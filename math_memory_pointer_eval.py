import argparse
import csv
import json
import random
import shlex
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
    query_to_text,
    safe_model_name,
)
from lever_lm.math_memory.embeddings import build_embedder, load_or_create_embeddings
from lever_lm.math_memory.model import PointerMemoryLeverLM
from lever_lm.math_memory.scoring import build_scorer


def _random_candidate_ids(
    query_id: int,
    memory_size: int,
    candidate_num: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed + query_id * 1_000_003)
    return rng.sample(range(memory_size), candidate_num)


def _semantic_candidate_ids_for_batch(
    query_embs: torch.Tensor,
    memory_embeddings: torch.Tensor,
    candidate_num: int,
) -> torch.Tensor:
    if candidate_num > memory_embeddings.shape[0]:
        raise ValueError(
            f"candidate_num={candidate_num} exceeds memory_size={memory_embeddings.shape[0]}"
        )
    query = F.normalize(query_embs.float(), dim=-1)
    memory = F.normalize(memory_embeddings.float(), dim=-1)
    scores = query @ memory.T
    return torch.topk(scores, k=candidate_num, dim=-1).indices


def _mixed_candidate_ids_for_batch(
    query_ids: List[int],
    semantic_ids: torch.Tensor,
    memory_size: int,
    random_candidate_num: int,
    seed: int,
) -> torch.Tensor:
    rows = []
    for query_id, semantic_row in zip(query_ids, semantic_ids.cpu().tolist()):
        semantic_set = set(semantic_row)
        random_pool = [
            memory_id
            for memory_id in range(memory_size)
            if memory_id not in semantic_set
        ]
        rng = random.Random(seed + query_id * 1_000_003)
        random_ids = rng.sample(random_pool, random_candidate_num)
        candidate_ids = list(semantic_row) + random_ids
        rng.shuffle(candidate_ids)
        rows.append(candidate_ids)
    return torch.tensor(rows, dtype=torch.long, device=semantic_ids.device)


def _candidate_ids_for_batch(
    mode: str,
    query_batch: List[Dict],
    query_embs: torch.Tensor,
    memory_embeddings: torch.Tensor,
    memory_size: int,
    candidate_num: int,
    candidate_seed: int,
    random_candidate_num: int,
    semantic_candidate_num: int,
    device,
) -> torch.Tensor:
    if mode == "random":
        return torch.tensor(
            [
                _random_candidate_ids(
                    query["query_id"], memory_size, candidate_num, candidate_seed
                )
                for query in query_batch
            ],
            dtype=torch.long,
            device=device,
        )
    if mode == "semantic":
        return _semantic_candidate_ids_for_batch(
            query_embs=query_embs,
            memory_embeddings=memory_embeddings,
            candidate_num=candidate_num,
        ).to(device)
    if mode == "mixed":
        if random_candidate_num + semantic_candidate_num != candidate_num:
            raise ValueError(
                "For candidate_mode=mixed, random_candidate_num + semantic_candidate_num "
                "must equal candidate_num"
            )
        semantic_ids = _semantic_candidate_ids_for_batch(
            query_embs=query_embs,
            memory_embeddings=memory_embeddings,
            candidate_num=semantic_candidate_num,
        ).to(device)
        return _mixed_candidate_ids_for_batch(
            query_ids=[query["query_id"] for query in query_batch],
            semantic_ids=semantic_ids,
            memory_size=memory_size,
            random_candidate_num=random_candidate_num,
            seed=candidate_seed,
        )
    raise ValueError(f"Unsupported candidate mode: {mode}")


def _selection_mode(candidate_mode: str, candidate_num: int, random_num: int, semantic_num: int) -> str:
    if candidate_mode == "random":
        return f"pointer_random{candidate_num}_greedy"
    if candidate_mode == "semantic":
        return f"pointer_semantic_top{candidate_num}_greedy"
    if candidate_mode == "mixed":
        return f"pointer_mixed_random{random_num}_semantic{semantic_num}_greedy"
    return f"pointer_{candidate_mode}_greedy"


def _retrieval_diversity(retrievals: List[List[int]]) -> Dict:
    if not retrievals:
        return {
            "unique_first_memory_count": 0,
            "unique_second_memory_count": 0,
            "unique_pair_count": 0,
        }
    first = [row[0] for row in retrievals if row]
    second = [row[1] for row in retrievals if len(row) > 1]
    pairs = [tuple(row) for row in retrievals]
    return {
        "unique_first_memory_count": len(set(first)),
        "unique_second_memory_count": len(set(second)),
        "unique_pair_count": len(set(pairs)),
    }


@torch.inference_mode()
def _pointer_memory_ids(
    checkpoint_path: str,
    test_queries: List[Dict],
    memories: List[Dict],
    embedding_cache_dir: str,
    embedding_model: str,
    embedding_device: str,
    embedding_dtype: str,
    embedding_batch_size: int,
    embedding_max_length: int,
    embedding_device_map: str | None,
    embedding_max_memory: str | None,
    mock_emb_dim: int,
    device: str,
    candidate_num: int,
    candidate_seed: int,
    candidate_mode: str,
    random_candidate_num: int,
    semantic_candidate_num: int,
    infer_batch_size: int,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = payload["metadata"]
    embedding_model = embedding_model or metadata["embedding_model"]
    candidate_num = candidate_num or metadata["candidate_num"]

    embedder = build_embedder(
        embedding_model,
        device=embedding_device,
        dtype=embedding_dtype,
        max_length=embedding_max_length,
        mock_emb_dim=mock_emb_dim,
        device_map=embedding_device_map,
        max_memory=embedding_max_memory,
    )
    safe_name = safe_model_name(embedding_model)
    cache_dir = Path(embedding_cache_dir)
    memory_embeddings = load_or_create_embeddings(
        str(cache_dir / f"memory_{safe_name}.pt"),
        [memory["text"] for memory in memories],
        embedder,
        batch_size=embedding_batch_size,
    )
    query_embeddings = load_or_create_embeddings(
        str(cache_dir / f"test_queries_{safe_name}_n{len(test_queries)}.pt"),
        [query_to_text(query) for query in test_queries],
        embedder,
        batch_size=embedding_batch_size,
    )

    model = PointerMemoryLeverLM(
        encoder_emb_dim=metadata["encoder_emb_dim"],
        candidate_num=metadata["candidate_num"],
        n_embd=metadata["n_embd"],
        n_head=metadata["n_head"],
        n_layer=metadata["n_layer"],
        max_positions=metadata["max_positions"],
        normalize_encoder_emb=metadata.get("normalize_encoder_emb", True),
        pointer_key_source=metadata.get("pointer_key_source", "contextual"),
    )
    model.load_state_dict(payload["model"], strict=True)
    run_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(run_device)
    model.eval()
    memory_embeddings = memory_embeddings.to(run_device)

    all_memory_ids: List[List[int]] = []
    all_candidate_ids: List[List[int]] = []
    for start in tqdm(
        range(0, len(test_queries), infer_batch_size),
        desc="Pointer retrieval",
        ncols=100,
    ):
        query_batch = test_queries[start : start + infer_batch_size]
        query_embs = query_embeddings[start : start + infer_batch_size].to(run_device)
        candidate_ids = _candidate_ids_for_batch(
            mode=candidate_mode,
            query_batch=query_batch,
            query_embs=query_embs,
            memory_embeddings=memory_embeddings,
            memory_size=len(memories),
            candidate_num=candidate_num,
            candidate_seed=candidate_seed,
            random_candidate_num=random_candidate_num,
            semantic_candidate_num=semantic_candidate_num,
            device=run_device,
        )
        batch_memory_ids = model.generate_memory_ids(
            query_embs=query_embs,
            candidate_memory_ids=candidate_ids,
            memory_embedding_table=memory_embeddings,
        )
        all_memory_ids.extend(batch_memory_ids.cpu().tolist())
        all_candidate_ids.extend(candidate_ids.cpu().tolist())
    return all_memory_ids, all_candidate_ids, metadata


def _write_outputs(output_dir: Path, metrics: Dict, predictions: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    method = metrics["method"]
    with (output_dir / f"{method}_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with (output_dir / f"{method}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (output_dir / "eval_command.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["command"] + "\n")

    csv_path = output_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "candidate_mode",
                "selection_mode",
                "shot_num",
                "candidate_seed",
                "candidate_num",
                "random_candidate_num",
                "semantic_candidate_num",
                "accuracy",
                "correct",
                "total",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "method": metrics["method"],
                "candidate_mode": metrics["candidate_mode"],
                "selection_mode": metrics["selection_mode"],
                "shot_num": metrics["shot_num"],
                "candidate_seed": metrics["candidate_seed"],
                "candidate_num": metrics["candidate_num"],
                "random_candidate_num": metrics.get("random_candidate_num", ""),
                "semantic_candidate_num": metrics.get("semantic_candidate_num", ""),
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pointer Lever-LM on MMLU-Pro math.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-num", type=int, default=None)
    parser.add_argument("--candidate-seed", type=int, default=42)
    parser.add_argument(
        "--candidate-mode",
        choices=["random", "semantic", "mixed"],
        default="random",
    )
    parser.add_argument("--random-candidate-num", type=int, default=32)
    parser.add_argument("--semantic-candidate-num", type=int, default=32)
    parser.add_argument("--compute-final-delta", action="store_true")
    parser.add_argument("--shot-num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=4)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--scorer-device-map", default=None)
    parser.add_argument("--scorer-max-memory", default=None)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device-map", default=None)
    parser.add_argument("--embedding-max-memory", default=None)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--infer-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.shot_num != 2:
        raise ValueError("Pointer v1 supports --shot-num 2 only")
    if args.candidate_mode == "mixed":
        effective_candidate_num = args.candidate_num or 64
        if args.random_candidate_num + args.semantic_candidate_num != effective_candidate_num:
            raise ValueError(
                "--random-candidate-num + --semantic-candidate-num must equal "
                "--candidate-num for mixed eval"
            )

    memories = load_experiences(args.experience_file)
    _train_queries, test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    if args.test_limit is not None:
        test_queries = test_queries[: args.test_limit]

    retrievals, candidate_sets, checkpoint_metadata = _pointer_memory_ids(
        checkpoint_path=args.checkpoint,
        test_queries=test_queries,
        memories=memories,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_dtype=args.embedding_dtype,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        embedding_device_map=args.embedding_device_map,
        embedding_max_memory=args.embedding_max_memory,
        mock_emb_dim=args.mock_emb_dim,
        device=args.device,
        candidate_num=args.candidate_num,
        candidate_seed=args.candidate_seed,
        candidate_mode=args.candidate_mode,
        random_candidate_num=args.random_candidate_num,
        semantic_candidate_num=args.semantic_candidate_num,
        infer_batch_size=args.infer_batch_size,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorer = build_scorer(
        model_name=args.scorer_model,
        device=args.scorer_device,
        dtype=args.scorer_dtype,
        batch_size=args.scorer_batch_size,
        max_length=args.scorer_max_length,
        device_map=args.scorer_device_map,
        max_memory=args.scorer_max_memory,
    )

    predictions = []
    correct = 0
    final_deltas = []
    for query, memory_ids, candidate_ids in tqdm(
        list(zip(test_queries, retrievals, candidate_sets)),
        desc="Evaluating pointer_lever_lm",
        ncols=100,
    ):
        prediction, scores = scorer.predict(query, memory_ids, memories)
        final_delta = None
        if args.compute_final_delta:
            empty_score, full_score = scorer.score_gold_sequences(
                query, [[], memory_ids], memories
            )
            final_delta = full_score - empty_score
            final_deltas.append(final_delta)
        is_correct = prediction == query["answer"]
        correct += int(is_correct)
        row = {
            "query_id": query["query_id"],
            "question_id": query["question_id"],
            "prediction": prediction,
            "answer": query["answer"],
            "correct": is_correct,
            "memory_ids": memory_ids,
            "memory_source_ids": [memories[memory_id]["source_id"] for memory_id in memory_ids],
            "candidate_ids": candidate_ids,
            "candidate_source_ids": [
                memories[memory_id]["source_id"] for memory_id in candidate_ids
            ],
            "scores": scores,
        }
        if final_delta is not None:
            row["final_delta"] = final_delta
        predictions.append(row)

    total = len(test_queries)
    candidate_num = len(candidate_sets[0]) if candidate_sets else args.candidate_num
    metrics = {
        "method": f"pointer_lever_lm_seed{args.candidate_seed}",
        "base_method": "pointer_lever_lm",
        "shot_num": args.shot_num,
        "candidate_num": candidate_num,
        "candidate_seed": args.candidate_seed,
        "candidate_mode": args.candidate_mode,
        "random_candidate_num": args.random_candidate_num if args.candidate_mode == "mixed" else None,
        "semantic_candidate_num": (
            args.semantic_candidate_num
            if args.candidate_mode == "mixed"
            else (candidate_num if args.candidate_mode == "semantic" else None)
        ),
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "split_seed": args.seed,
        "train_ratio": args.train_ratio,
        "selection_mode": _selection_mode(
            args.candidate_mode,
            candidate_num,
            args.random_candidate_num,
            args.semantic_candidate_num,
        ),
        "pointer_key_source": checkpoint_metadata.get("pointer_key_source", "contextual"),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        **_retrieval_diversity(retrievals),
    }
    if final_deltas:
        metrics["mean_final_delta"] = sum(final_deltas) / len(final_deltas)
    _write_outputs(Path(args.output_dir), metrics, predictions)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
