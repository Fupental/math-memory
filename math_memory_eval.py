import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
    query_to_text,
    safe_model_name,
)
from lever_lm.math_memory.embeddings import build_embedder, load_or_create_embeddings
from lever_lm.math_memory.model import MathMemoryLeverLM
from lever_lm.math_memory.scoring import build_scorer


def _rs_memory_ids(query_id: int, memory_size: int, shot_num: int, seed: int) -> List[int]:
    rng = random.Random(seed + query_id * 1_000_003)
    memory_ids = rng.sample(range(memory_size), shot_num)
    rng.shuffle(memory_ids)
    return memory_ids


@torch.inference_mode()
def _lever_lm_memory_ids(
    checkpoint_path: str,
    test_queries: List[Dict],
    memories: List[Dict],
    embedding_cache_dir: str,
    embedding_model: str,
    embedding_device: str,
    embedding_dtype: str,
    embedding_batch_size: int,
    embedding_max_length: int,
    mock_emb_dim: int,
    device: str,
    shot_num: int,
    infer_batch_size: int,
    selection_mode: str,
) -> List[List[int]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = payload["metadata"]
    embedding_model = embedding_model or metadata["embedding_model"]

    embedder = build_embedder(
        embedding_model,
        device=embedding_device,
        dtype=embedding_dtype,
        max_length=embedding_max_length,
        mock_emb_dim=mock_emb_dim,
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

    model = MathMemoryLeverLM(
        memory_size=metadata["memory_size"],
        encoder_emb_dim=metadata["encoder_emb_dim"],
        n_embd=metadata["n_embd"],
        n_head=metadata["n_head"],
        n_layer=metadata["n_layer"],
        max_positions=metadata["max_positions"],
        model_backend=metadata.get("model_backend", "gpt2"),
    )
    model.load_state_dict(payload["model"], strict=False)
    run_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(run_device)
    model.eval()
    memory_embeddings = memory_embeddings.to(run_device)
    debias_query_embs = (
        query_embeddings.to(run_device) if selection_mode == "debiased" else None
    )

    all_memory_ids: List[List[int]] = []
    for start in tqdm(
        range(0, len(test_queries), infer_batch_size),
        desc="Lever-LM retrieval",
        ncols=100,
    ):
        query_batch = query_embeddings[start : start + infer_batch_size].to(run_device)
        batch_memory_ids = model.generate_memory_ids(
            query_embs=query_batch,
            memory_embedding_table=memory_embeddings,
            shot_num=shot_num,
            selection_mode=selection_mode,
            debias_query_embs=debias_query_embs,
            debias_batch_size=embedding_batch_size,
        )
        all_memory_ids.extend(batch_memory_ids.cpu().tolist())
    return all_memory_ids


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


def _write_metrics(output_dir: Path, metrics: Dict, predictions: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    method = metrics["method"]
    with (output_dir / f"{method}_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with (output_dir / f"{method}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "shot_num", "accuracy", "correct", "total"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "method": metrics["method"],
                "shot_num": metrics["shot_num"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval on MMLU-Pro math.")
    parser.add_argument("--method", required=True, choices=["lever_lm", "rs"])
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--selection-mode", choices=["raw", "debiased"], default="raw")
    parser.add_argument("--compute-final-delta", action="store_true")
    parser.add_argument("--shot-num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rs-seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=4)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--infer-batch-size", type=int, default=32)
    args = parser.parse_args()

    memories = load_experiences(args.experience_file)
    _train_queries, test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    if args.test_limit is not None:
        test_queries = test_queries[: args.test_limit]

    if args.method == "rs":
        rs_seed = args.seed if args.rs_seed is None else args.rs_seed
        retrievals = [
            _rs_memory_ids(query["query_id"], len(memories), args.shot_num, rs_seed)
            for query in test_queries
        ]
        output_method = f"rs_seed{rs_seed}" if args.rs_seed is not None else "rs"
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for method=lever_lm")
        rs_seed = None
        retrievals = _lever_lm_memory_ids(
            checkpoint_path=args.checkpoint,
            test_queries=test_queries,
            memories=memories,
            embedding_cache_dir=args.embedding_cache_dir,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_dtype=args.embedding_dtype,
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_length=args.embedding_max_length,
            mock_emb_dim=args.mock_emb_dim,
            device=args.device,
            shot_num=args.shot_num,
            infer_batch_size=args.infer_batch_size,
            selection_mode=args.selection_mode,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        output_method = (
            "lever_lm" if args.selection_mode == "raw" else f"lever_lm_{args.selection_mode}"
        )

    scorer = build_scorer(
        model_name=args.scorer_model,
        device=args.scorer_device,
        dtype=args.scorer_dtype,
        batch_size=args.scorer_batch_size,
        max_length=args.scorer_max_length,
    )

    predictions = []
    correct = 0
    final_deltas = []
    for query, memory_ids in tqdm(
        list(zip(test_queries, retrievals)), desc=f"Evaluating {args.method}", ncols=100
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
            "scores": scores,
        }
        if final_delta is not None:
            row["final_delta"] = final_delta
        predictions.append(row)

    total = len(test_queries)
    metrics = {
        "method": output_method,
        "base_method": args.method,
        "shot_num": args.shot_num,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "split_seed": args.seed,
        "rs_seed": rs_seed,
        "train_ratio": args.train_ratio,
        "selection_mode": args.selection_mode if args.method == "lever_lm" else "random",
        **_retrieval_diversity(retrievals),
    }
    if final_deltas:
        metrics["mean_final_delta"] = sum(final_deltas) / len(final_deltas)
    _write_metrics(Path(args.output_dir), metrics, predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
