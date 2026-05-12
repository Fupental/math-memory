import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
    query_to_text,
    safe_model_name,
)
from lever_lm.math_memory.embeddings import build_embedder, load_or_create_embeddings
from lever_lm.math_memory.model import MathMemoryLeverLM


def _checkpoint_sort_key(path: Path):
    name = path.name
    if name == "init.pt":
        return (0, 0, name)
    if name.startswith("step_") and name.endswith(".pt"):
        try:
            return (1, int(name[len("step_") : -len(".pt")]), name)
        except ValueError:
            return (1, 0, name)
    if name == "best.pt":
        return (2, 0, name)
    if name == "last.pt":
        return (3, 0, name)
    return (4, 0, name)


def _resolve_checkpoints(args) -> List[Path]:
    checkpoints: List[Path] = []
    if args.checkpoint:
        checkpoints.extend(Path(path) for path in args.checkpoint)
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints.extend(checkpoint_dir.glob("*.pt"))
    unique = sorted({path.resolve() for path in checkpoints}, key=_checkpoint_sort_key)
    if not unique:
        raise ValueError("No checkpoints found. Use --checkpoint or --checkpoint-dir.")
    return unique


def _quantile(values: List[int], q: float) -> int:
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def _load_query_embeddings(args, embedding_model: str, embedder, queries: List[Dict]) -> torch.Tensor:
    safe_name = safe_model_name(embedding_model)
    cache_dir = Path(args.embedding_cache_dir)

    if args.split == "test":
        cache_name = f"test_queries_{safe_name}_n{len(queries)}.pt"
    elif args.split == "train":
        ratio_name = str(args.train_ratio).replace(".", "p")
        cache_name = f"grpo_train_queries_{safe_name}_seed{args.seed}_ratio{ratio_name}.pt"
    else:
        ratio_name = str(args.train_ratio).replace(".", "p")
        cache_name = (
            f"diagnose_all_queries_{safe_name}_seed{args.seed}_ratio{ratio_name}.pt"
        )

    return load_or_create_embeddings(
        str(cache_dir / cache_name),
        [query_to_text(query) for query in queries],
        embedder,
        batch_size=args.embedding_batch_size,
    )


def _load_memory_embeddings(args, embedding_model: str, embedder, memories: List[Dict]) -> torch.Tensor:
    safe_name = safe_model_name(embedding_model)
    return load_or_create_embeddings(
        str(Path(args.embedding_cache_dir) / f"memory_{safe_name}.pt"),
        [memory["text"] for memory in memories],
        embedder,
        batch_size=args.embedding_batch_size,
    )


def _select_queries(args) -> List[Dict]:
    train_queries, test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    if args.split == "train":
        queries = train_queries
    elif args.split == "test":
        queries = test_queries
    else:
        queries = train_queries + test_queries
    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        raise ValueError("No queries selected")
    return queries


def _diagnose_checkpoint(
    checkpoint: Path,
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    prefix_id: int,
    target_id: int,
    id_to_source: Dict[int, str],
    device: torch.device,
    batch_size: int,
) -> Dict:
    payload = torch.load(checkpoint, map_location="cpu")
    metadata = payload["metadata"]
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
    model.to(device)
    model.eval()

    memory_embeddings = memory_embeddings.to(device)
    ranks: List[int] = []
    top1_ids: List[int] = []

    with torch.inference_mode():
        for start in range(0, query_embeddings.shape[0], batch_size):
            query_batch = query_embeddings[start : start + batch_size].to(device)
            prefix = torch.tensor(
                [
                    [model.bos_token_id, model.query_token_id, prefix_id]
                    for _ in range(query_batch.shape[0])
                ],
                dtype=torch.long,
                device=device,
            )
            memory_batch = memory_embeddings[prefix[:, 2:]]
            output = model(
                input_ids=prefix,
                query_emb=query_batch,
                memory_embs=memory_batch,
                labels=None,
            )
            logits = output.logits[:, -1, : model.memory_size].clone()
            logits[:, prefix_id] = -torch.inf
            order = torch.argsort(logits, dim=-1, descending=True).cpu()
            top1_ids.extend(order[:, 0].tolist())
            positions = (order == target_id).nonzero()
            ranks.extend((positions[:, 1] + 1).tolist())

    top1_counter = Counter(top1_ids)
    rank_counter = Counter(ranks)
    n = len(ranks)
    row = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "current_step": metadata.get("current_step"),
        "update_count": metadata.get("update_count"),
        "best_step": metadata.get("best_step"),
        "n": n,
        "target_rank_mean": sum(ranks) / n,
        "target_rank_min": min(ranks),
        "target_rank_p25": _quantile(ranks, 0.25),
        "target_rank_median": _quantile(ranks, 0.50),
        "target_rank_p75": _quantile(ranks, 0.75),
        "target_rank_max": max(ranks),
        "target_top1_count": sum(rank <= 1 for rank in ranks),
        "target_top2_count": sum(rank <= 2 for rank in ranks),
        "target_top5_count": sum(rank <= 5 for rank in ranks),
        "target_top10_count": sum(rank <= 10 for rank in ranks),
        "target_top32_count": sum(rank <= 32 for rank in ranks),
        "target_top64_count": sum(rank <= 64 for rank in ranks),
        "target_top128_count": sum(rank <= 128 for rank in ranks),
        "unique_second_step_top1": len(top1_counter),
        "second_step_top1": [
            {
                "memory_id": memory_id,
                "source_id": id_to_source[memory_id],
                "count": count,
            }
            for memory_id, count in top1_counter.most_common(20)
        ],
        "rank_counts": [
            {"rank": rank, "count": count}
            for rank, count in rank_counter.most_common(20)
        ],
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose successor memory rank after a forced prefix memory."
    )
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--infer-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--prefix-source-id", default="G615")
    parser.add_argument("--target-source-id", default="G391")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoints = _resolve_checkpoints(args)
    first_payload = torch.load(checkpoints[0], map_location="cpu")
    first_metadata = first_payload["metadata"]
    embedding_model = args.embedding_model or first_metadata.get(
        "embedding_model", "Qwen/Qwen3-Embedding-0.6B"
    )

    memories = load_experiences(args.experience_file)
    source_to_id = {memory["source_id"]: memory["memory_id"] for memory in memories}
    id_to_source = {memory["memory_id"]: memory["source_id"] for memory in memories}
    if args.prefix_source_id not in source_to_id:
        raise ValueError(f"Unknown prefix source id: {args.prefix_source_id}")
    if args.target_source_id not in source_to_id:
        raise ValueError(f"Unknown target source id: {args.target_source_id}")
    prefix_id = source_to_id[args.prefix_source_id]
    target_id = source_to_id[args.target_source_id]

    queries = _select_queries(args)
    embedder = build_embedder(
        embedding_model,
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        max_length=args.embedding_max_length,
        mock_emb_dim=args.mock_emb_dim,
    )
    query_embeddings = _load_query_embeddings(args, embedding_model, embedder, queries).float()
    memory_embeddings = _load_memory_embeddings(args, embedding_model, embedder, memories).float()
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    rows = []
    for checkpoint in checkpoints:
        row = _diagnose_checkpoint(
            checkpoint=checkpoint,
            query_embeddings=query_embeddings,
            memory_embeddings=memory_embeddings,
            prefix_id=prefix_id,
            target_id=target_id,
            id_to_source=id_to_source,
            device=device,
            batch_size=args.infer_batch_size,
        )
        rows.append(row)
        print(
            f"{row['checkpoint_name']}: mean_rank={row['target_rank_mean']:.3f} "
            f"top1={row['target_top1_count']}/{row['n']} "
            f"top64={row['target_top64_count']}/{row['n']} "
            f"top128={row['target_top128_count']}/{row['n']} "
            f"unique_top1={row['unique_second_step_top1']}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path(checkpoints[0]).parent / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"successor_rank_{args.prefix_source_id}_to_{args.target_source_id}_{args.split}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "prefix_source_id": args.prefix_source_id,
                "prefix_memory_id": prefix_id,
                "target_source_id": args.target_source_id,
                "target_memory_id": target_id,
                "split": args.split,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    flat_fields = [
        "checkpoint_name",
        "checkpoint",
        "current_step",
        "update_count",
        "best_step",
        "n",
        "target_rank_mean",
        "target_rank_min",
        "target_rank_p25",
        "target_rank_median",
        "target_rank_p75",
        "target_rank_max",
        "target_top1_count",
        "target_top2_count",
        "target_top5_count",
        "target_top10_count",
        "target_top32_count",
        "target_top64_count",
        "target_top128_count",
        "unique_second_step_top1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in flat_fields})

    print(f"Saved diagnostics to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
