import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
    query_to_text,
    safe_model_name,
)
from lever_lm.math_memory.embeddings import build_embedder, load_or_create_embeddings
from lever_lm.math_memory.model import MathMemoryLeverLM


CHECKPOINT_ORDER = {
    "init.pt": 0,
    "best.pt": 10_000_000,
    "last.pt": 10_000_001,
}


def _checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    if path.name in CHECKPOINT_ORDER:
        return CHECKPOINT_ORDER[path.name], path.name
    if path.name.startswith("step_") and path.name.endswith(".pt"):
        try:
            return int(path.name[len("step_") : -len(".pt")]), path.name
        except ValueError:
            pass
    return 20_000_000, path.name


def _resolve_checkpoints(checkpoint_dir: str, names: Optional[str]) -> List[Path]:
    root = Path(checkpoint_dir)
    if names:
        checkpoints = [root / name.strip() for name in names.split(",") if name.strip()]
    else:
        checkpoints = list(root.glob("*.pt"))
    checkpoints = sorted({path.resolve() for path in checkpoints}, key=_checkpoint_sort_key)
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    if not checkpoints:
        raise ValueError("No checkpoints found")
    return checkpoints


def _select_queries(args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_queries, test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    all_queries = train_queries + test_queries
    return train_queries, test_queries, all_queries


def _load_embeddings(
    args,
    embedding_model: str,
    memories: List[Dict[str, Any]],
    train_queries: List[Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    all_queries: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    embedder = build_embedder(
        embedding_model,
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        max_length=args.embedding_max_length,
        mock_emb_dim=args.mock_emb_dim,
    )
    safe_name = safe_model_name(embedding_model)
    cache_dir = Path(args.embedding_cache_dir)
    ratio_name = str(args.train_ratio).replace(".", "p")
    memory_embeddings = load_or_create_embeddings(
        str(cache_dir / f"memory_{safe_name}.pt"),
        [memory["text"] for memory in memories],
        embedder,
        batch_size=args.embedding_batch_size,
    ).float()
    query_embeddings = {
        "train": load_or_create_embeddings(
            str(cache_dir / f"grpo_train_queries_{safe_name}_seed{args.seed}_ratio{ratio_name}.pt"),
            [query_to_text(query) for query in train_queries],
            embedder,
            batch_size=args.embedding_batch_size,
        ).float(),
        "test": load_or_create_embeddings(
            str(cache_dir / f"test_queries_{safe_name}_n{len(test_queries)}.pt"),
            [query_to_text(query) for query in test_queries],
            embedder,
            batch_size=args.embedding_batch_size,
        ).float(),
        "all": load_or_create_embeddings(
            str(cache_dir / f"diagnose_all_queries_{safe_name}_seed{args.seed}_ratio{ratio_name}.pt"),
            [query_to_text(query) for query in all_queries],
            embedder,
            batch_size=args.embedding_batch_size,
        ).float(),
    }
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return memory_embeddings, query_embeddings


def _load_model(checkpoint: Path, device: torch.device) -> Tuple[MathMemoryLeverLM, Dict[str, Any]]:
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
    return model, metadata


def _counter_rows(counter: Counter, id_to_source: Dict[int, str], limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {
            "memory_id": int(memory_id),
            "source_id": id_to_source[int(memory_id)],
            "count": int(count),
        }
        for memory_id, count in counter.most_common(limit)
    ]


def _pair_counter_rows(counter: Counter, id_to_source: Dict[int, str], limit: int = 10) -> List[Dict[str, Any]]:
    rows = []
    for pair, count in counter.most_common(limit):
        rows.append(
            {
                "ids": [int(pair[0]), int(pair[1])],
                "pair": [id_to_source[int(pair[0])], id_to_source[int(pair[1])]],
                "count": int(count),
            }
        )
    return rows


def _quantile(values: Sequence[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * q)]


def _make_prefix(
    model: MathMemoryLeverLM,
    batch_size: int,
    device: torch.device,
    prefix_ids: Sequence[int],
) -> torch.Tensor:
    rows = [[model.bos_token_id, model.query_token_id, *prefix_ids] for _ in range(batch_size)]
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.inference_mode()
def _next_logits(
    model: MathMemoryLeverLM,
    query_embs: torch.Tensor,
    memory_embeddings: torch.Tensor,
    prefix_ids: Sequence[int],
) -> torch.Tensor:
    device = query_embs.device
    prefix = _make_prefix(model, query_embs.shape[0], device, prefix_ids)
    if prefix_ids:
        memory_embs = memory_embeddings[torch.tensor(prefix_ids, device=device)].unsqueeze(0)
        memory_embs = memory_embs.expand(query_embs.shape[0], -1, -1)
    else:
        memory_embs = torch.empty(
            query_embs.shape[0],
            0,
            model.encoder_emb_dim,
            dtype=query_embs.dtype,
            device=device,
        )
    output = model(
        input_ids=prefix,
        query_emb=query_embs,
        memory_embs=memory_embs,
        labels=None,
    )
    logits = output.logits[:, -1, : model.memory_size].clone()
    if prefix_ids:
        logits[:, list(prefix_ids)] = -torch.inf
    return logits


@torch.inference_mode()
def _batched_logits(
    model: MathMemoryLeverLM,
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    prefix_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, query_embeddings.shape[0], batch_size):
        batch = query_embeddings[start : start + batch_size].to(device)
        rows.append(_next_logits(model, batch, memory_embeddings, prefix_ids).cpu())
    return torch.cat(rows, dim=0)


def _rank_summary(logits: torch.Tensor, target_id: int) -> Dict[str, Any]:
    order = torch.argsort(logits, dim=-1, descending=True)
    ranks = (order == target_id).nonzero()[:, 1].add(1).tolist()
    return {
        "mean_rank": sum(ranks) / len(ranks),
        "min_rank": min(ranks),
        "p25": _quantile(ranks, 0.25),
        "median": _quantile(ranks, 0.50),
        "p75": _quantile(ranks, 0.75),
        "max_rank": max(ranks),
        "top1_count": sum(rank <= 1 for rank in ranks),
        "top32_count": sum(rank <= 32 for rank in ranks),
        "top64_count": sum(rank <= 64 for rank in ranks),
        "top128_count": sum(rank <= 128 for rank in ranks),
    }


def _first_hop_rows(
    checkpoints: List[Path],
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    target_id: int,
    id_to_source: Dict[int, str],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows = []
    memory_embeddings = memory_embeddings.to(device)
    for checkpoint in checkpoints:
        model, metadata = _load_model(checkpoint, device)
        logits = _batched_logits(model, query_embeddings, memory_embeddings, [], device, batch_size)
        top1 = logits.argmax(dim=-1).tolist()
        row = {
            "checkpoint": checkpoint.name,
            "current_step": metadata.get("current_step"),
            "update_count": metadata.get("update_count"),
            "n": int(logits.shape[0]),
            "unique_first_step_top1": len(set(top1)),
            "first_step_top1": json.dumps(
                _counter_rows(Counter(top1), id_to_source), ensure_ascii=False
            ),
        }
        row.update(_rank_summary(logits, target_id))
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _greedy_pairs(
    model: MathMemoryLeverLM,
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> List[Tuple[int, int]]:
    pairs = []
    memory_embeddings = memory_embeddings.to(device)
    for start in range(0, query_embeddings.shape[0], batch_size):
        batch = query_embeddings[start : start + batch_size].to(device)
        first_logits = _next_logits(model, batch, memory_embeddings, [])
        first = first_logits.argmax(dim=-1)
        second_ids = []
        for prefix_id in first.tolist():
            one_logits = _next_logits(
                model,
                batch[len(second_ids) : len(second_ids) + 1],
                memory_embeddings,
                [int(prefix_id)],
            )
            second_ids.append(int(one_logits.argmax(dim=-1).item()))
        pairs.extend((int(a), int(b)) for a, b in zip(first.tolist(), second_ids))
    return pairs


def _query_ablation_rows(
    checkpoints: List[Path],
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    prefix_id: int,
    id_to_source: Dict[int, str],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows = []
    modes = {
        "real": query_embeddings,
        "shuffled": query_embeddings[torch.randperm(query_embeddings.shape[0])],
        "zero": torch.zeros_like(query_embeddings),
    }
    memory_embeddings = memory_embeddings.to(device)
    for checkpoint in checkpoints:
        model, metadata = _load_model(checkpoint, device)
        for mode_name, mode_embeddings in modes.items():
            first_logits = _batched_logits(model, mode_embeddings, memory_embeddings, [], device, batch_size)
            first_top1 = first_logits.argmax(dim=-1).tolist()
            second_logits = _batched_logits(
                model, mode_embeddings, memory_embeddings, [prefix_id], device, batch_size
            )
            second_top1 = second_logits.argmax(dim=-1).tolist()
            pairs = _greedy_pairs(model, mode_embeddings, memory_embeddings, device, batch_size)
            rows.append(
                {
                    "checkpoint": checkpoint.name,
                    "current_step": metadata.get("current_step"),
                    "input": mode_name,
                    "first_unique_top1": len(set(first_top1)),
                    "first_top1": json.dumps(
                        _counter_rows(Counter(first_top1), id_to_source), ensure_ascii=False
                    ),
                    "second_after_prefix_unique_top1": len(set(second_top1)),
                    "second_after_prefix_top1": json.dumps(
                        _counter_rows(Counter(second_top1), id_to_source), ensure_ascii=False
                    ),
                    "greedy_pair_unique": len(set(pairs)),
                    "greedy_pair_top": json.dumps(
                        _pair_counter_rows(Counter(pairs), id_to_source), ensure_ascii=False
                    ),
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _debiased_top1(logits: torch.Tensor, masked_ids: Sequence[int]) -> torch.Tensor:
    bias = logits.mean(dim=0, keepdim=True)
    residual = logits - bias
    if masked_ids:
        residual[:, list(masked_ids)] = -torch.inf
    return residual.argmax(dim=-1)


def _logit_bias_rows(
    checkpoints: List[Path],
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    prefix_id: int,
    id_to_source: Dict[int, str],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows = []
    memory_embeddings = memory_embeddings.to(device)
    for checkpoint in checkpoints:
        model, metadata = _load_model(checkpoint, device)
        for stage, prefix_ids in [("first", []), ("second_after_prefix", [prefix_id])]:
            logits = _batched_logits(
                model, query_embeddings, memory_embeddings, prefix_ids, device, batch_size
            )
            raw_top1 = logits.argmax(dim=-1).tolist()
            debiased_top1 = _debiased_top1(logits, prefix_ids).tolist()
            rows.append(
                {
                    "checkpoint": checkpoint.name,
                    "current_step": metadata.get("current_step"),
                    "stage": stage,
                    "raw_unique_top1": len(set(raw_top1)),
                    "debiased_unique_top1": len(set(debiased_top1)),
                    "raw_top1": json.dumps(
                        _counter_rows(Counter(raw_top1), id_to_source), ensure_ascii=False
                    ),
                    "debiased_top1": json.dumps(
                        _counter_rows(Counter(debiased_top1), id_to_source), ensure_ascii=False
                    ),
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _summarize_history(history_path: Path, steps: Sequence[int]) -> List[Dict[str, Any]]:
    if not history_path.exists():
        return []
    rows = list(csv.DictReader(history_path.open(encoding="utf-8")))
    selected = []
    for step in steps:
        if step < 0 or step >= len(rows):
            continue
        row = rows[step]
        selected.append(
            {
                "step": int(row["step"]),
                "reward_mean": _to_float(row.get("reward_mean")),
                "final_delta_mean": _to_float(row.get("final_delta_mean")),
                "entropy": _to_float(row.get("entropy")),
                "entropy_step0": _to_float(row.get("entropy_step0")),
                "entropy_step1": _to_float(row.get("entropy_step1")),
                "marginal_entropy": _to_float(row.get("marginal_entropy")),
                "marginal_entropy_step0": _to_float(row.get("marginal_entropy_step0")),
                "marginal_entropy_step1": _to_float(row.get("marginal_entropy_step1")),
                "reference_kl": _to_float(row.get("reference_kl")),
                "reference_kl_step0": _to_float(row.get("reference_kl_step0")),
                "reference_kl_step1": _to_float(row.get("reference_kl_step1")),
                "approx_kl": _to_float(row.get("approx_kl")),
                "clip_fraction": _to_float(row.get("clip_fraction")),
                "mean_abs_logprob_delta": _to_float(row.get("mean_abs_logprob_delta")),
                "max_prob_step0": _to_float(row.get("max_prob_step0")),
                "max_prob_step1": _to_float(row.get("max_prob_step1")),
                "unique_memory_count": _to_float(row.get("unique_memory_count")),
            }
        )
    return selected


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_action_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        parsed = dict(row)
        parsed["step"] = int(parsed["step"])
        parsed["action_memory_id"] = int(parsed["action_memory_id"])
        parsed["count"] = int(parsed["count"])
        for key in [
            "mean_a1",
            "mean_g1",
            "mean_old_logprob_step1",
            "mean_new_logprob_step1",
            "mean_logprob_delta_step1",
        ]:
            parsed[key] = float(parsed[key])
        rows.append(parsed)
    return rows


def _action_trace_summary(
    trace_path: Path,
    wanted_source_ids: Sequence[str],
    steps: Sequence[int],
) -> List[Dict[str, Any]]:
    rows = _read_action_trace(trace_path)
    if not rows:
        return []
    wanted = set(wanted_source_ids)
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)

    summary = []
    for step in steps:
        current = by_step.get(step, [])
        if not current:
            continue
        current_sorted = sorted(current, key=lambda row: row["mean_a1"], reverse=True)
        rank_by_action = {
            row["action_source_id"]: rank for rank, row in enumerate(current_sorted, start=1)
        }
        for row in current_sorted:
            if row["action_source_id"] in wanted or rank_by_action[row["action_source_id"]] <= 5:
                summary.append(
                    {
                        "step": step,
                        "rank_by_mean_a1": rank_by_action[row["action_source_id"]],
                        "action_source_id": row["action_source_id"],
                        "action_memory_id": row["action_memory_id"],
                        "count": row["count"],
                        "mean_a1": row["mean_a1"],
                        "mean_g1": row["mean_g1"],
                        "mean_logprob_delta_step1": row["mean_logprob_delta_step1"],
                    }
                )
    return summary


def _first_action_trace_summary(
    trace_path: Path,
    wanted_source_ids: Sequence[str],
    steps: Sequence[int],
) -> List[Dict[str, Any]]:
    if not trace_path.exists():
        return []
    rows = []
    for row in csv.DictReader(trace_path.open(encoding="utf-8")):
        parsed = dict(row)
        parsed["step"] = int(parsed["step"])
        parsed["action_memory_id"] = int(parsed["action_memory_id"])
        parsed["count"] = int(parsed["count"])
        for key in [
            "mean_r0",
            "mean_r1",
            "mean_g0",
            "mean_a0",
            "mean_logprob_delta_step0",
        ]:
            parsed[key] = float(parsed[key])
        rows.append(parsed)
    wanted = set(wanted_source_ids)
    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)

    summary = []
    for step in steps:
        current = by_step.get(step, [])
        if not current:
            continue
        current_sorted = sorted(current, key=lambda row: row["mean_a0"], reverse=True)
        rank_by_action = {
            row["action_source_id"]: rank for rank, row in enumerate(current_sorted, start=1)
        }
        for row in current_sorted:
            if row["action_source_id"] in wanted or rank_by_action[row["action_source_id"]] <= 5:
                summary.append(
                    {
                        "step": step,
                        "rank_by_mean_a0": rank_by_action[row["action_source_id"]],
                        "action_source_id": row["action_source_id"],
                        "action_memory_id": row["action_memory_id"],
                        "count": row["count"],
                        "mean_r0": row["mean_r0"],
                        "mean_r1": row["mean_r1"],
                        "mean_g0": row["mean_g0"],
                        "mean_a0": row["mean_a0"],
                        "mean_logprob_delta_step0": row["mean_logprob_delta_step0"],
                    }
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose GRPO memory-policy collapse and query-blind behavior."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-names", default=None)
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
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--prefix-source-id", default="G615")
    parser.add_argument("--first-target-source-id", default="G615")
    parser.add_argument("--trace-actions", default="G634,G391")
    parser.add_argument("--first-trace-actions", default="G615,G391,G634")
    parser.add_argument("--summary-steps", default="0,10,20,30,40,50,75,100,150,200,219")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoints = _resolve_checkpoints(args.checkpoint_dir, args.checkpoint_names)
    first_payload = torch.load(checkpoints[0], map_location="cpu")
    embedding_model = args.embedding_model or first_payload["metadata"].get(
        "embedding_model", "Qwen/Qwen3-Embedding-0.6B"
    )

    memories = load_experiences(args.experience_file)
    source_to_id = {memory["source_id"]: memory["memory_id"] for memory in memories}
    id_to_source = {memory["memory_id"]: memory["source_id"] for memory in memories}
    if args.prefix_source_id not in source_to_id:
        raise ValueError(f"Unknown --prefix-source-id: {args.prefix_source_id}")
    if args.first_target_source_id not in source_to_id:
        raise ValueError(f"Unknown --first-target-source-id: {args.first_target_source_id}")
    prefix_id = source_to_id[args.prefix_source_id]
    first_target_id = source_to_id[args.first_target_source_id]

    train_queries, test_queries, all_queries = _select_queries(args)
    memory_embeddings, query_embeddings = _load_embeddings(
        args,
        embedding_model,
        memories,
        train_queries,
        test_queries,
        all_queries,
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or Path(args.checkpoint_dir) / "diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_rows = _first_hop_rows(
        checkpoints,
        query_embeddings["all"],
        memory_embeddings,
        first_target_id,
        id_to_source,
        device,
        args.infer_batch_size,
    )
    ablation_rows = _query_ablation_rows(
        checkpoints,
        query_embeddings["test"],
        memory_embeddings,
        prefix_id,
        id_to_source,
        device,
        args.infer_batch_size,
    )
    bias_rows = _logit_bias_rows(
        checkpoints,
        query_embeddings["test"],
        memory_embeddings,
        prefix_id,
        id_to_source,
        device,
        args.infer_batch_size,
    )

    steps = [int(item) for item in args.summary_steps.split(",") if item.strip()]
    history_rows = _summarize_history(Path(args.checkpoint_dir) / "grpo_history.csv", steps)
    trace_actions = [item.strip() for item in args.trace_actions.split(",") if item.strip()]
    action_rows = _action_trace_summary(
        Path(args.checkpoint_dir) / "grpo_action_trace.csv",
        trace_actions,
        steps,
    )
    first_trace_actions = [
        item.strip() for item in args.first_trace_actions.split(",") if item.strip()
    ]
    first_action_rows = _first_action_trace_summary(
        Path(args.checkpoint_dir) / "grpo_first_action_trace.csv",
        first_trace_actions,
        steps,
    )

    _write_csv(output_dir / "first_hop_trace.csv", first_rows)
    _write_csv(output_dir / "query_ablation.csv", ablation_rows)
    _write_csv(output_dir / "logit_bias_decomposition.csv", bias_rows)
    _write_csv(output_dir / "ppo_history_summary.csv", history_rows)
    _write_csv(output_dir / "action_trace_summary.csv", action_rows)
    _write_csv(output_dir / "first_action_trace_summary.csv", first_action_rows)
    summary = {
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "checkpoints": [path.name for path in checkpoints],
        "prefix_source_id": args.prefix_source_id,
        "first_target_source_id": args.first_target_source_id,
        "num_train_queries": len(train_queries),
        "num_test_queries": len(test_queries),
        "num_all_queries": len(all_queries),
        "outputs": {
            "first_hop_trace": str((output_dir / "first_hop_trace.csv").resolve()),
            "query_ablation": str((output_dir / "query_ablation.csv").resolve()),
            "logit_bias_decomposition": str((output_dir / "logit_bias_decomposition.csv").resolve()),
            "ppo_history_summary": str((output_dir / "ppo_history_summary.csv").resolve()),
            "action_trace_summary": str((output_dir / "action_trace_summary.csv").resolve()),
            "first_action_trace_summary": str(
                (output_dir / "first_action_trace_summary.csv").resolve()
            ),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
