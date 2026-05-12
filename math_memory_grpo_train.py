import argparse
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
    query_to_text,
    safe_model_name,
)
from lever_lm.math_memory.embeddings import build_embedder, load_or_create_embeddings
from lever_lm.math_memory.model import MathMemoryLeverLM, checkpoint_metadata
from lever_lm.math_memory.scoring import build_scorer


ScoreCache = Dict[Tuple[int, Tuple[int, ...]], float]
CorrectnessCache = Dict[Tuple[int, Tuple[int, ...]], bool]


def _parse_checkpoint_steps(value: str) -> List[int]:
    if not value:
        return []
    steps = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise ValueError("--checkpoint-steps entries must be >= 0")
        steps.append(step)
    return sorted(set(steps))


def _score_sequences_cached(
    scorer,
    query: Dict[str, Any],
    memory_sequences: List[List[int]],
    memories: List[Dict[str, Any]],
    cache: ScoreCache,
) -> List[float]:
    missing = []
    for memory_ids in memory_sequences:
        key = (query["query_id"], tuple(memory_ids))
        if key not in cache:
            missing.append(memory_ids)

    if missing:
        scores = scorer.score_gold_sequences(query, missing, memories)
        for memory_ids, score in zip(missing, scores):
            cache[(query["query_id"], tuple(memory_ids))] = float(score)

    return [cache[(query["query_id"], tuple(memory_ids))] for memory_ids in memory_sequences]


def _predict_correct_cached(
    scorer,
    query: Dict[str, Any],
    memory_ids: List[int],
    memories: List[Dict[str, Any]],
    cache: CorrectnessCache,
) -> bool:
    key = (query["query_id"], tuple(memory_ids))
    if key not in cache:
        prediction, _scores = scorer.predict(query, memory_ids, memories)
        cache[key] = prediction == query["answer"]
    return cache[key]


def _normalize_group(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, unbiased=False, keepdim=True)
    normalized = (values - mean) / std.clamp_min(eps)
    return torch.where(std > eps, normalized, torch.zeros_like(normalized))


def _compute_rewards(
    scorer,
    queries: List[Dict[str, Any]],
    memory_ids: torch.Tensor,
    memories: List[Dict[str, Any]],
    cache: ScoreCache,
    correctness_cache: CorrectnessCache,
    credit_mode: str,
    credit_gamma: float,
    reward_mode: str,
    correctness_bonus: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    batch_size, group_size, shot_num = memory_ids.shape
    if shot_num != 2:
        raise ValueError("GRPO reward currently expects shot_num=2")

    rewards = torch.zeros(batch_size, group_size, 2, dtype=torch.float32)
    step_rewards = torch.zeros(batch_size, group_size, 2, dtype=torch.float32)
    r0_values = []
    r1_values = []
    final_delta_values = []
    final_correct_values = []

    for batch_idx, query in enumerate(queries):
        rollout_ids = memory_ids[batch_idx].cpu().tolist()
        first_step = [[row[0]] for row in rollout_ids]
        full_step = [list(row) for row in rollout_ids]
        s0 = _score_sequences_cached(scorer, query, [[]], memories, cache)[0]
        s1 = _score_sequences_cached(scorer, query, first_step, memories, cache)
        s2 = _score_sequences_cached(scorer, query, full_step, memories, cache)

        for group_idx in range(group_size):
            final_delta = s2[group_idx] - s0
            final_correct = _predict_correct_cached(
                scorer=scorer,
                query=query,
                memory_ids=full_step[group_idx],
                memories=memories,
                cache=correctness_cache,
            )
            if reward_mode == "delta_logprob":
                r0 = s1[group_idx] - s0
                r1 = s2[group_idx] - s1[group_idx]
            elif reward_mode == "delta_plus_correctness":
                r0 = s1[group_idx] - s0
                r1 = s2[group_idx] - s1[group_idx] + correctness_bonus * int(final_correct)
            elif reward_mode == "correctness":
                r0 = 0.0
                r1 = correctness_bonus * int(final_correct)
            else:
                raise ValueError(f"Unsupported reward_mode: {reward_mode}")
            if credit_mode == "reward_to_go":
                g0 = r0 + r1
            elif credit_mode == "step":
                g0 = r0
            elif credit_mode == "discounted":
                g0 = r0 + credit_gamma * r1
            else:
                raise ValueError(f"Unsupported credit_mode: {credit_mode}")
            rewards[batch_idx, group_idx, 0] = g0
            rewards[batch_idx, group_idx, 1] = r1
            step_rewards[batch_idx, group_idx, 0] = r0
            step_rewards[batch_idx, group_idx, 1] = r1
            r0_values.append(r0)
            r1_values.append(r1)
            final_delta_values.append(final_delta)
            final_correct_values.append(float(final_correct))

    stats = {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "r0_mean": float(torch.tensor(r0_values).mean().item()) if r0_values else 0.0,
        "r1_mean": float(torch.tensor(r1_values).mean().item()) if r1_values else 0.0,
        "final_delta_mean": (
            float(torch.tensor(final_delta_values).mean().item())
            if final_delta_values
            else 0.0
        ),
        "final_correct_rate": (
            float(torch.tensor(final_correct_values).mean().item())
            if final_correct_values
            else 0.0
        ),
    }
    return rewards, step_rewards, stats


def _compute_correct_rate(
    scorer,
    queries: List[Dict[str, Any]],
    memory_ids: torch.Tensor,
    memories: List[Dict[str, Any]],
) -> float:
    correct = 0
    total = 0
    for batch_idx, query in enumerate(queries):
        for rollout_ids in memory_ids[batch_idx].cpu().tolist():
            prediction, _scores = scorer.predict(query, rollout_ids, memories)
            correct += int(prediction == query["answer"])
            total += 1
    return correct / total if total else 0.0


def _action_trace_rows(
    step: int,
    memory_ids: torch.Tensor,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    prefix_memory_id: int,
    memories: List[Dict[str, Any]],
    top_actions: int,
) -> List[Dict[str, Any]]:
    stats: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"a1": [], "g1": [], "old_logprob": [], "new_logprob": []}
    )
    memory_rows = memory_ids.detach().cpu()
    reward_rows = rewards.detach().cpu()
    advantage_rows = advantages.detach().cpu()
    old_rows = old_logprobs.detach().cpu()
    new_rows = new_logprobs.detach().cpu()
    batch_size, group_size, _shot_num = memory_rows.shape
    for batch_idx in range(batch_size):
        for group_idx in range(group_size):
            if int(memory_rows[batch_idx, group_idx, 0]) != prefix_memory_id:
                continue
            action_id = int(memory_rows[batch_idx, group_idx, 1])
            stats[action_id]["a1"].append(float(advantage_rows[batch_idx, group_idx, 1]))
            stats[action_id]["g1"].append(float(reward_rows[batch_idx, group_idx, 1]))
            stats[action_id]["old_logprob"].append(float(old_rows[batch_idx, group_idx, 1]))
            stats[action_id]["new_logprob"].append(float(new_rows[batch_idx, group_idx, 1]))

    rows = []
    for action_id, values in stats.items():
        count = len(values["a1"])
        mean_a1 = sum(values["a1"]) / count
        mean_g1 = sum(values["g1"]) / count
        mean_old = sum(values["old_logprob"]) / count
        mean_new = sum(values["new_logprob"]) / count
        rows.append(
            {
                "step": step,
                "prefix_memory_id": prefix_memory_id,
                "prefix_source_id": memories[prefix_memory_id]["source_id"],
                "action_memory_id": action_id,
                "action_source_id": memories[action_id]["source_id"],
                "count": count,
                "mean_a1": mean_a1,
                "mean_g1": mean_g1,
                "mean_old_logprob_step1": mean_old,
                "mean_new_logprob_step1": mean_new,
                "mean_logprob_delta_step1": mean_new - mean_old,
            }
        )
    rows.sort(key=lambda row: row["mean_a1"], reverse=True)
    return rows[:top_actions]


def _first_action_trace_rows(
    step: int,
    memory_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    memories: List[Dict[str, Any]],
    top_actions: int,
) -> List[Dict[str, Any]]:
    stats: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "r0": [],
            "r1": [],
            "g0": [],
            "a0": [],
            "old_logprob": [],
            "new_logprob": [],
        }
    )
    memory_rows = memory_ids.detach().cpu()
    step_reward_rows = step_rewards.detach().cpu()
    return_rows = returns.detach().cpu()
    advantage_rows = advantages.detach().cpu()
    old_rows = old_logprobs.detach().cpu()
    new_rows = new_logprobs.detach().cpu()
    batch_size, group_size, _shot_num = memory_rows.shape
    for batch_idx in range(batch_size):
        for group_idx in range(group_size):
            action_id = int(memory_rows[batch_idx, group_idx, 0])
            stats[action_id]["r0"].append(float(step_reward_rows[batch_idx, group_idx, 0]))
            stats[action_id]["r1"].append(float(step_reward_rows[batch_idx, group_idx, 1]))
            stats[action_id]["g0"].append(float(return_rows[batch_idx, group_idx, 0]))
            stats[action_id]["a0"].append(float(advantage_rows[batch_idx, group_idx, 0]))
            stats[action_id]["old_logprob"].append(float(old_rows[batch_idx, group_idx, 0]))
            stats[action_id]["new_logprob"].append(float(new_rows[batch_idx, group_idx, 0]))

    rows = []
    for action_id, values in stats.items():
        count = len(values["a0"])
        mean_r0 = sum(values["r0"]) / count
        mean_r1 = sum(values["r1"]) / count
        mean_g0 = sum(values["g0"]) / count
        mean_a0 = sum(values["a0"]) / count
        mean_old = sum(values["old_logprob"]) / count
        mean_new = sum(values["new_logprob"]) / count
        rows.append(
            {
                "step": step,
                "action_memory_id": action_id,
                "action_source_id": memories[action_id]["source_id"],
                "count": count,
                "mean_r0": mean_r0,
                "mean_r1": mean_r1,
                "mean_g0": mean_g0,
                "mean_a0": mean_a0,
                "mean_old_logprob_step0": mean_old,
                "mean_new_logprob_step0": mean_new,
                "mean_logprob_delta_step0": mean_new - mean_old,
            }
        )
    rows.sort(key=lambda row: row["mean_a0"], reverse=True)
    return rows[:top_actions]


def _marginal_entropy(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = torch.finfo(probs.dtype).eps
    step0 = probs[:, :, 0, :].reshape(-1, probs.shape[-1]).mean(dim=0)
    step1 = probs[:, :, 1, :].reshape(-1, probs.shape[-1]).mean(dim=0)
    entropy0 = -(step0 * step0.clamp_min(eps).log()).sum()
    entropy1 = -(step1 * step1.clamp_min(eps).log()).sum()
    return (entropy0 + entropy1) / 2.0, entropy0, entropy1


def _exact_kl_from_probs(policy_probs: torch.Tensor, reference_probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(policy_probs.dtype).eps
    policy = policy_probs.clamp_min(eps)
    reference = reference_probs.clamp_min(eps)
    return (policy * (policy.log() - reference.log())).sum(dim=-1)


def _save_checkpoint(
    path: Path,
    model: MathMemoryLeverLM,
    base_metadata: Dict[str, Any],
    extra_metadata: Dict[str, Any],
) -> None:
    metadata = checkpoint_metadata(model, {**base_metadata, **extra_metadata})
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)


def _split_grpo_indices(
    num_items: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    indices = list(range(num_items))
    if val_ratio <= 0 or num_items <= 1:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(num_items * val_ratio))
    val_count = min(val_count, num_items - 1)
    return indices[val_count:], indices[:val_count]


def _load_anchor_rows(anchor_file: str, valid_query_ids: set) -> List[Dict[str, Any]]:
    payload = json.load(open(anchor_file, encoding="utf-8"))
    rows = payload.get("data", payload)
    if not isinstance(rows, list):
        raise ValueError("--sft-anchor-file must contain a list or a {'data': [...]} payload")
    filtered = [row for row in rows if row.get("query_id") in valid_query_ids]
    if not filtered:
        raise ValueError("--sft-anchor-file has no rows matching GRPO train queries")
    return filtered


def _compute_sft_anchor_loss(
    model: MathMemoryLeverLM,
    anchor_rows: List[Dict[str, Any]],
    query_id_to_index: Dict[int, int],
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = [anchor_rows[rng.randrange(len(anchor_rows))] for _ in range(batch_size)]
    input_rows = []
    query_emb_rows = []
    memory_emb_rows = []
    for row in rows:
        memory_ids = [int(item) for item in row["memory_ids"]]
        input_rows.append(
            [model.bos_token_id, model.query_token_id, *memory_ids, model.eos_token_id]
        )
        query_emb_rows.append(query_embeddings[query_id_to_index[int(row["query_id"])]])
        memory_emb_rows.append(memory_embeddings[memory_ids])

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    query_emb = torch.stack(query_emb_rows, dim=0).to(device)
    memory_embs = torch.stack(memory_emb_rows, dim=0).to(device)
    output = model(
        input_ids=input_ids,
        query_emb=query_emb,
        memory_embs=memory_embs,
        labels=input_ids,
    )
    return output.loss


@torch.inference_mode()
def _generate_policy_retrievals(
    model: MathMemoryLeverLM,
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    shot_num: int,
    selection_mode: str,
    debias_query_embs: Optional[torch.Tensor],
    debias_batch_size: int,
    infer_batch_size: int,
) -> List[List[int]]:
    retrievals: List[List[int]] = []
    model.eval()
    for start in range(0, query_embeddings.shape[0], infer_batch_size):
        query_batch = query_embeddings[start : start + infer_batch_size]
        batch_ids = model.generate_memory_ids(
            query_embs=query_batch,
            memory_embedding_table=memory_embeddings,
            shot_num=shot_num,
            selection_mode=selection_mode,
            debias_query_embs=debias_query_embs,
            debias_batch_size=debias_batch_size,
        )
        retrievals.extend(batch_ids.cpu().tolist())
    return retrievals


def _retrieval_diversity(retrievals: List[List[int]]) -> Dict[str, int]:
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


def _evaluate_policy(
    model: MathMemoryLeverLM,
    reference_model: Optional[MathMemoryLeverLM],
    scorer,
    queries: List[Dict[str, Any]],
    query_embeddings: torch.Tensor,
    reference_query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    memories: List[Dict[str, Any]],
    shot_num: int,
    selection_mode: str,
    debias_query_embs: Optional[torch.Tensor],
    debias_batch_size: int,
    infer_batch_size: int,
) -> Dict[str, Any]:
    retrievals = _generate_policy_retrievals(
        model=model,
        query_embeddings=query_embeddings,
        memory_embeddings=memory_embeddings,
        shot_num=shot_num,
        selection_mode=selection_mode,
        debias_query_embs=debias_query_embs,
        debias_batch_size=debias_batch_size,
        infer_batch_size=infer_batch_size,
    )
    correct = 0
    final_deltas = []
    for query, memory_ids in zip(queries, retrievals):
        prediction, _scores = scorer.predict(query, memory_ids, memories)
        correct += int(prediction == query["answer"])
        empty_score, full_score = scorer.score_gold_sequences(
            query, [[], memory_ids], memories
        )
        final_deltas.append(full_score - empty_score)

    pair_changed_ratio = ""
    if reference_model is not None:
        reference_retrievals = _generate_policy_retrievals(
            model=reference_model,
            query_embeddings=reference_query_embeddings,
            memory_embeddings=memory_embeddings,
            shot_num=shot_num,
            selection_mode=selection_mode,
            debias_query_embs=debias_query_embs,
            debias_batch_size=debias_batch_size,
            infer_batch_size=infer_batch_size,
        )
        changed = sum(
            int(tuple(row) != tuple(ref_row))
            for row, ref_row in zip(retrievals, reference_retrievals)
        )
        pair_changed_ratio = changed / len(retrievals) if retrievals else ""

    diversity = _retrieval_diversity(retrievals)
    total = len(queries)
    return {
        "val_accuracy": correct / total if total else 0.0,
        "val_correct": correct,
        "val_total": total,
        "val_mean_final_delta": sum(final_deltas) / len(final_deltas)
        if final_deltas
        else 0.0,
        "val_pair_changed_ratio_vs_ref": pair_changed_ratio,
        **{f"val_{key}": value for key, value in diversity.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tune math-memory Lever-LM.")
    parser.add_argument("--init-mode", choices=["checkpoint", "scratch"], default="checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--shot-num", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--selection-mode",
        choices=["raw", "debiased_topk", "debiased_policy"],
        default="raw",
    )
    parser.add_argument("--debias-pool-size", type=int, default=256)
    parser.add_argument(
        "--credit-mode",
        choices=["reward_to_go", "step", "discounted"],
        default="reward_to_go",
    )
    parser.add_argument("--credit-gamma", type=float, default=0.3)
    parser.add_argument(
        "--reward-mode",
        choices=["delta_logprob", "delta_plus_correctness", "correctness"],
        default="delta_logprob",
    )
    parser.add_argument("--correctness-bonus", type=float, default=1.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--marginal-entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--ref-kl-coef", type=float, default=0.0)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--sft-anchor-file", default=None)
    parser.add_argument("--sft-anchor-coef", type=float, default=0.0)
    parser.add_argument("--sft-anchor-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--correctness-every", type=int, default=10)
    parser.add_argument("--grpo-val-ratio", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument(
        "--best-metric",
        choices=["train_window_final_delta", "val_accuracy", "val_final_delta"],
        default="train_window_final_delta",
    )
    parser.add_argument("--eval-infer-batch-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=16)
    parser.add_argument("--trace-prefix-source-id", default=None)
    parser.add_argument("--trace-top-actions", type=int, default=20)
    args = parser.parse_args()

    if args.init_mode == "checkpoint" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --init-mode checkpoint")
    if args.shot_num != 2:
        raise ValueError("This GRPO trainer currently supports --shot-num 2 only")
    if args.group_size <= 1:
        raise ValueError("--group-size must be > 1 for group-normalized advantages")
    if args.clip_eps <= 0:
        raise ValueError("--clip-eps must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.debias_pool_size < 0:
        raise ValueError("--debias-pool-size must be >= 0")
    if args.credit_gamma < 0:
        raise ValueError("--credit-gamma must be >= 0")
    if args.correctness_bonus < 0:
        raise ValueError("--correctness-bonus must be >= 0")
    if args.marginal_entropy_coef < 0:
        raise ValueError("--marginal-entropy-coef must be >= 0")
    if args.ref_kl_coef < 0:
        raise ValueError("--ref-kl-coef must be >= 0")
    if args.ref_kl_coef > 0 and not (args.reference_checkpoint or args.checkpoint):
        raise ValueError("--ref-kl-coef requires --reference-checkpoint or --checkpoint")
    if args.sft_anchor_coef < 0:
        raise ValueError("--sft-anchor-coef must be >= 0")
    if args.sft_anchor_coef > 0 and not args.sft_anchor_file:
        raise ValueError("--sft-anchor-coef > 0 requires --sft-anchor-file")
    if args.sft_anchor_batch_size <= 0:
        raise ValueError("--sft-anchor-batch-size must be > 0")
    if not 0 <= args.grpo_val_ratio < 1:
        raise ValueError("--grpo-val-ratio must be in [0, 1)")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.best_metric != "train_window_final_delta":
        if args.grpo_val_ratio <= 0:
            raise ValueError("--best-metric val_* requires --grpo-val-ratio > 0")
        if args.eval_every <= 0:
            raise ValueError("--best-metric val_* requires --eval-every > 0")
    if args.eval_infer_batch_size <= 0:
        raise ValueError("--eval-infer-batch-size must be > 0")
    if args.best_window <= 0:
        raise ValueError("--best-window must be > 0")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "grpo_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    payload = None
    metadata: Dict[str, Any] = {}
    if args.init_mode == "checkpoint":
        payload = torch.load(args.checkpoint, map_location="cpu")
        metadata = payload["metadata"]
    embedding_model = args.embedding_model or metadata.get(
        "embedding_model", "Qwen/Qwen3-Embedding-0.6B"
    )

    memories = load_experiences(args.experience_file)
    source_to_id = {memory["source_id"]: memory["memory_id"] for memory in memories}
    trace_prefix_memory_id = None
    if args.trace_prefix_source_id:
        if args.trace_prefix_source_id not in source_to_id:
            raise ValueError(f"Unknown --trace-prefix-source-id: {args.trace_prefix_source_id}")
        trace_prefix_memory_id = source_to_id[args.trace_prefix_source_id]
    if args.init_mode == "checkpoint" and metadata["memory_size"] != len(memories):
        raise ValueError(
            f"Checkpoint memory_size={metadata['memory_size']} does not match "
            f"experience count={len(memories)}"
        )

    train_queries, _test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    if args.query_limit is not None:
        train_queries = train_queries[: args.query_limit]
    if not train_queries:
        raise ValueError("No train queries available")

    embedder = build_embedder(
        embedding_model,
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        max_length=args.embedding_max_length,
        mock_emb_dim=args.mock_emb_dim,
    )
    safe_name = safe_model_name(embedding_model)
    cache_dir = Path(args.embedding_cache_dir)
    memory_embeddings = load_or_create_embeddings(
        str(cache_dir / f"memory_{safe_name}.pt"),
        [memory["text"] for memory in memories],
        embedder,
        batch_size=args.embedding_batch_size,
    )
    ratio_name = str(args.train_ratio).replace(".", "p")
    query_embeddings = load_or_create_embeddings(
        str(cache_dir / f"grpo_train_queries_{safe_name}_seed{args.seed}_ratio{ratio_name}.pt"),
        [query_to_text(query) for query in train_queries],
        embedder,
        batch_size=args.embedding_batch_size,
    )
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if args.init_mode == "checkpoint":
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
        base_checkpoint = str(Path(args.checkpoint).resolve())
        base_training_stage = metadata.get("training_stage", "sft")
    else:
        model = MathMemoryLeverLM(
            memory_size=len(memories),
            encoder_emb_dim=memory_embeddings.shape[-1],
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            max_positions=args.max_positions,
            model_backend="gpt2",
        )
        base_checkpoint = None
        base_training_stage = None
    model.to(run_device)
    model.train()

    reference_model = None
    reference_checkpoint = args.reference_checkpoint or args.checkpoint
    if args.ref_kl_coef > 0:
        ref_payload = torch.load(reference_checkpoint, map_location="cpu")
        ref_metadata = ref_payload["metadata"]
        if ref_metadata["memory_size"] != len(memories):
            raise ValueError(
                f"Reference memory_size={ref_metadata['memory_size']} does not match "
                f"experience count={len(memories)}"
            )
        reference_model = MathMemoryLeverLM(
            memory_size=ref_metadata["memory_size"],
            encoder_emb_dim=ref_metadata["encoder_emb_dim"],
            n_embd=ref_metadata["n_embd"],
            n_head=ref_metadata["n_head"],
            n_layer=ref_metadata["n_layer"],
            max_positions=ref_metadata["max_positions"],
            model_backend=ref_metadata.get("model_backend", "gpt2"),
        )
        reference_model.load_state_dict(ref_payload["model"], strict=False)
        reference_model.to(run_device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    memory_embeddings = memory_embeddings.to(run_device)
    query_embeddings = query_embeddings.to(run_device)
    grpo_train_indices, grpo_val_indices = _split_grpo_indices(
        len(train_queries), args.grpo_val_ratio, args.seed
    )
    grpo_train_queries = [train_queries[index] for index in grpo_train_indices]
    grpo_val_queries = [train_queries[index] for index in grpo_val_indices]
    if not grpo_train_queries:
        raise ValueError("No GRPO train queries available after validation split")
    query_id_to_index = {
        int(query["query_id"]): index for index, query in enumerate(train_queries)
    }
    anchor_rows = None
    if args.sft_anchor_coef > 0:
        anchor_rows = _load_anchor_rows(
            args.sft_anchor_file,
            {int(query["query_id"]) for query in grpo_train_queries},
        )
    if args.selection_mode in {"debiased_topk", "debiased_policy"}:
        if args.debias_pool_size == 0:
            debias_query_embs = query_embeddings
        else:
            debias_query_embs = query_embeddings[: min(args.debias_pool_size, len(query_embeddings))]
    else:
        debias_query_embs = None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scorer = build_scorer(
        model_name=args.scorer_model,
        device=args.scorer_device,
        dtype=args.scorer_dtype,
        batch_size=args.scorer_batch_size,
        max_length=args.scorer_max_length,
    )

    history_path = output_dir / "grpo_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "policy_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "mean_abs_logprob_delta",
                "entropy_step0",
                "entropy_step1",
                "marginal_entropy",
                "marginal_entropy_step0",
                "marginal_entropy_step1",
                "reference_kl",
                "reference_kl_step0",
                "reference_kl_step1",
                "sft_anchor_loss",
                "max_prob_step0",
                "max_prob_step1",
                "reward_mean",
                "reward_std",
                "r0_mean",
                "r1_mean",
                "final_delta_mean",
                "final_correct_rate",
                "correct_rate",
                "unique_memory_count",
                "window_final_delta_mean",
                "is_best",
                "bad_windows",
            ],
        )
        writer.writeheader()

    eval_history_path = output_dir / "grpo_eval_history.csv"
    if args.eval_every > 0 and grpo_val_queries:
        with eval_history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "val_accuracy",
                    "val_correct",
                    "val_total",
                    "val_mean_final_delta",
                    "val_pair_changed_ratio_vs_ref",
                    "val_unique_first_memory_count",
                    "val_unique_second_memory_count",
                    "val_unique_pair_count",
                    "is_best",
                    "bad_windows",
                ],
            )
            writer.writeheader()

    action_trace_path = output_dir / "grpo_action_trace.csv"
    if trace_prefix_memory_id is not None:
        with action_trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "prefix_memory_id",
                    "prefix_source_id",
                    "action_memory_id",
                    "action_source_id",
                    "count",
                    "mean_a1",
                    "mean_g1",
                    "mean_old_logprob_step1",
                    "mean_new_logprob_step1",
                    "mean_logprob_delta_step1",
                ],
            )
            writer.writeheader()

    first_action_trace_path = output_dir / "grpo_first_action_trace.csv"
    with first_action_trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "action_memory_id",
                "action_source_id",
                "count",
                "mean_r0",
                "mean_r1",
                "mean_g0",
                "mean_a0",
                "mean_old_logprob_step0",
                "mean_new_logprob_step0",
                "mean_logprob_delta_step0",
            ],
        )
        writer.writeheader()

    base_metadata = {
        "training_stage": "grpo",
        "init_mode": args.init_mode,
        "base_checkpoint": base_checkpoint,
        "base_training_stage": base_training_stage,
        "reward_mode": f"{args.reward_mode}_{args.credit_mode}",
        "embedding_model": embedding_model,
        "experience_file": str(Path(args.experience_file).resolve()),
        "history_file": str(history_path.resolve()),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "shot_num": args.shot_num,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "selection_mode": args.selection_mode,
        "debias_pool_size": args.debias_pool_size,
        "credit_mode": args.credit_mode,
        "credit_gamma": args.credit_gamma,
        "correctness_bonus": args.correctness_bonus,
        "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "marginal_entropy_coef": args.marginal_entropy_coef,
        "kl_coef": args.kl_coef,
        "ref_kl_coef": args.ref_kl_coef,
        "reference_checkpoint": (
            str(Path(reference_checkpoint).resolve()) if reference_checkpoint else None
        ),
        "sft_anchor_file": (
            str(Path(args.sft_anchor_file).resolve()) if args.sft_anchor_file else None
        ),
        "sft_anchor_coef": args.sft_anchor_coef,
        "sft_anchor_batch_size": args.sft_anchor_batch_size,
        "max_steps": args.max_steps,
        "checkpoint_steps": checkpoint_steps,
        "best_window": args.best_window,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "grpo_val_ratio": args.grpo_val_ratio,
        "eval_every": args.eval_every,
        "eval_history_file": str(eval_history_path.resolve())
        if args.eval_every > 0 and grpo_val_queries
        else None,
        "best_metric": args.best_metric,
        "grpo_train_query_count": len(grpo_train_queries),
        "grpo_val_query_count": len(grpo_val_queries),
        "trace_prefix_source_id": args.trace_prefix_source_id,
        "trace_top_actions": args.trace_top_actions,
    }
    score_cache: ScoreCache = {}
    correctness_cache: CorrectnessCache = {}
    rng = random.Random(args.seed)
    anchor_rng = random.Random(args.seed + 17)
    best_window_values = deque(maxlen=args.best_window)
    best_window_reward = float("-inf")
    best_metric_value = float("-inf")
    best_step = None
    bad_windows = 0
    stopped_step = None

    if 0 in checkpoint_steps:
        _save_checkpoint(
            output_dir / "init.pt",
            model,
            base_metadata,
            {
                "current_step": -1,
                "update_count": 0,
                "best_step": best_step,
                "best_window_final_delta_mean": None,
                "best_metric": args.best_metric,
                "best_metric_value": None,
                "bad_windows": bad_windows,
                "early_stopped": False,
                "stopped_step": None,
                "checkpoint_label": "init",
            },
        )

    progress = tqdm(range(args.max_steps), desc="GRPO", ncols=100)
    for step in progress:
        batch_indices = [
            grpo_train_indices[rng.randrange(len(grpo_train_indices))]
            for _ in range(args.batch_size)
        ]
        batch_queries = [train_queries[index] for index in batch_indices]
        batch_query_embs = query_embeddings[batch_indices].to(run_device)

        model.eval()
        with torch.no_grad():
            memory_ids, old_logprobs, _old_entropies = model.sample_memory_ids(
                query_embs=batch_query_embs,
                memory_embedding_table=memory_embeddings,
                shot_num=args.shot_num,
                group_size=args.group_size,
                temperature=args.temperature,
                top_k=args.top_k,
                selection_mode=args.selection_mode,
                debias_query_embs=debias_query_embs,
                debias_batch_size=args.embedding_batch_size,
            )

        rewards_cpu, step_rewards_cpu, reward_stats = _compute_rewards(
            scorer=scorer,
            queries=batch_queries,
            memory_ids=memory_ids,
            memories=memories,
            cache=score_cache,
            correctness_cache=correctness_cache,
            credit_mode=args.credit_mode,
            credit_gamma=args.credit_gamma,
            reward_mode=args.reward_mode,
            correctness_bonus=args.correctness_bonus,
        )
        advantages_cpu = _normalize_group(rewards_cpu)
        advantages = advantages_cpu.to(run_device)
        old_logprobs = old_logprobs.detach().to(run_device)
        memory_ids = memory_ids.to(run_device)

        # Keep dropout disabled for policy-gradient logprob computation. Gradients
        # still flow in eval mode, and this keeps old/new/reference policies
        # comparable when measuring PPO ratio and reference KL.
        model.eval()
        new_logprobs, entropies, max_probs, probs = model.compute_action_logprobs(
            query_embs=batch_query_embs,
            memory_embedding_table=memory_embeddings,
            memory_ids=memory_ids,
            temperature=args.temperature,
            selection_mode=args.selection_mode,
            debias_query_embs=debias_query_embs,
            debias_batch_size=args.embedding_batch_size,
            return_max_probs=True,
            return_probs=True,
        )
        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
        ) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = entropies.mean()
        marginal_entropy, marginal_entropy_step0, marginal_entropy_step1 = (
            _marginal_entropy(probs)
        )
        if reference_model is not None:
            with torch.no_grad():
                _ref_logprobs, _ref_entropies, ref_probs = (
                    reference_model.compute_action_logprobs(
                        query_embs=batch_query_embs,
                        memory_embedding_table=memory_embeddings,
                        memory_ids=memory_ids,
                        temperature=args.temperature,
                        selection_mode=args.selection_mode,
                        debias_query_embs=debias_query_embs,
                        debias_batch_size=args.embedding_batch_size,
                        return_probs=True,
                    )
                )
            reference_kl_steps = _exact_kl_from_probs(probs, ref_probs)
            reference_kl = reference_kl_steps.mean()
            reference_kl_step0 = reference_kl_steps[:, :, 0].mean()
            reference_kl_step1 = reference_kl_steps[:, :, 1].mean()
        else:
            reference_kl = torch.zeros((), dtype=policy_loss.dtype, device=run_device)
            reference_kl_step0 = reference_kl
            reference_kl_step1 = reference_kl
        if args.sft_anchor_coef > 0:
            sft_anchor_loss = _compute_sft_anchor_loss(
                model=model,
                anchor_rows=anchor_rows,
                query_id_to_index=query_id_to_index,
                query_embeddings=query_embeddings,
                memory_embeddings=memory_embeddings,
                rng=anchor_rng,
                batch_size=args.sft_anchor_batch_size,
                device=run_device,
            )
        else:
            sft_anchor_loss = torch.zeros((), dtype=policy_loss.dtype, device=run_device)
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (
            (ratio < 1.0 - args.clip_eps) | (ratio > 1.0 + args.clip_eps)
        ).float().mean()
        mean_abs_logprob_delta = log_ratio.abs().mean()
        loss = (
            policy_loss
            - args.entropy_coef * entropy
            - args.marginal_entropy_coef * marginal_entropy
            + args.kl_coef * approx_kl
            + args.ref_kl_coef * reference_kl
            + args.sft_anchor_coef * sft_anchor_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        correct_rate = ""
        if args.correctness_every > 0 and step % args.correctness_every == 0:
            model.eval()
            correct_rate = _compute_correct_rate(
                scorer=scorer,
                queries=batch_queries,
                memory_ids=memory_ids.detach().cpu(),
                memories=memories,
            )

        unique_memory_count = len(set(memory_ids.detach().cpu().reshape(-1).tolist()))
        best_window_values.append(reward_stats["final_delta_mean"])
        window_reward = sum(best_window_values) / len(best_window_values)
        train_window_improved = window_reward > best_window_reward + args.early_stop_min_delta
        if train_window_improved:
            best_window_reward = window_reward

        val_stats: Dict[str, Any] = {}
        is_eval_step = (
            args.eval_every > 0
            and bool(grpo_val_queries)
            and step % args.eval_every == 0
        )
        if is_eval_step:
            val_query_embs = query_embeddings[grpo_val_indices].to(run_device)
            val_stats = _evaluate_policy(
                model=model,
                reference_model=reference_model,
                scorer=scorer,
                queries=grpo_val_queries,
                query_embeddings=val_query_embs,
                reference_query_embeddings=val_query_embs,
                memory_embeddings=memory_embeddings,
                memories=memories,
                shot_num=args.shot_num,
                selection_mode=args.selection_mode,
                debias_query_embs=debias_query_embs,
                debias_batch_size=args.embedding_batch_size,
                infer_batch_size=args.eval_infer_batch_size,
            )

        if args.best_metric == "train_window_final_delta":
            is_best = train_window_improved
            metric_value = window_reward
        elif args.best_metric == "val_accuracy":
            is_best = bool(is_eval_step) and (
                val_stats["val_accuracy"] > best_metric_value + args.early_stop_min_delta
            )
            metric_value = val_stats["val_accuracy"] if is_eval_step else None
        elif args.best_metric == "val_final_delta":
            is_best = bool(is_eval_step) and (
                val_stats["val_mean_final_delta"]
                > best_metric_value + args.early_stop_min_delta
            )
            metric_value = val_stats["val_mean_final_delta"] if is_eval_step else None
        else:
            raise ValueError(f"Unsupported best_metric: {args.best_metric}")

        if is_best:
            if metric_value is not None:
                best_metric_value = metric_value
            best_step = step
            bad_windows = 0
        elif (
            args.early_stop_patience > 0
            and (
                args.best_metric == "train_window_final_delta"
                and len(best_window_values) == args.best_window
                or args.best_metric != "train_window_final_delta"
                and is_eval_step
            )
        ):
            bad_windows += 1

        row = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_fraction": float(clip_fraction.detach().cpu().item()),
            "mean_abs_logprob_delta": float(
                mean_abs_logprob_delta.detach().cpu().item()
            ),
            "entropy_step0": float(entropies[:, :, 0].mean().detach().cpu().item()),
            "entropy_step1": float(entropies[:, :, 1].mean().detach().cpu().item()),
            "marginal_entropy": float(marginal_entropy.detach().cpu().item()),
            "marginal_entropy_step0": float(
                marginal_entropy_step0.detach().cpu().item()
            ),
            "marginal_entropy_step1": float(
                marginal_entropy_step1.detach().cpu().item()
            ),
            "reference_kl": float(reference_kl.detach().cpu().item()),
            "reference_kl_step0": float(reference_kl_step0.detach().cpu().item()),
            "reference_kl_step1": float(reference_kl_step1.detach().cpu().item()),
            "sft_anchor_loss": float(sft_anchor_loss.detach().cpu().item()),
            "max_prob_step0": float(max_probs[:, :, 0].mean().detach().cpu().item()),
            "max_prob_step1": float(max_probs[:, :, 1].mean().detach().cpu().item()),
            "reward_mean": reward_stats["reward_mean"],
            "reward_std": reward_stats["reward_std"],
            "r0_mean": reward_stats["r0_mean"],
            "r1_mean": reward_stats["r1_mean"],
            "final_delta_mean": reward_stats["final_delta_mean"],
            "final_correct_rate": reward_stats["final_correct_rate"],
            "correct_rate": correct_rate,
            "unique_memory_count": unique_memory_count,
            "window_final_delta_mean": window_reward,
            "is_best": int(is_best),
            "bad_windows": bad_windows,
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        if is_eval_step:
            eval_row = {
                "step": step,
                **val_stats,
                "is_best": int(is_best),
                "bad_windows": bad_windows,
            }
            with eval_history_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(eval_row.keys()))
                writer.writerow(eval_row)

        if trace_prefix_memory_id is not None:
            trace_rows = _action_trace_rows(
                step=step,
                memory_ids=memory_ids,
                rewards=rewards_cpu,
                advantages=advantages_cpu,
                old_logprobs=old_logprobs,
                new_logprobs=new_logprobs,
                prefix_memory_id=trace_prefix_memory_id,
                memories=memories,
                top_actions=args.trace_top_actions,
            )
            if trace_rows:
                with action_trace_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                    writer.writerows(trace_rows)

        first_trace_rows = _first_action_trace_rows(
            step=step,
            memory_ids=memory_ids,
            step_rewards=step_rewards_cpu,
            returns=rewards_cpu,
            advantages=advantages_cpu,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            memories=memories,
            top_actions=args.trace_top_actions,
        )
        if first_trace_rows:
            with first_action_trace_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(first_trace_rows[0].keys())
                )
                writer.writerows(first_trace_rows)

        if is_best:
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                base_metadata,
                {
                    "current_step": step,
                    "update_count": step + 1,
                    "best_step": best_step,
                    "best_window_final_delta_mean": best_window_reward,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value,
                    "bad_windows": bad_windows,
                    "early_stopped": False,
                    "stopped_step": None,
                },
            )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            _save_checkpoint(
                output_dir / "last.pt",
                model,
                base_metadata,
                {
                    "current_step": step,
                    "update_count": step + 1,
                    "best_step": best_step,
                    "best_window_final_delta_mean": best_window_reward,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value,
                    "bad_windows": bad_windows,
                    "early_stopped": False,
                    "stopped_step": None,
                },
            )

        update_count = step + 1
        if update_count in checkpoint_steps:
            _save_checkpoint(
                output_dir / f"step_{update_count:06d}.pt",
                model,
                base_metadata,
                {
                    "current_step": step,
                    "update_count": update_count,
                    "best_step": best_step,
                    "best_window_final_delta_mean": best_window_reward,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value,
                    "bad_windows": bad_windows,
                    "early_stopped": False,
                    "stopped_step": None,
                    "checkpoint_label": f"step_{update_count:06d}",
                },
            )

        progress.set_postfix(
            loss=f"{row['loss']:.4f}",
            reward=f"{row['reward_mean']:.4f}",
            ent=f"{row['entropy']:.2f}",
            bad=bad_windows,
        )

        if args.early_stop_patience > 0 and bad_windows >= args.early_stop_patience:
            stopped_step = step
            print(
                f"Early stopping at step={step}; best_step={best_step} "
                f"best_window_final_delta_mean={best_window_reward:.6f}"
            )
            break

    final_step = stopped_step if stopped_step is not None else args.max_steps - 1
    _save_checkpoint(
        output_dir / "last.pt",
        model,
        base_metadata,
        {
            "current_step": final_step,
            "update_count": final_step + 1 if final_step >= 0 else 0,
            "best_step": best_step,
            "best_window_final_delta_mean": best_window_reward,
            "best_metric": args.best_metric,
            "best_metric_value": best_metric_value,
            "bad_windows": bad_windows,
            "early_stopped": stopped_step is not None,
            "stopped_step": stopped_step,
        },
    )
    print(f"Saved GRPO checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
