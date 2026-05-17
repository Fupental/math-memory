import argparse
import csv
import json
import random
import shlex
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
    query_to_text,
    safe_model_name,
)
from lever_lm.math_memory.embeddings import build_embedder, load_or_create_embeddings
from lever_lm.math_memory.model import PointerMemoryLeverLM, pointer_checkpoint_metadata
from lever_lm.math_memory.scoring import build_scorer
from math_memory_grpo_train import (
    _compute_rewards,
    _normalize_group,
    _parse_checkpoint_steps,
)
from math_memory_pointer_eval import _candidate_ids_for_batch


def _save_command(output_dir: Path) -> None:
    with (output_dir / "train_command.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n")


def _load_pointer_model(checkpoint_path: str, device: torch.device) -> Tuple[PointerMemoryLeverLM, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = payload["metadata"]
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
    model.to(device)
    return model, metadata


def _sample_pointer_rollouts(
    model: PointerMemoryLeverLM,
    query_embs: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_embeddings: torch.Tensor,
    group_size: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    batch_size, candidate_num = candidate_ids.shape
    flat_query = query_embs.repeat_interleave(group_size, dim=0)
    flat_candidate_ids = candidate_ids.repeat_interleave(group_size, dim=0)
    flat_candidate_embs = memory_embeddings[flat_candidate_ids]

    out1 = model(flat_query, flat_candidate_embs)
    dist1 = Categorical(logits=out1.logits1 / temperature)
    first_local = dist1.sample()
    old_logprob1 = dist1.log_prob(first_local)
    entropy1 = dist1.entropy()

    out2 = model(flat_query, flat_candidate_embs, selected_indices=first_local)
    dist2 = Categorical(logits=out2.logits2 / temperature)
    second_local = dist2.sample()
    old_logprob2 = dist2.log_prob(second_local)
    entropy2 = dist2.entropy()

    local_actions = torch.stack([first_local, second_local], dim=-1)
    memory_ids = flat_candidate_ids.gather(1, local_actions).reshape(
        batch_size, group_size, 2
    )
    old_logprobs = torch.stack([old_logprob1, old_logprob2], dim=-1).reshape(
        batch_size, group_size, 2
    )
    entropies = torch.stack([entropy1, entropy2], dim=-1).reshape(
        batch_size, group_size, 2
    )
    local_actions = local_actions.reshape(batch_size, group_size, 2)
    if candidate_num != model.candidate_num:
        raise ValueError(f"Expected candidate_num={model.candidate_num}, got {candidate_num}")
    return memory_ids, local_actions, old_logprobs.detach(), entropies.detach()


def _pointer_logprobs_and_stats(
    model: PointerMemoryLeverLM,
    query_embs: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_embeddings: torch.Tensor,
    local_actions: torch.Tensor,
    temperature: float,
    ref_model: PointerMemoryLeverLM | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, group_size, _shot = local_actions.shape
    flat_query = query_embs.repeat_interleave(group_size, dim=0)
    flat_candidate_ids = candidate_ids.repeat_interleave(group_size, dim=0)
    flat_candidate_embs = memory_embeddings[flat_candidate_ids]
    flat_actions = local_actions.reshape(batch_size * group_size, 2)

    out1 = model(flat_query, flat_candidate_embs)
    logits1 = out1.logits1 / temperature
    dist1 = Categorical(logits=logits1)
    logp1 = dist1.logits
    probs1 = dist1.probs
    action_logp1 = dist1.log_prob(flat_actions[:, 0])
    entropy1 = dist1.entropy()

    out2 = model(flat_query, flat_candidate_embs, selected_indices=flat_actions[:, 0])
    logits2 = out2.logits2 / temperature
    dist2 = Categorical(logits=logits2)
    logp2 = dist2.logits
    probs2 = dist2.probs
    action_logp2 = dist2.log_prob(flat_actions[:, 1])
    entropy2 = dist2.entropy()

    ref_kl = torch.zeros_like(action_logp1)
    if ref_model is not None:
        with torch.no_grad():
            ref_out1 = ref_model(flat_query, flat_candidate_embs)
            ref_logp1 = Categorical(logits=ref_out1.logits1 / temperature).logits
            ref_out2 = ref_model(
                flat_query, flat_candidate_embs, selected_indices=flat_actions[:, 0]
            )
            ref_logp2 = Categorical(logits=ref_out2.logits2 / temperature).logits
        diff1 = torch.where(probs1 > 0, logp1 - ref_logp1, torch.zeros_like(logp1))
        diff2 = torch.where(probs2 > 0, logp2 - ref_logp2, torch.zeros_like(logp2))
        ref_kl1 = (probs1 * diff1).sum(dim=-1)
        ref_kl2 = (probs2 * diff2).sum(dim=-1)
        ref_kl = 0.5 * (ref_kl1 + ref_kl2)

    logprobs = torch.stack([action_logp1, action_logp2], dim=-1).reshape(
        batch_size, group_size, 2
    )
    entropies = torch.stack([entropy1, entropy2], dim=-1).reshape(
        batch_size, group_size, 2
    )
    max_probs = torch.stack(
        [probs1.max(dim=-1).values, probs2.max(dim=-1).values], dim=-1
    ).reshape(batch_size, group_size, 2)
    ref_kl = ref_kl.reshape(batch_size, group_size)
    return logprobs, entropies, max_probs, ref_kl


def _advantages_from_rewards(rewards: torch.Tensor) -> torch.Tensor:
    step_advantages = []
    for step in range(rewards.shape[-1]):
        step_advantages.append(_normalize_group(rewards[:, :, step]))
    return torch.stack(step_advantages, dim=-1)


def _save_checkpoint(
    path: Path,
    model: PointerMemoryLeverLM,
    metadata_extra: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": pointer_checkpoint_metadata(model, metadata_extra),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for Pointer Lever-LM math-memory retrieval."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device-map", default=None)
    parser.add_argument("--embedding-max-memory", default=None)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--candidate-num", type=int, default=64)
    parser.add_argument("--candidate-seed", type=int, default=42)
    parser.add_argument(
        "--candidate-mode",
        choices=["random", "semantic", "mixed"],
        default="random",
    )
    parser.add_argument("--random-candidate-num", type=int, default=32)
    parser.add_argument("--semantic-candidate-num", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        choices=["delta_logprob", "delta_plus_correctness", "correctness"],
        default="delta_logprob",
    )
    parser.add_argument(
        "--credit-mode",
        choices=["step", "reward_to_go", "discounted"],
        default="step",
    )
    parser.add_argument("--credit-gamma", type=float, default=0.3)
    parser.add_argument("--correctness-bonus", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ref-kl-coef", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--best-metric", choices=["train_window_final_delta"], default="train_window_final_delta")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--scorer-device-map", default=None)
    parser.add_argument("--scorer-max-memory", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.group_size <= 1:
        raise ValueError("--group-size must be > 1 for group normalization")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.candidate_mode == "mixed":
        if args.random_candidate_num + args.semantic_candidate_num != args.candidate_num:
            raise ValueError(
                "--random-candidate-num + --semantic-candidate-num must equal --candidate-num"
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_command(output_dir)
    history_path = output_dir / "pointer_grpo_history.csv"
    checkpoint_steps = set(_parse_checkpoint_steps(args.checkpoint_steps))

    run_device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, base_metadata = _load_pointer_model(args.checkpoint, run_device)
    reference_path = args.reference_checkpoint or args.checkpoint
    ref_model = None
    if args.ref_kl_coef > 0:
        ref_model, _ref_metadata = _load_pointer_model(reference_path, run_device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    if args.candidate_num != model.candidate_num:
        raise ValueError(
            f"Checkpoint expects candidate_num={model.candidate_num}; got {args.candidate_num}"
        )

    memories = load_experiences(args.experience_file)
    train_queries, _test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    if args.train_limit is not None:
        train_queries = train_queries[: args.train_limit]
    query_by_id = {query["query_id"]: query for query in train_queries}

    embedding_model = args.embedding_model or base_metadata.get(
        "embedding_model", "Qwen/Qwen3-Embedding-0.6B"
    )
    embedder = build_embedder(
        embedding_model,
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        max_length=args.embedding_max_length,
        mock_emb_dim=args.mock_emb_dim,
        device_map=args.embedding_device_map,
        max_memory=args.embedding_max_memory,
    )
    safe_name = safe_model_name(embedding_model)
    cache_dir = Path(args.embedding_cache_dir)
    memory_embeddings = load_or_create_embeddings(
        str(cache_dir / f"memory_{safe_name}.pt"),
        [memory["text"] for memory in memories],
        embedder,
        batch_size=args.embedding_batch_size,
    ).to(run_device)
    query_embeddings_tensor = load_or_create_embeddings(
        str(cache_dir / f"train_queries_{safe_name}_seed{args.seed}_ratio{args.train_ratio}.pt"),
        [query_to_text(query) for query in train_queries],
        embedder,
        batch_size=args.embedding_batch_size,
    )
    query_embeddings = {
        query["query_id"]: query_embeddings_tensor[index]
        for index, query in enumerate(train_queries)
    }
    del embedder
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    metadata_extra_base = {
        "training_stage": "pointer_grpo",
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "reference_checkpoint": str(Path(reference_path).resolve()),
        "embedding_model": embedding_model,
        "experience_file": str(Path(args.experience_file).resolve()),
        "memory_size": len(memories),
        "shot_num": 2,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "candidate_seed": args.candidate_seed,
        "candidate_mode": args.candidate_mode,
        "random_candidate_num": args.random_candidate_num,
        "semantic_candidate_num": args.semantic_candidate_num,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "reward_mode": args.reward_mode,
        "credit_mode": args.credit_mode,
        "credit_gamma": args.credit_gamma,
        "correctness_bonus": args.correctness_bonus,
        "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef,
        "ref_kl_coef": args.ref_kl_coef,
        "best_metric": args.best_metric,
        "loss_history_file": str(history_path.resolve()),
    }
    _save_checkpoint(output_dir / "init.pt", model, {**metadata_extra_base, "current_step": 0})

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "policy_loss",
                "entropy",
                "ref_kl",
                "approx_kl",
                "clip_fraction",
                "reward_mean",
                "reward_std",
                "r0_mean",
                "r1_mean",
                "final_delta_mean",
                "final_correct_rate",
                "window_final_delta",
                "best_metric_value",
                "best_step",
                "unique_pair_count",
                "unique_first_memory_count",
                "unique_second_memory_count",
                "max_prob_step0",
                "max_prob_step1",
            ],
        )
        writer.writeheader()

    score_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    correctness_cache: Dict[Tuple[int, Tuple[int, ...]], bool] = {}
    metric_window = deque(maxlen=max(1, args.best_window))
    best_metric_value = None
    best_step = None
    bad_steps = 0
    rng = random.Random(args.seed)
    max_steps = 2 if args.fast_dev_run else args.max_steps

    for step in tqdm(range(1, max_steps + 1), desc="Pointer GRPO", ncols=120):
        batch_queries = [rng.choice(train_queries) for _ in range(args.batch_size)]
        query_embs = torch.stack(
            [query_embeddings[query["query_id"]] for query in batch_queries], dim=0
        ).to(run_device)
        candidate_ids = _candidate_ids_for_batch(
            mode=args.candidate_mode,
            query_batch=batch_queries,
            query_embs=query_embs,
            memory_embeddings=memory_embeddings,
            memory_size=len(memories),
            candidate_num=args.candidate_num,
            candidate_seed=args.candidate_seed,
            random_candidate_num=args.random_candidate_num,
            semantic_candidate_num=args.semantic_candidate_num,
            device=run_device,
        )

        model.eval()
        with torch.no_grad():
            memory_ids, local_actions, old_logprobs, _old_entropies = _sample_pointer_rollouts(
                model=model,
                query_embs=query_embs,
                candidate_ids=candidate_ids,
                memory_embeddings=memory_embeddings,
                group_size=args.group_size,
                temperature=args.temperature,
            )
        rewards, _step_rewards, reward_stats = _compute_rewards(
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
        advantages = _advantages_from_rewards(rewards).to(run_device)
        rewards = rewards.to(run_device)
        old_logprobs = old_logprobs.to(run_device)
        local_actions = local_actions.to(run_device)

        model.train()
        new_logprobs, entropies, max_probs, ref_kl = _pointer_logprobs_and_stats(
            model=model,
            query_embs=query_embs,
            candidate_ids=candidate_ids,
            memory_embeddings=memory_embeddings,
            local_actions=local_actions,
            temperature=args.temperature,
            ref_model=ref_model,
        )
        logprob_delta = new_logprobs - old_logprobs
        ratio = torch.exp(logprob_delta)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        entropy = entropies.mean()
        ref_kl_mean = ref_kl.mean()
        loss = policy_loss - args.entropy_coef * entropy + args.ref_kl_coef * ref_kl_mean

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        metric_window.append(reward_stats["final_delta_mean"])
        window_final_delta = sum(metric_window) / len(metric_window)
        current_metric = window_final_delta
        is_best = False
        if best_metric_value is None or current_metric > best_metric_value + args.early_stop_min_delta:
            best_metric_value = current_metric
            best_step = step
            bad_steps = 0
            is_best = True
        elif args.early_stop_patience > 0:
            bad_steps += 1

        pair_rows = [tuple(row) for row in memory_ids.reshape(-1, 2).cpu().tolist()]
        first_rows = [row[0] for row in pair_rows]
        second_rows = [row[1] for row in pair_rows]
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "loss",
                    "policy_loss",
                    "entropy",
                    "ref_kl",
                    "approx_kl",
                    "clip_fraction",
                    "reward_mean",
                    "reward_std",
                    "r0_mean",
                    "r1_mean",
                    "final_delta_mean",
                    "final_correct_rate",
                    "window_final_delta",
                    "best_metric_value",
                    "best_step",
                    "unique_pair_count",
                    "unique_first_memory_count",
                    "unique_second_memory_count",
                    "max_prob_step0",
                    "max_prob_step1",
                ],
            )
            writer.writerow(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "ref_kl": float(ref_kl_mean.detach().cpu()),
                    "approx_kl": float((-logprob_delta).mean().detach().cpu()),
                    "clip_fraction": float(
                        (torch.abs(ratio - 1.0) > args.clip_eps).float().mean().detach().cpu()
                    ),
                    "reward_mean": reward_stats["reward_mean"],
                    "reward_std": reward_stats["reward_std"],
                    "r0_mean": reward_stats["r0_mean"],
                    "r1_mean": reward_stats["r1_mean"],
                    "final_delta_mean": reward_stats["final_delta_mean"],
                    "final_correct_rate": reward_stats["final_correct_rate"],
                    "window_final_delta": window_final_delta,
                    "best_metric_value": best_metric_value,
                    "best_step": best_step,
                    "unique_pair_count": len(set(pair_rows)),
                    "unique_first_memory_count": len(set(first_rows)),
                    "unique_second_memory_count": len(set(second_rows)),
                    "max_prob_step0": float(max_probs[:, :, 0].mean().detach().cpu()),
                    "max_prob_step1": float(max_probs[:, :, 1].mean().detach().cpu()),
                }
            )

        metadata_extra = {
            **metadata_extra_base,
            "current_step": step,
            "best_step": best_step,
            "best_metric_value": best_metric_value,
            "bad_steps": bad_steps,
            "last_final_delta_mean": reward_stats["final_delta_mean"],
            "last_window_final_delta": window_final_delta,
        }
        _save_checkpoint(output_dir / "last.pt", model, metadata_extra)
        if is_best:
            _save_checkpoint(output_dir / "best.pt", model, metadata_extra)
        if step in checkpoint_steps or (args.save_every > 0 and step % args.save_every == 0):
            _save_checkpoint(output_dir / f"step_{step:06d}.pt", model, metadata_extra)

        if (
            args.early_stop_patience > 0
            and bad_steps >= args.early_stop_patience
        ):
            print(
                f"Early stopping at step={step}; "
                f"best_step={best_step} best_metric_value={best_metric_value:.6f}"
            )
            break

    print(f"Saved Pointer GRPO checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
