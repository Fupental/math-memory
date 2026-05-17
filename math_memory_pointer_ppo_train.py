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
import torch.nn as nn
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
from math_memory_grpo_train import _compute_rewards, _normalize_group, _parse_checkpoint_steps
from math_memory_pointer_eval import _candidate_ids_for_batch
from math_memory_pointer_grpo_train import _load_pointer_model
from math_memory_ppo_train import _explained_variance, _standardize


class PointerValueHead(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.value = nn.Linear(n_embd, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value(hidden).squeeze(-1)


def _save_command(output_dir: Path) -> None:
    with (output_dir / "train_command.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n")


def _save_checkpoint(
    path: Path,
    model: PointerMemoryLeverLM,
    value_head: PointerValueHead,
    metadata_extra: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "value_head": value_head.state_dict(),
            "metadata": pointer_checkpoint_metadata(model, metadata_extra),
        },
        path,
    )


def _load_value_head(
    checkpoint_path: str,
    n_embd: int,
    device: torch.device,
) -> PointerValueHead:
    value_head = PointerValueHead(n_embd=n_embd).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "value_head" in payload:
        value_head.load_state_dict(payload["value_head"], strict=True)
    return value_head


def _transform_returns(
    returns: torch.Tensor,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    if mode == "none":
        return returns
    if mode == "group_zscore":
        by_step = [_normalize_group(returns[:, :, step]) for step in range(returns.shape[-1])]
        return torch.stack(by_step, dim=-1)
    if temperature <= 0:
        raise ValueError("--reward-transform-temperature must be > 0")
    if mode == "sigmoid":
        return torch.sigmoid(returns / temperature)
    if mode == "tanh":
        return torch.tanh(returns / temperature)
    raise ValueError(f"Unsupported reward_transform: {mode}")


def _pointer_logprobs_values_and_stats(
    model: PointerMemoryLeverLM,
    value_head: PointerValueHead,
    query_embs: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_embeddings: torch.Tensor,
    local_actions: torch.Tensor,
    temperature: float,
    ref_model: PointerMemoryLeverLM | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    candidate_embs = memory_embeddings[candidate_ids]
    out1 = model(query_embs, candidate_embs)
    dist1 = Categorical(logits=out1.logits1 / temperature)
    action_logp1 = dist1.log_prob(local_actions[:, 0])
    entropy1 = dist1.entropy()
    probs1 = dist1.probs
    logp1 = dist1.logits

    out2 = model(query_embs, candidate_embs, selected_indices=local_actions[:, 0])
    dist2 = Categorical(logits=out2.logits2 / temperature)
    action_logp2 = dist2.log_prob(local_actions[:, 1])
    entropy2 = dist2.entropy()
    probs2 = dist2.probs
    logp2 = dist2.logits

    sel1_pos = 2 + model.candidate_num
    sel2_pos = sel1_pos + 2
    value1 = value_head(out1.hidden_states[:, sel1_pos])
    value2 = value_head(out2.hidden_states[:, sel2_pos])

    ref_kl = torch.zeros_like(action_logp1)
    if ref_model is not None:
        with torch.no_grad():
            ref_out1 = ref_model(query_embs, candidate_embs)
            ref_logp1 = Categorical(logits=ref_out1.logits1 / temperature).logits
            ref_out2 = ref_model(
                query_embs,
                candidate_embs,
                selected_indices=local_actions[:, 0],
            )
            ref_logp2 = Categorical(logits=ref_out2.logits2 / temperature).logits
        diff1 = torch.where(probs1 > 0, logp1 - ref_logp1, torch.zeros_like(logp1))
        diff2 = torch.where(probs2 > 0, logp2 - ref_logp2, torch.zeros_like(logp2))
        ref_kl = 0.5 * ((probs1 * diff1).sum(dim=-1) + (probs2 * diff2).sum(dim=-1))

    logprobs = torch.stack([action_logp1, action_logp2], dim=-1)
    entropies = torch.stack([entropy1, entropy2], dim=-1)
    max_probs = torch.stack(
        [probs1.max(dim=-1).values, probs2.max(dim=-1).values], dim=-1
    )
    values = torch.stack([value1, value2], dim=-1)
    return logprobs, entropies, max_probs, ref_kl, values


@torch.no_grad()
def _sample_pointer_rollouts(
    model: PointerMemoryLeverLM,
    value_head: PointerValueHead,
    query_embs: torch.Tensor,
    candidate_ids: torch.Tensor,
    memory_embeddings: torch.Tensor,
    group_size: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    batch_size, candidate_num = candidate_ids.shape
    if candidate_num != model.candidate_num:
        raise ValueError(f"Expected candidate_num={model.candidate_num}, got {candidate_num}")
    flat_query = query_embs.repeat_interleave(group_size, dim=0)
    flat_candidate_ids = candidate_ids.repeat_interleave(group_size, dim=0)
    flat_candidate_embs = memory_embeddings[flat_candidate_ids]

    out1 = model(flat_query, flat_candidate_embs)
    dist1 = Categorical(logits=out1.logits1 / temperature)
    first_local = dist1.sample()

    out2 = model(flat_query, flat_candidate_embs, selected_indices=first_local)
    dist2 = Categorical(logits=out2.logits2 / temperature)
    second_local = dist2.sample()

    sel1_pos = 2 + model.candidate_num
    sel2_pos = sel1_pos + 2
    value1 = value_head(out1.hidden_states[:, sel1_pos])
    value2 = value_head(out2.hidden_states[:, sel2_pos])

    flat_actions = torch.stack([first_local, second_local], dim=-1)
    memory_ids = flat_candidate_ids.gather(1, flat_actions).reshape(batch_size, group_size, 2)
    local_actions = flat_actions.reshape(batch_size, group_size, 2)
    old_logprobs = torch.stack(
        [dist1.log_prob(first_local), dist2.log_prob(second_local)], dim=-1
    ).reshape(batch_size, group_size, 2)
    old_values = torch.stack([value1, value2], dim=-1).reshape(batch_size, group_size, 2)
    old_entropies = torch.stack([dist1.entropy(), dist2.entropy()], dim=-1).reshape(
        batch_size, group_size, 2
    )
    return memory_ids, local_actions, old_logprobs, old_values, old_entropies


def _flatten_rollout_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO fine-tuning for Pointer Lever-LM math-memory retrieval."
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
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--scorer-device-map", default=None)
    parser.add_argument("--scorer-max-memory", default=None)
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
        default="discounted",
    )
    parser.add_argument("--credit-gamma", type=float, default=0.3)
    parser.add_argument("--correctness-bonus", type=float, default=1.0)
    parser.add_argument(
        "--reward-transform",
        choices=["none", "group_zscore", "sigmoid", "tanh"],
        default="none",
    )
    parser.add_argument("--reward-transform-temperature", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--value-lr", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--value-clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ref-kl-coef", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument(
        "--best-metric",
        choices=["train_window_final_delta"],
        default="train_window_final_delta",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.batch_size <= 0 or args.group_size <= 0:
        raise ValueError("--batch-size and --group-size must be > 0")
    if args.ppo_epochs <= 0 or args.ppo_minibatch_size <= 0:
        raise ValueError("--ppo-epochs and --ppo-minibatch-size must be > 0")
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
    history_path = output_dir / "pointer_ppo_history.csv"
    checkpoint_steps = set(_parse_checkpoint_steps(args.checkpoint_steps))

    run_device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, base_metadata = _load_pointer_model(args.checkpoint, run_device)
    value_head = _load_value_head(args.checkpoint, model.n_embd, run_device)
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
    if not train_queries:
        raise ValueError("No training queries available")

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

    value_lr = args.value_lr if args.value_lr is not None else args.lr
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": value_head.parameters(), "lr": value_lr, "weight_decay": args.weight_decay},
        ]
    )

    metadata_extra_base = {
        "training_stage": "pointer_ppo",
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
        "reward_transform": args.reward_transform,
        "reward_transform_temperature": args.reward_transform_temperature,
        "correctness_bonus": args.correctness_bonus,
        "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch_size": args.ppo_minibatch_size,
        "clip_eps": args.clip_eps,
        "value_clip_eps": args.value_clip_eps,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "ref_kl_coef": args.ref_kl_coef,
        "target_kl": args.target_kl,
        "best_metric": args.best_metric,
        "loss_history_file": str(history_path.resolve()),
    }
    _save_checkpoint(output_dir / "init.pt", model, value_head, {**metadata_extra_base, "current_step": 0})

    fieldnames = [
        "step",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "ref_kl",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "reward_mean",
        "reward_std",
        "transformed_return_mean",
        "transformed_return_std",
        "advantage_mean",
        "advantage_std",
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
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    score_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    correctness_cache: Dict[Tuple[int, Tuple[int, ...]], bool] = {}
    metric_window = deque(maxlen=max(1, args.best_window))
    best_metric_value = None
    best_step = None
    bad_steps = 0
    rng = random.Random(args.seed)
    max_steps = 2 if args.fast_dev_run else args.max_steps

    for step in tqdm(range(1, max_steps + 1), desc="Pointer PPO", ncols=120):
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
        value_head.eval()
        with torch.no_grad():
            (
                memory_ids,
                local_actions,
                old_logprobs,
                old_values,
                _old_entropies,
            ) = _sample_pointer_rollouts(
                model=model,
                value_head=value_head,
                query_embs=query_embs,
                candidate_ids=candidate_ids,
                memory_embeddings=memory_embeddings,
                group_size=args.group_size,
                temperature=args.temperature,
            )
        returns_cpu, _step_rewards, reward_stats = _compute_rewards(
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
        returns = returns_cpu.to(run_device)
        transformed_returns = _transform_returns(
            returns,
            mode=args.reward_transform,
            temperature=args.reward_transform_temperature,
        )
        old_values = old_values.detach().to(run_device)
        old_logprobs = old_logprobs.detach().to(run_device)
        local_actions = local_actions.to(run_device)
        raw_advantages = transformed_returns - old_values
        advantages = _standardize(raw_advantages)

        flat_query_embs = query_embs.repeat_interleave(args.group_size, dim=0)
        flat_candidate_ids = candidate_ids.repeat_interleave(args.group_size, dim=0)
        flat_actions = _flatten_rollout_rows(local_actions)
        flat_old_logprobs = _flatten_rollout_rows(old_logprobs)
        flat_old_values = _flatten_rollout_rows(old_values)
        flat_returns = _flatten_rollout_rows(transformed_returns)
        flat_advantages = _flatten_rollout_rows(advantages)
        rollout_count = flat_actions.shape[0]
        epoch_metrics: Dict[str, List[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "ref_kl": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
            "max_prob_step0": [],
            "max_prob_step1": [],
        }

        model.train()
        value_head.train()
        early_kl = False
        for _epoch in range(args.ppo_epochs):
            permutation = torch.randperm(rollout_count, device=run_device)
            for start in range(0, rollout_count, args.ppo_minibatch_size):
                index = permutation[start : start + args.ppo_minibatch_size]
                mb_query = flat_query_embs[index]
                mb_candidate_ids = flat_candidate_ids[index]
                mb_actions = flat_actions[index]
                mb_old_logprobs = flat_old_logprobs[index]
                mb_old_values = flat_old_values[index]
                mb_returns = flat_returns[index]
                mb_advantages = flat_advantages[index]

                new_logprobs, entropies, max_probs, ref_kl, values = (
                    _pointer_logprobs_values_and_stats(
                        model=model,
                        value_head=value_head,
                        query_embs=mb_query,
                        candidate_ids=mb_candidate_ids,
                        memory_embeddings=memory_embeddings,
                        local_actions=mb_actions,
                        temperature=args.temperature,
                        ref_model=ref_model,
                    )
                )
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - args.clip_eps,
                    1.0 + args.clip_eps,
                ) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_unclipped = (values - mb_returns).pow(2)
                clipped_values = mb_old_values + (values - mb_old_values).clamp(
                    -args.value_clip_eps,
                    args.value_clip_eps,
                )
                value_clipped = (clipped_values - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_unclipped, value_clipped).mean()

                entropy = entropies.mean()
                ref_kl_mean = ref_kl.mean()
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                    + args.ref_kl_coef * ref_kl_mean
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(value_head.parameters()),
                        args.grad_clip,
                    )
                optimizer.step()

                approx_kl = float((-log_ratio).mean().detach().cpu())
                epoch_metrics["loss"].append(float(loss.detach().cpu()))
                epoch_metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                epoch_metrics["value_loss"].append(float(value_loss.detach().cpu()))
                epoch_metrics["entropy"].append(float(entropy.detach().cpu()))
                epoch_metrics["ref_kl"].append(float(ref_kl_mean.detach().cpu()))
                epoch_metrics["approx_kl"].append(approx_kl)
                epoch_metrics["clip_fraction"].append(
                    float((torch.abs(ratio - 1.0) > args.clip_eps).float().mean().detach().cpu())
                )
                epoch_metrics["explained_variance"].append(
                    _explained_variance(mb_returns, values)
                )
                epoch_metrics["max_prob_step0"].append(
                    float(max_probs[:, 0].mean().detach().cpu())
                )
                epoch_metrics["max_prob_step1"].append(
                    float(max_probs[:, 1].mean().detach().cpu())
                )

                if args.target_kl > 0 and approx_kl > args.target_kl:
                    early_kl = True
                    break
            if early_kl:
                break

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
        metadata_extra = {
            **metadata_extra_base,
            "current_step": step,
            "best_step": best_step,
            "best_metric_value": best_metric_value,
            "bad_steps": bad_steps,
            "last_final_delta_mean": reward_stats["final_delta_mean"],
            "last_window_final_delta": window_final_delta,
        }
        _save_checkpoint(output_dir / "last.pt", model, value_head, metadata_extra)
        if is_best:
            _save_checkpoint(output_dir / "best.pt", model, value_head, metadata_extra)
        if step in checkpoint_steps or (args.save_every > 0 and step % args.save_every == 0):
            _save_checkpoint(output_dir / f"step_{step:06d}.pt", model, value_head, metadata_extra)

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(
                {
                    "step": step,
                    "loss": _mean_or_zero(epoch_metrics["loss"]),
                    "policy_loss": _mean_or_zero(epoch_metrics["policy_loss"]),
                    "value_loss": _mean_or_zero(epoch_metrics["value_loss"]),
                    "entropy": _mean_or_zero(epoch_metrics["entropy"]),
                    "ref_kl": _mean_or_zero(epoch_metrics["ref_kl"]),
                    "approx_kl": _mean_or_zero(epoch_metrics["approx_kl"]),
                    "clip_fraction": _mean_or_zero(epoch_metrics["clip_fraction"]),
                    "explained_variance": _mean_or_zero(epoch_metrics["explained_variance"]),
                    "reward_mean": reward_stats["reward_mean"],
                    "reward_std": reward_stats["reward_std"],
                    "transformed_return_mean": float(transformed_returns.mean().detach().cpu()),
                    "transformed_return_std": float(
                        transformed_returns.std(unbiased=False).detach().cpu()
                    ),
                    "advantage_mean": float(raw_advantages.mean().detach().cpu()),
                    "advantage_std": float(raw_advantages.std(unbiased=False).detach().cpu()),
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
                    "max_prob_step0": _mean_or_zero(epoch_metrics["max_prob_step0"]),
                    "max_prob_step1": _mean_or_zero(epoch_metrics["max_prob_step1"]),
                }
            )

        if args.early_stop_patience > 0 and bad_steps >= args.early_stop_patience:
            print(
                f"Early stopping at step={step}; "
                f"best_step={best_step} best_metric_value={best_metric_value:.6f}"
            )
            break

    print(f"Saved Pointer PPO checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
