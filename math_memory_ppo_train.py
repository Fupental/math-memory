import argparse
import copy
import csv
import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

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
from math_memory_grpo_train import (
    CorrectnessCache,
    ScoreCache,
    _compute_correct_rate,
    _compute_rewards,
    _compute_sft_anchor_loss,
    _evaluate_policy,
    _exact_kl_from_probs,
    _load_anchor_rows,
    _marginal_entropy,
    _parse_checkpoint_steps,
    _split_grpo_indices,
)


def _flatten_rollouts(tensor: torch.Tensor) -> torch.Tensor:
    batch_size, group_size = tensor.shape[:2]
    return tensor.reshape(batch_size * group_size, 1, *tensor.shape[2:])


def _standardize(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std.detach().cpu()) <= eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(eps)


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    returns_flat = returns.detach().reshape(-1)
    values_flat = values.detach().reshape(-1)
    var_returns = returns_flat.var(unbiased=False)
    if float(var_returns.cpu()) <= 1e-8:
        return 0.0
    residual_var = (returns_flat - values_flat).var(unbiased=False)
    return float((1.0 - residual_var / var_returns).cpu())


def _mean_or_zero(items: List[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _save_ppo_checkpoint(
    path: Path,
    model: MathMemoryLeverLM,
    critic_model: MathMemoryLeverLM | None,
    base_metadata: Dict[str, Any],
    extra_metadata: Dict[str, Any],
) -> None:
    metadata = checkpoint_metadata(model, {**base_metadata, **extra_metadata})
    payload = {
        "model": model.state_dict(),
        "metadata": metadata,
    }
    if critic_model is not None and critic_model is not model:
        payload["critic_model"] = critic_model.state_dict()
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="PPO fine-tune SFT math-memory Lever-LM.")
    parser.add_argument("--init-mode", choices=["checkpoint", "scratch"], default="checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
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
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--shot-num", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=32)
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
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--critic-weight-decay", type=float, default=None)
    parser.add_argument("--critic-grad-clip", type=float, default=None)
    parser.add_argument("--critic-mode", choices=["shared", "separate"], default="separate")
    parser.add_argument("--critic-init", choices=["copy_actor"], default="copy_actor")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--value-clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--marginal-entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--ref-kl-coef", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--sft-anchor-file", default=None)
    parser.add_argument("--sft-anchor-coef", type=float, default=0.0)
    parser.add_argument("--sft-anchor-batch-size", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--correctness-every", type=int, default=0)
    parser.add_argument("--grpo-val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--best-metric",
        choices=["train_window_final_delta", "val_accuracy", "val_final_delta"],
        default="val_accuracy",
    )
    parser.add_argument("--eval-infer-batch-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=16)
    parser.add_argument("--model-backend", choices=["gpt2", "simple"], default="gpt2")
    args = parser.parse_args()

    if args.init_mode == "checkpoint" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --init-mode checkpoint")
    if args.init_mode == "scratch" and args.checkpoint:
        raise ValueError("--checkpoint must not be set when --init-mode scratch")
    if args.init_mode == "scratch" and not args.embedding_model:
        raise ValueError("--embedding-model is required when --init-mode scratch")
    if args.shot_num != 2:
        raise ValueError("This PPO trainer currently supports --shot-num 2 only")
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.ppo_epochs <= 0:
        raise ValueError("--ppo-epochs must be > 0")
    if args.ppo_minibatch_size <= 0:
        raise ValueError("--ppo-minibatch-size must be > 0")
    if args.clip_eps <= 0:
        raise ValueError("--clip-eps must be > 0")
    if args.value_clip_eps <= 0:
        raise ValueError("--value-clip-eps must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.debias_pool_size < 0:
        raise ValueError("--debias-pool-size must be >= 0")
    if args.credit_gamma < 0:
        raise ValueError("--credit-gamma must be >= 0")
    if args.correctness_bonus < 0:
        raise ValueError("--correctness-bonus must be >= 0")
    if args.value_coef < 0:
        raise ValueError("--value-coef must be >= 0")
    if args.entropy_coef < 0:
        raise ValueError("--entropy-coef must be >= 0")
    if args.marginal_entropy_coef < 0:
        raise ValueError("--marginal-entropy-coef must be >= 0")
    if args.ref_kl_coef < 0:
        raise ValueError("--ref-kl-coef must be >= 0")
    if args.target_kl < 0:
        raise ValueError("--target-kl must be >= 0")
    if args.sft_anchor_coef < 0:
        raise ValueError("--sft-anchor-coef must be >= 0")
    if args.sft_anchor_coef > 0 and not args.sft_anchor_file:
        raise ValueError("--sft-anchor-coef > 0 requires --sft-anchor-file")
    if not 0 <= args.grpo_val_ratio < 1:
        raise ValueError("--grpo-val-ratio must be in [0, 1)")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.best_metric != "train_window_final_delta":
        if args.grpo_val_ratio <= 0:
            raise ValueError("--best-metric val_* requires --grpo-val-ratio > 0")
        if args.eval_every <= 0:
            raise ValueError("--best-metric val_* requires --eval-every > 0")
    if args.best_window <= 0:
        raise ValueError("--best-window must be > 0")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    critic_lr = args.critic_lr if args.critic_lr is not None else args.lr
    critic_weight_decay = (
        args.critic_weight_decay
        if args.critic_weight_decay is not None
        else args.weight_decay
    )
    critic_grad_clip = (
        args.critic_grad_clip
        if args.critic_grad_clip is not None
        else args.grad_clip
    )
    if critic_lr <= 0:
        raise ValueError("--critic-lr must be > 0")
    if critic_weight_decay < 0:
        raise ValueError("--critic-weight-decay must be >= 0")
    if critic_grad_clip < 0:
        raise ValueError("--critic-grad-clip must be >= 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppo_config.json").open("w", encoding="utf-8") as f:
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
    )
    ratio_name = str(args.train_ratio).replace(".", "p")
    query_embeddings = load_or_create_embeddings(
        str(cache_dir / f"ppo_train_queries_{safe_name}_seed{args.seed}_ratio{ratio_name}.pt"),
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
        model_backend = metadata.get("model_backend", "gpt2")
    else:
        model = MathMemoryLeverLM(
            memory_size=len(memories),
            encoder_emb_dim=memory_embeddings.shape[-1],
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            max_positions=args.max_positions,
            model_backend=args.model_backend,
        )
        base_checkpoint = None
        base_training_stage = None
        model_backend = args.model_backend
    model.to(run_device)
    model.eval()
    if args.critic_mode == "separate":
        critic_model = copy.deepcopy(model)
        if payload is not None and "critic_model" in payload:
            critic_model.load_state_dict(payload["critic_model"], strict=False)
    else:
        critic_model = model
    critic_model.to(run_device)
    critic_model.eval()

    reference_checkpoint = args.reference_checkpoint
    reference_model = None
    reference_source = "none"
    if args.ref_kl_coef > 0:
        if reference_checkpoint is not None:
            ref_payload = torch.load(reference_checkpoint, map_location="cpu")
            ref_metadata = ref_payload["metadata"]
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
            reference_source = "checkpoint"
        elif args.init_mode == "checkpoint":
            ref_payload = torch.load(args.checkpoint, map_location="cpu")
            ref_metadata = ref_payload["metadata"]
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
            reference_checkpoint = args.checkpoint
            reference_source = "checkpoint"
        else:
            reference_model = copy.deepcopy(model)
            reference_source = "initial_policy"
        reference_model.to(run_device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    memory_embeddings = memory_embeddings.to(run_device)
    query_embeddings = query_embeddings.to(run_device)
    ppo_train_indices, ppo_val_indices = _split_grpo_indices(
        len(train_queries), args.grpo_val_ratio, args.seed
    )
    ppo_train_queries = [train_queries[index] for index in ppo_train_indices]
    ppo_val_queries = [train_queries[index] for index in ppo_val_indices]
    if not ppo_train_queries:
        raise ValueError("No PPO train queries available after validation split")
    query_id_to_index = {
        int(query["query_id"]): index for index, query in enumerate(train_queries)
    }

    anchor_rows = None
    if args.sft_anchor_coef > 0:
        anchor_rows = _load_anchor_rows(
            args.sft_anchor_file,
            {int(query["query_id"]) for query in ppo_train_queries},
        )

    if args.selection_mode in {"debiased_topk", "debiased_policy"}:
        if args.debias_pool_size == 0:
            debias_query_embs = query_embeddings
        else:
            debias_query_embs = query_embeddings[: min(args.debias_pool_size, len(query_embeddings))]
    else:
        debias_query_embs = None

    if args.critic_mode == "separate":
        actor_parameters = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("value_head.")
        ]
        actor_optimizer = torch.optim.AdamW(
            actor_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        critic_optimizer = torch.optim.AdamW(
            critic_model.parameters(), lr=critic_lr, weight_decay=critic_weight_decay
        )
    else:
        actor_parameters = list(model.parameters())
        actor_optimizer = torch.optim.AdamW(
            actor_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        critic_optimizer = None
    scorer = build_scorer(
        model_name=args.scorer_model,
        device=args.scorer_device,
        dtype=args.scorer_dtype,
        batch_size=args.scorer_batch_size,
        max_length=args.scorer_max_length,
        device_map=args.scorer_device_map,
        max_memory=args.scorer_max_memory,
    )

    history_path = output_dir / "ppo_history.csv"
    history_fields = [
        "step",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "mean_abs_logprob_delta",
        "reference_kl",
        "sft_anchor_loss",
        "marginal_entropy",
        "max_prob_step0",
        "max_prob_step1",
        "old_value_mean",
        "value_mean",
        "return_mean",
        "advantage_mean",
        "explained_variance",
        "reward_mean",
        "reward_std",
        "r0_mean",
        "r1_mean",
        "final_delta_mean",
        "final_correct_rate",
        "correct_rate",
        "unique_memory_count",
        "ppo_epochs_ran",
        "minibatch_updates",
        "window_final_delta_mean",
        "is_best",
        "bad_windows",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()

    eval_history_path = output_dir / "ppo_eval_history.csv"
    if args.eval_every > 0 and ppo_val_queries:
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

    base_metadata = {
        "training_stage": "ppo",
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
        "value_clip_eps": args.value_clip_eps,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "marginal_entropy_coef": args.marginal_entropy_coef,
        "kl_coef": args.kl_coef,
        "ref_kl_coef": args.ref_kl_coef,
        "reference_checkpoint": (
            str(Path(reference_checkpoint).resolve()) if reference_checkpoint else None
        ),
        "reference_source": reference_source,
        "target_kl": args.target_kl,
        "sft_anchor_file": (
            str(Path(args.sft_anchor_file).resolve()) if args.sft_anchor_file else None
        ),
        "sft_anchor_coef": args.sft_anchor_coef,
        "sft_anchor_batch_size": args.sft_anchor_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "critic_mode": args.critic_mode,
        "separate_critic": args.critic_mode == "separate",
        "critic_init": args.critic_init,
        "critic_lr": critic_lr,
        "critic_weight_decay": critic_weight_decay,
        "critic_grad_clip": critic_grad_clip,
        "batch_size": args.batch_size,
        "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch_size": args.ppo_minibatch_size,
        "n_embd": model.lm_model.config.n_embd,
        "n_head": model.lm_model.config.n_head,
        "n_layer": model.lm_model.config.n_layer,
        "max_positions": model.lm_model.config.n_positions,
        "model_backend": model_backend,
        "max_steps": args.max_steps,
        "checkpoint_steps": checkpoint_steps,
        "best_window": args.best_window,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "grpo_val_ratio": args.grpo_val_ratio,
        "eval_every": args.eval_every,
        "eval_history_file": str(eval_history_path.resolve())
        if args.eval_every > 0 and ppo_val_queries
        else None,
        "best_metric": args.best_metric,
        "ppo_train_query_count": len(ppo_train_queries),
        "ppo_val_query_count": len(ppo_val_queries),
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
        _save_ppo_checkpoint(
            output_dir / "init.pt",
            model,
            critic_model,
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

    progress = tqdm(range(args.max_steps), desc="PPO", ncols=100)
    for step in progress:
        batch_indices = [
            ppo_train_indices[rng.randrange(len(ppo_train_indices))]
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
            _critic_logprobs_check, _critic_entropies_check, old_values = (
                critic_model.compute_action_logprobs(
                    query_embs=batch_query_embs,
                    memory_embedding_table=memory_embeddings,
                    memory_ids=memory_ids.to(run_device),
                    temperature=args.temperature,
                    selection_mode=args.selection_mode,
                    debias_query_embs=debias_query_embs,
                    debias_batch_size=args.embedding_batch_size,
                    return_values=True,
                )
            )

        returns_cpu, _step_rewards_cpu, reward_stats = _compute_rewards(
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
        old_logprobs = old_logprobs.detach().to(run_device)
        old_values = old_values.detach().to(run_device)
        memory_ids = memory_ids.to(run_device)
        raw_advantages = returns - old_values
        advantages = _standardize(raw_advantages)

        flat_query_embs = batch_query_embs.repeat_interleave(args.group_size, dim=0)
        flat_memory_ids = _flatten_rollouts(memory_ids)
        flat_old_logprobs = _flatten_rollouts(old_logprobs)
        flat_old_values = _flatten_rollouts(old_values)
        flat_returns = _flatten_rollouts(returns)
        flat_advantages = _flatten_rollouts(advantages)

        rollout_count = flat_memory_ids.shape[0]
        update_order = list(range(rollout_count))
        epoch_metrics: Dict[str, List[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "mean_abs_logprob_delta": [],
            "reference_kl": [],
            "sft_anchor_loss": [],
            "marginal_entropy": [],
            "max_prob_step0": [],
            "max_prob_step1": [],
            "value_mean": [],
        }
        ppo_epochs_ran = 0
        minibatch_updates = 0
        stop_early_for_kl = False

        for _epoch in range(args.ppo_epochs):
            ppo_epochs_ran += 1
            rng.shuffle(update_order)
            epoch_kl_values = []
            for start in range(0, rollout_count, args.ppo_minibatch_size):
                mb_indices = update_order[start : start + args.ppo_minibatch_size]
                index_tensor = torch.tensor(mb_indices, dtype=torch.long, device=run_device)
                mb_query_embs = flat_query_embs[index_tensor]
                mb_memory_ids = flat_memory_ids[index_tensor]
                mb_old_logprobs = flat_old_logprobs[index_tensor]
                mb_old_values = flat_old_values[index_tensor]
                mb_returns = flat_returns[index_tensor]
                mb_advantages = flat_advantages[index_tensor]

                model.eval()
                new_logprobs, entropies, max_probs, probs = (
                    model.compute_action_logprobs(
                        query_embs=mb_query_embs,
                        memory_embedding_table=memory_embeddings,
                        memory_ids=mb_memory_ids,
                        temperature=args.temperature,
                        selection_mode=args.selection_mode,
                        debias_query_embs=debias_query_embs,
                        debias_batch_size=args.embedding_batch_size,
                        return_max_probs=True,
                        return_probs=True,
                    )
                )
                _critic_logprobs, _critic_entropies, values = (
                    critic_model.compute_action_logprobs(
                        query_embs=mb_query_embs,
                        memory_embedding_table=memory_embeddings,
                        memory_ids=mb_memory_ids,
                        temperature=args.temperature,
                        selection_mode=args.selection_mode,
                        debias_query_embs=debias_query_embs,
                        debias_batch_size=args.embedding_batch_size,
                        return_values=True,
                    )
                )
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
                ) * mb_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                value_pred_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -args.value_clip_eps,
                    args.value_clip_eps,
                )
                value_losses = (values - mb_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_losses, value_losses_clipped
                ).mean()

                entropy = entropies.mean()
                marginal_entropy, _marginal0, _marginal1 = _marginal_entropy(probs)
                if reference_model is not None:
                    with torch.no_grad():
                        _ref_logprobs, _ref_entropies, ref_probs = (
                            reference_model.compute_action_logprobs(
                                query_embs=mb_query_embs,
                                memory_embedding_table=memory_embeddings,
                                memory_ids=mb_memory_ids,
                                temperature=args.temperature,
                                selection_mode=args.selection_mode,
                                debias_query_embs=debias_query_embs,
                                debias_batch_size=args.embedding_batch_size,
                                return_probs=True,
                            )
                        )
                    reference_kl = _exact_kl_from_probs(probs, ref_probs).mean()
                else:
                    reference_kl = torch.zeros(
                        (), dtype=policy_loss.dtype, device=run_device
                    )

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
                    sft_anchor_loss = torch.zeros(
                        (), dtype=policy_loss.dtype, device=run_device
                    )

                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio < 1.0 - args.clip_eps)
                    | (ratio > 1.0 + args.clip_eps)
                ).float().mean()
                mean_abs_logprob_delta = log_ratio.abs().mean()
                actor_loss = (
                    policy_loss
                    - args.entropy_coef * entropy
                    - args.marginal_entropy_coef * marginal_entropy
                    + args.kl_coef * approx_kl
                    + args.ref_kl_coef * reference_kl
                    + args.sft_anchor_coef * sft_anchor_loss
                )
                critic_loss = args.value_coef * value_loss
                loss = actor_loss + critic_loss

                if args.critic_mode == "separate":
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(actor_parameters, args.grad_clip)
                    actor_optimizer.step()

                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    if critic_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            critic_model.parameters(), critic_grad_clip
                        )
                    critic_optimizer.step()
                else:
                    actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(actor_parameters, args.grad_clip)
                    actor_optimizer.step()
                minibatch_updates += 1

                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "mean_abs_logprob_delta": mean_abs_logprob_delta,
                    "reference_kl": reference_kl,
                    "sft_anchor_loss": sft_anchor_loss,
                    "marginal_entropy": marginal_entropy,
                    "max_prob_step0": max_probs[:, :, 0].mean(),
                    "max_prob_step1": max_probs[:, :, 1].mean(),
                    "value_mean": values.mean(),
                }
                for key, value in metrics.items():
                    epoch_metrics[key].append(float(value.detach().cpu()))
                epoch_kl_values.append(float(approx_kl.detach().cpu()))

            if args.target_kl > 0 and _mean_or_zero(epoch_kl_values) > args.target_kl:
                stop_early_for_kl = True
                break

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
            and bool(ppo_val_queries)
            and step % args.eval_every == 0
        )
        if is_eval_step:
            val_query_embs = query_embeddings[ppo_val_indices].to(run_device)
            val_stats = _evaluate_policy(
                model=model,
                reference_model=reference_model,
                scorer=scorer,
                queries=ppo_val_queries,
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
            "loss": _mean_or_zero(epoch_metrics["loss"]),
            "policy_loss": _mean_or_zero(epoch_metrics["policy_loss"]),
            "value_loss": _mean_or_zero(epoch_metrics["value_loss"]),
            "entropy": _mean_or_zero(epoch_metrics["entropy"]),
            "approx_kl": _mean_or_zero(epoch_metrics["approx_kl"]),
            "clip_fraction": _mean_or_zero(epoch_metrics["clip_fraction"]),
            "mean_abs_logprob_delta": _mean_or_zero(
                epoch_metrics["mean_abs_logprob_delta"]
            ),
            "reference_kl": _mean_or_zero(epoch_metrics["reference_kl"]),
            "sft_anchor_loss": _mean_or_zero(epoch_metrics["sft_anchor_loss"]),
            "marginal_entropy": _mean_or_zero(epoch_metrics["marginal_entropy"]),
            "max_prob_step0": _mean_or_zero(epoch_metrics["max_prob_step0"]),
            "max_prob_step1": _mean_or_zero(epoch_metrics["max_prob_step1"]),
            "old_value_mean": float(old_values.mean().detach().cpu()),
            "value_mean": _mean_or_zero(epoch_metrics["value_mean"]),
            "return_mean": float(returns.mean().detach().cpu()),
            "advantage_mean": float(advantages.mean().detach().cpu()),
            "explained_variance": _explained_variance(returns, old_values),
            "reward_mean": reward_stats["reward_mean"],
            "reward_std": reward_stats["reward_std"],
            "r0_mean": reward_stats["r0_mean"],
            "r1_mean": reward_stats["r1_mean"],
            "final_delta_mean": reward_stats["final_delta_mean"],
            "final_correct_rate": reward_stats["final_correct_rate"],
            "correct_rate": correct_rate,
            "unique_memory_count": unique_memory_count,
            "ppo_epochs_ran": ppo_epochs_ran,
            "minibatch_updates": minibatch_updates,
            "window_final_delta_mean": window_reward,
            "is_best": int(is_best),
            "bad_windows": bad_windows,
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
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

        if is_best:
            _save_ppo_checkpoint(
                output_dir / "best.pt",
                model,
                critic_model,
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
            _save_ppo_checkpoint(
                output_dir / "last.pt",
                model,
                critic_model,
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
            _save_ppo_checkpoint(
                output_dir / f"step_{update_count:06d}.pt",
                model,
                critic_model,
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
            value=f"{row['value_loss']:.3f}",
            kl=f"{row['approx_kl']:.4f}",
            reward=f"{row['reward_mean']:.3f}",
            early_kl=int(stop_early_for_kl),
        )

        if args.early_stop_patience > 0 and bad_windows >= args.early_stop_patience:
            stopped_step = step
            print(
                f"Early stopping at step={step}; best_step={best_step} "
                f"best_metric_value={best_metric_value:.6f}"
            )
            break

    final_step = stopped_step if stopped_step is not None else args.max_steps - 1
    _save_ppo_checkpoint(
        output_dir / "last.pt",
        model,
        critic_model,
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
    print(f"Saved PPO checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
