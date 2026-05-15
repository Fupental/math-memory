import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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


class MemorySequenceDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        query_embeddings: Dict[int, torch.Tensor],
        memory_embeddings: torch.Tensor,
        memory_size: int,
    ) -> None:
        self.rows = rows
        self.query_embeddings = query_embeddings
        self.memory_embeddings = memory_embeddings
        self.eos_token_id = memory_size
        self.bos_token_id = memory_size + 1
        self.query_token_id = memory_size + 2

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        memory_ids = row["memory_ids"]
        input_ids = [self.bos_token_id, self.query_token_id, *memory_ids, self.eos_token_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "query_emb": self.query_embeddings[row["query_id"]].float(),
            "memory_embs": self.memory_embeddings[memory_ids].float(),
        }


class MemoryBeamGroupDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        query_embeddings: Dict[int, torch.Tensor],
        memory_embeddings: torch.Tensor,
        memory_size: int,
        reward_field: str,
    ) -> None:
        self.query_embeddings = query_embeddings
        self.memory_embeddings = memory_embeddings
        self.eos_token_id = memory_size
        self.bos_token_id = memory_size + 1
        self.query_token_id = memory_size + 2
        self.reward_field = reward_field

        grouped: Dict[int, List[Dict]] = {}
        for row in rows:
            if reward_field not in row:
                raise ValueError(
                    f"RCE training requires reward field '{reward_field}'. "
                    "Use a scored generated file or choose another field."
                )
            grouped.setdefault(row["query_id"], []).append(row)
        self.groups = [
            {"query_id": query_id, "rows": grouped[query_id]}
            for query_id in sorted(grouped)
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        input_ids = []
        memory_ids = []
        rewards = []
        for row in group["rows"]:
            row_memory_ids = row["memory_ids"]
            memory_ids.append(row_memory_ids)
            input_ids.append(
                [
                    self.bos_token_id,
                    self.query_token_id,
                    *row_memory_ids,
                    self.eos_token_id,
                ]
            )
            rewards.append(float(row[self.reward_field]))
        return {
            "query_id": group["query_id"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "query_emb": self.query_embeddings[group["query_id"]].float(),
            "memory_embs": self.memory_embeddings[torch.tensor(memory_ids, dtype=torch.long)].float(),
            "rewards": torch.tensor(rewards, dtype=torch.float),
            "beam_mask": torch.ones(len(input_ids), dtype=torch.bool),
        }


def rce_collate(batch):
    max_beams = max(item["input_ids"].shape[0] for item in batch)
    seq_len = batch[0]["input_ids"].shape[1]
    shot_num = batch[0]["memory_embs"].shape[1]
    emb_dim = batch[0]["memory_embs"].shape[-1]
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_beams, seq_len, dtype=torch.long)
    memory_embs = torch.zeros(batch_size, max_beams, shot_num, emb_dim, dtype=torch.float)
    rewards = torch.zeros(batch_size, max_beams, dtype=torch.float)
    beam_mask = torch.zeros(batch_size, max_beams, dtype=torch.bool)
    query_emb = torch.stack([item["query_emb"] for item in batch], dim=0)
    query_ids = []

    for batch_index, item in enumerate(batch):
        beam_count = item["input_ids"].shape[0]
        input_ids[batch_index, :beam_count] = item["input_ids"]
        memory_embs[batch_index, :beam_count] = item["memory_embs"]
        rewards[batch_index, :beam_count] = item["rewards"]
        beam_mask[batch_index, :beam_count] = item["beam_mask"]
        query_ids.append(item["query_id"])

    return {
        "query_ids": query_ids,
        "input_ids": input_ids,
        "query_emb": query_emb,
        "memory_embs": memory_embs,
        "rewards": rewards,
        "beam_mask": beam_mask,
    }


def save_rce_weights(dataset: MemoryBeamGroupDataset, output_dir: Path, temperature: float) -> Path:
    records = []
    for group in dataset.groups:
        rewards = torch.tensor(
            [float(row[dataset.reward_field]) for row in group["rows"]],
            dtype=torch.float,
        )
        weights = torch.softmax(rewards / temperature, dim=-1)
        for beam_index, (row, reward, weight) in enumerate(
            zip(group["rows"], rewards.tolist(), weights.tolist())
        ):
            record = {
                "query_id": group["query_id"],
                "beam_index": beam_index,
                "memory_ids": row["memory_ids"],
                "reward_field": dataset.reward_field,
                "reward": reward,
                "rce_temperature": temperature,
                "rce_weight": weight,
            }
            if "total_delta" in row:
                record["total_delta"] = row["total_delta"]
            if "score" in row:
                record["score"] = row["score"]
            records.append(record)

    json_path = output_dir / "rce_weights.json"
    csv_path = output_dir / "rce_weights.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "reward_field": dataset.reward_field,
                "temperature": temperature,
                "data": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "beam_index",
            "memory_ids",
            "reward_field",
            "reward",
            "rce_temperature",
            "rce_weight",
            "total_delta",
            "score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["memory_ids"] = json.dumps(row["memory_ids"], ensure_ascii=False)
            writer.writerow(row)
    return json_path


def _split_indices(num_items: int, train_ratio: float, seed: int):
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if num_items <= 1:
        return indices, []
    val_count = max(1, int(num_items * (1.0 - train_ratio)))
    return indices[val_count:], indices[:val_count]


def _rce_loss(model, batch, device, temperature: float):
    input_ids = batch["input_ids"].to(device)
    query_emb = batch["query_emb"].to(device)
    memory_embs = batch["memory_embs"].to(device)
    rewards = batch["rewards"].to(device)
    beam_mask = batch["beam_mask"].to(device)

    batch_size, num_beams, seq_len = input_ids.shape
    flat_input_ids = input_ids.reshape(batch_size * num_beams, seq_len)
    flat_query_emb = query_emb.repeat_interleave(num_beams, dim=0)
    flat_memory_embs = memory_embs.reshape(
        batch_size * num_beams,
        memory_embs.shape[-2],
        memory_embs.shape[-1],
    )

    output = model(
        input_ids=flat_input_ids,
        query_emb=flat_query_emb,
        memory_embs=flat_memory_embs,
        labels=None,
    )
    shift_logits = output.logits[:, :-1, :].contiguous()
    shift_labels = flat_input_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view(batch_size, num_beams, seq_len - 1)
    sequence_losses = token_losses.mean(dim=-1)

    masked_rewards = rewards.masked_fill(~beam_mask, -torch.inf)
    weights = torch.softmax(masked_rewards / temperature, dim=-1)
    weights = weights.masked_fill(~beam_mask, 0.0)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    return (weights * sequence_losses).sum(dim=-1).mean()


def _run_epoch(
    model,
    dataloader,
    optimizer,
    device,
    train: bool,
    loss_type: str,
    rce_temperature: float,
    max_batches=None,
):
    model.train(train)
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=100, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if loss_type == "sft":
                input_ids = batch["input_ids"].to(device)
                query_emb = batch["query_emb"].to(device)
                memory_embs = batch["memory_embs"].to(device)
                output = model(
                    input_ids=input_ids,
                    query_emb=query_emb,
                    memory_embs=memory_embs,
                    labels=input_ids,
                )
                loss = output.loss
                item_count = input_ids.shape[0]
            elif loss_type == "rce":
                loss = _rce_loss(model, batch, device, rce_temperature)
                item_count = batch["query_emb"].shape[0]
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * item_count
            total_items += item_count
    return total_loss / max(total_items, 1)


def _validation_queries(dataset, val_indices: List[int], query_by_id: Dict[int, Dict]) -> List[Dict]:
    query_ids = []
    seen = set()
    if isinstance(dataset, MemoryBeamGroupDataset):
        candidates = [dataset.groups[index]["query_id"] for index in val_indices]
    else:
        candidates = [dataset.rows[index]["query_id"] for index in val_indices]
    for query_id in candidates:
        if query_id not in seen:
            query_ids.append(query_id)
            seen.add(query_id)
    return [query_by_id[query_id] for query_id in query_ids]


@torch.inference_mode()
def _evaluate_validation_accuracy(
    model,
    val_queries: List[Dict],
    memories: List[Dict],
    query_embeddings: Dict[int, torch.Tensor],
    memory_embeddings: torch.Tensor,
    scorer,
    device,
    shot_num: int,
    infer_batch_size: int,
):
    if not val_queries:
        return None
    was_training = model.training
    model.eval()
    memory_embeddings = memory_embeddings.to(device)
    predictions = []
    retrievals = []
    for start in range(0, len(val_queries), infer_batch_size):
        query_batch = val_queries[start : start + infer_batch_size]
        query_embs = torch.stack(
            [query_embeddings[query["query_id"]] for query in query_batch], dim=0
        ).to(device)
        batch_memory_ids = model.generate_memory_ids(
            query_embs=query_embs,
            memory_embedding_table=memory_embeddings,
            shot_num=shot_num,
            selection_mode="raw",
        )
        retrievals.extend(batch_memory_ids.cpu().tolist())

    correct = 0
    for query, memory_ids in tqdm(
        list(zip(val_queries, retrievals)),
        desc="Online val Qwen3",
        leave=False,
        ncols=100,
    ):
        prediction, _scores = scorer.predict(query, memory_ids, memories)
        is_correct = prediction == query["answer"]
        correct += int(is_correct)
        predictions.append(
            {
                "query_id": query["query_id"],
                "question_id": query["question_id"],
                "prediction": prediction,
                "answer": query["answer"],
                "correct": is_correct,
                "memory_ids": memory_ids,
                "memory_source_ids": [
                    memories[memory_id]["source_id"] for memory_id in memory_ids
                ],
            }
        )
    if was_training:
        model.train()
    total = len(val_queries)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def _metric_improved(metric_name: str, current, best, min_delta: float) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if metric_name == "val_accuracy":
        return current > best + min_delta
    return current < best - min_delta


def main():
    parser = argparse.ArgumentParser(description="Train 2-layer Lever-LM for memories.")
    parser.add_argument("--generated-file", required=True)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--model-backend", choices=["gpt2", "simple"], default="gpt2")
    parser.add_argument("--loss-type", choices=["sft", "rce"], default="sft")
    parser.add_argument("--rce-reward-field", default="total_delta")
    parser.add_argument("--rce-temperature", type=float, default=1.0)
    parser.add_argument("--best-metric", choices=["val_loss", "val_accuracy"], default="val_loss")
    parser.add_argument("--online-eval-every", type=int, default=1)
    parser.add_argument("--online-eval-limit", type=int, default=None)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    if args.rce_temperature <= 0:
        raise ValueError("--rce-temperature must be > 0")
    if args.online_eval_every <= 0:
        raise ValueError("--online-eval-every must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    generated = json.load(open(args.generated_file, encoding="utf-8"))
    metadata = generated["metadata"]
    rows = generated["data"]
    if not rows:
        raise ValueError("Generated file contains no training rows")

    memories = load_experiences(args.experience_file)
    train_queries, _test_queries = load_mmlu_pro_math_split(
        seed=metadata["seed"],
        train_ratio=metadata["train_ratio"],
        mock_data=metadata.get("mock_data", False),
        mock_records=metadata.get("mock_records", 20),
    )
    query_by_id = {query["query_id"]: query for query in train_queries}
    used_query_ids = sorted({row["query_id"] for row in rows})
    missing = [query_id for query_id in used_query_ids if query_id not in query_by_id]
    if missing:
        raise ValueError(f"Generated data references unknown train query ids: {missing[:5]}")

    embedder = build_embedder(
        args.embedding_model,
        device=args.embedding_device,
        dtype=args.embedding_dtype,
        max_length=args.embedding_max_length,
        mock_emb_dim=args.mock_emb_dim,
    )
    safe_name = safe_model_name(args.embedding_model)
    cache_dir = Path(args.embedding_cache_dir)
    memory_texts = [memory["text"] for memory in memories]
    memory_embeddings = load_or_create_embeddings(
        str(cache_dir / f"memory_{safe_name}.pt"),
        memory_texts,
        embedder,
        batch_size=args.embedding_batch_size,
    )
    query_texts = [query_to_text(query_by_id[query_id]) for query_id in used_query_ids]
    query_embeddings_tensor = load_or_create_embeddings(
        str(cache_dir / f"train_queries_{safe_name}_seed{metadata['seed']}.pt"),
        query_texts,
        embedder,
        batch_size=args.embedding_batch_size,
    )
    query_embeddings = {
        query_id: query_embeddings_tensor[index]
        for index, query_id in enumerate(used_query_ids)
    }
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.loss_type == "sft":
        dataset = MemorySequenceDataset(
            rows=rows,
            query_embeddings=query_embeddings,
            memory_embeddings=memory_embeddings,
            memory_size=len(memories),
        )
        collate_fn = None
    else:
        dataset = MemoryBeamGroupDataset(
            rows=rows,
            query_embeddings=query_embeddings,
            memory_embeddings=memory_embeddings,
            memory_size=len(memories),
            reward_field=args.rce_reward_field,
        )
        collate_fn = rce_collate
    train_indices, val_indices = _split_indices(len(dataset), args.train_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    ) if val_indices else None
    val_queries = _validation_queries(dataset, val_indices, query_by_id)
    if args.online_eval_limit is not None:
        val_queries = val_queries[: args.online_eval_limit]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MathMemoryLeverLM(
        memory_size=len(memories),
        encoder_emb_dim=memory_embeddings.shape[-1],
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        model_backend=args.model_backend,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scorer = None
    if args.best_metric == "val_accuracy":
        scorer = build_scorer(
            model_name=args.scorer_model,
            device=args.scorer_device,
            dtype=args.scorer_dtype,
            batch_size=args.scorer_batch_size,
            max_length=args.scorer_max_length,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rce_weights_path = None
    if args.loss_type == "rce":
        rce_weights_path = save_rce_weights(dataset, output_dir, args.rce_temperature)
        print(f"Saved RCE weights to {rce_weights_path}")
    history_path = output_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_correct",
                "val_total",
                "is_best",
                "bad_epochs",
            ],
        )
        writer.writeheader()

    best_metric_value = None
    best_epoch = None
    bad_epochs = 0
    max_epochs = 1 if args.fast_dev_run else args.max_epochs
    max_batches = 2 if args.fast_dev_run else None

    for epoch in range(max_epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            loss_type=args.loss_type,
            rce_temperature=args.rce_temperature,
            max_batches=max_batches,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                loss_type=args.loss_type,
                rce_temperature=args.rce_temperature,
                max_batches=max_batches,
            )
        val_accuracy = None
        val_correct = None
        val_total = None
        val_predictions = None
        if (
            args.best_metric == "val_accuracy"
            and scorer is not None
            and epoch % args.online_eval_every == 0
        ):
            val_metrics = _evaluate_validation_accuracy(
                model=model,
                val_queries=val_queries,
                memories=memories,
                query_embeddings=query_embeddings,
                memory_embeddings=memory_embeddings,
                scorer=scorer,
                device=device,
                shot_num=metadata["shot_num"],
                infer_batch_size=args.infer_batch_size,
            )
            if val_metrics is not None:
                val_accuracy = val_metrics["accuracy"]
                val_correct = val_metrics["correct"]
                val_total = val_metrics["total"]
                val_predictions = val_metrics["predictions"]
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss if val_loss is not None else 'nan'} "
            f"val_accuracy={val_accuracy if val_accuracy is not None else 'nan'}"
        )

        is_best = False
        current_metric = val_accuracy if args.best_metric == "val_accuracy" else val_loss
        if current_metric is not None:
            if _metric_improved(
                args.best_metric,
                current_metric,
                best_metric_value,
                args.early_stop_min_delta,
            ):
                best_metric_value = current_metric
                best_epoch = epoch
                bad_epochs = 0
                is_best = True
            elif args.early_stop_patience > 0:
                bad_epochs += 1

        early_stopped = (
            current_metric is not None
            and args.early_stop_patience > 0
            and bad_epochs >= args.early_stop_patience
        )

        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_accuracy",
                    "val_correct",
                    "val_total",
                    "is_best",
                    "bad_epochs",
                ],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loss is not None else "",
                    "val_accuracy": val_accuracy if val_accuracy is not None else "",
                    "val_correct": val_correct if val_correct is not None else "",
                    "val_total": val_total if val_total is not None else "",
                    "is_best": int(is_best),
                    "bad_epochs": bad_epochs,
                }
            )

        payload = {
            "model": model.state_dict(),
            "metadata": checkpoint_metadata(
                model,
                {
                    "embedding_model": args.embedding_model,
                    "shot_num": metadata["shot_num"],
                    "generated_file": str(Path(args.generated_file).resolve()),
                    "experience_file": str(Path(args.experience_file).resolve()),
                    "loss_history_file": str(history_path.resolve()),
                    "loss_type": args.loss_type,
                    "rce_reward_field": args.rce_reward_field if args.loss_type == "rce" else None,
                    "rce_temperature": args.rce_temperature if args.loss_type == "rce" else None,
                    "rce_weights_file": str(rce_weights_path.resolve()) if rce_weights_path else None,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value,
                    "online_eval_every": args.online_eval_every,
                    "online_eval_limit": args.online_eval_limit,
                    "val_accuracy": val_accuracy,
                    "val_correct": val_correct,
                    "val_total": val_total,
                    "max_epochs": max_epochs,
                    "early_stop_patience": args.early_stop_patience,
                    "early_stop_min_delta": args.early_stop_min_delta,
                    "current_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_metric_value if args.best_metric == "val_loss" else None,
                    "best_val_accuracy": best_metric_value if args.best_metric == "val_accuracy" else None,
                    "bad_epochs": bad_epochs,
                    "early_stopped": early_stopped,
                    "stopped_epoch": epoch if early_stopped else None,
                },
            ),
        }
        torch.save(payload, output_dir / "last.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")
            if val_predictions is not None:
                with (output_dir / "best_val_predictions.json").open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(val_predictions, f, indent=2, ensure_ascii=False)

        if early_stopped:
            print(
                f"Early stopping at epoch={epoch}; "
                f"best_epoch={best_epoch} best_{args.best_metric}={best_metric_value:.6f}"
            )
            break

    print(f"Saved checkpoint to {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
