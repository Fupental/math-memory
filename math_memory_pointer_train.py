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
from torch.utils.data import DataLoader, Dataset, Subset
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


class PointerBeamGroupDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        query_embeddings: Dict[int, torch.Tensor],
        memory_embeddings: torch.Tensor,
        memory_size: int,
        candidate_num: int,
        reward_field: str,
        seed: int,
        candidate_mode: str = "random",
    ) -> None:
        self.query_embeddings = query_embeddings
        self.memory_embeddings = memory_embeddings
        self.memory_size = memory_size
        self.candidate_num = candidate_num
        self.reward_field = reward_field
        self.seed = seed
        self.candidate_mode = candidate_mode
        self.epoch = 0
        if candidate_mode not in {"random", "generated"}:
            raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

        grouped: Dict[int, List[Dict]] = {}
        for row in rows:
            if reward_field not in row:
                raise ValueError(
                    f"Pointer RCE requires reward field '{reward_field}'. "
                    "Use a scored generated file or choose another field."
                )
            if len(row["memory_ids"]) != 2:
                raise ValueError("Pointer v1 supports fixed 2-shot rows only")
            grouped.setdefault(row["query_id"], []).append(row)
        self.generated_candidates: Dict[int, List[int]] = {}
        if candidate_mode == "generated":
            for query_id, query_rows in grouped.items():
                first_candidates = query_rows[0].get("candidate_ids")
                if first_candidates is None:
                    raise ValueError(
                        "candidate_mode=generated requires each row to contain candidate_ids"
                    )
                candidate_ids = [int(memory_id) for memory_id in first_candidates]
                if len(candidate_ids) != candidate_num:
                    raise ValueError(
                        f"Query {query_id} has {len(candidate_ids)} generated candidates; "
                        f"expected candidate_num={candidate_num}"
                    )
                if len(set(candidate_ids)) != len(candidate_ids):
                    raise ValueError(f"Query {query_id} generated candidate_ids contain duplicates")
                bad_ids = [
                    memory_id
                    for memory_id in candidate_ids
                    if memory_id < 0 or memory_id >= memory_size
                ]
                if bad_ids:
                    raise ValueError(f"Query {query_id} has out-of-range candidate ids: {bad_ids[:5]}")
                for row in query_rows:
                    row_candidates = [int(memory_id) for memory_id in row.get("candidate_ids", [])]
                    if row_candidates != candidate_ids:
                        raise ValueError(
                            f"Query {query_id} has inconsistent candidate_ids across beam rows"
                        )
                    missing_oracle = [
                        memory_id
                        for memory_id in row["memory_ids"]
                        if int(memory_id) not in set(candidate_ids)
                    ]
                    if missing_oracle:
                        raise ValueError(
                            f"Query {query_id} oracle memories are missing from candidates: "
                            f"{missing_oracle}"
                        )
                self.generated_candidates[query_id] = candidate_ids
        self.groups = [
            {"query_id": query_id, "rows": grouped[query_id]}
            for query_id in sorted(grouped)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.groups)

    def _candidate_ids(self, query_id: int, rows: List[Dict]) -> List[int]:
        if self.candidate_mode == "generated":
            return list(self.generated_candidates[query_id])
        oracle_ids = []
        for row in rows:
            for memory_id in row["memory_ids"]:
                if memory_id not in oracle_ids:
                    oracle_ids.append(memory_id)
        if len(oracle_ids) > self.candidate_num:
            raise ValueError(
                f"Query {query_id} has {len(oracle_ids)} oracle memories, "
                f"more than candidate_num={self.candidate_num}"
            )
        negative_pool = [
            memory_id
            for memory_id in range(self.memory_size)
            if memory_id not in set(oracle_ids)
        ]
        rng = random.Random(self.seed + self.epoch * 1_000_000_007 + query_id * 1_000_003)
        negatives = rng.sample(negative_pool, self.candidate_num - len(oracle_ids))
        candidate_ids = oracle_ids + negatives
        rng.shuffle(candidate_ids)
        return candidate_ids

    def __getitem__(self, index):
        group = self.groups[index]
        query_id = group["query_id"]
        rows = group["rows"]
        candidate_ids = self._candidate_ids(query_id, rows)
        local_index = {memory_id: idx for idx, memory_id in enumerate(candidate_ids)}
        labels = []
        rewards = []
        for row in rows:
            labels.append([local_index[memory_id] for memory_id in row["memory_ids"]])
            rewards.append(float(row[self.reward_field]))
        return {
            "query_id": query_id,
            "query_emb": self.query_embeddings[query_id].float(),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "candidate_embs": self.memory_embeddings[
                torch.tensor(candidate_ids, dtype=torch.long)
            ].float(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float),
            "beam_mask": torch.ones(len(labels), dtype=torch.bool),
        }


def pointer_collate(batch):
    max_beams = max(item["labels"].shape[0] for item in batch)
    candidate_num = batch[0]["candidate_embs"].shape[0]
    emb_dim = batch[0]["candidate_embs"].shape[-1]
    batch_size = len(batch)
    labels = torch.zeros(batch_size, max_beams, 2, dtype=torch.long)
    rewards = torch.zeros(batch_size, max_beams, dtype=torch.float)
    beam_mask = torch.zeros(batch_size, max_beams, dtype=torch.bool)
    query_emb = torch.stack([item["query_emb"] for item in batch], dim=0)
    candidate_ids = torch.stack([item["candidate_ids"] for item in batch], dim=0)
    candidate_embs = torch.zeros(
        batch_size, candidate_num, emb_dim, dtype=torch.float
    )
    query_ids = []
    for batch_index, item in enumerate(batch):
        beam_count = item["labels"].shape[0]
        labels[batch_index, :beam_count] = item["labels"]
        rewards[batch_index, :beam_count] = item["rewards"]
        beam_mask[batch_index, :beam_count] = item["beam_mask"]
        candidate_embs[batch_index] = item["candidate_embs"]
        query_ids.append(item["query_id"])
    return {
        "query_ids": query_ids,
        "query_emb": query_emb,
        "candidate_ids": candidate_ids,
        "candidate_embs": candidate_embs,
        "labels": labels,
        "rewards": rewards,
        "beam_mask": beam_mask,
    }


def _split_indices(num_items: int, train_ratio: float, seed: int):
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if num_items <= 1:
        return indices, []
    val_count = max(1, int(num_items * (1.0 - train_ratio)))
    return indices[val_count:], indices[:val_count]


def _metric_improved(metric_name: str, current, best, min_delta: float) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if metric_name == "val_accuracy":
        return current > best + min_delta
    return current < best - min_delta


def save_rce_weights(dataset: PointerBeamGroupDataset, output_dir: Path, temperature: float) -> Path:
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
            records.append(
                {
                    "query_id": group["query_id"],
                    "beam_index": beam_index,
                    "memory_ids": row["memory_ids"],
                    "reward_field": dataset.reward_field,
                    "reward": reward,
                    "rce_temperature": temperature,
                    "rce_weight": weight,
                    "total_score": row.get("total_score"),
                    "total_delta": row.get("total_delta"),
                    "score": row.get("score"),
                }
            )

    json_path = output_dir / "pointer_rce_weights.json"
    csv_path = output_dir / "pointer_rce_weights.csv"
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
            "total_score",
            "score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["memory_ids"] = json.dumps(row["memory_ids"], ensure_ascii=False)
            writer.writerow(row)
    return json_path


def _pointer_rce_loss(model, batch, device, temperature: float):
    query_emb = batch["query_emb"].to(device)
    candidate_embs = batch["candidate_embs"].to(device)
    labels = batch["labels"].to(device)
    rewards = batch["rewards"].to(device)
    beam_mask = batch["beam_mask"].to(device)

    batch_size, num_beams, _shot = labels.shape
    flat_query = query_emb.repeat_interleave(num_beams, dim=0)
    flat_candidates = candidate_embs.repeat_interleave(num_beams, dim=0)
    flat_labels = labels.reshape(batch_size * num_beams, 2)
    output = model(
        query_embs=flat_query,
        candidate_embs=flat_candidates,
        selected_indices=flat_labels[:, 0],
    )
    loss1 = F.cross_entropy(output.logits1, flat_labels[:, 0], reduction="none")
    loss2 = F.cross_entropy(output.logits2, flat_labels[:, 1], reduction="none")
    sequence_losses = (loss1 + loss2).view(batch_size, num_beams)

    masked_rewards = rewards.masked_fill(~beam_mask, -torch.inf)
    weights = torch.softmax(masked_rewards / temperature, dim=-1)
    weights = weights.masked_fill(~beam_mask, 0.0)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    return (weights * sequence_losses).sum(dim=-1).mean()


def _run_epoch(
    model,
    dataset,
    dataloader,
    optimizer,
    device,
    train: bool,
    epoch: int,
    rce_temperature: float,
    max_batches=None,
):
    dataset.set_epoch(epoch if train else 0)
    model.train(train)
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=100, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            loss = _pointer_rce_loss(model, batch, device, rce_temperature)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            item_count = batch["query_emb"].shape[0]
            total_loss += float(loss.detach().cpu()) * item_count
            total_items += item_count
    return total_loss / max(total_items, 1)


def _validation_queries(dataset, val_indices: List[int], query_by_id: Dict[int, Dict]) -> List[Dict]:
    query_ids = []
    seen = set()
    for index in val_indices:
        query_id = dataset.groups[index]["query_id"]
        if query_id not in seen:
            query_ids.append(query_id)
            seen.add(query_id)
    return [query_by_id[query_id] for query_id in query_ids]


def _random_candidate_ids(query_id: int, memory_size: int, candidate_num: int, seed: int) -> List[int]:
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
    raise ValueError(f"Unsupported eval candidate mode: {mode}")


@torch.inference_mode()
def _evaluate_validation_accuracy(
    model,
    val_queries: List[Dict],
    memories: List[Dict],
    query_embeddings: Dict[int, torch.Tensor],
    memory_embeddings: torch.Tensor,
    scorer,
    device,
    candidate_num: int,
    candidate_seed: int,
    candidate_mode: str,
    random_candidate_num: int,
    semantic_candidate_num: int,
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
            device=device,
        )
        batch_memory_ids = model.generate_memory_ids(
            query_embs=query_embs,
            candidate_memory_ids=candidate_ids,
            memory_embedding_table=memory_embeddings,
        )
        retrievals.extend(batch_memory_ids.cpu().tolist())

    correct = 0
    for query, memory_ids in tqdm(
        list(zip(val_queries, retrievals)),
        desc="Pointer online val Qwen3",
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


def _save_command(output_dir: Path) -> None:
    with (output_dir / "train_command.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Pointer Lever-LM for math memories.")
    parser.add_argument("--generated-file", required=True)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-dtype", default="bf16")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device-map", default=None)
    parser.add_argument("--embedding-max-memory", default=None)
    parser.add_argument("--mock-emb-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-seed", type=int, default=42)
    parser.add_argument("--candidate-num", type=int, default=64)
    parser.add_argument(
        "--candidate-mode",
        choices=["random", "generated"],
        default="random",
        help=(
            "Training candidate pool. random preserves the original Random64 behavior; "
            "generated uses candidate_ids stored in the generated training file."
        ),
    )
    parser.add_argument(
        "--eval-candidate-mode",
        choices=["random", "semantic", "mixed"],
        default="random",
        help="Candidate recall mode used by online val_accuracy evaluation.",
    )
    parser.add_argument("--random-candidate-num", type=int, default=32)
    parser.add_argument("--semantic-candidate-num", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument(
        "--pointer-key-source",
        choices=["contextual", "semantic"],
        default="contextual",
        help=(
            "contextual uses Transformer MEM hidden states as pointer keys; "
            "semantic uses key_proj(adapter(z_i)) directly."
        ),
    )
    parser.add_argument("--rce-reward-field", default="total_delta")
    parser.add_argument("--rce-temperature", type=float, default=0.1)
    parser.add_argument("--best-metric", choices=["val_loss", "val_accuracy"], default="val_loss")
    parser.add_argument("--online-eval-every", type=int, default=1)
    parser.add_argument("--online-eval-limit", type=int, default=None)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument("--scorer-device-map", default=None)
    parser.add_argument("--scorer-max-memory", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    if args.candidate_num <= 1:
        raise ValueError("--candidate-num must be > 1")
    if args.rce_temperature <= 0:
        raise ValueError("--rce-temperature must be > 0")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.online_eval_every <= 0:
        raise ValueError("--online-eval-every must be > 0")
    if args.eval_candidate_mode == "mixed":
        if args.random_candidate_num + args.semantic_candidate_num != args.candidate_num:
            raise ValueError(
                "--random-candidate-num + --semantic-candidate-num must equal "
                "--candidate-num for mixed eval"
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_command(output_dir)

    generated = json.load(open(args.generated_file, encoding="utf-8"))
    metadata = generated["metadata"]
    rows = generated["data"]
    if metadata.get("shot_num") != 2:
        raise ValueError("Pointer v1 supports generated data with shot_num=2")
    if not rows:
        raise ValueError("Generated file contains no training rows")

    memories = load_experiences(args.experience_file)
    if args.candidate_num > len(memories):
        raise ValueError("--candidate-num cannot exceed memory bank size")
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
        device_map=args.embedding_device_map,
        max_memory=args.embedding_max_memory,
    )
    safe_name = safe_model_name(args.embedding_model)
    cache_dir = Path(args.embedding_cache_dir)
    memory_embeddings = load_or_create_embeddings(
        str(cache_dir / f"memory_{safe_name}.pt"),
        [memory["text"] for memory in memories],
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

    dataset = PointerBeamGroupDataset(
        rows=rows,
        query_embeddings=query_embeddings,
        memory_embeddings=memory_embeddings,
        memory_size=len(memories),
        candidate_num=args.candidate_num,
        reward_field=args.rce_reward_field,
        seed=args.seed,
        candidate_mode=args.candidate_mode,
    )
    train_indices, val_indices = _split_indices(len(dataset), args.train_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pointer_collate,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pointer_collate,
        )
        if val_indices
        else None
    )
    val_queries = _validation_queries(dataset, val_indices, query_by_id)
    if args.online_eval_limit is not None:
        val_queries = val_queries[: args.online_eval_limit]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = PointerMemoryLeverLM(
        encoder_emb_dim=memory_embeddings.shape[-1],
        candidate_num=args.candidate_num,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        max_positions=args.candidate_num + 8,
        pointer_key_source=args.pointer_key_source,
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
            device_map=args.scorer_device_map,
            max_memory=args.scorer_max_memory,
        )

    rce_weights_path = save_rce_weights(dataset, output_dir, args.rce_temperature)
    print(f"Saved Pointer RCE weights to {rce_weights_path}")
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
            model=model,
            dataset=dataset,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            epoch=epoch,
            rce_temperature=args.rce_temperature,
            max_batches=max_batches,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model=model,
                dataset=dataset,
                dataloader=val_loader,
                optimizer=optimizer,
                device=device,
                train=False,
                epoch=epoch,
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
                candidate_num=args.candidate_num,
                candidate_seed=args.candidate_seed,
                candidate_mode=args.eval_candidate_mode,
                random_candidate_num=args.random_candidate_num,
                semantic_candidate_num=args.semantic_candidate_num,
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
            "metadata": pointer_checkpoint_metadata(
                model,
                {
                    "embedding_model": args.embedding_model,
                    "memory_size": len(memories),
                    "shot_num": metadata["shot_num"],
                    "generated_file": str(Path(args.generated_file).resolve()),
                    "experience_file": str(Path(args.experience_file).resolve()),
                    "loss_history_file": str(history_path.resolve()),
                    "loss_type": "pointer_rce",
                    "rce_reward_field": args.rce_reward_field,
                    "rce_temperature": args.rce_temperature,
                    "rce_weights_file": str(rce_weights_path.resolve()),
                    "best_metric": args.best_metric,
                    "best_metric_value": best_metric_value,
                    "online_eval_every": args.online_eval_every,
                    "online_eval_limit": args.online_eval_limit,
                    "candidate_seed": args.candidate_seed,
                    "candidate_mode": args.candidate_mode,
                    "eval_candidate_mode": args.eval_candidate_mode,
                    "random_candidate_num": args.random_candidate_num,
                    "semantic_candidate_num": args.semantic_candidate_num,
                    "generated_candidate_mode": metadata.get("candidate_mode"),
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

    print(f"Saved Pointer checkpoint to {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
