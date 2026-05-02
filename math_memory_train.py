import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
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


def _split_indices(num_items: int, train_ratio: float, seed: int):
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if num_items <= 1:
        return indices, []
    val_count = max(1, int(num_items * (1.0 - train_ratio)))
    return indices[val_count:], indices[:val_count]


def _run_epoch(model, dataloader, optimizer, device, train: bool, max_batches=None):
    model.train(train)
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=100, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
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
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = input_ids.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
    return total_loss / max(total_items, 1)


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
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")

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

    dataset = MemorySequenceDataset(
        rows=rows,
        query_embeddings=query_embeddings,
        memory_embeddings=memory_embeddings,
        memory_size=len(memories),
    )
    train_indices, val_indices = _split_indices(len(dataset), args.train_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
    ) if val_indices else None

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MathMemoryLeverLM(
        memory_size=len(memories),
        encoder_emb_dim=memory_embeddings.shape[-1],
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "is_best",
                "bad_epochs",
            ],
        )
        writer.writeheader()

    best_val = float("inf")
    best_epoch = None
    bad_epochs = 0
    max_epochs = 1 if args.fast_dev_run else args.max_epochs
    max_batches = 2 if args.fast_dev_run else None

    for epoch in range(max_epochs):
        train_loss = _run_epoch(
            model, train_loader, optimizer, device, train=True, max_batches=max_batches
        )
        val_loss = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, optimizer, device, train=False, max_batches=max_batches
            )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss if val_loss is not None else 'nan'}"
        )

        is_best = False
        if val_loss is not None:
            if val_loss < best_val - args.early_stop_min_delta:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                is_best = True
            elif args.early_stop_patience > 0:
                bad_epochs += 1

        early_stopped = (
            val_loss is not None
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
                    "is_best",
                    "bad_epochs",
                ],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loss is not None else "",
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
                    "max_epochs": max_epochs,
                    "early_stop_patience": args.early_stop_patience,
                    "early_stop_min_delta": args.early_stop_min_delta,
                    "current_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val if best_epoch is not None else None,
                    "bad_epochs": bad_epochs,
                    "early_stopped": early_stopped,
                    "stopped_epoch": epoch if early_stopped else None,
                },
            ),
        }
        torch.save(payload, output_dir / "last.pt")
        if is_best:
            best_val = val_loss
            torch.save(payload, output_dir / "best.pt")

        if early_stopped:
            print(
                f"Early stopping at epoch={epoch}; "
                f"best_epoch={best_epoch} best_val_loss={best_val:.6f}"
            )
            break

    print(f"Saved checkpoint to {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
