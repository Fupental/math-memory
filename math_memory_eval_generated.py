import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from lever_lm.math_memory.data import load_experiences, load_mmlu_pro_math_split
from lever_lm.math_memory.scoring import build_scorer


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
                "method": method,
                "shot_num": metrics["shot_num"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            }
        )


def _load_train_query_map(metadata: Dict) -> Dict[int, Dict]:
    train_queries, _test_queries = load_mmlu_pro_math_split(
        seed=metadata["seed"],
        train_ratio=metadata["train_ratio"],
        mock_data=metadata.get("mock_data", False),
        mock_records=metadata.get("mock_records", 20),
    )
    return {query["query_id"]: query for query in train_queries}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated memory sequences as fixed retrievals."
    )
    parser.add_argument("--generated-file", required=True)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="generated_sequences")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=4)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    args = parser.parse_args()

    generated = json.load(open(args.generated_file, encoding="utf-8"))
    metadata = generated["metadata"]
    rows = generated["data"]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Generated file contains no rows to evaluate")

    memories = load_experiences(args.experience_file)
    query_by_id = _load_train_query_map(metadata)
    missing_query_ids = sorted(
        {row["query_id"] for row in rows if row["query_id"] not in query_by_id}
    )
    if missing_query_ids:
        raise ValueError(
            f"Generated rows reference unknown train query ids: {missing_query_ids[:5]}"
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
    for row in tqdm(rows, desc="Evaluating generated sequences", ncols=100):
        query = query_by_id[row["query_id"]]
        memory_ids = row["memory_ids"]
        prediction, scores = scorer.predict(query, memory_ids, memories)
        is_correct = prediction == query["answer"]
        correct += int(is_correct)
        predictions.append(
            {
                "query_id": query["query_id"],
                "question_id": query["question_id"],
                "repeat": row.get("repeat"),
                "rank": row.get("rank"),
                "sequence_score": row.get("score"),
                "prediction": prediction,
                "answer": query["answer"],
                "correct": is_correct,
                "memory_ids": memory_ids,
                "memory_source_ids": [
                    memories[memory_id]["source_id"] for memory_id in memory_ids
                ],
                "scores": scores,
            }
        )

    total = len(rows)
    metrics = {
        "method": args.method_name,
        "shot_num": metadata["shot_num"],
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "generated_file": str(Path(args.generated_file).resolve()),
        "experience_file": str(Path(args.experience_file).resolve()),
        "score_mode": metadata.get("score_mode"),
        "unique_query_count": len({row["query_id"] for row in rows}),
    }
    _write_metrics(Path(args.output_dir), metrics, predictions)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
