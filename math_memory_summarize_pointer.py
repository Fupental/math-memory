import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(
        description="Summarize repeated Pointer Lever-LM evaluations."
    )
    parser.add_argument("--metrics-dir", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    metric_files = sorted(metrics_dir.glob("pointer_lever_lm*_metrics.json"))
    if not metric_files:
        raise ValueError(
            f"No pointer_lever_lm*_metrics.json files found in {metrics_dir}"
        )

    rows = []
    for path in metric_files:
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        if metrics.get("base_method") != "pointer_lever_lm":
            continue
        rows.append(
            {
                "method": metrics["method"],
                "split_seed": metrics.get("split_seed"),
                "candidate_seed": metrics.get("candidate_seed"),
                "candidate_mode": metrics.get("candidate_mode", "random"),
                "selection_mode": metrics.get("selection_mode", ""),
                "candidate_num": metrics.get("candidate_num"),
                "random_candidate_num": metrics.get("random_candidate_num", ""),
                "semantic_candidate_num": metrics.get("semantic_candidate_num", ""),
                "shot_num": metrics["shot_num"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "mean_final_delta": metrics.get("mean_final_delta", ""),
                "unique_pair_count": metrics.get("unique_pair_count", ""),
                "metrics_file": str(path.resolve()),
            }
        )

    if not rows:
        raise ValueError(f"No Pointer metric rows found in {metrics_dir}")

    accuracies = [row["accuracy"] for row in rows]
    summary = {
        "method": "pointer_lever_lm_repeated",
        "metrics_dir": str(metrics_dir.resolve()),
        "num_runs": len(rows),
        "split_seed": rows[0]["split_seed"],
        "candidate_num": rows[0]["candidate_num"],
        "candidate_mode": rows[0]["candidate_mode"],
        "selection_mode": rows[0]["selection_mode"],
        "shot_num": rows[0]["shot_num"],
        "total_per_run": rows[0]["total"],
        "mean_accuracy": mean(accuracies),
        "std_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "standard_error": (
            stdev(accuracies) / math.sqrt(len(accuracies))
            if len(accuracies) > 1
            else 0.0
        ),
        "runs": rows,
    }

    summary_json = metrics_dir / "pointer_repeated_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_csv = metrics_dir / "pointer_repeated_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "split_seed",
                "candidate_seed",
                "candidate_mode",
                "selection_mode",
                "candidate_num",
                "random_candidate_num",
                "semantic_candidate_num",
                "shot_num",
                "accuracy",
                "correct",
                "total",
                "mean_final_delta",
                "unique_pair_count",
                "metrics_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
