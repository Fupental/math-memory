import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(
        description="Summarize repeated random-sampling math-memory evaluations."
    )
    parser.add_argument("--metrics-dir", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    metric_files = sorted(metrics_dir.glob("rs_seed*_metrics.json"))
    if not metric_files:
        raise ValueError(f"No rs_seed*_metrics.json files found in {metrics_dir}")

    rows = []
    for path in metric_files:
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        if metrics.get("base_method") != "rs":
            continue
        rows.append(
            {
                "method": metrics["method"],
                "split_seed": metrics.get("split_seed"),
                "rs_seed": metrics.get("rs_seed"),
                "shot_num": metrics["shot_num"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "metrics_file": str(path.resolve()),
            }
        )

    if not rows:
        raise ValueError(f"No RS metric rows found in {metrics_dir}")

    accuracies = [row["accuracy"] for row in rows]
    summary = {
        "method": "rs_repeated",
        "metrics_dir": str(metrics_dir.resolve()),
        "num_runs": len(rows),
        "split_seed": rows[0]["split_seed"],
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

    summary_json = metrics_dir / "rs_repeated_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_csv = metrics_dir / "rs_repeated_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "split_seed",
                "rs_seed",
                "shot_num",
                "accuracy",
                "correct",
                "total",
                "metrics_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
