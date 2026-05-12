import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_predictions(path: str) -> Dict[int, Dict[str, Any]]:
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError(f"Prediction file must contain a list: {path}")

    rows: Dict[int, Dict[str, Any]] = {}
    for row in data:
        query_id = int(row["query_id"])
        if query_id in rows:
            raise ValueError(f"Duplicate query_id={query_id} in {path}")
        rows[query_id] = row
    return rows


def _pair(row: Dict[str, Any]) -> Tuple[int, ...]:
    return tuple(int(item) for item in row.get("memory_ids", []))


def _source_pair(row: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str(item) for item in row.get("memory_source_ids", []))


def _score(row: Dict[str, Any], label: str) -> Optional[float]:
    scores = row.get("scores") or {}
    if label not in scores:
        return None
    return float(scores[label])


def _answer_margin(row: Dict[str, Any]) -> Optional[float]:
    scores = row.get("scores") or {}
    answer = row.get("answer")
    if answer not in scores:
        return None
    wrong_scores = [float(value) for key, value in scores.items() if key != answer]
    if not wrong_scores:
        return None
    return float(scores[answer]) - max(wrong_scores)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired math-memory prediction files."
    )
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    baseline = _load_predictions(args.baseline_predictions)
    candidate = _load_predictions(args.candidate_predictions)
    common_ids = sorted(set(baseline) & set(candidate))
    if not common_ids:
        raise ValueError("No overlapping query_id values between prediction files")

    missing_baseline = sorted(set(candidate) - set(baseline))
    missing_candidate = sorted(set(baseline) - set(candidate))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {
        "both_correct": 0,
        "baseline_only_correct": 0,
        "candidate_only_correct": 0,
        "both_wrong": 0,
    }
    pair_counts = {
        "pair_unchanged": 0,
        "pair_changed": 0,
        "pair_unchanged_baseline_correct": 0,
        "pair_unchanged_candidate_correct": 0,
        "pair_changed_baseline_correct": 0,
        "pair_changed_candidate_correct": 0,
        "pair_changed_correct_to_wrong": 0,
        "pair_changed_wrong_to_correct": 0,
    }
    score_stats: Dict[str, List[float]] = {
        "answer_score_delta": [],
        "answer_margin_delta": [],
        "final_delta_delta": [],
    }
    alignment = {
        "final_delta_up_correct_to_wrong": 0,
        "final_delta_down_correct_to_wrong": 0,
        "final_delta_up_wrong_to_correct": 0,
        "final_delta_down_wrong_to_correct": 0,
        "answer_score_up_correct_to_wrong": 0,
        "answer_margin_up_correct_to_wrong": 0,
        "answer_margin_down_correct_to_wrong": 0,
    }

    for query_id in common_ids:
        base = baseline[query_id]
        cand = candidate[query_id]
        answer = base.get("answer")
        if answer != cand.get("answer"):
            raise ValueError(f"Answer mismatch for query_id={query_id}")

        base_correct = bool(base.get("correct"))
        cand_correct = bool(cand.get("correct"))
        if base_correct and cand_correct:
            counts["both_correct"] += 1
        elif base_correct and not cand_correct:
            counts["baseline_only_correct"] += 1
        elif not base_correct and cand_correct:
            counts["candidate_only_correct"] += 1
        else:
            counts["both_wrong"] += 1

        base_pair = _pair(base)
        cand_pair = _pair(cand)
        pair_changed = base_pair != cand_pair
        pair_key = "pair_changed" if pair_changed else "pair_unchanged"
        pair_counts[pair_key] += 1
        pair_counts[f"{pair_key}_baseline_correct"] += int(base_correct)
        pair_counts[f"{pair_key}_candidate_correct"] += int(cand_correct)
        if pair_changed and base_correct and not cand_correct:
            pair_counts["pair_changed_correct_to_wrong"] += 1
        if pair_changed and not base_correct and cand_correct:
            pair_counts["pair_changed_wrong_to_correct"] += 1

        base_answer_score = _score(base, answer)
        cand_answer_score = _score(cand, answer)
        base_margin = _answer_margin(base)
        cand_margin = _answer_margin(cand)
        base_final_delta = base.get("final_delta")
        cand_final_delta = cand.get("final_delta")
        if base_final_delta is not None:
            base_final_delta = float(base_final_delta)
        if cand_final_delta is not None:
            cand_final_delta = float(cand_final_delta)

        answer_score_delta = _delta(cand_answer_score, base_answer_score)
        margin_delta = _delta(cand_margin, base_margin)
        final_delta_delta = _delta(cand_final_delta, base_final_delta)
        for key, value in [
            ("answer_score_delta", answer_score_delta),
            ("answer_margin_delta", margin_delta),
            ("final_delta_delta", final_delta_delta),
        ]:
            if value is not None:
                score_stats[key].append(value)

        if base_correct and not cand_correct:
            if final_delta_delta is not None:
                if final_delta_delta > 0:
                    alignment["final_delta_up_correct_to_wrong"] += 1
                elif final_delta_delta < 0:
                    alignment["final_delta_down_correct_to_wrong"] += 1
            if answer_score_delta is not None and answer_score_delta > 0:
                alignment["answer_score_up_correct_to_wrong"] += 1
            if margin_delta is not None:
                if margin_delta > 0:
                    alignment["answer_margin_up_correct_to_wrong"] += 1
                elif margin_delta < 0:
                    alignment["answer_margin_down_correct_to_wrong"] += 1
        if not base_correct and cand_correct and final_delta_delta is not None:
            if final_delta_delta > 0:
                alignment["final_delta_up_wrong_to_correct"] += 1
            elif final_delta_delta < 0:
                alignment["final_delta_down_wrong_to_correct"] += 1

        rows.append(
            {
                "query_id": query_id,
                "question_id": base.get("question_id"),
                "answer": answer,
                "baseline_prediction": base.get("prediction"),
                "candidate_prediction": cand.get("prediction"),
                "baseline_correct": int(base_correct),
                "candidate_correct": int(cand_correct),
                "pair_changed": int(pair_changed),
                "baseline_memory_ids": " ".join(map(str, base_pair)),
                "candidate_memory_ids": " ".join(map(str, cand_pair)),
                "baseline_memory_source_ids": " ".join(_source_pair(base)),
                "candidate_memory_source_ids": " ".join(_source_pair(cand)),
                "baseline_answer_score": base_answer_score,
                "candidate_answer_score": cand_answer_score,
                "answer_score_delta": answer_score_delta,
                "baseline_answer_margin": base_margin,
                "candidate_answer_margin": cand_margin,
                "answer_margin_delta": margin_delta,
                "baseline_final_delta": base_final_delta,
                "candidate_final_delta": cand_final_delta,
                "final_delta_delta": final_delta_delta,
            }
        )

    total = len(common_ids)
    baseline_correct = counts["both_correct"] + counts["baseline_only_correct"]
    candidate_correct = counts["both_correct"] + counts["candidate_only_correct"]
    paired_summary = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_predictions": str(Path(args.baseline_predictions).resolve()),
        "candidate_predictions": str(Path(args.candidate_predictions).resolve()),
        "total_common": total,
        "missing_from_baseline": len(missing_baseline),
        "missing_from_candidate": len(missing_candidate),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": baseline_correct / total,
        "candidate_accuracy": candidate_correct / total,
        "net_correct_change": candidate_correct - baseline_correct,
        "net_accuracy_change": (candidate_correct - baseline_correct) / total,
        **counts,
    }

    pair_summary = {
        **pair_counts,
        "pair_changed_ratio": pair_counts["pair_changed"] / total,
        "pair_unchanged_baseline_accuracy": (
            pair_counts["pair_unchanged_baseline_correct"]
            / pair_counts["pair_unchanged"]
            if pair_counts["pair_unchanged"]
            else None
        ),
        "pair_unchanged_candidate_accuracy": (
            pair_counts["pair_unchanged_candidate_correct"]
            / pair_counts["pair_unchanged"]
            if pair_counts["pair_unchanged"]
            else None
        ),
        "pair_changed_baseline_accuracy": (
            pair_counts["pair_changed_baseline_correct"] / pair_counts["pair_changed"]
            if pair_counts["pair_changed"]
            else None
        ),
        "pair_changed_candidate_accuracy": (
            pair_counts["pair_changed_candidate_correct"] / pair_counts["pair_changed"]
            if pair_counts["pair_changed"]
            else None
        ),
    }
    score_summary = {
        "has_baseline_final_delta": any(
            baseline[query_id].get("final_delta") is not None for query_id in common_ids
        ),
        "has_candidate_final_delta": any(
            candidate[query_id].get("final_delta") is not None for query_id in common_ids
        ),
        "mean_answer_score_delta": _mean(score_stats["answer_score_delta"]),
        "mean_answer_margin_delta": _mean(score_stats["answer_margin_delta"]),
        "mean_final_delta_delta": _mean(score_stats["final_delta_delta"]),
        **alignment,
    }

    _write_json(output_dir / "paired_summary.json", paired_summary)
    _write_json(output_dir / "pair_change_summary.json", pair_summary)
    _write_json(output_dir / "score_alignment_summary.json", score_summary)

    case_path = output_dir / "paired_cases.csv"
    with case_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "paired_summary": paired_summary,
                "pair_change_summary": pair_summary,
                "score_alignment_summary": score_summary,
                "paired_cases": str(case_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
