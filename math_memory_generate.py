import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from lever_lm.math_memory.data import (
    load_experiences,
    load_mmlu_pro_math_split,
)
from lever_lm.math_memory.scoring import build_scorer


def _top_beams(
    proposals: List[Tuple[List[int], float]],
    beam_size: int,
) -> List[Tuple[List[int], float]]:
    proposals.sort(key=lambda item: item[1], reverse=True)
    return proposals[:beam_size]


def generate_for_query(
    query: Dict,
    memories: List[Dict],
    scorer,
    candidate_num: int,
    repeat: int,
    beam_size: int,
    shot_num: int,
    seed: int,
    score_mode: str,
) -> List[Dict]:
    if score_mode not in {"delta_logprob", "absolute_logprob"}:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    memory_size = len(memories)
    if candidate_num < shot_num:
        raise ValueError("candidate_num must be >= shot_num")
    if candidate_num > memory_size:
        raise ValueError(
            f"candidate_num={candidate_num} exceeds memory_size={memory_size}"
        )

    rows = []
    score_cache: Dict[Tuple[int, ...], float] = {}

    def get_absolute_scores(memory_sequences: List[List[int]]) -> List[float]:
        missing_sequences = [
            memory_ids
            for memory_ids in memory_sequences
            if tuple(memory_ids) not in score_cache
        ]
        if missing_sequences:
            missing_scores = scorer.score_gold_sequences(
                query, missing_sequences, memories
            )
            for memory_ids, score in zip(missing_sequences, missing_scores):
                score_cache[tuple(memory_ids)] = score
        return [score_cache[tuple(memory_ids)] for memory_ids in memory_sequences]

    for repeat_idx in range(repeat):
        rng = random.Random(seed + query["query_id"] * 1_000_003 + repeat_idx)
        candidate_ids = rng.sample(range(memory_size), candidate_num)
        beams: List[Tuple[List[int], float]] = [([], 0.0)]

        for _step in range(shot_num):
            proposals: List[Tuple[List[int], float]] = []
            for prefix, _prefix_score in beams:
                next_ids = [
                    memory_id for memory_id in candidate_ids if memory_id not in prefix
                ]
                memory_sequences = [prefix + [memory_id] for memory_id in next_ids]
                absolute_scores = get_absolute_scores(memory_sequences)
                if score_mode == "delta_logprob":
                    baseline_score = get_absolute_scores([prefix])[0]
                    scores = [score - baseline_score for score in absolute_scores]
                else:
                    scores = absolute_scores
                proposals.extend(zip(memory_sequences, scores))
            beams = _top_beams(proposals, beam_size)

        for rank, (memory_ids, score) in enumerate(beams):
            rows.append(
                {
                    "query_id": query["query_id"],
                    "question_id": query["question_id"],
                    "repeat": repeat_idx,
                    "rank": rank,
                    "memory_ids": memory_ids,
                    "score": float(score),
                }
            )
    return rows


def _build_metadata(args, memories: List[Dict], train_queries: List[Dict], test_queries: List[Dict]) -> Dict:
    return {
        "task": "mmlu_pro_math_memory",
        "shot_num": args.shot_num,
        "candidate_num": args.candidate_num,
        "repeat": args.repeat,
        "beam_size": args.beam_size,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "mock_data": args.mock_data,
        "mock_records": args.mock_records,
        "scorer_model": args.scorer_model,
        "score_mode": args.score_mode,
        "memory_size": len(memories),
        "train_query_count": len(train_queries),
        "test_query_count": len(test_queries),
        "resume_enabled": True,
    }


def _metadata_matches(existing: Dict, expected: Dict) -> bool:
    keys = [
        "task",
        "shot_num",
        "candidate_num",
        "repeat",
        "beam_size",
        "seed",
        "train_ratio",
        "mock_data",
        "mock_records",
        "scorer_model",
        "score_mode",
        "memory_size",
        "train_query_count",
        "test_query_count",
    ]
    return all(existing.get(key) == expected.get(key) for key in keys)


def _progress_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial.jsonl")


def _load_resume_rows(output_path: Path, progress_path: Path, metadata: Dict) -> List[Dict]:
    rows: List[Dict] = []
    if output_path.exists():
        try:
            with output_path.open(encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = None
        if existing is not None and not _metadata_matches(existing.get("metadata", {}), metadata):
            raise ValueError(
                f"Existing output metadata does not match requested generation: {output_path}. "
                "Use --overwrite or choose a different --output-file."
            )

    if progress_path.exists():
        with progress_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if not output_path.exists():
        return rows

    if existing is None:
        return rows
    return existing.get("data", [])


def _dedupe_complete_rows(rows: List[Dict], expected_rows_per_query: int) -> List[Dict]:
    rows_by_query: Dict[int, List[Dict]] = {}
    for row in rows:
        rows_by_query.setdefault(row["query_id"], []).append(row)

    complete_rows: List[Dict] = []
    for query_id in sorted(rows_by_query):
        query_rows = rows_by_query[query_id]
        unique = {
            (row.get("repeat"), row.get("rank")): row
            for row in query_rows
        }
        if len(unique) == expected_rows_per_query:
            complete_rows.extend(
                unique[key] for key in sorted(unique, key=lambda item: (item[0], item[1]))
            )
    return complete_rows


def _write_output_atomic(output_path: Path, metadata: Dict, rows: List[Dict]) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output = {"metadata": metadata, "data": rows}
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_path)


def _rewrite_progress(progress_path: Path, rows: List[Dict]) -> None:
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(progress_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2-shot experience-memory supervision for Lever-LM."
    )
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--shot-num", type=int, default=2)
    parser.add_argument("--candidate-num", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=4)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore and replace existing generated JSON/progress files.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["delta_logprob", "absolute_logprob"],
        default="delta_logprob",
        help=(
            "delta_logprob ranks each added memory by its marginal gain over the "
            "current prefix; absolute_logprob preserves the previous behavior."
        ),
    )
    args = parser.parse_args()

    memories = load_experiences(args.experience_file)
    train_queries, test_queries = load_mmlu_pro_math_split(
        seed=args.seed,
        train_ratio=args.train_ratio,
        mock_data=args.mock_data,
        mock_records=args.mock_records,
    )
    if args.anchor_limit is not None:
        train_queries = train_queries[: args.anchor_limit]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = _progress_path(output_path)
    metadata = _build_metadata(args, memories, train_queries, test_queries)
    expected_rows_per_query = args.repeat * args.beam_size
    expected_total_rows = len(train_queries) * expected_rows_per_query

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    rows = _dedupe_complete_rows(
        _load_resume_rows(output_path, progress_path, metadata),
        expected_rows_per_query=expected_rows_per_query,
    )
    completed_query_ids = {row["query_id"] for row in rows}
    if rows:
        _rewrite_progress(progress_path, rows)
        _write_output_atomic(output_path, metadata, rows)
        print(
            f"Resuming generation from {len(completed_query_ids)}/{len(train_queries)} "
            f"completed queries and {len(rows)}/{expected_total_rows} rows."
        )

    if len(rows) == expected_total_rows:
        print(f"Generated data already complete: {output_path}")
        return

    scorer = build_scorer(
        model_name=args.scorer_model,
        device=args.scorer_device,
        dtype=args.scorer_dtype,
        batch_size=args.scorer_batch_size,
        max_length=args.scorer_max_length,
    )

    for query in tqdm(train_queries, desc="Generating D_M", ncols=100):
        if query["query_id"] in completed_query_ids:
            continue
        query_rows = generate_for_query(
            query=query,
            memories=memories,
            scorer=scorer,
            candidate_num=args.candidate_num,
            repeat=args.repeat,
            beam_size=args.beam_size,
            shot_num=args.shot_num,
            seed=args.seed,
            score_mode=args.score_mode,
        )
        with progress_path.open("a", encoding="utf-8") as f:
            for row in query_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.extend(query_rows)
        completed_query_ids.add(query["query_id"])
        metadata["completed_query_count"] = len(completed_query_ids)
        metadata["is_complete"] = len(rows) == expected_total_rows
        _write_output_atomic(output_path, metadata, rows)

    metadata["completed_query_count"] = len(completed_query_ids)
    metadata["is_complete"] = len(rows) == expected_total_rows
    _write_output_atomic(output_path, metadata, rows)
    print(f"Saved {len(rows)} training sequences to {output_path}")


if __name__ == "__main__":
    main()
