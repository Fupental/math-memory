import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from lever_lm.math_memory.data import load_experiences, load_mmlu_pro_math_split
from lever_lm.math_memory.scoring import build_scorer


def _default_output_path(generated_file: Path) -> Path:
    return generated_file.with_name(f"{generated_file.stem}_scored{generated_file.suffix}")


def _load_train_query_map(metadata: Dict) -> Dict[int, Dict]:
    train_queries, _test_queries = load_mmlu_pro_math_split(
        seed=metadata["seed"],
        train_ratio=metadata["train_ratio"],
        mock_data=metadata.get("mock_data", False),
        mock_records=metadata.get("mock_records", 20),
    )
    return {query["query_id"]: query for query in train_queries}


def _group_rows_by_query(rows: List[Dict]) -> Dict[int, List[Dict]]:
    rows_by_query: Dict[int, List[Dict]] = {}
    for row in rows:
        rows_by_query.setdefault(row["query_id"], []).append(row)
    return rows_by_query


def _sequence_sort_key(memory_ids: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    return len(memory_ids), memory_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add empty/prefix/full Qwen logprob scores and total_delta to generated "
            "math-memory SFT sequences."
        )
    )
    parser.add_argument("--generated-file", required=True)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-device", default="cuda")
    parser.add_argument("--scorer-dtype", default="bf16")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=4096)
    args = parser.parse_args()

    generated_path = Path(args.generated_file)
    output_path = Path(args.output_file) if args.output_file else _default_output_path(generated_path)

    generated = json.load(open(generated_path, encoding="utf-8"))
    metadata = generated["metadata"]
    rows = generated["data"]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Generated file contains no rows to score")

    memories = load_experiences(args.experience_file)
    query_by_id = _load_train_query_map(metadata)
    missing_query_ids = sorted(
        {row["query_id"] for row in rows if row["query_id"] not in query_by_id}
    )
    if missing_query_ids:
        raise ValueError(
            "Generated rows reference query ids outside the generated train split: "
            f"{missing_query_ids[:5]}"
        )

    scorer = build_scorer(
        model_name=args.scorer_model,
        device=args.scorer_device,
        dtype=args.scorer_dtype,
        batch_size=args.scorer_batch_size,
        max_length=args.scorer_max_length,
    )

    rows_by_query = _group_rows_by_query(rows)
    score_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    new_rows_by_key: Dict[Tuple[int, int, int], Dict] = {}

    for query_id, query_rows in tqdm(
        rows_by_query.items(), desc="Scoring generated rows", ncols=100
    ):
        query = query_by_id[query_id]
        needed_sequences = {()}
        for row in query_rows:
            memory_ids = tuple(row["memory_ids"])
            if len(memory_ids) != 2:
                raise ValueError(
                    "This scorer currently expects 2-shot rows; "
                    f"got memory_ids={memory_ids}"
                )
            needed_sequences.add((memory_ids[0],))
            needed_sequences.add(memory_ids)

        needed_list = [
            list(memory_ids)
            for memory_ids in sorted(needed_sequences, key=_sequence_sort_key)
        ]
        scores = scorer.score_gold_sequences(query, needed_list, memories)
        for memory_ids, score in zip(needed_list, scores):
            score_cache[(query_id, tuple(memory_ids))] = float(score)

        for row in query_rows:
            memory_ids = tuple(row["memory_ids"])
            first_memory = (memory_ids[0],)
            empty_score = score_cache[(query_id, ())]
            prefix_score = score_cache[(query_id, first_memory)]
            full_score = score_cache[(query_id, memory_ids)]

            new_row = dict(row)
            new_row.update(
                {
                    "empty_score": empty_score,
                    "prefix_score": prefix_score,
                    "full_score": full_score,
                    "step0_delta": prefix_score - empty_score,
                    "step1_delta": full_score - prefix_score,
                    "total_delta": full_score - empty_score,
                }
            )
            new_rows_by_key[(row["query_id"], row["repeat"], row["rank"])] = new_row

    new_rows = [
        new_rows_by_key[(row["query_id"], row["repeat"], row["rank"])] for row in rows
    ]
    new_metadata = dict(metadata)
    new_metadata.update(
        {
            "score_details_added": True,
            "score_details_source_file": str(generated_path),
            "score_details_scorer_model": args.scorer_model,
            "score_details_scorer_dtype": args.scorer_dtype,
            "score_details_scorer_batch_size": args.scorer_batch_size,
            "score_details_scorer_max_length": args.scorer_max_length,
            "score_details_row_limit": args.limit,
            "score_details_fields": [
                "empty_score",
                "prefix_score",
                "full_score",
                "step0_delta",
                "step1_delta",
                "total_delta",
            ],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": new_metadata, "data": new_rows}, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_path)

    print(f"Saved scored generated file to {output_path}")
    print(f"rows: {len(new_rows)}")
    print(f"unique score cache entries: {len(score_cache)}")


if __name__ == "__main__":
    main()
