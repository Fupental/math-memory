import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from lever_lm.math_memory.data import load_experiences, load_mmlu_pro_math_split


def main():
    parser = argparse.ArgumentParser(
        description="Generate random fixed memory sequences for MMLU-Pro math."
    )
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--shot-num", type=int, default=2)
    parser.add_argument("--samples-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--mock-records", type=int, default=20)
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

    memory_size = len(memories)
    if args.shot_num > memory_size:
        raise ValueError(
            f"shot_num={args.shot_num} exceeds memory_size={memory_size}"
        )

    rows = []
    for query in tqdm(train_queries, desc="Generating random sequences", ncols=100):
        for sample_idx in range(args.samples_per_query):
            rng = random.Random(
                args.seed + query["query_id"] * 1_000_003 + sample_idx
            )
            memory_ids = rng.sample(range(memory_size), args.shot_num)
            rng.shuffle(memory_ids)
            rows.append(
                {
                    "query_id": query["query_id"],
                    "question_id": query["question_id"],
                    "repeat": sample_idx,
                    "rank": sample_idx,
                    "memory_ids": memory_ids,
                    "score": None,
                }
            )

    output = {
        "metadata": {
            "task": "mmlu_pro_math_memory",
            "method": "random_sequences",
            "shot_num": args.shot_num,
            "samples_per_query": args.samples_per_query,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "mock_data": args.mock_data,
            "mock_records": args.mock_records,
            "memory_size": memory_size,
            "train_query_count": len(train_queries),
            "test_query_count": len(test_queries),
            "score_mode": "random",
        },
        "data": rows,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(rows)} random sequences to {output_path}")


if __name__ == "__main__":
    main()
