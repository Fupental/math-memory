import argparse
import json
import random
import shlex
import sys
from pathlib import Path
from typing import Dict, List

from lever_lm.math_memory.data import load_experiences


def _command_text() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def _candidate_ids_for_row(row: Dict, seed: int, memory_size: int, candidate_num: int) -> List[int]:
    repeat_idx = int(row.get("repeat", 0))
    rng = random.Random(seed + int(row["query_id"]) * 1_000_003 + repeat_idx)
    return rng.sample(range(memory_size), candidate_num)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct and add original Random-k candidate_ids to legacy "
            "math-memory generated data."
        )
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--experience-file", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace it.")

    payload = json.load(input_path.open(encoding="utf-8"))
    metadata = dict(payload["metadata"])
    rows = payload["data"]
    memories = load_experiences(args.experience_file)

    seed = int(metadata["seed"])
    candidate_num = int(metadata["candidate_num"])
    memory_size = int(metadata.get("memory_size", len(memories)))
    if memory_size != len(memories):
        raise ValueError(
            f"metadata memory_size={memory_size} but experience file has {len(memories)} rows"
        )
    if candidate_num > memory_size:
        raise ValueError("candidate_num cannot exceed memory_size")

    missing_oracle_rows = []
    bad_candidate_rows = []
    new_rows = []
    for row in rows:
        new_row = dict(row)
        candidate_ids = _candidate_ids_for_row(
            row=row,
            seed=seed,
            memory_size=memory_size,
            candidate_num=candidate_num,
        )
        if len(candidate_ids) != candidate_num or len(set(candidate_ids)) != candidate_num:
            bad_candidate_rows.append(row)
        candidate_set = set(candidate_ids)
        if not all(int(memory_id) in candidate_set for memory_id in row["memory_ids"]):
            missing_oracle_rows.append(row)
        new_row["candidate_ids"] = candidate_ids
        new_row["candidate_source_ids"] = [
            memories[memory_id]["source_id"] for memory_id in candidate_ids
        ]
        new_rows.append(new_row)

    if bad_candidate_rows:
        raise ValueError(f"Found invalid reconstructed candidate rows: {len(bad_candidate_rows)}")
    if missing_oracle_rows:
        first = missing_oracle_rows[0]
        raise ValueError(
            "Reconstructed candidates do not contain oracle memories for "
            f"{len(missing_oracle_rows)} rows; first query_id={first['query_id']} "
            f"memory_ids={first['memory_ids']}"
        )

    metadata.update(
        {
            "candidate_mode": "random",
            "candidate_ids_reconstructed": True,
            "candidate_ids_source_file": str(input_path.resolve()),
            "candidate_ids_command": _command_text(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "data": new_rows}, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_path)

    print(
        json.dumps(
            {
                "input_file": str(input_path.resolve()),
                "output_file": str(output_path.resolve()),
                "rows": len(new_rows),
                "candidate_num": candidate_num,
                "memory_size": memory_size,
                "missing_oracle_rows": len(missing_oracle_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
