import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _natural_key(key: str) -> Tuple[str, int]:
    match = re.match(r"^([A-Za-z_]+)(\d+)$", str(key))
    if match:
        return match.group(1), int(match.group(2))
    return str(key), -1


def load_experiences(experience_file: str) -> List[Dict[str, Any]]:
    """Load the experience JSON as the memory vocabulary.

    Memory ids are dense integer token ids. The source id (for example "G17") is
    kept only for reporting and prompt display.
    """

    path = Path(experience_file)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict in {experience_file}, got {type(raw)}")

    memories = []
    for memory_id, source_id in enumerate(sorted(raw, key=_natural_key)):
        text = raw[source_id]
        if not isinstance(text, str):
            raise ValueError(f"Experience {source_id} is not a string")
        memories.append(
            {
                "memory_id": memory_id,
                "source_id": source_id,
                "text": text.strip(),
            }
        )
    return memories


def _normalize_mmlu_record(record: Dict[str, Any], query_id: int) -> Dict[str, Any]:
    answer = record["answer"]
    if isinstance(answer, int):
        answer_index = answer
        answer = LETTERS[answer_index]
    else:
        answer = str(answer).strip().upper()
        answer_index = record.get("answer_index")
        if answer_index is None and answer in LETTERS:
            answer_index = LETTERS.index(answer)

    return {
        "query_id": query_id,
        "question_id": record.get("question_id", query_id),
        "question": str(record["question"]).strip(),
        "options": [str(option).strip() for option in record["options"]],
        "answer": answer,
        "answer_index": int(answer_index),
        "category": record.get("category", "math"),
        "src": record.get("src", ""),
    }


def _mock_math_records(num_records: int = 20) -> List[Dict[str, Any]]:
    records = []
    for query_id in range(num_records):
        a = query_id + 2
        b = query_id % 5 + 3
        answer_value = a + b
        options = [str(answer_value + offset) for offset in [-2, -1, 0, 1]]
        records.append(
            {
                "query_id": query_id,
                "question_id": query_id,
                "question": f"What is {a} + {b}?",
                "options": options,
                "answer": "C",
                "answer_index": 2,
                "category": "math",
                "src": "mock",
            }
        )
    return records


def load_mmlu_pro_math_split(
    seed: int = 42,
    train_ratio: float = 0.8,
    mock_data: bool = False,
    mock_records: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load MMLU-Pro math and create a deterministic train/test split.

    The public Hugging Face dataset has no train split. The main experiment uses
    the official test split's math category and splits it deterministically.
    """

    if mock_data:
        records = _mock_math_records(mock_records)
    else:
        from datasets import load_dataset

        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        records = [
            record
            for record in ds.to_list()
            if str(record.get("category", "")).lower() == "math"
        ]
        records = sorted(records, key=lambda item: item.get("question_id", 0))
        records = [
            _normalize_mmlu_record(record, query_id)
            for query_id, record in enumerate(records)
        ]

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_len = int(len(shuffled) * train_ratio)
    train_queries = shuffled[:train_len]
    test_queries = shuffled[train_len:]
    return train_queries, test_queries


def query_to_text(query: Dict[str, Any]) -> str:
    option_lines = [
        f"({LETTERS[i]}) {option}" for i, option in enumerate(query["options"])
    ]
    return query["question"] + "\n" + "\n".join(option_lines)


def build_answer_prompt(
    query: Dict[str, Any],
    memories: List[Dict[str, Any]],
    memory_ids: List[int],
) -> str:
    if memory_ids:
        memory_block = "\n".join(
            f"[{memories[memory_id]['source_id']}] {memories[memory_id]['text']}"
            for memory_id in memory_ids
        )
    else:
        memory_block = "None"
    option_lines = [
        f"({LETTERS[i]}) {option}" for i, option in enumerate(query["options"])
    ]
    return (
        "Use the following experiences if they are relevant to the math problem.\n"
        "Answer with only one option letter.\n\n"
        f"<experiences>\n{memory_block}\n</experiences>\n\n"
        f"<problem>\n{query['question']}\n\n"
        f"{chr(10).join(option_lines)}\n</problem>\n\n"
        "Answer:"
    )


def safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )

