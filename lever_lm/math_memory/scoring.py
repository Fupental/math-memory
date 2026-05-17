import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from lever_lm.math_memory.data import LETTERS, build_answer_prompt


def _dtype_from_name(name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "16"}:
        return torch.float16
    if name in {"fp32", "float32", "32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _normalize_optional_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _parse_max_memory(value: Optional[str]) -> Optional[Dict[Any, str]]:
    """Parse strings like '0:22GiB,1:22GiB,cpu:64GiB' for HF device_map."""

    value = _normalize_optional_arg(value)
    if value is None:
        return None
    result: Dict[Any, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--scorer-max-memory entries must look like '0:22GiB' or 'cpu:64GiB'"
            )
        key, memory = item.split(":", 1)
        key = key.strip()
        memory = memory.strip()
        if not key or not memory:
            raise ValueError(f"Invalid max_memory entry: {item!r}")
        result[int(key) if key.isdigit() else key] = memory
    return result or None


def _first_parameter_device(model, fallback: torch.device) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return fallback


class MockChoiceScorer:
    """Deterministic scorer for local smoke tests without loading Qwen3."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    @staticmethod
    def _hash_float(parts: List[Any]) -> float:
        text = "::".join(map(str, parts))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16**12)

    def score_gold_sequences(
        self,
        query: Dict[str, Any],
        memory_id_sequences: List[List[int]],
        memories: List[Dict[str, Any]],
    ) -> List[float]:
        del memories
        return [
            self._hash_float([query["query_id"], query["answer"], *memory_ids])
            for memory_ids in memory_id_sequences
        ]

    def predict(
        self,
        query: Dict[str, Any],
        memory_ids: List[int],
        memories: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, float]]:
        del memories
        labels = LETTERS[: len(query["options"])]
        scores = {
            label: self._hash_float([query["query_id"], label, *memory_ids])
            for label in labels
        }
        prediction = max(scores, key=scores.get)
        return prediction, scores


class QwenChoiceScorer:
    """Score option letters by conditional log probability under a causal LM."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bf16",
        batch_size: int = 4,
        max_length: int = 4096,
        trust_remote_code: bool = True,
        use_chat_template: bool = True,
        device_map: Optional[str] = None,
        max_memory: Optional[str] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device_map = _normalize_optional_arg(device_map)
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_chat_template = use_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "torch_dtype": _dtype_from_name(dtype),
            "trust_remote_code": trust_remote_code,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            parsed_max_memory = _parse_max_memory(max_memory)
            if parsed_max_memory is not None:
                model_kwargs["max_memory"] = parsed_max_memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if self.device_map is None:
            self.model.to(self.device)
            self.input_device = self.device
        else:
            self.input_device = _first_parameter_device(self.model, self.device)
        self.model.eval()

    def _render_prompt(
        self,
        query: Dict[str, Any],
        memory_ids: List[int],
        memories: List[Dict[str, Any]],
    ) -> str:
        prompt = build_answer_prompt(query, memories, memory_ids)
        if not self.use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        messages = [
            {
                "role": "system",
                "content": "You are a careful math expert. Return only the option letter.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def _encode_prompt_label(self, prompt: str, label: str) -> Tuple[List[int], List[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        label_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
        if not label_ids:
            raise ValueError(f"Label produced no tokens: {label!r}")

        max_prompt_len = self.max_length - len(label_ids)
        if max_prompt_len <= 0:
            raise ValueError(
                f"max_length={self.max_length} is too small for label {label!r}"
            )
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + label_ids
        label_mask = [0] * len(prompt_ids) + [1] * len(label_ids)
        return input_ids, label_mask

    def _single_token_label_ids(self, labels: List[str]) -> List[int]:
        label_token_ids: List[int] = []
        for label in labels:
            token_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(token_ids) != 1:
                return []
            label_token_ids.append(token_ids[0])
        return label_token_ids

    def _encode_prompt_for_next_token(self, prompt: str, reserve_tokens: int) -> List[int]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        max_prompt_len = self.max_length - reserve_tokens
        if max_prompt_len <= 0:
            raise ValueError(
                f"max_length={self.max_length} is too small for {reserve_tokens} reserved tokens"
            )
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        if not prompt_ids:
            raise ValueError("Prompt produced no tokens")
        return prompt_ids

    @torch.inference_mode()
    def _score_prompt_labels(self, prompt_labels: List[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for batch in _batched(prompt_labels, self.batch_size):
            encoded = [self._encode_prompt_label(prompt, label) for prompt, label in batch]
            max_len = max(len(input_ids) for input_ids, _ in encoded)
            input_rows = []
            attention_rows = []
            mask_rows = []
            for input_ids, label_mask in encoded:
                pad_len = max_len - len(input_ids)
                input_rows.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                attention_rows.append([1] * len(input_ids) + [0] * pad_len)
                mask_rows.append(label_mask + [0] * pad_len)

            input_tensor = torch.tensor(
                input_rows, dtype=torch.long, device=self.input_device
            )
            attention_mask = torch.tensor(
                attention_rows, dtype=torch.long, device=self.input_device
            )
            label_mask_tensor = torch.tensor(
                mask_rows, dtype=torch.bool, device=self.input_device
            )

            output = self.model(input_ids=input_tensor, attention_mask=attention_mask)
            log_probs = torch.log_softmax(output.logits[:, :-1, :], dim=-1)
            output_device = log_probs.device
            target_ids = input_tensor[:, 1:].to(output_device)
            target_mask = label_mask_tensor[:, 1:].to(output_device)
            token_scores = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            token_scores = token_scores.masked_fill(~target_mask, 0.0)
            scores.extend(token_scores.sum(dim=-1).float().cpu().tolist())
        return scores

    @torch.inference_mode()
    def _score_next_token_labels(
        self,
        prompts: List[str],
        labels: List[str],
    ) -> List[List[float]]:
        label_token_ids = self._single_token_label_ids(labels)
        if not label_token_ids:
            raise ValueError("Next-token scoring requires single-token labels")

        label_tensor = torch.tensor(
            label_token_ids, dtype=torch.long, device=self.input_device
        )
        all_scores: List[List[float]] = []
        for batch in _batched(prompts, self.batch_size):
            encoded = [
                self._encode_prompt_for_next_token(prompt, reserve_tokens=1)
                for prompt in batch
            ]
            max_len = max(len(input_ids) for input_ids in encoded)
            input_rows = []
            attention_rows = []
            for input_ids in encoded:
                pad_len = max_len - len(input_ids)
                input_rows.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                attention_rows.append([1] * len(input_ids) + [0] * pad_len)

            input_tensor = torch.tensor(
                input_rows, dtype=torch.long, device=self.input_device
            )
            attention_mask = torch.tensor(
                attention_rows, dtype=torch.long, device=self.input_device
            )

            output = self.model(input_ids=input_tensor, attention_mask=attention_mask)
            logits = output.logits
            output_device = logits.device
            input_tensor = input_tensor.to(output_device)
            attention_mask = attention_mask.to(output_device)
            label_tensor_on_output = label_tensor.to(output_device)
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_tensor.size(0), device=output_device)
            next_logits = logits[batch_indices, last_token_indices, :]
            log_probs = torch.log_softmax(next_logits, dim=-1)
            batch_scores = log_probs.index_select(dim=-1, index=label_tensor_on_output)
            all_scores.extend(batch_scores.float().cpu().tolist())
        return all_scores

    def score_gold_sequences(
        self,
        query: Dict[str, Any],
        memory_id_sequences: List[List[int]],
        memories: List[Dict[str, Any]],
    ) -> List[float]:
        label = " " + query["answer"]
        if self._single_token_label_ids([label]):
            prompts = [
                self._render_prompt(query, memory_ids, memories)
                for memory_ids in memory_id_sequences
            ]
            return [
                scores[0] for scores in self._score_next_token_labels(prompts, [label])
            ]

        prompt_labels = [
            (self._render_prompt(query, memory_ids, memories), label)
            for memory_ids in memory_id_sequences
        ]
        return self._score_prompt_labels(prompt_labels)

    def predict(
        self,
        query: Dict[str, Any],
        memory_ids: List[int],
        memories: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, float]]:
        labels = LETTERS[: len(query["options"])]
        prompt = self._render_prompt(query, memory_ids, memories)
        label_texts = [" " + label for label in labels]
        if self._single_token_label_ids(label_texts):
            raw_scores = self._score_next_token_labels([prompt], label_texts)[0]
        else:
            prompt_labels = [(prompt, label_text) for label_text in label_texts]
            raw_scores = self._score_prompt_labels(prompt_labels)
        scores = dict(zip(labels, raw_scores))
        prediction = max(scores, key=scores.get)
        return prediction, scores


def build_scorer(
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
    max_length: int,
    device_map: Optional[str] = None,
    max_memory: Optional[str] = None,
):
    if model_name == "mock":
        return MockChoiceScorer()
    return QwenChoiceScorer(
        model_name=model_name,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        max_length=max_length,
        device_map=device_map,
        max_memory=max_memory,
    )


def score_many_with_progress(
    scorer,
    query: Dict[str, Any],
    memory_id_sequences: List[List[int]],
    memories: List[Dict[str, Any]],
    batch_size: int,
    desc: str,
) -> List[float]:
    scores: List[float] = []
    for batch in tqdm(
        list(_batched(memory_id_sequences, batch_size)),
        desc=desc,
        leave=False,
        ncols=100,
    ):
        scores.extend(scorer.score_gold_sequences(query, batch, memories))
    return scores
