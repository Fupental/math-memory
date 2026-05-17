import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _dtype_from_name(name: str):
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "16"}:
        return torch.float16
    if name in {"fp32", "float32", "32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


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
                "--embedding-max-memory entries must look like '0:22GiB' or 'cpu:64GiB'"
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


class MockTextEmbedder:
    def __init__(self, emb_dim: int = 32) -> None:
        self.model_name = "mock"
        self.emb_dim = emb_dim

    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        del batch_size
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = []
            while len(values) < self.emb_dim:
                for byte in digest:
                    values.append((byte / 255.0) * 2.0 - 1.0)
                    if len(values) == self.emb_dim:
                        break
                digest = hashlib.sha256(digest).digest()
            rows.append(values)
        return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


class HFTextEmbedder:
    """Frozen text embedder used to replace CLIP for math-memory Lever-LM."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bf16",
        max_length: int = 1024,
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
        max_memory: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device_map = _normalize_optional_arg(device_map)
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
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
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if self.device_map is None:
            self.model.to(self.device)
            self.input_device = self.device
        else:
            self.input_device = _first_parameter_device(self.model, self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        embeddings = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Embedding with {self.model_name}",
            ncols=100,
        ):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.input_device)
            outputs = self.model(**inputs)
            attention_mask = inputs["attention_mask"].to(outputs.last_hidden_state.device)
            pooled = self._last_token_pool(outputs.last_hidden_state, attention_mask)
            pooled = F.normalize(pooled.float(), dim=-1)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)


def build_embedder(
    model_name: str,
    device: str = "cuda",
    dtype: str = "bf16",
    max_length: int = 1024,
    mock_emb_dim: int = 32,
    device_map: Optional[str] = None,
    max_memory: Optional[str] = None,
):
    if model_name == "mock":
        return MockTextEmbedder(mock_emb_dim)
    return HFTextEmbedder(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_length=max_length,
        device_map=device_map,
        max_memory=max_memory,
    )


def load_or_create_embeddings(
    cache_path: str,
    texts: List[str],
    embedder,
    batch_size: int,
) -> torch.Tensor:
    path = Path(cache_path)
    texts_hash = hashlib.sha256("\n\n".join(texts).encode("utf-8")).hexdigest()
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        if (
            isinstance(payload, dict)
            and payload.get("model_name") == embedder.model_name
            and payload.get("num_texts") == len(texts)
            and payload.get("texts_hash") == texts_hash
        ):
            return payload["embeddings"].float()

    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = embedder.encode(texts, batch_size=batch_size).float()
    payload: Dict[str, object] = {
        "model_name": embedder.model_name,
        "num_texts": len(texts),
        "texts_hash": texts_hash,
        "embeddings": embeddings,
    }
    torch.save(payload, path)
    return embeddings
