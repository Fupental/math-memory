from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


class MathMemoryLeverLM(nn.Module):
    """Two-layer Transformer that generates experience ids instead of text."""

    def __init__(
        self,
        memory_size: int,
        encoder_emb_dim: int,
        n_embd: int = 512,
        n_head: int = 8,
        n_layer: int = 2,
        adapter_hidden_mult: int = 4,
        max_positions: int = 16,
        normalize_encoder_emb: bool = True,
    ) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.eos_token_id = memory_size
        self.bos_token_id = memory_size + 1
        self.query_token_id = memory_size + 2
        self.vocab_size = memory_size + 3
        self.encoder_emb_dim = encoder_emb_dim
        self.normalize_encoder_emb = normalize_encoder_emb

        config = GPT2Config(
            vocab_size=self.vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_positions=max_positions,
            n_ctx=max_positions,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )
        self.lm_model = GPT2LMHeadModel(config)
        self.adapter = nn.Sequential(
            nn.Linear(encoder_emb_dim, n_embd * adapter_hidden_mult),
            nn.ReLU(),
            nn.Linear(n_embd * adapter_hidden_mult, n_embd),
        )

    def _adapt(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize_encoder_emb:
            embeddings = F.normalize(embeddings.float(), dim=-1)
        return self.adapter(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        query_emb: torch.Tensor,
        memory_embs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.lm_model.transformer.wte(input_ids)
        inputs_embeds[:, 1] = inputs_embeds[:, 1] + self._adapt(query_emb)

        if memory_embs is not None and memory_embs.numel() > 0:
            batch_size, memory_count, emb_dim = memory_embs.shape
            flat = memory_embs.reshape(batch_size * memory_count, emb_dim)
            adapted = self._adapt(flat).reshape(batch_size, memory_count, -1)
            inputs_embeds[:, 2 : 2 + memory_count] = (
                inputs_embeds[:, 2 : 2 + memory_count] + adapted
            )

        return self.lm_model(inputs_embeds=inputs_embeds, labels=labels)

    @torch.inference_mode()
    def generate_memory_ids(
        self,
        query_embs: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        shot_num: int = 2,
    ) -> torch.Tensor:
        device = query_embs.device
        batch_size = query_embs.shape[0]
        prefix = torch.tensor(
            [[self.bos_token_id, self.query_token_id] for _ in range(batch_size)],
            dtype=torch.long,
            device=device,
        )
        memory_embedding_table = memory_embedding_table.to(device)

        for _ in range(shot_num):
            chosen = prefix[:, 2:]
            if chosen.numel() == 0:
                memory_embs = torch.empty(
                    batch_size,
                    0,
                    self.encoder_emb_dim,
                    dtype=query_embs.dtype,
                    device=device,
                )
            else:
                memory_embs = memory_embedding_table[chosen]
            output = self.forward(
                input_ids=prefix,
                query_emb=query_embs,
                memory_embs=memory_embs,
                labels=None,
            )
            logits = output.logits[:, -1, :]
            logits[:, self.memory_size :] = -torch.inf
            for row_idx in range(batch_size):
                if chosen.shape[1] > 0:
                    logits[row_idx, chosen[row_idx]] = -torch.inf
            next_id = logits.argmax(dim=-1, keepdim=True)
            prefix = torch.cat([prefix, next_id], dim=1)

        return prefix[:, 2:]


def checkpoint_metadata(model: MathMemoryLeverLM, extra: Dict) -> Dict:
    metadata = {
        "memory_size": model.memory_size,
        "encoder_emb_dim": model.encoder_emb_dim,
        "n_embd": model.lm_model.config.n_embd,
        "n_head": model.lm_model.config.n_head,
        "n_layer": model.lm_model.config.n_layer,
        "max_positions": model.lm_model.config.n_positions,
        "eos_token_id": model.eos_token_id,
        "bos_token_id": model.bos_token_id,
        "query_token_id": model.query_token_id,
        "vocab_size": model.vocab_size,
    }
    metadata.update(extra)
    return metadata

