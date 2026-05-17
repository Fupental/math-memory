import math
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class _SimpleCausalLM(nn.Module):
    """Small Transformer LM used for CPU/mock smoke tests."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        max_positions: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_positions=max_positions,
            n_ctx=max_positions,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.transformer = SimpleNamespace(wte=nn.Embedding(vocab_size, n_embd))
        self.position_embedding = nn.Embedding(max_positions, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        batch_size, seq_len, _hidden = inputs_embeds.shape
        positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        hidden = inputs_embeds + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden_states)


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
        model_backend: str = "gpt2",
    ) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.eos_token_id = memory_size
        self.bos_token_id = memory_size + 1
        self.query_token_id = memory_size + 2
        self.vocab_size = memory_size + 3
        self.encoder_emb_dim = encoder_emb_dim
        self.normalize_encoder_emb = normalize_encoder_emb
        self.model_backend = model_backend

        if model_backend == "gpt2":
            from transformers import GPT2Config, GPT2LMHeadModel

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
        elif model_backend == "simple":
            self.lm_model = _SimpleCausalLM(
                vocab_size=self.vocab_size,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                max_positions=max_positions,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
            )
        else:
            raise ValueError(f"Unsupported model_backend: {model_backend}")
        self.adapter = nn.Sequential(
            nn.Linear(encoder_emb_dim, n_embd * adapter_hidden_mult),
            nn.ReLU(),
            nn.Linear(n_embd * adapter_hidden_mult, n_embd),
        )
        self.value_head = nn.Linear(n_embd, 1)

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
        output_hidden_states: bool = False,
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

        return self.lm_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

    def _prefix_memory_embs(
        self,
        prefix: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        query_embs: torch.Tensor,
    ) -> torch.Tensor:
        chosen = prefix[:, 2:]
        if chosen.numel() == 0:
            return torch.empty(
                prefix.shape[0],
                0,
                self.encoder_emb_dim,
                dtype=query_embs.dtype,
                device=prefix.device,
            )
        return memory_embedding_table[chosen]

    def _mask_memory_logits(
        self,
        logits: torch.Tensor,
        chosen: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.clone()
        logits[:, self.memory_size :] = -torch.inf
        if chosen.numel() > 0:
            logits.scatter_(1, chosen, -torch.inf)
        return logits

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
            return logits
        values, _indices = torch.topk(logits, k=top_k, dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        return logits.masked_fill(logits < cutoff, -torch.inf)

    @torch.no_grad()
    def _mean_memory_logits_for_prefix(
        self,
        prefix_ids: Sequence[int],
        debias_query_embs: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Estimate global memory-token bias for one fixed prefix."""

        device = memory_embedding_table.device
        rows = []
        for start in range(0, debias_query_embs.shape[0], batch_size):
            query_batch = debias_query_embs[start : start + batch_size].to(device)
            prefix = torch.tensor(
                [
                    [self.bos_token_id, self.query_token_id, *prefix_ids]
                    for _ in range(query_batch.shape[0])
                ],
                dtype=torch.long,
                device=device,
            )
            memory_embs = self._prefix_memory_embs(
                prefix, memory_embedding_table, query_batch
            )
            output = self.forward(
                input_ids=prefix,
                query_emb=query_batch,
                memory_embs=memory_embs,
                labels=None,
            )
            rows.append(output.logits[:, -1, : self.memory_size].detach())
        return torch.cat(rows, dim=0).mean(dim=0)

    @torch.no_grad()
    def _bias_for_prefix_rows(
        self,
        prefix: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        debias_query_embs: Optional[torch.Tensor],
        debias_batch_size: int,
    ) -> torch.Tensor:
        if debias_query_embs is None:
            return torch.zeros(
                prefix.shape[0],
                self.memory_size,
                dtype=memory_embedding_table.dtype,
                device=memory_embedding_table.device,
            )

        chosen = prefix[:, 2:].detach().cpu().tolist()
        unique_prefixes = sorted({tuple(row) for row in chosen})
        bias_cache = {
            prefix_ids: self._mean_memory_logits_for_prefix(
                prefix_ids=prefix_ids,
                debias_query_embs=debias_query_embs,
                memory_embedding_table=memory_embedding_table,
                batch_size=debias_batch_size,
            )
            for prefix_ids in unique_prefixes
        }
        return torch.stack([bias_cache[tuple(row)] for row in chosen], dim=0)

    def _selection_policy_logits(
        self,
        raw_logits: torch.Tensor,
        chosen: torch.Tensor,
        selection_mode: str,
        temperature: float,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if selection_mode not in {"raw", "debiased_topk", "debiased_policy"}:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")
        policy_logits = raw_logits
        if selection_mode == "debiased_policy":
            if bias is None:
                raise ValueError("debiased_policy requires a bias tensor")
            policy_logits = raw_logits.clone()
            policy_logits[:, : self.memory_size] = (
                policy_logits[:, : self.memory_size] - bias
            )
            policy_logits = self._mask_memory_logits(policy_logits, chosen)
        return policy_logits / temperature

    def _sample_logits(
        self,
        raw_logits: torch.Tensor,
        chosen: torch.Tensor,
        selection_mode: str,
        temperature: float,
        top_k: Optional[int],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if selection_mode == "raw":
            return self._top_k_logits(raw_logits / temperature, top_k)
        if bias is None:
            raise ValueError(f"{selection_mode} requires a bias tensor")

        debiased_logits = raw_logits.clone()
        debiased_logits[:, : self.memory_size] = (
            debiased_logits[:, : self.memory_size] - bias
        )
        debiased_logits = self._mask_memory_logits(debiased_logits, chosen)
        candidate_logits = self._top_k_logits(debiased_logits, top_k)
        candidate_mask = torch.isfinite(candidate_logits)
        if selection_mode == "debiased_topk":
            return (raw_logits / temperature).masked_fill(~candidate_mask, -torch.inf)
        if selection_mode == "debiased_policy":
            return (debiased_logits / temperature).masked_fill(
                ~candidate_mask, -torch.inf
            )
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    @torch.inference_mode()
    def generate_memory_ids(
        self,
        query_embs: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        shot_num: int = 2,
        selection_mode: str = "raw",
        debias_query_embs: Optional[torch.Tensor] = None,
        debias_batch_size: int = 128,
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
            logits = self._mask_memory_logits(output.logits[:, -1, :], chosen)
            if selection_mode == "debiased":
                selection_mode = "debiased_policy"
            if selection_mode in {"debiased_topk", "debiased_policy"}:
                bias = self._bias_for_prefix_rows(
                    prefix=prefix,
                    memory_embedding_table=memory_embedding_table,
                    debias_query_embs=debias_query_embs,
                    debias_batch_size=debias_batch_size,
                )
                logits[:, : self.memory_size] = logits[:, : self.memory_size] - bias
                logits = self._mask_memory_logits(logits, chosen)
            elif selection_mode != "raw":
                raise ValueError(f"Unsupported selection_mode: {selection_mode}")
            next_id = logits.argmax(dim=-1, keepdim=True)
            prefix = torch.cat([prefix, next_id], dim=1)

        return prefix[:, 2:]

    @torch.no_grad()
    def sample_memory_ids(
        self,
        query_embs: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        shot_num: int = 2,
        group_size: int = 8,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        selection_mode: str = "raw",
        debias_query_embs: Optional[torch.Tensor] = None,
        debias_batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample memory trajectories and return full-policy action logprobs."""

        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        device = query_embs.device
        batch_size = query_embs.shape[0]
        flat_query_embs = query_embs.repeat_interleave(group_size, dim=0)
        flat_size = flat_query_embs.shape[0]
        prefix = torch.tensor(
            [[self.bos_token_id, self.query_token_id] for _ in range(flat_size)],
            dtype=torch.long,
            device=device,
        )
        memory_embedding_table = memory_embedding_table.to(device)
        logprob_steps = []
        entropy_steps = []

        for _step in range(shot_num):
            chosen = prefix[:, 2:]
            memory_embs = self._prefix_memory_embs(
                prefix, memory_embedding_table, flat_query_embs
            )
            output = self.forward(
                input_ids=prefix,
                query_emb=flat_query_embs,
                memory_embs=memory_embs,
                labels=None,
            )
            raw_logits = self._mask_memory_logits(output.logits[:, -1, :], chosen)
            bias = None
            if selection_mode in {"debiased_topk", "debiased_policy"}:
                bias = self._bias_for_prefix_rows(
                    prefix=prefix,
                    memory_embedding_table=memory_embedding_table,
                    debias_query_embs=debias_query_embs,
                    debias_batch_size=debias_batch_size,
                )
            policy_logits = self._selection_policy_logits(
                raw_logits=raw_logits,
                chosen=chosen,
                selection_mode=selection_mode,
                temperature=temperature,
                bias=bias,
            )
            sample_logits = self._sample_logits(
                raw_logits=raw_logits,
                chosen=chosen,
                selection_mode=selection_mode,
                temperature=temperature,
                top_k=top_k,
                bias=bias,
            )
            sample_dist = Categorical(logits=sample_logits)
            full_dist = Categorical(logits=policy_logits)
            next_id = sample_dist.sample()
            logprob_steps.append(full_dist.log_prob(next_id))
            entropy_steps.append(full_dist.entropy())
            prefix = torch.cat([prefix, next_id.unsqueeze(-1)], dim=1)

        memory_ids = prefix[:, 2:].reshape(batch_size, group_size, shot_num)
        logprobs = torch.stack(logprob_steps, dim=-1).reshape(
            batch_size, group_size, shot_num
        )
        entropies = torch.stack(entropy_steps, dim=-1).reshape(
            batch_size, group_size, shot_num
        )
        return memory_ids, logprobs, entropies

    def compute_action_logprobs(
        self,
        query_embs: torch.Tensor,
        memory_embedding_table: torch.Tensor,
        memory_ids: torch.Tensor,
        temperature: float = 1.0,
        return_max_probs: bool = False,
        selection_mode: str = "raw",
        debias_query_embs: Optional[torch.Tensor] = None,
        debias_batch_size: int = 128,
        return_probs: bool = False,
        return_values: bool = False,
    ):
        """Recompute logprobs for sampled memory ids under the current policy."""

        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if memory_ids.dim() != 3:
            raise ValueError("memory_ids must have shape [batch, group, shot]")

        device = query_embs.device
        batch_size, group_size, shot_num = memory_ids.shape
        flat_memory_ids = memory_ids.reshape(batch_size * group_size, shot_num).to(device)
        flat_query_embs = query_embs.repeat_interleave(group_size, dim=0)
        flat_size = flat_query_embs.shape[0]
        prefix = torch.tensor(
            [[self.bos_token_id, self.query_token_id] for _ in range(flat_size)],
            dtype=torch.long,
            device=device,
        )
        memory_embedding_table = memory_embedding_table.to(device)
        logprob_steps = []
        entropy_steps = []
        max_prob_steps = []
        prob_steps = []
        value_steps = []

        for step in range(shot_num):
            chosen = prefix[:, 2:]
            memory_embs = self._prefix_memory_embs(
                prefix, memory_embedding_table, flat_query_embs
            )
            output = self.forward(
                input_ids=prefix,
                query_emb=flat_query_embs,
                memory_embs=memory_embs,
                labels=None,
                output_hidden_states=return_values,
            )
            if return_values:
                last_hidden = output.hidden_states[-1][:, -1, :]
                value_steps.append(self.value_head(last_hidden).squeeze(-1))
            raw_logits = self._mask_memory_logits(output.logits[:, -1, :], chosen)
            bias = None
            if selection_mode in {"debiased_topk", "debiased_policy"}:
                bias = self._bias_for_prefix_rows(
                    prefix=prefix,
                    memory_embedding_table=memory_embedding_table,
                    debias_query_embs=debias_query_embs,
                    debias_batch_size=debias_batch_size,
                )
            logits = self._selection_policy_logits(
                raw_logits=raw_logits,
                chosen=chosen,
                selection_mode=selection_mode,
                temperature=temperature,
                bias=bias,
            )
            dist = Categorical(logits=logits)
            probs = torch.softmax(logits, dim=-1)
            if return_max_probs:
                max_prob_steps.append(probs.max(dim=-1).values)
            if return_probs:
                prob_steps.append(probs[:, : self.memory_size])
            action = flat_memory_ids[:, step]
            logprob_steps.append(dist.log_prob(action))
            entropy_steps.append(dist.entropy())
            prefix = torch.cat([prefix, action.unsqueeze(-1)], dim=1)

        logprobs = torch.stack(logprob_steps, dim=-1).reshape(
            batch_size, group_size, shot_num
        )
        entropies = torch.stack(entropy_steps, dim=-1).reshape(
            batch_size, group_size, shot_num
        )
        values = None
        if return_values:
            values = torch.stack(value_steps, dim=-1).reshape(
                batch_size, group_size, shot_num
            )
        if return_max_probs:
            max_probs = torch.stack(max_prob_steps, dim=-1).reshape(
                batch_size, group_size, shot_num
            )
            if return_probs:
                probs = torch.stack(prob_steps, dim=1).reshape(
                    batch_size, group_size, shot_num, self.memory_size
                )
                if return_values:
                    return logprobs, entropies, max_probs, probs, values
                return logprobs, entropies, max_probs, probs
            if return_values:
                return logprobs, entropies, max_probs, values
            return logprobs, entropies, max_probs
        if return_probs:
            probs = torch.stack(prob_steps, dim=1).reshape(
                batch_size, group_size, shot_num, self.memory_size
            )
            if return_values:
                return logprobs, entropies, probs, values
            return logprobs, entropies, probs
        if return_values:
            return logprobs, entropies, values
        return logprobs, entropies


def checkpoint_metadata(model: MathMemoryLeverLM, extra: Dict) -> Dict:
    metadata = {
        "memory_size": model.memory_size,
        "encoder_emb_dim": model.encoder_emb_dim,
        "n_embd": model.lm_model.config.n_embd,
        "n_head": model.lm_model.config.n_head,
        "n_layer": model.lm_model.config.n_layer,
        "max_positions": model.lm_model.config.n_positions,
        "model_backend": model.model_backend,
        "eos_token_id": model.eos_token_id,
        "bos_token_id": model.bos_token_id,
        "query_token_id": model.query_token_id,
        "vocab_size": model.vocab_size,
        "has_value_head": True,
    }
    metadata.update(extra)
    return metadata


class PointerMemoryLeverLM(nn.Module):
    """Pointer selector over a fixed candidate set of memory embeddings."""

    BOS_TOKEN = 0
    QUERY_TOKEN = 1
    MEM_TOKEN = 2
    SEL1_TOKEN = 3
    SELECTED_TOKEN = 4
    SEL2_TOKEN = 5
    TYPE_VOCAB_SIZE = 6

    def __init__(
        self,
        encoder_emb_dim: int,
        candidate_num: int = 64,
        n_embd: int = 512,
        n_head: int = 8,
        n_layer: int = 2,
        adapter_hidden_mult: int = 4,
        max_positions: int = 80,
        normalize_encoder_emb: bool = True,
        pointer_key_source: str = "contextual",
    ) -> None:
        super().__init__()
        if pointer_key_source not in {"contextual", "semantic"}:
            raise ValueError(
                "pointer_key_source must be either 'contextual' or 'semantic'"
            )
        self.encoder_emb_dim = encoder_emb_dim
        self.candidate_num = candidate_num
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_positions = max_positions
        self.normalize_encoder_emb = normalize_encoder_emb
        self.pointer_key_source = pointer_key_source

        self.type_embedding = nn.Embedding(self.TYPE_VOCAB_SIZE, n_embd)
        self.position_embedding = nn.Embedding(max_positions, n_embd)
        self.adapter = nn.Sequential(
            nn.Linear(encoder_emb_dim, n_embd * adapter_hidden_mult),
            nn.ReLU(),
            nn.Linear(n_embd * adapter_hidden_mult, n_embd),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(n_embd)
        self.query_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.key_proj = nn.Linear(n_embd, n_embd, bias=False)

    def _adapt(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize_encoder_emb:
            embeddings = F.normalize(embeddings.float(), dim=-1)
        return self.adapter(embeddings)

    @staticmethod
    def _attention_mask(seq_len: int, prefix_len: int, device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        if prefix_len < seq_len:
            mask[:prefix_len, prefix_len:] = True
            for row in range(prefix_len, seq_len):
                mask[row, row + 1 :] = True
        return mask

    def _type_pos_embed(self, token_type: int, position: int, batch_size: int, device):
        type_ids = torch.full(
            (batch_size,), token_type, dtype=torch.long, device=device
        )
        pos_ids = torch.full((batch_size,), position, dtype=torch.long, device=device)
        return self.type_embedding(type_ids) + self.position_embedding(pos_ids)

    def forward(
        self,
        query_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
    ):
        if candidate_embs.dim() != 3:
            raise ValueError("candidate_embs must have shape [batch, candidate, dim]")
        batch_size, candidate_num, _emb_dim = candidate_embs.shape
        if candidate_num != self.candidate_num:
            raise ValueError(
                f"Expected candidate_num={self.candidate_num}, got {candidate_num}"
            )

        device = query_embs.device
        prefix_len = 2 + candidate_num
        seq_len = prefix_len + (3 if selected_indices is not None else 1)
        if seq_len > self.max_positions:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_positions={self.max_positions}"
            )

        inputs = torch.zeros(batch_size, seq_len, self.n_embd, device=device)
        inputs[:, 0] = self._type_pos_embed(self.BOS_TOKEN, 0, batch_size, device)
        inputs[:, 1] = (
            self._type_pos_embed(self.QUERY_TOKEN, 1, batch_size, device)
            + self._adapt(query_embs)
        )

        adapted_candidates = self._adapt(
            candidate_embs.reshape(batch_size * candidate_num, -1)
        ).reshape(batch_size, candidate_num, self.n_embd)
        mem_type = self.type_embedding(
            torch.full(
                (batch_size, candidate_num),
                self.MEM_TOKEN,
                dtype=torch.long,
                device=device,
            )
        )
        mem_positions = self.position_embedding(
            torch.arange(2, 2 + candidate_num, device=device)
        ).unsqueeze(0)
        inputs[:, 2 : 2 + candidate_num] = adapted_candidates + mem_type + mem_positions

        sel1_pos = prefix_len
        inputs[:, sel1_pos] = self._type_pos_embed(
            self.SEL1_TOKEN, sel1_pos, batch_size, device
        )
        sel2_pos = None
        if selected_indices is not None:
            selected_indices = selected_indices.to(device=device, dtype=torch.long)
            selected = adapted_candidates[
                torch.arange(batch_size, device=device), selected_indices
            ]
            selected_pos = prefix_len + 1
            sel2_pos = prefix_len + 2
            inputs[:, selected_pos] = (
                self._type_pos_embed(
                    self.SELECTED_TOKEN, selected_pos, batch_size, device
                )
                + selected
            )
            inputs[:, sel2_pos] = self._type_pos_embed(
                self.SEL2_TOKEN, sel2_pos, batch_size, device
            )

        attn_mask = self._attention_mask(seq_len, prefix_len, device)
        hidden = self.transformer(inputs, mask=attn_mask)
        hidden = self.final_norm(hidden)

        if self.pointer_key_source == "semantic":
            keys = self.key_proj(adapted_candidates)
        else:
            keys = self.key_proj(hidden[:, 2 : 2 + candidate_num])
        q1 = self.query_proj(hidden[:, sel1_pos])
        logits1 = torch.einsum("bd,bcd->bc", q1, keys) / math.sqrt(self.n_embd)
        logits2 = None
        if selected_indices is not None:
            q2 = self.query_proj(hidden[:, sel2_pos])
            logits2 = torch.einsum("bd,bcd->bc", q2, keys) / math.sqrt(self.n_embd)
            logits2 = logits2.clone()
            logits2.scatter_(1, selected_indices.unsqueeze(1), -torch.inf)

        return SimpleNamespace(logits1=logits1, logits2=logits2, hidden_states=hidden)

    @torch.inference_mode()
    def generate_memory_ids(
        self,
        query_embs: torch.Tensor,
        candidate_memory_ids: torch.Tensor,
        memory_embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        candidate_memory_ids = candidate_memory_ids.to(query_embs.device)
        memory_embedding_table = memory_embedding_table.to(query_embs.device)
        candidate_embs = memory_embedding_table[candidate_memory_ids]
        first = self.forward(query_embs, candidate_embs).logits1.argmax(dim=-1)
        output = self.forward(
            query_embs=query_embs,
            candidate_embs=candidate_embs,
            selected_indices=first,
        )
        second = output.logits2.argmax(dim=-1)
        local = torch.stack([first, second], dim=-1)
        return candidate_memory_ids.gather(1, local)


def pointer_checkpoint_metadata(model: PointerMemoryLeverLM, extra: Dict) -> Dict:
    metadata = {
        "model_type": "pointer_lever_lm",
        "encoder_emb_dim": model.encoder_emb_dim,
        "candidate_num": model.candidate_num,
        "n_embd": model.n_embd,
        "n_head": model.n_head,
        "n_layer": model.n_layer,
        "max_positions": model.max_positions,
        "type_vocab_size": model.TYPE_VOCAB_SIZE,
        "normalize_encoder_emb": model.normalize_encoder_emb,
        "pointer_key_source": model.pointer_key_source,
    }
    metadata.update(extra)
    return metadata
