"""Qwen2 in plain PyTorch, with attention that reads and writes the paged KV cache.

A forward pass takes a flat batch of tokens described by `Batch`:
- prefill: several sequences' prompts concatenated; each attends causally within itself;
- decode: one new token per sequence, attending to that sequence's cached tokens plus itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from minivllm.cache import KVCache
from minivllm.config import ModelConfig


@dataclass
class Batch:
    input_ids: torch.Tensor  # [T]
    positions: torch.Tensor  # [T]
    slots: torch.Tensor  # [T] flat cache slot for each token's K/V
    is_prefill: bool
    seq_lens: list[int]  # prefill: tokens per sequence (sums to T); decode: context length incl. new token
    block_tables: torch.Tensor | None = None  # decode: [B, max_blocks]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [T, heads, head_dim]; rotate-half convention (as in HF Llama/Qwen2)."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # type: ignore[index]
        emb = torch.cat([freqs, freqs], dim=-1)
        # Same precision as HF: cos/sin computed in fp32, applied in the activation dtype.
        cos, sin = emb.cos().to(x.dtype)[:, None, :], emb.sin().to(x.dtype)[:, None, :]
        half = x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.h, self.kvh, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.d, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kvh * self.d, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kvh * self.d, bias=True)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False)
        self.rotary = Rotary(self.d, cfg.rope_theta)

    def forward(self, x: torch.Tensor, batch: Batch, cache: KVCache) -> torch.Tensor:
        t = x.shape[0]
        q = self.rotary(self.q_proj(x).view(t, self.h, self.d), batch.positions)
        k = self.rotary(self.k_proj(x).view(t, self.kvh, self.d), batch.positions)
        v = self.v_proj(x).view(t, self.kvh, self.d)
        cache.write(self.layer, batch.slots, k, v)
        out = self._prefill(q, k, v, batch) if batch.is_prefill else self._decode(q, batch, cache)
        return self.o_proj(out.reshape(t, self.h * self.d))

    def _prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: Batch) -> torch.Tensor:
        # No prefix caching: a prompt only attends to itself, so the fresh K/V are all it needs.
        outs, start = [], 0
        for n in batch.seq_lens:
            qs, ks, vs = (t[start : start + n].transpose(0, 1) for t in (q, k, v))  # [heads, n, d]
            outs.append(
                F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True).transpose(0, 1)
            )
            start += n
        return torch.cat(outs, dim=0)

    def _decode(self, q: torch.Tensor, batch: Batch, cache: KVCache) -> torch.Tensor:
        assert batch.block_tables is not None
        k, v = cache.gather(self.layer, batch.block_tables)  # [B, L, kvh, d]
        lens = torch.tensor(batch.seq_lens, device=q.device)
        valid = torch.arange(k.shape[1], device=q.device)[None, :] < lens[:, None]  # [B, L]
        out = F.scaled_dot_product_attention(
            q[:, :, None, :],  # [B, h, 1, d]
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=valid[:, None, None, :],
            enable_gqa=True,
        )
        return out[:, :, 0, :]


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, batch: Batch, cache: KVCache) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), batch, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @torch.inference_mode()
    def forward(self, batch: Batch, cache: KVCache, last_only: bool = True) -> torch.Tensor:
        """Returns logits (float32): one row per sequence (its last token) or, with
        last_only=False, one per input token."""
        x = self.embed_tokens(batch.input_ids)
        for layer in self.layers:
            x = layer(x, batch, cache)
        x = self.norm(x)
        if last_only and batch.is_prefill:
            ends = torch.tensor(batch.seq_lens, device=x.device).cumsum(0) - 1
            x = x[ends]
        return self.lm_head(x).float()

    def load_hf_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load HF Qwen2ForCausalLM weights (names prefixed `model.`; `lm_head.weight` optional when tied)."""
        mapped = {}
        for name, w in sd.items():
            key = name.removeprefix("model.")
            if key == "lm_head.weight" and self.cfg.tie_word_embeddings:
                continue
            mapped[key] = w
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        missing = [m for m in missing if not (m == "lm_head.weight" and self.cfg.tie_word_embeddings)]
        if missing or unexpected:
            raise ValueError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
