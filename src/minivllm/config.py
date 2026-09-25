"""Model configuration (the subset of a HF Qwen2 config.json this engine uses)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 32768
    eos_token_ids: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        eos = d.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            rope_theta=float(d.get("rope_theta", 1_000_000.0)),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", True)),
            max_position_embeddings=int(d.get("max_position_embeddings", 32768)),
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
