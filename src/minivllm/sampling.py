"""Per-request sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None) -> int:
    """logits: [vocab] float32."""
    if params.temperature == 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_p, idx = torch.sort(probs, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < params.top_p  # always keeps the top token
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        choice = torch.multinomial(sorted_p / sorted_p.sum(), 1, generator=generator)
        return int(idx[choice].item())
    return int(torch.multinomial(probs, 1, generator=generator).item())
