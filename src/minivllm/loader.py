"""Loading a Qwen2 checkpoint (local directory or HF hub id) and its tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from minivllm.config import ModelConfig
from minivllm.model import Qwen2

FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "*.safetensors"]
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class Tokenizer:
    """HF `tokenizers` wrapper with Qwen's ChatML format."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        self.tok = HFTokenizer.from_file(str(path / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self.tok.decode(ids, skip_special_tokens=True))

    def chat(self, messages: list[dict[str, str]]) -> list[int]:
        return self.encode(chat_prompt(messages))


def chat_prompt(messages: list[dict[str, str]]) -> str:
    """Qwen2.5's chat template (ChatML), including its default system message."""
    parts = []
    if not messages or messages[0].get("role") != "system":
        parts.append(f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n")
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@dataclass
class Loaded:
    model: Qwen2
    tokenizer: Tokenizer
    path: Path


def resolve(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=FILES))


def load(model: str, dtype: torch.dtype = torch.float16, device: str | torch.device = "cuda") -> Loaded:
    from safetensors.torch import load_file

    path = resolve(model)
    raw = json.loads((path / "config.json").read_text())
    gen = path / "generation_config.json"
    if gen.exists():
        eos = json.loads(gen.read_text()).get("eos_token_id")
        if eos is not None:
            raw["eos_token_id"] = eos
    cfg = ModelConfig.from_dict(raw)
    with torch.device("meta"):
        m = Qwen2(cfg)
    m = m.to_empty(device=device).to(dtype)
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        sd.update(load_file(str(f), device=str(device)))
    m.load_hf_state_dict(sd)
    # Buffers were created on the meta device; rebuild the rotary frequencies for real.
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim)
    )
    m.rotary.inv_freq = inv_freq
    if cfg.tie_word_embeddings:
        m.lm_head.weight = m.embed_tokens.weight
    return Loaded(m.eval(), Tokenizer(path), path)
