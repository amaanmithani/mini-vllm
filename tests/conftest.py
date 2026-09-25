from typing import Any

import pytest
import torch

from minivllm.config import ModelConfig
from minivllm.model import Qwen2

TINY = {
    "vocab_size": 97,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "max_position_embeddings": 512,
    "eos_token_id": 96,
}


def hf_tiny(seed: int = 0) -> Any:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(**TINY, attn_implementation="eager")
    hf = Qwen2ForCausalLM(cfg).eval()
    # Non-zero biases so their handling is actually tested.
    with torch.no_grad():
        for layer in hf.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.bias.normal_(0, 0.5)
    return hf


def ours_from(hf: Any) -> Qwen2:
    m = Qwen2(ModelConfig.from_dict(TINY)).eval()
    m.load_hf_state_dict(hf.state_dict())
    return m


@pytest.fixture
def pair() -> tuple[Any, Qwen2]:
    hf = hf_tiny()
    return hf, ours_from(hf)
