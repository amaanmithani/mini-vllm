import json
from pathlib import Path
from typing import Any

import torch

from minivllm import Engine, SamplingParams
from minivllm.loader import load


def test_load_checkpoint_dir_matches_hf(tmp_path: Path, pair: tuple[Any, Any]) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    hf, _ = pair
    hf.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [95, 96]}))
    vocab = {f"w{i}": i for i in range(97)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))

    loaded = load(str(tmp_path), dtype=torch.float32, device="cpu")
    assert loaded.model.cfg.eos_token_ids == (95, 96)
    ids = loaded.tokenizer.encode("w5 w9 w12 w40")
    assert ids == [5, 9, 12, 40] and loaded.tokenizer.decode(ids).split() == ["w5", "w9", "w12", "w40"]
    assert loaded.tokenizer.chat([{"role": "user", "content": "w1"}])

    got = Engine(loaded.model, num_blocks=32, block_size=4).generate(
        [ids], SamplingParams(max_tokens=6, ignore_eos=True)
    )
    want = hf.generate(
        torch.tensor([ids]), max_new_tokens=6, min_new_tokens=6, do_sample=False, pad_token_id=0
    )
    assert got[0] == want[0, len(ids) :].tolist()
