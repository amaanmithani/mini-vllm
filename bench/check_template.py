"""Our chat template + tokenizer vs HF's apply_chat_template, on every benchmark prompt.
Writes results/template_check.json."""

import json
from pathlib import Path

import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from minivllm.loader import Tokenizer, chat_prompt

ROOT = Path(__file__).resolve().parents[1]
path = Path(
    snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["tokenizer*", "vocab.json", "merges.txt"])
)
hf, ours = AutoTokenizer.from_pretrained(path), Tokenizer(path)
texts = json.loads((ROOT / "bench" / "prompts.json").read_text())
same_text = same_ids = 0
for t in texts:
    msgs = [{"role": "user", "content": t}]
    ht = hf.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ot = chat_prompt(msgs)
    same_text += ht == ot
    same_ids += hf.encode(ht, add_special_tokens=False) == ours.encode(ot)
out = {
    "transformers": transformers.__version__,
    "prompts": len(texts),
    "template_text_match_rate": same_text / len(texts),
    "token_id_match_rate": same_ids / len(texts),
}
(ROOT / "results" / "template_check.json").write_text(json.dumps(out, indent=2) + "\n")
print(out)
