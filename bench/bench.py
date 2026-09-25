"""T4 benchmark: mini-vllm vs Hugging Face `generate` on Qwen2.5-0.5B-Instruct (fp16).

1. Equality: greedy outputs of mini-vllm vs HF generate (one request at a time, EOS allowed) on the
   first --eq-prompts prompts: exact-match rate and the position of the first differing token.
2. Serving: closed loop with C requests in flight (C = 1..64) over --requests prompts, EOS ignored
   so both systems do the same work, in two workloads:
   - fixed: every request generates --out-len tokens (the best case for static batching);
   - mixed: each request has its own output length, drawn once (seed 0) uniformly from 16..2*out_len-16,
     the same lengths for both systems. HF static batches run until their longest member finishes and
     then drop the extra tokens; that waste is exactly what continuous batching avoids.
   mini-vllm keeps C requests in the engine; HF runs consecutive groups of C, left-padded. Reports
   useful output tokens/s and p50/p99 request latency (from when a request is started to its last token).

HF's generation_config for Qwen2.5-Instruct sets repetition_penalty 1.05 (and sampling defaults);
all of those are neutralised here so HF runs plain greedy decoding, like mini-vllm.
Writes results/t4.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from minivllm import Engine, SamplingParams
from minivllm.cache import KVCache
from minivllm.loader import load

ROOT = Path(__file__).resolve().parents[1]


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


GREEDY_HF = {"do_sample": False, "repetition_penalty": 1.0, "temperature": None, "top_p": None, "top_k": None}


def minivllm_closed_loop(
    engine: Engine, prompts: list[list[int]], c: int, lengths: list[int]
) -> dict[str, Any]:
    todo = deque(zip(prompts, lengths, strict=True))
    inflight: dict[int, float] = {}
    lat: list[float] = []
    tokens = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while todo or engine.has_work():
        while todo and len(inflight) < c:
            p, n = todo.popleft()
            s = engine.add_request(p, SamplingParams(max_tokens=n, ignore_eos=True))
            inflight[s.id] = time.perf_counter()
        for o in engine.step():
            tokens += 1
            if o.finished:
                lat.append(time.perf_counter() - inflight.pop(o.seq.id))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "output_tokens": tokens,
        "seconds": wall,
        "tokens_per_s": tokens / wall,
        "p50_latency_s": pct(lat, 0.5),
        "p99_latency_s": pct(lat, 0.99),
        "preemptions": engine.stats.preemptions,
    }


def hf_static(hf: Any, tok: Any, prompts: list[list[int]], c: int, lengths: list[int]) -> dict[str, Any]:
    lat: list[float] = []
    tokens = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(prompts), c):
        group = prompts[i : i + c]
        want = lengths[i : i + c]
        out_len = max(want)  # a static batch runs until its longest request is done
        start = time.perf_counter()
        width = max(len(p) for p in group)
        ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in group], device="cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in group], device="cuda")
        hf.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=out_len,
            min_new_tokens=out_len,
            pad_token_id=tok.pad_token_id,
            **GREEDY_HF,
        )
        torch.cuda.synchronize()
        tokens += sum(want)  # useful tokens: what each request asked for
        # Every request in a static batch is returned when the batch finishes.
        lat += [time.perf_counter() - start] * len(group)
    wall = time.perf_counter() - t0
    return {
        "output_tokens": tokens,
        "seconds": wall,
        "tokens_per_s": tokens / wall,
        "p50_latency_s": pct(lat, 0.5),
        "p99_latency_s": pct(lat, 0.99),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--requests", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--eq-prompts", type=int, default=64)
    ap.add_argument("--eq-len", type=int, default=64)
    ap.add_argument("--kv-gib", type=float, default=6.0)
    ap.add_argument("--out", default="results/t4.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded = load(a.model, dtype=torch.float16, device="cuda")
    hf_tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    texts = json.loads((ROOT / "bench" / "prompts.json").read_text())
    prompts = [loaded.tokenizer.chat([{"role": "user", "content": t}]) for t in texts]
    # Our tokenizer + chat template must produce exactly what HF's does.
    hf_ids = [
        hf_tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True)
        for t in texts
    ]
    template_match = sum(p == list(h) for p, h in zip(prompts, hf_ids, strict=True)) / len(prompts)

    per_block = KVCache.bytes_per_block(loaded.model.cfg, 16, torch.float16)
    blocks = int(a.kv_gib * 2**30 // per_block)

    def engine() -> Engine:
        return Engine(loaded.model, blocks, 16, max_batch_tokens=8192, detokenize=loaded.tokenizer.decode)

    # 1) Greedy equality, one request at a time, EOS allowed.
    eq = []
    for p in prompts[: a.eq_prompts]:
        ours = engine().generate([p], SamplingParams(max_tokens=a.eq_len))[0]
        ref = hf.generate(
            torch.tensor([p], device="cuda"),
            max_new_tokens=a.eq_len,
            pad_token_id=hf_tok.pad_token_id,
            **GREEDY_HF,
        )[0, len(p) :].tolist()
        first = next((i for i, (x, y) in enumerate(zip(ours, ref, strict=False)) if x != y), None)
        if first is None and len(ours) != len(ref):
            first = min(len(ours), len(ref))
        eq.append(
            {
                "match": first is None,
                "first_diff": first,
                "ours_len": len(ours),
                "hf_len": len(ref),
                "ours": loaded.tokenizer.decode(ours)[:300],
                "hf": hf_tok.decode(ref, skip_special_tokens=True)[:300],
            }
        )
        print("eq", len(eq), eq[-1]["match"], eq[-1]["first_diff"], flush=True)

    # 2) Serving throughput and latency.
    work = (prompts * ((a.requests // len(prompts)) + 1))[: a.requests]
    engine().generate(work[:4], SamplingParams(max_tokens=8, ignore_eos=True))  # warm-up
    hf_static(hf, hf_tok, work[:4], 4, [8] * 4)
    rng = random.Random(0)
    workloads = {
        "fixed": [a.out_len] * len(work),
        "mixed": [rng.randint(16, 2 * a.out_len - 16) for _ in work],
    }
    serving = []
    for wname, lengths in workloads.items():
        for c in (int(x) for x in a.concurrency.split(",")):
            mv = minivllm_closed_loop(engine(), work, c, lengths)
            torch.cuda.empty_cache()
            try:
                h = hf_static(hf, hf_tok, work, c, lengths)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                h = {"oom": True}
            serving.append({"workload": wname, "concurrency": c, "minivllm": mv, "hf_generate": h})
            print(json.dumps(serving[-1]), flush=True)

    props = torch.cuda.get_device_properties(0)
    import transformers

    out = {
        "env": {
            "gpu": props.name,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
        },
        "config": {
            "model": a.model,
            "dtype": "float16",
            "requests": a.requests,
            "out_len": a.out_len,
            "kv_cache_gib": a.kv_gib,
            "kv_blocks": blocks,
            "block_size": 16,
            "prompt_tokens_mean": statistics.mean(len(p) for p in work),
        },
        "chat_template_match_rate": template_match,
        "equality": {
            "prompts": len(eq),
            "max_new_tokens": a.eq_len,
            "exact_match_rate": sum(e["match"] for e in eq) / len(eq),
            "cases": eq,
        },
        "serving": serving,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
