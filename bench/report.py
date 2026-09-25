"""Render the README results sections from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def splice(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.index(start), text.index(end)
    return text[: i + len(start)] + "\n" + body.strip() + "\n" + text[j:]


def serving(res: dict[str, Any], workload: str) -> str:
    rows = [s for s in res["serving"] if s["workload"] == workload]
    out = [
        "| concurrency | mini-vllm tok/s | HF generate tok/s | mini-vllm / HF | mini-vllm p50 / p99 s | HF p50 / p99 s |",
        "|---|---|---|---|---|---|",
    ]
    for s in rows:
        m, h = s["minivllm"], s["hf_generate"]
        if h.get("oom"):
            out.append(
                f"| {s['concurrency']} | {m['tokens_per_s']:,.0f} | OOM | – | {m['p50_latency_s']:.1f} / {m['p99_latency_s']:.1f} | – |"
            )
            continue
        out.append(
            f"| {s['concurrency']} | {m['tokens_per_s']:,.0f} | {h['tokens_per_s']:,.0f} | "
            f"{m['tokens_per_s'] / h['tokens_per_s']:.2f}× | {m['p50_latency_s']:.1f} / {m['p99_latency_s']:.1f} | "
            f"{h['p50_latency_s']:.1f} / {h['p99_latency_s']:.1f} |"
        )
    return "\n".join(out)


def main() -> None:
    res = json.loads((ROOT / "results" / "t4.json").read_text())
    tmpl = json.loads((ROOT / "results" / "template_check.json").read_text())
    e, c, eq = res["env"], res["config"], res["equality"]
    mism = [x["first_diff"] for x in eq["cases"] if not x["match"]]
    ml = c.get("mixed_lengths")
    if ml is None:  # older runs didn't record it; the lengths come from a fixed seed, so rebuild them
        import random
        import statistics

        rng = random.Random(0)
        lengths = [rng.randint(16, 2 * c["out_len"] - 16) for _ in range(c["requests"])]
        ml = {"min": min(lengths), "max": max(lengths), "mean": statistics.mean(lengths)}
    body = f"""
{e["gpu"]}, torch {e["torch"]}, transformers {e["transformers"]}, CUDA {e["cuda"]}. Model `{c["model"]}` in fp16.
{c["requests"]} requests (prompts from `bench/prompts.json`, mean {c["prompt_tokens_mean"]:.0f} tokens after the chat template),
EOS ignored so both systems generate the same tokens. KV cache: {c["kv_cache_gib"]} GiB = {c["kv_blocks"]:,} blocks of {c["block_size"]}.
Closed loop: C requests in flight; HF runs consecutive groups of C as left-padded static batches.

**Mixed output lengths** ({ml["min"]}–{ml["max"]} tokens per request, mean {ml["mean"]:.0f}; the same lengths for both).
A static batch runs until its longest request finishes; continuous batching starts the next request as soon as one ends.

{serving(res, "mixed")}

**Fixed output length** ({c["out_len"]} tokens for every request, the best case for static batching).

{serving(res, "fixed")}

Throughput counts only the tokens each request asked for. Latency runs from when a request is started until its last
token, for both systems.
"""
    eq_body = f"""
- **Chat template and tokenizer:** identical to HF `apply_chat_template` + tokenizer on all {tmpl["prompts"]} benchmark
  prompts (text and token ids; transformers {tmpl["transformers"]}, `bench/check_template.py`). The Kaggle run's own
  template check reported 0 %. The check itself was wrong, not the tokenizer: it compared against the raw return value
  of HF's tokenizer call. The check in `bench/bench.py` now uses `encode`, and the standalone comparison above is the
  evidence.
- **Greedy output vs HF `generate` on the T4 (fp16):** {sum(x["match"] for x in eq["cases"])}/{eq["prompts"]} prompts
  identical for {eq["max_new_tokens"]} tokens. The other {len(mism)} diverge after {min(mism)}–{max(mism)} matching tokens.
  In fp16 two nearly equal logits can come out in either order depending on the kernel, and after that the greedy paths
  split. On CPU in fp32 (CI), logits and greedy outputs are identical to HF's.
- **Tests on the T4:** {next(ln.strip() for ln in reversed(open(ROOT / "results" / "t4-tests.txt").read().splitlines()) if " passed" in ln)} (`results/t4-tests.txt`).
"""
    t = (ROOT / "README.md").read_text()
    t = splice(t, "perf", body)
    t = splice(t, "correctness", eq_body)
    (ROOT / "README.md").write_text(t)


if __name__ == "__main__":
    main()
