# mini-vllm spec

## Goal
A small, readable LLM inference engine with the two ideas that make vLLM fast: a paged KV cache and
continuous batching. It serves Qwen2.5-0.5B-Instruct behind an OpenAI-compatible API, and it is measured
against Hugging Face `generate` on a T4.

## In scope
- **Model:** Qwen2 architecture written from scratch in PyTorch (RMSNorm, rotary embeddings, grouped-query
  attention with qkv bias, SwiGLU MLP, tied embeddings). Weights are loaded from the HF safetensors checkpoint.
- **Paged KV cache:** fixed-size blocks (default 16 tokens) in one preallocated pool per layer; a free-list
  allocator (O(1) allocate and free); per-sequence block tables; slot mapping for writes.
- **Scheduler:** continuous batching. Each step is either a prefill of waiting sequences (under a token
  budget) or one decode token for every running sequence. When the pool runs out, the most recently admitted
  sequence is preempted: its blocks are freed and it is recomputed later.
- **Sampling:** greedy, temperature and top-p, per request; stop on EOS, stop strings or max tokens.
- **Server:** FastAPI, `/v1/completions` and `/v1/chat/completions` (streaming and not), `/v1/models`, plus
  a ModelMux provider config so the gateway can route to it.

## Out of scope
Tensor parallelism, prefix caching, speculative decoding, quantisation, CUDA graphs, chunked prefill,
LoRA, and models other than the Qwen2 family.

## Success metrics
- **Correctness:** logits match HF `Qwen2ForCausalLM` on the same weights, paged vs contiguous, batched vs
  alone (tiny random model on CPU in CI; the real model on the T4). Greedy output equality vs HF `generate` is
  reported as a match rate, because fp16 batched kernels can legitimately flip near-ties.
- **Performance on a T4:** output tokens/s and p50/p99 request latency at concurrency 1–64, against HF
  `generate` with static padded batches, on a fixed prompt set.
- Every README number is produced by a committed script from committed JSON. Coverage >= 85 %.
