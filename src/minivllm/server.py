"""OpenAI-compatible HTTP server.

One background thread owns the engine and runs `step()` while there is work; HTTP handlers submit
requests through a thread-safe inbox and receive tokens through per-request asyncio queues, so
concurrent requests are batched together by the engine (that is the point of continuous batching).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from minivllm.engine import Engine, Sequence
from minivllm.sampling import SamplingParams


@dataclass
class _Pending:
    prompt_ids: list[int]
    params: SamplingParams
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[tuple[int, str | None] | None]
    seq: Sequence | None = None
    cancelled: bool = False


class EngineWorker:
    """Runs the engine in a thread; `submit` returns an async iterator of (token, finish_reason)."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._inbox: list[_Pending] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._by_seq: dict[int, _Pending] = {}
        self.thread = threading.Thread(target=self._loop, daemon=True, name="minivllm-engine")
        self.thread.start()

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                inbox, self._inbox = self._inbox, []
            for p in inbox:
                if p.cancelled:
                    continue
                try:
                    p.seq = self.engine.add_request(p.prompt_ids, p.params)
                except ValueError as e:
                    p.loop.call_soon_threadsafe(p.queue.put_nowait, (-1, f"error: {e}"))
                    p.loop.call_soon_threadsafe(p.queue.put_nowait, None)
                    continue
                self._by_seq[p.seq.id] = p
            for sid, p in list(self._by_seq.items()):
                if p.cancelled and p.seq is not None:
                    self.engine.abort(p.seq)
                    del self._by_seq[sid]
            if not self.engine.has_work():
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            for out in self.engine.step():
                pend = self._by_seq.get(out.seq.id)
                if pend is None:
                    continue
                reason = out.seq.finish_reason if out.finished else None
                pend.loop.call_soon_threadsafe(pend.queue.put_nowait, (out.new_token, reason))
                if out.finished:
                    pend.loop.call_soon_threadsafe(pend.queue.put_nowait, None)
                    del self._by_seq[out.seq.id]

    async def submit(
        self, prompt_ids: list[int], params: SamplingParams
    ) -> AsyncIterator[tuple[int, str | None]]:
        loop = asyncio.get_running_loop()
        p = _Pending(prompt_ids, params, loop, asyncio.Queue())
        with self._lock:
            self._inbox.append(p)
        self._wake.set()
        try:
            while (item := await p.queue.get()) is not None:
                if item[0] == -1:
                    raise ValueError(item[1])
                yield item
        finally:
            p.cancelled = True  # client went away (or finished): let the engine free the blocks
            self._wake.set()

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        self.thread.join(timeout=5)


def create_app(worker: EngineWorker, tokenizer: Any, model_name: str) -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="mini-vllm")

    def params_from(body: dict[str, Any]) -> SamplingParams:
        stop = body.get("stop") or []
        try:
            return SamplingParams(
                max_tokens=int(body.get("max_tokens") or 64),
                temperature=float(body.get("temperature", 1.0)),  # OpenAI default
                top_p=float(body.get("top_p", 1.0)),
                stop=[stop] if isinstance(stop, str) else list(stop),
                seed=body.get("seed"),
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e)) from e

    def usage(prompt: list[int], n: int) -> dict[str, int]:
        return {"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}

    def clip(text: str, stop: list[str]) -> str:
        for s in stop:
            i = text.find(s)
            if i >= 0:
                text = text[:i]
        return text

    async def run(prompt_ids: list[int], params: SamplingParams, stream: bool, chat: bool) -> Any:
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        def choice(text: str, reason: str | None, delta: bool) -> dict[str, Any]:
            if chat:
                key = "delta" if delta else "message"
                body: dict[str, Any] = (
                    {"role": "assistant", "content": text} if not delta else {"content": text}
                )
                return {"index": 0, key: body, "finish_reason": reason}
            return {"index": 0, "text": text, "finish_reason": reason}

        if not stream:
            ids: list[int] = []
            reason = None
            try:
                async for tok, r in worker.submit(prompt_ids, params):
                    ids.append(tok)
                    reason = r
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            text = clip(tokenizer.decode(ids), params.stop)
            return JSONResponse(
                {"id": rid, "object": obj, "created": created, "model": model_name,
                 "choices": [choice(text, reason, False)], "usage": usage(prompt_ids, len(ids))}
            )  # fmt: skip

        async def events() -> AsyncIterator[str]:
            ids: list[int] = []
            sent = ""
            try:
                async for tok, r in worker.submit(prompt_ids, params):
                    ids.append(tok)
                    text = clip(tokenizer.decode(ids), params.stop)
                    # Hold back a trailing replacement char: a multi-byte character may be incomplete.
                    stable = text[:-1] if text.endswith("�") and r is None else text
                    delta, sent = stable[len(sent) :], stable
                    if delta or r:
                        chunk = {"id": rid, "object": f"{obj}.chunk" if chat else obj, "created": created,
                                 "model": model_name, "choices": [choice(delta, r, True)]}  # fmt: skip
                        yield f"data: {json.dumps(chunk)}\n\n"
            except ValueError as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "mini-vllm"}]}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        e = worker.engine
        return {
            "running": len(e.running),
            "waiting": len(e.waiting),
            "free_blocks": e.cache.allocator.num_free,
            "total_blocks": e.cache.num_blocks,
            "preemptions": e.stats.preemptions,
        }

    @app.post("/v1/completions")
    async def completions(body: dict[str, Any]) -> Any:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(400, "prompt must be a non-empty string")
        return await run(tokenizer.encode(prompt), params_from(body), bool(body.get("stream")), chat=False)

    @app.post("/v1/chat/completions")
    async def chat(body: dict[str, Any]) -> Any:
        msgs = body.get("messages")
        if (
            not isinstance(msgs, list)
            or not msgs
            or not all(
                isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("role") for m in msgs
            )
        ):
            raise HTTPException(400, "messages must be a non-empty list of {role, content}")
        return await run(tokenizer.chat(msgs), params_from(body), bool(body.get("stream")), chat=True)

    return app


def main(argv: list[str] | None = None, serve: Callable[..., None] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="minivllm", description="Serve a Qwen2 model with an OpenAI-compatible API."
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--kv-cache-gib", type=float, default=4.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-tokens", type=int, default=4096)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    import torch

    from minivllm.cache import KVCache
    from minivllm.loader import load

    dtype = getattr(torch, a.dtype)
    loaded = load(a.model, dtype=dtype, device=a.device)
    per_block = KVCache.bytes_per_block(loaded.model.cfg, a.block_size, dtype)
    blocks = max(1, int(a.kv_cache_gib * 2**30 // per_block))
    engine = Engine(
        loaded.model, blocks, a.block_size, a.max_batch_tokens, detokenize=loaded.tokenizer.decode
    )
    app = create_app(EngineWorker(engine), loaded.tokenizer, a.model)
    if serve is None:
        import uvicorn

        serve = uvicorn.run
    serve(app, host=a.host, port=a.port, log_level="warning")
