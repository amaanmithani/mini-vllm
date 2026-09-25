import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from minivllm import Engine
from minivllm.loader import chat_prompt
from minivllm.server import EngineWorker, create_app, main


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 90 + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + i % 26) for i in ids)

    def chat(self, messages: list[dict[str, str]]) -> list[int]:
        return self.encode("".join(m["content"] for m in messages))


@pytest.fixture
def client(pair: tuple[Any, Any]) -> Any:
    _, ours = pair
    worker = EngineWorker(Engine(ours, num_blocks=64, block_size=4))
    with TestClient(create_app(worker, FakeTokenizer(), "tiny")) as c:
        yield c
    worker.close()


def test_completion_greedy_and_usage(client: Any) -> None:
    body = {"prompt": "hello", "max_tokens": 6, "temperature": 0}
    a = client.post("/v1/completions", json=body).json()
    b = client.post("/v1/completions", json=body).json()
    assert a["choices"][0]["text"] == b["choices"][0]["text"]
    assert a["usage"] == {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
    assert a["choices"][0]["finish_reason"] == "length" and a["object"] == "text_completion"


def test_chat_and_stream_agree(client: Any) -> None:
    body = {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 8, "temperature": 0}
    full = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"]
    text, done, reasons = "", False, []
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        for line in r.iter_lines():
            if not line:
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                done = True
                continue
            ch = json.loads(payload)["choices"][0]
            text += ch["delta"].get("content", "")
            reasons.append(ch["finish_reason"])
    assert done and text == full and reasons[-1] == "length"


def test_stop_string_is_clipped(client: Any) -> None:
    body = {"prompt": "abc", "max_tokens": 10, "temperature": 0}
    full = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    stop = full[3:5]
    clipped = client.post("/v1/completions", json={**body, "stop": stop}).json()["choices"][0]["text"]
    assert stop not in clipped and full.startswith(clipped)


def test_concurrent_requests_share_the_engine(client: Any) -> None:
    import concurrent.futures

    body = {"prompt": "concurrency", "max_tokens": 5, "temperature": 0}
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: client.post("/v1/completions", json=body).json(), range(8)))
    assert len({o["choices"][0]["text"] for o in outs}) == 1
    h = client.get("/health").json()
    assert h["free_blocks"] == h["total_blocks"] and h["running"] == 0


def test_validation(client: Any) -> None:
    assert client.post("/v1/completions", json={"prompt": ""}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "x", "top_p": 5}).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user"}]}).status_code == 400
    too_long = client.post("/v1/completions", json={"prompt": "x" * 300, "max_tokens": 10})
    assert too_long.status_code == 400 and "needs" in too_long.json()["detail"]
    with client.stream("POST", "/v1/completions", json={"prompt": "x" * 300, "stream": True}) as r:
        assert "error" in "".join(r.iter_lines())
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"


def test_chat_template() -> None:
    p = chat_prompt([{"role": "user", "content": "hi"}])
    assert p.startswith("<|im_start|>system\nYou are Qwen") and p.endswith("<|im_start|>assistant\n")
    assert "<|im_start|>system\nbe brief" in chat_prompt([{"role": "system", "content": "be brief"}])


def test_main_builds_the_app(monkeypatch: pytest.MonkeyPatch, pair: tuple[Any, Any]) -> None:
    from minivllm import loader

    _, ours = pair
    monkeypatch.setattr(loader, "load", lambda *a, **k: loader.Loaded(ours, FakeTokenizer(), None))  # type: ignore[arg-type]
    seen: dict[str, Any] = {}
    main(
        ["--device", "cpu", "--dtype", "float32", "--kv-cache-gib", "0.001"],
        serve=lambda app, **kw: seen.update(kw, app=app),
    )
    assert seen["port"] == 8000 and seen["app"].title == "mini-vllm"
