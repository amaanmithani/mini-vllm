from typing import Any

import pytest
import torch

from minivllm import Engine, SamplingParams
from minivllm.cache import BlockAllocator, KVCache, OutOfBlocks
from minivllm.model import Batch, Qwen2
from minivllm.sampling import sample

GREEDY = SamplingParams(max_tokens=12, ignore_eos=True)


def prompts(n: int, seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 96, (int(torch.randint(3, 40, (1,), generator=g)),), generator=g).tolist()
        for _ in range(n)
    ]


def hf_greedy(hf: Any, ids: list[int], n: int) -> list[int]:
    out = hf.generate(
        torch.tensor([ids]), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0
    )
    return out[0, len(ids) :].tolist()


def test_prefill_logits_match_hf(pair: tuple[Any, Qwen2]) -> None:
    hf, ours = pair
    ids = prompts(1)[0]
    with torch.no_grad():
        want = hf(torch.tensor([ids])).logits[0]
    cache = KVCache(ours.cfg, num_blocks=8, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    blocks = cache.allocator.allocate(-(-len(ids) // 16))
    batch = Batch(
        input_ids=torch.tensor(ids),
        positions=torch.arange(len(ids)),
        slots=torch.tensor([blocks[p // 16] * 16 + p % 16 for p in range(len(ids))]),
        is_prefill=True,
        seq_lens=[len(ids)],
    )
    got = ours(batch, cache, last_only=False)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_greedy_generation_matches_hf(pair: tuple[Any, Qwen2], block_size: int) -> None:
    hf, ours = pair
    engine = Engine(ours, num_blocks=64, block_size=block_size)
    for ids in prompts(3, seed=block_size):
        assert engine.generate([ids], GREEDY)[0] == hf_greedy(hf, ids, 12)


def test_batched_equals_alone(pair: tuple[Any, Qwen2]) -> None:
    _, ours = pair
    ps = prompts(6, seed=3)
    together = Engine(ours, num_blocks=64, block_size=4).generate(ps, GREEDY)
    alone = [Engine(ours, num_blocks=64, block_size=4).generate([p], GREEDY)[0] for p in ps]
    assert together == alone


def test_preemption_recomputes_and_matches(pair: tuple[Any, Qwen2]) -> None:
    _, ours = pair
    ps = prompts(4, seed=5)
    params = SamplingParams(max_tokens=20, ignore_eos=True)
    roomy = Engine(ours, num_blocks=128, block_size=4).generate(ps, params)
    tight = Engine(ours, num_blocks=18, block_size=4)
    out = tight.generate(ps, params)
    assert out == roomy
    assert tight.stats.preemptions > 0
    assert tight.cache.allocator.num_free == 18  # everything returned


def test_eos_and_stop_strings(pair: tuple[Any, Qwen2]) -> None:
    _, ours = pair
    ids = prompts(1, seed=7)[0]
    free_run = Engine(ours, num_blocks=64).generate([ids], SamplingParams(max_tokens=10, ignore_eos=True))[0]
    # Pretend the 4th generated token is EOS.
    eos_cfg = type(ours.cfg)(**{**ours.cfg.__dict__, "eos_token_ids": (free_run[3],)})
    ours.cfg = eos_cfg
    e = Engine(ours, num_blocks=64)
    seq = e.add_request(ids, SamplingParams(max_tokens=10))
    list(e.run())
    first = free_run.index(free_run[3])
    assert seq.output_ids == free_run[: first + 1] and seq.finish_reason == "stop"
    # Stop strings, with a toy detokenizer.
    e2 = Engine(ours, num_blocks=64, detokenize=lambda t: " ".join(map(str, t)))
    s2 = e2.add_request(ids, SamplingParams(max_tokens=10, ignore_eos=True, stop=[str(free_run[1])]))
    list(e2.run())
    assert len(s2.output_ids) <= 2 and s2.finish_reason == "stop"


def test_request_validation_and_abort(pair: tuple[Any, Qwen2]) -> None:
    _, ours = pair
    e = Engine(ours, num_blocks=4, block_size=4)
    with pytest.raises(ValueError, match="empty"):
        e.add_request([])
    with pytest.raises(ValueError, match="needs"):
        e.add_request([1] * 10, SamplingParams(max_tokens=10))
    s = e.add_request([1, 2, 3], SamplingParams(max_tokens=5))
    e.step()
    e.abort(s)
    assert s.finish_reason == "abort" and not e.has_work() and e.cache.allocator.num_free == 4
    w = e.add_request([1, 2], SamplingParams(max_tokens=2))
    e.abort(w)
    assert not e.has_work()
    assert e.step() == []


def test_long_sequence_runs_alone_over_budget(pair: tuple[Any, Qwen2]) -> None:
    _, ours = pair
    e = Engine(ours, num_blocks=64, block_size=4, max_batch_tokens=8)
    out = e.generate([list(range(1, 20)), [5, 6]], SamplingParams(max_tokens=3, ignore_eos=True))
    assert [len(o) for o in out] == [3, 3]


def test_allocator() -> None:
    a = BlockAllocator(3)
    x = a.allocate(2)
    assert x == [0, 1] and a.num_free == 1
    with pytest.raises(OutOfBlocks):
        a.allocate(2)
    a.free(x)
    with pytest.raises(ValueError, match="twice"):
        a.free([0])
    with pytest.raises(ValueError):
        BlockAllocator(0)


def test_sampling() -> None:
    logits = torch.tensor([0.0, 5.0, 1.0, 4.9])
    assert sample(logits, SamplingParams(), None) == 1
    g = torch.Generator().manual_seed(0)
    assert sample(logits, SamplingParams(temperature=1.0, top_p=0.01), g) == 1
    draws = {
        sample(logits, SamplingParams(temperature=1.0), torch.Generator().manual_seed(s)) for s in range(40)
    }
    assert draws <= {0, 1, 2, 3} and {1, 3} <= draws
    for bad in ({"max_tokens": 0}, {"temperature": -1}, {"top_p": 0}):
        with pytest.raises(ValueError):
            SamplingParams(**bad)  # type: ignore[arg-type]


def test_seeded_sampling_is_reproducible(pair: tuple[Any, Qwen2]) -> None:
    _, ours = pair
    p = SamplingParams(max_tokens=8, temperature=0.9, top_p=0.9, seed=11, ignore_eos=True)
    a = Engine(ours, num_blocks=32).generate([[1, 2, 3]], p)
    b = Engine(ours, num_blocks=32).generate([[1, 2, 3]], p)
    assert a == b
