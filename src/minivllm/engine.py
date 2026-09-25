"""Continuous batching over a paged KV cache.

Each `step()` is either a prefill (admit waiting sequences whose prompts fit the token budget and the
free blocks) or a decode (one token for every running sequence). Prefill takes priority, as in vLLM's
original scheduler, so new requests start quickly. When a decode step needs a block and none is free,
the most recently admitted running sequence is preempted by recomputation: its blocks are freed and it
goes back to the front of the waiting queue with its prompt and the tokens generated so far.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

import torch

from minivllm.cache import KVCache
from minivllm.config import ModelConfig
from minivllm.model import Batch, Qwen2
from minivllm.sampling import SamplingParams, sample


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    id: int
    prompt_ids: list[int]
    params: SamplingParams
    arrival: float = field(default_factory=time.perf_counter)
    output_ids: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    finish_reason: str | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    preemptions: int = 0
    generator: torch.Generator | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    def __len__(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class StepOutput:
    seq: Sequence
    new_token: int
    finished: bool


@dataclass
class EngineStats:
    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    preemptions: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0


class Engine:
    def __init__(
        self,
        model: Qwen2,
        num_blocks: int,
        block_size: int = 16,
        max_batch_tokens: int = 4096,
        max_running: int = 256,
        detokenize: object | None = None,
    ) -> None:
        p = next(model.parameters())
        self.model = model
        self.cfg: ModelConfig = model.cfg
        self.cache = KVCache(self.cfg, num_blocks, block_size, p.dtype, p.device)
        self.device = p.device
        self.block_size = block_size
        self.max_batch_tokens = max_batch_tokens
        self.max_running = max_running
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.stats = EngineStats()
        self._ids = itertools.count()
        # Optional token-ids -> text function, used only for stop strings.
        self.detokenize = detokenize

    # -- requests -----------------------------------------------------------------------------------
    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> Sequence:
        params = params or SamplingParams()
        if not prompt_ids:
            raise ValueError("empty prompt")
        need = self._blocks_for(len(prompt_ids) + params.max_tokens)
        if need > self.cache.num_blocks:
            raise ValueError(f"request needs {need} blocks but the cache has {self.cache.num_blocks}")
        seq = Sequence(next(self._ids), list(prompt_ids), params)
        if params.seed is not None:
            seq.generator = torch.Generator(device="cpu").manual_seed(params.seed)
        self.waiting.append(seq)
        return seq

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, seq: Sequence) -> None:
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)
            self._free(seq)
        seq.status, seq.finish_reason = Status.FINISHED, "abort"

    # -- scheduling ---------------------------------------------------------------------------------
    def _blocks_for(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)

    def _free(self, seq: Sequence) -> None:
        self.cache.allocator.free(seq.blocks)
        seq.blocks = []

    def _schedule_prefill(self) -> list[Sequence]:
        admitted: list[Sequence] = []
        budget = self.max_batch_tokens
        while self.waiting and len(self.running) + len(admitted) < self.max_running:
            seq = self.waiting[0]
            n = len(seq)  # after a preemption the generated tokens are recomputed too
            need = self._blocks_for(n)
            # Keep one block of headroom per running sequence so the next decode step
            # doesn't immediately preempt what was just admitted.
            # A single sequence longer than the budget still runs alone (else it could never finish
            # after being preempted with a long output).
            over_budget = n > budget and (admitted or self.running)
            if over_budget or need + len(self.running) > self.cache.allocator.num_free:
                break
            self.waiting.popleft()
            seq.blocks = self.cache.allocator.allocate(need)
            admitted.append(seq)
            budget -= n
        return admitted

    def _ensure_decode_slots(self) -> None:
        """Every running sequence needs room for one more token; preempt from the back if not."""
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            # This step writes position len(seq) - 1; is it past the sequence's last block?
            if (len(seq) - 1) // self.block_size >= len(seq.blocks):
                if self.cache.allocator.num_free == 0:
                    self._preempt(self.running.pop())  # newest first; may be `seq` itself
                    continue
                seq.blocks += self.cache.allocator.allocate(1)
            i += 1

    def _preempt(self, seq: Sequence) -> None:
        self._free(seq)
        seq.status = Status.WAITING
        seq.preemptions += 1
        self.stats.preemptions += 1
        self.waiting.appendleft(seq)

    # -- execution ----------------------------------------------------------------------------------
    def _slot(self, seq: Sequence, pos: int) -> int:
        return seq.blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    def step(self) -> list[StepOutput]:
        admitted = self._schedule_prefill()
        if admitted:
            out = self._run_prefill(admitted)
            self.running += admitted
            self.stats.prefill_steps += 1
        else:
            self._ensure_decode_slots()
            if not self.running:
                return []
            out = self._run_decode(self.running)
            self.stats.decode_steps += 1
        self.stats.steps += 1
        for o in out:
            if o.finished:
                self.running.remove(o.seq)
                self._free(o.seq)
        return out

    def _run_prefill(self, seqs: list[Sequence]) -> list[StepOutput]:
        ids: list[int] = []
        pos: list[int] = []
        slots: list[int] = []
        lens: list[int] = []
        for s in seqs:
            toks = s.tokens
            ids += toks
            pos += range(len(toks))
            slots += [self._slot(s, p) for p in range(len(toks))]
            lens.append(len(toks))
        batch = Batch(
            input_ids=torch.tensor(ids, device=self.device),
            positions=torch.tensor(pos, device=self.device),
            slots=torch.tensor(slots, device=self.device),
            is_prefill=True,
            seq_lens=lens,
        )
        self.stats.prefill_tokens += len(ids)
        logits = self.model(batch, self.cache)
        return self._emit_all(seqs, logits)

    def _run_decode(self, seqs: list[Sequence]) -> list[StepOutput]:
        max_blocks = max(len(s.blocks) for s in seqs)
        tables = torch.tensor(
            [s.blocks + [0] * (max_blocks - len(s.blocks)) for s in seqs], device=self.device
        )
        batch = Batch(
            input_ids=torch.tensor([s.tokens[-1] for s in seqs], device=self.device),
            positions=torch.tensor([len(s) - 1 for s in seqs], device=self.device),
            slots=torch.tensor([self._slot(s, len(s) - 1) for s in seqs], device=self.device),
            is_prefill=False,
            seq_lens=[len(s) for s in seqs],
            block_tables=tables,
        )
        self.stats.decode_tokens += len(seqs)
        logits = self.model(batch, self.cache)
        return self._emit_all(seqs, logits)

    def _emit_all(self, seqs: list[Sequence], logits: torch.Tensor) -> list[StepOutput]:
        """Greedy sequences are decided on the device in one argmax, so only token ids cross to the
        host; sequences that sample copy just their own row."""
        greedy = torch.argmax(logits, dim=-1).tolist()
        out = []
        for i, s in enumerate(seqs):
            tok = greedy[i] if s.params.temperature == 0 else sample(logits[i].cpu(), s.params, s.generator)
            out.append(self._emit(s, tok))
        return out

    def _emit(self, seq: Sequence, tok: int) -> StepOutput:
        seq.status = Status.RUNNING
        seq.output_ids.append(tok)
        now = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = now
        reason = None
        if not seq.params.ignore_eos and tok in self.cfg.eos_token_ids:
            reason = "stop"
        elif len(seq.output_ids) >= seq.params.max_tokens:
            reason = "length"
        elif seq.params.stop and self.detokenize is not None:
            text = self.detokenize(seq.output_ids)  # type: ignore[operator]
            if any(s in text for s in seq.params.stop):
                reason = "stop"
        if reason:
            seq.status, seq.finish_reason, seq.finish_time = Status.FINISHED, reason, now
        return StepOutput(seq, tok, reason is not None)

    # -- convenience --------------------------------------------------------------------------------
    def run(self) -> Iterator[StepOutput]:
        while self.has_work():
            yield from self.step()

    def generate(self, prompts: list[list[int]], params: SamplingParams | None = None) -> list[list[int]]:
        seqs = [self.add_request(p, params) for p in prompts]
        for _ in self.run():
            pass
        return [s.output_ids for s in seqs]
