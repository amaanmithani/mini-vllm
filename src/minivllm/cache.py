"""Paged KV cache: one preallocated pool of fixed-size blocks per layer, handed out by a free list."""

from __future__ import annotations

import torch

from minivllm.config import ModelConfig


class OutOfBlocks(Exception):
    """The pool has fewer free blocks than requested."""


class BlockAllocator:
    """Free-list allocator over block ids 0..n-1. allocate/free are O(1) per block."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("need at least one block")
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))  # pop() hands out low ids first
        self._used = [False] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"asked for {n} blocks, {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        for b in out:
            self._used[b] = True
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if not self._used[b]:
                raise ValueError(f"block {b} freed twice")
            self._used[b] = False
            self._free.append(b)


class KVCache:
    """k[layer] and v[layer] are [num_blocks, block_size, kv_heads, head_dim]. A token at logical
    position p of a sequence lives in block table[p // block_size], row p % block_size; its flat slot
    is block * block_size + row."""

    def __init__(
        self, cfg: ModelConfig, num_blocks: int, block_size: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_hidden_layers)]
        self.allocator = BlockAllocator(num_blocks)

    @staticmethod
    def bytes_per_block(cfg: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        elt = torch.tensor([], dtype=dtype).element_size()
        return 2 * cfg.num_hidden_layers * block_size * cfg.num_key_value_heads * cfg.head_dim * elt

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [T, kv_heads, head_dim] written at flat slots [T]."""
        self.k[layer].view(-1, *self.k[layer].shape[2:])[slots] = k
        self.v[layer].view(-1, *self.v[layer].shape[2:])[slots] = v

    def gather(self, layer: int, block_tables: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """block_tables: [B, max_blocks] -> k, v: [B, max_blocks * block_size, kv_heads, head_dim]."""
        b, nb = block_tables.shape
        k = self.k[layer][block_tables].reshape(b, nb * self.block_size, *self.k[layer].shape[2:])
        v = self.v[layer][block_tables].reshape(b, nb * self.block_size, *self.v[layer].shape[2:])
        return k, v
