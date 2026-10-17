"""Encoder embedding cache keyed by content hash.

Two entry phases. While a request still has rows to gather, the entry is pinned by
per-uid counts. Once the last holder is done the tensor stays as an LRU-cached
entry (bounded by ``FREETOKEN_MM_ENCODER_MB``, 1 GiB default), so a repeated image on a
later turn skips re-encode; cached entries are evicted oldest-first, holders are never
evicted."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _encoder_cap_bytes() -> int:
    raw = os.environ.get("FREETOKEN_MM_ENCODER_MB", "1024")
    try:
        return max(0, int(raw)) * 2**20
    except ValueError:
        return 1024 * 2**20


@dataclass
class _Entry:
    tensor: torch.Tensor | None = None
    # uid -> embedding rows the request still has to gather, summed over its occurrences of the image
    remaining: Dict[int, int] = field(default_factory=dict)


class EncoderCache:
    def __init__(self, storage: str = "cpu") -> None:
        assert storage in ("cpu", "cuda")
        self._storage = storage
        # insertion order = LRU order; touched entries are moved to the back
        self._entries: Dict[int, _Entry] = {}
        self._cap = _encoder_cap_bytes()

    def _touch(self, item_hash: int) -> None:
        entry = self._entries.pop(item_hash, None)
        if entry is not None:
            self._entries[item_hash] = entry

    def _evict_to_cap(self) -> None:
        used = sum(
            t.numel() * t.element_size()
            for e in self._entries.values()
            if e.tensor is not None
            for t in (e.tensor,)
        )
        for h in list(self._entries):
            if used <= self._cap:
                break
            entry = self._entries[h]
            if entry.remaining:  # held by an active request: not evictable
                continue
            if entry.tensor is not None:
                used -= entry.tensor.numel() * entry.tensor.element_size()
            del self._entries[h]

    def register(self, item_hash: int, uid: int, rows: int) -> None:
        """Claim rows of the image for uid before any of its chunks run; a repeated image adds up."""
        if rows <= 0:
            return
        entry = self._entries.setdefault(item_hash, _Entry())
        entry.remaining[uid] = entry.remaining.get(uid, 0) + rows
        self._touch(item_hash)

    def has(self, item_hash: int) -> bool:
        entry = self._entries.get(item_hash)
        return entry is not None and entry.tensor is not None

    def put(self, item_hash: int, embedding: torch.Tensor) -> None:
        entry = self._entries.get(item_hash)
        assert entry is not None, f"image {item_hash} encoded before any request registered it"
        if entry.tensor is not None:
            return
        if self._storage == "cpu":
            stored = torch.empty(
                embedding.shape, dtype=embedding.dtype, device="cpu",
                pin_memory=torch.cuda.is_available(),
            )
            stored.copy_(embedding, non_blocking=True)
        else:
            stored = embedding
        entry.tensor = stored
        self._touch(item_hash)
        self._evict_to_cap()

    def get_slice(self, item_hash: int, lo: int, hi: int, device: torch.device) -> torch.Tensor:
        view = self._entries[item_hash].tensor[lo:hi]
        if view.device == device:
            return view
        return view.to(device, non_blocking=True)

    def consume(self, item_hash: int, uid: int, rows: int) -> None:
        """Account rows uid gathered; the last row of the last holder frees the entry."""
        entry = self._entries[item_hash]
        left = entry.remaining[uid] - rows
        assert left >= 0, f"request {uid} gathered more rows of image {item_hash} than it registered"
        if left:
            entry.remaining[uid] = left
            return
        del entry.remaining[uid]
        if not entry.remaining and entry.tensor is None:
            del self._entries[item_hash]  # never finished encoding: nothing worth caching

    def release(self, uid: int, hashes: list[int]) -> None:
        """Drop uid's claims whatever it gathered (abort); half-encoded entries are freed, completed ones stay cached."""
        for h in hashes:
            entry = self._entries.get(h)
            if entry is None:
                continue
            entry.remaining.pop(uid, None)
            if not entry.remaining and entry.tensor is None:
                del self._entries[h]

    def stats(self) -> tuple[int, int]:
        stored = [e.tensor for e in self._entries.values() if e.tensor is not None]
        return len(stored), sum(t.numel() * t.element_size() for t in stored)


__all__ = ["EncoderCache"]
