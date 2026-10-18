# Pinned, grow-only slab pool for the GDN (gated-delta-net) prefill transients.
#
# WHY: the scheduler caps each prefill step at --max-extend-length (max_extend_tokens, default
# 8192), so a GDN prefill materializes only ONE <=8192 chunk's worth of big intermediates at a
# time (o, h, v_new, u, w) -- never the whole prompt, and its VRAM footprint is a FIXED
# function of that chunk cap, not of prompt length. But those buffers are (re)allocated on
# EVERY chunked-prefill step, and on Windows the caching allocator cannot coalesce the blocks
# it frees (expandable_segments is hard-disabled), so after enough steps the
# reserved-but-unallocated pool fragments and the biggest transient (chunk_fwd_o's `o`, ~96 MiB
# at 8192) stops finding a contiguous run -> OOM. This module removes that by pinning the
# (bounded) transients in slabs that are never freed/re-fragmented.
#
# HOW: SlabPool holds a persistent reference to one grow-only flat slab per (tag, dtype,
# device). The caching allocator can only free a block whose refcount hits 0, so a slab we keep
# pinned is never freed and never re-fragmented between steps: its contiguous run stays
# resident on the hot path. carve() re-shapes the flat reservation to the exact `shape`; a
# flat buffer .view(shape) yields standard contiguous strides, byte-identical to a fresh
# torch.empty(shape), so the triton kernels see exactly what they saw before. Every element of
# these outputs is overwritten by the kernel each call (kernel-written accumulators), so a
# shared slab needs no re-zero. GDN prefill layers run sequentially on one CUDA stream, so a
# slab is safe to reuse across layers/calls.
#
# Bounded: because the chunk is capped, each slab's high-water is bounded (the full GDN set is
# ~0.5 GiB at 8192). It grows only if a caller passes a chunk larger than ever seen.
#
# Extensibility: SlabPool is model-agnostic -- any transient (any shape/dtype) can be carved
# under its own tag, so new prefill buffers or other linear-attention families (KDA) join by
# one carve() call. Off the CUDA path (CPU / hybrid side) carve() falls back to a fresh
# allocation, so the pinned slabs only back the GPU prefill kernels.

from __future__ import annotations

import gc

import torch


class SlabPool:
    """A pinned, grow-only pool of contiguous GPU slabs, keyed by (tag, dtype, device)."""

    def __init__(self) -> None:
        # (tag, dtype, device) -> (numel, flat buffer). Grown high-water-only; never shrunk or
        # freed while the pool lives, which is precisely what keeps each region contiguous.
        self._slabs: dict[tuple, tuple[int, torch.Tensor]] = {}

    def carve(self, tag: str, shape, ref: torch.Tensor) -> torch.Tensor:
        """Return a contiguous tensor of ``shape`` from a pinned grow-only slab for ``tag``.

        ``ref`` supplies (dtype, device, is_cuda); ``shape`` is the exact target shape (the
        caller passes the real shape, e.g. the B=1 varlen prefill shape). Off the CUDA path a
        fresh allocation is returned so the pinned slabs only back the GPU prefill kernels.
        """
        if not ref.is_cuda:
            return torch.empty(tuple(shape), dtype=ref.dtype, device=ref.device)
        numel = 1
        for s in shape:
            numel *= int(s)
        return self._carve_flat(tag, numel, ref.dtype, ref.device)[:numel].view(*shape)

    def _carve_flat(self, tag: str, numel: int, dtype, device) -> torch.Tensor:
        key = (tag, dtype, device)
        ent = self._slabs.get(key)
        if ent is None or ent[0] < numel:
            buf = torch.empty(numel, dtype=dtype, device=device)
            self._slabs[key] = (numel, buf)
            return buf
        return ent[1]

    def release(self) -> None:
        """Drop every pinned slab so its blocks return to the allocator (e.g. before a
        rebuild that must move GPU memory)."""
        self._slabs = {}
        gc.collect()

    @property
    def reserved_bytes(self) -> int:
        """Total bytes currently pinned (the size of the fixed prefill-inference region)."""
        return sum(buf.numel() * buf.element_size() for _n, buf in self._slabs.values())


# The default pool: every GDN prefill transient is carved from here under its own tag.
_default = SlabPool()


def prefill_scratch(shape, tag: str, ref: torch.Tensor) -> torch.Tensor:
    """Carve ``shape`` from the default GDN-prefill slab pool (see :class:`SlabPool`)."""
    return _default.carve(tag, shape, ref)


__all__ = ["SlabPool", "prefill_scratch"]
