from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_dtype: torch.dtype | None = None,
        kv_quant: str | None = None,
        arena: torch.Tensor | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        # ``kv_dtype`` selects the STORAGE dtype independently of compute dtype
        # (--kv-cache-dtype fp8_e4m3): attention math stays bf16, the pool clamps on write
        # at the storage max (scale 1.0, saturating) and the kernels dequant on read.
        storage_dtype = dtype if kv_dtype is None else kv_dtype
        # turbo4 (--kv-cache-dtype turbo4): 4-bit TurboQuant, the uint8 kv_dtype is a PACKED
        # carrier -- head_dim halves into nibble pairs and a parallel bf16 scale buffer holds
        # one dequant scale per (token, head, slab). See kvcache/turboquant.py.
        self._kv_quant = kv_quant if storage_dtype != dtype else None
        if self._kv_quant == "turbo4":
            storage_head_dim = head_dim // 2
            self._tq_dim = head_dim
        else:
            storage_head_dim = head_dim
            self._tq_dim = 0
        self._quant_limit = None
        if self._kv_quant is None and storage_dtype != dtype:
            self._quant_limit = float(torch.finfo(storage_dtype).max)
        self._device = device
        # Shared flat arena (uint8, engine-owned): when set, every buffer below is a view
        # off it instead of a private allocation, so runtime resizes re-slice without ever
        # moving the base address.
        self._arena = arena
        self._arena_next = 0
        self._kv_buffer = self._arena_alloc(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, storage_head_dim),
            storage_dtype,
        )
        self._dscale_buffer = None
        if self._kv_quant == "turbo4":
            # dequant scales ride the compute dtype (bf16), in their own section after the
            # packed K/V; zero-init so never-written tokens dequant to a finite 0.
            ds = self._arena_alloc(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads), dtype
            )
            ds.zero_()
            self._dscale_buffer = ds
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, storage_head_dim)
        if self._kv_quant == "turbo4":
            from .turboquant import TurboQuantConstants

            self._tq = TurboQuantConstants(head_dim, device)

    def _arena_alloc(self, shape, dtype) -> torch.Tensor:
        """Next 16-byte-aligned section of the shared arena, viewed as ``dtype``; a plain
        ``torch.empty`` when no arena was provided. Left-anchored: the KV section starts at
        offset 0, so the block's base address is stable across resizes and no storage is
        ever re-asked from the allocator (the MoE side anchors its bank slot-caches from
        the arena's RIGHT end -- the two sides meet at the sliding boundary, total bytes
        capped by the budget that sized the arena). See ``rebuild`` for the content
        caveat: re-slicing shifts the non-first slabs' views, it does not carry their
        bytes along."""
        if self._arena is None:
            return torch.empty(tuple(int(s) for s in shape), dtype=dtype, device=self._device)
        numel = 1
        for s in shape:
            numel *= int(s)
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()
        # 16-byte alignment: matches torch's own allocator and keeps the fused-copy
        # descriptor's data_ptr checks happy for anything sliced off the arena.
        off = (self._arena_next + 15) & ~15
        assert off + nbytes <= int(self._arena.numel()), (
            f"arena overflow: KV side needs {off + nbytes} B "
            f"(this section at {off}..{off + nbytes}), arena {int(self._arena.numel())} B"
        )
        view = self._arena.narrow(0, off, nbytes).view(dtype)
        self._arena_next = off + nbytes
        return view.view(tuple(int(s) for s in shape))

    def rebuild(self, num_pages: int) -> None:
        """Re-slice the KV views for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. With a shared arena the storage
        block never moves -- views are re-sliced from the arena's LEFT end (offset 0), so
        the base address is stable and no byte is ever reallocated or freed (the point of
        the pipe: resizes slide the KV/MoE boundary inside one block instead of re-asking
        the allocator). That is ALLOCATOR stability, not content stability: in the
        (kv, layer, page, ...) layout only the very first slab (K, storage layer 0) sits
        at an offset independent of the page count; the V slab and every layer-1+ slab
        sit at offsets that scale with P, so a re-slice shifts those views. Cross-rebuild
        CONTENT is not part of the pool contract anyway: CacheManager.rebuild wipes the prefix
        tree (its page-index -> byte mapping follows the page count) and the kernels pos-mask
        every load, so nothing reads the shifted live-slab bytes. The host RAM tier is kept
        across the rebuild, but that is independent of this: its parked payload is compact
        per-(kv,layer) rows gathered at snapshot time and scattered into freshly allocated
        slots by ``restore_rows``, so it never depends on the live slab offsets. Without an
        arena, the old buffers are freed
        before the new ones are allocated (legacy path). Views and ``_storage_shape``
        are refreshed; object identity is preserved so cached references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        ds_dtype = None if self._dscale_buffer is None else self._dscale_buffer.dtype
        device = self._device
        self._arena_next = 0
        if self._arena is None:
            # free-before-alloc only in the self-owned path; arena views keep their storage
            self._k_buffer = None
            self._v_buffer = None
            self._kv_buffer = None
            self._dscale_buffer = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        self._kv_buffer = self._arena_alloc(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim), dtype
        )
        self._dscale_buffer = None
        if ds_dtype is not None:
            # turbo4: scales are geometry-shaped like the token axis; zero-init matches the
            # constructor (never-written tokens must dequant to a finite 0).
            ds = self._arena_alloc(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads), ds_dtype
            )
            ds.zero_()
            self._dscale_buffer = ds
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        total = int(buf.numel() * buf.element_size())
        if self._dscale_buffer is not None:
            total += int(self._dscale_buffer.numel() * self._dscale_buffer.element_size())
        return total // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    _TIER_CHUNK_BYTES = 64 << 20

    @staticmethod
    def _tier_step(view: torch.Tensor, n_slots: int) -> int:
        budget = MHAKVCache._TIER_CHUNK_BYTES
        if view.is_cuda:
            try:
                free, _total = torch.cuda.mem_get_info(view.device)
                budget = max(1 << 22, min(budget, int(free) // 2))
            except Exception:
                pass
        per_slot = int(view.shape[0] * view.shape[2]) * view.element_size()
        return max(1, min(n_slots, budget // max(per_slot, 1)))

    def snapshot_rows(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Host-RAM copy of the packed K/V (and turbo4 dequant scales) at the given token
        slots -- the store side of the mm RAM tier. Byte-level on purpose: the packed/quantized
        representation round-trips untouched, so restore needs no re-quantize.

        Gathered in slot chunks straight into host: one whole-slab gather of a long prompt
        (~3 GiB at 500K tokens) can exceed the free-VRAM margin left after a budget slide
        and OOM the worker; per-chunk temporaries stay inside the chunk budget."""
        idx = slots.to(torch.int64).to(self._device)
        rows = self._kv_buffer.shape[1] * 2
        tokens = self._storage_shape[0]
        out = {"kv": self._gather_host(self._kv_buffer.view(rows, tokens, -1), idx)}
        if self._dscale_buffer is not None:
            out["ds"] = self._gather_host(self._dscale_buffer.view(rows, tokens, -1), idx)
        return out

    def _gather_host(self, view: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        n = int(idx.numel())
        step = self._tier_step(view, n)
        parts = [view[:, idx[a:a + step]].cpu() for a in range(0, n, step)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def restore_rows(self, slots: torch.Tensor, data: dict[str, torch.Tensor]) -> None:
        """Promote a ``snapshot_rows`` payload back into the pool at (freshly allocated)
        token slots -- the swap-in half of the mm RAM tier."""
        # index_copy_ (not ``[:, idx].copy_``: advanced indexing gathers into a temporary,
        # so copying into it never reaches the slab). Chunked for the same reason as
        # snapshot: the whole host payload promoted at once competes with live decode
        # memory; per-chunk .to(device) stays inside the chunk budget.
        idx = slots.to(torch.int64).to(self._device)
        rows = self._kv_buffer.shape[1] * 2
        tokens = self._storage_shape[0]
        self._scatter_host(self._kv_buffer.view(rows, tokens, -1), idx, data["kv"])
        if self._dscale_buffer is not None and "ds" in data:
            self._scatter_host(self._dscale_buffer.view(rows, tokens, -1), idx, data["ds"])

    def _scatter_host(self, view: torch.Tensor, idx: torch.Tensor, data: torch.Tensor) -> None:
        n = int(idx.numel())
        if data.shape[1] > n:
            # A text-tier restore may trim the boundary to a shorter common prefix than the
            # stored payload; the extra columns are rows past the restore point -- drop them so
            # index_copy_'s source height matches the index count.
            data = data[:, :n]
        step = self._tier_step(view, n)
        for a in range(0, n, step):
            view.index_copy_(1, idx[a:a + step], data[:, a:a + step].to(self._device))

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_dscale(self, index: int) -> torch.Tensor | None:
        """turbo4: the (tokens, kv_heads) bf16 dequant scales of the K slab (None otherwise)."""
        if self._dscale_buffer is None:
            return None
        return self._dscale_buffer[0][self._dense(index)]

    def v_dscale(self, index: int) -> torch.Tensor | None:
        if self._dscale_buffer is None:
            return None
        return self._dscale_buffer[1][self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        if self._kv_quant == "turbo4":
            self._store_kv_turbo4(k, v, out_loc, dense)
            return
        if self._quant_limit is None:
            from freetoken.kernel import store_cache

            store_cache(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
            return
        # Quantizing store (fp8 KV): saturating clamp then torch byte-scatter. Plain
        # index_copy_ (not a new JIT store-kernel element size) keeps this capture-safe
        # with no nvcc; the scatter rides a uint8 view because index_copy_ has no fp8
        # kernel -- a 1-byte row copy is exactly what the bf16 store kernel does too.
        tokens = self._storage_shape[0]
        idx = out_loc.to(torch.int64)
        limit = self._quant_limit
        buf_dtype = self._kv_buffer.dtype
        self._k_buffer[dense].view(tokens, -1).view(torch.uint8).index_copy_(
            0,
            idx,
            k.clamp(-limit, limit).to(buf_dtype).reshape(k.shape[0], -1).view(torch.uint8),
        )
        self._v_buffer[dense].view(tokens, -1).view(torch.uint8).index_copy_(
            0,
            idx,
            v.clamp(-limit, limit).to(buf_dtype).reshape(v.shape[0], -1).view(torch.uint8),
        )

    def _store_kv_turbo4(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        dense: int,
    ) -> None:
        """TurboQuant 4-bit store: rotate + quantize + nibble-pack (kvcache/turboquant.py),
        then the same uint8 row-scatter the fp8 path uses (capture-safe, no nvcc, no new
        store-kernel element sizes). Scales go to the parallel bf16 buffer."""
        tokens = self._storage_shape[0]
        idx = out_loc.to(torch.int64)
        kv_heads = self._dscale_buffer.shape[3]
        for source, buffer, dscale_row in (
            (k, self._k_buffer[dense], self._dscale_buffer[0][dense]),
            (v, self._v_buffer[dense], self._dscale_buffer[1][dense]),
        ):
            packed, dscale = self._tq.quantize(source)
            buffer.view(tokens, -1).index_copy_(
                0, idx, packed.view(packed.shape[0] // kv_heads, -1)
            )
            dscale_row.index_copy_(0, idx, dscale.view(packed.shape[0] // kv_heads, kv_heads))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
