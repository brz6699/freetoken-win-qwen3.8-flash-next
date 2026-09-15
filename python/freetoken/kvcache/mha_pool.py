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
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, storage_head_dim),
            device=device,
            dtype=storage_dtype,
        )
        self._dscale_buffer = (
            torch.zeros(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads),
                dtype=dtype,  # dequant scales ride the compute dtype (bf16)
                device=device,
            )
            if self._kv_quant == "turbo4"
            else None
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, storage_head_dim)
        if self._kv_quant == "turbo4":
            from .turboquant import TurboQuantConstants

            self._tq = TurboQuantConstants(head_dim, device)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        old_dscale = self._dscale_buffer
        self._dscale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        if old_dscale is not None:
            # turbo4: scales are geometry-shaped like the token axis; zero-init matches the
            # constructor (never-written tokens must dequant to a finite 0).
            self._dscale_buffer = torch.zeros(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads),
                dtype=old_dscale.dtype,
                device=device,
            )
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

    def snapshot_rows(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Host-RAM copy of the packed K/V (and turbo4 dequant scales) at the given token
        slots -- the store side of the mm RAM tier. Byte-level on purpose: the packed/quantized
        representation round-trips untouched, so restore needs no re-quantize."""
        idx = slots.to(torch.int64).to(self._device)
        rows = self._kv_buffer.shape[1] * 2
        tokens = self._storage_shape[0]
        out = {"kv": self._kv_buffer.view(rows, tokens, -1)[:, idx].cpu()}
        if self._dscale_buffer is not None:
            out["ds"] = self._dscale_buffer.view(rows, tokens, -1)[:, idx].cpu()
        return out

    def restore_rows(self, slots: torch.Tensor, data: dict[str, torch.Tensor]) -> None:
        """Promote a ``snapshot_rows`` payload back into the pool at (freshly allocated)
        token slots -- the swap-in half of the mm RAM tier."""
        # index_copy_ (not ``[:, idx].copy_``: advanced indexing gathers into a temporary,
        # so copying into it never reaches the slab).
        idx = slots.to(torch.int64).to(self._device)
        rows = self._kv_buffer.shape[1] * 2
        tokens = self._storage_shape[0]
        self._kv_buffer.view(rows, tokens, -1).index_copy_(1, idx, data["kv"])
        if self._dscale_buffer is not None and "ds" in data:
            self._dscale_buffer.view(rows, tokens, -1).index_copy_(1, idx, data["ds"])

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
