"""MLA latent-KV pools (GLM-5.2 / DeepSeek-style latent attention).

``MLAKVCache`` is the MHA pool's sibling for latent-KV models: ONE slab holding the
per-token latent ``ckv (kv_lora_rank) | kpe (qk_rope_head_dim)`` -- there is no
separate V (``v_cache`` aliases ``k_cache``, same convention as dsv4_paged_pool's
single-latent tiers). ``DSAKVCache`` extends it with the DeepSeek-Sparse-Attention
index-key slab: one ``index_head_dim``-wide bf16 row per token per full-indexer
layer, addressed by the SAME physical rows as the latent slab (GLM-5.2, page 1;
``KpoolDSAKVCache`` overrides the geometry to a 1/ratio shadow). ``rebuild``
resizes ALL slabs atomically so the allocator can never hand out a slot one slab
has and the other lacks.

Storage lives here -- not in the attention backend -- so the engine's rebuild path
(``MHAKVCache.rebuild``-shaped: fresh allocation, object identity preserved, views
re-derived by callers per forward) and the KV cost model (which budgets the index-K
bytes off the attention-group spec) stay correct by construction.
"""

from __future__ import annotations

import sys

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """Paged latent-KV pool: ``[1, num_layers, num_pages, page_size, 1, latent_dim]``.

    The leading singleton keeps the buffer shape-compatible with MHAKVCache's
    (tokens = shape[2] * shape[3]).

    ``layer_ids`` backs only a SUBSET of the model's layers (hybrid linear x
    MLA/DSA) while callers keep addressing by GLOBAL layer id -- the same remap
    contract as MHAKVCache; None keeps the identity mapping.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: "tuple[int, ...] | None" = None,
        kv_dtype: torch.dtype | None = None,
        kv_quant: str | None = None,
        host_offload: bool = False,
    ) -> None:
        self._latent_dim = latent_dim
        if layer_ids is None:
            self._num_layers = num_layers
            self._layer_index: dict[int, int] | None = None
        else:
            self._num_layers = len(layer_ids)
            self._layer_index = {int(g): i for i, g in enumerate(layer_ids)}
        self._page_size = page_size
        self._dtype = dtype
        self._device = device
        # Storage-only KV quantization (--kv-cache-dtype), MHAKVCache convention:
        # attention compute stays bf16, the pool packs/clamps on write and the gather
        # kernel dequants on read. turbo4 needs the pow-2 latent row as ONE quantized
        # vector (rotspace rotates the whole row, so a ckv|kpe split would mix after
        # the rotation) -- pure-NoPE checkpoints (rope dim 0) satisfy this exactly.
        storage_dtype = dtype if kv_dtype is None else kv_dtype
        self._kv_quant = kv_quant if storage_dtype != dtype else None
        if self._kv_quant == "turbo4" and latent_dim & (latent_dim - 1):
            raise ValueError(
                f"turbo4 latent storage needs a power-of-two latent_dim, got {latent_dim}"
            )
        self._storage_dtype = storage_dtype
        self._storage_head = latent_dim // 2 if self._kv_quant == "turbo4" else latent_dim
        # fp8 storage: saturating clamp on write at the storage max (the same
        # write-side hook MHAKVCache uses); compute stays bf16 either way.
        self._quant_limit = None
        if storage_dtype != dtype and self._kv_quant != "turbo4":
            self._quant_limit = float(torch.finfo(storage_dtype).max)
        self._dscale_buffer: torch.Tensor | None = None
        self._tq = None
        if self._kv_quant == "turbo4":
            from .turboquant import TurboQuantConstants

            self._tq = TurboQuantConstants(latent_dim, device)
        self._host_offload = host_offload
        self._gather_bytes = 0
        self._host_banks: list = []
        self._alloc(num_pages)

    def _local_layer(self, layer_id: int) -> int:
        return layer_id if self._layer_index is None else self._layer_index[layer_id]

    def _host_alloc(self, shape, dtype) -> torch.Tensor:
        """Pinned host tensor for the bulk body (KV host tier); falls back to a plain
        device tensor when the pin path is exhausted (the tier is an optimization, not a
        requirement). Windows takes the registered-mmap path directly -- see
        ``qsa_pool._alloc_host``; elsewhere fresh cudaHostAlloc is tried first."""
        from freetoken.moe.host_banks import HostBank
        from freetoken.utils import init_logger

        log = init_logger(__name__)
        shape = tuple(int(s) for s in shape)

        def _registered() -> torch.Tensor:
            bank = HostBank(shape, dtype, backing="mmap")  # lazy mmap: address space only
            bank.tensor.zero_()  # commit every page; registration demands resident pages
            bank.pin()
            self._host_banks.append(bank)
            log.info(
                f"kv host tier: pinned {bank.nbytes / 2**30:.2f} GiB via cudaHostRegister (register-direct)"
            )
            return bank.tensor

        if sys.platform.startswith("win"):
            try:
                return _registered()
            except Exception as exc:
                log.info(f"kv host tier: pin failed ({exc}), keeping the latent pool on GPU")
                return torch.empty(shape, dtype=dtype, device=self._device)
        try:
            from freetoken.kernel.pinned import alloc_pinned_tensor

            return alloc_pinned_tensor(*shape, dtype=dtype)
        except Exception:
            try:
                return _registered()
            except Exception as exc:
                log.info(f"kv host tier: pin failed ({exc}), keeping the latent pool on GPU")
                return torch.empty(shape, dtype=dtype, device=self._device)

    def _alloc(self, num_pages: int) -> None:
        self._num_pages = num_pages
        body_shape = (1, self._num_layers, num_pages, self._page_size, 1, self._storage_head)
        if self._host_offload:
            self._kv_buffer = self._host_alloc(body_shape, self._storage_dtype)
        else:
            self._kv_buffer = torch.empty(
                body_shape, device=self._device, dtype=self._storage_dtype
            )
        # turbo4 dequant scales ride the compute dtype (bf16), one per latent row;
        # zero-init so a never-written slot dequant-scales to a finite 0.
        self._dscale_buffer = None
        if self._kv_quant == "turbo4":
            ds_shape = (self._num_layers, num_pages * self._page_size)
            self._dscale_buffer = (
                self._host_alloc(ds_shape, self._dtype)
                if self._host_offload
                else torch.zeros(ds_shape, dtype=self._dtype, device=self._device)
            )
            self._dscale_buffer.zero_()

    def host_bytes(self) -> int:
        """Bytes of this pool's latent rows (+ turbo4 scales) parked in pinned host RAM."""
        if not self._host_offload or self._kv_buffer.is_cuda:
            return 0
        total = int(self._kv_buffer.numel()) * self._kv_buffer.element_size()
        if self._dscale_buffer is not None:
            total += int(self._dscale_buffer.numel()) * self._dscale_buffer.element_size()
        return total

    def count_gather(self, rows: int) -> None:
        """Fold one decode step's gather volume into the advisory :meth:`gather_bytes`.
        ``rows`` counts latent rows the selection names across the layers that read them."""
        self._gather_bytes += int(rows) * self._gather_row_bytes()

    def gather_bytes(self) -> int:
        """Lifetime bytes this pool had to serve to the sparse-index gathers. Advisory: it
        lets /v1/stats show the reads track the bounded selection width, not the history."""
        return self._gather_bytes

    def _gather_row_bytes(self) -> int:
        """Bytes one latent row pulls (K == V here), plus its turbo4 scale when in that layout."""
        row = self._storage_head * self._kv_buffer.element_size()
        if self._dscale_buffer is not None:
            row += self._dscale_buffer.element_size()
        return row

    # -- views (addressed by GLOBAL layer id; remapped when layer_ids was given) --
    def k_cache(self, layer_id: int) -> torch.Tensor:
        """Paged latent view ``[num_pages, page_size, latent_dim]``."""
        return self._kv_buffer[0, self._local_layer(layer_id)].view(
            self._num_pages, self._page_size, -1
        )

    def v_cache(self, layer_id: int) -> torch.Tensor:
        # MLA: K == V (single latent); same buffer, dsv4_paged_pool precedent.
        return self.k_cache(layer_id)

    def latent_rows(self, layer_id: int) -> torch.Tensor:
        """Row-flat latent view ``[num_pages * page_size, storage_head]`` (bf16 rows,
        or the uint8 nibble-packed rows under turbo4 -- half width)."""
        return self._kv_buffer[0, self._local_layer(layer_id)].view(-1, self._storage_head)

    def k_dscale(self, layer_id: int) -> torch.Tensor | None:
        """turbo4: bf16 dequant scale per latent row (None for bf16 storage)."""
        if self._dscale_buffer is None:
            return None
        return self._dscale_buffer[self._local_layer(layer_id)]

    # -- host RAM tier (KV parked on host, swapped into VRAM on demand) --------------
    def _row_idx(self, slots: torch.Tensor) -> torch.Tensor:
        # Body-device indices for the host tier: gather/index_select accept CPU indices on a
        # CUDA target, but CUDA indices on a CPU view are rejected, so with a host body the
        # indices must be CPU. Device-side slab ops (`index_copy_` on `_index_k_buffer`) require
        # index/src/dst all on one device, so those call sites convert back with `.to(_device)`.
        idx = slots.to(torch.int64)
        return idx.cpu() if self._host_offload else idx.to(self._device)

    def snapshot_rows(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Host-RAM copy of the latent rows (and turbo4 dequant scales) at the given token
        slots -- MHAKVCache convention: byte-level so the packed/quantized representation
        round-trips untouched and restore needs no re-quantize. Backing arrays are
        ``[layers, tokens, head]`` / ``[layers, tokens]``, so the gather is along dim 1."""
        # Body-device indices: normally host_offload means a host body (CPU idx), but the
        # pin-failure fallback keeps the body on GPU while the flag stays True -- derive
        # from the buffer, not the flag (index_copy_ needs all three on one device).
        idx = self._row_idx(slots).to(self._kv_buffer.device)
        tokens = self._num_pages * self._page_size
        out = {"kv": self._kv_buffer.view(self._num_layers, tokens, -1)[:, idx].cpu()}
        if self._dscale_buffer is not None:
            out["ds"] = self._dscale_buffer[:, idx].cpu()
        return out

    def restore_rows(self, slots: torch.Tensor, data: dict[str, torch.Tensor]) -> None:
        """Promote a ``snapshot_rows`` payload back into the pool at freshly allocated token
        slots -- the swap-in half. index_copy_ along dim 1 (advanced indexing gathers into a
        temporary, so copying into it never reaches the slab)."""
        idx = self._row_idx(slots).to(self._kv_buffer.device)
        tokens = self._num_pages * self._page_size
        n = idx.shape[0]
        # the body device follows the tier: pinned host bodies take host-side copies,
        # device bodies the H2D hop (index_copy_ needs index, src and dst on one device)
        bdev = self._kv_buffer.device
        self._kv_buffer.view(self._num_layers, tokens, -1).index_copy_(
            1, idx, data["kv"].to(bdev)[:, :n]
        )
        if self._dscale_buffer is not None and "ds" in data:
            self._dscale_buffer.index_copy_(
                1, idx, data["ds"].to(bdev)[:, :n]
            )

    # -- writes -----------------------------------------------------------------
    def store_kv(
        self,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Scatter this forward's latent rows: ``c_kv`` [T, kv_lora_rank] and
        ``k_rope`` [T, qk_rope_head_dim] land in the row's two halves.

        Two narrow ``index_put_`` scatters. TODO: fuse into kernel/csrc
        store.cu (two-width store).
        """
        rows = self.latent_rows(layer_id)
        if not self._kv_buffer.is_cuda:
            # Host-tier body: assemble the latent row on the compute device (the two halves
            # become one contiguous row, which is also what turbo4 already does) and write it
            # through into the pinned slab in one launch.
            from freetoken.kernel.triton.host_store import store_rows_to_host

            full = c_kv if k_rope.shape[-1] == 0 else torch.cat([c_kv, k_rope], dim=-1)
            if self._kv_quant == "turbo4":
                packed, dscale = self._tq.quantize(full)
                store_rows_to_host(rows, packed, out_loc)
                # The dscale slab is laid out 1-D per layer ([tokens]) and this step's
                # scales are 1-D [T]; present both as row-flat [n, 1] so the helper's
                # per-row byte-width check compares element widths (1 bf16 == 1 bf16)
                # instead of total lengths, and the kernel scatters one scale per row.
                store_rows_to_host(
                    self._dscale_buffer[self._local_layer(layer_id)].view(-1, 1),
                    dscale.view(-1, 1),
                    out_loc,
                )
                return
            if self._quant_limit is not None:
                limit = self._quant_limit
                full = full.clamp(-limit, limit).to(self._storage_dtype)
            store_rows_to_host(rows, full, out_loc)
            return
        if self._kv_quant == "turbo4":
            # The whole latent row is ONE rot-space vector (heads=1): rotate+pack it
            # and scatter the uint8 rows with the same row-copy the bf16 path does
            # (capture-safe, no new store-kernel element sizes). Scale -> parallel buffer.
            full = c_kv if k_rope.shape[-1] == 0 else torch.cat([c_kv, k_rope], dim=-1)
            packed, dscale = self._tq.quantize(full)
            idx = out_loc.to(torch.int64)
            rows.index_copy_(0, idx, packed)
            self._dscale_buffer[self._local_layer(layer_id)].index_copy_(0, idx, dscale)
            return
        split = rows.shape[1] - k_rope.shape[-1]
        if self._quant_limit is not None:
            # fp8 destinations need the cast spelled out (index_put is dtype-strict);
            # clamp first so the rounding saturates at the storage max, not at inf.
            limit = self._quant_limit
            rows[out_loc, :split] = c_kv.clamp(-limit, limit).to(self._storage_dtype)
            rows[out_loc, split:] = k_rope.clamp(-limit, limit).to(self._storage_dtype)
            return
        rows[out_loc, :split] = c_kv
        rows[out_loc, split:] = k_rope

    def _detach_host_banks(self) -> None:
        # Retire the middle-tier banks of the outgoing generation so each rebuild replaces
        # its host banks instead of stacking another full latent tier on top (banks ride
        # _LIVE_BUFFERS for their lifetime; detach() drops that keep-alive). The slab views
        # go first so no view outlives the closed mmap.
        self._kv_buffer = None
        self._dscale_buffer = None
        for bank in self._host_banks:
            bank.detach()
        self._host_banks.clear()

    def rebuild(self, num_pages: int) -> None:
        """In-place resize (frees the old slab first; object identity preserved --
        callers re-derive views per forward, same contract as MHAKVCache.rebuild)."""
        self._kv_buffer = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._detach_host_banks()
        self._alloc(num_pages)

    @staticmethod
    def _index_row_bytes_per_token(spec) -> int:
        return spec.index_head_dim * spec.num_index_layers * 2 // spec.index_ratio

    @classmethod
    def host_bytes_per_token(cls, config) -> int:
        """Per-token bytes the host tier parks in pinned RAM: the latent rows with the
        indexer-slab term (which stays device-side) subtracted."""
        from .base import spec_kv_bytes_per_token

        total = 0
        for spec in config.model_config.kv_cache_group_specs():
            total += spec_kv_bytes_per_token(spec, config) - cls._index_row_bytes_per_token(spec)
        return total

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        host_offload = bool(getattr(config, "kv_offload", False))
        per_token = 0
        for spec in config.model_config.kv_cache_group_specs():
            if host_offload:
                term = cls._index_row_bytes_per_token(spec)
                if term == 0:
                    # no indexer dims -> nothing scales with pages on the GPU side;
                    # price the whole row there rather than divide-by-zero the page solve
                    term = spec_kv_bytes_per_token(spec, config)
                per_token += term
            else:
                per_token += spec_kv_bytes_per_token(spec, config)
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = self._num_pages * self._page_size
        if self._host_offload and not buf.is_cuda:
            # host-resident body: the latent tier contributes no VRAM per token, the
            # slide denominates in the indexer slab only (DSAKVCache adds it on top)
            return 0, 0
        total = int(buf.numel() * buf.element_size())
        if self._dscale_buffer is not None:
            total += int(self._dscale_buffer.numel() * self._dscale_buffer.element_size())
        return total // tokens, 0

    # -- pool properties ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        # STORAGE dtype (MHAKVCache convention): bf16, fp8, or the uint8 turbo4
        # carrier -- the backend reads it to decide its unpack-on-read branch.
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


class DSAKVCache(MLAKVCache):
    """MLA latent pool + the DSA index-key slab (one row per token per full-indexer
    layer, bf16, slot order = the backend's full-layer order)."""

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        layer_ids: "tuple[int, ...] | None" = None,
        kv_dtype: torch.dtype | None = None,
        kv_quant: str | None = None,
        host_offload: bool = False,
    ) -> None:
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        super().__init__(
            latent_dim, num_layers, num_pages, page_size, dtype, device,
            layer_ids=layer_ids, kv_dtype=kv_dtype, kv_quant=kv_quant,
            host_offload=host_offload,
        )

    def _index_rows(self, num_pages: int) -> int:
        """Index-slab row count: one row per token (KpoolDSAKVCache overrides to the
        1/ratio shadow + scratch layout)."""
        return num_pages * self._page_size

    def _alloc(self, num_pages: int) -> None:
        # Both slabs in one allocation step so rebuild can never leave one grown
        # and the other stale.
        super()._alloc(num_pages)
        # bf16 == the 2 bytes/token/layer the KV cost model budgets for this slab
        # (cache_status._kv_cost_model); keep the two in lockstep.
        self._index_k_buffer = torch.zeros(
            self._num_index_layers,
            self._index_rows(num_pages),
            self._index_head_dim,
            dtype=torch.bfloat16,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        self._index_k_buffer = None
        super().rebuild(num_pages)

    def unit_bytes(self) -> tuple[int, int]:
        # The index slab rides the same token budget as the latent slab; each slab's per-token
        # cost is floor-divided on its own, matching the cost model's two separate terms.
        kv, swa = super().unit_bytes()
        idx = self._index_k_buffer
        tokens = self._num_pages * self._page_size
        return kv + int(idx.numel() * idx.element_size()) // tokens, swa

    def index_k_cache(self, slot: int) -> torch.Tensor:
        """Row-flat index keys for a full-indexer layer slot: ``[rows, index_head_dim]``."""
        return self._index_k_buffer[slot]

    def store_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, slot: int) -> None:
        self._index_k_buffer[slot][out_loc] = k

    def _index_slot_rows(self, slots: torch.Tensor) -> torch.Tensor:
        """Index-slab rows for the given token slots: one row per token here (one row per
        KpoolDSAKVCache overrides to the 1/ratio shadow)."""
        return self._row_idx(slots)

    def snapshot_rows(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        # The DSA index slab rides the KV rows so a restored prefix re-selects the same top-k.
        # `_index_k_buffer` is always device-resident (index keys stay on GPU even under kv-offload),
        # so indices for slab ops are on-device regardless of body tier; `_row_idx` returns CPU
        # indices only for the host-body gather (which tolerates CPU indices), so convert here.
        out = super().snapshot_rows(slots)
        rows = self._index_slot_rows(slots).to(self._device)
        out["idx"] = self._index_k_buffer[:, rows].cpu()
        return out

    def restore_rows(self, slots: torch.Tensor, data: dict[str, torch.Tensor]) -> None:
        super().restore_rows(slots, data)
        # index_copy_ requires index/src/dst all on the same device; the slab is CUDA, so
        # rows must be a CUDA tensor even when the body is host-resident with CPU indices.
        rows = self._index_slot_rows(slots).to(self._device)
        self._index_k_buffer.index_copy_(1, rows, data["idx"].to(self._device)[:, : rows.shape[0]])


class KpoolDSAKVCache(DSAKVCache):
    """DSAKVCache with the index slab as a 1/ratio SHADOW of the KV pages, plus
    per-request scratch rows and tail rings.

    A pooled entry lives at ``token_slot // index_ratio`` (``page_size %
    index_ratio == 0``), so compressed rows follow page sharing/eviction for
    free. Rows whose pool does not close write to the request's scratch row
    instead (scoring never reads it). The tail rings hold the in-progress
    pool's raw K + gate at ``pos % index_ratio`` for pools straddling two
    forwards; never cleared -- prefix-cache resume points are page-aligned,
    so a stale ring is never read.
    """

    def __init__(self, *args, num_req_slots: int, index_ratio: int, **kwargs) -> None:
        self._num_req_slots = num_req_slots
        self._index_ratio = index_ratio
        super().__init__(*args, **kwargs)
        assert self._page_size % index_ratio == 0, (
            f"kpool needs page_size ({self._page_size}) divisible by "
            f"index_ratio ({index_ratio})"
        )

    def _index_rows(self, num_pages: int) -> int:
        # 1/ratio shadow of every token slot + one scratch row per request slot.
        return num_pages * self._page_size // self._index_ratio + self._num_req_slots

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row (== shadow row count); request ``table_idx`` offsets it."""
        return self._num_pages * self._page_size // self._index_ratio

    def _alloc(self, num_pages: int) -> None:
        super()._alloc(num_pages)
        self._tail_k = torch.zeros(
            self._num_index_layers, self._num_req_slots, self._index_ratio,
            self._index_head_dim, dtype=torch.bfloat16, device=self._device,
        )
        self._tail_gate = torch.zeros_like(self._tail_k)

    def rebuild(self, num_pages: int) -> None:
        self._tail_k = None
        self._tail_gate = None
        super().rebuild(num_pages)

    def tail_k(self, slot: int) -> torch.Tensor:
        """Tail raw keys for an indexer layer slot: ``[num_req_slots, ratio, head_dim]``."""
        return self._tail_k[slot]

    def tail_gate(self, slot: int) -> torch.Tensor:
        return self._tail_gate[slot]

    def _index_slot_rows(self, slots: torch.Tensor) -> torch.Tensor:
        # kpool index rows are the 1/ratio SHADOW of the KV slots (the tail/pending rings
        # are transient -- rebuilt per forward, and a page-aligned boundary never splits a
        # group -- so only the shadow rows ride the tier).
        return self._row_idx(slots) // self._index_ratio


__all__ = ["MLAKVCache", "DSAKVCache", "KpoolDSAKVCache"]
