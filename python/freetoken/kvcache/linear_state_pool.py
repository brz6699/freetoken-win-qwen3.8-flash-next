from __future__ import annotations

import math

import torch
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.utils import div_even

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def _linear_local_dims(
    group: LinearGatedDeltaGroupConfig, tp_size: int
) -> tuple[int, int, int]:
    """TP-local ``(n_layers, conv_dim, v_heads)`` for the GDN state tensors -- the single
    source of the sharding math shared by the pool allocation and the byte estimate."""
    local_k_heads = div_even(group.num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(group.num_value_heads, tp_size, allow_replicate=True)
    local_conv_dim = 2 * local_k_heads * group.key_head_dim + local_v_heads * group.value_head_dim
    return len(group.layer_ids), local_conv_dim, local_v_heads


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers.

    Indexed by ``Req.table_idx`` (0..max_running_req), the same per-request slot the
    page table uses, so the scheduler's existing admit/free of ``table_idx`` covers the
    state's lifetime. One fixed slot per running request; no paging, no eviction.

    A model can declare extra per-request tensors on the same slots through
    ``ModelConfig.slot_states`` (see ``SlotStateSpec``); they advance, snapshot, COW and
    rebuild with the GDN state and are read back through ``slot_state(name, layer_id)``.
    Consumers must re-read them each forward: ``rebuild`` replaces the tensors.
    """

    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int | None = None,
        slot_states: tuple[SlotStateSpec, ...] = (),
    ) -> None:
        if tp_size is None:
            tp_size = get_tp_info().size

        self._group = group
        self._num_slots = num_slots
        self._device = device
        self._conv_dtype = dtype

        n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, group.conv_kernel_dim - 1),
            dtype=dtype,
            device=device,
        )
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype); the dtype is
        # overridable via FREETOKEN_MAMBA_SSM_DTYPE (see ssm_state_dtype).
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, group.key_head_dim, group.value_head_dim),
            dtype=ssm_state_dtype(),
            device=device,
        )
        self._local_index = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}

        # Per-step state snapshots a speculative verify forward writes into (see
        # ensure_verify_rows): ``_verify_blocks`` blocks of ``_verify_rows`` rows. None until the
        # first verify asks for them.
        self._verify_rows = 0
        self._verify_blocks = 0
        self.verify_rec: torch.Tensor | None = None
        self.verify_conv: torch.Tensor | None = None

        self._slot_specs = tuple(slot_states)
        names = [spec.name for spec in self._slot_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate slot_state names: {names}")
        self._state_layer_index = {
            spec.name: {lid: i for i, lid in enumerate(spec.layer_ids)}
            for spec in self._slot_specs
        }
        self.slot_states: dict[str, torch.Tensor] = self._alloc_slot_states(num_slots)

        # Free-list allocator over slots 1..num_slots-1 (slot 0 reserved as a padding sink,
        # sglang MambaPool convention). Live working slots, ping-pong track slots, and
        # radix-tree-donated snapshots are all drawn from this single free-list, so memory
        # flows between them by demand. Unused by the op harness (which assigns slots by hand).
        self.padding_slot = 0
        self._free_slots: list[int] = list(range(1, num_slots))

    def _alloc_slot_states(self, num_slots: int) -> dict[str, torch.Tensor]:
        return {
            spec.name: torch.full(
                (max(1, len(spec.layer_ids)), num_slots, *spec.shape),
                spec.fill_value,
                dtype=spec.dtype if spec.dtype is not None else self._conv_dtype,
                device=self._device,
            )
            for spec in self._slot_specs
        }

    def has_slot_state(self, name: str) -> bool:
        return name in self.slot_states

    def slot_state(self, name: str, layer_id: int | None = None) -> torch.Tensor:
        """One declared sibling state, ``[num_slots, *shape]``; ``layer_id`` picks the layer row."""
        t = self.slot_states[name]
        if layer_id is None:
            assert not self._state_layer_index[name], (
                f"slot_state {name!r} is per-layer, pass layer_id"
            )
            return t[0]
        return t[self._state_layer_index[name][layer_id]]

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(
                f"LinearStatePool exhausted: need {n}, have {len(self._free_slots)}"
            )
        return [self._free_slots.pop() for _ in range(n)]

    def reclaim_all_slots(self) -> None:
        """Restore the free-list to all non-padding slots. Idle-only: the caller (e.g. a
        CacheManager rebuild that discards the tree owning donated snapshots) must guarantee no
        running request holds a slot, otherwise live state would be handed out twice."""
        self._free_slots = list(range(1, self._num_slots))

    def rebuild(self, num_slots: int) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE.

        Geometry (layers, conv dim, head dims) and dtypes are taken from the existing
        tensors; only the slot count changes. Object identity is preserved so cached
        references (ctx.linear_state_pool) stay valid. Idle-only.

        The first ``min(old, new)`` slots carry their content and ownership over from the
        old tensors; slots past ``num_slots`` are dropped and brand-new slots start zeroed.
        Preserving the overlap is what keeps a resumed decode from reading cold (zeroed)
        recurrent state for its first chunk after a budget slide -- without it the delta-rule
        recurrence has to refill from scratch and the output degenerates (repeated-punctuation
        runs) for ~one chunk before it recovers.
        """
        old_conv, old_rec = self.conv_states, self.recurrent_states
        old_slot = self.slot_states
        old_num = self._num_slots
        old_free = set(self._free_slots)

        n_layers, _, local_conv_dim, km1 = old_conv.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = old_rec.shape
        conv_dtype, rec_dtype = old_conv.dtype, old_rec.dtype
        device = self._device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        ov = min(old_num, num_slots)

        # Interleaved alloc -> overlap-copy -> release, per KIND, so the transient peak is one
        # old/new pair instead of both pools whole: a grow slide is exactly where old+new
        # together exceed free VRAM and the whole rebuild OOMs. Rebinding self.* before
        # dropping the local is what makes the old tensor actually free for the next alloc.
        # Slot 0 is the padding sink; copies cover only the usable range 1..ov.
        new_conv = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1), dtype=conv_dtype, device=device
        )
        if ov > 1:
            new_conv[:, :ov].copy_(old_conv[:, :ov])
        self.conv_states = new_conv
        del old_conv

        new_rec = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=rec_dtype,
            device=device,
        )
        if ov > 1:
            new_rec[:, :ov].copy_(old_rec[:, :ov])
        self.recurrent_states = new_rec
        del old_rec

        new_slot = self._alloc_slot_states(num_slots)
        if ov > 1:
            for name, nt in new_slot.items():
                ot = old_slot.get(name)
                if ot is not None:
                    nt[:, :ov].copy_(ot[:, :ov])
        self.slot_states = new_slot
        del old_slot

        # Hand the released segments back to the driver (no expandable_segments on Windows).
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self._num_slots = num_slots
        # The verify snapshots are indexed slot-major over the OLD slot count, so a resize has to
        # rebuild them (their content is one forward deep -- nothing to carry over).
        if self._verify_rows:
            self._alloc_verify()
        # Keep exactly the slots that were already allocated and still exist; everything else
        # (the padding slot aside) is free. Slots dropped by a shrink return to the free-list,
        # and previously-free overlap slots are not handed out a second time.
        allocated = {s for s in range(1, ov) if s not in old_free}
        self._free_slots = [s for s in range(1, num_slots) if s not in allocated]

    def free(self, slots) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        self._free_slots.extend(int(s) for s in slots)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, slots] = spec.fill_value

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers) from slot ``src`` to
        ``dst``. Used for COW-on-restore (donated snapshot -> fresh live slot)."""
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])
        for t in self.slot_states.values():
            t[:, dst].copy_(t[:, src])

    def ensure_verify_rows(self, rows: int, blocks: int) -> None:
        """Give each of the ``blocks`` in-flight requests ``rows`` extra snapshot rows for the MTP
        verify window, so a commit can pick the state of the prefix it accepted instead of
        re-running that prefix. A verify forward writes one snapshot per query row (the recurrent
        kernel does it in-kernel, the conv window alongside), so ``rows`` = draft_len + 1 covers
        every possible cut. Blocks are indexed by position in the batch, not by state slot: at
        most ``max_running_req`` requests are in flight, and the extra slots of the pool exist for
        prefix snapshots, which never need a verify window of their own."""
        rows, blocks = max(1, int(rows)), max(1, int(blocks))
        if self.verify_rec is not None and self._verify_rows >= rows and self._verify_blocks >= blocks:
            return
        self._verify_rows = rows
        self._verify_blocks = blocks
        self._alloc_verify()

    def _alloc_verify(self) -> None:
        n_layers, _, local_v_heads, key_dim, value_dim = self.recurrent_states.shape
        _, _, local_conv_dim, km1 = self.conv_states.shape
        n = self._verify_blocks * self._verify_rows
        # Block-major (blocks, rows, ...): the verify kernel derives the per-request block
        # stride from ``snap_buffer.shape[1]``, so the rows axis MUST sit at dim 1. A flat
        # ``[blocks*rows]`` here would make the kernel stride by the head count and scatter
        # snapshots across blocks (out of bounds once bs > 1). Element mapping inside one
        # snapshot row matches recurrent_states' (the kernel indexes both as
        # ``hv*K*V + o_v*K + o_k``), so ``load_verify_row`` is a plain block copy.
        self.verify_rec = torch.zeros(
            (n_layers, self._verify_blocks, self._verify_rows,
             local_v_heads, key_dim, value_dim),
            dtype=self.recurrent_states.dtype,
            device=self._device,
        )
        self.verify_conv = torch.zeros(
            (n_layers, n, local_conv_dim, km1),
            dtype=self.conv_states.dtype,
            device=self._device,
        )

    def load_verify_row(self, slot: int, block: int, row: int) -> None:
        """Copy the snapshot of ``row`` into the live state of the request holding ``slot``. The
        source is addressed by the request's position in the last verify batch (see
        ensure_verify_rows), its destination by the slot its state lives in."""
        flat = block * self._verify_rows + row
        self.recurrent_states[:, slot].copy_(self.verify_rec[:, block, row])
        self.conv_states[:, slot].copy_(self.verify_conv[:, flat])

    def snapshot_state(self, src: int) -> tuple:
        """Host-RAM copy of one slot's whole-sequence state (conv + recurrent + declared
        slot_states) -- the GDN half of the mm RAM tier. Small enough (~MB) to park next to
        the KV rows without touching the pool's slot accounting."""
        conv = self.conv_states[:, src].cpu().clone()
        rec = self.recurrent_states[:, src].cpu().clone()
        extras = {name: t[:, src].cpu().clone() for name, t in self.slot_states.items()}
        return (conv, rec, extras)

    def restore_state(self, dst: int, snap: tuple) -> None:
        """Write a ``snapshot_state`` payload into slot ``dst`` (host -> device). Runs on the
        engine stream before the first forward that reads ``dst``."""
        conv, rec, extras = snap
        self.conv_states[:, dst].copy_(conv)
        self.recurrent_states[:, dst].copy_(rec)
        for name, t in extras.items():
            if name in self.slot_states:
                self.slot_states[name][:, dst].copy_(t)

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, table_idx] = spec.fill_value

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        for t in self.slot_states.values():
            per += t[:, 0].numel() * t.element_size()
        return int(per)


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig,
    tp_size: int,
    dtype: torch.dtype,
    slot_states: tuple[SlotStateSpec, ...] = (),
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local), plus any
    declared slot_states."""
    n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

    conv_elems = local_conv_dim * (group.conv_kernel_dim - 1)
    rec_elems = local_v_heads * group.key_head_dim * group.value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    total = n_layers * (conv_bytes + rec_bytes)

    for spec in slot_states:
        item = (spec.dtype if spec.dtype is not None else dtype).itemsize
        total += max(1, len(spec.layer_ids)) * math.prod(spec.shape) * item
    return int(total)


def mtp_verify_rows(config) -> int:
    """Snapshot rows per in-flight request that an MTP verify window needs: one per draft row plus
    the bonus row the target contributes at the cut. 0 without MTP."""
    if not getattr(config, "enable_mtp", False):
        return 0
    return max(1, int(getattr(config, "mtp_draft_len", 0) or 0)) + 1


def mtp_verify_bytes(config) -> int:
    """Fixed cost of the verify snapshots: one row-block per in-flight request (the pool's extra
    slots hold prefix snapshots, which never need a verify window of their own)."""
    rows = mtp_verify_rows(config)
    if rows <= 0:
        return 0
    linear_group = config.model_config.linear_attention_group()
    if linear_group is None:
        return 0
    per_row = linear_state_bytes_per_req(
        linear_group, config.tp_info.size, config.dtype
    )
    return int(config.max_running_req) * rows * per_row


__all__ = [
    "LinearStatePool",
    "linear_state_bytes_per_req",
    "fit_linear_pool_slots",
    "mamba_slots_for_growth",
    "mtp_verify_bytes",
    "mtp_verify_rows",
]


def state_pool_bytes(config, num_slots: int | None = None) -> int:
    """Total GDN state-pool bytes at ``num_slots`` PHYSICAL slots (default: the startup
    slot count). The engine adds this to the KV family's fixed cost when budgeting --
    the state pool is a sibling pool, not a KV tier."""
    linear_group = config.model_config.linear_attention_group()
    slot_states = getattr(config.model_config, "slot_states", ())
    if linear_group is None:
        if slot_states:
            raise ValueError("slot_states ride the linear-state slots; model has no linear group")
        return 0
    slots = num_slots if num_slots is not None else _linear_pool_num_slots(config)
    per_req = linear_state_bytes_per_req(
        linear_group, config.tp_info.size, config.dtype, slot_states
    )
    return per_req * slots + mtp_verify_bytes(config)


def _linear_pool_num_slots(config) -> int:
    """LinearStatePool slot count. Hybrid-radix non-evictable peak is 4 slots per running request
    (1 live + 2 ping-pong + 1 committed snapshot locked through decode), plus a cross-request
    snapshot cache and a padding sink; naive GDN keeps the old (max_running_req + 1)."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1  # live + dummy/padding
    ratio = config.linear_state_cache_ratio
    n_cache = max(4, int(ratio * mr))
    return 4 * mr + n_cache + 1  # live + 2 ping-pong + locked committed snapshot + cache + padding


def _linear_pool_min_slots(config) -> int:
    """Floor on LinearStatePool slots that still runs: the non-evictable working set with a
    zero snapshot cache. Hybrid-radix needs 4 per running request (1 live + 2 ping-pong + 1
    committed snapshot locked through decode) + the padding sink; naive needs 1 per request +
    padding. Below this, a full max_running_req batch can't get its slots and admission
    deadlocks -- so a runtime rebuild rejects a smaller request."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1
    return 4 * mr + 1


def mamba_slots_for_growth(config, cur_pages: int, new_pages: int) -> int | None:
    """GDN state-pool usable-slot target when a budget slide grew KV from ``cur_pages`` to
    ``new_pages``.

    Only the hybrid-radix cross-request snapshot cache (the portion of ``_linear_pool_num_slots``
    above ``_linear_pool_min_slots``) is elastic; the non-evictable working set is fixed by
    ``max_running_req``. A larger KV pool retains more simultaneous prefixes, so the snapshot
    cache is scaled by the page-growth ratio (capped at 2x the startup cache) to keep prefix
    reuse from evicting/recycling live GDN snapshots -- a recycled snapshot restored into a fresh
    live slot gives the linear-attention recurrence stale prior state, which degenerates long
    reasoning turns into repetition. Returns None when there is nothing elastic to grow (dense /
    naive, or a KV shrink), so the caller leaves the pool at its current size."""
    startup = _linear_pool_num_slots(config)
    floor = _linear_pool_min_slots(config)
    elastic0 = startup - floor
    if elastic0 <= 0:
        return None  # no snapshot cache to scale (dense / naive working-set-only pool)
    ratio = new_pages / max(1, cur_pages)
    if ratio <= 1.0:
        return None  # shrink keeps the working set; nothing to lift the snapshot cache onto
    new_elastic = math.ceil(min(2.0, ratio) * elastic0)
    return floor + new_elastic - 1  # physical minus the padding sink -> usable count


# Fraction of the free VRAM measured right before the GDN pool is allocated that the pool may
# take. The remainder is left for CUDA-graph capture scratch + activations that run afterwards;
# filling 100% of the measured free reliably OOMs the later graph capture.
_GDN_FIT_HEADROOM = 0.90


def fit_linear_pool_slots(config, free_bytes: int) -> int:
    """Largest LinearStatePool slot count that fits in ``free_bytes`` of free VRAM, clamped to
    the [non-evictable floor, startup default] range.

    Shrinks ONLY the cross-request snapshot cache (the portion of the slot count above the
    floor): that cache is a pure reuse optimization and dropping it never affects correctness,
    it just reduces cross-request snapshot hits. The floor is what admission needs to run, so
    if even it does not fit this raises with an actionable message instead of OOMing on the
    (unfit) allocation.

    ``free_bytes`` must be the free VRAM measured AFTER the model weights, MoE banks, KV pool
    and any MTP bank are resident -- the caller passes ``self._sync_get_memory()[0]`` taken at
    the point of no return just before the pool tensor is allocated.
    """
    default = _linear_pool_num_slots(config)
    minimum = _linear_pool_min_slots(config)
    per_req = linear_state_bytes_per_req(
        config.model_config.linear_attention_group(),
        config.tp_info.size, config.dtype,
        getattr(config.model_config, "slot_states", ()),
    )
    if per_req <= 0:
        return default
    # The verify snapshots ride the same free VRAM as the slots, so they come off the top.
    usable = int(free_bytes * _GDN_FIT_HEADROOM) - mtp_verify_bytes(config)
    fit = usable // per_req
    if fit < minimum:
        raise MemoryError(
            f"GDN (GatedDeltaNet) state pool needs at least {minimum} slots "
            f"(~{minimum * per_req / 1e9:.2f} GiB) but only "
            f"{usable / 1e9:.2f} GiB of free VRAM is available. Lower --max-running-req, "
            f"reduce --moe-cache-size / --num-tokens, or set FREETOKEN_MAMBA_SSM_DTYPE=bfloat16 "
            f"to halve the recurrent (SSM) state."
        )
    return max(minimum, min(default, fit))
