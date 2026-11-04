"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import align_ceil, div_ceil

if TYPE_CHECKING:
    import torch

# Physical-ceiling fraction used when --memory-ratio is omitted. With no ratio declared the
# ceiling is simply "everything still free after weights" (ratio 1.0). The arena takes only the
# kv+moe footprint, so the CUDA-graph / activation headroom is whatever VRAM the arena leaves
# unused -- consumed and self-sized by the live capture rather than a guessed fraction. The ratio
# only matters when explicitly given, where its (1-ratio) remainder is the headroom by definition.
DEFAULT_MEMORY_RATIO = 1.0

def effective_ratio(memory_ratio: float | None) -> float:
    """Resolve a possibly-None memory_ratio into the concrete ceiling fraction to apply."""
    return DEFAULT_MEMORY_RATIO if memory_ratio is None else float(memory_ratio)


def layer_slot_bytes(sources: dict[str, "list[torch.Tensor]"], row_align: int = 16) -> list[int]:
    """Bytes one expert slot of EACH layer occupies on GPU: per-layer row bytes summed over banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is already TP-sharded
    upstream, so the per-row byte count is the per-rank slot size for one layer. ``row_align`` is
    the lcm of the active kernels' READ-contract atoms (``MoEKernel.row_align_bytes``); each layer's
    summed bytes are ROUND_UP-ed to it so the arena's gcd(cpp, peb) lattice lands on points every
    backend reads cleanly without misaligned tile loads.
    """
    num_layers = len(next(iter(sources.values())))
    return [
        align_ceil(sum(t[l][0].numel() * t[l].element_size() for t in sources.values()), row_align)
        for l in range(num_layers)
    ]


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"], row_align: int = 16) -> int:
    """Bytes one expert slot occupies on GPU: mean over layers when bank layouts differ, then aligned.

    ``row_align`` (lcm of active kernels' READ-contract atoms) is applied BOTH inside per-layer
    sums AND to the final rounded mean, so the value the engine and budget_slide agree on is an
    atom-multiple by construction regardless of how uneven the layer bank layouts are.
    """
    per_layer = layer_slot_bytes(sources, row_align)
    return align_ceil(-(-sum(per_layer) // len(per_layer)), row_align)


def net_cache_budget_bytes(
    memory_ratio: float | None, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools.

    With an explicit ``memory_ratio``, the budget is ``ratio x baseline`` minus weights and
    fixed (non-paged) cache; the ``(1-ratio)`` remainder is the CUDA-graph/activation
    headroom. With ``memory_ratio=None`` (the default), sizing follows Algorithm 2: declared
    KV + MoE sizes are authoritative, so the ceiling is the full post-weights free VRAM
    (equivalent to ratio=1.0 here) and the fit step clamps each pool to its declared size
    inside that ceiling. Single source of truth for startup auto-sizing and the runtime
    rebuild fit check.
    """
    ratio = effective_ratio(memory_ratio)
    return int(ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )


def fit_declared_cache_sizes(
    *,
    declared_moe: int,
    declared_pages: int | None,
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    kv_reserve_pages: int,
    num_experts: int,
    total_experts: int,
    max_slots: int,
    prefill_overlap: bool,
) -> tuple[int, int]:
    """Fit DECLARED cache sizes into the MEASURED byte budget, respecting declared when it fits.

    Both MoE slots and KV pages are treated as USER-DECLARED ceilings (not targets to grow into):
    when ``(declared_moe x peb) + (declared_pages x cpp)`` fits inside ``budget_bytes``, the boot
    geometry equals the declaration exactly and the leftover VRAM is left for graph capture +
    activations. When the declaration exceeds ``budget_bytes``, it is clamped DOWN to the largest
    sub-fit that keeps the KV-first ordering -- never raised above what the user asked for. The
    MoE side then fills whatever headroom the KV clamp leaves, down to the kernel floor of one
    FULL expert layer (``num_experts``, or 2x with prefill overlap). Envelope-conserving:
    ``moe * peb + pages * cpp <= budget_bytes``."""
    assert per_expert_bytes > 0 and cache_per_page > 0

    hi = min(total_experts, max_slots)
    # MoE floor = one full layer's experts (num_experts, or 2x with prefill overlap): the offload
    # MoE kernel hard-requires >= num_experts resident slots to run a GEMM. Always reserved first.
    floor = min(hi, max(num_experts, 2 * num_experts if prefill_overlap else num_experts))
    floor_bytes = floor * per_expert_bytes
    kv_reserve_bytes = kv_reserve_pages * cache_per_page

    # Smallest possible plan: MoE floor + KV hard reserve. Anything smaller than this cannot run
    # at all -- raise before clamping so the failure explains WHICH knob to turn.
    min_needed = floor_bytes + kv_reserve_bytes
    if budget_bytes < min_needed:
        raise ValueError(
            f"cache budget too small: minimum plan (MoE floor={floor} slots x {per_expert_bytes} B"
            f" + KV reserve={kv_reserve_pages} pages x {cache_per_page} B) needs {min_needed} B"
            f" > budget {budget_bytes} B (raise --memory-ratio, lower --kv-reserve-tokens, or free VRAM)"
        )

    # Clamp declared KV down so MoE floor still fits. ``kv_cap >= kv_reserve_pages`` follows from
    # the min_needed check above, so the max(...) below never contradicts the cap.
    declared_kv = declared_pages or 0
    kv_cap = (budget_bytes - floor_bytes) // cache_per_page
    kv_pages = max(kv_reserve_pages, min(declared_kv, kv_cap))
    kv_bytes = kv_pages * cache_per_page

    # MoE takes the remainder, capped at the declared size: declared is the ceiling, the runtime
    # budget_slide can grow later only if KV idle hands bytes back inside the SAME envelope.
    afford_moe = (budget_bytes - kv_bytes) // per_expert_bytes
    moe = max(floor, min(declared_moe, hi, afford_moe))
    return moe, kv_pages
