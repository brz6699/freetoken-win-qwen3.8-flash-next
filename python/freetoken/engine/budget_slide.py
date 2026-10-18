"""Runtime KV <-> MoE budget sliding, layered on top of the static startup plan.

Startup (cache_budget.plan_cache_budget) splits the VRAM headroom ONCE, MoE-first.
This module turns that snapshot into a live balance: KV page occupancy is sampled
every batch, and when a sustained shift is visible, the byte envelope is rebalanced
between the KV page pool and the MoE expert-slot cache -- KV grows toward its
smoothed occupancy (+headroom) by giving expert slots back, and shrinks back by
giving them more. The rebalance rides the EXISTING runtime rebuild path
(CacheRebuildBackendMsg -> scheduler._pending_rebuild), so it only lands at the
scheduler's idle safe point and inherits its reject/rollback protections.

Model-agnostic by construction: every quantity is handed in as plain integers by
the scheduler (bytes-per-page from the pool class's kv_cost, bytes-per-slot from
the offload cache's bank schema, expert counts from the cache geometry), so the
QSA hybrid, plain-MoE (GLM-class) and dense no-cache-model shapes all take the
same arithmetic. Models without an offload cache keep the dense branch: KV-only
resize with the MoE term inert. Pure arithmetic -- unit-testable without a
device, mirroring cache_budget.py's no-GPU style.
"""

from __future__ import annotations

import math
import os
import time

# KV ladder tiers (fractions of the declared KV ceiling). The decision layer snaps the
# smoothed demand UP to the smallest tier that still covers it, so a growing agent session
# crosses a tier boundary a handful of times instead of continuously. This matters because
# EVERY accepted rebuild wipes the prefix tree (stale page indices from the old geometry);
# continuous occupancy-following therefore wipes the tree every few turns and the prefix
# hit ratio never climbs off zero. On tiers, settled load proposes desired == current and
# the proposal gate returns None. --kv-tiers N (N evenly spaced tiers, translated into the
# FREETOKEN_KV_TIERS percentage list here) overrides the fractions; a single 100 disables
# the ladder (plain follow).
_TIER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _tier_ratios() -> tuple[float, ...]:
    raw = os.getenv("FREETOKEN_KV_TIERS", "").strip()
    if raw:
        try:
            vals = tuple(sorted({max(0.05, float(x) / 100.0) for x in raw.split(",") if x.strip()}))
            if vals:
                return vals
        except ValueError:
            pass
    return _TIER_FRACTIONS


def _snap_to_tier(pages: int, max_pages: int) -> int:
    """Smallest ladder tier (ceil of ratio x ceiling) that still covers ``pages``."""
    for ratio in _tier_ratios():
        tier = math.ceil(max_pages * ratio)
        if pages <= tier:
            return tier
    return max_pages


def propose_targets(
    *,
    ema_used_pages: float,
    cur_pages: int,
    cur_moe: int | None,
    cache_per_page: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    max_slots: int,
    min_kv_pages: int,
    headroom: float,
    min_step_ratio: float,
    prefill_overlap: bool,
    demand_pages: int = 0,
    max_kv_pages: int | None = None,
    allow_small: bool = False,
    arena_bytes: int | None = None,
    kv_fixed_bytes: int = 0,
) -> dict | None:
    """Rebalance (num_pages, moe_cache_size) inside the CURRENT byte envelope.

    Desired KV size = smoothed occupancy with proportional headroom, floored at the
    model's own minimum KV reserve and at ``demand_pages`` -- a demand-aware floor
    the scheduler sizes from what will still need the pool (a high-water mark of the
    admitted prompt footprints, since agent turns only grow), CAPPED at cur_pages so it
    only ever prevents a shrink, never forces a grow (admission's grow-before-reject owns
    growth). Without it the idle shrink slides the pool back below the last agent prompt's
    footprint, the next prompt forces a grow rebuild at admission, and BOTH rebuilds wipe
    the prefix tree -- the measured cause of every big turn re-prefilling cold. The
    byte delta between desired and current pages moves one-for-one through the
    expert-slot cache (total bytes stay constant), clamped by the slot
    floor/ceiling that ``OffloadMoeCache.validate_rebuild`` enforces -- the floor
    is ``num_experts`` normally, ``2*num_experts`` while prefill overlap is on
    (its double buffer needs a full extra layer-set). Returns ``rebuild_cache``
    kwargs, or None when the move is too small to be worth a teardown (the slider
    debounces on top of this gate).
    """
    desired_pages = max(min_kv_pages, math.ceil(ema_used_pages * (1.0 + headroom)))
    # Shrink-only floor: never slide below what the recent workload will ask back.
    desired_pages = max(desired_pages, min(int(demand_pages), cur_pages))
    if max_kv_pages is not None:
        # Grow KV to demand but cap at the declared --num-tokens: the slide may hand MoE
        # bytes back as context grows, yet never promises more KV than was asked for.
        desired_pages = min(desired_pages, max_kv_pages)
        # Snap UP to the ladder tier covering the demand: repeated samples inside one tier
        # band re-propose the SAME target, so the equal-target gate below quiets them and
        # the tree-wiping rebuild only lands when a tier is actually crossed.
        desired_pages = _snap_to_tier(desired_pages, int(max_kv_pages))
    if cur_moe is None:
        # No slot cache to trade with (dense / owned cache): KV-only resize. The pool still
        # carves the shared arena (left-anchored), so the proposal respects the boundary.
        if arena_bytes and cache_per_page > 0:
            desired_pages = min(
                desired_pages,
                (arena_bytes - kv_fixed_bytes) // cache_per_page,
            )
        if desired_pages == cur_pages:
            return None
        if not allow_small and abs(desired_pages - cur_pages) < max(1, round(cur_pages * min_step_ratio)):
            return None
        return {"num_pages": int(desired_pages)}

    hi = min(total_experts, max_slots)
    floor = min(hi, max(num_experts, 2 * num_experts if prefill_overlap else num_experts))
    # Slide in EXPERT-SLOT units. cpp and peb both sit on the gcd(cpp, peb) byte lattice,
    # so any (pages, slots) trade moves whole-lattice bytes; the integer division below
    # truncates the trade in the SHRINKING direction, so the (moe, pages) byte total never
    # exceeds the current geometry's -- the drift is bounded (at most one page's bytes per
    # slide) and does NOT accumulate, because every proposal re-derives from the CURRENT
    # geometry. (The old lcm(cpp, peb) step assumed the two unit sizes share large common
    # factors; GLM-class latent-KV geometries put a 0.2 MiB page against a 13.5 MiB slot --
    # nearly coprime, lcm 747 GiB vs a 7 GiB envelope -- so the chunk count rounded to
    # zero on every proposal and the slider never moved for that family. Slot units fixed
    # that; the min-step gate below keeps every move worth at least several MoE slots.)
    delta_slots = round((desired_pages - cur_pages) * cache_per_page / per_expert_bytes)
    new_moe = max(floor, min(cur_moe - delta_slots, hi))
    # Pages follow what the MoE side ACTUALLY afforded after clamping, so the byte
    # envelope stays constant even when the slot floor/cap truncates the trade.
    new_pages = cur_pages + (cur_moe - new_moe) * per_expert_bytes // cache_per_page
    if new_pages < min_kv_pages:
        # Keep the KV floor exact and let the slot count absorb the rounding:
        # price the extra pages back against the same envelope.
        new_pages = min_kv_pages
        new_moe = min(
            hi, max(floor, cur_moe - (new_pages - cur_pages) * cache_per_page // per_expert_bytes)
        )
    # The slot floor can truncate the trade below the demand floor: re-assert it (the
    # cap at cur_pages keeps it shrink-only even when the demand exceeds the pool).
    demand_pages = min(int(demand_pages), cur_pages)
    if new_pages < demand_pages:
        new_pages = demand_pages
        new_moe = min(
            hi, max(floor, cur_moe - (new_pages - cur_pages) * cache_per_page // per_expert_bytes)
        )
    if max_kv_pages is not None and new_pages > max_kv_pages:
        # The slot delta rounds UP, so the back-derived page count can overshoot the
        # user's declared KV ceiling by a page or two (e.g. 333 slots = 2049.1 pages of
        # trade against a 2048-page gap to the cap). The ceiling is a hard contract --
        # users size max-tokens to the model's acceptance -- so clamp it back; MoE
        # keeps the bytes the extra pages would have taken.
        new_moe = min(
            hi,
            max(
                floor,
                new_moe + (new_pages - max_kv_pages) * cache_per_page // per_expert_bytes,
            ),
        )
        new_pages = max_kv_pages
    if arena_bytes:
        # Hard envelope: both pools carve the ONE shared cache arena (KV left-anchored,
        # MoE banks right-anchored), allocated once at cold start and never grown. A
        # target whose (moe, pages) footprint exceeds it aliases the two carves onto
        # each other (corrupted experts/KV -> garbled output), so the proposal itself
        # must land inside the boundary: the MoE half (the elastic side) absorbs the
        # excess first; if the slot floor still overflows, the page count gives ground
        # back so the pair sits exactly on the boundary.
        def _fits(moe, pages):
            return (
                moe * per_expert_bytes + pages * cache_per_page + kv_fixed_bytes
                <= arena_bytes
            )

        if per_expert_bytes > 0 and not _fits(new_moe, new_pages):
            new_moe = max(
                floor,
                min(
                    new_moe,
                    (arena_bytes - kv_fixed_bytes - new_pages * cache_per_page)
                    // per_expert_bytes,
                ),
            )
            if not _fits(new_moe, new_pages):
                new_pages = max(
                    min_kv_pages,
                    (arena_bytes - kv_fixed_bytes - new_moe * per_expert_bytes)
                    // cache_per_page,
                )
                new_moe = max(
                    floor,
                    min(
                        new_moe,
                        (arena_bytes - kv_fixed_bytes - new_pages * cache_per_page)
                        // per_expert_bytes,
                    ),
                )
    if not allow_small and (
        abs(new_moe - cur_moe) < max(4, round(cur_moe * min_step_ratio))
        and abs(new_pages - cur_pages) < max(1, round(cur_pages * min_step_ratio))
    ):
        return None
    if new_pages == cur_pages and new_moe == cur_moe:
        # A no-op "rebuild" still costs a full teardown (prefix-cache wipe) for zero resize --
        # e.g. a prompt that exceeds the max_kv_pages cap while the pool is already at the cap.
        # Skip it so the caller just falls through to its normal context_length_exceeded reject
        # instead of bouncing the cache and forcing the next request to re-prefill cold.
        return None
    return {"num_pages": int(new_pages), "moe_cache_size": int(new_moe)}


class BudgetSlider:
    """Asymmetric-smoothed KV occupancy sampler + rebalancer for one scheduler.

    Growth is tracked fast (alpha 0.1) so admission pressure shows up within a few
    batches; shrink is tracked slow (alpha 0.02) so a brief gap in traffic does not
    pull KV down only for the next burst to push it back -- the asymmetric pair is
    what makes the two-sided slide converge instead of oscillating. ``propose`` is
    handed the CURRENT geometry (it reads no device state), debounces to at most one
    move per ``min_interval`` seconds, and resets the smoother on every accepted
    move so one sustained load level produces exactly one resize.
    """

    def __init__(
        self,
        *,
        min_kv_pages: int,
        headroom: float = 0.25,
        min_interval: float = 30.0,
        min_step_ratio: float = 0.05,
        page_tokens: int = 1,
        **geometry,
    ) -> None:
        self._min_kv_pages = min_kv_pages
        self._headroom = headroom
        self._min_interval = min_interval
        self._min_step_ratio = min_step_ratio
        self._page_tokens = max(1, int(page_tokens))
        self._geometry = geometry
        self._ema: float | None = None
        self._last_move = 0.0
        self._low_streak = 0

    def observe(self, used_pages: int, total_pages: int) -> None:
        """Feed one KV occupancy sample (pages, non-evictable semantics).

        Growth stays fast (alpha 0.1) so admission pressure shows up within a few batches.
        Shrink is slow on a single low sample (alpha 0.02) so a brief gap does not pull KV
        down prematurely, but once low occupancy is CONFIRMED across a short streak it speeds
        up (alpha 0.15) so the pool hands bytes back to the MoE cache before the next burst
        arrives -- otherwise a long-context round leaves KV oversized and the expert cache
        starved into round two. The streak requirement keeps the anti-blip property a single
        slow alpha would otherwise lose."""
        if self._ema is None:
            self._ema = float(used_pages)
            return
        if used_pages >= self._ema:
            alpha = 0.1
            self._low_streak = 0
        else:
            self._low_streak += 1
            alpha = 0.15 if self._low_streak >= 4 else 0.02
        self._ema += alpha * (used_pages - self._ema)

    def observe_need(self, tokens: int) -> None:
        """Admission rejection is hard evidence of KV pressure: the request never
        became a batch, so no observe() sample will ever describe it. Jump the
        smoother to at least the page footprint this request would have needed."""
        needed = -(-tokens // self._page_tokens)
        self._ema = float(needed) if self._ema is None else max(self._ema, float(needed))

    def propose(
        self,
        cur_pages: int,
        cur_moe: int | None,
        *,
        force: bool = False,
        demand_pages: int = 0,
        allow_small: bool = False,
    ) -> dict | None:
        """Next rebalance as rebuild_cache kwargs, or None when not due/not worth it.

        ``force`` skips the interval debounce: used by the grow-before-reject admission
        path, where the demand is a concrete pending request rather than a sampled
        drift, so waiting out the interval only costs an unnecessary bounce.
        ``demand_pages`` is a shrink-only floor (capped at ``cur_pages`` inside
        ``propose_targets``) so an idle shrink never slides the pool below what the
        recent workload will ask back at the next admission. ``allow_small`` also skips
        the min-step size gate: a forced admission grow must land even when it is only a
        few pages, so that a prompt that just barely exceeds the current ceiling is not
        dropped because the grow-back would be "too small to be worth a teardown"."""
        now = time.monotonic()
        if self._ema is None:
            return None
        if not force and now - self._last_move < self._min_interval:
            return None
        out = propose_targets(
            ema_used_pages=self._ema,
            cur_pages=cur_pages,
            cur_moe=cur_moe,
            min_kv_pages=self._min_kv_pages,
            headroom=self._headroom,
            min_step_ratio=self._min_step_ratio,
            demand_pages=demand_pages,
            allow_small=allow_small,
            **self._geometry,
        )
        if out is None:
            return None
        self._last_move = now
        self._ema = None  # re-measure the new steady state under the new geometry
        return out
