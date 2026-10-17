"""Pure-policy checks for engine.budget_slide (no device needed; run standalone)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from freetoken.engine.budget_slide import BudgetSlider, propose_targets  # noqa: E402

GEOM = dict(
    cache_per_page=8192,
    per_expert_bytes=100000,
    num_experts=1280,
    total_experts=6400,
    max_slots=6400,
    prefill_overlap=False,
)


def test_grow_kv_borrows_from_expert_slots():
    out = propose_targets(
        ema_used_pages=950, cur_pages=1000, cur_moe=5120, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, **GEOM,
    )
    assert out is not None
    # desired pages = ceil(950*1.25)=1188 -> delta 188 pages ~= 15 slot-steps
    assert out["num_pages"] == 1183 and out["moe_cache_size"] == 5105


def test_shrink_kv_gives_experts_back():
    out = propose_targets(
        ema_used_pages=200, cur_pages=1000, cur_moe=5120, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, **GEOM,
    )
    assert out is not None
    assert out["num_pages"] < 1000 and out["moe_cache_size"] > 5120
    # Byte envelope stays ~constant: freed-by-KV ~= spent-on-experts.
    kv_freed = (1000 - out["num_pages"]) * GEOM["cache_per_page"]
    moe_spent = (out["moe_cache_size"] - 5120) * GEOM["per_expert_bytes"]
    assert abs(kv_freed - moe_spent) <= GEOM["cache_per_page"] + GEOM["per_expert_bytes"]


def test_slot_floor_and_cap_clamp():
    # Floor: experts may not drop below num_experts even when KV wants more bytes.
    out = propose_targets(
        ema_used_pages=1500, cur_pages=1000, cur_moe=1300, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, **GEOM,
    )
    assert out == {"num_pages": 1244, "moe_cache_size": 1280}
    # Cap (marlin-style): at the ceiling a further KV shrink cannot add slots.
    capped = dict(GEOM, max_slots=992)
    out = propose_targets(
        ema_used_pages=200, cur_pages=1000, cur_moe=992, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, **capped,
    )
    assert out is None  # both sides at their bounds -> nothing worth doing


def test_overlap_floor_is_two_layers():
    geom = dict(GEOM, prefill_overlap=True)
    out = propose_targets(
        ema_used_pages=1500, cur_pages=1000, cur_moe=2600, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, **geom,
    )
    assert out["moe_cache_size"] == 2 * GEOM["num_experts"]


def test_kv_floor_and_dense_branch_and_deadband():
    # KV never shrinks below its own reserve floor.
    out = propose_targets(
        ema_used_pages=1, cur_pages=1000, cur_moe=5120, min_kv_pages=300,
        headroom=0.25, min_step_ratio=0.05, **GEOM,
    )
    assert out["num_pages"] >= 300
    # Dense branch: no slot cache -> KV-only resize.
    out = propose_targets(
        ema_used_pages=950, cur_pages=1000, cur_moe=None, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, cache_per_page=8192, per_expert_bytes=1,
        num_experts=1, total_experts=1, max_slots=1, prefill_overlap=False,
    )
    assert out == {"num_pages": 1188}
    # Deadband: within-step changes return None (no teardown for noise).
    out = propose_targets(
        ema_used_pages=800, cur_pages=1000, cur_moe=5120, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, **GEOM,
    )
    assert out is None


def test_slider_debounce_and_reset():
    s = BudgetSlider(min_kv_pages=100, **GEOM)
    assert s.propose(1000, 5120) is None  # no sample yet
    s.observe(950, 1000)
    out = s.propose(1000, 5120)
    assert out is not None and out["num_pages"] > 1000
    s.observe(950, 1000)
    assert s.propose(out["num_pages"], out["moe_cache_size"]) is None  # interval gate
    s._last_move = 0.0
    assert s.propose(out["num_pages"], out["moe_cache_size"]) is None  # ema reset on apply
    # Admission demand re-arms the smoother; force skips the interval gate.
    s.observe_need(20000)
    out2 = s.propose(out["num_pages"], out["moe_cache_size"], force=True)
    assert out2 is not None and out2["num_pages"] > out["num_pages"]
    assert out2["moe_cache_size"] < out["moe_cache_size"]


def test_declared_sizes_fit_measured_budget():
    from freetoken.engine.cache_budget import fit_declared_cache_sizes

    kw = dict(
        budget_bytes=200_000_000, per_expert_bytes=100_000, cache_per_page=8192,
        kv_reserve_pages=100, num_experts=1280, total_experts=6400, max_slots=6400,
        prefill_overlap=False,
    )
    # Declared over affordable: experts fill to what the budget leaves after the KV
    # floor; KV takes the remainder.
    moe, pages = fit_declared_cache_sizes(declared_moe=5120, declared_pages=None, **kw)
    assert moe == 1991 and pages == 109
    # Declared below the floors: floors win (working set must stay runnable).
    moe, pages = fit_declared_cache_sizes(declared_moe=50, declared_pages=None, **kw)
    assert moe == 1280 and pages > 100
    # Declared KV over affordable: clamped, never below the reserve.
    moe, pages = fit_declared_cache_sizes(declared_moe=5120, declared_pages=400, **kw)
    assert moe == 1991 and pages == 109
    # Even both floors together do not fit -> loud failure, not an under-sized cache.
    try:
        fit_declared_cache_sizes(declared_moe=5120, declared_pages=None, budget_bytes=100_000_000, **{
            k: v for k, v in kw.items() if k != "budget_bytes"})
    except ValueError:
        pass
    else:
        raise AssertionError("too-small budget must raise, not silently under-size")


def test_demand_page_floor_is_shrink_only():
    # MoE path: an idle shrink toward EMA 0 is floored at the recent prompt's
    # footprint (150 pages) instead of sliding to the KV floor.
    out = propose_targets(
        ema_used_pages=0.0, cur_pages=200, cur_moe=5120, min_kv_pages=10,
        headroom=0.25, min_step_ratio=0.05, demand_pages=150, **GEOM,
    )
    assert out is not None and out["num_pages"] == 150
    # The bytes the shrink freed from KV went to the expert slots, envelope ~constant
    # within one page + one slot (same band test_shrink_kv_gives_experts_back uses; the
    # exact slot count is a floor-division detail, not the invariant).
    assert out["moe_cache_size"] > 5120
    kv_freed = (200 - out["num_pages"]) * GEOM["cache_per_page"]
    moe_spent = (out["moe_cache_size"] - 5120) * GEOM["per_expert_bytes"]
    assert abs(kv_freed - moe_spent) <= GEOM["cache_per_page"] + GEOM["per_expert_bytes"]
    # Demand above the pool is capped at cur_pages: it prevents a shrink, never
    # forces a grow -> nothing to move.
    out = propose_targets(
        ema_used_pages=0.0, cur_pages=1000, cur_moe=5120, min_kv_pages=100,
        headroom=0.25, min_step_ratio=0.05, demand_pages=5000, **GEOM,
    )
    assert out is None
    # The MoE-trade byte round-trip can overshoot the shrink (floor-division lands new_pages
    # below the demand footprint); the re-assert snaps it back to the demand floor.
    out = propose_targets(
        ema_used_pages=0.0, cur_pages=300, cur_moe=5120, min_kv_pages=10,
        headroom=0.25, min_step_ratio=0.05, demand_pages=200, **GEOM,
    )
    assert out is not None and out["num_pages"] == 200 and out["moe_cache_size"] >= GEOM["num_experts"]
    # Dense path: the same floor applies with the MoE term inert.
    out = propose_targets(
        ema_used_pages=1.0, cur_pages=200, cur_moe=None, min_kv_pages=10,
        headroom=0.25, min_step_ratio=0.05, demand_pages=180, cache_per_page=8192,
        per_expert_bytes=1, num_experts=1, total_experts=1, max_slots=1,
        prefill_overlap=False,
    )
    assert out == {"num_pages": 180}
    out = propose_targets(
        ema_used_pages=1.0, cur_pages=200, cur_moe=None, min_kv_pages=10,
        headroom=0.25, min_step_ratio=0.05, demand_pages=900, cache_per_page=8192,
        per_expert_bytes=1, num_experts=1, total_experts=1, max_slots=1,
        prefill_overlap=False,
    )
    assert out is None  # capped at cur_pages -> no-op


def test_slider_demand_passthrough():
    s = BudgetSlider(min_kv_pages=10, **GEOM)
    s._ema = 0.0
    # The demand floor handed to propose() holds the pool at the recent prompt's footprint.
    out = s.propose(300, 5120, force=True, demand_pages=250)
    assert out is not None and out["num_pages"] == 250
    # propose() resets the smoother on an accepted move; re-seed it WITHOUT the per-call
    # demand floor and the same state now slides to the KV floor (the floor is per-call).
    s._ema = 0.0
    out2 = s.propose(300, 5120, force=True)
    assert out2 is not None and out2["num_pages"] < 250


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("budget_slide policy: all assertions passed")
