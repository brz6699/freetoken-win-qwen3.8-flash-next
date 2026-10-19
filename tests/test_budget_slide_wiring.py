"""Scheduler-side wiring smoke for the budget slider (no CUDA needed; run standalone)."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from freetoken.scheduler.scheduler import Scheduler  # noqa: E402


class _Pool:
    @classmethod
    def kv_cost(cls, config):
        # (cache_per_page, fixed, page_tokens, min_reserve_tokens)
        return (8192, 0, 64, 4096)


def _self(**cfg_over):
    cfg = dict(
        budget_slide=True, num_page_override=None, num_token_override=None,
        kv_reserve_tokens=8192,
        # Real EngineConfig always carries max_seq_len (engine/config.py); the
        # dense-branch ceiling in _build_budget_slider derives from it.
        max_seq_len=131072,
    )
    cfg.update(cfg_over)
    eng = SimpleNamespace(_pool_cls=_Pool, moe_offload_cache=None, num_pages=1000)
    geo = lambda: {"num_pages": 1000, "moe_cache_size": None, "num_mamba_slots": None, "num_swa_pages": None}  # noqa: E731
    return SimpleNamespace(
        config=SimpleNamespace(**cfg), engine=eng, _current_cache_geometry=geo,
        _kv_demand_tokens=0, _kv_page_tokens=1,
    )


def test_slider_constructed_and_disabled_paths():
    s = Scheduler._build_budget_slider(_self())
    assert s is not None
    assert Scheduler._build_budget_slider(_self(budget_slide=False)) is None
    # Declared sizes are starting points, not a freeze: the slider stays live.
    assert Scheduler._build_budget_slider(_self(num_token_override=32768)) is not None


def test_tick_queues_a_rebuild_at_idle():
    self = _self()
    self._pending_rebuild = None
    self._budget_slider = Scheduler._build_budget_slider(self)
    # High occupancy vs a small current pool -> grow KV (dense branch: num_pages only).
    self._budget_slider.observe(900, 1000)
    Scheduler._maybe_queue_budget_slide(self)
    msg = self._pending_rebuild
    assert msg is not None and msg.request_id == "budget-slide"
    assert msg.num_pages > 1000 and msg.moe_cache_size is None
    # A second tick must not stack on top of a pending rebuild.
    Scheduler._maybe_queue_budget_slide(self)
    assert self._pending_rebuild is msg


def test_deadband_leaves_nothing_queued():
    self = _self()
    self._pending_rebuild = None
    self._budget_slider = Scheduler._build_budget_slider(self)
    self._budget_slider.observe(800, 1000)  # matches cur occupancy -> within deadband
    Scheduler._maybe_queue_budget_slide(self)
    assert self._pending_rebuild is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("budget_slide wiring: all assertions passed")
