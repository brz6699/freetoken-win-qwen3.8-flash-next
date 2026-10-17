"""CPU smoke test for the shared cache arena (no GPU needed; run standalone).

Verifies the pipe-model invariants: the KV side left-anchors at arena offset 0 (block
base address stable, first slab byte-stable across grow/shrink), the QSA index tiers pack right after
the KV cursor, the MoE bank slot-caches right-anchor at the arena end, the two sides'
cursors never cross on a shared arena, and an undersized arena fails loud.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from freetoken.distributed.info import set_tp_info  # noqa: E402
from freetoken.kvcache.mha_pool import MHAKVCache  # noqa: E402
from freetoken.kvcache.qsa_pool import QSAKVCache  # noqa: E402
from freetoken.moe.offload_cache import OffloadMoeCache  # noqa: E402

set_tp_info(0, 1)
CPU = torch.device("cpu")
DT = torch.bfloat16


def off(t: torch.Tensor, arena: torch.Tensor) -> int:
    return t.data_ptr() - arena.data_ptr()


def make_moe(cache_size, arena, num_experts=16, num_layers=2, hidden=8, inter=16):
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=CPU,
        quant_format="bf16",
        arena=arena,
    )
    gu = [torch.zeros((num_experts, 2 * inter, hidden), dtype=DT) for _ in range(num_layers)]
    dn = [torch.zeros((num_experts, hidden, inter), dtype=DT) for _ in range(num_layers)]
    cache.set_bank_sources({"gate_up": gu, "down": dn})
    return cache


def test_mha_left_anchor_stable():
    L, H, D, P, PS = 2, 4, 8, 4, 4
    arena = torch.empty(1 << 20, dtype=torch.uint8, device=CPU)
    pool = MHAKVCache(H, L, D, P, PS, DT, CPU, arena=arena)
    assert pool._kv_buffer.data_ptr() == arena.data_ptr(), "KV side must start at offset 0"
    assert pool.k_cache(0).data_ptr() == arena.data_ptr()
    assert off(pool.v_cache(0), arena) == L * P * PS * H * D * 2
    # Sentinel on the K slab's storage layer 0, pages 0..1 -- the only slab whose offset
    # is independent of the page count (arena byte 0). The V slab and layer-1+ slabs sit
    # at offsets that scale with P, so a re-slice shifts their views (content across a
    # rebuild is not the pool's contract -- see MHAKVCache.rebuild's docstring).
    pool._kv_buffer[0, 0, :2].fill_(1.0)
    sent = pool._kv_buffer[0, 0, :2].clone()
    pool.rebuild(2 * P)  # grow
    assert pool._kv_buffer.data_ptr() == arena.data_ptr(), "grow must not move the KV base"
    assert torch.equal(pool._kv_buffer[0, 0, :2], sent), "grow keeps the first slab's rows"
    pool.rebuild(1)  # shrink below the sentinel region
    assert pool._kv_buffer.data_ptr() == arena.data_ptr()
    assert torch.equal(pool._kv_buffer[0, 0, :1], sent[:1]), "shrink keeps page 0"
    print("  mha left-anchor: block base stable; first slab byte-stable across grow/shrink  OK")


def test_qsa_tiers_pack_after_kv():
    L, H, D, P, PS = 2, 2, 16, 8, 16
    ratio, idx_dim, idx_layers, slots = 4, 8, 2, 3
    arena = torch.empty(1 << 20, dtype=torch.uint8, device=CPU)
    q = QSAKVCache(
        num_kv_heads=H, num_layers=L, head_dim=D, num_pages=P, page_size=PS,
        dtype=DT, device=CPU, index_head_dim=idx_dim, num_index_layers=idx_layers,
        index_ratio=ratio, num_req_slots=slots, arena=arena,
    )
    kv_n = q._kv_buffer.numel() * 2
    cmp_n = q._cmp_k_buffer.numel() * 2
    assert off(q._kv_buffer, arena) == 0
    assert off(q._cmp_k_buffer, arena) == kv_n, "cmp slab packs right after the KV cursor"
    assert off(q._pending_ring, arena) == kv_n + cmp_n
    assert (q._cmp_k_buffer == 0).all() and (q._pending_ring == 0).all(), "index tiers zero-init"
    assert q.cmp_scratch_base == P * PS // ratio
    q.rebuild(2 * P)  # grow: KV base stable, tiers re-anchor after the new KV size
    kv_n2 = q._kv_buffer.numel() * 2
    assert q._kv_buffer.data_ptr() == arena.data_ptr()
    assert off(q._cmp_k_buffer, arena) == kv_n2
    assert (q._cmp_k_buffer == 0).all() and (q._pending_ring == 0).all()
    print("  qsa tiers: pack after KV cursor, zero-init, re-anchor on grow  OK")


def test_moe_right_anchor_and_rebuild():
    arena = torch.empty(1 << 20, dtype=torch.uint8, device=CPU)
    cache = make_moe(64, arena)
    for new in (64, 96, 32):
        if new != 64:
            cache.rebuild(new)
        c1, c2 = cache.bank_caches["gate_up"], cache.bank_caches["down"]
        assert off(c1, arena) + c1.numel() * 2 == arena.numel(), "last bank ends at the arena end"
        assert off(c2, arena) + c2.numel() * 2 == off(c1, arena), "banks pack contiguously"
        assert off(c1, arena) % 8 == 0 and off(c2, arena) % 8 == 0
    print("  moe right-anchor: end-anchored, contiguous, stable across rebuild  OK")


def test_both_sides_never_cross():
    # Engine order: MoE banks right-anchor first, then the KV side is sized to the
    # remainder and left-anchors -- the two cursors may meet but never cross.
    L, H, D, P, PS = 2, 4, 8, 4, 4
    arena = torch.empty(1 << 20, dtype=torch.uint8, device=CPU)
    cache = make_moe(64, arena)
    moe_start = min(off(c, arena) for c in cache.bank_caches.values())
    pool = MHAKVCache(H, L, D, P, PS, DT, CPU, arena=arena)
    assert pool._arena_next <= moe_start, f"KV cursor {pool._arena_next} crossed MoE start {moe_start}"
    # Slide the boundary: KV grows into the slack, MoE shrinks back -- still no cross.
    slack = moe_start - pool._arena_next
    bytes_per_page = 2 * L * PS * H * D * 2  # K+V x layers x page tokens x heads x dim x 2B
    pool.rebuild(P + slack // bytes_per_page - 1)
    assert pool._arena_next <= moe_start
    cache.rebuild(32)
    moe_start2 = min(off(c, arena) for c in cache.bank_caches.values())
    assert pool._arena_next <= moe_start2
    print("  shared arena: KV cursor + MoE cursor never cross on boundary slides  OK")


def test_overflow_fails_loud():
    arena = torch.empty(1024, dtype=torch.uint8, device=CPU)
    fired = False
    try:
        MHAKVCache(4, 2, 8, 64, 4, DT, CPU, arena=arena)  # 64 pages x 1024 B >> 1 KiB
    except AssertionError as e:
        fired = "arena overflow: KV side" in str(e)
    assert fired, "KV-side overflow assert must fire"
    fired = False
    try:
        make_moe(512, arena)  # 512 slots x 12 KiB >> 1 KiB
    except AssertionError as e:
        fired = "arena overflow: MoE side" in str(e)
    assert fired, "MoE-side overflow assert must fire"
    print("  overflow: undersized arena fails loud on both sides  OK")


def test_legacy_path_still_works():
    pool = MHAKVCache(4, 2, 8, 4, 4, DT, CPU)
    assert pool._arena is None
    pool.rebuild(8)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=16, cache_size=32, device=CPU, quant_format="bf16"
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.zeros((16, 32, 8), dtype=DT) for _ in range(2)],
            "down": [torch.zeros((16, 8, 16), dtype=DT) for _ in range(2)],
        }
    )
    cache.rebuild(48)
    print("  legacy (arena=None): private allocations + rebuild unchanged  OK")


if __name__ == "__main__":
    test_mha_left_anchor_stable()
    test_qsa_tiers_pack_after_kv()
    test_moe_right_anchor_and_rebuild()
    test_both_sides_never_cross()
    test_overflow_fails_loud()
    test_legacy_path_still_works()
    print("arena smoke: all assertions passed")
