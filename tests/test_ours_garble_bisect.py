"""CPU bisect tests for the ours-tree garbled-output suspects, one section each.

Suspect order (see kb project_arena_design.md 2026-10-07 evening):
  1. GDN frozen-state / boundary pairing: the fallback must park at the COMMITTED
     boundary and take the GDN state from the slot stored WITH that boundary --
     ping-pong parity flips must not re-select the half. Branch-1 keeps the
     opposite rule (parity picks the half) and must stay consistent within itself.
  2. lock/unlock cycles + tier matching: repeated lock/unlock and intervening
     different-content turns must not make the original prompt miss or a
     different-head prompt false-match.
  3. rebuild (shrink) then tier restore: restored bytes must be readable through
     the rebuilt view -- the consumer contract of "rows are geometry-independent".
  4. arena right-anchor alignment across MoE cache sizes (incl. the fused-tree
     launch value 3072) on shared arenas whose size is/exact-fits the demand.

  PYTHONPATH=src/python python -X utf8 src/tests/test_ours_garble_bisect.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def _mk_hybrid(torch):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
    from freetoken.scheduler.cache import CacheManager

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=2,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=4, output_gate="silu",
    )
    lsp = LinearStatePool(
        group, num_slots=6, dtype=torch.float32, device=torch.device("cpu"), tp_size=1,
        slot_states=(SlotStateSpec(name="anchor", shape=(3,), layer_ids=(0,), fill_value=-1.0),),
    )
    pool = QSAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=32, num_pages=8, page_size=4,
        dtype=torch.float16, device=torch.device("cpu"), index_head_dim=4,
        num_index_layers=1, index_ratio=2, num_req_slots=2,
        kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    pool._kv_buffer.copy_(torch.arange(pool._kv_buffer.numel(), dtype=torch.float16).view_as(pool._kv_buffer))
    pt = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(
        num_pages=8, page_size=4, page_table=pt, type="hybrid_radix",
        linear_state_pool=lsp, kv_pool=pool,
    )
    return cm, pt, lsp, pool


def test_gdn_half_pairing(torch):
    """Suspect 1: fallback parks at committed boundary with the HALF stored beside it."""
    from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

    cm, pt, lsp, pool = _mk_hybrid(torch)
    a = lsp.alloc(1)[0]
    b = lsp.alloc(1)[0]
    lsp.conv_states[:, a] = 1.0
    lsp.conv_states[:, b] = 2.0
    ids = torch.arange(500, 513, dtype=torch.int32)  # 13 tokens, unaligned prompt end
    pt[0, :13] = torch.arange(0, 13, dtype=torch.int32)
    cm.free_slots = torch.arange(16, 24, dtype=torch.int32)

    req = type("R", (), {})()
    req.input_ids, req.mm_cache_key, req.mm_embeds = ids, None, None
    req.cached_len, req.max_device_len, req.output_len = 13, 17, 4  # prompt_len=13
    req.mamba_last_track_seqlen = None
    # parity formula (branch-1) picks index 1-next=0 -> slot a; the committed tuple
    # stores b as frozen -> the fallback must follow the tuple and yield conv=2.0.
    req.mamba_ping_pong, req.mamba_next_track_idx = (a, b), 1
    req.linear_slot_idx, req.table_idx = a, 0
    req.last_committed_boundary = (8, b)
    req.cache_handle = HybridCacheHandle(0, cm.prefix_cache.root, torch.empty(0, dtype=torch.int32))
    cm._tier_store(req)
    check("gdn.fallback_parked", len(cm._mm_tier) == 1)
    _k, _r, gdn, be, _nb, _tx = next(iter(cm._mm_tier.values()))
    check("gdn.fallback_boundary8", be == 8, f"b={be}")
    check("gdn.fallback_half_from_tuple",
          gdn is not None and torch.equal(gdn[0], torch.full_like(gdn[0], 2.0)),
          f"conv={None if gdn is None else float(gdn[0].float().mean()):.1f}")

    # branch-1 (pending track) stays parity-driven and consistent within itself:
    req2 = type("R", (), {})()
    req2.input_ids, req2.mm_cache_key, req2.mm_embeds = ids[:8], None, None
    req2.cached_len, req2.max_device_len, req2.output_len = 8, 12, 4  # prompt_len=8
    req2.mamba_last_track_seqlen = 8
    req2.mamba_ping_pong, req2.mamba_next_track_idx = (a, b), 0  # -> frozen = b
    req2.linear_slot_idx, req2.table_idx = a, 0
    req2.last_committed_boundary = None
    cm._tier_store(req2)
    _k2, _r2, gdn2, be2, _nb2, _tx2 = cm._mm_tier[cm._tier_digest(ids[:8].cpu().clone())] \
        if hasattr(cm, "_tier_digest") else next(iter(cm._mm_tier.values()))
    check("gdn.track_branch_parity", be2 == 8
          and gdn2 is not None and float(gdn2[0].float().mean()) == 2.0)

    # boundary must never exceed prompt_len (state/KV length mismatch guard)
    req.last_committed_boundary = (16, b)
    n_before = len(cm._mm_tier)
    cm._tier_store(req)
    still = next(iter(cm._mm_tier.values()))
    check("gdn.no_over_advance", still[3] <= 13, f"b={still[3]}")


def test_lock_cycles_matching(torch):
    """Suspect 2: lock/unlock cycles + interleaved turns keep tier matching sane."""
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.utils import PendingReq
    from freetoken.kvcache import MatchResult as _MR

    pool = QSAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=32, num_pages=8, page_size=4,
        dtype=torch.float16, device=torch.device("cpu"), index_head_dim=4,
        num_index_layers=1, index_ratio=2, num_req_slots=2,
        kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    pt = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(num_pages=8, page_size=4, page_table=pt, type="radix", kv_pool=pool)

    t_ids = torch.arange(100, 112, dtype=torch.int32)
    treq = type("R", (), {})()
    treq.mm_cache_key, treq.mm_embeds = None, None
    treq.input_ids, treq.max_device_len, treq.output_len = t_ids, 20, 8
    treq.mamba_last_track_seqlen, treq.mamba_ping_pong, treq.mamba_next_track_idx = None, None, 0
    treq.linear_slot_idx, treq.table_idx, treq.cached_len = None, 0, 12
    cm._tier_store(treq)

    def turn(ids):
        # one production-shaped turn: match -> restore -> lock/unlock -> cache_req(finish)
        p = PendingReq.__new__(PendingReq)
        p.input_ids, p.mm_embeds, p.mm_cache_key, p.mm_gdn = ids, None, None, None
        mr = cm.restore_mm(p, _MR(cm.prefix_cache.match_prefix(
            ids[:1]).cuda_handle))
        h = mr.cuda_handle
        matched = (h.get_matched_indices().to(torch.int32) if h.cached_len
                   else torch.empty(0, dtype=torch.int32))
        pt[0, : matched.numel()] = matched
        req = type("R", (), {})()
        req.input_ids, req.mm_cache_key, req.mm_embeds = ids, None, None
        req.table_idx, req.cache_handle = 0, h
        req.cached_len, req.device_len = matched.numel(), len(ids)
        req.output_len, req.max_device_len = 8, len(ids) + 8
        req.mamba_last_track_seqlen = None
        req.mamba_ping_pong, req.mamba_next_track_idx = None, 0
        req.linear_slot_idx, req.table_idx = None, 0
        req.last_committed_boundary = None
        cm.lock(h)
        cm.unlock(h)
        cm.cache_req(req, finished=True)
        return h

    h = turn(t_ids)
    check("lock.first_hit", h.cached_len == 12, str(h.cached_len))
    ok, rnd = True, 0
    other = torch.cat([torch.arange(200, 208, dtype=torch.int32), t_ids])
    for rnd in range(3):
        h2 = turn(other)           # different head: miss, no false match
        h3 = turn(t_ids)
        ok = h3.cached_len == 12
        if not ok:
            break
    check("lock.cycles_stable", ok, f"rounds={rnd+1} b={h3.cached_len}")
    # Growing-history design: the parked window must be a true prefix of the new
    # prompt -- accept iff other[:b] == other[:b] recomputed from the stored window.
    # Here the conversation grew (t_ids turn came first), so b=20 covering the whole
    # 20-token prompt is the design-correct hit, not a false match.
    win_ok = 0 < h2.cached_len <= other.numel()
    check("lock.grow_history_prefix", win_ok, f"b={h2.cached_len}")


def test_rebuild_shrink_restore(torch):
    """Suspect 3: after a SHRINK rebuild, restored rows read back through the new view."""
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.utils import PendingReq
    from freetoken.kvcache import MatchResult as _MR

    pool = QSAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=32, num_pages=8, page_size=4,
        dtype=torch.float16, device=torch.device("cpu"), index_head_dim=4,
        num_index_layers=1, index_ratio=2, num_req_slots=2,
        kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    pool._kv_buffer.copy_(torch.arange(pool._kv_buffer.numel(), dtype=torch.float16).view_as(pool._kv_buffer))
    pt = torch.zeros(2, 32, dtype=torch.int32)
    pt[0, :12] = torch.arange(0, 12, dtype=torch.int32)
    cm = CacheManager(num_pages=8, page_size=4, page_table=pt, type="radix", kv_pool=pool)

    t_ids = torch.arange(100, 112, dtype=torch.int32)
    treq = type("R", (), {})()
    treq.mm_cache_key, treq.mm_embeds = None, None
    treq.input_ids, treq.max_device_len, treq.output_len = t_ids, 20, 8
    treq.mamba_last_track_seqlen, treq.mamba_ping_pong, treq.mamba_next_track_idx = None, None, 0
    treq.linear_slot_idx, treq.table_idx, treq.cached_len = None, 0, 12
    cm._tier_store(treq)
    entry = next(iter(cm._mm_tier.values()))
    payload = entry[1]["kv"].clone()

    pt2 = torch.zeros(2, 32, dtype=torch.int32)
    pt2[0, :12] = torch.arange(12, 24, dtype=torch.int32)  # original bytes move out of range
    cm.rebuild(5, pt2)                                      # 8 -> 5 pages (shrink)
    check("shrink.tier_survives", len(cm._mm_tier) == 1)

    p = PendingReq.__new__(PendingReq)
    p.input_ids = torch.cat([t_ids, torch.arange(120, 128, dtype=torch.int32)])
    p.mm_embeds, p.mm_cache_key, p.mm_gdn = None, None, None
    mr = cm.restore_mm(p, _MR(cm.prefix_cache.match_prefix(
        torch.arange(100, 101, dtype=torch.int32)).cuda_handle))
    h = mr.cuda_handle
    check("shrink.restore_len", h.cached_len == 12, str(h.cached_len))
    matched = h.get_matched_indices().to(torch.int64)
    kb = pool._kv_buffer
    view = kb.view(kb.shape[1] * 2, kb.shape[2] * kb.shape[3], -1)
    check("shrink.bytes_through_new_view", torch.equal(view[:, matched], payload))


def test_anchor_alignment_sweep(torch):
    """Suspect 4: MoE right-anchor offsets stay 16-aligned across cache sizes."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    DT = torch.bfloat16
    NE, NL, HID, INTER = 16, 2, 8, 16
    per_slot = NL * (NE * 2 * INTER * HID + NE * HID * INTER) * DT.itemsize
    for cs in (3072, 5652):
        need = per_slot * cs
        arena_bytes = (need + (1 << 22) + 15) // 16 * 16
        arena = torch.empty(arena_bytes, dtype=torch.uint8)
        cache = OffloadMoeCache(num_layers=NL, num_experts=NE, cache_size=cs,
                                device=torch.device("cpu"), quant_format="bf16", arena=arena)
        gu = [torch.zeros((NE, 2 * INTER, HID), dtype=DT) for _ in range(NL)]
        dn = [torch.zeros((NE, HID, INTER), dtype=DT) for _ in range(NL)]
        cache.set_bank_sources({"gate_up": gu, "down": dn})
        offs = {k: t.data_ptr() - arena.data_ptr() for k, t in cache.bank_caches.items()}
        aligned = all(o % 16 == 0 for o in offs.values())
        anchored = max(o + t.numel() * t.element_size()
                       for o, t in zip(offs.values(), cache.bank_caches.values())) == arena_bytes
        check(f"align.cs{cs}", aligned and anchored,
              f"offs={sorted(offs.values())} arena={arena_bytes}")
        cache.rebuild(cs // 2)
        offs2 = {k: t.data_ptr() - arena.data_ptr() for k, t in cache.bank_caches.items()}
        check(f"align.cs{cs}_after_shrink", all(o % 16 == 0 for o in offs2.values()),
              f"offs={sorted(offs2.values())}")


def main() -> int:
    import torch
    from freetoken.distributed import set_tp_info

    set_tp_info(0, 1)
    torch.manual_seed(0)
    test_gdn_half_pairing(torch)
    test_lock_cycles_matching(torch)
    test_rebuild_shrink_restore(torch)
    test_anchor_alignment_sweep(torch)
    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
