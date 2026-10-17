"""mm RAM-tier tests (CPU-only, no engine): store-on-finish / restore-on-match.

1. Pool row round trip: snapshot_rows/restore_rows through the QSA pool (K/V + cmp rows,
   page-slot -> cmp-row = slot // index_ratio).
2. Tier store on a finished mm request: entry boundary from the pending ×CHUNK track,
   bytes accounted, second store of the same key replaces (no double count).
3. Restore on an exact content-key hit: fresh pages allocated, rows promoted, matched
   indices land on the promoted bytes, GDN host payload rides on the MatchResult.
4. Guards: different content key misses; a boundary inside a pad run is refused
   (mirrors match_req's splice invariant); LRU eviction frees bytes over the cap.

  python -X utf8 verification/kv_tier_test.py
"""

import sys

FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def main() -> int:
    import torch
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.utils import PendingReq

    import torch
    from freetoken.distributed import set_tp_info

    set_tp_info(0, 1)
    torch.manual_seed(0)
    dev = torch.device("cpu")
    PAGE = 4

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=2,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=4, output_gate="silu",
    )
    lsp = LinearStatePool(
        group, num_slots=4, dtype=torch.float32, device=dev, tp_size=1,
        slot_states=(SlotStateSpec(name="anchor", shape=(3,), layer_ids=(0,), fill_value=-1.0),),
    )
    pool = QSAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=32, num_pages=8, page_size=PAGE,
        dtype=torch.float16, device=dev, index_head_dim=4, num_index_layers=1,
        index_ratio=2, num_req_slots=2, kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    # deterministic content in the slabs
    kv_ref = torch.arange(pool._kv_buffer.numel(), dtype=torch.float16).view_as(pool._kv_buffer)
    pool._kv_buffer.copy_(kv_ref)
    ds_ref = torch.arange(pool._dscale_buffer.numel(), dtype=torch.bfloat16).view_as(pool._dscale_buffer)
    pool._dscale_buffer.copy_(ds_ref)
    cmp_ref = torch.arange(pool._cmp_k_buffer.numel(), dtype=torch.float16).view_as(pool._cmp_k_buffer)
    pool._cmp_k_buffer.copy_(cmp_ref)

    page_table = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(
        num_pages=8, page_size=PAGE, page_table=page_table, type="hybrid_radix",
        linear_state_pool=lsp, kv_pool=pool,
    )
    check("cap.default", cm._mm_tier_cap > 0, str(cm._mm_tier_cap))

    # ---------------- store ----------------------------------------------------
    B = 16  # track boundary = 4 pages of 4
    # req1 lives in pages 4..19 bases? -> pages with bases 4,8,12,16 (token slots 4..19);
    # claim those page bases so the restore allocation lands on DIFFERENT pages.
    page_table[0, :B] = torch.arange(4, 4 + B, dtype=torch.int32)
    cm.free_slots = torch.tensor([0, 20, 24, 28], dtype=torch.int32)
    key = torch.full((B,), 1_000_000_007, dtype=torch.int32)
    src = lsp.alloc(1)[0]
    lsp.conv_states[:, src] = 7.0
    lsp.recurrent_states[:, src] = 9.0
    lsp.slot_states["anchor"][:, src] = 3.0
    req1 = type("R", (), {})()
    req1.mm_cache_key, req1.mm_embeds = key, torch.zeros(1, 2)
    req1.max_device_len, req1.output_len, req1.cached_len = B + 8, 8, B
    req1.mamba_last_track_seqlen = B
    req1.mamba_ping_pong, req1.mamba_next_track_idx = (src, src), 0
    req1.linear_slot_idx = src
    req1.table_idx = 0
    cm._tier_store(req1)
    check("store.entry", len(cm._mm_tier) == 1)
    digest = cm._tier_digest(key)
    ekey, rows, gdn, b_e, nbytes, _tx = cm._mm_tier[digest]
    check("store.boundary", b_e == B and nbytes > 0, f"b={b_e} bytes={nbytes}")
    check("store.bytes", cm._mm_tier_bytes == nbytes, f"{cm._mm_tier_bytes} vs {nbytes}")
    check("store.gdn", gdn is not None and torch.equal(gdn[0], torch.full_like(gdn[0], 7.0)))

    # replacing the same key must not double-count bytes
    cm._tier_store(req1)
    check("store.replace", len(cm._mm_tier) == 1 and cm._mm_tier_bytes == nbytes)

    # ---------------- restore --------------------------------------------------
    # A follow-up turn: identical prompt (key == ids at non-pad positions, pads carry the
    # content ids; here the whole prompt is the media span -> key differs from raw ids).
    ids = torch.full((B,), 42, dtype=torch.int32)
    pending = PendingReq.__new__(PendingReq)
    pending.input_ids, pending.mm_embeds, pending.mm_cache_key = ids, torch.zeros(1, 2), key
    pending.mm_gdn = None
    from freetoken.kvcache import MatchResult
    from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle
    m0 = MatchResult(HybridCacheHandle(0, cm.prefix_cache.root, torch.empty(0, dtype=torch.int32)))
    mr = cm.restore_mm(pending, m0)
    h = mr.cuda_handle
    check("restore.len", h.cached_len == B, str(h.cached_len))
    matched = h.get_matched_indices()
    check("restore.indices", matched.numel() == B, str(matched.numel()))
    # promoted bytes: K/V rows at the NEW slots equal the ORIGINAL rows [4..20)
    kb = pool._kv_buffer
    kv64 = kb.view(kb.shape[1] * 2, kb.shape[2] * kb.shape[3], -1)
    orig = torch.arange(4, 4 + B, dtype=torch.int64)
    check("restore.kv_rows", torch.equal(kv64[:, matched.to(torch.int64)], kv64[:, orig]))
    ds = pool._dscale_buffer
    check("restore.has_dscale", ds is not None)
    ds64 = ds.view(ds.shape[1] * 2, ds.shape[2], -1)
    check("restore.dscale_rows", torch.equal(ds64[:, matched.to(torch.int64)], ds64[:, orig]))
    exp_rows = torch.arange(4, 4 + B, dtype=torch.int64) // 2
    ok_cmp = torch.equal(pool._cmp_k_buffer[:, matched.to(torch.int64) // 2], cmp_ref[:, exp_rows])
    check("restore.cmp_rows", ok_cmp)
    check("restore.gdn_host", mr.gdn_host is not None and torch.equal(mr.gdn_host[0], gdn[0]))
    # live-slot write path the scheduler uses
    dst = lsp.alloc(1)[0]
    lsp.restore_state(dst, mr.gdn_host)
    check("restore.state", torch.equal(lsp.conv_states[:, dst], torch.full_like(gdn[0], 7.0))
          and torch.equal(lsp.slot_states["anchor"][:, dst], torch.full_like(gdn[2]["anchor"], 3.0)))

    # ---------------- guards ----------------------------------------------------
    other = PendingReq.__new__(PendingReq)
    other.input_ids = ids
    other.mm_embeds = torch.zeros(1, 2)
    other.mm_cache_key = torch.full((B,), 1_200_000_011, dtype=torch.int32)
    other.mm_gdn = None
    mr2 = cm.restore_mm(other, m0)
    check("guard.key_miss", mr2.cuda_handle.cached_len == 0)

    # boundary inside a pad run: ids pad=42 at [2..6], key differs there -> span (2,7);
    # b=4 splits the run (refuse), b=1 clears it ahead (accept), b=8 clears past the end.
    ids_p = torch.full((10,), 5, dtype=torch.int32)
    ids_p[2:7] = 42
    key_p = ids_p.clone()
    key_p[2:7] = 1_000_000_007
    p = PendingReq.__new__(PendingReq)
    p.input_ids, p.mm_cache_key, p.mm_embeds = ids_p, key_p, torch.zeros(1, 1)
    check("bound.inside_run", not cm._mm_restore_ok(p, 4))
    check("bound.before_run", cm._mm_restore_ok(p, 2))
    check("bound.after_run", cm._mm_restore_ok(p, 7))

    # plain-radix caches: restored handle must survive lock_handle's isinstance assert
    cm_p = CacheManager(
        num_pages=8, page_size=PAGE, page_table=torch.zeros(2, 32, dtype=torch.int32),
        type="radix", kv_pool=pool,
    )
    req_p = type("R", (), {})()
    req_p.mm_cache_key, req_p.mm_embeds = key, torch.zeros(1, 2)
    req_p.max_device_len, req_p.output_len, req_p.cached_len = B + 8, 8, B
    req_p.mamba_last_track_seqlen = None
    req_p.mamba_ping_pong, req_p.mamba_next_track_idx = None, 0
    req_p.linear_slot_idx, req_p.table_idx = None, 0
    cm_p._tier_store(req_p)
    from freetoken.kvcache import MatchResult as _MR
    p0 = _MR(cm_p.prefix_cache.match_prefix(ids[:4]).cuda_handle)
    mr_p = cm_p.restore_mm(pending, p0)
    hp = mr_p.cuda_handle
    cm_p.lock(hp)
    cm_p.unlock(hp)  # lock_handle's isinstance(RadixCacheHandle) must hold on the subclass
    check("plainradix.lock_unlock", hp.cached_len == B
          and hp.get_matched_indices().numel() == B)

    # LRU eviction by bytes
    cm._mm_tier_cap = nbytes + (nbytes // 2)
    key2 = torch.full((B,), 1_500_000_009, dtype=torch.int32)
    req1.mm_cache_key = key2
    cm._tier_store(req1)
    check("evict.lru", len(cm._mm_tier) == 1 and cm._tier_digest(key2) in cm._mm_tier
          and cm._mm_tier_bytes <= cm._mm_tier_cap)

    # ---------------- text tier (park on finish / prefix-match restore) ----------
    from freetoken.kvcache.radix_cache import RadixCacheHandle  # noqa: F811
    cm_t = CacheManager(
        num_pages=8, page_size=PAGE, page_table=torch.zeros(2, 32, dtype=torch.int32),
        type="radix", kv_pool=pool,
    )
    treq = type("R", (), {})()
    t_ids = torch.arange(100, 112, dtype=torch.int32)  # 12-token prompt, b = align_down(12, 4)
    treq.mm_cache_key, treq.mm_embeds = None, None
    treq.input_ids, treq.max_device_len, treq.output_len = t_ids, 20, 8
    treq.mamba_last_track_seqlen, treq.mamba_ping_pong, treq.mamba_next_track_idx = None, None, 0
    treq.linear_slot_idx, treq.table_idx, treq.cached_len = None, 0, 12
    cm_t._tier_store(treq)
    check("text.store", len(cm_t._mm_tier) == 1, f"n={len(cm_t._mm_tier)} b={cm_t._mm_tier_bytes}")

    def _t0(mgr):
        return _MR(mgr.prefix_cache.match_prefix(torch.arange(100, 101, dtype=torch.int32)).cuda_handle)

    tp = PendingReq.__new__(PendingReq)
    tp.input_ids = torch.cat([t_ids, torch.arange(120, 128, dtype=torch.int32)])  # grown history
    tp.mm_embeds, tp.mm_cache_key, tp.mm_gdn = None, None, None
    mr_t = cm_t.restore_mm(tp, _t0(cm_t))
    check("text.restore_prefix", mr_t.cuda_handle.cached_len == 12, str(mr_t.cuda_handle.cached_len))

    # turn 3: history grows again -- the SAME parked window still matches the head
    tp2 = PendingReq.__new__(PendingReq)
    tp2.input_ids = torch.cat([tp.input_ids, torch.arange(130, 138, dtype=torch.int32)])
    tp2.mm_embeds, tp2.mm_cache_key, tp2.mm_gdn = None, None, None
    mr_t2 = cm_t.restore_mm(tp2, _t0(cm_t))
    check("text.grow_history", mr_t2.cuda_handle.cached_len == 12)

    # different conversation head -> no false match
    tp3 = PendingReq.__new__(PendingReq)
    tp3.input_ids = torch.cat([torch.arange(200, 208, dtype=torch.int32), t_ids])
    tp3.mm_embeds, tp3.mm_cache_key, tp3.mm_gdn = None, None, None
    mr_t3 = cm_t.restore_mm(tp3, _t0(cm_t))
    check("text.no_false_match", mr_t3.cuda_handle.cached_len == 0)

    # budget-slide rebuild: the RAM tier survives (rows are geometry-independent), tree wipes
    cm_t.rebuild(8, torch.zeros(2, 32, dtype=torch.int32))
    check("text.tier_survives_rebuild", len(cm_t._mm_tier) == 1)
    tp.mm_gdn = None
    mr_t4 = cm_t.restore_mm(tp, _t0(cm_t))
    check("text.restore_after_rebuild", mr_t4.cuda_handle.cached_len == 12)

    # ---------------- hybrid chunked-prefill tier fallback (last_committed_boundary) ----
    # A 13-token prompt, prefill chunked at 8 (page-aligned xCHUNK boundary). The commit
    # consumes mamba_last_track_seqlen, so at finish neither the pending-track branch nor
    # the page-aligned live-slot branch fires (13 % 4 != 0) -- the tier must park at the
    # deepest committed boundary (8) or the agent's next turn re-prefills cold after every
    # rebuild wipe. Dedicated manager: the commit inserts into ITS tree (tree match is 0,
    # mirroring a just-wiped tree, so the tier entry must beat the tree to be visible).
    pool2 = QSAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=32, num_pages=8, page_size=PAGE,
        dtype=torch.float16, device=dev, index_head_dim=4, num_index_layers=1,
        index_ratio=2, num_req_slots=2, kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    pool2._kv_buffer.copy_(torch.arange(pool2._kv_buffer.numel(), dtype=torch.float16).view_as(pool2._kv_buffer))
    lsp2 = LinearStatePool(
        group, num_slots=4, dtype=torch.float32, device=dev, tp_size=1,
        slot_states=(SlotStateSpec(name="anchor", shape=(3,), layer_ids=(0,), fill_value=-1.0),),
    )
    pt2 = torch.zeros(2, 32, dtype=torch.int32)
    cm2 = CacheManager(
        num_pages=8, page_size=PAGE, page_table=pt2, type="hybrid_radix",
        linear_state_pool=lsp2, kv_pool=pool2,
    )
    c_ids = torch.arange(500, 513, dtype=torch.int32)  # 13 tokens, unaligned prompt end
    # Original prompt lives in slots 4..16 so the restore below lands on DIFFERENT slots.
    pt2[0, :13] = torch.arange(4, 4 + 13, dtype=torch.int32)
    c_src = lsp2.alloc(1)[0]
    lsp2.conv_states[:, c_src] = 5.0
    lsp2.recurrent_states[:, c_src] = 6.0
    lsp2.slot_states["anchor"][:, c_src] = 1.0
    c_live = lsp2.alloc(1)[0]
    cm2.free_slots = torch.tensor([0, 20, 24, 28], dtype=torch.int32)
    creq = type("R", (), {})()
    creq.input_ids, creq.mm_cache_key, creq.mm_embeds = c_ids, None, None
    creq.cached_len, creq.max_device_len, creq.output_len = 13, 17, 4  # prompt_len = 13
    creq.mamba_last_track_seqlen = 8
    creq.mamba_ping_pong, creq.mamba_next_track_idx = (c_src, c_src), 0
    creq.linear_slot_idx, creq.table_idx = c_live, 0
    creq.last_committed_boundary = None
    creq.cache_handle = HybridCacheHandle(0, cm2.prefix_cache.root, torch.empty(0, dtype=torch.int32))
    cm2._cache_req_hybrid(creq, finished=False)  # chunk commit at 8
    check("chunk.commit_boundary", creq.last_committed_boundary is not None
          and creq.last_committed_boundary[0] == 8, str(creq.last_committed_boundary))
    check("chunk.seqlen_consumed", creq.mamba_last_track_seqlen is None)
    creq.cached_len = 13  # prompt end after the tail of the chunk
    cm2._cache_req_hybrid(creq, finished=True)   # finish -> _tier_store fallback
    check("chunk.tier_parked", len(cm2._mm_tier) == 1, f"n={len(cm2._mm_tier)}")
    cekey, crows, cgdn, cb, cnb, _tx = cm2._mm_tier[cm2._mm_tier and next(iter(cm2._mm_tier))]
    check("chunk.tier_b", cb == 8, f"b={cb}")
    check("chunk.tier_gdn", cgdn is not None and torch.equal(cgdn[0], torch.full_like(cgdn[0], 5.0)))
    cm2.free_slots = torch.tensor([0, 20, 24, 28], dtype=torch.int32)
    cpend = PendingReq.__new__(PendingReq)
    cpend.input_ids = torch.cat([c_ids, torch.arange(513, 519, dtype=torch.int32)])  # grown history
    cpend.mm_embeds, cpend.mm_cache_key, cpend.mm_gdn = None, None, None
    cmr = cm2.restore_mm(cpend, _MR(HybridCacheHandle(0, cm2.prefix_cache.root, torch.empty(0, dtype=torch.int32))))
    ch = cmr.cuda_handle
    check("chunk.restore_b", ch.cached_len == 8, str(ch.cached_len))
    check("chunk.restore_gdn_host", cmr.gdn_host is not None and torch.equal(cmr.gdn_host[0], cgdn[0]))
    cmatched = ch.get_matched_indices()
    ckv64 = pool2._kv_buffer.view(pool2._kv_buffer.shape[1] * 2, pool2._kv_buffer.shape[2] * pool2._kv_buffer.shape[3], -1)
    check("chunk.restore_rows", torch.equal(
        ckv64[:, cmatched.to(torch.int64)], ckv64[:, torch.arange(4, 12, dtype=torch.int64)]))
    # prompt end had aligned to a page and no chunk crossed a boundary -> no committed
    # boundary, no track, unaligned... but page-aligned live slot still parks (regression guard).
    creq2 = type("R", (), {})()
    creq2.input_ids = c_ids[:12]
    creq2.mm_cache_key, creq2.mm_embeds = None, None
    creq2.cached_len, creq2.max_device_len, creq2.output_len = 12, 16, 4
    creq2.mamba_last_track_seqlen, creq2.mamba_ping_pong, creq2.mamba_next_track_idx = None, None, 0
    creq2.linear_slot_idx, creq2.table_idx = c_live, 0
    creq2.last_committed_boundary = None
    cm2._tier_store(creq2)  # page-aligned live-slot branch parks; no commit needed to test it
    check("chunk.live_slot_parks", len(cm2._mm_tier) == 2)

    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
