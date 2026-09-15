"""Round-4 adversarial probes: A two pad spans, B COW no-op observable, C page ownership,
D splice-invariant fuzz ps in 1/4/8, E text uncapped + keyless mm no-reuse.
World: real CacheManager(hybrid_radix) + real LinearStatePool, engine-exact lifecycle.
Conservation oracle: free + tree == N_PAGES."""
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq

FAIL = []
PAD = 248056
N_PAGES = 512
CID_A = 1_000_000_001
CID_B = 1_000_000_002


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def make_group():
    return LinearGatedDeltaGroupConfig(
        name="g", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=2, output_gate="silu",
    )


def fp32(x):
    """The value a float64 sum becomes after being stored in the fp32 pool tensor."""
    return float(torch.tensor(x, dtype=torch.float32).item())


class World:
    def __init__(self, ids_all, ps):
        self.ps = ps
        self.ids_all = ids_all
        self.pool = LinearStatePool(make_group(), num_slots=16, dtype=torch.float32,
                                    device=torch.device("cpu"), tp_size=1)
        self.page_table = torch.zeros(32, N_PAGES * ps, dtype=torch.int32)
        self.cm = CacheManager(N_PAGES, ps, self.page_table, "hybrid_radix",
                               linear_state_pool=self.pool)
        self.pc = self.cm.prefix_cache

    def fp(self, slot):
        return float(self.pool.conv_states[0, slot, 0, 0])

    def advance(self, slot, n):
        self.pool.conv_states[0, slot, 0, 0] = float(sum(self.ids_all[:n]))

    def pages_conserved(self):
        tree_tokens = self.pc.full_evictable + self.pc.full_protected
        held = len(self.cm.free_slots) + tree_tokens // self.ps
        return tree_tokens % self.ps == 0 and held == N_PAGES, (
            f"free={len(self.cm.free_slots)} tree={tree_tokens}tok held={held}/{N_PAGES}")

    def tree_slots(self):
        slots = []
        stack = list(self.pc.root.children.values())
        while stack:
            n = stack.pop()
            slots.extend(n.value.tolist())
            stack.extend(n.children.values())
        return slots


def admit(world, uid, ids, key, mm=True, n_gen=9):
    sp = SamplingParams(max_tokens=n_gen)
    pr = PendingReq(uid, ids, sp,
                    mm_embeds=torch.zeros(4, 8) if mm else None,
                    mm_cache_key=key if mm else None)
    mr = world.cm.match_req(pr)
    world.cm.lock(mr.cuda_handle)
    live = world.pool.alloc(1)[0]
    pp = tuple(world.pool.alloc(2))
    req = Req(input_ids=ids, table_idx=0, cached_len=mr.cuda_handle.cached_len,
              output_len=n_gen, uid=uid, sampling_params=sp,
              cache_handle=mr.cuda_handle,
              mm_embeds=torch.zeros(4, 8) if mm else None,
              mm_cache_key=key if mm else None,
              linear_slot_idx=live, mamba_ping_pong=pp)
    world.cm.allocate_paged([req])
    return req, mr


def run_turn(world, req, plen, n_gen):
    live = req.linear_slot_idx
    cached0 = req.cached_len
    world.advance(live, plen)
    extend = plen - cached0
    c = (extend - 1) // 64
    track = None
    if c >= 1:
        track = cached0 + c * 64
        frozen = req.mamba_ping_pong[req.mamba_next_track_idx]
        world.advance(frozen, track)
        req.mamba_last_track_seqlen = track
        req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))
    world.cm.cache_req(req, finished=False)
    fin = plen + n_gen - 1
    for j in range(1, n_gen):
        world.advance(live, plen + j)
        req.append_host(torch.tensor([100 + j], dtype=torch.int32))
        if plen + j <= fin:
            world.cm.allocate_paged([req])
        req.complete_one()
    assert req.cached_len == fin, f"cached_len={req.cached_len} want {fin}"
    world.cm.cache_req(req, finished=True)
    return fin, track


def key_of(ids, cid):
    k = ids.clone()
    k[ids == PAD] = cid
    return k


def mm_span_of(ids):
    pads = torch.nonzero(ids == PAD).squeeze(1)
    return (int(pads[0]), int(pads[-1]) + 1)


def invariant_ok(ids, cl, span):
    total = int((ids == PAD).sum().item())
    row_pads = int((ids[cl:] == PAD).sum().item())
    ok = (row_pads == total) or (row_pads == 0 and cl >= span[1])
    return ok, row_pads, total


def no_double_ownership(world, tag):
    tree = world.tree_slots()
    free = set(world.cm.free_slots.tolist())
    dup_tree = len(tree) != len(set(tree))
    shared = len(free & set(tree))
    ok = (not dup_tree) and shared == 0
    check(f"{tag}.ownership", ok,
          f"tree_slots={len(tree)} dup_in_tree={dup_tree} free_and_tree={shared}")
    return ok


RESP = [100 + i for i in range(9)]
Q2 = list(range(5000, 5011))
Q3 = list(range(6000, 6011))
print("scaffold ok")

# ================== A: TWO PAD SPANS (ps=8) ==================
# layout: 40 text | S1 64 pads [40,104) | 30 text | S2 64 pads [134,198) | 10 text = 208
# turn-1 track = ((208-1)//64)*64 = 192 -> strictly INSIDE S2 (in-span, 2nd image)
PRE = list(range(1000, 1000 + 40))
MID = list(range(2000, 2000 + 30))
T10 = list(range(3000, 3000 + 10))
P1 = torch.tensor(PRE + [PAD] * 64 + MID + [PAD] * 64 + T10, dtype=torch.int32)
assert len(P1) == 208
SP1 = mm_span_of(P1)
ALLI = list(P1) + [100 + i for i in range(80)]
w = World(ALLI, 8)

req1, mr1 = admit(w, 1, P1, key_of(P1, CID_A), n_gen=4)
check("A.i.cold", mr1.cuda_handle.cached_len == 0 and mr1.mamba_value is None,
      f"cached_len={mr1.cuda_handle.cached_len}")
ok, rp, tot = invariant_ok(P1, 0, SP1)
check("A.i.cold_invariant", ok, f"row_pads={rp} total={tot}")
fin1, trk1 = run_turn(w, req1, 208, 4)
check("A.i.track_in_2nd_span", trk1 == 192 and 134 < 192 < 198,
      f"track={trk1} inside S2 [134,198): in-span snapshot on the SECOND image")
ok = w.pages_conserved()
check("A.i.t1_conserved", ok[0], ok[1])

P2 = torch.cat([P1, torch.tensor(RESP + Q2, dtype=torch.int32)])
SP2 = mm_span_of(P2)
raw2 = w.pc.match_prefix(key_of(P2, CID_A)[: len(P2) - 1])
check("A.i.raw_match_2nd_span", raw2.cached_len == 192 and raw2.mamba_value is not None,
      f"raw cached_len={raw2.cached_len} mamba={raw2.mamba_value} (live snapshot at 192)")

req2, mr2 = admit(w, 2, P2, key_of(P2, CID_A), n_gen=4)
cl2 = mr2.cuda_handle.cached_len
check("A.i.pulled_to_span_start", cl2 == 40,
      f"admission cached_len={cl2} (raw 192 in 2nd span -> align_down(40,8)=40)")
check("A.i.mamba_dropped", mr2.mamba_value is None,
      f"(raw match had live snapshot {raw2.mamba_value}; the cap must drop it)")
ok, rp2, tot2 = invariant_ok(P2, cl2, SP2)
check("A.i.splice_invariant", ok, f"row_pads={rp2} total={tot2} (both images fully in rows)")

live2 = req2.linear_slot_idx
restore_src = mr2.mamba_value
if restore_src is not None:
    w.pool.copy_from(restore_src, live2)
w.advance(live2, len(P2))
check("A.i.cow_noop_live_state",
      w.fp(live2) == fp32(sum(ALLI[:len(P2)])) and w.fp(live2) != fp32(sum(ALLI[:192])),
      f"live fp={w.fp(live2)} fwd_end={sum(ALLI[:len(P2)])} donated@192={w.fp(raw2.mamba_value)} "
      f"(no over-advance restore; restore_src was {restore_src})")
fin2, trk2 = run_turn(w, req2, len(P2), 4)
ok = w.pages_conserved()
check("A.i.t2_conserved", ok[0], ok[1])
no_double_ownership(w, "A.i.t2")

# A.ii: turn 2 with a DIFFERENT second image -> walk divergence inside the gap
W2 = World(list(P1) + [100 + i for i in range(80)], 8)
req1b, mr1b = admit(W2, 1, P1, key_of(P1, CID_A))
fin1b, _ = run_turn(W2, req1b, 208, 9)
P2b = torch.cat([P1, torch.tensor(RESP + Q2, dtype=torch.int32)])
KEY2B = key_of(P2b, CID_A)
KEY2B[134:198] = CID_B
req2b, mr2b = admit(W2, 2, P2b, KEY2B)
cl2b = mr2b.cuda_handle.cached_len
okb, rp2b, tot2b = invariant_ok(P2b, cl2b, SP2)
check("A.ii.diff_img2_safe", okb and cl2b <= 40,
      f"cached_len={cl2b} row_pads={rp2b} total={tot2b} "
      f"(divergence at S2 start 134: snapshot stays on split suffix -> no false resumption)")
fin2b, _ = run_turn(W2, req2b, len(P2b), 9)
ok = W2.pages_conserved()
check("A.ii.conserved", ok[0], ok[1])

# A.iii: track in the TAIL (past both spans) -> pass-through, 0 pads, mamba live
GAP = list(range(2000, 2000 + 13))
TAIL50 = list(range(3000, 3000 + 50))
P1c = torch.tensor(PRE + [PAD] * 64 + GAP + [PAD] * 64 + TAIL50, dtype=torch.int32)
assert len(P1c) == 231
SPC = mm_span_of(P1c)
ALLI_C = list(P1c) + [100 + i for i in range(80)]
W3 = World(ALLI_C, 8)
req1c, mr1c = admit(W3, 1, P1c, key_of(P1c, CID_A))
fin1c, trk1c = run_turn(W3, req1c, 231, 9)
check("A.iii.track_in_tail", trk1c == 192 and trk1c >= 181,
      f"track={trk1c} past the last span (tail [181,231))")
P2c = torch.cat([P1c, torch.tensor(RESP + Q2, dtype=torch.int32)])
req2c, mr2c = admit(W3, 2, P2c, key_of(P2c, CID_A))
cl2c = mr2c.cuda_handle.cached_len
okc, rp2c, tot2c = invariant_ok(P2c, cl2c, SPC)
check("A.iii.past_span_passes", cl2c == 192 and rp2c == 0 and okc,
      f"cached_len={cl2c} clears both spans; row_pads={rp2c} (splice no-op, full reuse kept)")
check("A.iii.mamba_live_at_boundary",
      mr2c.mamba_value is not None and W3.fp(mr2c.mamba_value) == fp32(sum(ALLI_C[:192])),
      f"mamba={mr2c.mamba_value} fp == prefix sum @192 (restore lands exactly at cached end)")
W3.pool.copy_from(mr2c.mamba_value, req2c.linear_slot_idx)
W3.advance(req2c.linear_slot_idx, len(P2c))
check("A.iii.cow_then_forward_exact",
      W3.fp(req2c.linear_slot_idx) == fp32(sum(ALLI_C[:len(P2c)])),
      "COW restore @192 then prefill advance -> fp at the full prompt end")
fin2c, _ = run_turn(W3, req2c, len(P2c), 9)
ok = W3.pages_conserved()
check("A.iii.conserved", ok[0], ok[1])
no_double_ownership(W3, "A.iii")
print("section A done, failures:", len(FAIL))

# ================== D: splice-invariant fuzz, ps in {1,4,8} ==================
# shapes: single span w/ track mid-span (S1), long span w/ track near the end (S2),
# two spans w/ track past the 2nd span (S3). 4 same-image turns each; after EVERY
# match: pads-in-rows invariant; after EVERY turn: page conservation; after t2/t4:
# no double ownership.
SHAPES = {
    "S1": (10, 97, 15),     # T10 + 97 pads [10,107) + 15 tail -> plen 122, track 64 in-span
    "S2": (2, 192, 20),     # T2 + 192 pads [2,194) + 20 tail -> plen 214, track 192 in-span
    "S3": (40, 64, 13, 64, 50),  # T40 + 64p + 13t + 64p + 50t -> plen 231, track 192 past
}

for ps in (1, 4, 8):
    for sname, shp in SHAPES.items():
        if len(shp) == 3:
            head, span, tailn = shp
            ids = list(range(1000, 1000 + head)) + [PAD] * span + list(range(5000, 5000 + tailn))
        else:
            head, s1, mid, s2, tailn = shp
            ids = (list(range(1000, 1000 + head)) + [PAD] * s1
                   + list(range(2000, 2000 + mid)) + [PAD] * s2
                   + list(range(5000, 5000 + tailn)))
        p0 = torch.tensor(ids, dtype=torch.int32)
        plen0 = len(p0)
        span0 = mm_span_of(p0)
        track1 = ((plen0 - 1) // 64) * 64
        allids = list(p0) + [100 + i for i in range(240)]
        W = World(allids, ps)
        cur = p0
        prev_fin = None
        tag = f"D.ps{ps}.{sname}"
        ok = W.pages_conserved()
        check(f"{tag}.t0_conserved", ok[0], ok[1])
        for t in (1, 2, 3, 4):
            plen_t = len(cur)
            # aligned finish on turn 3 so turn 4 can match past the span
            if t == 3:
                n_gen = (1 - plen_t) % ps
                if n_gen == 0:
                    n_gen = ps
            else:
                n_gen = 9
            key_t = key_of(cur, CID_A)
            req_t, mr_t = admit(W, 100 + t, cur, key_t, n_gen=n_gen)
            cl = mr_t.cuda_handle.cached_len
            span_t = mm_span_of(cur)
            tot_pads = int((cur == PAD).sum().item())
            row_pads = int((cur[cl:] == PAD).sum().item())
            inv_ok = (row_pads == tot_pads) or (row_pads == 0 and cl >= span_t[1])
            check(f"{tag}.t{t}.invariant", inv_ok,
                  f"cached_len={cl} span={span_t} row_pads={row_pads} total={tot_pads} "
                  f"mamba={mr_t.mamba_value}")
            raw_t = W.pc.match_prefix(key_t[: plen_t - 1])
            if raw_t.cached_len != 0 and span_t[0] <= raw_t.cached_len < span_t[1]:
                want = (span_t[0] // ps) * ps
                check(f"{tag}.t{t}.capped", cl == want and mr_t.mamba_value is None,
                      f"raw={raw_t.cached_len} in-span -> cl={cl} want {want}, mamba dropped")
            elif raw_t.cached_len >= span_t[1]:
                check(f"{tag}.t{t}.past", cl == raw_t.cached_len and row_pads == 0,
                      f"raw={raw_t.cached_len} past span -> unchanged, 0 pads in rows")
                if mr_t.mamba_value is not None:
                    check(f"{tag}.t{t}.fp_exact",
                          W.fp(mr_t.mamba_value) == fp32(sum(allids[:cl])),
                          f"snapshot fp == prefix sum @ matched boundary {cl}")
            fin_t, trk_t = run_turn(W, req_t, plen_t, n_gen)
            if t == 2 or t == 4:
                no_double_ownership(W, f"{tag}.t{t}")
            ok = W.pages_conserved()
            check(f"{tag}.t{t}.conserved", ok[0], ok[1])
            cur = torch.cat([cur, torch.tensor(RESP + Q2, dtype=torch.int32)])
            prev_fin = fin_t

# ================== E: text uncapped + keyless mm ==================
# E1: text control in hybrid world -- snapshot-node reuse, no mm cap may apply
TXT = list(range(7000, 7064))
ALLI_T = TXT + [100 + i for i in range(60)]
WT = World(ALLI_T, 8)
pt1 = torch.tensor(TXT, dtype=torch.int32)
rt1, mrt1 = admit(WT, 1, pt1, None, mm=False)
check("E1.cold", mrt1.cuda_handle.cached_len == 0, f"cached_len={mrt1.cuda_handle.cached_len}")
fin_t1, _ = run_turn(WT, rt1, 64, 9)   # fin = 72, page-aligned -> finish donate @72
cont = torch.tensor(TXT + [100 + i for i in range(9)] + Q2, dtype=torch.int32)
rt2, mrt2 = admit(WT, 2, cont, None, mm=False)
check("E1.text_72_reuse", mrt2.cuda_handle.cached_len == 72 and mrt2.mamba_value is not None,
      f"cached_len={mrt2.cuda_handle.cached_len} mamba={mrt2.mamba_value} "
      f"(snapshot node @72 reused; mm cap must not touch text)")
check("E1.text_fp_exact", WT.fp(mrt2.mamba_value) == fp32(sum(ALLI_T[:72])),
      "snapshot fp == prefix sum @72")
fin_t2, _ = run_turn(WT, rt2, len(cont), 9)
ok = WT.pages_conserved()
check("E1.conserved", ok[0], ok[1])

# E2: keyless mm -- no cross-request reuse even against a warm tree
PK = torch.tensor(list(range(4000, 4010)) + [PAD] * 64 + list(range(4100, 4110)),
                  dtype=torch.int32)
ALLI_K = list(PK) + [100 + i for i in range(60)]
WK = World(ALLI_K, 8)
rk1, mrk1 = admit(WK, 1, PK, key_of(PK, CID_A))
fin_k1, _ = run_turn(WK, rk1, len(PK), 9)
rk2, mrk2 = admit(WK, 2, PK, None, mm=True)   # keyless, identical raw ids
check("E2.keyless_no_reuse", mrk2.cuda_handle.cached_len == 0,
      f"cached_len={mrk2.cuda_handle.cached_len} (keyless mm never reuses; tree is warm)")
ok, rpk, totk = invariant_ok(PK, 0, mm_span_of(PK))
check("E2.keyless_invariant", ok, f"row_pads={rpk} total={totk}")
fin_k2, _ = run_turn(WK, rk2, len(PK), 9)
ok = WK.pages_conserved()
check("E2.conserved", ok[0], ok[1])
no_double_ownership(WK, "E2")

print()
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("RESULT: ALL PASS")
