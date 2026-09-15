"""P4 splice-invariant regression (production crash reproduction).

The live-serve crash (2026-09-10, turn 2 of same-image multi-turn):
  RuntimeError: multimodal splice mismatch: 10 image_pad token(s) in the batch rows
                but 1000 soft-token row(s)
Root cause: the GDN chunk commit donates a snapshot at the deepest mid-prefill x64
boundary (linear.py: c = (extend_len-1)//64; boundary = cached_len + c*64) -- which
lands MID-PAD-SPAN for a vision prompt. A follow-up turn then matches to that
mid-span node; its prefill rows hold only the span's tail while the worker always
supplies ALL of the image's soft rows -> the model's splice count invariant breaks.

Fix under test (CacheManager.match_req): an mm match may either stay BEFORE the first
pad run (whole span re-prefilled, all pads in the rows) or CLEAR the last pad run
(span fully cached, zero pads in the rows, model scatters nothing); a boundary in
between is pulled back to align_down(span_start, page_size) and drops the donated
GDN snapshot (mamba_value) so it is never COW-restored past the real boundary.

This test drives the REAL CacheManager(hybrid_radix, page_size=8) + REAL
LinearStatePool through the exact engine lifecycle and asserts the SPICE INVARIANT
(the model-side contract) after every match:

    n_pads(rows [cached_len, input_len)) == mm_rows   OR   n_pads == 0 with
    cached_len >= span_end   (span fully cached)

Legs:
  T1  turn 1 (cold, unaligned finish) -> mid-span chunk-commit node exists
  T2  turn 2 same image: match falls in the mid-span node's territory -> pulled back
      to before the span; ALL pads in the rows; mamba_value dropped; full lifecycle OK
  T3  turn 3 same image (aligned finish B2 cached): FULL-prefix reuse across the span
      -> zero pads in the rows, splice no-op, mamba restore consistent at B2
  T4  different image -> no false hit past the text prefix
  T5  keyless mm -> still no reuse
  T6  text control -> match behavior unchanged (no cap applies)

  python -X utf8 verification/p4_splice_test.py
"""

import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq

FAIL = []
PAD = 248056          # production image_pad id
PS = 8                # production-class page size (unaligned finishes)
N_PAGES = 512


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def make_group():
    return LinearGatedDeltaGroupConfig(
        name="g", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=2, output_gate="silu",
    )


class World:
    def __init__(self, ids_all):
        self.ids_all = ids_all
        self.pool = LinearStatePool(make_group(), num_slots=16, dtype=torch.float32,
                                    device=torch.device("cpu"), tp_size=1)
        self.page_table = torch.zeros(32, N_PAGES * PS, dtype=torch.int32)
        self.cm = CacheManager(N_PAGES, PS, self.page_table, "hybrid_radix",
                               linear_state_pool=self.pool)
        self.pc = self.cm.prefix_cache

    def fp(self, slot):
        return float(self.pool.conv_states[0, slot, 0, 0])

    def advance(self, slot, n):
        self.pool.conv_states[0, slot, 0, 0] = float(sum(self.ids_all[:n]))

    def pages_conserved(self):
        tree_tokens = self.pc.full_evictable + self.pc.full_protected
        held = len(self.cm.free_slots) + tree_tokens // PS
        return tree_tokens % PS == 0 and held == N_PAGES, (
            f"free={len(self.cm.free_slots)} tree={tree_tokens}tok held={held}/{N_PAGES}")


def admit(world, uid, ids, key, mm=True, n_gen=9, plen=None):
    """PrefillAdder-shaped admission: match_req + Req with fresh GDN slots."""
    sp = SamplingParams(max_tokens=n_gen)
    pr = PendingReq(uid, ids, sp,
                    mm_embeds=torch.zeros(4, 8) if mm else None,
                    mm_cache_key=key if mm else None)
    mr = world.cm.match_req(pr)
    world.cm.lock(mr.cuda_handle)          # PrefillAdder (prefill.py:69)
    live = world.pool.alloc(1)[0]
    pp = tuple(world.pool.alloc(2))
    req = Req(input_ids=ids, table_idx=0, cached_len=mr.cuda_handle.cached_len,
              output_len=n_gen, uid=uid, sampling_params=sp,
              cache_handle=mr.cuda_handle,
              mm_embeds=torch.zeros(4, 8) if mm else None,
              mm_cache_key=key if mm else None,
              linear_slot_idx=live, mamba_ping_pong=pp)
    plen = plen if plen is not None else len(ids)
    world.cm.allocate_paged([req])
    return req, mr


def run_turn(world, req, plen, n_gen):
    """Single-chunk prefill + n_gen generated tokens, engine-exact (hybrid path)."""
    live = req.linear_slot_idx
    cached0 = req.cached_len
    world.advance(live, plen)
    # prefill forward: GDN x64 track at the deepest mid-extend boundary (linear.py:118-122)
    extend = plen - cached0
    c = (extend - 1) // 64
    track = None
    if c >= 1:
        track = cached0 + c * 64
        frozen = req.mamba_ping_pong[req.mamba_next_track_idx]
        world.advance(frozen, track)
        req.mamba_last_track_seqlen = track
        req.mamba_next_track_idx = 1 - req.mamba_next_track_idx

    # prefill forward tail: KV valid to plen, first token (100) sampled at plen
    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))
    world.cm.cache_req(req, finished=False)   # chunk commit (after first append)
    fin = plen + n_gen - 1
    for j in range(1, n_gen):
        world.advance(live, plen + j)
        req.append_host(torch.tensor([100 + j], dtype=torch.int32))
        if plen + j < fin:
            world.cm.allocate_paged([req])
        req.complete_one()
    assert req.cached_len == fin, f"cached_len={req.cached_len} want {fin}"
    world.cm.cache_req(req, finished=True)
    return fin, track


def n_pads_in_rows(ids, cached_len, row_end):
    return int((ids[cached_len:row_end] == PAD).sum().item())


# ===================================================================== #
# T1: turn 1 -- cold mm prompt whose GDN track lands MID-SPAN           #
# ===================================================================== #
PRE = 10                                  # text before the span
SPAN = 955                                 # image_pad run (not a multiple of 64: the x64
                                           # GDN track 960 then lands strictly INSIDE it)
post = list(range(3000, 3015))             # question text after the span (15 tokens)
prompt1 = torch.tensor(list(range(1000, 1000 + PRE)) + [PAD] * SPAN + post,
                       dtype=torch.int32)
assert len(prompt1) == 980
span_start, span_end = PRE, PRE + SPAN     # 10 .. 965
key1 = prompt1.clone()
key1[prompt1 == PAD] = 1_000_000_001

all_ids = list(prompt1) + [100 + i for i in range(40)]
world = World(all_ids)

req1, mr1 = admit(world, 1, prompt1, key1, n_gen=9)
check("T1.cold", mr1.cuda_handle.cached_len == 0 and mr1.mamba_value is None,
      f"cached_len={mr1.cuda_handle.cached_len}")
rows = n_pads_in_rows(prompt1, 0, len(prompt1))
check("T1.cold_rows_all_pads", rows == SPAN, f"pads in rows={rows} (want {SPAN})")

fin1, track1 = run_turn(world, req1, len(prompt1), n_gen=9)
check("T1.finish_unaligned", fin1 % PS != 0, f"fin={fin1} (unaligned -> no finish-donate)")
check("T1.midspan_track", track1 is not None and span_start < track1 < span_end,
      f"track={track1} in span ({span_start},{span_end}) -- the production killer shape")
ok = world.pages_conserved()
check("T1.pages_conserved", ok[0], ok[1])

# ===================================================================== #
# T2: turn 2, SAME image -- the crash leg.                              #
# input = prompt1 + turn-1 response (9 tokens, re-tokenized) + q2 (11)   #
# ===================================================================== #
resp1 = [100 + i for i in range(9)]        # turn-1's generated tokens (re-tokenized)
q2 = list(range(5000, 5011))
prompt2 = torch.cat([prompt1, torch.tensor(resp1 + q2, dtype=torch.int32)])
key2 = prompt2.clone()
key2[prompt2 == PAD] = 1_000_000_001       # same image -> same content id

req2, mr2 = admit(world, 2, prompt2, key2, n_gen=9)
m2 = mr2.cuda_handle.cached_len
raw = world.pc.match_prefix(key2[: len(prompt2) - 1]).cached_len
check("T2.raw_match_midspan", span_start <= raw < span_end,
      f"raw tree match={raw} falls INSIDE the pad span ({span_start},{span_end}) "
      f"-- the pre-fix admission reused it and the splice crashed (production: 10 vs 1000)")
check("T2.admission_pulled_back", m2 == (span_start // PS) * PS,
      f"admission cached_len={m2} pulled back to align_down({span_start},{PS})")
check("T2.mamba_dropped", mr2.mamba_value is None,
      "(donated snapshot at the uncapped boundary must not be restored)")
row_end = len(prompt2)
pads2 = n_pads_in_rows(prompt2, m2, row_end)
check("T2.splice_invariant",
      pads2 == SPAN or (pads2 == 0 and m2 >= span_end),
      f"pads in rows={pads2}, mm_rows={SPAN} -- the production crash was 10 vs 1000 here")
if pads2 == SPAN:
    # full re-prefill of the span: everything from the boundary is request-owned
    check("T2.prefix_before_span", m2 <= span_start, f"boundary {m2} before the span")

fin2, track2 = run_turn(world, req2, len(prompt2), n_gen=9)
ok = world.pages_conserved()
check("T2.pages_conserved", ok[0], ok[1])

# ===================================================================== #
# T3: turn 3, SAME image -- aligned finish B2 => FULL-prefix reuse      #
# across the span: zero pads in the rows, splice is a no-op.            #
# ===================================================================== #
q3 = list(range(6000, 6011))
prompt3 = torch.cat([prompt2, torch.tensor(q3, dtype=torch.int32)])
# make B2 page-aligned: fin3 = len(prompt3) + n_gen - 1 ≡ 0 (mod PS)
n_gen3 = (1 - len(prompt3)) % PS
if n_gen3 == 0:
    n_gen3 = PS
key3 = prompt3.clone()
key3[prompt3 == PAD] = 1_000_000_001

req3, mr3 = admit(world, 3, prompt3, key3, n_gen=n_gen3)
m3 = mr3.cuda_handle.cached_len
check("T3.full_span_reuse", m3 >= span_end,
      f"cached_len={m3} clears the last pad run (span fully cached; want >= {span_end})")
pads3 = n_pads_in_rows(prompt3, m3, len(prompt3))
check("T3.zero_pads_rows", pads3 == 0,
      f"pads in rows={pads3} (splice no-op: mm rows redundant, KV in cached pages)")
check("T3.mamba_consistent", mr3.mamba_value is not None,
      "(restore the donated state exactly at the matched boundary)")
if mr3.mamba_value is not None:
    check("T3.mamba_at_boundary", world.fp(mr3.mamba_value) == float(sum(all_ids[:m3])),
          "state fingerprint == prefix sum at the matched boundary (no over-advance)")

fin3, track3 = run_turn(world, req3, len(prompt3), n_gen=n_gen3)
ok = world.pages_conserved()
check("T3.pages_conserved", ok[0], ok[1])

# ===================================================================== #
# T4: different image -- no false hit past the shared text prefix       #
# ===================================================================== #
prompt4 = torch.cat([prompt1, torch.tensor(resp1 + q2, dtype=torch.int32)])
key4 = prompt4.clone()
key4[prompt4 == PAD] = 1_000_000_002       # DIFFERENT image
req4, mr4 = admit(world, 4, prompt4, key4, n_gen=4)
m4 = mr4.cuda_handle.cached_len
check("T4.no_false_hit", m4 < span_start,
      f"cached_len={m4} stops before the pad span (content ids differ)")
pads4 = n_pads_in_rows(prompt4, m4, len(prompt4))
check("T4.splice_invariant", pads4 == SPAN or (pads4 == 0 and m4 >= span_end),
      f"pads in rows={pads4}")

# ===================================================================== #
# T5: keyless mm -- still no cross-request reuse                        #
# ===================================================================== #
req5, mr5 = admit(world, 5, prompt1, None, mm=True, n_gen=4)
check("T5.keyless_no_reuse", mr5.cuda_handle.cached_len == 0,
      f"cached_len={mr5.cuda_handle.cached_len}")

# ===================================================================== #
# T6: text control -- no cap applies, plain prefix reuse intact         #
# ===================================================================== #
ttext = torch.tensor(list(range(7000, 7064)) + list(range(7064, 7100)), dtype=torch.int32)
# n_gen=9 -> fin = 64+9-1 = 72 (page-aligned finish-donate; unaligned finishes donate
# nothing -- pre-existing semantics, same class as the ps=4 probe)
rt, mrt = admit(world, 6, ttext[:64], None, mm=False, n_gen=9)
fin_t, _ = run_turn(world, rt, 64, n_gen=9)
# continuation re-tokenizes the same response (100..107) then new text: the only text
# reuse point is the snapshot node @72 (a mid-node match has no GDN snapshot to resume
# from, so a divergence before 72 correctly yields 0 -- unchanged text semantics)
cont = torch.cat([ttext[:64],
                  torch.tensor([100 + i for i in range(8)], dtype=torch.int32),
                  torch.tensor(list(range(7100, 7132)), dtype=torch.int32)])
rt2, mr2t = admit(world, 7, cont, None, mm=False, n_gen=4)
check("T6.text_reuse_unchanged",
      mr2t.cuda_handle.cached_len == 72 and mr2t.mamba_value is not None,
      f"cached_len={mr2t.cuda_handle.cached_len} mamba={mr2t.mamba_value} "
      "(snapshot node @72 reused; no mm cap applies)")

print()
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("RESULT: ALL PASS")
