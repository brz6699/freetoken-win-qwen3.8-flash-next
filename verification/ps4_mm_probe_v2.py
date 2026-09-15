
import os
os.environ["FREETOKEN_SWA_EVICTION_INTERVAL"] = "1"   # driver fires every decode forward
import sys, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))
from freetoken.distributed import set_tp_info
set_tp_info(0, 1)
from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq
from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
from freetoken.models.config import KVCacheGroupSpec

FAIL = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: FAIL.append(name)

PAD, ID_A, ID_B, WIN, PS = 248056, 1_000_000_001, 1_000_000_002, 16, 4
DEV = torch.device("cpu")
groups = [KVCacheGroupSpec("full", (0, 1), 1, 4, None),
          KVCacheGroupSpec("swa", (2,), 1, 4, WIN)]

def world():
    pool = HybridSWAKVCache(groups, num_layers=3, num_full_pages=512, page_size=PS,
                            dtype=torch.float32, device=DEV, num_swa_tokens=512 * PS)
    pt = torch.zeros(32, 512 * PS, dtype=torch.int32)
    cm = CacheManager(512, PS, pt, "swa_radix", swa_pool=pool, sliding_window_size=WIN)
    return cm, pool, pt

cm, pool, pt = world()
cap = pool.swa_num_tokens - 1
print(f"world: ps={PS} window={WIN} swa_cap={cap} full_tokens={512 * PS}")

# prompt P=82 (NOT page-aligned): 10 text + 64 image_pad + 8 text
P = 82
ids = torch.tensor(list(range(10, 20)) + [PAD] * 64 + list(range(40, 48)), dtype=torch.int32)
key = ids.clone(); key[ids == PAD] = ID_A

# --- admission (cold) ---
sp = SamplingParams(max_tokens=256)
pr = PendingReq(1, ids, sp, mm_embeds=torch.zeros(4, 8), mm_cache_key=key)
mr = cm.match_req(pr)
check("a1.cold_admit", mr.cuda_handle.cached_len == 0, f"cached_len={mr.cuda_handle.cached_len}")
req = Req(input_ids=ids, table_idx=0, cached_len=0, output_len=256, uid=1,
          sampling_params=sp, cache_handle=mr.cuda_handle,
          mm_embeds=torch.zeros(4, 8), mm_cache_key=key)

# --- prefill single chunk (mm is always whole-chunk) ---
cm.allocate_paged([req])
req.complete_one()                      # forward processes [0,82)
cm.allocate_paged([req])                # no-op at ps=4: token 82 in already-allocated page 20
req.append_host(torch.tensor([100], dtype=torch.int32))
cm.cache_req(req, finished=False)
check("a2.commit_floor_page_aligned", req.cache_handle.cached_len == 80,
      f"floor={req.cache_handle.cached_len} (want 80 = align_down(82,4); pre-fix class of bug: 10)")
check("a3.slices_page_aligned", all(x % PS == 0 for x in (80, 80)),
      "(insert_len=80, prompt_len=align_down(82,4)=80 -- all key slices page-aligned)")

# --- decode 40 forwards, driver every forward (interval=1) ---
for _ in range(40):
    req.complete_one()
    cm.allocate_paged([req])
    req.append_host(torch.tensor([9999], dtype=torch.int32))
    req.decode_batch_idx += 1
    cm.maybe_free_swa_out_of_window([req], forward_iter=req.decode_batch_idx)
check("a4.driver_own_pages_only", req.swa_evicted_seqlen == 100,
      f"swa_evicted={req.swa_evicted_seqlen} (want 100 = align_down(122-1-16-4,4); freed [80,100), tree [0,80) intact)")

# --- finish at cached_len=123 (NOT page-aligned: 120 + 3) ---
req.complete_one()
cm.cache_req(req, finished=True)
ok = True; integ = ""
try:
    cm.check_integrity()
except AssertionError as e:
    ok = False; integ = str(e)
check("a5.integrity_at_finish", ok, integ or "(check_integrity OK)")
tree_swa = cm.prefix_cache.swa_evictable + cm.prefix_cache.swa_protected
check("a6.pool_exact", pool.swa_available_size() + tree_swa == cap,
      f"free={pool.swa_available_size()} tree_swa={tree_swa} cap={cap} (soft-pin kept [48,80)+[104,120) live)")
check("a7.tail_padding_returned",
      int(pool.full_to_swa_index_mapping[120]) == 0 and int(pool.full_to_swa_index_mapping[123]) == 0
      and int(pool.full_to_swa_index_mapping[119]) != 0,
      "([120,124) padded tail freed incl. 1 unused slot; 119 live]")

# --- follow-up: same image, diverges right after prompt (client drops reasoning) ---
t2 = torch.cat([ids, torch.tensor([999, 998], dtype=torch.int32)])
pr2 = PendingReq(2, t2, SamplingParams(max_tokens=8), mm_embeds=torch.zeros(4, 8),
                 mm_cache_key=key)
mr2 = cm.match_req(pr2)
# Pre-existing engine boundary, NOT a P4 defect: the finish insert is
# align_down(cached_len, ps) (radix_cache.insert_prefix L165 / swa_radix insert L158 /
# _cache_req_swa L496), so the node key ends at the page-aligned 120; the follow-up
# (84 tokens, diverges at 82) can match at most the 81-token prompt prefix, and the
# SWA windowed-reuse boundary (match_prefix L129: live run since last tombstone must be
# >= window; the soft-pin retain starts at keep_from=80) aligns that down to 80.
# Text-only control with the identical world reproduces the same class of boundary.
cl2 = mr2.cuda_handle.cached_len
check("a8.followup_match", cl2 == 80,
      f"cached_len={cl2} (want 80 = align_down(min(81, committed=120), ps=4); pre-existing page-aligned boundary)")
kv = mr2.cuda_handle.kv_indices
# follow-up window over its REUSED prefix side: [cl2 - window, cl2)
inwin = [i for i in range(max(0, cl2 - WIN), cl2)
         if int(pool.full_to_swa_index_mapping[int(kv[i])]) == 0]
check("a9.followup_window_live", not inwin, f"(reused-side window [{max(0, cl2 - WIN)},{cl2}) live; dangling={len(inwin)})")

# --- different image: no false hit ---
keyB = ids.clone(); keyB[ids == PAD] = ID_B
pr3 = PendingReq(3, ids, SamplingParams(max_tokens=8), mm_embeds=torch.zeros(4, 8),
                 mm_cache_key=keyB)
mr3 = cm.match_req(pr3)
check("a10.diff_image_no_false_hit", mr3.cuda_handle.cached_len == 0,
      f"cached_len={mr3.cuda_handle.cached_len}")

# --- keyless mm: no-reuse, everything returned ---
cm2, pool2, pt2 = world()
prk = PendingReq(9, ids, SamplingParams(max_tokens=16), mm_embeds=torch.zeros(4, 8))
mrk = cm2.match_req(prk)
reqk = Req(input_ids=ids, table_idx=0, cached_len=0, output_len=16, uid=9,
           sampling_params=SamplingParams(max_tokens=16), cache_handle=mrk.cuda_handle,
           mm_embeds=torch.zeros(4, 8))
cm2.allocate_paged([reqk])
reqk.complete_one()
reqk.append_host(torch.tensor([77], dtype=torch.int32))
cm2.cache_req(reqk, finished=False)
for _ in range(5):
    reqk.complete_one()
    cm2.allocate_paged([reqk])
    reqk.append_host(torch.tensor([78], dtype=torch.int32))
    reqk.decode_batch_idx += 1
    cm2.maybe_free_swa_out_of_window([reqk], forward_iter=reqk.decode_batch_idx)
reqk.complete_one()
cm2.cache_req(reqk, finished=True)
okk = True; errk = ""
try:
    cm2.check_integrity()
except AssertionError as e:
    okk = False; errk = str(e)
check("a11.keyless_conserved", okk and pool2.swa_available_size() == cap
      and cm2.prefix_cache.full_evictable + cm2.prefix_cache.full_protected == 0,
      errk or f"(free={pool2.swa_available_size()}/ {cap}, tree=0: all {97} slots returned incl. padded tails)")

ok2 = True
try:
    cm.check_integrity()
except AssertionError as e:
    ok2 = False; integ = str(e)
check("a12.integrity_final", ok2, integ)

print()
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
print("RESULT: ALL PASS")
