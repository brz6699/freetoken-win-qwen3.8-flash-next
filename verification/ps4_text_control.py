"""Text-only control for the ps=4 non-page-aligned finish follow-up boundary.

Mirrors the verifier's ps=4 world exactly (window=16, ps=4, P=82 text prompt,
40 decode steps with driver every forward, finish at cached_len=123), but with
a TEXT request (no mm). If the follow-up match also returns 80 (not 81), the
1-token loss is pre-existing page-alignment engine semantics, not P4.
"""
import os
os.environ["FREETOKEN_SWA_EVICTION_INTERVAL"] = "1"
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))
from freetoken.distributed import set_tp_info

try:
    set_tp_info(0, 1)
except RuntimeError:
    pass

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
from freetoken.models.config import KVCacheGroupSpec
from freetoken.scheduler.utils import PendingReq

WIN, PS = 16, 4
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

P = 82
ids = torch.tensor(list(range(10, 20)) + list(range(30, 94)), dtype=torch.int32)
print(f"prompt_len={len(ids)} (not page-aligned), ps={PS}")

# --- admission (cold) ---
sp = SamplingParams(max_tokens=256)
pr = PendingReq(1, ids, sp)
mr = cm.match_req(pr)
print("cold match:", mr.cuda_handle.cached_len)
req = Req(input_ids=ids, table_idx=0, cached_len=0, output_len=256, uid=1,
          sampling_params=sp, cache_handle=mr.cuda_handle)

# --- prefill single chunk ---
cm.allocate_paged([req])
req.complete_one()
cm.allocate_paged([req])
req.append_host(torch.tensor([100], dtype=torch.int32))
cm.cache_req(req, finished=False)
print("chunk commit floor:", req.cache_handle.cached_len)

# --- 40 decode steps, driver every forward ---
for _ in range(40):
    req.complete_one()
    cm.allocate_paged([req])
    req.append_host(torch.tensor([9999], dtype=torch.int32))
    req.decode_batch_idx += 1
    cm.maybe_free_swa_out_of_window([req], forward_iter=req.decode_batch_idx)
print("swa_evicted:", req.swa_evicted_seqlen)

# --- finish at cached_len=123 (not page-aligned) ---
req.complete_one()
cm.cache_req(req, finished=True)
try:
    cm.check_integrity()
    print("integrity_at_finish: OK")
except AssertionError as e:
    print("integrity_at_finish: FAIL", e)
tree_swa = cm.prefix_cache.swa_evictable + cm.prefix_cache.swa_protected
print(f"pool: free={pool.swa_available_size()} tree_swa={tree_swa} cap={cap}")

# --- follow-up: same prompt, diverges at 82 (client drops the 122-token response) ---
t2 = torch.cat([ids, torch.tensor([999, 998], dtype=torch.int32)])
pr2 = PendingReq(2, t2, SamplingParams(max_tokens=8))
mr2 = cm.match_req(pr2)
cl = mr2.cuda_handle.cached_len
print(f"TEXT follow-up match cached_len={cl} (mm probe got 80; verifier expected 81)")
kv = mr2.cuda_handle.kv_indices
inwin = [i for i in range(max(0, cl - 1 - WIN), cl)
         if int(pool.full_to_swa_index_mapping[int(kv[i])]) == 0]
print(f"follow-up window live: dangling={len(inwin)}")

# --- keyless sanity: different prefix no false hit ---
pr3 = PendingReq(3, torch.cat([torch.tensor([7777], dtype=torch.int32), t2[1:]]),
                 SamplingParams(max_tokens=8))
mr3 = cm.match_req(pr3)
print("diverge-at-0 match:", mr3.cuda_handle.cached_len)

ok = True
try:
    cm.check_integrity()
    print("integrity_final: OK")
except AssertionError as e:
    ok = False
    print("integrity_final: FAIL", e)
print("RESULT:", "ALL OK" if ok else "FAIL")
