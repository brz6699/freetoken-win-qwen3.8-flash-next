"""Minimal: S3 shape (231), 2 turns, print slot fingerprints at every step."""
import sys
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "package"))
from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq

PAD = 248056
N_PAGES = 512
CID_A = 1_000_000_001
PS = 8

def make_group():
    return LinearGatedDeltaGroupConfig(
        name="g", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=2, output_gate="silu")

ids_all = (list(range(1000, 1040)) + [PAD] * 64 + list(range(2000, 2013))
           + [PAD] * 64 + list(range(5000, 5050)) + [100 + i for i in range(80)])
pool = LinearStatePool(make_group(), num_slots=16, dtype=torch.float32,
                       device=torch.device("cpu"), tp_size=1)
page_table = torch.zeros(32, N_PAGES * PS, dtype=torch.int32)
cm = CacheManager(N_PAGES, PS, page_table, "hybrid_radix", linear_state_pool=pool)
pc = cm.prefix_cache

def fp(slot):
    return float(pool.conv_states[0, slot, 0, 0])

def advance(slot, n):
    pool.conv_states[0, slot, 0, 0] = float(sum(ids_all[:n]))

def snapshot(tag):
    vals = {s: fp(s) for s in range(1, 16) if fp(s) != 0.0}
    snaps = [(n.uuid, len(n.value), fp(n.mamba_value) if n.mamba_value is not None else None,
              n.mamba_value) for n in pc._snapshot_nodes()]
    print(f"{tag}: nonzero_slots={vals} snapshot_nodes={snaps}")

P1 = torch.tensor(ids_all[:231], dtype=torch.int32)
def key_of(ids):
    k = ids.clone()
    k[ids == PAD] = CID_A
    return k

def admit(uid, ids, key, n_gen):
    sp = SamplingParams(max_tokens=n_gen)
    pr = PendingReq(uid, ids, sp, mm_embeds=torch.zeros(4, 8), mm_cache_key=key)
    mr = cm.match_req(pr)
    cm.lock(mr.cuda_handle)
    live = pool.alloc(1)[0]
    pp = tuple(pool.alloc(2))
    req = Req(input_ids=ids, table_idx=0, cached_len=mr.cuda_handle.cached_len,
              output_len=n_gen, uid=uid, sampling_params=sp, cache_handle=mr.cuda_handle,
              mm_embeds=torch.zeros(4, 8), mm_cache_key=key,
              linear_slot_idx=live, mamba_ping_pong=pp)
    cm.allocate_paged([req])
    print(f"admit{uid}: cached_len={mr.cuda_handle.cached_len} mamba={mr.mamba_value} "
          f"live={live} pp={pp} tree_nodes={[(n.uuid, len(n.value)) for n in pc._snapshot_nodes()]}")
    return req, mr

def run_turn(req, plen, n_gen):
    live = req.linear_slot_idx
    cached0 = req.cached_len
    advance(live, plen)
    print(f"  after prefill-advance: fp(live)={fp(live)} want={sum(ids_all[:plen])}")
    extend = plen - cached0
    c = (extend - 1) // 64
    track = None
    if c >= 1:
        track = cached0 + c * 64
        frozen = req.mamba_ping_pong[req.mamba_next_track_idx]
        advance(frozen, track)
        req.mamba_last_track_seqlen = track
        req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
        print(f"  track={track} frozen={frozen} fp(frozen)={fp(frozen)} want={sum(ids_all[:track])}")
    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))
    cm.cache_req(req, finished=False)
    snapshot("  after chunk-commit")
    fin = plen + n_gen - 1
    for j in range(1, n_gen):
        advance(live, plen + j)
        req.append_host(torch.tensor([100 + j], dtype=torch.int32))
        if plen + j < fin:
            cm.allocate_paged([req])
        req.complete_one()
    print(f"  after decode: fp(live)={fp(live)} want={sum(ids_all[:fin])} fin={fin}")
    cm.cache_req(req, finished=True)
    snapshot("  after finish")
    return fin

resp = [100 + i for i in range(9)]
q2 = list(range(5000, 5011))
ids_all = ids_all[:231] + resp + q2 + [9000 + i for i in range(40)]
print("ids_all len:", len(ids_all))
P1 = torch.tensor(ids_all[:231], dtype=torch.int32)
req1, mr1 = admit(1, P1, key_of(P1), 9)
snapshot("after admit1")
run_turn(req1, 231, 9)
P2 = torch.tensor(ids_all[:251], dtype=torch.int32)
req2, mr2 = admit(2, P2, key_of(P2), 9)
print(f"t2 match mamba={mr2.mamba_value} fp={fp(mr2.mamba_value)} vs sum231={sum(ids_all[:231])} "
      f"vs sum192={sum(ids_all[:192])}")
snapshot("after admit2 (before t2 forward)")
