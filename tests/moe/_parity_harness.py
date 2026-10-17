"""Standalone (pytest-free) check for the recurrence-window hybrid fetch path.

Runs the CPU reference mirror on CPU (works while the live serve holds the GPU) and
checks the recurrence-window gate: only misses whose expert was seen in a prior step earn a
fetch slot, capped by the bandwidth-balanced split; one-shot/stale misses stay non-resident
(rewrite to -1 -> computed on CPU). Mirrors the eligibility logic in offload_kernels.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

Q = 1 << 16
WINDOW = max(1, int(os.getenv("FREETOKEN_HYBRID_WINDOW", "64")))


def _balanced_fetch(num_missing, frac_q16):
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def main():
    import torch

    torch.manual_seed(0)
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415
    cache = OffloadMoeCache(
        num_layers=2, num_experts=num_experts, cache_size=cache_size,
        device=torch.device("cpu"), quant_format="bf16", decode_target="hybrid",
        hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
    )
    frac_q16 = round(frac * Q)
    seen = set()
    fails = []
    for step in range(1, 65):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        ids_list = ids.tolist()
        uniq = []
        for e in ids_list:
            if e not in uniq:
                uniq.append(e)
        # resident-before-this-step = experts with slot_for_id != -1 already.
        resident_before = {
            int(cache.slot_for_id[0, e].item()) for e in uniq if int(cache.slot_for_id[0, e].item()) != -1
        }
        cache.ensure_experts_hybrid(0, ids)
        missing = int(cache.num_missing_full.item())
        fetched = int(cache.num_indices.item())
        # Eligible misses = routed-this-step, missing-before-this-step, seen prior step
        # (matches kernel: rec >= cutoff; cutoff<=0 over these steps -> rec >= 0 -> seen before).
        elig_count = sum(1 for e in uniq if e in seen and e not in resident_before)
        expected = min(elig_count, _balanced_fetch(missing, frac_q16))
        if fetched != expected:
            fails.append(f"step {step}: fetched={fetched} expected={expected} missing={missing} elig={elig_count}")
        if fetched > elig_count:
            fails.append(f"step {step}: fetched={fetched} exceeds eligible={elig_count}")
        seen.update(uniq)
    if fails:
        print("RESULT: FAILURES")
        for f in fails:
            print(" ", f)
        return 1
    print("RESULT: ALL PASS (recurrence-window eligibility gate, CPU reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
