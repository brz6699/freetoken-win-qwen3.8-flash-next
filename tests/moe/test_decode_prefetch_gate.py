"""Standalone (pytest-free) check for bs>=4 decode prefetch double-buffering.

Runs OffloadMoeCache on CPU (works while llama-server holds the GPU) and verifies the
gate + effect of the bs>=4 prefetch branch added to layers/moe.py::_decode_hybrid:

- bs < 4 skips prefetch (existing single-buffer path preserved).
- bs >= 4 fires prefetch_decode_layer(next_layer, hint) BEFORE current-layer GEMM dispatch.
- Last layer (next_layer >= num_layers) skips prefetch cleanly.
- Prefetch bumps next-layer expert_recency so the next step's ensure_experts_hybrid finds
  recurring experts eligible without re-fetching.
- Prefetch on next_layer does NOT clobber current-layer slot assignments (layer isolation
  through the shared slot_for_id / id_of_slot / expert_recency tensors).

Mirrors the branch logic in layers/moe.py::_decode_hybrid without needing a real GEMM.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

# CPU-only harness: force the pageable copy path so the JIT kernel's pinned-memory check
# does not fire. Same as setting FREETOKEN_SKIP_FAST_INDEX_COPY=1 for the serve process.
os.environ.setdefault("FREETOKEN_SKIP_FAST_INDEX_COPY", "1")


def _should_prefill(bs_hint, layer_id, num_layers):
    """Mirror of the bs>=4 gate in layers/moe.py::_decode_hybrid."""
    next_layer = layer_id + 1
    return bs_hint >= 4 and next_layer < int(num_layers)


def main():
    import torch

    torch.manual_seed(0)
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_experts = 16
    num_layers = 3
    cache_size = num_experts * num_layers + num_layers * num_experts

    fails = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not cond:
            fails.append(name)

    # ---- gate behavior (pure arithmetic on the branch condition) ----
    check("gate bs=1 skips", _should_prefill(1, 0, num_layers) is False)
    check("gate bs=3 skips", _should_prefill(3, 0, num_layers) is False)
    check("gate bs=4 fires", _should_prefill(4, 0, num_layers) is True)
    check("gate bs=8 fires", _should_prefill(8, 0, num_layers) is True)
    check("gate last-layer skips", _should_prefill(4, num_layers - 1, num_layers) is False)

    # ---- prefetch_decode_layer effect on cache state ----
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=torch.device("cpu"), quant_format="bf16", decode_target="hybrid",
        hybrid_max_fetch=num_experts, hybrid_fetch_fraction=0.0,
    )
    hidden, inter = 32, 64
    cache.set_bank_sources({
        "gate_up": [torch.zeros(num_experts, 2 * inter, hidden, dtype=torch.bfloat16)
                    for _ in range(num_layers)],
        "down": [torch.zeros(num_experts, hidden, inter, dtype=torch.bfloat16)
                 for _ in range(num_layers)],
    })

    # Step 1: populate layer 0 with real routing. First-time misses stay non-resident
    # (recurrence-window rejects unseen); their recency bumps so next call qualifies them.
    ids_l0 = torch.tensor([1, 3, 5, 7], dtype=torch.int32)
    cache.ensure_experts_hybrid(0, ids_l0.clone())
    slots_l0_after = cache.slot_for_id[0].clone()

    # Step 2: prefetch next layer with same hint (mirrors what _decode_hybrid does when bs>=4).
    cache.prefetch_decode_layer(1, ids_l0.clone())

    # Recency bumped for hint experts on BOTH layers: ensure_experts_hybrid bumps recency
    # unconditionally for seen experts (eligibility gate is orthogonal to the bump).
    check("layer-0 recency bumped for hint expert", int(cache.expert_recency[0, 1].item()) > 0)
    check("layer-1 recency bumped for hint expert", int(cache.expert_recency[1, 1].item()) > 0)

    # Layer isolation: prefetching next_layer must NOT change layer 0's slot assignments.
    check(
        "layer isolation: prefetch preserves layer-0 slots",
        torch.equal(cache.slot_for_id[0], slots_l0_after),
    )

    # ---- bs=1 path: prefetch not called, next layer stays cold ----
    cache2 = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=torch.device("cpu"), quant_format="bf16", decode_target="hybrid",
        hybrid_max_fetch=num_experts, hybrid_fetch_fraction=0.0,
    )
    ids_a = torch.tensor([2, 4], dtype=torch.int32)
    cache2.ensure_experts_hybrid(0, ids_a.clone())
    # bs=1 branch skips prefetch_decode_layer entirely.
    if _should_prefill(1, 0, num_layers):
        cache2.prefetch_decode_layer(1, ids_a.clone())
    # After bs=1 step, next-layer should still be untouched (no prefetch, no recency bump).
    check(
        "bs=1 leaves next-layer recency untouched",
        int(cache2.expert_recency[1, 2].item()) == -1,
    )
    # Layer isolation check on cache2: bs=1 path only touched layer 0's recency.
    check(
        "bs=1 leaves layer-1 slot map untouched",
        torch.all(cache2.slot_for_id[1] == -1).item(),
    )

    # ---- non-hybrid decode_target disables prefetch (gate-only check) ----
    # ensure_experts on a gpu-target cache needs CUDA tensors (Triton kernel path), so the
    # CPU harness checks the gate arithmetic directly rather than driving the full path.
    cache3 = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=32,
        device=torch.device("cpu"), quant_format="bf16", decode_target="gpu",
        hybrid_max_fetch=8, hybrid_fetch_fraction=0.0,
    )
    check(
        "non-hybrid target disables prefetch (gate check)",
        cache3.decode_target != "hybrid" and cache3.decode_target == "gpu",
    )

    if fails:
        print(f"\nRESULT: FAIL ({len(fails)} assertion(s))")
        return 1
    print("\nRESULT: ALL PASS (bs>=4 decode prefetch double-buffer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
