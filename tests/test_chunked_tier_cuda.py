"""Minimal CUDA test: turbo4 pool + chunked mm-tier snapshot/restore on cuda:0.

Uses only a tiny pool (a few MiB) so it runs alongside the resident llama-server
without OOM'ing it. Exercises the REAL on-device D2H/H2D chunked copy (the path
CPU cannot reproduce) and checks, byte-for-byte, that content survives a multi-chunk
gather and a scatter into fresh slots -- for both the packed KV and the turbo4 dequant
scales.

  E:\\FreeToken-Ours\\ft-venv\\Scripts\\python.exe -X utf8 src/tests/test_chunked_tier_cuda.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    import torch
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    free, _total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    if free < 400 * (1 << 20):
        print(f"SKIP: only {free // (1 << 20)} MiB free; need ~400 MiB headroom")
        return 0
    print(f"cuda:0 free {free // (1 << 20)} MiB before test")
    from freetoken.distributed import set_tp_info
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    set_tp_info(0, 1)
    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    PAGE, PAGES, L, H, HD = 64, 24, 8, 2, 256
    pool = QSAKVCache(
        num_kv_heads=H, num_layers=L, head_dim=HD, num_pages=PAGES, page_size=PAGE,
        dtype=torch.bfloat16, device=dev, index_head_dim=4,
        num_index_layers=1, index_ratio=2, num_req_slots=4,
        kv_dtype=torch.uint8, kv_quant="turbo4",
    )

    # distinct packed kv bytes + distinct scales
    pool._kv_buffer.copy_(torch.randint(0, 256, pool._kv_buffer.shape,
                                        dtype=torch.uint8, device=dev))
    pool._dscale_buffer.copy_(
        torch.arange(1, pool._dscale_buffer.numel() + 1, dtype=torch.bfloat16, device=dev)
        .view_as(pool._dscale_buffer) / 11
    )

    kv_rows = pool._kv_buffer.shape[1] * 2
    kv_tokens = pool._storage_shape[0]
    ds_rows = pool._dscale_buffer.shape[1] * 2
    ds_tokens = pool._storage_shape[0]
    # reference: capture on-device then move once (stable host copy)
    kv_ref = pool._kv_buffer.view(kv_rows, kv_tokens, -1)
    ds_ref = pool._dscale_buffer.view(ds_rows, ds_tokens, -1)

    # b must exceed _tier_step's floor (max(4MiB,...) -> ~96 slots here) so the chunked
    # loop actually runs MULTIPLE chunks; that boundary math is the regression surface.
    b = 1500
    src = torch.arange(0, b, dtype=torch.int64, device=dev)
    kv_ref_host = kv_ref[:, src].cpu()
    ds_ref_host = ds_ref[:, src].cpu()

    # force MULTIPLE chunks even on the small pool
    MHAKVCache._TIER_CHUNK_BYTES = 1024
    kstep = MHAKVCache._tier_step(kv_ref, b)
    print(f"kv_rows={kv_rows} tokens={kv_tokens} kv_step={kstep} chunks={-(-b // kstep)}")

    snap = pool.snapshot_rows(src)
    check("cuda.kv_gather", torch.equal(snap["kv"], kv_ref_host), f"n={b}")
    check("cuda.ds_gather", torch.equal(snap["ds"], ds_ref_host), f"n={b}")

    # scatter into fresh, scattered slots and read back
    torch.manual_seed(1)
    dest = torch.randperm(kv_tokens)[:b].to(torch.int64).to(dev)
    pool._kv_buffer.zero_()
    pool._dscale_buffer.zero_()
    pool.restore_rows(dest, snap)
    check("cuda.kv_restore", torch.equal(
        pool._kv_buffer.view(kv_rows, kv_tokens, -1)[:, dest].cpu(), kv_ref_host), f"n={b}")
    check("cuda.ds_restore", torch.equal(
        pool._dscale_buffer.view(ds_rows, ds_tokens, -1)[:, dest].cpu(), ds_ref_host), f"n={b}")

    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
