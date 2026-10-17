"""CPU test: does the chunked mm-tier snapshot/restore preserve logical KV content?

Reproduces the user's hunch: the old path copied the whole slab at once (no per-chunk
offset math); the new path copies fixed-size consecutive segments, so an indexer computes
each segment's slot displacement. This checks, byte-for-byte, that:
  1. a MULTI-CHUNK gather returns exactly the logical content of the requested slots
     (forcing many chunks via a tiny _TIER_CHUNK_BYTES);
  2. that content survives a SHINK rebuild (arena re-slice) and a scatter into fresh
     (possibly non-contiguous) slots, positionally.

Every buffer element is made distinct (arange) so any transposition, off-by-one, or wrong
segment boundary shows up as an inequality. Quantization-agnostic: the tier copies stored
bytes, so plain fp16 isolates the indexer from turbo4.

  E:\\FreeToken-Ours\\ft-venv\\Scripts\\python.exe -X utf8 src/tests/test_chunked_tier_offset.py
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
    from freetoken.distributed import set_tp_info
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    set_tp_info(0, 1)
    torch.manual_seed(0)

    PAGE, PAGES, L, H, HD = 64, 20, 3, 2, 8
    pool = QSAKVCache(
        num_kv_heads=H, num_layers=L, head_dim=HD, num_pages=PAGES, page_size=PAGE,
        dtype=torch.float16, device=torch.device("cpu"), index_head_dim=4,
        num_index_layers=1, index_ratio=2, num_req_slots=4,
        kv_dtype=torch.float16, kv_quant=None,
    )
    # distinct values: token t, kv l, head h, dim d -> arange
    pool._kv_buffer.copy_(
        torch.arange(pool._kv_buffer.numel(), dtype=torch.float16).view_as(pool._kv_buffer)
    )

    rows = pool._kv_buffer.shape[1] * 2
    tokens = pool._storage_shape[0]
    ref_view = pool._kv_buffer.view(rows, tokens, -1)

    # force MANY chunks
    MHAKVCache._TIER_CHUNK_BYTES = 512  # tiny -> step is only a handful of slots
    step = MHAKVCache._tier_step(ref_view, 100)
    print(f"rows={rows} tokens={tokens} tiny_step={step}")

    # --- 1. multi-chunk gather of a contiguous prefix [0..b) ---
    b = 100
    src = torch.arange(0, b, dtype=torch.int64)
    snap = pool.snapshot_rows(src)
    ref = ref_view[:, src].cpu()
    check("gather.multi_chunk_matches_ref", torch.equal(snap["kv"], ref),
          f"n={b} chunks={-(-b // step)}")

    # --- 1b. multi-chunk gather of a NON-contiguous, unsorted slot set ---
    torch.manual_seed(1)
    src2 = torch.randperm(tokens)[:60]
    snap2 = pool.snapshot_rows(src2)
    ref2 = ref_view[:, src2].cpu()
    check("gather.noncontig_unsorted", torch.equal(snap2["kv"], ref2), f"n=60")

    # --- 2. shrink rebuild (arena re-slice) then scatter into fresh slots ---
    pool.rebuild(PAGES // 2)  # 20 -> 10 pages
    new_view = pool._kv_buffer.view(rows, pool._storage_shape[0], -1)
    # fresh, scattered destination slots within the new pool
    torch.manual_seed(2)
    new_tokens = pool._storage_shape[0]
    dest = torch.randperm(new_tokens)[:b]
    pool._kv_buffer.zero_()
    pool.restore_rows(dest, snap)  # snap was taken at the ORIGINAL geometry
    got = new_view[:, dest].cpu()
    check("restore.after_shrink_positional", torch.equal(got, ref),
          f"dest scattered, n={b}")

    # --- 3. turbo4 dequant-scale (ds) round-trip through the same chunked copy ---
    pool2 = QSAKVCache(
        num_kv_heads=H, num_layers=L, head_dim=32, num_pages=PAGES, page_size=PAGE,
        dtype=torch.bfloat16, device=torch.device("cpu"), index_head_dim=4,
        num_index_layers=1, index_ratio=2, num_req_slots=4,
        kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    if pool2._dscale_buffer is None:
        check("turbo4.dscale_present", False, "no dscale buffer built for turbo4")
    else:
        pool2._kv_buffer.copy_(
            torch.randint(0, 256, pool2._kv_buffer.shape, dtype=torch.uint8)
        )
        pool2._dscale_buffer.copy_(
            torch.arange(1, pool2._dscale_buffer.numel() + 1, dtype=torch.bfloat16)
            .view_as(pool2._dscale_buffer) / 7
        )
        ds_rows = pool2._dscale_buffer.shape[1] * 2
        ds_tokens = pool2._storage_shape[0]
        b2 = 96
        src3 = torch.arange(0, b2, dtype=torch.int64)
        # capture the reference as a HOST COPY before any mutation below (a view would go stale)
        ds_ref = pool2._dscale_buffer.view(ds_rows, ds_tokens, -1)[:, src3].cpu()
        snap3 = pool2.snapshot_rows(src3)
        check("turbo4.dscale_gather", "ds" in snap3
              and torch.equal(snap3["ds"], ds_ref), f"n={b2}")
        # scatter ds into fresh slots and read back
        torch.manual_seed(3)
        dest3 = torch.randperm(ds_tokens)[:b2]
        pool2._dscale_buffer.zero_()
        # scatter ONLY the ds view (the kv scatter is already covered above)
        pool2._scatter_host(pool2._dscale_buffer.view(ds_rows, ds_tokens, -1),
                            dest3, snap3["ds"])
        ds_got = pool2._dscale_buffer.view(ds_rows, ds_tokens, -1)[:, dest3].cpu()
        check("turbo4.dscale_restore", torch.equal(ds_got, ds_ref),
              f"dest scattered, n={b2}")

    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
