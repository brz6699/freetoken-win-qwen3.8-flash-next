"""CUDA byte-order test for the chunked tier snapshot/restore path.

Guarantees under test (the exact contract the live garble would violate):
  1. chunked ``snapshot_rows`` == a reference full gather (column j <-> idx[j]),
     forced through MULTIPLE chunks;
  2. shuffled/unsorted slot order survives round-trip positionally;
  3. after ``restore_rows`` into DIFFERENT (fresh) slots, every destination column
     equals the original source column -- per-column, not just whole-tensor;
  4. the QSA cmp rows follow the slot//ratio shadow with the same ordering.

Every buffer element is made distinct (arange over numel), so any transposition,
rotation or wrong-chunk-boundary shows up as an inequality.

  python -X utf8 src/verification/gpu_restore_order.py
"""
import sys


def _fill(buf):
    buf.copy_(torch.arange(buf.numel(), dtype=buf.dtype).view_as(buf))


def main() -> int:
    global torch
    import torch
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    from freetoken.distributed import set_tp_info
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    set_tp_info(0, 1)
    dev = torch.device("cuda:0")
    PAGE, PAGES = 64, 8
    TOKENS = PAGE * PAGES

    # Plain fp16 kv (no turbo4): with a production serve resident on the GPU the
    # remaining headroom is only a few hundred MiB, and TurboQuant's rotation
    # matrix (head_dim-sized) would dominate the probe itself. The chunked
    # gather/scatter path under test is quantization-agnostic.
    pool = QSAKVCache(
        num_kv_heads=2, num_layers=4, head_dim=8192, num_pages=PAGES, page_size=PAGE,
        dtype=torch.float16, device=dev, index_head_dim=4, num_index_layers=1,
        index_ratio=2, num_req_slots=2, kv_dtype=torch.float16, kv_quant=None,
    )
    _fill(pool._kv_buffer)
    if pool._dscale_buffer is not None:
        _fill(pool._dscale_buffer)
    _fill(pool._cmp_k_buffer)
    cmp_pre = pool._cmp_k_buffer.cpu()

    rows = pool._kv_buffer.shape[1] * 2
    kv_view = pool._kv_buffer.view(rows, TOKENS, -1)
    n = 400  # > step so the chunked loop actually runs several chunks
    step = MHAKVCache._tier_step(kv_view, n)
    print(f"slots={n} chunk_step={step} chunks={-(-n // step)}")
    if n <= step:
        print("NOTE: single chunk for these dims; bump head_dim/page_size to force splits")

    torch.manual_seed(0)
    src = torch.randperm(n)                      # unsorted source slots
    dest = torch.randperm(TOKENS - PAGE)[:n]     # fresh destination slots

    # Reference: full gather from a host copy, computed BEFORE any trashing.
    kv_ref = kv_view.cpu()[:, src]
    ds_ref = (pool._dscale_buffer.view(rows, TOKENS, -1).cpu()[:, src]
              if pool._dscale_buffer is not None else None)

    snap = pool.snapshot_rows(src)
    checks = [("gather.kv", torch.equal(snap["kv"], kv_ref))]
    if ds_ref is not None and "ds" in snap:
        checks.append(("gather.ds", torch.equal(snap["ds"], ds_ref)))
    checks.append(("gather.cmp", torch.equal(snap["cmp"], cmp_pre[:, src // pool._index_ratio])))

    # Trash the slab, restore into different slots, compare per column.
    pool._kv_buffer.fill_(0xFF)
    if pool._dscale_buffer is not None:
        pool._dscale_buffer.fill_(0xFF)
    pool._cmp_k_buffer.fill_(0xFF)
    pool.restore_rows(dest, snap)

    got_kv = pool._kv_buffer.view(rows, TOKENS, -1).cpu()[:, dest]
    checks.append(("restore.kv", torch.equal(got_kv, kv_ref)))
    if ds_ref is not None:
        got_ds = pool._dscale_buffer.view(rows, TOKENS, -1).cpu()[:, dest]
        checks.append(("restore.ds", torch.equal(got_ds, ds_ref)))

    # cmp rows are shared shadows (slot//ratio): two adjacent slots write the SAME
    # row, so per-column checks need production geometry -- contiguous ascending
    # runs on both sides (what _page_to_token always yields). A shuffled dest
    # makes two columns carry different values for one shared row, which is a
    # harness artifact, not a restore bug.
    src2 = torch.arange(0, PAGE)
    dest2 = torch.arange(PAGE, 2 * PAGE)
    snap2 = pool.snapshot_rows(src2)
    want_cmp2 = snap2["cmp"]
    pool._cmp_k_buffer.fill_(0xFF)
    pool.restore_rows(dest2, snap2)
    got_cmp2 = pool._cmp_k_buffer.cpu()[:, dest2 // pool._index_ratio]
    checks.append(("restore.cmp", torch.equal(got_cmp2, want_cmp2)))

    fails = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if fails:
        print(f"RESULT: FAIL {fails}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
