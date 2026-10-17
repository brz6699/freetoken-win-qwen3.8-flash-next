"""GPU-only probe for the tier restore_rows device promotion (host-RAM -> cuda).

Exercises the exact path that crashed live: snapshot_rows returns CPU tensors,
restore_rows must promote them to the pool device before the in-place index_copy_.
Kept tiny so it fits in the ~2 GiB left on a GPU that another process occupies.

  python -X utf8 verification/gpu_restore_probe.py
"""
import sys


def main() -> int:
    import torch
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    from freetoken.distributed import set_tp_info
    from freetoken.kvcache.qsa_pool import QSAKVCache

    set_tp_info(0, 1)
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    PAGE = 4

    pool = QSAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=32, num_pages=8, page_size=PAGE,
        dtype=torch.float16, device=dev, index_head_dim=4, num_index_layers=1,
        index_ratio=2, num_req_slots=2, kv_dtype=torch.uint8, kv_quant="turbo4",
    )
    # Distinct content per slot so a failed copy is detectable.
    pool._kv_buffer.copy_(torch.arange(pool._kv_buffer.numel(), dtype=pool._kv_buffer.dtype).view_as(pool._kv_buffer))
    if pool._dscale_buffer is not None:
        pool._dscale_buffer.copy_(torch.arange(pool._dscale_buffer.numel(), dtype=pool._dscale_buffer.dtype).view_as(pool._dscale_buffer))
    pool._cmp_k_buffer.copy_(torch.arange(pool._cmp_k_buffer.numel(), dtype=pool._cmp_k_buffer.dtype).view_as(pool._cmp_k_buffer))

    A = torch.arange(0, 8, dtype=torch.int32)      # original slots
    B = torch.arange(8, 16, dtype=torch.int32)     # fresh slots
    orig = pool.snapshot_rows(A)
    print(f"snapshot device: kv={orig['kv'].device} cmp={orig.get('cmp', torch.empty(0)).device}")
    assert orig["kv"].device.type == "cpu", "tier rows must be host-resident"

    # Simulate the pages being recycled: trash the whole slab.
    pool._kv_buffer.fill_(0xFF)
    if pool._dscale_buffer is not None:
        pool._dscale_buffer.fill_(0xFF)
    pool._cmp_k_buffer.fill_(0xFF)

    pool.restore_rows(B, orig)          # <-- the live crash was here (cpu->cuda index_copy_)
    after = pool.snapshot_rows(B)
    for k in ("kv", "ds", "cmp"):
        if k in orig:
            ok = torch.equal(after[k], orig[k])
            print(f"[{'PASS' if ok else 'FAIL'}] restore.{k}")
            if not ok:
                print(f"RESULT: FAIL {k}"); return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
