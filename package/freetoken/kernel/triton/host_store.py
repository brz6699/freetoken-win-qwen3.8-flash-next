"""Write-through row scatter into a pinned-host KV slab (the KV host offload's store path).

With ``--kv-offload`` the bulk K/V rows live in ``cudaHostAlloc``'d (portable + mapped)
memory, so the normal store path -- ``kernel/store.py``'s ``StoreKernel``, which matches a
CUDA destination -- cannot be used: it would need a device-side destination for the bytes.
This is the same shape as ``ple.py``'s gather in the other direction: one program per row,
read the row from the device-side tensor, write the bytes into the host slab through the
address the GPU must dereference (``kernel/pinned.device_ptr``, which resolves the
WDDM-mapped alias rather than the host VA).

Byte-level on purpose: rows reach the slab exactly as the caller prepared them, so the
quantized-on-store representations (fp8 saturating clamp, turbo4 nibble packing) need no
re-quantize on read and the kernel never branches on the storage dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One row is a few KB; many small programs keep the PCIe write path busy without a wide block.
_NUM_WARPS = 1


@triton.jit
def _host_store_kernel(
    k_src_ptr,
    v_src_ptr,
    k_dst_ptr,
    v_dst_ptr,
    idx_ptr,
    row_bytes_k,
    row_bytes_v,
    BLOCK: tl.constexpr,
) -> None:
    row_out = tl.program_id(0)
    dst_row = tl.load(idx_ptr + row_out).to(tl.int64)
    # the slab is a host allocation: rebuild the typed pointer from the mapped address
    k_base = k_dst_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    v_base = v_dst_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))

    offs = tl.arange(0, BLOCK)
    mask_k = offs < row_bytes_k
    vals_k = tl.load(
        k_src_ptr + row_out.to(tl.int64) * row_bytes_k + offs, mask=mask_k
    )
    tl.store(k_base + dst_row * row_bytes_k + offs, vals_k, mask=mask_k)

    mask_v = offs < row_bytes_v
    vals_v = tl.load(
        v_src_ptr + row_out.to(tl.int64) * row_bytes_v + offs, mask=mask_v
    )
    tl.store(v_base + dst_row * row_bytes_v + offs, vals_v, mask=mask_v)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def store_rows_to_host(
    dst: torch.Tensor, src: torch.Tensor, rows: torch.Tensor
) -> None:
    """Scatter ``src[i]`` into row ``rows[i]`` of ``dst``, which may live in pinned host RAM.

    ``dst`` is a row-flat view ``[num_tokens, width]`` (any storage dtype), ``src`` is
    ``[T, width]`` on the compute device with the same element count per row, ``rows`` is a
    device int tensor of destination token slots. Byte-for-byte: the caller has already
    clamped/cast/packed the values for the destination dtype.
    """
    d = _as_bytes(dst)
    s = _as_bytes(src)
    row_bytes = int(s.shape[-1])
    if int(d.shape[-1]) != row_bytes:
        raise ValueError(
            f"host store row width mismatch: dst {int(d.shape[-1])} B, src {row_bytes} B"
        )
    n = int(rows.numel())
    if not n:
        return
    idx = rows if rows.dtype == torch.int32 else rows.to(torch.int32)
    _host_store_kernel[(n,)](
        s,
        s,
        int(_mapped(d)),
        int(_mapped(d)),
        idx,
        row_bytes,
        row_bytes,
        BLOCK=triton.next_power_of_2(row_bytes),
        num_warps=_NUM_WARPS,
    )


def store_kv_rows_to_host(
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """K/V twin of :func:`store_rows_to_host` -- one launch for both halves."""
    dk, dv = _as_bytes(k_dst), _as_bytes(v_dst)
    sk, sv = _as_bytes(k), _as_bytes(v)
    n = int(rows.numel())
    if not n:
        return
    idx = rows if rows.dtype == torch.int32 else rows.to(torch.int32)
    _host_store_kernel[(n,)](
        sk,
        sv,
        int(_mapped(dk)),
        int(_mapped(dv)),
        idx,
        int(sk.shape[-1]),
        int(sv.shape[-1]),
        BLOCK=triton.next_power_of_2(
            max(int(sk.shape[-1]), int(sv.shape[-1]))
        ),
        num_warps=_NUM_WARPS,
    )


def _mapped(t: torch.Tensor) -> int:
    from freetoken.kernel.pinned import device_ptr

    return int(device_ptr(t))


def host_element_type(dtype: torch.dtype):
    """Triton pointer element type for a host-resident KV slab, matching the storage dtypes
    the attend kernels assume (uint8 turbo4 carrier, fp8 clamped storage, plain bf16). A
    kernel receiving a mapped address rather than a tensor needs it to rebuild the pointer."""
    if dtype == torch.uint8:
        return tl.uint8
    if dtype == torch.float8_e4m3fn:
        return tl.float8e4nv
    return tl.bfloat16


__all__ = [
    "store_rows_to_host",
    "store_kv_rows_to_host",
    "host_element_type",
]
