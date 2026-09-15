# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""Sparse paged GQA over the QSA selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_dscale_ptr,
    v_dscale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_dscale_row,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    codebook_c0,
    codebook_step,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED4: tl.constexpr,
) -> None:
    # row * stride can overflow int32 for large row counts.
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    if PACKED4:
        # turbo4 stores nibble pairs (even in the low nibble); split the query the same
        # way and run the two-way split dot, SGLang tq-decode precedent. Q arrives
        # already rotated (rotspace: dot(q,k) == dot(Rq,Rk)).
        half_offsets = tl.arange(0, HEAD_DIM // 2)
        query_even = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + (half_offsets * 2)[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
        query_odd = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + (half_offsets * 2 + 1)[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )
    else:
        query = tl.load(
            q_ptr
            + row * stride_q_row
            + (first_head + head_offsets[:, None]) * stride_q_head
            + dim_offsets[None, :],
            mask=head_offsets[:, None] < GROUP_SIZE,
            other=0.0,
        )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request.to(tl.int64) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if PACKED4:
            # The scale buffers are token-linear (slot = page * PAGE_SIZE + offset); a
            # never-written slot dequant-scales to 0, keeping masked reads finite.
            slot = safe_page * PAGE_SIZE + page_offset
            packed_k = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + half_offsets[:, None],
                mask=valid[None, :],
                other=0,
            )
            k_lo = (
                (packed_k & 0x0F).to(tl.float32) * codebook_step + codebook_c0
            ).to(tl.bfloat16)
            k_hi = (
                ((packed_k >> 4) & 0x0F).to(tl.float32) * codebook_step + codebook_c0
            ).to(tl.bfloat16)
            k_scale = tl.load(
                k_dscale_ptr + slot * stride_dscale_row + kv_head,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            scores = tl.dot(query_even, k_lo) + tl.dot(query_odd, k_hi)
            scores *= k_scale[None, :]
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            ).to(tl.bfloat16)  # fp8 KV storage dequant; no-op for bf16 caches
            scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if PACKED4:
            packed_v = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + half_offsets[None, :],
                mask=valid[:, None],
                other=0,
            )
            v_lo = (
                (packed_v & 0x0F).to(tl.float32) * codebook_step + codebook_c0
            ).to(tl.bfloat16)
            v_hi = (
                ((packed_v >> 4) & 0x0F).to(tl.float32) * codebook_step + codebook_c0
            ).to(tl.bfloat16)
            v_scale = tl.load(
                v_dscale_ptr + slot * stride_dscale_row + kv_head,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            # The per-token scale folds into the probabilities once (linearity of the
            # dot); the normalizer must stay UNSCALED (it is the softmax denominator).
            scaled_probabilities = (probabilities * v_scale[None, :]).to(tl.bfloat16)
            accumulator_even = tl.dot(
                scaled_probabilities, v_lo, acc=accumulator_even * alpha[:, None]
            )
            accumulator_odd = tl.dot(
                scaled_probabilities, v_hi, acc=accumulator_odd * alpha[:, None]
            )
        else:
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)  # fp8 KV storage dequant; no-op for bf16 caches
            accumulator = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator * alpha[:, None],
            )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    safe_normalizer = tl.maximum(normalizer, 1.0e-20)[:, None]
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if PACKED4:
        # The wrapper inverts the rotation on the FULL attention output, so partials and
        # direct output both stay in the rotated domain, interleaved back into HEAD_DIM.
        normalized_even = tl.where(
            has_values[:, None], accumulator_even / safe_normalizer, 0.0
        )
        normalized_odd = tl.where(
            has_values[:, None], accumulator_odd / safe_normalizer, 0.0
        )
        if NUM_SPLITS == 1:
            output_base = (
                output_ptr
                + row * stride_output_row
                + (first_head + head_offsets[:, None]) * stride_output_head
            )
            tl.store(
                output_base + (half_offsets * 2)[None, :],
                normalized_even.to(output_ptr.dtype.element_ty),
                mask=output_mask,
            )
            tl.store(
                output_base + (half_offsets * 2 + 1)[None, :],
                normalized_odd.to(output_ptr.dtype.element_ty),
                mask=output_mask,
            )
        else:
            partial_lse = tl.where(
                has_values,
                max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
                -float("inf"),
            )
            partial_base = (
                partial_output_ptr
                + (
                    (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                    + first_head
                    + head_offsets[:, None]
                )
                * HEAD_DIM
            )
            tl.store(
                partial_base + (half_offsets * 2)[None, :],
                normalized_even,
                mask=output_mask,
            )
            tl.store(
                partial_base + (half_offsets * 2 + 1)[None, :],
                normalized_odd,
                mask=output_mask,
            )
            tl.store(
                partial_lse_ptr
                + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets,
                partial_lse,
                mask=head_offsets < GROUP_SIZE,
            )
    else:
        normalized_output = tl.where(
            has_values[:, None],
            accumulator / safe_normalizer,
            0.0,
        )
        if NUM_SPLITS == 1:
            tl.store(
                output_ptr
                + row * stride_output_row
                + (first_head + head_offsets[:, None]) * stride_output_head
                + dim_offsets[None, :],
                normalized_output,
                mask=output_mask,
            )
        else:
            partial_lse = tl.where(
                has_values,
                max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
                -float("inf"),
            )
            tl.store(
                partial_output_ptr
                + (
                    (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                    + first_head
                    + head_offsets[:, None]
                )
                * HEAD_DIM
                + dim_offsets[None, :],
                normalized_output,
                mask=output_mask,
            )
            tl.store(
                partial_lse_ptr
                + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets,
                partial_lse,
                mask=head_offsets < GROUP_SIZE,
            )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None].to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    k_dscale: torch.Tensor | None = None,
    v_dscale: torch.Tensor | None = None,
    tq=None,  # kvcache.turboquant.TurboQuantConstants, required for packed uint8 caches
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16/FP8/TurboQuant-4bit K/V caches."""

    packed4 = k_cache.dtype == torch.uint8
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    kv_dim = k_cache.shape[3] * 2 if packed4 else k_cache.shape[3]
    if q.shape[2] != kv_dim or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    # Compute stays bf16; the paged storage may be fp8 (clamped float8_e4m3fn) or turbo4
    # (nibble-packed uint8 + per-token dequant scales) and the kernel dequants on read.
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    assert k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
    if packed4:
        assert tq is not None and k_dscale is not None and v_dscale is not None, (
            "turbo4 (uint8) KV caches need tq constants and dequant scale buffers"
        )
        assert k_dscale.dtype == v_dscale.dtype == torch.bfloat16
        assert k_dscale.stride(1) == v_dscale.stride(1) == 1
        assert k_dscale.shape == v_dscale.shape
    else:
        assert k_dscale is None and v_dscale is None
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    if not q.shape[0]:
        return out

    if packed4:
        # Rotspace equivalence: the pool stored K/V rotated by R, so rotate Q by R and
        # invert the FULL attention output by R^T. The kernel itself never unrotates.
        q = tq.rotate(q)
        out = torch.empty_like(q)

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        k_dscale if packed4 else k_cache,
        v_dscale if packed4 else v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        k_dscale.stride(0) if packed4 else 0,
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        tq.c0 if packed4 else 0.0,
        tq.step if packed4 else 1.0,
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PACKED4=packed4,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits > 1:
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output,
            partial_lse,
            out,
            out.stride(0),
            out.stride(1),
            q.shape[0],
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=2,
            num_stages=1,
        )
    if packed4:
        out = tq.unrotate(out)
    return out


__all__ = ["qsa_sparse_paged_attention"]
