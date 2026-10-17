"""Qwen3.8-Flash-Next vision tower (Qwen2.5-VL-lineage ViT, opt-in FREETOKEN_LOAD_VISION=1).

Port of HF ``modeling_qwen4_exp.Qwen4ExpVisionModel``: Conv3D patchify of (2,16,16) pixel
blocks, a learned 48x48 ``pos_embed`` bilinearly resampled per image (align_corners=True,
patches enumerated in spatial-merge-block order so the merger's ``view`` is free), 2-D RoPE
over the (row, col) of each patch (rotating ``head_dim//2`` frequencies duplicated), 27
global bidirectional blocks (LayerNorm eps 1e-6, ``gelu_pytorch_tanh`` MLPs, attention
scale ``head_dim**-0.5``), and the merger (pre-shuffle LN -> 2x2 concat -> 4608 FC ->
EXACT GELU -> FC -> text hidden). The exact-vs-tanh GELU split between MLP and merger
follows HF and is load-bearing for bit-parity.

Per-grid contract: ``forward(pixel_values [seq, 3*2*16*16], grid_thw [1, 3]) ->
[seq // spatial_merge_size**2, out_hidden_size]``. Attention never crosses grids, so the
pipeline loops per grid (no varlen packing). A grid is one image (t == 1) or one video
(t > 1, frames packed t-major by the processor).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList

if TYPE_CHECKING:
    from freetoken.models.qwen4_exp.config import Qwen4ExpVisionConfig


# --------------------------------------------------------------------------------------
# grid_thw precomputes (port of the bilinear branch of transformers.vision_utils; the
# engine calls these eagerly, so no traceable-op gymnastics are needed)
# --------------------------------------------------------------------------------------

def _interp_indices_weights(
    grid_thw: torch.Tensor, side: int, merge: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-patch (indices, weights) resampling the side*side pos_embed table onto each
    (h, w) grid, in spatial-merge-block patch order. ``(total_thw, 4)`` each."""
    device = grid_thw.device
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(counts.sum(), device=device) - starts) % (heights * widths)
    # decode `within` under merge-block order: (block_row, block_col, in_row, in_col)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row, col = block_row * merge + in_row, block_col * merge + in_col

    def axis(index, size):
        # F.interpolate(mode="bilinear", align_corners=True): src in [0, side-1]
        src = index.to(torch.float32) * (side - 1) / torch.clamp(size - 1, min=1)
        floor = torch.floor(src)
        raw_taps = floor.long()[:, None] + torch.arange(0, 2, device=device)
        taps = raw_taps.clamp(0, side - 1)
        distance = (src[:, None] - floor[:, None] - torch.arange(0, 2, device=device)).abs()
        return taps, (1 - distance).clamp(min=0)

    h_taps, h_w = axis(row, heights)
    w_taps, w_w = axis(col, widths)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return indices, weights


def _vision_position_ids(
    t: int, h: int, w: int, merge: int, device: torch.device
) -> torch.Tensor:
    """(h, w) patch coordinates in merge-block order, repeated over the t frames."""
    hpos, wpos = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    block = (h // merge, merge, w // merge, merge)
    hpos = hpos.reshape(block).transpose(1, 2).flatten()
    wpos = wpos.reshape(block).transpose(1, 2).flatten()
    return torch.stack([hpos, wpos], dim=-1).repeat(t, 1)


class _VisionRotary:
    """Vision 2-D rope: 2 axes x (head_dim//4) frequencies, duplicated to head_dim.

    inv_freq is computed per call, not stored: the engine builds the model under
    ``torch.device("meta")`` and this is not a state-dict leaf, so an __init__-time
    tensor would stay a meta tensor forever ("Cannot copy out of meta tensor").
    """

    def __init__(self, dim: int, theta: float):  # dim == head_dim // 2
        self._dim = int(dim)
        self._theta = float(theta)

    def cos_sin(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (
            self._theta
            ** (
                torch.arange(
                    0, self._dim, 2, dtype=torch.float32, device=position_ids.device
                )
                / self._dim
            )
        )
        freqs = (position_ids.unsqueeze(-1) * inv).flatten(1)  # [N, dim]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, head_dim]
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# Fused kernels measure fine through n=65536 (the checkpoint's 16.7M-px cap);
# the ceiling below is a deterministic bound, not an observed wall — an
# earlier "every fused backend dies at 65536" reading was a misattributed
# async error from a [1,n,heads,dim] layout bug (heads=65536 hit the kernel's
# grid cap; the launch error surfaced later on an unrelated line).
_FUSED_N_MAX = 65535


def _attention_4d(q4: torch.Tensor, k4: torch.Tensor, v4: torch.Tensor,
                  scale: float, chunk: int = 256) -> torch.Tensor:
    """Full bidirectional softmax attention on [batch=1, heads, n, head_dim].

    n <= _FUSED_N_MAX calls F.scaled_dot_product_attention exactly as HF's
    vision attention does (same 4-D view layout), so the dispatcher selects the
    same fused mem-efficient kernel as the HF golden reference: O(n) memory
    (+43 MiB at n=19590, where the math backend's fp32 [heads,n,n] scores are
    22.9 GiB — the production OOM). Beyond the ceiling the q rows run in blocks
    (unreachable from the live path; media.py caps at 32768 patches): softmax
    is per-q-row so every query still attends every key, the fp32
    scores/softmax/probs dtype flow mirrors the math backend, and the transient
    is bounded regardless of which backends this torch build happens to have.
    """
    n = q4.shape[2]
    if n <= _FUSED_N_MAX:
        return F.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    q, k, v = (t.squeeze(0).contiguous() for t in (q4, k4, v4))
    kf, vf = k.float(), v.float()
    out = torch.empty(q.shape[0], n, v.shape[-1], device=q.device, dtype=torch.float32)
    for i in range(0, n, chunk):
        sc = q[:, i:i + chunk].float() @ kf.transpose(-1, -2)
        sc.mul_(scale)
        sc = F.softmax(sc, dim=-1)
        out[:, i:i + chunk] = sc @ vf
    return out.unsqueeze(0).to(v4.dtype)


# --------------------------------------------------------------------------------------
# parameter leaves (attribute paths mirror the checkpoint's post-rename keys exactly)
# --------------------------------------------------------------------------------------

class LayerNorm(BaseOP):
    """Plain nn.LayerNorm parity (weight+bias, fp32 internal math)."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.bias = torch.empty(dim)
        self._dim = dim
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self._dim,), self.weight, self.bias, self._eps)


class _Conv3DProj(BaseOP):
    """patch_embed.proj: kernel == stride, so conv == linear over the flattened block
    (the processor already packs each (c,t,h,w) block contiguously = the conv fan-in)."""

    def __init__(self, out: int, in_ch: int, kt: int, kp: int):
        self.weight = torch.empty(out, in_ch, kt, kp, kp)
        self.bias = torch.empty(out)
        self._flat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._flat is None:
            self._flat = self.weight.reshape(self.weight.shape[0], -1)
        return F.linear(x.reshape(-1, self._flat.shape[1]).to(self.weight.dtype),
                        self._flat, self.bias)


class _EmbeddingTable(BaseOP):
    def __init__(self, rows: int, cols: int):
        self.weight = torch.empty(rows, cols)


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------

class Qwen4ExpVisionMLP(BaseOP):
    def __init__(self, vc: "Qwen4ExpVisionConfig"):
        self.linear_fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x), approximate="tanh"))


class Qwen4ExpVisionAttention(BaseOP):
    """Bidirectional MHA, fused qkv with bias, 2-D rope in fp32, scale head_dim**-0.5.

    Single image per call, so every patch attends to every patch (no mask / cu_seqlens).
    """

    def __init__(self, vc: "Qwen4ExpVisionConfig"):
        self.num_heads = vc.num_heads
        self.head_dim = vc.head_dim
        self.qkv = LinearReplicated(vc.hidden_size, 3 * vc.hidden_size, has_bias=True)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)
        self._scale = vc.head_dim**-0.5

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = (
            self.qkv.forward(x).view(n, 3, self.num_heads, self.head_dim).unbind(1)
        )
        # HF applies rope fp32 then casts back; cos/sin are [N, head_dim] fp32.
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        qf, kf = q.float(), k.float()
        q = (qf * c + _rotate_half(qf) * s).to(q.dtype)
        k = (kf * c + _rotate_half(kf) * s).to(k.dtype)
        # HF layout: transpose(0,1).unsqueeze(0) -> [1, heads, n, head_dim], exactly
        # HF's sequence (both sides' q are contiguous [n, heads, head_dim] after the
        # fp32-rope cast, so the strides match too), and the dispatcher then selects
        # the SAME fused kernel as the golden reference.
        o = _attention_4d(
            *(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)), self._scale
        )
        # HF does transpose(1, 2).contiguous() + reshape(seq, -1); same values.
        return self.proj.forward(o.transpose(1, 2).reshape(n, -1))


class Qwen4ExpVisionBlock(BaseOP):
    def __init__(self, vc: "Qwen4ExpVisionConfig"):
        self.norm1 = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.norm2 = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.attn = Qwen4ExpVisionAttention(vc)
        self.mlp = Qwen4ExpVisionMLP(vc)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin)
        return x + self.mlp.forward(self.norm2.forward(x))


class Qwen4ExpVisionPatchEmbed(BaseOP):
    def __init__(self, vc: "Qwen4ExpVisionConfig"):
        self.proj = _Conv3DProj(
            vc.hidden_size, vc.in_channels, vc.temporal_patch_size, vc.patch_size
        )
        self._fan_in = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self._fan_in:
            raise ValueError(f"pixel rows {x.shape[-1]} != patch fan-in {self._fan_in}")
        return self.proj.forward(x)


class Qwen4ExpVisionPatchMerger(BaseOP):
    """Pre-shuffle LN, then 2x2 concat -> FC(hidden*4) -> EXACT gelu -> FC(text hidden)."""

    def __init__(self, vc: "Qwen4ExpVisionConfig"):
        self.norm = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        merged = vc.hidden_size * vc.spatial_merge_size**2
        self.linear_fc1 = LinearReplicated(merged, merged, has_bias=True)
        self.linear_fc2 = LinearReplicated(merged, vc.out_hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x).view(-1, self.linear_fc1.weight.shape[1])
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


# --------------------------------------------------------------------------------------
# tower
# --------------------------------------------------------------------------------------

class Qwen4ExpVisionModel(BaseOP):
    """Pixels -> text-space soft tokens for ONE image per call."""

    def __init__(self, vc: "Qwen4ExpVisionConfig"):
        self.patch_embed = Qwen4ExpVisionPatchEmbed(vc)
        self.pos_embed = _EmbeddingTable(vc.num_position_embeddings, vc.hidden_size)
        self.blocks = OPList([Qwen4ExpVisionBlock(vc) for _ in range(vc.depth)])
        self.merger = Qwen4ExpVisionPatchMerger(vc)
        self._vc = vc
        self._rotary = _VisionRotary(vc.head_dim // 2, vc.rope_theta)
        self._side = int(math.isqrt(vc.num_position_embeddings))
        if self._side * self._side != vc.num_position_embeddings:
            raise ValueError(
                f"pos_embed rows {vc.num_position_embeddings} must be a perfect square"
            )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """``pixel_values`` [t*h*w, in_ch*kt*p*p], ``grid_thw`` one row [t, h, w].

        t >= 1: frames are packed t-major (the processor pairs them into temporal
        units), pos-embed/rope repeat the spatial pattern per frame and the
        merger's 2x2 pooling never crosses frames (rows are t-major). Attention
        is global over the whole grid, matching HF's per-video cu_seqlens span.
        """
        if grid_thw.shape[0] != 1:
            raise NotImplementedError(
                "qwen4_exp vision encodes one grid per call; the pipeline loops"
            )
        vc = self._vc
        t, h, w = (int(v) for v in grid_thw[0])
        merge = vc.spatial_merge_size
        if h % merge or w % merge:
            raise ValueError(f"vision grid ({h}, {w}) must divide by merge {merge}")

        idx, wts = _interp_indices_weights(
            grid_thw.to(pixel_values.device), self._side, merge
        )
        pos_ids = _vision_position_ids(t, h, w, merge, pixel_values.device)
        x = self.patch_embed.forward(pixel_values)
        pos = (self.pos_embed.weight[idx] * wts.unsqueeze(-1)).sum(1)
        x = x + pos.to(x.dtype)
        cos, sin = self._rotary.cos_sin(pos_ids)
        for blk in self.blocks.op_list:
            x = blk.forward(x, cos, sin)
        return self.merger.forward(x)


__all__ = ["Qwen4ExpVisionModel"]
