"""TurboQuant 4-bit KV storage (PolarQuant, arXiv:2504.19874; SGLang #23135 precedent).

Storage per (token, kv_head): head_dim 4-bit codebook indices packed TWO PER BYTE
(uint8 head_dim//2) plus ONE bf16 dequant scale. Per-vector recipe:
  L2 norm -> unit vector -> rotate by R = D2 H D1 (sign-flipped normalized Sylvester
  Hadamard, fixed seed) -> nearest index on a uniform grid -> dscale = norm / ||c[idx]||.
The rotation is orthogonal and SHARED by K and V, so no buffer is ever stored
unrotated and the attention kernel never inverts it: the caller rotates Q by R
(``rotate``) and inverts the attention output by R^T (``unrotate``) -- the rotspace
"Query Rotation" equivalence dot(q, k) == dot(Rq, Rk).

Uniform codebook (centroids = linspace(-r, r, 16), r = 2.5/sqrt(head_dim), the
sigma=1/sqrt(d) Gaussian's ~99.4% range) instead of Lloyd-Max: nearest centroid
becomes clamp(round((y - c0) / step), 0, 15) and dequant is one FMA. The ~0.8 dB
SNR loss versus Lloyd-Max is irrelevant under QSA (the indexer routes on bf16 keys;
the quantized main KV only perturbs content reads).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl

# Fixed rotation seed: store-side and read-side constants must agree across every
# process that rebuilds them independently (pool, backend, smoke tests).
TQ_SEED = 42
BITS = 4
N_CENTROIDS = 1 << BITS
CLIP_SIGMAS = 2.5


def _orthonormal_hadamard(dim: int) -> torch.Tensor:
    """Sylvester construction, scaled so H @ H.T == I."""
    if dim < 1 or dim & (dim - 1):
        raise ValueError(f"TurboQuant needs a power-of-two head_dim, got {dim}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(dim)


@triton.jit
def _tq_pack_kernel(
    Y,          # (rows, DIM) float32/bf16 — rotated unit vectors
    Norms,      # (rows,) float32 — original L2 norms
    Packed,     # (rows, DIM // 2) uint8 out — (odd << 4) | even, SGLang order
    DScale,     # (rows,) bf16 out — norm / max(||c[idx]||, eps)
    C0,         # first centroid
    STEP,       # centroid spacing
    stride_y,
    stride_p,
    DIM: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, DIM // 2)
    y_base = Y + row.to(tl.int64) * stride_y
    y_even = tl.load(y_base + offs * 2).to(tl.float32)
    y_odd = tl.load(y_base + offs * 2 + 1).to(tl.float32)

    # Nearest uniform-grid index (boundaries are the grid midpoints); 15.0 == the 4-bit
    # centroid count minus one (triton jit cannot read module globals).
    idx_even = tl.clamp(tl.floor((y_even - C0) / STEP + 0.5), 0.0, 15.0)
    idx_odd = tl.clamp(tl.floor((y_odd - C0) / STEP + 0.5), 0.0, 15.0)

    # dequant scale = original norm / centroid-vector length (PolarQuant norm correction).
    c_even = idx_even * STEP + C0
    c_odd = idx_odd * STEP + C0
    qnorm = tl.sqrt(tl.sum(c_even * c_even) + tl.sum(c_odd * c_odd))
    norm = tl.load(Norms + row)
    safe_qnorm = tl.where(qnorm > 1.0e-10, qnorm, 1.0)
    dscale = tl.where(norm > 0.0, norm / safe_qnorm, 0.0).to(tl.bfloat16)

    packed = ((idx_odd.to(tl.int32) << 4) | idx_even.to(tl.int32)) & 0xFF
    tl.store(
        Packed + row.to(tl.int64) * stride_p + offs, packed.to(tl.uint8)
    )
    tl.store(DScale + row, dscale)


class TurboQuantConstants:
    """Per-(head_dim, device) rotation + codebook constants and the fused pack path.

    Row-vector convention: the store-side rotation of a vector x (row) is
    ``x @ ROT_T`` with ROT_T = R^T = D1 H D2; the read side rotates Q with the same
    ROT_T and inverts the attention output with ROT = R = (ROT_T)^T.
    """

    def __init__(self, head_dim: int, device: torch.device, seed: int = TQ_SEED) -> None:
        if head_dim < 32 or head_dim & (head_dim - 1):
            raise ValueError(f"TurboQuant needs a power-of-two head_dim >= 32, got {head_dim}")
        self.head_dim = head_dim
        self.device = device
        rng = np.random.default_rng(seed)
        signs1 = rng.choice([-1.0, 1.0], size=head_dim)
        signs2 = rng.choice([-1.0, 1.0], size=head_dim)
        h = _orthonormal_hadamard(head_dim)
        rot_t = h * signs1[:, None] * signs2[None, :]  # R^T = D1 H D2
        self.rot_t = rot_t.to(torch.bfloat16).to(device)
        self.rot = rot_t.t().contiguous().to(torch.bfloat16).to(device)  # R
        radius = CLIP_SIGMAS / math.sqrt(head_dim)
        self.c0 = -radius
        self.step = (2.0 * radius) / (N_CENTROIDS - 1)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate head_dim-sized rows into the WHT domain: (..., D) -> (..., D)."""
        shape = x.shape
        return (x.reshape(-1, self.head_dim) @ self.rot_t).view(shape)

    def unrotate(self, o: torch.Tensor) -> torch.Tensor:
        """Invert the rotation on attention outputs: (..., D) @ R."""
        shape = o.shape
        return (o.reshape(-1, self.head_dim) @ self.rot).view(shape)

    def quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(T, heads * head_dim) compute-dtype -> packed (T*heads, D/2) uint8 + (T*heads,) bf16 dscale."""
        dim = self.head_dim
        rows = x.numel() // dim
        xf = x.reshape(rows, dim).float()
        norms = torch.linalg.norm(xf, dim=-1)
        unit = (xf / norms.clamp_min(1.0e-30).unsqueeze(-1)).to(torch.bfloat16)
        y = unit @ self.rot_t
        packed = torch.empty((rows, dim // 2), dtype=torch.uint8, device=x.device)
        dscale = torch.empty((rows,), dtype=torch.bfloat16, device=x.device)
        _tq_pack_kernel[(rows,)](
            y,
            norms,
            packed,
            dscale,
            self.c0,
            self.step,
            y.stride(0),
            packed.stride(0),
            DIM=dim,
            num_warps=2,
        )
        return packed, dscale


__all__ = ["TurboQuantConstants", "TQ_SEED"]
