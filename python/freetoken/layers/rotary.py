from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        proportional: bool = False,
        attention_factor: float = 1.0,
        is_neox: bool = True,
        mrope_section: Tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # Interleaved M-RoPE (Qwen3.8 / Qwen4Exp): frequency j draws its position from
        # channel sel[j] of the per-token (t, h, w) triple. None keeps this a pure 1-D
        # rope instance (every text-only serving path).
        self._mrope_section = mrope_section
        self._mrope_sel: torch.Tensor | None = None
        # NeoX (half-rotation, HF default) vs GPT-J interleaved (adjacent pairs,
        # ``rope_interleave`` models: GLM MLA lineage). Both underlying kernels
        # accept the flag; the cos/sin cache layout is identical.
        self.is_neox = is_neox
        if proportional:
            assert 0 < rotary_dim <= head_size
            assert rotary_dim % 2 == 0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size)
            )
            if rotary_dim < head_size:
                inv_freq[rotary_dim // 2 :] = 0.0
        else:
            # Standard (NeoX) rope. Supports partial rotary (rotary_dim < head_size):
            # rope is applied to the first ``rotary_dim`` dims of each head, the rest pass
            # through. Frequencies are spaced over ``rotary_dim`` (matches HF default
            # partial rope, e.g. Qwen3.5 partial_rotary_factor, MiniMax-M2's
            # ``apply_rotary_pos_emb``). Full rope is rotary_dim == head_size and is
            # unaffected. ``head_size`` is passed to flashinfer separately so it rotates
            # only the first ``rotary_dim`` dims.
            assert 0 < rotary_dim <= head_size
            assert rotary_dim % 2 == 0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
            )
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import apply_rope_with_cos_sin_cache_inplace
        else:
            from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
            is_neox=self.is_neox,
        )
        return query, key

    def _mrope_freq_channels(self, device: torch.device) -> torch.Tensor:
        """[rotary_dim//2] channel (0=t, 1=h, 2=w) per frequency, mirroring HF
        ``apply_interleaved_mrope``: channel d owns indices ``offset_d : 3*section_d : 3``."""
        if self._mrope_sel is None or self._mrope_sel.device != device:
            sel = torch.zeros(self.rotary_dim // 2, dtype=torch.long)
            for d, offset in ((1, 1), (2, 2)):
                sel[offset : 3 * self._mrope_section[d] : 3] = d
            self._mrope_sel = sel.to(device)
        return self._mrope_sel

    def forward_mrope(
        self,
        positions3: torch.Tensor,  # [T, 3] int32 (t/h/w per token)
        query: torch.Tensor,  # [T, num_q_heads * head_size], rotated in place
        key: torch.Tensor,  # [T, num_kv_heads * head_size], rotated in place
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """M-RoPE twin of :meth:`forward` for image-bearing batches.

        Per-token cos/sin are assembled from the SAME fp32 cache the 1-D flashinfer path
        reads, so text tokens (t==h==w) rotate bit-identically to ``forward``; only the
        gather index changes. Eager-only by design (image batches never ride a graph).
        """
        r = self.rotary_dim // 2
        sel = self._mrope_freq_channels(query.device)
        channels = positions3.long()  # [T, 3]
        pos_sel = torch.gather(channels, 1, sel.unsqueeze(0).expand(channels.shape[0], r))
        dims = torch.arange(r, device=query.device)
        cos = self._cos_sin_cache[pos_sel, dims]  # [T, r] fp32
        sin = self._cos_sin_cache[pos_sel, r + dims]

        for x in (query, key):
            heads = x.shape[1] // self.head_size
            v = x.view(-1, heads, self.head_size)
            rot = v[..., : self.rotary_dim].float()
            if not self.is_neox:
                raise NotImplementedError("mrope is only wired for NeoX rope layout")
            a, b = rot[..., :r], rot[..., r:]
            c, s = cos.unsqueeze(1), sin.unsqueeze(1)  # [T, 1, r]
            v[..., : self.rotary_dim] = torch.cat((a * c - b * s, b * c + a * s), dim=-1).to(
                v.dtype
            )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox: bool = True,
    mrope_section: Tuple[int, ...] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(
            head_dim, rotary_dim, max_position, base, is_neox=is_neox, mrope_section=mrope_section
        )
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                is_neox=is_neox,
                mrope_section=mrope_section,
            )

        case "proportional":
            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                proportional=True,
                is_neox=is_neox,
                mrope_section=mrope_section,
            )

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                post_process,
                is_neox=is_neox,
                mrope_section=mrope_section,
            )

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def get_mscale(scale: float, mscale: float = 1.0) -> float:
                if scale <= 1:
                    return 1.0
                return 0.1 * mscale * math.log(scale) + 1.0

            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                mscale = rope_scaling.get("mscale")
                mscale_all_dim = rope_scaling.get("mscale_all_dim")
                # Truthiness, not presence: HF falls back to get_mscale(factor) when
                # mscale_all_dim is 0 (a real DeepSeek-lineage default).
                if mscale and mscale_all_dim:
                    attention_factor = get_mscale(factor, mscale) / get_mscale(
                        factor, mscale_all_dim
                    )
                else:
                    attention_factor = get_mscale(factor)

            def _find_correction_dim(num_rotations: float) -> float:
                return (
                    rotary_dim
                    * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = _find_correction_dim(beta_fast)
            high = _find_correction_dim(beta_slow)
            if rope_scaling.get("truncate", True):
                low = math.floor(low)
                high = math.ceil(high)
            low = max(low, 0)
            # rotary_dim - 1, per HF's find_correction_range and this repo's own faithful copy in
            # models/deepseek_v4/ops.py. Clamping to rotary_dim//2 - 1 instead forces the ramp to
            # reach 1.0 at the last entry, fully interpolating the longest-wavelength dims that
            # the reference deliberately leaves partly extrapolated.
            high = min(high, rotary_dim - 1)
            if low == high:  # HF nudges instead of flooring the gap at 1 ("truncate": false)
                high += 0.001

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                post_process,
                attention_factor=float(attention_factor),
                is_neox=is_neox,
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox: bool = True,
    mrope_section: Tuple[int, ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(
                head_dim, rotary_dim, max_position, base, rope_map, is_neox, mrope_section
            )
    return _get_rope(
        head_dim, rotary_dim, max_position, base, rope_map, is_neox, mrope_section
    )


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
