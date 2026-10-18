from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import periph_flashinfer_ok

        from freetoken.kernel.triton.norm import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        # Triton wrapper handles both CUDA and CPU; flashinfer is faster on CUDA but
        # rejects CPU tensors, so keep the Triton binding as the CPU fallback even when
        # flashinfer backs the GPU path (the CPU-hidden MTP head).
        self.rmsnorm_cpu = rmsnorm
        self.rmsnorm = rmsnorm
        if periph_flashinfer_ok():
            from flashinfer import rmsnorm

            self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self.rmsnorm if x.is_cuda else self.rmsnorm_cpu
        return fn(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        fn = self.rmsnorm if x.is_cuda else self.rmsnorm_cpu
        fn(x, self.weight, self.eps, out=x)


class GemmaRMSNorm(BaseOP):
    """Gemma4-style RMSNorm backed directly by sgl_kernel.

    Gemma4 scales by the raw checkpoint weight. ``with_scale=False`` uses a
    runtime ones vector that is intentionally not part of ``state_dict``.
    """

    def __init__(self, size: int, eps: float, with_scale: bool = True) -> None:
        from freetoken.kernel.backend import is_sgl_kernel_installed
        from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.size = size
        self.with_scale = with_scale
        self.rmsnorm_cpu = rmsnorm
        self.fused_add_rmsnorm_cpu = fused_add_rmsnorm
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
        if is_sgl_kernel_installed():
            from sgl_kernel import fused_add_rmsnorm, rmsnorm

            self.rmsnorm = rmsnorm
            self.fused_add_rmsnorm = fused_add_rmsnorm
        if with_scale:
            self.weight = torch.empty(size)
        else:
            self._ones_weight: torch.Tensor | None = None

    def _kernel_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_scale:
            return self.weight
        if self._ones_weight is None:
            self._ones_weight = torch.ones(self.size, device=x.device, dtype=x.dtype)
        return self._ones_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self.rmsnorm if x.is_cuda else self.rmsnorm_cpu
        if x.dim() == 2:
            return fn(x, self._kernel_weight(x), self.eps)
        original_shape = x.shape
        out = fn(
            x.contiguous().reshape(-1, original_shape[-1]),
            self._kernel_weight(x),
            self.eps,
        )
        return out.reshape(original_shape)

    def forward_add_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fn = self.fused_add_rmsnorm if x.is_cuda else self.fused_add_rmsnorm_cpu
        fn(x, residual, self._kernel_weight(x), self.eps)
        return x, residual


class GemmaPlusOneRMSNorm(BaseOP):
    """(1 + w)-scaled RMSNorm (Gemma semantics: the checkpoint stores ``scale - 1``
    and the effective multiplier is ``1 + weight``, added in fp32 at runtime --
    never folded into the bf16 weight, which would round away the precision the
    format exists to keep). Used by MiniMax-M3 (``use_gemma_norm``) for the decoder
    layernorms, the per-head q/k norms, and the indexer q/k norms.

    Per-head 3D inputs are collapsed to 2D before the kernel call: flashinfer's
    ``gemma_rmsnorm`` CUDA binding is CHECK_DIM(2) (it rejects 3D outright on
    wheels without the CuTe path), and the per-head weight makes the 2D view
    exactly equivalent. The M3 call sites pass contiguous buffers, so the views
    are free; a non-contiguous 3D input is rejected rather than silently copied
    (an in-place norm on a copy would be dropped).
    """

    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import periph_flashinfer_ok

        from freetoken.kernel.triton.norm import gemma_rmsnorm

        self.eps = eps
        self.size = size
        self.weight = torch.empty(size)
        self.gemma_rmsnorm_cpu = gemma_rmsnorm
        self.gemma_rmsnorm = gemma_rmsnorm
        if periph_flashinfer_ok():
            from flashinfer.norm import gemma_rmsnorm

            self.gemma_rmsnorm = gemma_rmsnorm

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        assert x.is_contiguous(), "per-head gemma norm needs a contiguous buffer"
        return x.view(-1, self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self.gemma_rmsnorm if x.is_cuda else self.gemma_rmsnorm_cpu
        return fn(self._flat(x), self.weight, self.eps).view(x.shape)

    def forward_inplace(self, x: torch.Tensor) -> None:
        fn = self.gemma_rmsnorm if x.is_cuda else self.gemma_rmsnorm_cpu
        flat = self._flat(x)
        fn(flat, self.weight, self.eps, out=flat)


class GemmaPlusOneRMSNormFused(BaseOP):
    """(1 + w)-scaled RMSNorm with the fused-add-residual API of ``RMSNormFused``
    (Gemma semantics, see :class:`GemmaPlusOneRMSNorm`). Drop-in for the decoder
    layernorm seam: ``forward(x, residual)`` returns ``(normed, residual)``."""

    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import periph_flashinfer_ok

        from freetoken.kernel.triton.norm import (
            gemma_fused_add_rmsnorm,
            gemma_rmsnorm,
        )

        self.eps = eps
        self.weight = torch.empty(size)
        self.gemma_rmsnorm_cpu = gemma_rmsnorm
        self.gemma_fused_add_rmsnorm_cpu = gemma_fused_add_rmsnorm
        self.gemma_rmsnorm = gemma_rmsnorm
        self.gemma_fused_add_rmsnorm = gemma_fused_add_rmsnorm
        if periph_flashinfer_ok():
            from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

            self.gemma_rmsnorm = gemma_rmsnorm
            self.gemma_fused_add_rmsnorm = gemma_fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            fn = self.gemma_rmsnorm if x.is_cuda else self.gemma_rmsnorm_cpu
            return fn(x, self.weight, self.eps), x
        fn = (
            self.gemma_fused_add_rmsnorm
            if x.is_cuda
            else self.gemma_fused_add_rmsnorm_cpu
        )
        fn(x, residual, self.weight, self.eps)
        return x, residual


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import periph_flashinfer_ok

        from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm_cpu = rmsnorm
        self.fused_add_rmsnorm_cpu = fused_add_rmsnorm
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
        if periph_flashinfer_ok():
            from flashinfer import fused_add_rmsnorm, rmsnorm

            self.rmsnorm = rmsnorm
            self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            fn = self.rmsnorm if x.is_cuda else self.rmsnorm_cpu
            return fn(x, self.weight, self.eps), x
        fn = (
            self.fused_add_rmsnorm
            if x.is_cuda
            else self.fused_add_rmsnorm_cpu
        )
        fn(x, residual, self.weight, self.eps)
        return x, residual


_GATE_ACTIVATIONS = ("silu", "swish", "sigmoid")


class GatedRMSNorm(BaseOP):
    """``rmsnorm(x) * activation(z)`` in one fla kernel: the GDN / KDA output norm."""

    def __init__(self, size: int, eps: float, activation: str = "silu") -> None:
        # rms_norm_gated drops the gate entirely (no error) for a name it does not know
        assert activation in _GATE_ACTIVATIONS, f"unsupported gate activation {activation!r}"
        self.weight = torch.empty(size)
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation=self.activation,
        )


class LayerNorm(BaseOP):
    """LayerNorm with bias on torch's fused kernel; the decoders use the RMSNorm family."""

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
