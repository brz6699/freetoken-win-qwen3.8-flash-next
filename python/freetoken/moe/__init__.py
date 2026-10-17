from __future__ import annotations

from typing import Protocol

from freetoken.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_STRATEGIES = Registry[MoeBackendCreator]("MoE Backend")

# Backends that serve experts from CPU (pinned) host banks through an
# ``OffloadMoeCache`` -- the GPU only holds the two-layer prefill double buffer.
# They differ only in how *decode* gets the experts: ``offload`` streams the
# missing experts over PCIe into a GPU slot cache and runs the GEMM on the GPU;
# ``cpu`` ships the activations to the CPU, computes the experts there (high RAM
# bandwidth), and ships the results back; ``hybrid`` keeps a full GPU slot cache
# AND a CPU executor -- it fetches at most K missing experts/layer over PCIe
# (computed on the GPU with the cache hits) and computes the remaining misses on
# the CPU, overlapped, then merges (capped PCIe + CPU overflow). All build their
# cache the same way, so model layer construction and the engine wiring key off
# this set rather than a bare ``== "offload"`` check.
OFFLOAD_MOE_STRATEGIES = frozenset({"offload", "cpu", "hybrid"})


def is_offload_moe_strategy(backend: str) -> bool:
    return backend in OFFLOAD_MOE_STRATEGIES


@SUPPORTED_MOE_STRATEGIES.register("fused")
def create_fused_moe_strategy():
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_STRATEGIES.register("offload")
def create_offload_moe_strategy():
    from .offload import OffloadMoeBackend

    return OffloadMoeBackend()


@SUPPORTED_MOE_STRATEGIES.register("cpu")
def create_cpu_moe_strategy():
    from .cpu_offload import CpuOffloadMoeBackend

    return CpuOffloadMoeBackend()


@SUPPORTED_MOE_STRATEGIES.register("hybrid")
def create_hybrid_moe_strategy():
    from .cpu_offload import HybridMoeBackend

    return HybridMoeBackend()


def create_moe_strategy(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_STRATEGIES[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_strategy",
    "SUPPORTED_MOE_STRATEGIES",
    "OFFLOAD_MOE_STRATEGIES",
    "is_offload_moe_strategy",
]
