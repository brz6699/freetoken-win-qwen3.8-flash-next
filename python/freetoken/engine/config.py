from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # --hf-overrides: applied to the checkpoint config the model is built from (cached_load_hf_config)
    hf_overrides: Mapping[str, Any] = field(default_factory=dict)
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Kernel-per-layer-kind quant backend selection (--quant-backend): comma-separated
    # layer[.kind]=name entries, parsed into a QuantBackend before the model is built.
    quant_backend: str | None = None
    # PLE table backend (--ple-backend): "disk" reads rows from the checkpoint files per fill,
    # "pinned" preloads the table into page-locked host RAM. The disk backend's C++ store
    # (_ple_store) is built on Linux only, so other platforms default to "pinned".
    ple_backend: str = "disk" if sys.platform == "linux" else "pinned"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    # --moe-cache-mem: slot-cache budget in bytes, turned into a slot count once the expert banks are loaded
    moe_cache_mem: int | None = None
    # --moe-cache-scope: unified (one LRU pool per expert quant) or per-layer (one pool per layer)
    moe_cache_scope: str = "unified"
    # --moe-cache-split: how pools share the slots; equal or profile:<path> (by expert use from a profile file)
    moe_cache_split: str = "equal"
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # Runtime KV<->MoE rebalancing on top of the startup plan: KV occupancy is sampled
    # per batch and the byte envelope is traded with the expert-slot cache at the idle
    # safe point (engine/budget_slide.py). Disable to freeze the startup split.
    budget_slide: bool = True
    moe_cache_policy: str = "lru"
    # A prefill chunk of at most this many tokens admits only its routed experts into the slot
    # cache (GPU decode only), like decode; 0 streams whole layers.
    moe_prefill_demand_tokens: int = 0
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # MTP speculative decoding (--mtp-draft-len): the checkpoint's native 1-layer head drafts
    # this many tokens on the GPU per decode step; the full model verifies them in one extend
    # forward and keeps the greedy-longest prefix. Setting the length IS the switch (None =
    # off); single greedy requests only.
    mtp_draft_len: int | None = None
    # MTP draft+verify compute device (--mtp-device): "gpu" runs the whole MTP path on the GPU;
    # "cpu" hides the head draft + verify compute on the CPU behind the GPU main-model forward.
    mtp_device: str = "gpu"
    # Storage dtype of the MTP head's routed-expert bank (--mtp-expert-quant):
    # "none" bf16 | "fp8" e4m3 + per-tensor scale | "q4_0" nibble-packed groups of 32.
    mtp_expert_quant: str = "none"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    # None = Algorithm 2 (declared sizes drive the budget: --num-tokens/--num-pages first,
    # then --moe-cache-size, with full VRAM as the ceiling). A float = Algorithm 1: ratio x
    # measured baseline minus weights is the budget envelope. Priority: ratio > KV > MoE.
    memory_ratio: float | None = None
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    # DeepSeek-V4.1 decoder SWA bounded replay: "bounded" (default, the tech-report 128-token
    # window) or "exact" (run the decoder layers on every prompt token).
    swa_decoder_replay: str = "bounded"
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Paged KV pool STORAGE dtype: "bf16" (default) | "fp8_e4m3" | "turbo4". fp8 halves the
    # KV pool at a one-time write-side quantization; turbo4 (TurboQuant/PolarQuant 4-bit,
    # kvcache/turboquant.py) halves it again -- nibble-packed uint8 buffers plus a per
    # (token, head) bf16 dequant scale. Attention compute stays bf16 either way and QSA
    # index keys stay bf16. QSA models (Qwen3.8-Flash-Next) only -- enforced by the engine.
    # Every per-byte budget formula must price kv_dtype/kv_cache_dtype, never dtype.
    kv_cache_dtype: str = "bf16"


    @property
    def enable_mtp(self) -> bool:
        """Derived from --mtp-draft-len: naming a draft length is what turns MTP on."""
        return self.mtp_draft_len is not None

    @cached_property
    def kv_dtype(self) -> torch.dtype:
        if self.kv_cache_dtype in ("fp8_e4m3", "fp8"):
            return torch.float8_e4m3fn
        if self.kv_cache_dtype == "turbo4":
            # uint8 is the PACKED carrier: two 4-bit indices per byte, head_dim halved.
            return torch.uint8
        if self.kv_cache_dtype not in ("bf16", "bfloat16"):
            raise ValueError(f"unknown --kv-cache-dtype {self.kv_cache_dtype!r}")
        return self.dtype

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path, self.hf_overrides)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> "tuple":
        """The encoder towers this process builds: the family registers them, the checkpoint config
        carries their section, and --mm-disable / --text-model-only did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        # The checkpoint's quant dialect is installed once here, before the weight
        # readers run: models/weight.py resolves each module through get_quant_config().
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        parse_config = _load_attr(spec.module, spec.parse_config)

        return replace(
            parse_config(hf_config),
            quant=quant,
            enable_mtp=self.enable_mtp,
            mtp_draft_len=self.mtp_draft_len,
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
