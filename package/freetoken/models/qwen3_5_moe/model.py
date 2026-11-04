from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class FinalNormMixer:
    """The ``hc``-free twin of qwen4_exp's top mixer: this family keeps one hidden stream, so
    collapsing the residual to the lm_head input is just the final RMSNorm. ``mix`` keeps the
    tuple shape the MTP backends unpack (``mix(residual)[0]``)."""

    def __init__(self, norm: GemmaRMSNorm):
        self.norm = norm

    def mix(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.norm.forward(residual),)


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # plain objects, so neither shows up in the state dict: the mixer shim just delegates to
        # ``norm`` above (already a child), and the head reference is set by the engine on --mtp.
        self.hyper_connection_mixer = FinalNormMixer(self.norm)
        self._mtp_head = None
        self._mtp_hidden: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        x = embed_input_ids(self.embed_tokens, input_ids, batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # fused add+norm leaves the summed hidden in ``residual`` and the normed one in ``x``. Both
        # the LM head and the nextn head read the NORMED stream: llama.cpp assigns ``t_h_nextn``
        # after ``output_norm`` (src/models/qwen35moe.cpp), so the head's ``hnorm`` always sees an
        # already-normed h and never a raw sum whose scale drifts with sequence depth.
        x, residual = self.norm.forward_add_residual(x, residual)
        self._last_residual = residual
        self._mtp_hidden = x
        if self._mtp_head is not None:
            self._mtp_head.write_ctx_kv(batch, x)
        return x


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        # The nextn block these GGUFs carry, as a draft head. Built only with --mtp so an
        # --mtp-off serve pays nothing (the loader then drops the mtp.* keys as well).
        self.mtp = None
        if getattr(config, "enable_mtp", False):
            from .mtp import Qwen3_5MTP

            self.mtp = Qwen3_5MTP(config)
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Load the draft head's routed experts; returns the pinned host bytes the engine
        reserves from its pin budget (0 when they went GPU-resident or were not needed)."""
        head = getattr(self, "mtp", None)
        if head is None or getattr(engine_config, "use_dummy_weight", False):
            return 0
        from .mtp import load_mtp_experts

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        quant = str(getattr(engine_config, "mtp_expert_quant", "none") or "none").lower()
        return load_mtp_experts(engine_config.model_path, head.bank, device=device, quant=quant)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.5 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
