"""Multi-Token Prediction (MTP) head for Qwen3.8-Flash-Next.

The checkpoint ships a 1-layer MTP head (``mtp.*``, 31 tensors) that the text-only loader
otherwise drops. This module owns it and, given the main model's final residual + the last
token, drafts the next tokens inside the same forward pass (self-speculation). Design notes:

* The routed MTP experts are bf16 stacked (not NVFP4, not per-expert), so they live in a
  :class:`MtpExpertBank` (GPU-resident when VRAM allows, pinned host RAM otherwise) and only
  the active top-k rows are gathered per step. The dense head alone is ~180MB.
* The MTP layer is a full-attention decoder block. We run it with DENSE GQA over a private
  paged KV (``mtp_k``/``mtp_v`` indexed by the same physical ``out_loc`` the main model uses).
  The checkpoint's QSA indexer is ignored -- dense attention attends to every key, a superset
  of the sparse selection -- which keeps this self-contained (no per-layer QSA slab/ring state).
* Input fusion follows the training reference (vLLM ``Qwen4ExpMultiTokenPredictor``,
  "residual_linear_shared"): ``a = fc_embedding(pre_fc_norm_embedding(E(x_{t+1})))`` is injected
  as a UNIT-WEIGHT broadcast into the multi stream (HC ``combine`` with ``s=None``), while
  ``pre_fc_norm_hidden`` is a FLAT GemmaRMSNorm over the whole ``hc*hidden`` stream (one
  statistic, NOT per-branch) feeding ``fc_hidden`` per branch -- the stream stays
  ``[T, hc*hidden]`` MULTI through the decoder block. The block's top mixer emits both the
  collapsed ``[T, hidden]`` for the shared lm_head AND the multi stream, which chains verbatim
  into the next draft (no collapse, no repeat).
* Context KV (``write_ctx_kv``): the head's attention context for target position ``t`` is the
  K/V of the FUSED row ``(E(x_{t+1}), H_t)`` at rope position ``t`` (the EAGLE shift convention,
  vLLM ``llm_base_proposer``: "row at position s consumes the token at s+1 with hidden at s,
  positions unchanged"). The main-model hook writes prefill rows in-batch (shifted by one) and
  defers each chunk's last row to the next forward; the scheduler rewrites slots ``L..L+m+1``
  with the true accepted rows at the end of every MTP step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
)
from freetoken.layers.rotary import get_rope
from freetoken.models.qwen3_5_moe.moe import _SharedExpert

from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


_Q4_GROUP = 32


def _quant_fp8(w: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    w = w.float()
    scale = (w.abs().amax() / 448.0).reshape(())
    return (w / scale).to(torch.float8_e4m3fn), scale.to(dtype)


def _quant_g32(w: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """ggml q4_0-style: symmetric groups of 32 on the LAST axis, values [-8, 7] packed two
    per uint8 (nibble + 8), one bf16 scale per group."""
    w = w.float()
    cols = w.shape[-1]
    wg = w.reshape(-1, cols // _Q4_GROUP, _Q4_GROUP)
    scale = wg.abs().amax(dim=-1, keepdim=True) / 7.0
    q = torch.round(wg / scale).clamp_(-8, 7)
    u = (q + 8).to(torch.uint8)
    packed = u[..., 0::2] | (u[..., 1::2] << 4)
    shape = (*w.shape[:-1], cols // 2)
    return packed.reshape(shape), scale.squeeze(-1).reshape(w.shape[:-1] + (-1,)).to(dtype)


def _dequant_g32(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    lo = (packed & 0xF).float() - 8.0
    hi = (packed >> 4).float() - 8.0
    q = torch.stack((lo, hi), dim=-1).flatten(-2)  # unpack in the pack order
    g = q.reshape(*q.shape[:-1], q.shape[-1] // _Q4_GROUP, _Q4_GROUP)
    return (g * scale.unsqueeze(-1)).to(dtype).reshape(q.shape)


class MtpExpertBank:
    """Quantizable bank for the MTP layer's routed experts; gather the active top-k per step.

    Layout matches the checkpoint's stacked tensors exactly: ``gate_up [E, 2*I, H]`` (gate|up fused
    on the output-row axis) and ``down [E, H, I]``. Storage is bf16, fp8-e4m3 (+ per-tensor scale)
    or q4_0-packed (+ per-group scales) per ``--mtp-expert-quant``; ``gather`` returns bf16 either
    way. GPU-resident when ``load`` gets a device and VRAM allows (drafts then pay zero PCIe);
    otherwise pinned CPU RAM with only the selected experts moving per step.
    """

    def __init__(self, num_experts: int, moe_inter: int, hidden: int, dtype: torch.dtype):
        self.num_experts = num_experts
        self.moe_inter = moe_inter
        self.hidden = hidden
        self.dtype = dtype
        # Lazy: the ~5GB expert weights are allocated (and pinned) only in load(), so a server
        # with --mtp off pays just the ~180MB dense head, never the expert RAM.
        self.gate_up = None  # [E, 2*I, H] (half-width when q4-packed), set by load()
        self.down = None  # [E, H, I], set by load()
        # Expert bytes held ON-DEVICE by the GPU-resident mode (0 = pinned host RAM path).
        # The engine prices this into every cache-budget fit check (startup auto-size + live
        # cache rebuild) so pool growth can never OOM the resident bank.
        self.gpu_bytes = 0
        self.quant = "none"
        self.stored_bytes = 0  # weights + scales, whichever storage mode load() chose
        self._scales: "Tuple[torch.Tensor, ...] | None" = None

    def load(self, gate_up: torch.Tensor, down: torch.Tensor,
             device: "torch.device | None" = None, quant: str = "none") -> None:
        """Allocate the bank from the checkpoint's stacked experts (bf16 ``[E, 2*I, H]`` /
        ``[E, H, I]``) under ``quant`` ("none" | "fp8" | "q4_0").

        With ``device`` given (GPU-resident MTP mode), the packed bank stays on that device:
        draft gathers become pure on-device indexing and the per-draft ~100MB host->GPU stream
        disappears (drafts stop sharing PCIe with the main model's expert streaming). On OOM the
        method rolls back and keeps the pinned host-RAM path, so a tight card serves exactly as
        before the feature existed.
        """
        quant = (quant or "none").lower()
        if quant == "none":
            gu, dn, scales = gate_up.to(self.dtype), down.to(self.dtype), None
        elif quant == "fp8":
            gu, s_gu = _quant_fp8(gate_up, self.dtype)
            dn, s_dn = _quant_fp8(down, self.dtype)
            scales = (s_gu, s_dn)
        elif quant in ("q4_0", "q4"):
            gu, s_gu = _quant_g32(gate_up, self.dtype)
            dn, s_dn = _quant_g32(down, self.dtype)
            scales = (s_gu, s_dn)
        else:
            raise ValueError(f"unknown --mtp-expert-quant {quant!r}")
        parts = (gu, dn) + (scales or ())
        self.stored_bytes = sum(int(t.numel()) * t.element_size() for t in parts)
        if device is not None:
            try:
                self.gate_up, self.down = gu.to(device), dn.to(device)
                scales = None if scales is None else tuple(s.to(device) for s in scales)
                self.gpu_bytes = self.stored_bytes
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                self.gate_up = self.down = None
                torch.cuda.empty_cache()  # retry as the pinned host path below
        if self.gate_up is None:
            pinned = []
            for t in parts:
                try:  # best-effort pin (e.g. a CPU-only test env cannot pin)
                    pinned.append(t.pin_memory())
                except Exception:
                    pinned.append(t)
            self.gate_up, self.down = pinned[0], pinned[1]
            scales = tuple(pinned[2:]) if scales is not None else None
        self.quant = quant if quant != "q4" else "q4_0"
        self._scales = scales

    def gather(self, topk_idx: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(gate_up, down)`` bf16 for the selected experts, on ``device``.

        ``topk_idx [T, K]`` long. Returns ``gate_up [T, K, 2*I, H]``, ``down [T, K, H, I]``.
        Quantized storage is dequantized right after the index so only the gathered rows are
        ever expanded.
        """
        assert self.gate_up is not None and self.down is not None, (
            "MtpExpertBank.load() must be called before gather() (enable --mtp)"
        )
        idx = topk_idx.to(self.gate_up.device)
        gu, dn = self.gate_up[idx], self.down[idx]
        if self.quant == "none":
            return gu.to(device), dn.to(device)
        s_gu, s_dn = self._scales  # type: ignore[misc]
        if self.quant == "fp8":
            return (gu.to(self.dtype) * s_gu.to(device)), (dn.to(self.dtype) * s_dn.to(device))
        # q4 scales are per (expert, output row): index them with the same top-k so each
        # gathered row keeps its own group scales. (fp8 scales are scalars, no indexing.)
        return (
            _dequant_g32(gu, s_gu[idx].to(device), self.dtype).to(device),
            _dequant_g32(dn, s_dn[idx].to(device), self.dtype).to(device),
        )


class Qwen4ExpMTPMoE(BaseOP):
    """The MTP layer's MoE: same gate/shared-expert/shared-gate as :class:`Qwen4ExpMoE`, but the
    routed experts come from the :class:`MtpExpertBank` (top-k gathered per step) instead of the
    NVFP4 offload cache (the MTP experts are bf16, a different packing)."""

    def __init__(self, config: ModelConfig, bank: MtpExpertBank, *, prefix: str = ""):
        self._bank = bank
        self._topk = int(config.num_experts_per_tok)
        self._renorm = bool(config.norm_topk_prob)
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate",
        )
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(
            config.hidden_size, 1, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.shared_expert_gate",
        )

    def _routed(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = x.shape
        # Same router as the main model's MoE: softmax over ALL experts first, then top-k, then
        # renormalize (on Windows this is the pure-torch path fused_topk falls back to --
        # bit-identical router, just unfused). Selecting top-k of RAW logits and renormalizing by
        # their SUM instead is unbounded: when the selected logits straddle zero the sum is ~0 and
        # the weights explode (observed mlp_out std ~1e9, head output random => acceptance 0).
        from freetoken.moe.fused import fused_topk

        topk_w, topk_i = fused_topk(
            hidden_states=x,
            gating_output=router_logits,
            topk=self._topk,
            renormalize=self._renorm,
        )
        gu, dn = self._bank.gather(topk_i, x.device)  # [T,K,2I,H], [T,K,H,I]
        # Per (token, selected-expert) GEMM: act[t,k] = gate_up[t,k] @ x[t]. The two I halves are
        # gate|up (fused on the row axis); silu-and-mul then the down projection back to hidden.
        act = torch.einsum("tkch,th->tkc", gu, x.to(gu.dtype))  # [T,K,2I]
        g, u = act.chunk(2, dim=-1)  # [T,K,I] each
        act = F.silu(g) * u  # [T,K,I]
        y = torch.einsum("tkhi,tki->tkh", dn, act.to(dn.dtype))  # [T,K,H]
        y = (y * topk_w.to(y.dtype).unsqueeze(-1)).sum(1)  # [T,H]
        return y.view(num_tokens, hidden_dim).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self._routed(hidden_states, router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


class Qwen4ExpMTPAttention(BaseOP):
    """Dense GQA for the MTP layer. Same projections/norms/rope as :class:`Qwen4ExpAttention` (the
    q-projection carries the output gate, so it is twice the attention width), but the attend is a
    plain paged GQA over a private ``mtp_k``/``mtp_v`` pool instead of the QSA sparse backend.

    The pool is indexed by the same physical ``out_loc`` the main model writes, so the MTP K/V for a
    token sits beside that token's main-model K/V. New K/V is written here before the attend.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        args = config.qwen4_args
        self.hidden_size = args.hidden_size
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.qo_attn_dim = self.num_q * self.head_dim
        self.kv_attn_dim = self.num_kv * self.head_dim
        self._qkv_split = [self.qo_attn_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.sm_scale = config.attn_sm_scale or self.head_dim**-0.5
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
            mrope_section=tuple(rotary.mrope_section) if rotary.mrope_section is not None else None,
            mrope_layout=rotary.mrope_layout,
        )
        # Private paged KV, filled by the engine before the first MTP forward (physical-loc indexed).
        self.mtp_k: Optional[torch.Tensor] = None
        self.mtp_v: Optional[torch.Tensor] = None

    def attach_pool(self, k_pool: torch.Tensor, v_pool: torch.Tensor) -> None:
        """``k_pool``/``v_pool``: ``[total_tokens, num_kv, head_dim]`` on the model device."""
        self.mtp_k = k_pool
        self.mtp_v = v_pool

    def forward(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        ctx = get_global_ctx()
        device = x.device
        # The head may run on CPU (hidden behind GPU compute) while the scheduler keeps its
        # page-table / out_loc metadata on GPU. Bring that index metadata onto the head device
        # once so the per-request gathers (and the torch.cat below) see a single device.
        page_table = ctx.page_table.to(device) if ctx.page_table.device != device else ctx.page_table
        out_loc = batch.out_loc
        if out_loc is not None and out_loc.device != device:
            out_loc = out_loc.to(device)
        if self.mtp_k is None or out_loc is None:
            raise RuntimeError("MTP KV pool not attached; call attach_pool before forward")

        T = x.shape[0]
        qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()  # [T, num_q, head_dim]
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)  # [T, qo]
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)  # [T, num_kv, head_dim]
        v = v.contiguous().view(-1, self.num_kv, self.head_dim)  # [T, num_kv, head_dim]
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(
            batch.get_attn_positions(), q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim)
        )
        q = q.view(-1, self.num_q, self.head_dim)
        k = k.view(-1, self.num_kv, self.head_dim)

        dtype = x.dtype
        out = torch.empty(T, self.num_q, self.head_dim, device=device, dtype=dtype)

        # Per-(padded-)request dense attend. In a pure decode batch each request contributes exactly
        # one new token at ``out_loc[i]``; the query attends to its full prefix (gathered by the
        # page-table row) plus its own freshly-written K/V.
        offset = 0
        for i, req in enumerate(batch.padded_reqs):
            n = req.extend_len
            if n <= 0:
                continue
            new_loc = out_loc[offset : offset + n]
            self.mtp_k[new_loc] = k[offset : offset + n]
            self.mtp_v[new_loc] = v[offset : offset + n]
            qi = q[offset : offset + n]  # [n, num_q, head_dim]

            prefix_len = req.cached_len
            if prefix_len > 0:
                prefix = page_table[req.table_idx, :prefix_len]
            else:
                prefix = torch.empty(0, dtype=torch.long, device=device)
            # Prior draft entries in a K-draft chain: their MTP-space K/V sits in the scratch
            # slots (beyond the page-allocated pool region), not in the page-table row.
            extra = getattr(batch, "mtp_extra_locs", None)
            if extra is not None and extra.numel() > 0:
                prefix = torch.cat([prefix, extra.to(torch.long).to(device)])
            prefix_len = prefix.shape[0]
            locs = torch.cat([prefix, new_loc.to(torch.long)])  # [prefix_len + n]
            keys = self.mtp_k[locs]  # [P+n, num_kv, head_dim]
            vals = self.mtp_v[locs]  # [P+n, num_kv, head_dim]

            # Causal visibility for the n new queries: query r (0-based) sees all prefix_len prior
            # keys plus new[0..r] (its own position and earlier new tokens).
            col = torch.arange(prefix_len + n, device=device)  # [P+n]
            row = torch.arange(n, device=device)  # [n]
            vis = col.unsqueeze(0) <= (prefix_len + row.unsqueeze(1))  # [n, P+n] bool
            qs = qi.transpose(0, 1).unsqueeze(0)  # [1, num_q, n, head_dim]
            ks = keys.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv, P+n, head_dim]
            vs = vals.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv, P+n, head_dim]
            o = F.scaled_dot_product_attention(
                qs, ks, vs, attn_mask=vis.unsqueeze(0).unsqueeze(0),
                scale=self.sm_scale, enable_gqa=True,
            )  # [1, num_q, n, head_dim]
            out[offset : offset + n] = o.squeeze(0).transpose(0, 1).to(dtype)  # [n, num_q, head_dim]
            offset += n

        gated = out.view(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)


class Qwen4ExpMTP(BaseOP):
    """The 1-layer MTP head (reference: vLLM ``Qwen4ExpMultiTokenPredictor``). ``forward`` maps
    rows ``(E(x_{t+1}), multi_t) -> (draft logits for x_{t+2}, multi_{t+1})``; the dense part is
    ~180MB, the routed experts come from the bank, and ``write_ctx_kv``/``write_rows`` maintain
    the attention's context KV of fused rows (see their docstrings)."""

    def __init__(self, config: ModelConfig, bank: MtpExpertBank, *, prefix: str = "mtp"):
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.bank = bank  # plain object (not a BaseOP/Tensor) -> never in the state dict
        eps = config.rms_norm_eps

        # Input fusion (reference: vLLM Qwen4ExpMultiTokenPredictor). The embedding branch
        # ``a`` is injected into the multi stream with UNIT weight (HC combine, s=None);
        # pre_fc_norm_hidden is FLAT over hc*hidden (one statistic), fc_hidden is applied
        # per branch (shared weight over the last dim), and the stream stays multi.
        # Every Linear looks itself up in the checkpoint's quant config under ``prefix``:
        # the head is bf16 there (ignore: "mtp.*"), so an empty prefix would fall into the
        # "targets all Linear" NVFP4 rule and build a packed layout the checkpoint cannot fill.
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            config.hidden_size, eps, num_groups=1
        )
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.ple_state_width, eps, num_groups=1
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_hidden",
        )
        # Top mixer: collapses the multi stream to [T, hidden] for the shared lm_head.
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer"
        )

        # The single MTP decoder block: dense attention + hyper-connections + bank MoE.
        self.self_attn = Qwen4ExpMTPAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen4ExpMTPMoE(config, bank, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")

        # Context-KV writer state (see write_ctx_kv). ``_embed`` is attached by the engine
        # (the head otherwise never owns the embedding table). Per-request bookkeeping keyed
        # by table_idx: ``_ctx_pending[table] = (pos, multi_row)`` is the deferred row whose
        # next-token id was not known yet at the writing forward; ``_ctx_upto[table]`` is the
        # highest slot index whose context row is provably true. ``_ctx_write_off`` disables
        # the hook around verify / rollback re-runs (their rows may carry draft tokens; the
        # scheduler rewrites those slots with the true accepted rows instead).
        self._embed = None
        self._ctx_pending: dict[int, tuple[int, torch.Tensor]] = {}
        self._ctx_upto: dict[int, int] = {}
        self._ctx_write_off = False

    def attach_embed(self, embed_tokens) -> None:
        """Give the context-KV writer the main model's embedding table (engine wiring)."""
        self._embed = embed_tokens

    def write_rows(
        self,
        tokens_next: torch.Tensor,
        multi: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Write the context KV of the FUSED rows ``(E(x_{t+1}), H_t)`` at rope position ``t``.

        ``tokens_next [n]``: the token at ``t+1`` for each row; ``multi [n, hc*hidden]``: the main
        model's pre-top-mixer multi stream at ``t``; ``slots [n]``: physical MTP-pool slots;
        ``positions``: rope positions, 1-D ``[n]`` for a plain model or ``[3, n]`` (t/h/w rows) for
        an mrope model -- passed verbatim to ``attn.rotary``. These K/V are bit-consistent with what
        a head ``forward`` computes for the same row: the same fusion front, then the unit-weight
        embedding combine and ``attn_hyper_connection.mix`` -- attention K/V is the qkv of that
        mix output (vLLM combine_and_mix: mix input = norm(combine(R, a, None))).
        """
        attn = self.self_attn
        assert multi.dim() == 2, f"write_rows expects [n, hc*hidden] 2D multi, got {tuple(multi.shape)}"
        # Run the whole writer on the head device (mirrors forward): the main-forward inputs
        # arrive on the GPU, the head's norms/qkv/rope and KV pool may be CPU on the hidden path.
        mdev = next(self.parameters()).device
        multi = multi.to(mdev)
        slots = slots.to(mdev)
        positions = positions.to(mdev)
        n = multi.shape[0]
        a, R = self._fuse(tokens_next, multi)
        R = self.attn_hyper_connection.combine(R, a, None)  # unit-weight broadcast (reference)
        x, _ = self.attn_hyper_connection.mix(R)
        qg, k, v = attn.qkv_proj.forward(x).split(attn._qkv_split, dim=-1)
        k = k.contiguous().view(-1, attn.num_kv, attn.head_dim)
        v = v.contiguous().view(-1, attn.num_kv, attn.head_dim)
        attn.k_norm.forward_inplace(k)
        # In-place rope takes (positions, q, k); no query exists here, pass a throwaway.
        dummy_q = torch.zeros(n, attn.qo_attn_dim, device=multi.device, dtype=multi.dtype)
        _, k = attn.rotary.forward(positions, dummy_q, k.view(-1, attn.kv_attn_dim))
        k = k.view(-1, attn.num_kv, attn.head_dim)
        attn.mtp_k[slots] = k
        attn.mtp_v[slots] = v

    def write_ctx_kv(self, batch: Batch, multi: torch.Tensor) -> None:
        """Maintain the private MTP context KV from a MAIN-model forward. ``multi [T, hc*hidden]``
        is the pre-top-mixer stream this forward computed for each processed row.

        Convention: context slot ``t`` holds the fused row ``(E(x_{t+1}), H_t)`` at rope ``t``.
        Row ``t`` needs ``x_{t+1}``, which a forward at position ``t`` does not have yet, so the
        last row of every forward is DEFERRED (``_ctx_pending``) and written by the next forward
        of the same request from its first input token:

        * prefill/extend rows ``[pos_start, pos_end-1]``: shifted in-batch (row r consumes row
          r+1's input id, same multi row); the chunk's last row is stashed.
        * decode (only single-request eager decodes reach this hook -- multi-request decode is
          graph-replayed and the scheduler marks those requests' context dirty for rebuild):
          consumes the pending row at ``cached_len-1`` and stashes the current row.
        * verify / rollback re-runs: skipped via ``_ctx_write_off``; their slots are rewritten
          with the TRUE accepted rows by the scheduler at the end of each MTP step.
        """
        attn = self.self_attn
        if (
            attn.mtp_k is None
            or self._embed is None
            or batch.out_loc is None
            or self._ctx_write_off
        ):
            return
        page_table = get_global_ctx().page_table
        dev = multi.device
        # rope positions: an mrope model feeds [3, n] (the t/h/w axis rows) and a plain model
        # feeds a 1-D [n]. ``attn_pos`` is the batch's rope rows; slice along dim 1 (mrope)
        # or dim 0 (plain). The scheduler only sets ``mrope_positions`` for mrope models, so
        # this is the same switch that decides the shape ``get_attn_positions`` returns.
        mrope = batch.mrope_positions is not None
        attn_pos = batch.get_attn_positions()

        def attn_pos_for(lo: int, hi: int) -> torch.Tensor:
            return attn_pos[:, lo : hi] if mrope else attn_pos[lo : hi]

        def attn_pos_scalar(p: int, mrope_delta: int) -> torch.Tensor:
            # A deferred / last-text-token row sits in the generated-text tail, where all three
            # mrope axes share the coordinate ``p + delta`` (text delta is 0; matches the
            # scheduler's text branch of _make_mrope_positions).
            if mrope:
                return torch.full((3, 1), p + mrope_delta, dtype=torch.int32, device=dev)
            return torch.tensor([p], dtype=torch.int32, device=dev)

        if batch.phase == "decode":
            # Eager decode with one row per real request (the MTP anchor forward). Each request
            # consumes its pending row at ``cached_len-1`` and stashes its current row.
            off = 0
            n_real = len(batch.reqs)
            for i, req in enumerate(batch.padded_reqs):
                n = req.extend_len
                if n <= 0:
                    continue
                if i >= n_real:  # graph-padding duplicate row: not a real request
                    off += n
                    continue
                T = req.table_idx
                L = req.cached_len
                pend = self._ctx_pending.pop(T, None)
                upto = self._ctx_upto.get(T, -1)
                if pend is not None and pend[0] == L - 1 and L - 1 > upto:
                    slot = page_table[T, L - 1 : L].to(torch.long)
                    pos = attn_pos_scalar(L - 1, req.mrope_delta)
                    self.write_rows(
                        batch.input_ids[off : off + 1], pend[1], slot, pos
                    )
                    self._ctx_upto[T] = L - 1
                self._ctx_pending[T] = (L, multi[off : off + 1].clone())  # keep 2D [1, hc*hidden]
                off += n
            return

        # prefill / extend: one contiguous row range per real request
        off = 0
        n_real = len(batch.reqs)
        for i, req in enumerate(batch.padded_reqs):
            n = req.extend_len
            if n <= 0:
                continue
            if i >= n_real:  # graph-padding duplicate row: not a real range
                off += n
                continue
            T = req.table_idx
            pos_start = req.cached_len
            pos_end = req.device_len  # == pos_start + n
            # Reset per-request state on any prefill: table_idx is recycled across requests,
            # and a fresh (or full-rebuild) prefill rewrites the whole range anyway.
            pend = self._ctx_pending.pop(T, None)
            if pos_start > 0 and pend is not None and pend[0] == pos_start - 1:
                slot = page_table[T, pos_start - 1 : pos_start].to(torch.long)
                pos = attn_pos_scalar(pos_start - 1, req.mrope_delta)
                self.write_rows(batch.input_ids[off : off + 1], pend[1], slot, pos)
            # ``upto`` bookkeeping: pos_start == 0 means a fresh request (table_idx recycled)
            # -- OVERWRITE the stale value; a continuation chunk may keep the prior max.
            new_upto = pos_end - 2 if n > 1 else pos_start - 1  # shifted rows written: none
            if n > 1:
                self.write_rows(
                    batch.input_ids[off + 1 : off + n],  # x_{t+1} per row (shift by one)
                    multi[off : off + n - 1],            # H_t per row
                    batch.out_loc[off : off + n - 1].to(torch.long),
                    attn_pos_for(off, off + n - 1),
                )
            if pos_start > 0:
                base = pend[0] if (pend is not None and pend[0] == pos_start - 1) else -1
                new_upto = max(self._ctx_upto.get(T, base), new_upto, base)
            self._ctx_upto[T] = new_upto
            self._ctx_pending[T] = (pos_end - 1, multi[off + n - 1 : off + n].clone())  # 2D [1, hc*hidden]
            off += n

    def _fuse(self, input_ids: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference fusion front (vLLM Qwen4ExpMultiTokenPredictor.forward): embedding branch
        ``a [T, hidden]`` (fed to the first mix as a unit-weight combine) and the per-branch
        hidden branch ``R [T, hc*hidden]`` (FLAT norm -> shared fc_hidden per branch, MULTI)."""
        return self._fuse_with(self._embed, input_ids, residual)

    def _fuse_with(self, embed_tokens, input_ids: torch.Tensor, residual: torch.Tensor):
        T, H = residual.shape[0], self.hidden_size
        # Bridge the shared embed tower to the head device (see forward for the gpu/cpu rule).
        mdev = next(self.parameters()).device
        edev = next(embed_tokens.parameters()).device
        e = embed_tokens.forward(input_ids.to(edev)).to(mdev)  # [T, hidden]
        a = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(e))  # [T, hidden]
        rn = self.pre_fc_norm_hidden.forward(residual.to(mdev))              # flat, [T, hc*H]
        R = self.fc_hidden.forward(rn.reshape(-1, H)).reshape(T, -1)          # per-branch
        return a, R

    def forward(
        self,
        batch: Batch,
        residual: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(draft_logits [T, vocab], multi_hidden [T, hc*hidden])`` for the token two
        steps ahead of each row's hidden position (row t = (E(x_{t+1}), H_t) predicts x_{t+2}).

        ``residual`` is the main model's pre-top-mixer multi stream ``[T, hc*hidden]`` (j=1) or
        the previous draft's ``multi_hidden`` (chain, verbatim -- no collapse/repeat, reference);
        ``input_ids`` are the tokens AT ``t+1`` for each row (anchor / previous draft token).
        ``multi_hidden`` chains into the next draft; the collapsed stream feeds the lm_head.
        """
        # One-shot internal diagnostic: when the caller sets ``self._diag`` (a dict) before a
        # forward, record the stats of each intermediate so a 0%-acceptance head can be traced
        # to its degenerate stage (input fusion / attention / MoE / collapse) in one run.
        diag = getattr(self, "_diag", None)
        if diag is not None:
            self._diag = None

        def _rec(name: str, t: torch.Tensor) -> None:
            if diag is not None:
                if not torch.isfinite(t).all():
                    diag[name] = "NONFINITE"
                else:
                    diag[name] = (
                        f"min={float(t.min()):.4g} max={float(t.max()):.4g} "
                        f"mean={float(t.mean()):.4g} std={float(t.std()):.4g} "
                        f"l2={float(t.norm()):.4g}"
                    )

        # Device bridge for the CPU-hidden path (--mtp-device cpu): the head module, its KV pool
        # and rope cache are on CPU (mdev); the shared embed_tokens/lm_head stay on GPU. Compute the
        # two shared-tower ops on their own device and hop the activations onto mdev between them.
        # On the gpu path mdev==their device so every .to() below is a no-op.
        mdev = next(self.parameters()).device
        residual = residual.to(mdev)
        input_ids = input_ids.to(next(embed_tokens.parameters()).device)

        # --- input fusion (reference "residual_linear_shared", see _fuse) ---
        T, hc, H = residual.shape[0], self.hc_count, self.hidden_size
        e = embed_tokens.forward(input_ids).to(mdev)  # [T, hidden]
        _rec("embed", e)
        a = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(e))  # [T, hidden]
        _rec("a_emb_branch", a)
        rn = self.pre_fc_norm_hidden.forward(residual)  # FLAT norm, [T, hc*hidden]
        _rec("rn_normed_residual", rn)
        R = self.fc_hidden.forward(rn.reshape(-1, H)).reshape(T, hc * H)  # per-branch, MULTI
        _rec("R_hidden_branch", R)
        R = self.attn_hyper_connection.combine(R, a, None)  # unit-weight embedding broadcast
        _rec("R_with_emb", R)

        # --- one decoder block over the hyper-connection streams ---
        block_input, inject = self.attn_hyper_connection.mix(R)
        _rec("attn_mix_in", block_input)
        if diag is not None:
            diag["attn_inject"] = f"min={float(inject.min()):.4g} max={float(inject.max()):.4g}"
        block_output = self.self_attn.forward(block_input, batch)
        _rec("attn_out", block_output)
        R = self.attn_hyper_connection.combine(R, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(R)
        _rec("mlp_mix_in", block_input)
        mlp_out = self.mlp.forward(block_input)
        _rec("mlp_out", mlp_out)
        multi_hidden = self.mlp_hyper_connection.combine(R, mlp_out, inject)  # [T, hc*hidden]

        # --- collapse + draft logits (reference combine_and_mix of the top mixer) ---
        sample = self.hyper_connection_mixer.mix(multi_hidden)[0]  # [T, hidden]
        _rec("mtp_out", sample)
        logits = lm_head.forward(sample.to(next(lm_head.parameters()).device))  # [T, vocab]
        _rec("logits", logits)
        # The MULTI stream chains verbatim into the next draft (reference: the next step's
        # backbone hidden is the prior draft's pre-final-mixer multi, never a collapse).
        return logits, multi_hidden


__all__ = ["MtpExpertBank", "Qwen4ExpMTP", "Qwen4ExpMTPAttention", "Qwen4ExpMTPMoE"]
