"""The nextn (MTP) head a ``qwen35`` / ``qwen35moe`` GGUF carries on its last block.

llama.cpp's converter writes the head as ``blk.N.nextn.*`` plus one ordinary decoder block in
``blk.N.*`` (full attention + MoE, no GDN), and its graph (``src/models/qwen35moe.cpp``) is::

    cur  = eh_proj @ cat(enorm(E(x_{t+1})), hnorm(h_t))          # inpSA
    cur  = wo(attn(wnorm(cur)) * sigmoid(gate)) + cur
    cur  = moe(wpost(cur)) + sigmoid(g_shexp @ wpost(cur)) * shexp(wpost(cur)) + cur
    out  = shared_head_norm(cur) -> lm_head

so the row at rope position ``t`` consumes the NEXT token's embedding and THIS position's
hidden -- the EAGLE shift. :meth:`Qwen3_5MTP.forward` returns that row's draft logits plus the
pre-``shared_head_norm`` hidden, which chains verbatim into the next draft of the chain.

The head attends over its OWN paged KV (``attach_pool``): its q/k/v live in a different space
from the main model's, so context rows are written by :meth:`write_rows` under the same shift --
slot ``t`` holds the fused row ``(E(x_{t+1}), h_t)``. Only the head's own block is its business;
the routed experts come from :class:`MtpExpertBank` (gathered top-k per step) so the draft never
waits on the main model's expert streaming.

Chaining convention (llama.cpp ``src/models/qwen35moe.cpp``): the head publishes
``res->t_h_nextn`` AFTER ``shared_head_norm`` and feeds that same tensor to the LM head, so the
hidden chained into the next draft (the next ``hnorm`` input) is the POST-norm one. The main
model's seed hidden is likewise post-``output_norm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
)
from freetoken.layers.rotary import get_rope
from freetoken.models.gguf.experts import _expert_rows
from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.models.qwen4_exp.mtp import MtpExpertBank
from freetoken.third_party.gguf import quants
from freetoken.third_party.gguf.constants import GGMLQuantizationType
from freetoken.utils import init_logger

from .moe import Qwen3_5DenseMLP, _SharedExpert

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

logger = init_logger(__name__)


class Qwen3_5MTPAttention(BaseOP):
    """Gated full attention over the head's private KV -- the same projections, q/k norms and
    partial mrope as :class:`~freetoken.models.qwen3_5_moe.attention.Qwen3_5Attention`, but the
    attend is a plain ``sdpa`` over ``mtp_k``/``mtp_v`` indexed by the physical locs the page
    table hands out (plus the draft-chain scratch slots in ``batch.mtp_extra_locs``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        head_dim = config.head_dim
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim
        # q half is twice the attention width because it carries the per-head output gate
        self._qkv_split = [self.qo_attn_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.sm_scale = config.attn_sm_scale or head_dim**-0.5
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        # Gemma-style norms: the GGUF stores the effective multiplier, these scale by the raw weight
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
            mrope_section=(
                tuple(rotary.mrope_section) if rotary.mrope_section is not None else None
            ),
            mrope_layout=rotary.mrope_layout,
        )
        self.mtp_k: Optional[torch.Tensor] = None
        self.mtp_v: Optional[torch.Tensor] = None

    def attach_pool(self, k_pool: torch.Tensor, v_pool: torch.Tensor) -> None:
        """``k_pool``/``v_pool``: ``[total_tokens, num_kv, head_dim]`` on the head's device."""
        self.mtp_k = k_pool
        self.mtp_v = v_pool

    def _project(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """``(q, k, v, gate)`` after the q/k norms and rope; q is ``[N, num_q, head_dim]``, k/v
        ``[N, num_kv, head_dim]``, gate ``[N, qo_attn_dim]``."""
        qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        v = v.contiguous().view(-1, self.num_kv, self.head_dim)
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        return q.view(-1, self.num_q, self.head_dim), k.view(-1, self.num_kv, self.head_dim), v, gate

    def forward(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        ctx = get_global_ctx()
        device = x.device
        # The head may sit on the CPU while the page table and out_loc stay on the GPU, so bring
        # the index metadata onto the head device once (see Qwen4ExpMTPAttention.attach_pool).
        page_table = ctx.page_table
        if page_table.device != device:
            page_table = page_table.to(device)
        out_loc = batch.out_loc
        assert self.mtp_k is not None and out_loc is not None, (
            "MTP KV pool not attached; the engine wires attach_pool when --mtp is on"
        )
        if out_loc.device != device:
            out_loc = out_loc.to(device)
        positions = batch.get_attn_positions()
        q, k, v, gate = self._project(x, positions)

        extra = getattr(batch, "mtp_extra_locs", None)
        if extra is not None and extra.numel() > 0:
            extra = extra.to(torch.long).to(device)
        out = torch.empty(q.shape[0], self.num_q, self.head_dim, device=device, dtype=x.dtype)
        offset = 0
        for req in batch.padded_reqs:
            n = req.extend_len
            if n <= 0:
                continue
            new_loc = out_loc[offset : offset + n]
            self.mtp_k[new_loc] = k[offset : offset + n]
            self.mtp_v[new_loc] = v[offset : offset + n]
            # The committed prefix from the request's page-table row, then this step's earlier
            # drafts from their scratch slots; query row r also sees rows <= r of its own chunk.
            prefix = (
                page_table[req.table_idx, : req.cached_len].to(torch.long)
                if req.cached_len > 0
                else torch.empty(0, dtype=torch.long, device=device)
            )
            parts = [p for p in (prefix, extra) if p is not None and p.numel() > 0]
            parts.append(new_loc.to(torch.long))
            locs = parts[0] if len(parts) == 1 else torch.cat(parts)
            keys = self.mtp_k[locs]
            vals = self.mtp_v[locs]
            if n == 1:
                # One query row per request: score the kv groups directly instead of through sdpa.
                # The head has far more q heads than kv heads, and sdpa's math path repeats each kv
                # head once per q group (hundreds of MB at an 8k context, on top of a nearly full
                # arena). Grouped einsums keep the stored k/v as they are; the grouping matches the
                # contiguous q-head layout ``_project`` produces (and sdpa's ``enable_gqa``).
                group = self.num_q // self.num_kv
                q_row = q[offset : offset + 1].reshape(self.num_kv, group, self.head_dim)
                scores = torch.einsum("ghc,kgc->ghk", q_row, keys) * self.sm_scale
                probs = torch.softmax(scores.float(), dim=-1).to(keys.dtype)
                o = torch.einsum("ghk,kgc->ghc", probs, vals)
                out[offset : offset + 1] = o.reshape(1, self.num_q, self.head_dim)
                offset += n
                continue
            plen = int(locs.shape[0]) - n
            # A single query row sees every stored row, so the causal mask is all-true. Passing one
            # anyway pushes sdpa onto the math backend, which repeats the GQA kv heads and expands
            # the mask to the query-head count -- a few hundred MB of copies per draft step, on top
            # of a nearly full arena. Only a multi-row chunk needs the mask.
            # sdpa wants ``[q_rows, k_rows]``: row r sees the stored prefix plus its own rows <= r.
            vis = None if n == 1 else (
                torch.arange(plen + n, device=device).unsqueeze(0)
                <= torch.arange(n, device=device).unsqueeze(1) + plen
            )
            o = F.scaled_dot_product_attention(
                q[offset : offset + n].transpose(0, 1).unsqueeze(0),
                keys.permute(1, 0, 2).unsqueeze(0),
                vals.permute(1, 0, 2).unsqueeze(0),
                attn_mask=None if vis is None else vis.unsqueeze(0),
                scale=self.sm_scale,
                enable_gqa=True,
            )  # [1, num_q, n, head_dim]
            out[offset : offset + n] = o.squeeze(0).transpose(0, 1).to(x.dtype)
            offset += n
        return self.o_proj.forward(out.view(-1, self.qo_attn_dim) * torch.sigmoid(gate))


class Qwen3_5MTPMoE(BaseOP):
    """The head's MoE: the family's router, softmax gating and sigmoid-gated shared expert, with
    the routed experts gathered from the bank per step instead of an offload cache."""

    def __init__(self, config: ModelConfig, bank: MtpExpertBank, *, prefix: str = ""):
        self._bank = bank
        self._topk = int(config.num_experts_per_tok)
        self._renorm = bool(config.norm_topk_prob)
        # routers stay bf16 whatever the checkpoint quantizes, like the main layers
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False, prefix=f"{prefix}.gate"
        )
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(
            config.hidden_size, 1, has_bias=False, prefix=f"{prefix}.shared_expert_gate"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from freetoken.moe.fused import fused_topk

        num_tokens, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(x)
        shared = self.shared_expert.forward(x)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(x))
        # softmax over ALL experts first, then top-k, then renormalize -- the same router the main
        # layers use, so a draft the head likes is weighted as the target would weight it.
        topk_w, topk_i = fused_topk(
            hidden_states=x, gating_output=router_logits, topk=self._topk, renormalize=self._renorm
        )
        gu, dn = self._bank.gather(topk_i, x.device)  # [T,K,2I,H], [T,K,H,I]
        act = torch.einsum("tkch,th->tkc", gu, x.to(gu.dtype))
        g, u = act.chunk(2, dim=-1)
        y = torch.einsum("tkhi,tki->tkh", dn, F.silu(g) * u)
        routed = (y * topk_w.to(y.dtype).unsqueeze(-1)).sum(1)
        return (routed.to(x.dtype) + shared).view(num_tokens, hidden_dim)


class Qwen3_5MTP(BaseOP):
    """The nextn block: fused input (``enorm`` / ``hnorm`` / ``eh_proj``), one gated-attention +
    MoE decoder layer, then ``shared_head_norm`` feeding the shared ``lm_head``."""

    def __init__(self, config: ModelConfig, *, prefix: str = "mtp"):
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.hidden_size = hidden
        # Plain object (not a BaseOP), so it never lands in the state dict; it allocates nothing
        # until :func:`load_mtp_experts` fills it, so a dense checkpoint pays only for the head.
        self.bank = MtpExpertBank(
            config.num_experts, config.moe_intermediate_size, hidden, torch.bfloat16
        )
        self.enorm = GemmaRMSNorm(hidden, eps=eps)
        self.hnorm = GemmaRMSNorm(hidden, eps=eps)
        # in = the concatenated pair of normed vectors
        self.eh_proj = LinearReplicated(
            2 * hidden, hidden, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.eh_proj"
        )
        self.attn_norm = GemmaRMSNorm(hidden, eps=eps)
        self.self_attn = Qwen3_5MTPAttention(config, prefix=f"{prefix}.self_attn")
        self.post_attention_norm = GemmaRMSNorm(hidden, eps=eps)
        self.mlp = (
            Qwen3_5MTPMoE(config, self.bank, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        # llama.cpp falls back to model.output_norm / model.output when the checkpoint carries no
        # nextn.shared_head_norm / shared_head_head, which is the case for these GGUFs; the
        # loader substitutes the fallbacks for whichever of these two is absent.
        self.shared_head_norm = GemmaRMSNorm(hidden, eps=eps)

        # Context-KV writer state (see write_ctx_kv). ``_embed`` is attached by the engine; the
        # rest is per-request bookkeeping keyed by table_idx.
        self._embed = None
        self._ctx_pending: "dict[int, tuple[int, torch.Tensor]]" = {}
        self._ctx_upto: "dict[int, int]" = {}
        self._ctx_write_off = False

    def attach_embed(self, embed_tokens) -> None:
        """Give the context-KV writer the main model's embedding table (engine wiring)."""
        self._embed = embed_tokens

    def _fuse(
        self, embed_tokens, input_ids: torch.Tensor, hidden: torch.Tensor, mdev: torch.device
    ) -> torch.Tensor:
        """``eh_proj @ cat(enorm(E(x_{t+1})), hnorm(h_t))`` -- the head's whole input front."""
        e = embed_tokens.forward(input_ids).to(mdev)
        joined = torch.cat([self.enorm.forward(e), self.hnorm.forward(hidden.to(mdev))], dim=-1)
        return self.eh_proj.forward(joined)

    def write_rows(
        self,
        tokens_next: torch.Tensor,
        hidden: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Write the context K/V of the fused rows ``(E(x_{t+1}), h_t)`` at rope ``positions``.

        ``tokens_next [n]`` is the token at ``t+1`` per row, ``hidden [n, hidden]`` the main
        model's final-normed hidden at ``t``, ``slots [n]`` physical MTP-pool slots. These K/V
        are what :meth:`forward` computes for the same row, so a rewritten slot and a row the head
        wrote itself agree."""
        attn = self.self_attn
        assert hidden.dim() == 2, f"write_rows expects [n, hidden], got {tuple(hidden.shape)}"
        # Everything the head owns lives on its own device (the CPU-hidden path); run the whole
        # writer there so the pool write and the rope cache stay on one side of the bridge.
        mdev = next(self.parameters()).device
        x = self._fuse(self._embed, tokens_next.to(next(self._embed.parameters()).device), hidden, mdev)
        x = attn.qkv_proj.forward(self.attn_norm.forward(x))
        _, k, v = x.split(attn._qkv_split, dim=-1)
        k = k.contiguous().view(-1, attn.num_kv, attn.head_dim)
        v = v.contiguous().view(-1, attn.num_kv, attn.head_dim)
        k = attn.k_norm.forward(k).reshape(-1, attn.kv_attn_dim)
        # the rope kernel takes a q/k pair; only k is kept here
        dummy = torch.zeros(k.shape[0], attn.qo_attn_dim, device=k.device, dtype=k.dtype)
        _, k = attn.rotary.forward(positions.to(mdev), dummy, k)
        attn.mtp_k[slots.to(mdev)] = k.view(-1, attn.num_kv, attn.head_dim)
        attn.mtp_v[slots.to(mdev)] = v

    def write_ctx_kv(self, batch: Batch, hidden: torch.Tensor) -> None:
        """Maintain the head's context KV from a MAIN-model forward; ``hidden [T, hidden]`` is this
        forward's final-normed hidden per processed row (what llama.cpp publishes as ``t_h_nextn``).

        Slot ``t`` holds ``(E(x_{t+1}), h_t)``, and row ``t`` needs ``x_{t+1}``, which a forward at
        ``t`` does not have yet -- so the last row of every forward is DEFERRED (``_ctx_pending``)
        and written by the next forward of that request from its first input token. Verify and
        rollback re-runs are skipped (``_ctx_write_off``); the scheduler rewrites their slots with
        the true accepted rows via :meth:`write_rows`.
        """
        attn = self.self_attn
        if attn.mtp_k is None or self._embed is None or batch.out_loc is None or self._ctx_write_off:
            return
        page_table = get_global_ctx().page_table
        dev = hidden.device
        mrope = batch.mrope_positions is not None
        attn_pos = batch.get_attn_positions()

        def attn_pos_for(lo: int, hi: int) -> torch.Tensor:
            return attn_pos[:, lo:hi] if mrope else attn_pos[lo:hi]

        def attn_pos_at(p: int, delta: int) -> torch.Tensor:
            # a deferred row sits in the generated-text tail, where all three mrope axes share the
            # coordinate p + delta (matches the scheduler's text branch of _make_mrope_positions)
            if mrope:
                return torch.full((3, 1), p + delta, dtype=torch.int32, device=dev)
            return torch.tensor([p], dtype=torch.int32, device=dev)

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
            pos_start, pos_end = req.cached_len, req.device_len
            pend = self._ctx_pending.pop(T, None)
            if batch.phase == "decode":
                # one eager row per real request: consume this request's pending row at L-1 and
                # stash the current one for the next step
                if pend is not None and pend[0] == pos_start - 1 and pos_start - 1 > self._ctx_upto.get(T, -1):
                    self.write_rows(
                        batch.input_ids[off : off + 1],
                        pend[1],
                        page_table[T, pos_start - 1 : pos_start].to(torch.long),
                        attn_pos_at(pos_start - 1, req.mrope_delta),
                    )
                    self._ctx_upto[T] = pos_start - 1
                self._ctx_pending[T] = (pos_start, hidden[off : off + 1].clone())
                off += n
                continue
            # prefill / extend: rows [pos_start, pos_end-1) shift in-batch, the chunk's last is held
            if pos_start > 0 and pend is not None and pend[0] == pos_start - 1:
                self.write_rows(
                    batch.input_ids[off : off + 1],
                    pend[1],
                    page_table[T, pos_start - 1 : pos_start].to(torch.long),
                    attn_pos_at(pos_start - 1, req.mrope_delta),
                )
            if n > 1:
                self.write_rows(
                    batch.input_ids[off + 1 : off + n],
                    hidden[off : off + n - 1],
                    batch.out_loc[off : off + n - 1].to(torch.long),
                    attn_pos_for(off, off + n - 1),
                )
            upto = pos_end - 2 if n > 1 else pos_start - 1
            if pos_start > 0:
                base = pend[0] if (pend is not None and pend[0] == pos_start - 1) else -1
                upto = max(self._ctx_upto.get(T, base), upto, base)
            self._ctx_upto[T] = upto
            self._ctx_pending[T] = (pos_end - 1, hidden[off + n - 1 : off + n].clone())
            off += n

    def forward(
        self,
        batch: Batch,
        residual: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """One draft step: ``(hidden at t, tokens at t+1)`` -> ``(draft logits, hidden for the next
        draft)``. ``residual`` is the main model's final-normed hidden on the first draft of a
        chain and the previous draft's own output afterwards."""
        # One-shot internal diagnostic: with ``_diag`` set by the scheduler, record each stage's
        # stats so a 0%-acceptance head traces to its degenerate stage in one run.
        diag = getattr(self, "_diag", None)
        if diag is not None:
            self._diag = None

        def _rec(name: str, t: torch.Tensor) -> None:
            if diag is None:
                return
            if not torch.isfinite(t).all():
                diag[name] = "NONFINITE"
            else:
                diag[name] = (
                    f"min={float(t.min()):.4g} max={float(t.max()):.4g} "
                    f"mean={float(t.mean()):.4g} std={float(t.std()):.4g}"
                )

        # Device bridge for --mtp-device cpu: the head, its rope cache and its KV pool are on CPU,
        # the shared embed/lm_head stay on the GPU. On the gpu path every .to() is a no-op.
        mdev = next(self.parameters()).device
        input_ids = input_ids.to(next(embed_tokens.parameters()).device)

        cur = inpSA = self._fuse(embed_tokens, input_ids, residual, mdev)  # noqa: N806
        _rec("fused", cur)
        cur = self.attn_norm.forward(cur)
        cur = self.self_attn.forward(cur, batch) + inpSA
        _rec("attn_out", cur)
        ffn_in = self.post_attention_norm.forward(cur)
        cur = self.mlp.forward(ffn_in) + cur
        _rec("mlp_out", cur)
        sample = self.shared_head_norm.forward(cur)
        _rec("head_in", sample)
        logits = lm_head.forward(sample.to(next(lm_head.parameters()).device))
        _rec("logits", logits)
        # llama.cpp chains the POST-shared_head_norm hidden (res->t_h_nextn is assigned after
        # build_norm and the LM head consumes the same tensor), so the next draft's hnorm sees
        # the normed stream exactly as trained -- not the raw pre-norm sum.
        return logits, sample


def load_mtp_experts(
    model_path: str, bank: MtpExpertBank, device: "torch.device | None" = None, quant: str = "none"
) -> int:
    """Fill the head's routed experts into ``bank`` from the GGUF's last block.

    ``blk.N`` is the only block the GGUF gives the head, and its experts are already ggml-packed
    (unsloth ships Q5_K / Q6_K there), so they are dequantized once at load into the stacked
    ``gate_up [E, 2I, H]`` / ``down [E, H, I]`` layout the bank wants, then stored under
    ``quant``. Dequantizing is loss-neutral against the file and keeps the draft off the main
    model's expert-streaming path; ``quant`` then only trades bank bytes for gather-time work.
    Returns the pinned host bytes the engine reserves (0 when the bank went GPU-resident)."""
    index = gguf_tensor_index(model_path)
    n_layer = next(
        (
            int(name.split(".")[1])
            for name in index
            if name.startswith("blk.") and name.endswith(".nextn.eh_proj.weight")
        ),
        None,
    )
    if n_layer is None:
        raise ValueError(f"{model_path}: no nextn (MTP) block to build a draft head from")
    blk = f"blk.{n_layer}"
    gate = index.get(f"{blk}.ffn_gate_exps.weight")
    merged = index.get(f"{blk}.ffn_gate_up_exps.weight")
    if gate is None and merged is None:  # a dense nextn block: its MLP is plain, nothing to gather
        return 0
    up = index.get(f"{blk}.ffn_up_exps.weight")
    down = index[f"{blk}.ffn_down_exps.weight"]

    def dense(info) -> torch.Tensor:
        """The expert stack of ``info`` as bf16 ``[E, rows, K]``; quantized types are walked
        expert by expert so the fp32 scratch of a 256-expert stack stays one expert wide."""
        rows = _expert_rows(info)  # [E, rows, row_bytes] packed, or already the float dtype
        if not is_quantized(int(info.ggml_type)):
            return rows.to(torch.bfloat16)
        qtype = GGMLQuantizationType(int(info.ggml_type))
        k, width = int(info.ne[0]), int(info.ne[1])
        return torch.stack(
            [
                torch.from_numpy(quants.dequantize(np.asarray(rows[e]), qtype)).reshape(width, k)
                for e in range(rows.shape[0])
            ]
        )

    if merged is not None:
        gate_up = dense(merged)  # the file already stacks gate rows then up rows per expert
    else:
        assert up is not None, f"{blk}: ffn_gate_exps without ffn_up_exps"
        gate_up = torch.cat([dense(gate), dense(up)], dim=1)  # [E, 2I, H] (gate rows, then up)
    bank.load(gate_up, dense(down), device=device, quant=quant)
    nbytes = bank.stored_bytes
    logger.info_rank0(
        f"MTP expert bank ({blk}): {nbytes / (1 << 30):.2f} GiB ({quant} storage),"
        f" {'GPU-resident' if bank.gpu_bytes else 'pinned host RAM'}"
    )
    return 0 if bank.gpu_bytes else nbytes


__all__ = ["MtpExpertBank", "Qwen3_5MTP", "Qwen3_5MTPAttention", "Qwen3_5MTPMoE", "load_mtp_experts"]
