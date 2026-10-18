from __future__ import annotations

import os
import time

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    div_ceil,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .mm import cut_image_spans, plan_mm_batch
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

# Budget-slider damping (see _maybe_queue_budget_slide): a proposal only lands if it moves an
# axis by at least this relative margin, and applies are spaced by at least this interval.
_SLIDE_DEADBAND = 0.15
_SLIDE_MIN_INTERVAL_S = 30.0


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class MtpBackend:
    """Device/placement backend for one MTP speculative-decoding step.

    The scheduler owns the step's orchestration and its spec-sampling acceptance math (both are
    identical regardless of device). This backend holds only the handful of engine-facing forward
    operations the step performs, so a GPU and a CPU variant can place that compute differently
    while sharing everything else. ``draft_device`` is where the step's small helper tensors
    (positions / page-table rows / attention metadata) are built; the head + verify compute follows
    it. Subclasses override the forward hooks; ``verify_logits`` returns the per-row residual and
    the matching ``[rows, vocab]`` logits that the shared acceptance consumes."""

    def __init__(self, sch: "Scheduler") -> None:
        self.sch = sch

    def draft_device(self) -> torch.device:
        return self.sch.device

    def anchor(self, batch, sample_args):
        raise NotImplementedError

    def draft_forward(self, db, res_j, fed_gpu):
        raise NotImplementedError

    def verify_logits(self, batch, vbatch):
        raise NotImplementedError

    def rerun(self, batch):
        raise NotImplementedError


class GpuMtpBackend(MtpBackend):
    """Draft head + verify run on the GPU through the engine's CUDA forward paths."""

    def anchor(self, batch, sample_args):
        eng = self.sch.engine
        out = eng.forward_batch(batch, sample_args, eager_only=True)
        return out, eng.model.model._last_residual

    def draft_forward(self, db, res_j, fed_gpu):
        return self.sch.engine.mtp_forward(db, res_j, fed_gpu)

    def verify_logits(self, batch, vbatch):
        eng = self.sch.engine
        eng.forward_logits(vbatch)
        vres = eng.model.model._last_residual  # [B*(K+1), hc*hidden]
        with eng.ctx.forward_batch(batch):
            v_logits_all = eng.model.lm_head.forward(
                eng.model.model.hyper_connection_mixer.mix(vres)[0]
            )  # [B*(K+1), vocab]
        return vres, v_logits_all

    def rerun(self, batch):
        return self.sch.engine.forward_logits(batch)


class CpuMtpBackend(MtpBackend):
    """Hide the MTP compute on the CPU behind the GPU's main-model work.

    The ~5GB bf16 head-expert bank stays pinned in host RAM (no PCIe round-trip), and CPU compute
    (~0.85x the GPU) runs concurrently with the GPU forward so the speculative step overlaps the
    main-model work instead of competing with it for PCIe bandwidth -- pure-GPU MTP lost decode
    throughput to exactly that contention. Only the device placement differs from the GPU path."""

    def draft_device(self) -> torch.device:
        return torch.device("cpu")

    def anchor(self, batch, sample_args):
        eng = self.sch.engine
        out = eng.forward_batch(batch, sample_args, eager_only=True)
        return out, eng.model.model._last_residual

    def draft_forward(self, db, res_j, fed_gpu):
        return self.sch.engine.mtp_forward(db, res_j, fed_gpu)

    def verify_logits(self, batch, vbatch):
        eng = self.sch.engine
        eng.forward_logits(vbatch)
        vres = eng.model.model._last_residual
        with eng.ctx.forward_batch(batch):
            v_logits_all = eng.model.lm_head.forward(
                eng.model.model.hyper_connection_mixer.mix(vres)[0]
            )
        return vres, v_logits_all

    def rerun(self, batch):
        return self.sch.engine.forward_logits(batch)


def _make_mtp_backend(sch: "Scheduler") -> MtpBackend:
    mode = str(getattr(sch.config, "mtp_device", "gpu") or "gpu").strip().lower()
    return CpuMtpBackend(sch) if mode == "cpu" else GpuMtpBackend(sch)


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            kv_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self._bidirectional_mm = any(getattr(g, "bidirectional_mm_blocks", False) for g in config.model_config.attention_groups)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            encoder_cache=self.engine.encoder_cache,
            keep_images_whole=self._bidirectional_mm,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self._model_is_mrope = config.model_config.model_is_mrope
        self._warned_cut_image = False
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Runtime KV<->MoE rebalancer: samples KV occupancy per batch and queues a rebuild
        # through the existing pending-rebuild path at the idle safe point.
        self._budget_slider = self._build_budget_slider()
        # High-water mark of admitted prompt footprints (tokens): the shrink-only demand floor,
        # so an idle shrink never slides the pool below what a recent big turn asks back.
        self._kv_demand_tokens = 0
        self._kv_page_tokens = 1
        # Number of accepted KV/MoE rebalances since startup (geometry epochs).
        self._slide_count = 0

        # Memory-growth probe: time-gated log of GPU mem + pool sizes, to catch which
        # resource (KV pages / MoE slots / mamba slots / allocator) ratchets up as free
        # VRAM falls. FREETOKEN_MEM_LOG_INTERVAL (s) sets cadence; <=0 disables.
        try:
            self._mem_log_interval = float(os.environ.get("FREETOKEN_MEM_LOG_INTERVAL", "5"))
        except ValueError:
            self._mem_log_interval = 5.0
        self._mem_log_last = 0.0
        # Prefill/decode interleaving: a long chunked prompt runs several consecutive
        # chunk-prefill forwards (strict prefill-priority above), which starves in-flight
        # decode. Every ``_interleave_every`` consecutive prefill chunks, hand one forward to
        # the decode manager so running requests keep emitting tokens. Counts consecutive
        # prefill batches; resets on any decode batch. No-op for short prompts (< K chunks).
        self._interleave_every = max(1, int(getattr(config, "prefill_interleave_every", 4)))
        self._prefill_run_len = 0
        # Hot-window cap on the floor (bounded-read backends): unset keeps the full-footprint floor.
        self._hot_window_tokens = max(0, int(os.getenv("FREETOKEN_HOT_WINDOW_TOKENS", "0")))

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
        slide: bool = False,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        rebuild_fn = (
            self.engine.rebuild_runtime_cache_slide
            if slide
            else self.engine.rebuild_runtime_cache
        )
        rebuild_fn(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Give the slider a turn before the safe-point check, so a rebalance it queues can
        # land on THIS iteration's idle window (not only a future one).
        self._maybe_queue_budget_slide()

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()
        self._drain_deferred()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        self._maybe_queue_budget_slide()
        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()
        self._drain_deferred()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, forward_output = last_data[0].batch, last_data[1]
        next_tokens_cpu = forward_output.next_tokens_cpu
        copy_done = forward_output.copy_done_event
        # committed token list for the single MTP request of the step, None on the plain path
        mtp_tokens = forward_output.mtp_tokens
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue

                # An MTP step commits m+2 tokens per request in one forward; the plain path
                # commits one. ``mtp_tokens`` is a per-request list of lists when set, indexed
                # by ``i``; ``next_tokens_cpu[i]`` covers the plain single-token path. Either
                # way every token streams through the same per-token EOS/stop/length checks,
                # stopping at the first terminal one.
                tokens = (
                    list(mtp_tokens[i])
                    if mtp_tokens is not None
                    else [int(next_tokens_cpu[i].item())]
                )
                finished = False
                for next_token in tokens:
                    req.append_host(torch.tensor([next_token], dtype=req.input_ids.dtype))
                    # EOS / stop-string -> "stop", output budget exhausted -> "length";
                    # EOS and stop strings win over length. The check runs AFTER each append on
                    # the true delivered count, so an MTP commit whose tail crosses the output
                    # budget mid-step still ships its accepted rows before finishing.
                    hit_length = req.input_ids.numel() >= req.max_device_len
                    hit_eos = (
                        not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                    )
                    matched_stop = (
                        self._match_stop_str(req)
                        if not hit_eos and req.sampling_params.stop_strs
                        else None
                    )
                    finished = hit_length or hit_eos or matched_stop is not None
                    finish_reason = (
                        ("stop" if (hit_eos or matched_stop is not None) else "length")
                        if finished
                        else None
                    )
                    if (
                        next_token == self.toolcall_anchor_id
                        and req.toolcall_anchor_len is None
                        and not finished
                    ):
                        req.toolcall_anchor_len = req.input_ids.numel()
                    reply.append(
                        DetokenizeMsg(
                            uid=req.uid,
                            next_token=next_token,
                            finished=finished,
                            finish_reason=finish_reason,
                            matched_stop=matched_stop,
                            stop_strs=req.sampling_params.stop_strs or None,
                        )
                    )
                    if finished:
                        break

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    if req.mtp_steps > 0:
                        # head-sanity line: first_match ~ 0/steps -> the draft head is
                        # systematically wrong (wiring/formula or missing KV history);
                        # > 0 but accepted ~ 0 -> the head is weak (drafts diverge later).
                        logger.info_rank0(
                            "MTP diag uid=%s steps=%d drafts=%d accepted=%d "
                            "first_match=%d full_accept=%d",
                            req.uid, req.mtp_steps, req.mtp_drafts_total,
                            req.mtp_accepted_total, req.mtp_first_match, req.mtp_full_accept,
                        )
                        # Same counters onto the terminal reply message; the tokenizer worker
                        # passes them through and StatsTracker folds them into the /v1/stats
                        # `mtp` section once per finished request (stdout parsing stays the
                        # optional fallback). reply[-1] is this req's last appended message.
                        last = reply[-1]
                        last.mtp_steps = req.mtp_steps
                        last.mtp_drafts = req.mtp_drafts_total
                        last.mtp_accepted = req.mtp_accepted_total
                        last.mtp_first_match = req.mtp_first_match
                        last.mtp_full_accept = req.mtp_full_accept
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        slider = getattr(self, "_budget_slider", None)
        if slider is not None:
            slider.observe(used, total)
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)
        self._mem_log_tick(used, total, mamba_slots, swa_tokens)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _mem_log_tick(self, kv_used: int, kv_total: int, mamba_slots, swa_tokens) -> None:
        """Time-gated memory-growth probe: torch mem counters + pool sizes, so we can see
        which resource (KV pages / MoE slots / mamba slots / allocator peak) ratchets up as
        free VRAM falls. Cadence via FREETOKEN_MEM_LOG_INTERVAL (s, default 5); <=0 disables.
        Diagnostic only -- no device sync, no behavior change."""
        if self._mem_log_interval <= 0:
            return
        now = time.monotonic()
        if now - self._mem_log_last < self._mem_log_interval:
            return
        self._mem_log_last = now
        if self.device.type != "cuda":
            return
        c = torch.cuda
        total = c.get_device_properties(self.device).total_memory
        reserved = c.memory_reserved(self.device)
        allocated = c.memory_allocated(self.device)
        peak = c.max_memory_reserved(self.device)
        peak_alloc = c.max_memory_allocated(self.device)
        demand = getattr(self, "_kv_demand_tokens", 0)
        m = "n/a" if not mamba_slots else f"{mamba_slots[0]}/{mamba_slots[1]}"
        swa = "n/a" if not swa_tokens else f"{swa_tokens[0]}/{swa_tokens[1]}"
        geo = {}
        try:
            geo = self._current_cache_geometry()
        except Exception:  # noqa: BLE001 - probe must never break the serving loop
            pass
        if geo.get("total_experts") is None:
            sgeo = getattr(getattr(self, "_budget_slider", None), "_geometry", {})
            if isinstance(sgeo, dict):
                geo["total_experts"] = sgeo.get("total_experts")
        GiB = 1024.0 ** 3
        logger.info_rank0(
            "mem-probe: free=%.2fGiB alloc=%.2fGiB(res %.2f, peak %.2f / peakAlloc %.2f) | "
            "KV=%d/%d pages (demand=%d tok) MoE=%s/%s mamba=%s swa=%s slides=%d",
            (total - reserved) / GiB, allocated / GiB,
            reserved / GiB, peak / GiB, peak_alloc / GiB,
            kv_used, kv_total, demand,
            geo.get("moe_cache_size"), geo.get("total_experts"), m, swa,
            getattr(self, "_slide_count", 0),
        )

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            if msg.mm_items and self.engine.encoder_cache is None:
                # no encoder runtime: fail loudly instead of decoding unexpanded placeholders
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            error="image input is not supported by this server",
                        )
                    ]
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            slider = getattr(self, "_budget_slider", None)
            if max_output_len <= 0 and slider is not None:
                # Grow-before-reject: a too-long prompt is concrete KV-demand evidence that never
                # becomes a batch, so observe() would never see it. While this msg is processed the
                # scheduler is idle by construction -> expand KV inside the byte envelope now and
                # re-check admission against the new ceiling.
                slider.observe_need(input_len)
                self._maybe_queue_budget_slide(force=True, allow_small=True)
                if self._pending_rebuild is not None and not (
                    self.prefill_manager.runnable or self.decode_manager.runnable
                ):
                    self._execute_pending_rebuild()
                    max_seq_len = self.engine.max_seq_len
                    max_output_len = max_seq_len - input_len
                if max_output_len <= 0 and self._grow_to_cover(input_len):
                    max_seq_len = self.engine.max_seq_len
                    max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                # `max_seq_len` is the LIVE (slide-resized) ceiling and can sit transiently below
                # the configured maximum right after a KV<->MoE rebalance. A prompt that still fits
                # the configured ceiling must NOT be dropped for that transient dip -- grow KV to
                # cover it (hard admission grow already ran above) and defer the request until the
                # resize lands, then admit on the next drain. Only a prompt over the CONFIGURED
                # ceiling is a real reject.
                hard_max = self._configured_max_seq_len()
                if input_len <= hard_max:
                    dq = getattr(self, "_deferred_msgs", None)
                    if dq is None:
                        from collections import deque

                        dq = self._deferred_msgs = deque()
                    dq.append(msg)
                    logger.debug_rank0(
                        f"Deferring request {msg.uid}: {input_len} tok fits configured {hard_max} "
                        f"but not current KV {max_seq_len}; waiting for grow."
                    )
                    return
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {hard_max} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # Demand floor for the rebalancer: the admitted prompt's full footprint, capped at
            # the pool size so it can only prevent a shrink, never force a grow.
            self._kv_demand_tokens = max(
                getattr(self, "_kv_demand_tokens", 0),
                min(input_len + msg.sampling_params.max_tokens, max_seq_len),
            )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if (
                req_to_free is not None
                and req_to_free.mm_items
                and self.engine.encoder_cache is not None
            ):
                # drop the aborted request's claims; entries it held alone die here
                self.engine.encoder_cache.release(
                    msg.uid, [item.hash for item in req_to_free.mm_items]
                )
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once
            elif getattr(req, "mm_gdn", None) is not None:
                pool.restore_state(req.linear_slot_idx, req.mm_gdn)
                req.mm_gdn = None

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            # Slider-originated slides ride the preserve-residency clone; manual and
            # rollback rebuilds keep the official full-reset path.
            self.rebuild_cache(**requested, slide=msg.request_id == "budget-slide")
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        # Handoff clamp recompute: the KV demand high-water was measured against the OLD
        # context cap, and the reachable-row clamp every QSA forward derives from
        # engine.max_seq_len was re-derived in _refresh_seq_state. Re-price the demand floor
        # against the geometry actually serving now (a shrink caps it lower; a grow leaves it),
        # so the next slider tick cannot demand pages past the new context and oscillate.
        self._kv_demand_tokens = min(
            getattr(self, "_kv_demand_tokens", 0), self.engine.max_seq_len
        )
        self._slide_count = getattr(self, "_slide_count", 0) + 1
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _build_budget_slider(self):
        """Wire the runtime KV<->MoE rebalancer, or None when it is inert.

        Off only with budget_slide=False. Declared --num-pages / --num-tokens are starting
        points, not a freeze: the startup fit already clamped them into the measured byte
        budget, and the slider keeps trading within that same envelope as the load profile
        drifts. Geometry comes from the pool class's kv_cost + the offload cache's own counts,
        so every model family runs the same arithmetic on its own numbers; no offload cache
        -> the KV-only branch (cur_moe None)."""
        config = self.config
        engine = self.engine
        if not getattr(config, "budget_slide", True):
            return None
        try:
            cache_per_page, fixed_kv, page_tokens, min_reserve = engine._pool_cls.kv_cost(config)
        except Exception as e:  # noqa: BLE001 — owned pools may need args; slider is optional
            logger.info_rank0(f"budget slider disabled (kv_cost unavailable: {e!r})")
            return None
        page_tokens = max(1, int(page_tokens))
        self._kv_page_tokens = page_tokens
        # The slider's HARD boundary is the shared cache arena (the one flat block both
        # pools carve, fixed at cold start) -- not the budget ceiling, whose headroom
        # above the arena belongs to graph capture / activations.
        arena = getattr(engine, "_cache_arena", None)
        arena_bytes = int(arena.numel()) if arena is not None else None
        mc = engine.moe_offload_cache
        if mc is None:
            geometry: dict = dict(
                cache_per_page=cache_per_page, per_expert_bytes=1, num_experts=1,
                total_experts=1, max_slots=1, prefill_overlap=False,
                arena_bytes=arena_bytes, kv_fixed_bytes=fixed_kv,
            )
        else:
            from freetoken.engine.cache_budget import expert_bytes_per_slot
            from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE

            total_experts = mc.num_layers * mc.num_experts
            geometry = dict(
                cache_per_page=cache_per_page,
                per_expert_bytes=expert_bytes_per_slot(mc.bank_sources),
                num_experts=mc.num_experts,
                total_experts=total_experts,
                max_slots=(MARLIN_MAX_CACHE_SIZE if mc.quant_format == "nvfp4_marlin" else total_experts),
                prefill_overlap=bool(mc.prefill_overlap),
                arena_bytes=arena_bytes, kv_fixed_bytes=fixed_kv,
            )
        # Grow KV to demand but never past the declared context; the byte envelope still bounds
        # the move (the MoE floor caps how many slots can be given back).
        geometry["max_kv_pages"] = int(
            config.num_page_override or -(-config.max_seq_len // page_tokens)
        )
        from freetoken.engine.budget_slide import BudgetSlider

        kv_floor = max(config.kv_reserve_tokens, min_reserve)
        return BudgetSlider(
            min_kv_pages=max(2, -(-kv_floor // page_tokens)), page_tokens=page_tokens, **geometry
        )

    def _maybe_queue_budget_slide(self, force: bool = False, allow_small: bool = False) -> None:
        """One rebalancer tick, run right before each idle safe-point check.

        Queues through the same pending-rebuild slot the HTTP endpoint uses (latest proposal
        wins), so the resize lands at the safe point with the engine's reject/rollback
        protections. ``force`` skips the interval debounce (admission-driven rebalances);
        ``allow_small`` also skips the min-step size gate."""
        slider = getattr(self, "_budget_slider", None)
        if slider is None or getattr(self, "_pending_rebuild", None) is not None:
            return
        geo = self._current_cache_geometry()
        page_tokens = getattr(self, "_kv_page_tokens", 1)
        demand = getattr(self, "_kv_demand_tokens", 0)
        # Relax the shrink-only floor halfway toward what the pool actually holds each idle tick
        # (never below it), so a big old turn does not pin KV and starve the MoE cache later.
        if not force:
            cm = getattr(self, "cache_manager", None)
            if cm is not None:
                live_tokens = cm.page_usage()[0] * page_tokens
                if demand > live_tokens:
                    demand = live_tokens + (demand - live_tokens) // 2
                    self._kv_demand_tokens = demand
        hot_cap = getattr(self, "_hot_window_tokens", 0)
        demand_tokens = min(demand, hot_cap) if hot_cap > 0 else demand
        targets = slider.propose(
            geo["num_pages"], geo["moe_cache_size"] or None,
            force=force, demand_pages=-(-demand_tokens // page_tokens), allow_small=allow_small,
        )
        if targets is None:
            return
        # Damp the follow-up chatter: the slider's demand signal renormalizes against the
        # just-resized pool, so without hysteresis every phase flip (short-burst concurrency
        # <-> lone long-context request) queues another rebuild within seconds. Each apply
        # costs a graph recapture, so only follow moves big enough to pay for it: both axes
        # within a relative dead band, or inside the minimum interval, is a no-op. Admission
        # grows bypass both — covering a concrete prompt is a hard bound, not a preference.
        if not force:
            last = getattr(self, "_slide_last_queued", 0.0)
            now = time.monotonic()
            if now - last < _SLIDE_MIN_INTERVAL_S:
                return

            def within(a, b):
                a, b = int(a or 0), int(b or 0)
                return abs(a - b) <= _SLIDE_DEADBAND * max(a, b, 1)

            if within(geo["num_pages"], targets.get("num_pages")) and within(
                geo["moe_cache_size"], targets.get("moe_cache_size")
            ):
                return
            self._slide_last_queued = now
        logger.info_rank0(f"budget slide queued: {targets}")
        self._pending_rebuild = CacheRebuildBackendMsg(
            request_id="budget-slide", mode="if_idle", **targets
        )


    def _configured_max_seq_len(self) -> int:
        """The CONFIGURED ceiling (not the live slide-resized one): the larger of the two bounds a
        prompt may legitimately reach -- min(KV pool at full pages, model rotary max). Used to
        decide whether a too-long-for-now prompt is a real overflow or just a transient KV dip."""
        cfg = self.config
        try:
            _cpp, _fixed, page_tokens, _mr = self.engine._pool_cls.kv_cost(cfg)
        except Exception:  # noqa: BLE001 — same optional-pool semantics as the builder
            return self.engine.max_seq_len
        page_tokens = max(1, int(page_tokens))
        model_max = int(cfg.max_seq_len)
        pages = cfg.num_page_override
        if pages:
            return min(int(pages) * page_tokens, model_max)
        return model_max

    def _drain_deferred(self) -> None:
        """Re-run admission on requests deferred during a transient KV dip, after the grow has had
        a chance to land. Each pass either admits (KV now covers it), re-defers (still short), or
        rejects (over the configured ceiling). One pass per loop iteration, so this never spins."""
        dq = getattr(self, "_deferred_msgs", None)
        if not dq:
            return
        for _ in range(len(dq)):
            self._process_one_msg(dq.popleft())

    def _grow_to_cover(self, need_tokens: int) -> bool:
        """Ceil-rounded follow-up to grow-before-reject. propose_targets rounds a move to
        the NEAREST joint lcm step, which can land a few pages below a concrete prompt's
        footprint (observed: a 77610-token prompt against a 77312-token post-grow ceiling).
        A hard admission demand must round UP, so price the residual against whole chunks
        here and queue the exact target through the same rebuild slot."""
        import math

        try:
            cpp, _fixed, page_tokens, _mr = self.engine._pool_cls.kv_cost(self.config)
        except Exception:  # noqa: BLE001 — same optional-pool semantics as the builder
            return False
        cpp = max(1, int(cpp))
        page_tokens = max(1, int(page_tokens))
        geo = self._current_cache_geometry()
        cur_pages = int(geo.get("num_pages") or 0)
        max_pages = int(
            self.config.num_page_override or -(-self.config.max_seq_len // page_tokens)
        )
        need_pages = min(-(-need_tokens // page_tokens), max_pages)
        if cur_pages <= 0 or cur_pages >= need_pages:
            return False
        mc = self.engine.moe_offload_cache
        if mc is None:
            targets: dict = {"num_pages": need_pages}
        else:
            from freetoken.engine.cache_budget import expert_bytes_per_slot
            from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE

            peb = max(1, int(expert_bytes_per_slot(mc.bank_sources)))
            total_experts = mc.num_layers * mc.num_experts
            hi = min(
                total_experts,
                MARLIN_MAX_CACHE_SIZE if mc.quant_format == "nvfp4_marlin" else total_experts,
            )
            floor = min(
                hi, max(mc.num_experts, 2 * mc.num_experts if mc.prefill_overlap else mc.num_experts)
            )
            chunk_pages = math.lcm(cpp, peb) // cpp
            steps = max(0, -(-(need_pages - cur_pages) // chunk_pages))
            new_pages = min(cur_pages + steps * chunk_pages, max_pages)
            cur_moe = int(geo.get("moe_cache_size") or 0)
            new_moe = max(floor, min(hi, cur_moe - (new_pages - cur_pages) * cpp // peb))
            targets = {"num_pages": int(new_pages), "moe_cache_size": int(new_moe)}
        logger.info_rank0(f"budget slide admission grow: {targets}")
        self._pending_rebuild = CacheRebuildBackendMsg(
            request_id="budget-slide", mode="if_idle", **targets
        )
        if not (self.prefill_manager.runnable or self.decode_manager.runnable):
            self._execute_pending_rebuild()
        return True

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        if self._model_is_mrope:
            batch.mrope_positions = _make_mrope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Plan the chunk's encoder jobs, gather rows and scatter rows over the batch; the engine runs them before the LM forward."""
        jobs, plan, rows, block_ends = plan_mm_batch(batch.padded_reqs, self.engine.encoder_cache)
        if plan:
            batch.mm_encoder_jobs = jobs
            batch.mm_gather_plan = plan
            batch.mm_rows = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
            batch.mm_block_ends = torch.tensor(block_ends, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        if self._bidirectional_mm and not self._warned_cut_image and (cut := cut_image_spans(batch.padded_reqs)):
            # only a bidirectional image span loses context when cut, and only an image longer than the chunk still gets cut
            lo, hi = cut[0]
            self._warned_cut_image = True
            logger.warning_rank0(
                f"an image of {hi - lo} tokens does not fit one prefill chunk (--max-extend-tokens {self.prefill_budget}, or the sliding-window pool's share of it): its earlier rows attend within the first part only"
            )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # Prefill-first by default, but interleave a decode step every K consecutive prefill
        # chunks so a long chunked prompt does not starve in-flight decode. ``schedule_next_
        # batch`` returns None when its manager has nothing runnable, so the fallback side of
        # ``or`` still fires when the preferred side is empty.
        decode_first = (
            getattr(self, "_prefill_run_len", 0) >= getattr(self, "_interleave_every", 4)
            and self.decode_manager.runnable
        )
        if decode_first:
            batch = (
                self.decode_manager.schedule_next_batch()
                or self.prefill_manager.schedule_next_batch(self.prefill_budget)
            )
        else:
            batch = (
                self.prefill_manager.schedule_next_batch(self.prefill_budget)
                or self.decode_manager.schedule_next_batch()
            )
        if batch is None:
            return None
        self._prefill_run_len = (getattr(self, "_prefill_run_len", 0) + 1) if batch.is_prefill else 0
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        if getattr(self, "_mtp_backend", None) is None:
            self._mtp_backend = _make_mtp_backend(self)
        if self._mtp_active(batch) and self._mtp_scratch_ready():
            forward_output = self._mtp_speculate(forward_input)
        else:
            self._note_mtp_skip(batch)
            forward_output = self.engine.forward_batch(batch, sample_args)
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _mtp_active(self, batch: Batch) -> bool:
        """MTP self-speculation applies to any not-aborted decode step where every request in the
        batch has room to grow. Works for greedy AND sampling (temperature/top-k/top-p) requests:
        the verify accepts drafts by spec-sampling (rejection) so the emitted sequence follows the
        request's exact sampling distribution. The 1-layer head chains drafts per request; verify
        is a single batched forward across requests, drafts and rollbacks loop per request."""
        if not (getattr(self.config, "enable_mtp", False) and batch.is_decode):
            return False
        if not batch.reqs:
            return False
        for req in batch.reqs:
            if req.aborted or not req.can_decode:
                return False
        return True

    def _mtp_scratch_ready(self) -> bool:
        """The verify step snapshots the request's GDN state into one scratch slot so a
        rejected draft can be rolled back. The slot comes from the same free list as
        live/ping-pong/tree slots; under concurrency the plain alloc can exhaust it, so free
        an LRU tree snapshot first (the standard ensure_mamba_slots mechanism). If the pool
        is still dry the step falls back to the plain graph path instead of dying."""
        gdn = self.engine.linear_state_pool
        if gdn is None:
            return True
        if gdn.num_free_slots < 1:
            self.cache_manager.ensure_mamba_slots(1)
        if gdn.num_free_slots < 1:
            if not getattr(self, "_mtp_pool_warned", False):
                self._mtp_pool_warned = True
                logger.warning_rank0(
                    f"MTP: linear-state pool exhausted (free={gdn.num_free_slots},"
                    f" slots={gdn.num_slots}) even after evicting an LRU tree snapshot"
                    " -- stepping back to plain decode"
                )
            return False
        return True

    def _note_mtp_skip(self, batch: Batch) -> None:
        """--mtp-draft-len is set but this step took the plain path: log the reason once per
        category so an apparently 'inactive' MTP is diagnosable instead of silent."""
        if not (getattr(self.config, "enable_mtp", False) and batch.is_decode):
            return
        reqs = batch.reqs
        if not reqs:
            return
        # Hookless graph-replayed decode: no write_ctx_kv runs, so every request's MTP context
        # rows go stale -- flag them for the full-prefix rebuild on its next eligible step.
        reason = None
        for r in reqs:
            r._mtp_ctx_dirty = True
            if r.aborted:
                continue
            if reason is None and not r.can_decode:
                reason = "request has no room to grow (KV/output budget)"
        if reason is None:
            return
        seen = getattr(self, "_mtp_skip_logged", None)
        if seen is None:
            seen = self._mtp_skip_logged = set()
        if reason not in seen:
            seen.add(reason)
            logger.info_rank0(f"MTP step skipped: {reason}")

    def _mtp_draft_batch(
        self, req: Req, device: torch.device, pos: int, cached_len: int, out_loc: int
    ) -> Batch:
        """A 1-token batch for the MTP head at REAL position ``pos``: the query attends to the
        committed prefix (the request's page-table row [0, cached_len) -- the head's context KV
        written by write_ctx_kv) plus prior-draft entries (``mtp_extra_locs``, set by the
        caller), and writes its own MTP-space K/V to reserved scratch slot ``out_loc`` (beyond
        the page-allocated pool region, engine._mtp_scratch_base)."""
        probe = type(
            "MtpProbe",
            (),
            {"extend_len": 1, "cached_len": cached_len, "table_idx": req.table_idx},
        )()
        db = Batch(reqs=[probe], phase="decode")
        db.padded_reqs = [probe]
        db.out_loc = torch.tensor([out_loc], dtype=torch.long, device=device)
        db.positions = torch.tensor([pos], dtype=torch.int32, device=device)
        if self._model_is_mrope:
            # the head's attention forward reads get_attn_positions(); an mrope rotary needs
            # the [3, 1] (t/h/w) rows, and the draft sits in the generated-text tail where all
            # three axes share ``pos + delta`` (text delta is 0).
            db.mrope_positions = torch.full(
                (3, 1), pos + req.mrope_delta, dtype=torch.int32, device=device
            )
        db.attn_metadata = None  # the head uses F.sdpa directly, not the attention backend
        return db

    def _mtp_extend_batch(
        self, reqs: list, ranges: list, allocate: bool
    ):
        """Prepare a multi-token (prefill) extend over positions ``[start, end)`` for each
        request in ``reqs`` (paired by index with ``ranges``): the main model's GDN uses the
        chunk path for these, so phase must be prefill. ``allocate=True`` allocates fresh KV
        for each range (the verify); ``False`` reuses slots already in the page table (the
        rollback re-run must not double-allocate). One Batch covers all requests so the verify
        forward is a single batched call."""
        # Verify / rollback batches always run through the GPU main-model forward, so their
        # metadata must live on the engine device even when the draft chain is CPU-resident.
        dev = self.device
        for req, (pos_start, pos_end) in zip(reqs, ranges):
            req.cached_len = pos_start
            req.device_len = pos_end
        vb = Batch(reqs=list(reqs), phase="prefill")
        self.engine.graph_runner.pad_batch(vb)
        if allocate:
            self.cache_manager.allocate_paged(list(reqs))
        vb.positions = _make_positions(vb, dev)
        if self._model_is_mrope:
            vb.mrope_positions = _make_mrope_positions(vb, dev)
        input_mapping = _make_input_tuple(vb, dev)
        vb.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            vb.fla_metadata = build_fla_metadata(vb, dev)
        self.engine.attn_backend.prepare_metadata(vb)
        return vb, input_mapping

    def _mtp_speculate(self, forward_input: ForwardInput) -> "ForwardOutput":
        """One MTP speculative step covering every greedy decode request in the batch. The main
        model emits one anchor per request in a single batched eager forward (which also stashes
        each request's pre-mixer residual); the 1-layer MTP head chains K cheap drafts per
        request; ONE verify forward through the full model confirms every request's longest
        greedy-consistent prefix in a single batched call. Per request, if drafts are rejected
        its GDN slot rolls back to a scratch snapshot and its committed prefix re-runs so the
        recurrent state stays exact. Each request commits anchor + accepted drafts + one bonus
        token (the main model's own prediction at the cut)."""
        from freetoken.engine import ForwardOutput
        # Match the normal decode path (engine/sample.py): pick the sampling kernels by
        # periph_flashinfer_ok() so MTP drafts/verifies sample with the SAME backend the rest of
        # the step uses (FREETOKEN_FLASHINFER_PERIPH=0 -> triton), never a divergent one.
        from freetoken.kernel.backend import periph_flashinfer_ok

        if periph_flashinfer_ok():
            import flashinfer.sampling as mtp_sampling
        else:
            import freetoken.kernel.triton.sampling as mtp_sampling

        batch, sample_args, input_mapping, output_mapping = forward_input
        reqs = list(batch.reqs)
        K = max(1, int(self.config.mtp_draft_len))
        dev = self._mtp_backend.draft_device()
        mtp = self.engine.model.mtp
        gdn = self.engine.linear_state_pool
        scratch_base = self.engine._mtp_scratch_base
        # Timing for the per-step summary line: step_ms brackets the whole step (CPU draft chains
        # + GPU forwards); verify_ms is the GPU verify window alone. Their gap is the CPU/GPU
        # overlap that bs>1 buys -- drafts for other requests compute while verify is on the GPU.
        step_start = torch.cuda.Event(enable_timing=True)
        step_start.record(self.engine.stream)
        verify_a = torch.cuda.Event(enable_timing=True)
        verify_b = torch.cuda.Event(enable_timing=True)

        # Snapshot per-request starting position before any helper mutates cached_len/device_len.
        starts = [(req, int(req.cached_len), req.table_idx) for req in reqs]

        # 0. MTP context-KV integrity per request: slots 0..L-1 must hold the true fused rows.
        # Repair by re-running the prefix through the prefill hook when stale (the GDN slot is
        # zeroed so the chunked path recomputes S_L from scratch). Independent per request.
        for req, L, T in starts:
            upto = mtp._ctx_upto.get(T, -1)
            pend = mtp._ctx_pending.get(T)
            covered = upto >= L - 1 or (pend is not None and pend[0] == L - 1 and upto == L - 2)
            if getattr(req, "_mtp_ctx_dirty", False) or not covered:
                req._mtp_ctx_dirty = False
                mtp._ctx_pending.pop(T, None)
                mtp._ctx_upto[T] = -1
                gdn_slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
                if gdn is not None:
                    gdn.reset(gdn_slot)
                rb, rmap = self._mtp_extend_batch([req], [(0, L)], allocate=False)
                rb.input_ids = self.token_pool[rmap]
                self._mtp_backend.rerun(rb)
                req.cached_len = L
                req.device_len = L + 1

        # 1. one batched eager decode forward: anchors per row + stashed pre-mixer residuals.
        out, residual_all = self._mtp_backend.anchor(batch, sample_args)
        self.token_pool[output_mapping] = out.next_tokens_gpu[: len(reqs)].to(torch.int32)

        # 2. draft chains per request, sequential (the scratch slots are reused across requests,
        # only one chain is live at a time). Reference pairing: draft j sits at rope position
        # L+j-1 fed token_pool[L+j] and chains the previous draft's MULTI stream verbatim.
        drafts_per_req: list[list[torch.Tensor]] = []
        probs_per_req: list[list[torch.Tensor]] = []   # per req: [K] x [1, vocab] q_j (head dist)
        d1_logits_first = None
        mtp_diag_first = None
        extra_locs = torch.arange(
            scratch_base, scratch_base + K - 1, dtype=torch.long, device=dev
        )
        for idx, (req, L, T) in enumerate(starts):
            # Spec-sampling: draw each draft from the head's OWN distribution q_j, not argmax.
            # The accept/reject below corrects q_j -> p_j (the main model's sampling dist), so the
            # emitted sequence follows the target exactly; an argmax draft would break that
            # correction. A greedy request collapses q_j to a near-delta at argmax (tiny T) so it
            # reduces exactly to the old argmax==argmax match with maximum acceptance.
            T_draft = 1e-6 if req.sampling_params.is_greedy else 1.0
            residual_i = residual_all[idx : idx + 1]
            drafts: list[torch.Tensor] = []
            d_probs: list[torch.Tensor] = []
            prev_multi = None
            for j in range(1, K + 1):
                fed_gpu = self.token_pool[T, L + j : L + j + 1]
                res_j = residual_i if j == 1 else prev_multi
                diag_on = idx == 0 and j == 1 and not getattr(self, "_mtp_diag_done", False)
                if diag_on:
                    mtp_diag_first = {}
                    mtp._diag = mtp_diag_first
                db = self._mtp_draft_batch(
                    req, dev, pos=L + j - 1, cached_len=L, out_loc=scratch_base + (j - 1)
                )
                if j > 1:
                    db.mtp_extra_locs = extra_locs[: j - 1]
                d_logits, prev_multi = self._mtp_backend.draft_forward(db, res_j, fed_gpu)
                if diag_on:
                    d1_logits_first = d_logits
                q_j = mtp_sampling.softmax(d_logits, T_draft)               # [1, vocab]
                d = mtp_sampling.sampling_from_probs(q_j).to(torch.int32).view(1)
                drafts.append(d)
                d_probs.append(q_j)
                self.token_pool[T, L + j + 1 : L + j + 2] = d
            drafts_per_req.append(drafts)
            probs_per_req.append(d_probs)

        # 3. verify forward: every request contributes K+1 rows (anchor + K drafts) over its own
        # position window [L+1, L+K+2), assembled into ONE batched prefill forward.
        snap_slots: list = []
        if gdn is not None:
            for req, L, T in starts:
                slot = gdn.alloc(1)[0]
                live = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
                gdn.copy_from(live, slot)
                snap_slots.append(slot)
        # Verify rows carry draft tokens (may be rejected): the context-KV hook must not write
        # them; the step-end rewrite installs the TRUE rows instead.
        mtp._ctx_write_off = True
        ranges = [(L + 1, L + K + 2) for _, L, _ in starts]
        vbatch, vmap = self._mtp_extend_batch(reqs, ranges, allocate=True)
        # K+1 rows per request is decode-sized work: take the on-demand top-k MoE movement
        # instead of the prefill whole-layer stream.
        vbatch.moe_force_decode = True
        vbatch.input_ids = self.token_pool[vmap]
        verify_a.record(self.engine.stream)
        vres, v_logits_all = self._mtp_backend.verify_logits(batch, vbatch)
        verify_b.record(self.engine.stream)
        v_pred_all = v_logits_all.argmax(dim=-1).to(torch.int32)  # [B*(K+1)]

        # 4. spec-sampling acceptance (Leviathan / vLLM rejection sampler), per request. Each
        # draft d_j is drawn from the head's own distribution q_j; keep it iff the main model's
        # sampling distribution p_j over that position would have produced it -- accept iff
        # u * q_j[d_j] < p_j[d_j] (the numerically-stable form of u < p_j[d_j]/q_j[d_j]).
        # p_j is the request's EXACT sampling distribution (temperature + top-k + top-p), so the
        # emitted sequence follows it exactly. On the first reject (position m) emit one token
        # from the normalized residual max(0, p_m - q_m); if all K survive, emit the main
        # model's own draw at the post-draft position p_K.
        committed_per_req: list[list[int]] = []
        rollback_needed: list = []     # (req, L, T, m)
        bonus_gpu_per_req: list = []   # per req: [1] int32 drawn bonus (GPU-resident)
        for idx, (req, L, T) in enumerate(starts):
            sp = req.sampling_params
            v_slice = v_logits_all[idx * (K + 1) : (idx + 1) * (K + 1)]   # [K+1, vocab]
            if sp.is_greedy:
                # target is a near-delta at argmax, so accept/reject reduces to argmax==argmax;
                # skip top-k/top-p (argmax has none).
                p_full = mtp_sampling.softmax(v_slice, 1e-6)             # [K+1, vocab]
            else:
                # temperature 0 is not "greedy" here only because top-k/top-p were set; its
                # sampling dist is the near-argmax limit, so a 0-temperature softmax (1/0 -> NaN)
                # is replaced by a tiny temperature (top-k/p on that delta are no-ops).
                t = sp.temperature if sp.temperature > 0.0 else 1e-6
                p_full = mtp_sampling.softmax(v_slice, t)
                if sp.top_k >= 1:
                    p_full = mtp_sampling.top_k_renorm_probs(p_full, sp.top_k)
                if 0.0 < sp.top_p < 1.0:
                    p_full = mtp_sampling.top_p_renorm_probs(p_full, sp.top_p)
            # The q-side arrives from the draft chain (CPU when mtp_device=cpu) and the p-side
            # from the GPU verify logits; acceptance is tiny math, so land it all on the logits
            # device instead of mixing (gather indices must match their input's device).
            q_full = torch.cat(probs_per_req[idx], dim=0).to(p_full.device)   # [K, vocab]
            d_tokens = torch.cat(drafts_per_req[idx], dim=0).to(torch.int32).to(p_full.device)  # [K]
            p_d = p_full[:K].gather(1, d_tokens[:, None]).squeeze(-1)   # [K] target prob of d_j
            q_d = q_full.gather(1, d_tokens[:, None]).squeeze(-1)       # [K] head prob of d_j
            u = torch.rand(K, device=p_full.device, dtype=torch.float32)
            accept = (u * q_d) < p_d                                    # [K]
            # ONE host sync for this request's scalars (anchor, K draft ids, K accept flags):
            # the draft chain and token_pool writes stay GPU-resident, only these few cross.
            vals = torch.cat([
                out.next_tokens_gpu[idx : idx + 1].view(1).to(torch.int32),
                d_tokens,
                accept.to(torch.int32),
            ]).to("cpu").tolist()
            anchor_i = vals[0]
            d_ids = vals[1 : 1 + K]
            acc_cpu = vals[1 + K : 1 + 2 * K]
            m = next((i for i in range(K) if not acc_cpu[i]), K)        # first reject (or K)
            if m < K:                                                   # rejected at m: residual draw
                resid = torch.clamp(p_full[m] - q_full[m], min=0.0)
                resid = resid / resid.sum()
                bonus_gpu_t = mtp_sampling.sampling_from_probs(resid[None, :]).to(torch.int32).view(1)
            else:                                                       # all kept: main model's own draw
                bonus_gpu_t = mtp_sampling.sampling_from_probs(p_full[K : K + 1]).to(torch.int32).view(1)
            bonus = int(bonus_gpu_t.item())
            committed = [anchor_i] + d_ids[:m] + [bonus]                # m+2 tokens
            req.mtp_drafts_total += K
            req.mtp_accepted_total += m
            req.mtp_steps += 1
            if m >= 1:
                req.mtp_first_match += 1
            if m == K:
                req.mtp_full_accept += 1
            committed_per_req.append(committed)
            rollback_needed.append((req, L, T, m))
            bonus_gpu_per_req.append(bonus_gpu_t)
            if idx == 0:
                self._mtp_head_diag_once(
                    mtp_diag_first,
                    d1_logits_first,
                    v_slice,
                    residual_all[0:1],
                    anchor_i,
                    d_ids,
                    v_pred_all[idx * (K + 1) : (idx + 1) * (K + 1)].tolist(),
                    m,
                )

        # 5. rollback per request whose tail was cut (settles GDN state to the committed prefix).
        # Split the reject-path CPU cost: rollback_ms = GDN slot restore (memcpy), redraft_ms =
        # recompute of the committed prefix. The reject-path doubling (~57->121ms at bs=1) shows
        # up as step_ms - verify_ms; these two tell which half to attack.
        rb_acc = 0.0
        rd_acc = 0.0
        for idx, (req, L, T, m) in enumerate(rollback_needed):
            slot_i = snap_slots[idx] if snap_slots else None
            if m < K and slot_i is not None:
                _t0 = time.perf_counter()
                live = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
                gdn.copy_from(slot_i, live)
                rb_acc += time.perf_counter() - _t0
                _t1 = time.perf_counter()
                rbatch, rmap = self._mtp_extend_batch(
                    [req], [(L + 1, L + 1 + m + 1)], allocate=False
                )
                rbatch.moe_force_decode = True
                rbatch.input_ids = self.token_pool[rmap]
                self._mtp_backend.rerun(rbatch)
                rd_acc += time.perf_counter() - _t1
            if slot_i is not None:
                _t2 = time.perf_counter()
                gdn.free(slot_i)
                rb_acc += time.perf_counter() - _t2
        self._mtp_rb_acc = getattr(self, "_mtp_rb_acc", 0.0) + rb_acc * 1e3
        self._mtp_rd_acc = getattr(self, "_mtp_rd_acc", 0.0) + rd_acc * 1e3
        mtp._ctx_write_off = False

        # 6. commit + emit per request: return the uncommitted whole-page tail the verify
        # reserved but the commit does not own; rewrite the MTP context slots with TRUE rows.
        ps = self.cache_manager.page_size
        bonus_gpu = []
        for idx, (req, L, T, m) in enumerate(rollback_needed):
            committed = committed_per_req[idx]
            keep_end = div_ceil(L + 2 + m, ps) * ps
            verify_end = div_ceil(L + K + 2, ps) * ps
            if verify_end > keep_end:
                self.cache_manager._free(self.cache_manager.page_table[T, keep_end:verify_end])
            req.cached_len = L + 2 + m
            req.device_len = L + 3 + m
            bonus_gpu_t = bonus_gpu_per_req[idx]
            self.token_pool[T, L + 2 + m : L + 3 + m] = bonus_gpu_t.to(torch.int32)

            # 6b. rewrite context slots L..L+m+1 with TRUE fused rows.
            n_rw = m + 2
            residual_i = residual_all[idx : idx + 1]
            v_off = idx * (K + 1)
            rw_multi = torch.cat([residual_i, vres[v_off : v_off + m + 1]], dim=0)
            rw_toks = self.token_pool[T, L + 1 : L + 1 + n_rw].to(torch.int32)
            rw_slots = self.cache_manager.page_table[T, L : L + n_rw].to(torch.long)
            rw_base = torch.arange(L, L + n_rw, dtype=torch.int32, device=dev)
            if self._model_is_mrope:
                rw_pos = (rw_base + req.mrope_delta).unsqueeze(0).expand(3, -1).contiguous()
            else:
                rw_pos = rw_base
            mtp.write_rows(rw_toks, rw_multi, rw_slots, rw_pos)
            mtp._ctx_pending.pop(T, None)
            mtp._ctx_upto[T] = L + m + 1
            bonus_gpu.append(bonus_gpu_t)

        # Per-step observability line: one line per MTP step showing batch size, drafts drawn by
        # the head (CPU-side draft chain), drafts accepted by the verify, the bonus tokens the
        # GPU verify contributed at each cut, and GPU-vs-wall timing so the CPU-draft / GPU-verify
        # overlap is visible (step_ms - verify_ms = time the CPU draft chains ran concurrently).
        step_end = torch.cuda.Event(enable_timing=True)
        step_end.record(self.engine.stream)
        step_end.synchronize()
        step_ms = step_start.elapsed_time(step_end)
        verify_ms = verify_a.elapsed_time(verify_b)
        bs = len(reqs)
        acc_total = sum(len(c) - 2 for c in committed_per_req)  # accepted drafts across reqs
        drafted_total = bs * K
        self._mtp_step_count = getattr(self, "_mtp_step_count", 0) + 1
        self._mtp_rb_n = getattr(self, "_mtp_rb_n", 0) + 1
        every = max(1, int(getattr(self.config, "decode_log_interval", 40)))
        if bs > 1 or self._mtp_step_count % every == 0:
            n = max(1, self._mtp_rb_n)
            rb_ms = getattr(self, "_mtp_rb_acc", 0.0) / n
            rd_ms = getattr(self, "_mtp_rd_acc", 0.0) / n
            self._mtp_rb_acc = 0.0
            self._mtp_rd_acc = 0.0
            self._mtp_rb_n = 0
            logger.info_rank0(
                f"MTP step bs={bs} drafted={drafted_total} accepted={acc_total} "
                f"bonus={bs} committed={sum(len(c) for c in committed_per_req)} "
                f"acc_rate={acc_total / max(1, drafted_total):.2f} "
                f"verify_ms={verify_ms:.1f} step_ms={step_ms:.1f} "
                f"rollback_ms={rb_ms:.1f} redraft_ms={rd_ms:.1f}"
            )

        next_tokens_gpu = torch.cat(bonus_gpu).to(torch.int32)
        all_committed = [t for lst in committed_per_req for t in lst]
        next_tokens_cpu = torch.tensor(all_committed, dtype=torch.int32)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.engine.stream)
        return ForwardOutput(
            next_tokens_gpu, next_tokens_cpu, copy_done_event, committed_per_req
        )

    def _mtp_head_diag_once(self, mtp_diag, d1_logits, v_logits, residual, anchor_i, d_ids, V, m):
        """First-MTP-step head sanity: a 0%-acceptance run must tell a DEGENERATE head
        (NaN/inf/all-zero logits -> kernel/dtype bug) from a merely divergent one. Cheap:
        a few top-k over tiny slices, once per process."""
        if getattr(self, "_mtp_diag_done", False) or d1_logits is None:
            return
        self._mtp_diag_done = True
        try:
            def _st(t):
                if not torch.isfinite(t).all():
                    return "NONFINITE"
                return (
                    f"min={float(t.min()):.4g} max={float(t.max()):.4g} "
                    f"mean={float(t.mean()):.4g} std={float(t.std()):.4g}"
                )

            head = d1_logits[0]  # head's logits for position L+2
            main = v_logits[0]  # main model's logits for the same position
            h_ids = [int(i) for i in head.topk(5).indices.cpu()]
            m_ids = [int(i) for i in main.topk(5).indices.cpu()]
            overlap = len(set(h_ids) & set(m_ids))
            logger.info_rank0(
                "MTP DIAG head: " + _st(head) + f" top1={int(head.argmax())} top5={h_ids}"
                " | model: " + _st(main) + f" top1={int(main.argmax())} top5={m_ids}"
                f" | top5_overlap={overlap}/5 m={m}"
            )
            if mtp_diag:
                logger.info_rank0(
                    "MTP DIAG head internals (first draft): "
                    + ", ".join(f"{k}={v}" for k, v in mtp_diag.items())
                )
        except Exception as e:  # never let a diagnostic break serving
            logger.info_rank0(f"MTP DIAG (error): {e!r}")


def _make_mrope_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """[3, N] rope rows: an image request's prompt tokens use their precomputed columns, everything else is sequence index + per-request delta."""
    needed = sum(r.extend_len for r in batch.padded_reqs)
    host = torch.empty((3, needed), dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        out = host[:, offset : offset + length]
        full = req.mrope_positions_full
        if full is not None and req.device_len <= full.shape[1]:
            out.copy_(full[:, req.cached_len : req.device_len])
        else:
            row = torch.arange(
                req.cached_len + req.mrope_delta,
                req.device_len + req.mrope_delta,
                dtype=torch.int32,
            )
            out.copy_(row.unsqueeze(0).expand(3, -1))
        offset += length
    return host.to(device, non_blocking=True)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)