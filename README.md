# FreeToken — Windows single-GPU build

Chinese version: [README_BRANCH_zh.md](README_BRANCH_zh.md).

A patched branch of [FreeToken](https://github.com/FlashML-org/FreeToken) (base `0.1.2+g816c324d0`, upstream main `58f4b9e`) tuned for **one consumer GPU + lots of system RAM** — e.g. RTX 5090 32 GB with ~254 GiB RAM — serving large MoE checkpoints through an OpenAI-/Anthropic-compatible HTTP API. Upstream Apache-2.0 LICENSE preserved; see [Provenance](#provenance).

Every item below follows the same shape: **(1) Problem observed** on this stack, **(2) Mechanism implemented**, **(3) Measured verification result**. Reference rig for every number: RTX 5090 SM120 PCIe-bound + Qwen3.8-Flash-Next(-ABLITERATED)-NVFP4.

---

## Feature points

### 1. Shared cudaMemPool CUDA-graph ladder with fixed-shape collapse

**(1)** Each captured bs in upstream stacks its own workspace and graph-private pool; with bs ∈ {1,2,4,8} ladder the dead-slot overhead eats contiguity headroom the verify peak needs. bs-histogram in `/v1/stats` showed ≥95% of traffic is bs=1 on this workload, yet bs=4/8 graphs were still capturing.

**(2)** `engine/graph.py` captures every ladder entry against a single shared `cudaMemPool` plus one shared `GraphCaptureBuffer` sized to `max_graph_bs`. Ladder entries share workspace instead of stacking it, so adding entries costs topology metadata (~few MB), not another workspace. Ladder = `[1, 2, 4] + range(8, max_bs+1, 8)`. `--cuda-graph-max-bs 1` collapses to single bs=1 capture; replay pads up to next ladder entry.

**(3)** bs=1-dominated traffic: bs=1 ladder leaves ~7.5 GiB contiguous free after PyTorch allocator overhead, comfortably covers MTP verify peak (~3–4 GiB contiguous at bs=4 with `--mtp-draft-len 2`). bs=4 mixed-decode bench: steady ~104 tok/s aggregate across rounds with bs=4 ladder. No quality hit from pad-to-next-ladder-entry.

---

### 2. Verify-phase statistics exclusion in MoE expert LRU

**(1)** MTP verify batch carries a K+1-row distribution that differs from decode-phase routing. Mixing those counts into hot-expert LRU promotes experts that only verify needs, evicting real decode working set.

**(2)** `MtpExpertBank` and the `OffloadMoeCache` chain (`load_experts` / `ensure_experts` / `ensure_experts_hybrid` / `prefetch_decode_layer`) thread a `record_stats: bool` kwarg driven by `ctx.batch.moe_force_decode` from scheduler. Verify admissions update hit/miss counters; they do **not** promote into LRU. Decode admissions do both.

**(3)** Expert residency tracks decode demand without verify-phase noise; bs=8 mixed-decode aggregate ~187/~193 tok/s across rounds. bs=4→bs=8 scaling ×1.85 — within PCIe BW ceiling, no LRU-thrash dip.

---

### 3. b12x Triton backend for NVFP4 MoE decode

**(1)** cuBLASLt on SM120 doesn't saturate PCIe BW for the NVFP4 MoE shapes on this model (`hidden=2560`, `num_experts=512/topk=10`, `moe_intermediate=640`); launch overhead per route dominates on bs ∈ {1,4,8} decode.

**(2)** `kernel/triton/nvfp4_fused_moe.py` is marlin-style: wide int32 loads (8 codes/word), deferred K reduction, register-only `_split8` shuffle, fp32 accumulate, per-route grid `(num_routes, cdiv(N, BLOCK_N))`. Block-scale epilogue folded into fp16 bits with global-scale applied post-K-loop + arrival-counter split-K reduction in single launch — no second kernel.

**(3)** ~87% PCIe BW ceiling hit on RTX 5090. Per-stream decode ~25 tok/s is length-independent up to ~50k context (KV read BW not yet bottleneck — weight streaming dominates). Aggregate bs=4 ~104 tok/s mixed-decode across ~10k/~50k prompts. Residual ~13% gap attributed to FP4-dequant ALU; this covers what DeepGEMM tiled cluster kernels offer in our shape range, so DeepGEMM not pulled in.

---

### 4. Hybrid QSA sparse attention + turbo4 KV storage tier

**(1)** bf16 paged KV eats too much VRAM for bs=1 long-context on a PCIe-bound stack. Need storage-side compression without touching compute path.

**(2)** Three tiers measured against bf16 reference attention (max-cosine over sparse + split-k paths):

| tier | bytes / token / KV layer | cosine vs bf16 | use when |
|---|---|---|---|
| `bf16` | 2048 | 1.0000 | default; smallest contexts |
| `fp8_e4m3` | 1024 | 0.9994 | general balance point |
| `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

Storage compressed; compute stays bf16. Indexer keys always bf16. QSA paged KV pool stores in turbo4 packed nibble rotspace; block top-k uses Triton `qsa_block_topk`. Engine gate keeps turbo4 on QSA + MLA-latent families.

**(3)** bs=1 holds ~50k-token context inside ~3 GiB KV pool with no verify-phase OOM. Needle retrieval 6/6 exact across 322k/450k × depth {30%, 60%, 90%}, worst-case depth-90% at 35.2s, rest 5–12s. Measured KV ≈6.9 KiB/token → full 1M-token pool ≈6.8 GiB. TTFT @ 901k context: 32.5s with prefill overlap, 51s without. Steady-state decode length-independent.

---

### 5. MoE prefill demand threshold gated on chunk row count

**(1)** Gating expert-fetch on whole prompt length over-fetches when cached-prefix incremental prefills add only small chunks per step but cumulative context is huge. Whole-prompt gating treats each incremental chunk as if it were cold-start.

**(2)** `--moe-prefill-demand-tokens` gates on chunk row count (`hidden_states.shape[0]`), not whole prompt length. Default `64`. Cached-prefix hits skip demand-prefill path; cold starts pay it once.

**(3)** Cached-prefix warm hits refill expert cache without re-fetching whole-prompt working set. bs=8 mixed-decode TTFT ~47s reflects chunked-prefill queueing across 8 streams × XLONG prompts × ~8192-token chunks — not demand-gating overhead. Round-to-round TTFT drop confirms warm prefix-cache effect.

---

### 6. Round-7 runtime rebalancing — eight independent rules

**(1)** Fixed VRAM arena must adapt to traffic mix changes without engine restart or weight reload. Each upstream knob on its own is too coarse; combined rules need to be independently gateable so disabling one doesn't perturb the others.

**(2)** Eight small rules, shared invariant: VRAM allocation happens once at startup; everything after is view reslicing and bookkeeping reset.

- **Recurrence-window MoE admission** — `ensure_experts_hybrid` fetches onto GPU on second sighting in window; first misses compute CPU-side. Gate fires per-layer at moment of miss, not periodic sweep. Removes cold-prefill cache thrash.
- **bs-sized transport routing** — `hybrid_fetch_fraction` (0–1 ratio read once from `ft bench bw`) replaces fixed `hybrid_max_fetch` count. Each decode step fetches `fraction × misses` experts; PCIe stays saturated and CPU GEMM overlaps across bs=1…8. Profile read at engine init only.
- **bs≥4 decode-step prefetch** — `_decode_hybrid` issues `prefetch_decode_layer(next_layer, hint)` before current-layer GEMM dispatch when bs≥4. Below bs=4 routing entropy exceeds prefetch-hit benefit. Tests in `tests/moe/test_decode_prefetch_gate.py`.
- **KV hot-window** — `FREETOKEN_HOT_WINDOW_TOKENS` caps `demand_tokens` signal the slider sees, letting KV grow into soft ceiling instead of pure shrink-only. Opt-in: unset preserves shrink-only exactly. Sized for multi-turn agent prompts where hard shrink-only floor evicts still-useful prefix pages.
- **Prefill/decode interleaving** — `_interleave_every` (default 4) consecutive prefill chunks force one decode step. Counted in chunks not seconds for deterministic replay. Override via `prefill_interleave_every`.
- **Asymmetric EMA shrink** — `BudgetSlider.propose_targets` rises fast (high EMA α, avoid under-provision in bursts) falls slow (low α, avoid ping-pong on load drop). `min_step_ratio` deadband ignores sub-threshold wobble; `demand_floor` keeps shrink-only above next-request working set. Fires only at idle transitions.
- **Special-token ckpt for tool-call reuse** — `--enable-special-token-ckpt` snapshots GDN state at tool-call opener token (structural boundary: pre-token content identical across turns, post-token drifts). Snapshot paid once per turn; prefix reuse hit on every echo. Replaces older "ckpt every N tokens" heuristic where most ckpts never matched.
- **BudgetSlider carrier** — `engine/budget_slide.py` re-derives KV pages ↔ MoE slots from current traffic. `mha_pool.rebuild` / `OffloadMoeCache.rebuild` re-slice views from arena left end; base pointer never moves, CUDA graphs stay valid. Fires only at idle transitions — the only moment with no live page-index references and no in-flight graph replay. Preserve-residency: pinned experts stay pinned.

**(3)** bs=4 mixed-decode aggregate steady ~104 tok/s across rounds; bs=8 ~187/~193 tok/s; bs=4→bs=8 scaling ×1.85 within PCIe BW ceiling. With `--max-running-requests 8` bs=6 fills bs=8 graph → aggregate lifts to ~76.9 at cost of p95 ≈7s (c8 tail ≈25s). Rebuild cadence observed: every few requests during bursts, every ~minute when idle. `budget_slide_test.py` green.

---

### 7. Windows compatibility layer + split residency + pin budget

**(1)** WDDM caps GPU-accessible pinned host memory at roughly half physical RAM; locked working-set pages count against same budget. Upstream's POSIX-only `resource` module, `expandable_segments`, and Proactor/zmq loop divergence don't degrade gracefully on Windows. Large MoE weights exceed pinned ceiling at default budget.

**(2)** `ctypes` `VirtualLock` replaces POSIX-only `resource`; expandable_segments graceful degradation path; Proactor loop adaptation; `FREETOKEN_PIN_BUDGET_GB` env knob (auto ≈0.4–0.5 × RAM by default). Below ceiling, head/tail MoE layers OS-lock CPU and decode on multi-threaded CPU executor (AVX-512 BF16 + VNNI, NVFP4 W4A8); remaining banks pin for GPU streaming. Locked ≠ pinned: locked banks decode CPU-side, pinned stream GPU-side.

**(3)** Steady-state on bs=1 ladder: ~14 GiB allocated / ~16 GiB contiguous free — verify peak fits comfortably. PyTorch reports less free than `nvidia-smi` by ~3 GiB (CUDA driver context + cuBLASLt workspace + cuDNN pools + CUDA-graph private pools sit outside PyTorch allocator visibility). `torch.cuda.empty_cache()` recovers that gap only at startup; runtime peak sizing must come from shrinking steady-state allocated. Startup OOM on last bank: lower `--num-tokens` first (turbo4 ≈6.9 KiB/token), then budget.

---

### 8. MM content-key prefix cache + splice invariant fix (Round-4 / P4)

**(1)** Repeated mm requests re-prefill identical image pixels; pad-span alignment in splice can overrun COW buffer when prefix matches land inside image_pad span. Production crash regression observed.

**(2)** Tokenizer worker hashes image pixel payload (sha1) → content id; mm prefix matching/insertion routes through `mm_cache_key` (image_pad spans replaced by content id) — same image → KV reuse, different image → zero false hits. `match_req` for prefixes landing inside pad span falls back to `align_down(span_start, page_size)` and drops GDN snapshot (prevents COW overrun); matches past last pad pass through unchanged. `qwen4_exp` / `gemma4` splice at `n_pads==0` is legal no-op; intermediate states still raise.

**(3)** Content-key reuse 13/13; adversarial cross-image isolation 174/174 green. `p4_splice_test.py` 21 legs green including production crash regression: track@960 crossing middle-span → align_down=8 fallback, GDN dropped, 955 pads all-in-row restored. Restart re-verify: 5 suites + 174 adversarial cases all green; three changes still in packaged tree.

---

### 9. MM video path + dual-pad fix (Round-5)

**(1)** Different checkpoints write either `image_token_id` or `video_token_id` (both wrapped inside vision_start/end). PIL without ffmpeg backend can't read mp4 containers. Need single serve path that handles both spellings without model-side changes.

**(2)** Detect both spellings in prepare; expansion uniformly writes `image_token_id`, so model-side embed mask / cache key / scheduler stay unchanged; `vision_geometry` passes `video_token_id` through. Video parts accept frame lists (data:/http/local paths) or single animated files (GIF/WebP/APNG); ≤8 frames sampled evenly; multi-frame uses processor `videos=` temporal pairing (falls back to per-frame stacking) with `t>1` grid; expand count comes from real grid, still renders exactly one pad placeholder. `extract_media_parts` mirrors template branch order. mp4 fallback via `imageio` (bundles static `imageio-ffmpeg`): read whole segment, sample ≤8 frames. 32768-patch budget covers whole video segment; guard past it returns clean 400 `context_length_exceeded`, streaming carries it as in-stream error frame.

**(3)** `serve_video_smoke.py` / `vision_video_test.py` / `vision_p3_media_test.py` green vs HF oracle bit-match. Live serve (:8001, default vision on) smoke: frame-list / GIF / image_url+video mixed legs all correct (static green ∉ {red, blue} animated). Measured prompt_tokens=238 on a representative reasoning leg; reasoning leg budget needs ≥512. Large image (19200 patches) peaks VRAM +984 MiB, first-token latency 2.9–4.2s. Image requests vs text: cold TTFT medians text 1518ms vs image 1526ms (+8ms). ViT forward not dominant — the ~1.5s floor is fixed MoE expert-cache cold-start which text pays too.

---

### 10. MM RAM tier swap-in/out (Round-6)

**(1)** Same mm content arriving hours later pays full ViT + prefill cost from cold KV. No mechanism to reuse prompt-segment KV across long gaps within pinned-budget arena.

**(2)** Completed mm requests stash prompt-segment KV + GDN state to host RAM. Subsequent same-content-key requests restore rows into freshly allocated VRAM pages and continue directly. `kvcache/mha_pool.py` / `qsa_pool.py`: `snapshot_rows` / `restore_rows` pack K/V compressed blocks + turbo4 dequant scales + QSA compressed-index rows (`slot // index_ratio`) to CPU; restore writes back via `index_copy_` in-place scatter (`[:, idx].copy_` would gather into a temp first and silently fail to write — known trap; documented inline). `pending_ring` is transient and not tiered. `linear_state_pool.snapshot_state` / `restore_state` pack conv+recurrent+slot_states whole-slot. Tier storage uses LRU-bytes eviction with `FREETOKEN_MM_TIER_MB` budget (default 1024). Restore gate: content-key bit-match + boundary beats tree match + doesn't cut a pad run → allocate new pages + row refill + `_TierHandle`. Restore integrated into admission after budget gate; `mm_gdn` propagated through PendingReq→Req; `_restore_linear_states` writes back on first prefill forward.

**(3)** `kv_tier_test.py` green on real QSA+turbo4 pools — row round-trip, byte accounting, same-key replacement, LRU eviction, pad-boundary GDN write-back. Vision two-suite regression green. Live serve (:8001) smoke: same-content repeated request takes restore path and answers correctly; image_url+video mixed legs judged correctly (static green ∉ animated {red, blue}); prompt_tokens=238 measured. Warm TTFT below cold TTFT confirms tier hits.

---

## Bench findings

Two measured curves that drive serve-profile choices.

**Concurrency sweet spot.** bs=4 maximises per-stream latency × aggregate on this rig. `max_tokens=96`, text-only, `--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --moe-prefill-hit-d2d`:

| concurrency | aggregate tok/s | per-stream tok/s | notes |
|---|---|---|---|
| 1 | 26.2 | 26.2 | |
| 2 | 47.5 | ~24 | |
| 4 | 66.9 | 17.2 | p50 ≈5.0s, p95 ≈5.2s — sweet spot |
| 6 | 64.6 | ~11 | |
| 8 | 68.8 | 9.9 | p95 climbs back |

Aggregate soft ceiling ≈65–77 tok/s — set by PCIe BW + per-step expert-fetch volume, not compute. Adding `--max-running-requests 8` lets bs=6 fill the captured bs=8 graph and lifts c6 aggregate to ~76.9 at cost of p95 ≈7s (c8 tail ≈25s) — throughput-first only.

**MoE cache size vs decode speed.** `ft ctl cache rebuild` live resize, no restart. Slots 1024 → 2048 → 3072 give long-text steady-state decode ≈62 → 75 → 84 char/s (≈33 → 40 → 44 tok/s; short text ≈52). Cumulative **+36% with zero quality loss** — cache stores exact weights, gains come purely from hit rate. Past ~3072 curve flattens (+5–8% but eats KV pool).

---

## Serve profiles

```powershell
# optional: only when expert weights exceed the pin budget (~half RAM under WDDM)
$env:FREETOKEN_PIN_BUDGET_GB = "120"

# turbo4 + 1M KV — longest-context profile, encoders on
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 2048 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d

# turbo4 + 512K KV — larger resident expert cache
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 524288 --moe-cache-size 3072 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d

# bs=1-dominated tight profile — single-graph ladder, MTP off, q4_0 bank, text-only
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 65536 --moe-cache-size 1024 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d `
  --cuda-graph-max-bs 1 --moe-prefill-demand-tokens 64 `
  --mtp-expert-quant q4_0 --text-model-only
```

Flags worth understanding:

- `--num-tokens` — paged KV pool size. turbo4 ≈6.9 KiB/token → ≈6.8 GiB full pool at 1M, ≈3.4 GiB at 512k, ≈0.4 GiB at 65k. Smaller pool leaves contiguous VRAM headroom for verify peak.
- `--moe-cache-size` — experts resident on GPU. Steady-state decode scales roughly with hit rate up to ~3072, then flattens.
- `--kv-cache-dtype` — see tier table under feature point 4.
- `--text-model-only` — skip all encoder towers; their VRAM goes to KV / expert caches. Use `--mm-disable vision audio` instead when only some kinds should be skipped. Vision / audio towers load by default and skip cleanly when disabled.
- `--moe-prefill-hit-d2d` — serves prefill expert hits via device-to-device copies.
- `--cuda-graph-max-bs N` — CUDA-graph ladder cap; `1` for bs=1-dominated, `4` adds bs=4 burst slot cheaply under shared mempool.
- `--moe-prefill-demand-tokens` — chunk-row threshold for demand-prefill; default `64` fits cached-prefix incremental prefill.
- `--mtp-draft-len` / `--mtp-device gpu` — opt-in. Measured acceptance ~0.70 with draft-len=2 gives net-positive on multi-stream aggregate but net-negative on bs=1-dominated traffic where verify overhead (~70ms/step) + redraft (~30ms) exceeds savings. Default-off on this branch.

## Requirements & install

Windows 11 (or WSL2) x64, NVIDIA GPU ≥24 GiB VRAM, ≥2× model-size RAM recommended. Python 3.12 (`cp312` wheels; kernel-cache wheel built for CUDA 13.x). Source-only; dependencies from public indexes.

```powershell
python -m venv ft-venv
ft-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
ft-venv\Scripts\pip install .
```

Optional: kernel-cache wheel attached to GitHub Releases (skips Triton JIT on first request; must match CUDA 13.x). Optional: `imageio` bundles static `imageio-ffmpeg` for mp4 container support on vision path. Then hit `POST /v1/chat/completions` like any OpenAI-compatible server.

## Verification suite

| script | covers |
|---|---|
| `p4_splice_test.py` | mm splice invariant (pad-span alignment, COW no-op) — 21 cases |
| `p4_finish_key_test.py` | finish-key + SWA regression legs |
| `p4_content_key_test.py` | content-hash prefix keys (same-image reuse / cross-image isolation) |
| `mm_prefill_chunk_test.py` | whole-segment mm prefill rollback |
| `r4_adv_probe.py` | adversarial fuzz: cross-span gaps, page ownership, `page_size ∈ {1,4,8}` × shapes × rounds |
| `kv_tier_test.py` | RAM-tier KV swap-in/out round-trip |
| `budget_slide_test.py` | runtime KV/expert pool resize without restart |
| `serve_video_smoke.py`, `vision_video_test.py`, `vision_p3_media_test.py` | end-to-end vision/media requests |

Run from package root with venv Python, e.g. `python verification/p4_splice_test.py`.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | cap for CUDA-pinned expert banks; exceeding triggers split residency (locked CPU layers) | auto (~0.4–0.5 × RAM) |
| `FREETOKEN_MM_TIER_MB` | mm RAM tier byte budget for swap-in/out eviction | 1024 |
| `FREETOKEN_HOT_WINDOW_TOKENS` | caps demand_tokens signal so KV grows to soft ceiling instead of pure shrink-only | unset (shrink-only preserved) |

## Provenance

Derived from upstream FreeToken commit `58f4b9e` (release `0.1.2+g816c324d0`). Upstream PRs merged in: #377 (nightly engine manifests), #375 (issue templates), #343 (NVFP4 detection behind compressed-tensors), #336 (safetensors index shards), #332 (GLM-5.3-Flash), #329 (exact Triton top-k/top-p), #319 (in-repo triton router), #311 (PLE n-gram table disk stream), #310 (nightly kernel-cache wheel fix), #243 (`_scaled_mm` sm_89 stall avoidance, baseline commit), plus #257 (Qwen3.8-Flash-Next support) and the foundational #1 / #2 / #3 / #4 / #5 / #6 / #7 / #25 / #75 / #117. Kernel-cache wheel unchanged from upstream. Upstream LICENSE (Apache-2.0) and copyright preserved in `package/`. Third-party checkpoint Qwen3.8-Flash-Next-ABLITERATED-NVFP4 downloaded at runtime from its HF hub.
