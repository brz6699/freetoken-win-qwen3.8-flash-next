# FreeToken — Windows single-GPU build

Chinese version: [README_BRANCH_zh.md](README_BRANCH_zh.md).

A patched branch of [FreeToken](https://github.com/FlashML-org/FreeToken) (base `0.1.2+g816c324d0`, upstream main `58f4b9e`) tuned for **one consumer GPU + lots of system RAM** — e.g. RTX 5090 32 GB with ~254 GiB RAM — serving large MoE checkpoints through an OpenAI-/Anthropic-compatible HTTP API. Upstream Apache-2.0 LICENSE preserved; see [Provenance](#provenance).

This README lists what this branch added on top of upstream `58f4b9e` plus a short tuning note for upstream-provided knobs. Each feature follows the same shape: **(1) Problem observed** on this stack, **(2) Mechanism implemented**, **(3) Measured verification**. Reference rig for every number: RTX 5090 SM120 PCIe-bound + Qwen3.8-Flash-Next(-ABLITERATED)-NVFP4.

---

## Feature points

### 1. Verify-phase statistics exclusion in MoE expert LRU

**(1)** MTP verify batch carries a K+1-row distribution that differs from decode-phase routing. Mixing those counts into hot-expert LRU promotes experts that only verify needs, evicting real decode working set. Upstream's recurrence-window admission doesn't separate verify from decode phases.

**(2)** `MtpExpertBank` and the `OffloadMoeCache` chain (`load_experts` / `ensure_experts` / `ensure_experts_hybrid` / `prefetch_decode_layer`) thread a `record_stats: bool` kwarg driven by `ctx.batch.moe_force_decode` from scheduler. Verify admissions update hit/miss counters; they do **not** promote into LRU. Decode admissions do both.

**(3)** Expert residency tracks decode demand without verify-phase noise; bs=8 mixed-decode aggregate ~187/~193 tok/s cross rounds. bs=4→bs=8 scaling ×1.85 — within PCIe BW ceiling, no LRU-thrash dip.

---

### 2. turbo4 KV storage tier + TurboQuant packed-nibble rotspace

**(1)** bf16 paged KV eats too much VRAM for bs=1 long-context on a PCIe-bound stack. Existing fp8_e4m3 tier still leaves ~50% headroom unused at 1M-token contexts. Need a fourth tier with cosine ≥0.98 vs bf16 that fits longest contexts inside ~6.8 GiB pool.

**(2)** Three tiers measured against bf16 reference attention (max-cosine over sparse + split-k paths):

| tier | bytes / token / KV layer | cosine vs bf16 | use when |
|---|---|---|---|
| `bf16` | 2048 | 1.0000 | default; smallest contexts |
| `fp8_e4m3` | 1024 | 0.9994 | general balance point |
| `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

Storage compressed; compute stays bf16. Indexer keys always bf16. `kvcache/turboquant.py` implements packed-nibble rotspace dequant with bf16 per-(token, head, slab) scales. `kvcache/dsv4_cost_model.py` prices the per-token byte cost including the scale overhead. Engine gate keeps turbo4 on QSA + MLA-latent families only.

**(3)** bs=1 holds ~50k-token context inside ~3 GiB KV pool, verify peak fits without OOM. Needle retrieval 6/6 exact across 322k/450k × depth {30%, 60%, 90%}, worst-case depth-90% at 35.2s, rest 5–12s. Measured KV ≈6.9 KiB/token → full 1M-token pool ≈6.8 GiB. TTFT @ 901k context: 32.5s with prefill overlap, 51s without. Steady-state decode length-independent.

---

### 3. QSA sparse attention extended for turbo4 packed-nibble rotspace

**(1)** Upstream `attention/qsa_sparse.py` assumes bf16-packed KV layout for the block-top-k scatter/gather path. Adding turbo4 tier requires the indexer's block_topk reads and the attention-row writes to handle packed nibbles + per-token bf16 scales without an extra dequant pass.

**(2)** `qsa_block_topk` Triton kernel updated to read packed-nibble blocks with co-loaded bf16 scale row, scatter writes go through `index_copy_` in-place so a single launch covers both K and V slab. Upstream file is 542 lines; ours extends to 647 with turbo4 branch + cost-model integration with `dsv4_cost_model.py`. Compute path itself stays bf16 — only storage layout changes.

**(3)** bs=1 ~50k-token context fits ~3 GiB KV pool. Block_topk overhead vs bf16 within measurement noise; cosine vs bf16 reference 0.9887 across 24 sanity legs.

---

### 4. BudgetSlider runtime rebalancing

**(1)** Upstream `engine/cache_budget.py` does startup-only sizing. Fixed KV/MoE pools sized for one traffic mix leave headroom wasted when mix shifts within hours. Long-lived agent traffic alternates between short high-rate bursts and long idle stretches; restart is unacceptable on this stack.

**(2)** `engine/budget_slide.py` adds a `BudgetSlider` class that re-derives KV pages ↔ MoE slots at idle transitions. Asymmetric EMA: α high on rise (avoid under-provision during bursts), low on fall (avoid ping-pong when load drops). `min_step_ratio` deadband ignores sub-threshold wobble; `demand_floor` keeps shrink-only above next-request working set. `mha_pool.rebuild` / `OffloadMoeCache.rebuild` re-slice views from arena left end; base pointer never moves, CUDA graphs stay valid. Fires only at idle transitions — the only moment with both no live page-index references and no in-flight graph replay. Preserve-residency: pinned experts stay pinned.

**(3)** bs=4 mixed-decode aggregate steady ~104 tok/s cross rounds; bs=8 ~187/~193 tok/s; bs=4→bs=8 scaling ×1.85 within PCIe BW ceiling. Rebuild cadence observed: every few requests during bursts, every ~minute when idle. `budget_slide_test.py` green.

---

### 5. MM content-key sha1 prefix cache + splice invariant fix

**(1)** Repeated mm requests re-prefill identical image pixels; pad-span alignment in splice can overrun COW buffer when prefix matches land inside image_pad span. Production crash regression observed.

**(2)** Tokenizer worker hashes image pixel payload (sha1) → content id; mm prefix matching/insertion routes through `mm_cache_key` (image_pad spans replaced by content id) — same image → KV reuse, different image → zero false hits. `match_req` for prefixes landing inside pad span falls back to `align_down(span_start, page_size)` and drops GDN snapshot (prevents COW overrun); matches past last pad pass through unchanged. `qwen4_exp` / `gemma4` splice at `n_pads==0` is legal no-op; intermediate states still raise.

**(3)** Content-key reuse 13/13; adversarial cross-image isolation 174/174 green. `p4_splice_test.py` 21 legs green including production crash regression: track@960 crossing middle-span → align_down=8 fallback, GDN dropped, 955 pads all-in-row restored. Restart re-verify: 5 suites + 174 adversarial cases all green; three changes still in packaged tree.

---

### 6. MM video path + dual-pad spelling fix

**(1)** Different checkpoints write either `image_token_id` or `video_token_id` (both wrapped inside vision_start/end). PIL without ffmpeg backend can't read mp4 containers. Upstream doesn't detect both spellings and silently drops video legs.

**(2)** Detect both spellings in prepare; expansion uniformly writes `image_token_id`, so model-side embed mask / cache key / scheduler stay unchanged; `vision_geometry` passes `video_token_id` through. Video parts accept frame lists (data:/http/local paths) or single animated files (GIF/WebP/APNG); ≤8 frames sampled evenly; multi-frame uses processor `videos=` temporal pairing (falls back to per-frame stacking) with `t>1` grid; expand count comes from real grid, still renders exactly one pad placeholder. mp4 fallback via `imageio` (bundles static `imageio-ffmpeg`): read whole segment, sample ≤8 frames. Past 32768-patch budget the guard returns clean 400 `context_length_exceeded`, streaming carries it as in-stream error frame.

**(3)** `serve_video_smoke.py` / `vision_video_test.py` / `vision_p3_media_test.py` green vs HF oracle bit-match. Live serve (:8001) smoke: frame-list / GIF / image_url+video mixed legs all correct (static green ∉ animated {red, blue}). Measured prompt_tokens=238 on a representative reasoning leg; reasoning leg budget ≥512. Image requests vs text: cold TTFT medians text 1518ms vs image 1526ms (+8ms); ViT forward not dominant — ~1.5s floor is fixed MoE expert-cache cold-start which text pays too. Large image (19200 patches) peaks VRAM +984 MiB, first-token latency 2.9–4.2s.

---

### 7. MM RAM tier swap-in/out

**(1)** Same mm content arriving hours later pays full ViT + prefill cost from cold KV. Pinned-budget arena on Windows has limited VRAM, so pure VRAM-resident mm cache can't survive hours-long gaps. Upstream has no mm-tier mechanism.

**(2)** Completed mm requests stash prompt-segment KV + GDN state to host RAM. Subsequent same-content-key requests restore rows into freshly allocated VRAM pages and continue directly. `kvcache/mha_pool.py` / `qsa_pool.py`: `snapshot_rows` / `restore_rows` pack K/V compressed blocks + turbo4 dequant scales + QSA compressed-index rows (`slot // index_ratio`) to CPU; restore writes back via `index_copy_` in-place scatter (`[:, idx].copy_` would gather into a temp first and silently fail to write — known trap; documented inline). `pending_ring` transient, not tiered. `linear_state_pool.snapshot_state` / `restore_state` pack conv+recurrent+slot_states whole-slot. Tier storage uses LRU-bytes eviction with `FREETOKEN_MM_TIER_MB` budget (default 1024). Restore gate: content-key bit-match + boundary beats tree match + doesn't cut a pad run → allocate new pages + row refill + `_TierHandle`. Restore integrated into admission after budget gate; `mm_gdn` propagated through PendingReq→Req; `_restore_linear_states` writes back on first prefill forward.

**(3)** `kv_tier_test.py` green on real QSA+turbo4 pools — row round-trip, byte accounting, same-key replacement, LRU eviction, pad-boundary GDN write-back. Vision two-suite regression green. Live serve (:8001) smoke: same-content repeat takes restore path and answers correctly; image_url+video mixed legs judged correctly; prompt_tokens=238 measured. Warm TTFT below cold TTFT confirms tier hits.

---

## Tuning notes for upstream-provided knobs

The mechanisms below exist upstream and this branch adopts them unchanged but tuned for PCIe-bound single-GPU stacks.

- **Shared cudaMemPool CUDA-graph ladder** — `engine/graph.py` already shares the cudaMemPool + graph handle across bs entries upstream. For bs-histogram ≥95% bs=1 traffic (`/v1/stats` `bs_hist`), `--cuda-graph-max-bs 1` collapses the ladder to single bs=1 capture; replay pads up to next entry. bs=1-dominated Windows stacks see ~7.5 GiB contiguous free after PyTorch allocator overhead with bs=1 ladder.
- **bs-sized transport routing** — `hybrid_fetch_fraction` (0–1 ratio read once from `ft bench bw`) replaces fixed `hybrid_max_fetch` count. Each decode step fetches `fraction × misses` experts; PCIe stays saturated and CPU GEMM overlaps across bs=1…8. Profile read at engine init only — mid-decode re-benching jitters TTFT.
- **Recurrence-window MoE admission** — `expert_recency` recurrence window gates expert fetch on second sighting; first misses compute CPU-side. Already in upstream `moe/offload_cache.py`. Combined with verify-phase stats exclusion above to keep hot LRU tracking decode demand.
- **bs≥4 decode-step prefetch** — `_decode_hybrid` issues `prefetch_decode_layer(next_layer, hint)` before current-layer GEMM dispatch when bs≥4. Tests in `tests/moe/test_decode_prefetch_gate.py`. Below bs=4 routing entropy exceeds prefetch-hit benefit, gate skips.
- **KV hot-window** — `FREETOKEN_HOT_WINDOW_TOKENS` caps the `demand_tokens` signal so KV grows to soft ceiling instead of pure shrink-only. Opt-in; unset preserves shrink-only exactly. Sized for multi-turn agent prompts where hard shrink-only floor evicts still-useful prefix pages.
- **Prefill/decode interleaving** — `_interleave_every` (default 4) consecutive prefill chunks force one decode step. Counted in chunks not seconds for deterministic replay. Override via `prefill_interleave_every`.
- **Special-token ckpt** — `--enable-special-token-ckpt` snapshots GDN state at tool-call opener token. Structural boundary: pre-token content identical across turns, post-token drifts. Snapshot paid once per turn, prefix reuse hit on every echo. Already upstream args.py.

---

## Bench findings

Two complementary bench regimes on this rig. Each isolates a different operating point — neither is "the answer" for the other. Both reflect the current packaged state with all exp mechanisms active by default.

| regime | what it measures | output length | prompt length |
|---|---|---|---|
| Short-output ladder | PCIe BW ceiling + per-step launch overhead; aggregate soft cap independent of prefill | `max_tokens=96` | short |
| Mixed-decode lc_mixed | steady-state agent workload with prefill overlap + warm prefix cache; aggregate after amortization kicks in | per-stream ∈ {1k, 2k, 3k, 4k} | ~10k / ~50k shared-prefix |

Same hardware, same serve profile (`--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --moe-prefill-hit-d2d`), but lc_mixed aggregate sits higher than ladder aggregate at identical bs because long outputs amortize fixed per-step overheads (launch, cache rebuild check, scheduler tick) over more decoded tokens.

**Ladder** — `max_tokens=96`, text-only:

| concurrency | aggregate tok/s | per-stream tok/s | notes |
|---|---|---|---|
| 1 | 26.2 | 26.2 | |
| 2 | 47.5 | ~24 | |
| 4 | 66.9 | 17.2 | p50 ≈5.0s, p95 ≈5.2s — sweet spot |
| 6 | 64.6 | ~11 | |
| 8 | 68.8 | 9.9 | p95 climbs back |

Aggregate soft ceiling ≈65–77 tok/s — set by PCIe BW + per-step expert-fetch volume, not compute. Adding `--max-running-requests 8` lets bs=6 fill the captured bs=8 graph and lifts c6 aggregate to ~76.9 at cost of p95 ≈7s (c8 tail ≈25s) — throughput-first only.

**lc_mixed** — `bench_lc_mixed.py`, MTP off, `--cuda-graph-max-bs 1`, two prompt tiers × bs ∈ {4, 8}, two rounds each:

| bs | prompt tier | aggregate tok/s | per-stream avg tok/s | TTFT cold → warm |
|---|---|---|---|---|
| 4 | ~10k | ~104 | ~25 | 3.2s → 2.4s |
| 4 | ~50k | ~104 | ~25 | 10.1s → 2.6s |
| 8 | ~10k | ~193 | ~24 | 46.6s → 46.8s |
| 8 | ~50k | ~187 | ~24 | 47.7s → 48.3s |

Three observations: per-stream decode ~25 tok/s is length-independent up to ~50k context (KV read BW not yet bottleneck — weight streaming dominates); bs=4→bs=8 scaling ×1.85 within PCIe BW ceiling; bs=8 TTFT ~47s reflects chunked-prefill queueing across 8 streams × XLONG prompts × ~8192-token chunks, not decode slowdown. Round-to-round TTFT drop confirms warm prefix cache hits via mm content-key path.

**MoE cache size vs decode speed** (`ft ctl cache rebuild` live resize, no restart). Slots 1024 → 2048 → 3072 give long-text steady-state decode ≈62 → 75 → 84 char/s (≈33 → 40 → 44 tok/s; short text ≈52). Cumulative **+36% with zero quality loss** — cache stores exact weights, gains come purely from hit rate. Past ~3072 curve flattens (+5–8% but eats KV pool).

---

## Serve profiles

The launcher in `_exp_garble/run_exp.py` is a thin shim that prepends the exp package dir to `sys.path` so `import freetoken` resolves to the exp tree, then forwards CLI flags unchanged. Equivalent to `ft serve` against the same package. Paths below are illustrative — substitute your own venv Python, launcher path, and model dir.

Default profile:

```powershell
python <launcher>.py `
  --model-path ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 2048 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d `
  --quant-backend moe.nvfp4=b12x --moe-prefill-demand-tokens 64 `
  --max-running-req 4 `
  --host 0.0.0.0 --port 8001
```

This is the 1M-token longest-context profile with b12x Triton backend (feature point 3) and bs=4 cap. Budget slide is enabled; encoder towers load by default; CUDA-graph ladder is auto-tuned from VRAM.

Tighter bs=1-dominated profile (verified separately, useful when bs_hist ≥95% bs=1 in `/v1/stats`):

```powershell
python <launcher>.py `
  --model-path ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 65536 --moe-cache-size 1024 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d `
  --quant-backend moe.nvfp4=b12x --moe-prefill-demand-tokens 64 `
  --cuda-graph-max-bs 1 --mtp-expert-quant q4_0 --text-model-only `
  --host 0.0.0.0 --port 8001
```

Flags worth understanding:

- `--quant-backend moe.nvfp4=b12x` — select the b12x Triton backend for NVFP4 MoE decode (feature point 3). Without it the auto-selector may pick `marlin` on SM80/99 or fall through to torch `_scaled_mm`. On SM120 with this checkpoint, `b12x` is the right choice; pair with `--moe-prefill-hit-d2d` so prefill hits reuse GPU-side copies. Accepts per-layer overrides: `linear=marlin,moe=b12x`.
- `--num-tokens` — paged KV pool size. turbo4 ≈6.9 KiB/token → ≈6.8 GiB full pool at 1M, ≈3.4 GiB at 512k, ≈0.4 GiB at 65k. Smaller pool leaves contiguous VRAM headroom for verify peak.
- `--moe-cache-size` — experts resident on GPU. Steady-state decode scales roughly with hit rate up to ~3072, then flattens.
- `--kv-cache-dtype` — see tier table under feature point 2.
- `--max-running-req N` — scheduler concurrency cap. Argparse prefix-matches `--max-running-requests`; both spellings work. Default lets bs grow up to ladder cap; `4` keeps the bs=4 graph hot.
- `--text-model-only` — skip all encoder towers; their VRAM goes to KV / expert caches. Use `--mm-disable vision audio` instead when only some kinds should be skipped. Vision / audio towers load by default and skip cleanly when disabled. Upstream-provided flags.
- `--moe-prefill-hit-d2d` — serves prefill expert hits via device-to-device copies. Upstream-provided.
- `--cuda-graph-max-bs N` — CUDA-graph ladder cap; `1` for bs=1-dominated, `4` adds bs=4 burst slot cheaply under shared mempool. Upstream-provided flag.
- `--moe-prefill-demand-tokens` — chunk-row threshold for demand-prefill (feature point 5); default `64` fits cached-prefix incremental prefill. Upstream-provided flag.
- `--mtp-expert-quant q4_0` — shrink the MTP draft expert bank to q4_0 to fit verify peak inside tight VRAM. Use with `--mtp-device gpu` to actually enable MTP; default-off on this branch (see tuning note under feature point 1).
- `--disable-budget-slide` — opt-out of the BudgetSlider from feature point 4, freezing KV/MoE pool sizes from startup. Use when the workload is single-model fixed-footprint and you want CUDA-graph captures pinned to stable addresses.

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

<a name="provenance"></a>

Derived from upstream FreeToken commit `58f4b9e` (release `0.1.2+g816c324d0`). Upstream PRs merged in: #377, #375, #343, #336, #332, #329, #319, #311, #310, #243, #257, plus foundational #1 / #2 / #3 / #4 / #5 / #6 / #7 / #25 / #75 / #117. Kernel-cache wheel unchanged from upstream. Upstream LICENSE (Apache-2.0) and copyright preserved in `package/`. Third-party checkpoint Qwen3.8-Flash-Next-ABLITERATED-NVFP4 downloaded at runtime from its HF hub.
