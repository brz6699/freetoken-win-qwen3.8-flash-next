# FreeToken — Windows single-GPU build

Chinese version: [README_BRANCH_zh.md](README_BRANCH_zh.md).

A patched branch of [FreeToken](https://github.com/FlashML-org/FreeToken) (base: `0.1.2+g816c324d0`, upstream main `58f4b9e`) tuned for **one consumer GPU + lots of system RAM** — e.g. RTX 5090 32 GB with ~250 GB RAM — serving large MoE checkpoints through an OpenAI-/Anthropic-compatible HTTP API.

Upstream is Apache-2.0; this branch keeps its LICENSE and copyright intact. See [Provenance](#provenance).

## Why this branch

Upstream FreeToken already targets MoE-offload inference, but its Windows path and KV-storage options were thin. This branch is the experimental line where those gaps get closed and measured against live traffic. The chain of reasoning: tiered KV storage (turbo4 ≈ 4× smaller per token than bf16) frees GPU-side room for the MoE expert LRU; a larger expert cache means more hot experts stay on-GPU; the multimodal content-key prefix cache keeps repeated images/videos from being re-prefilled; CUDA-graph capture is shaped around observed batch histograms to avoid dead-slot workspace; and MTP self-speculative decoding is opt-in with verify-phase statistics properly excluded from hot-expert LRU. Concretely this branch adds or stabilizes:

- **Windows compatibility** — working-set / pin-budget handling under WDDM, `ctypes`-based page locking (`VirtualLock`) replacing the POSIX-only `resource` module, graceful degradation paths for `expandable_segments` and the zmq/Proactor event-loop differences.
- **Split residency for expert banks** — when the CUDA pin budget is smaller than the expert weights, head/tail MoE layers stay OS-locked in RAM and decode on a multi-threaded CPU executor (AVX-512 BF16 + VNNI, NVFP4 W4A8), while remaining banks pin for GPU streaming. Tunable via `FREETOKEN_PIN_BUDGET_GB`.
- **Storage-side KV quantization for paged pools** (QSA and MLA/DSA families): three tiers measured against bf16 reference attention (max-cosine over sparse + split-k paths):

  | tier | bytes / token / KV layer | cosine vs bf16 | use when |
  |---|---|---|---|
  | `bf16` | 2048 | 1.0000 | default; smallest contexts |
  | `fp8_e4m3` | 1024 | 0.9994 | general balance point |
  | `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

  Storage compressed; compute stays bf16. Indexer keys always bf16.
- **Multimodal support by default** — vision / audio encoder towers load unless disabled, image requests reuse KV via content-hashed prefix keys (same image → KV reuse, different image → zero false hits), mm KV pages live in a RAM tier swapping against GPU pool. Use `--text-model-only` to skip all encoders and reclaim their VRAM for KV / expert caches; use `--mm-disable vision` (or `audio`) to skip only specific kinds.
- **Qwen3.8-Flash-Next serving profile** — QSA paged-KV pool with tiered storage above, NVFP4 routed experts via offload cache, mm content-key prefix cache exercised end-to-end by `verification/`.

## Experimental mechanisms

These are the moving parts tuned specifically for this branch's hardware target. Each ties to a specific bottleneck the live bench exposes.

**Shared CUDA-graph mempool with fixed-shape ladder.** `engine/graph.py` captures every bs in one shared `cudaMemPool` plus one shared `GraphCaptureBuffer` sized to `max_graph_bs`. Ladder entries share workspace instead of stacking it, so adding entries costs only topology metadata (~few MB each), not another workspace. Ladder is computed from `--cuda-graph-max-bs` via `[1, 2, 4] + range(8, max_bs+1, 8)`; for bs=1-dominated traffic (verify via `/v1/stats` `bs_hist`) `--cuda-graph-max-bs 1` collapses the ladder to a single bs=1 capture, removing dead-slot overhead without a quality hit. Replay pads up to the next ladder entry.

**Verify-phase statistics exclusion.** MTP verify batches carry a K+1-row distribution that would skew the hot-expert LRU if mixed into decode-phase routing counts. `MtpExpertBank` and the `OffloadMoeCache` chain (`load_experts` / `ensure_experts` / `ensure_experts_hybrid` / `prefetch_decode_layer`) thread a `record_stats: bool` kwarg, set from `ctx.batch.moe_force_decode` in scheduler. Verify admissions update hit/miss counters; they do not promote experts into the LRU. Result: expert residency reflects real decode demand, not verify-phase noise.

**b12x Triton backend for NVFP4 MoE decode.** `kernel/triton/nvfp4_fused_moe.py` is marlin-style: wide int32 loads (8 codes/word), deferred K reduction, register-only `_split8` shuffles, fp32 accumulate, per-route grid `(num_routes, cdiv(N, BLOCK_N))`. Hits ~87% PCIe BW ceiling on RTX 5090; residual gap is FP4-dequant ALU. The block-scale epilogue folds into fp16 bits with global-scale applied after the K-loop and arrival-counter split-K reduction — single-launch. This covers what DeepGEMM's tiled cluster kernels offer for our shape range, so we don't pull DeepGEMM in.

**Hybrid QSA sparse attention + turbo4 KV.** QSA paged KV pool stores in turbo4 packed nibble rotspace; compute stays bf16, indexer keys bf16. Block top-k path uses Triton `qsa_block_topk`. This combo lets bs=1 hold ~50k-token contexts inside ~3 GiB KV pool without OOM during verify peak.

**Moe prefill demand threshold.** `--moe-prefill-demand-tokens` gates expert-fetch on chunk row count (`hidden_states.shape[0]`), not whole prompt length. With cached-prefix incremental prefills the new-tokens-per-step is small but cumulative context is huge; the threshold matches chunk size, so cached-prefix hits skip the demand-prefill path while cold starts pay it once. Default `64`.

**Budget slide for runtime KV/expert rebalancing.** `engine/budget_slide.py` re-derives KV pool pages and MoE cache slots from current traffic without restarting the engine or reloading weights. Observed trigger cadence: every few requests during traffic bursts, every ~minute when idle. Rebuild is preserve-residency (existing expert weights stay pinned); only pool sizes change.

## Requirements

- Windows 11 (or WSL2) x64, NVIDIA GPU ≥ 24 GB VRAM, ≥ 2× model-size RAM recommended.
- Python 3.12 (shipped wheels are `cp312`; kernel-cache wheel built for CUDA 13.x).

## Install

Source-only; dependencies come from public indexes.

```powershell
python -m venv ft-venv
ft-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
ft-venv\Scripts\pip install .
```

### Dependencies

- `torch` (CUDA 13 build, `cp312`) — PyTorch official index: <https://download.pytorch.org/whl/cu130>
- `freetoken` — installed from this repository (`pip install .`); not yet on PyPI.
- Optional prebuilt kernel-cache wheel (skips Triton JIT on first request; must match CUDA 13.x) — attached to GitHub Releases.
- Optional `imageio` (bundles static `imageio-ffmpeg`) for mp4 container support on vision path.

## Serve

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
- `--kv-cache-dtype` — see tier table above.
- `--text-model-only` — skip all encoder towers; their VRAM goes to KV / expert caches. Use `--mm-disable vision audio` instead when only some kinds should be skipped.
- `--moe-prefill-hit-d2d` — serves prefill expert hits via device-to-device copies.
- `--cuda-graph-max-bs N` — CUDA-graph ladder cap; `1` for bs=1-dominated, `4` adds bs=4 burst slot cheaply under shared mempool, leave unset for auto-tune by VRAM.
- `--moe-prefill-demand-tokens` — chunk-row threshold for demand-prefill; default 64 fits cached-prefix incremental prefill.

Then hit `POST /v1/chat/completions` like any OpenAI-compatible server.

## Measured numbers

Reference rig: RTX 5090 32 GB + ~254 GiB RAM, Qwen3.8-Flash-Next(-ABLITERATED)-NVFP4.

**Concurrency ladder** (`max_tokens=96`, text-only, `--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --moe-prefill-hit-d2d`):

| concurrency | aggregate tok/s | per-stream tok/s | notes |
|---|---|---|---|
| 1 | 26.2 | 26.2 | |
| 2 | 47.5 | ~24 | |
| 4 | 66.9 | 17.2 | p50 ≈ 5.0s, p95 5.2s — sweet spot |
| 6 | 64.6 | ~11 | |
| 8 | 68.8 | 9.9 | p95 climbs back |

Sweet spot is concurrency 4: batching gain 1→4 is real (+155%), 4→8 slices same throughput finer. Aggregate soft ceiling ≈65–77 tok/s set by PCIe bandwidth and per-step expert-fetch volume, not compute. Adding `--max-running-requests 8` lets bs=6 fill the captured bs=8 graph and lifts c6 aggregate to 76.9 at cost of p95 ≈7s (c8 tail ≈25s) — throughput-first only.

**MoE cache vs decode speed** (`ft ctl cache rebuild`, no restart): slots 1024 → 2048 → 3072 give long-text steady-state decode ≈62 → 75 → 84 char/s (≈33 → 40 → 44 tok/s; short text ≈52). Cumulative **+36% with zero quality loss** — cache stores exact weights, gains come from hit rate alone. Past ~3072 curve flattens (+5–8% but eats KV pool).

**Long context with mixed-decode lengths** (`bench_lc_mixed.py`, prompts ~10k/~50k tokens shared-prefix, per-stream output targets ∈ {1024, 2048, 3072, 4096}, MTP off, `--cuda-graph-max-bs 1`):

| bs | prompt tier | aggregate tok/s | per-stream avg tok/s | TTFT cold → warm |
|---|---|---|---|---|
| 4 | ~10k | ~104 | ~25 | 3.2s → 2.4s |
| 4 | ~50k | ~104 | ~25 | 10.1s → 2.6s |
| 8 | ~10k | ~193 | ~24 | 46.6s → 46.8s |
| 8 | ~50k | ~187 | ~24 | 47.7s → 48.3s |

Three observations: per-stream decode ~25 tok/s is **length-independent** up to ~50k context (KV read BW not yet bottleneck — weight streaming dominates); aggregate scales bs=4→bs=8 by ×1.85, hitting PCIe BW ceiling; bs=8 TTFT ~47s reflects chunked-prefill queueing (8 streams × XLONG prompts × ~8192-token chunks), not decode slowdown. Round-to-round TTFT drop is warm prefix-cache effect.

**Vision** (default-on, vision tower bf16 ≈0.84 GiB): image requests cost **+8ms** over text (cold TTFT medians: text 1518ms vs image 1526ms). ViT forward is not dominant — the ~1.5s floor is fixed MoE expert-cache cold-start which text pays too. Large image (19200 patches) peaks VRAM +984 MiB with first token 2.9–4.2s. Repeated same-content requests refill from mm RAM tier; different images get zero false hits (content keys 13/13, adversarial 174/174 green). Past 32768 patches the guard returns clean 400 (`context_length_exceeded`); streaming carries it as in-stream error frame.

**Long-context retrieval** (turbo4 tier): needle 6/6 exact across 322k/450k × depth {30%, 60%, 90%}, worst-case depth-90% at 35.2s, rest 5–12s. KV ≈6.9 KiB/token measured → full 1M-token pool ≈6.8 GiB; TTFT at 901k context 32.5s with prefill overlap (51s without). Steady-state decode length-independent.

## Tuning notes

- **Pin ceiling**: WDDM caps GPU-accessible pinned host memory at roughly half physical RAM; locked working-set pages count against same budget. Keep `FREETOKEN_PIN_BUDGET_GB` below ceiling minus locked-layer share; raise in steps after freeing other RAM consumers.
- **Locked ≠ pinned**: locked banks (head/tail layers) decode on CPU executor; pinned banks stream to GPU. More pinned = faster decode; locking is fallback when pinned would exceed ceiling.
- Startup OOM on last bank: lower `--num-tokens` first (turbo4 ≈6.9 KiB/token), then the budget.
- **Verify-phase peak sizing**: at bs=4 with MTP `--mtp-draft-len 2`, verify peak ≈3–4 GiB contiguous. With shared CUDA-graph mempool + bs=1 ladder + `--num-tokens 65536 --moe-cache-size 1024 --mtp-expert-quant q4_0`, steady-state sits ≈14 GiB allocated, ≈16 GiB contiguous free — comfortably fits verify. PyTorch reports less free than `nvidia-smi` because CUDA driver context + cuBLASLt workspace + cuDNN pools + CUDA-graph private pools sit outside PyTorch allocator visibility (typically ~3 GiB gap on this stack). `torch.cuda.empty_cache()` alone recovers that gap only at startup; runtime peak sizing must come from shrinking steady-state allocated.
- **MTP is opt-in** — measured acceptance ~0.70 with `--mtp-draft-len 2` gives net-positive on multi-stream aggregate but net-negative on bs=1-dominated traffic where verify overhead (~70ms/step) + redraft (~30ms) exceeds savings from skipping forward passes. Default-off on this branch; enable with `--mtp-device gpu` when accepting longer per-token latency in exchange for higher aggregate throughput on multi-stream workloads.

## Verification suite

`verification/` holds regression tests gating each stage (all green on packaged state):

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

Encoder toggling is by CLI flag (`--text-model-only`, `--mm-disable ...`); vision / audio towers load by default and skip cleanly when disabled.

## Provenance

Derived from upstream FreeToken commit `58f4b9e` (release `0.1.2+g816c324d0`). Windows / quantization / mm changes documented per-area in `CHANGELOG.md` and round logs; kernel-cache wheel unchanged from upstream. Upstream LICENSE (Apache-2.0) and copyright preserved in `package/`.

Third-party checkpoint Qwen3.8-Flash-Next-ABLITERATED-NVFP4 not part of this repo; downloaded at runtime from its Hugging Face hub.
