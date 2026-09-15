# FreeToken — Windows single-GPU build

Chinese version: [README_BRANCH_zh.md](README_BRANCH_zh.md).

A patched branch of [FreeToken](https://github.com/FlashML-org/FreeToken) (base: `0.1.2+g816c324d0`, upstream main `58f4b9e`) tuned for **one consumer GPU + lots of system RAM** — e.g. RTX 5090 32 GB with ~250 GB RAM — serving large MoE checkpoints through an OpenAI-/Anthropic-compatible HTTP API.

Upstream is Apache-2.0; this branch keeps its LICENSE and copyright intact. See [Provenance](#provenance).

## Why this branch

Upstream FreeToken already targets MoE-offload inference, but its Windows path and KV-storage options were thin. The problems this branch solves, in one chain: tiered KV storage (turbo4 ≈ 4× smaller per token than bf16) frees GPU-side room for the MoE expert LRU; a larger expert cache means more hot experts stay on-GPU; and with `--vision-on`, the content-keyed mm tier keeps repeated images/videos from being re-prefilled, so one 32 GB card holds a large MoE, 100k+ tokens of context, and a rich multimodal history at the same time. Concretely, this branch adds or stabilizes:

- **Windows compatibility** — working-set/pin-budget handling under WDDM, `ctypes`-based page locking (`VirtualLock`) replacing the POSIX-only `resource` module, graceful degradation paths for `expandable_segments` and the zmq/Proactor event-loop differences.
- **Split residency for expert banks** — when the CUDA pin budget is smaller than the expert weights, head/tail MoE layers stay OS-locked in RAM and decode on a multi-threaded CPU executor (AVX-512 BF16 + VNNI, NVFP4 W4A8), while the remaining banks are pinned for GPU streaming. Budget is tunable via `FREETOKEN_PIN_BUDGET_GB`.
- **Storage-side KV quantization for the paged pools** (QSA and MLA/DSA families): three tiers, measured on a round-trip against bf16 reference attention (max-cosine over sparse + split-k paths):

  | tier | bytes / token / KV layer | cosine vs bf16 | use when |
  |---|---|---|---|
  | `bf16` | 2048 | 1.0000 | default; smallest contexts |
  | `fp8_e4m3` | 1024 | 0.9994 | general balance point |
  | `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

  Storage is compressed; attention compute stays bf16. Indexer keys always stay bf16.
- **Multimodal (vision) support** behind an explicit `--vision-on` flag: vision tower loads only when requested; image requests reuse KV through content-hashed prefix keys (same image → KV reuse, different image → zero false hits), and the mm KV pages live in a RAM tier that swaps in/out against the GPU pool.
- **Qwen3.8-Flash-Next serving profile** — QSA paged-KV pool with the tiered storage above, NVFP4 routed experts via the offload cache, and the mm content-key prefix cache exercised end-to-end by the `verification/` suite.

## Requirements

- Windows 11 (or WSL2) x64, NVIDIA GPU ≥ 24 GB VRAM, ≥ 2× model-size RAM recommended.
- Python 3.12 (the shipped wheels are `cp312`; the kernel-cache wheel is built for CUDA 13.x).

## Install

This repository is source-only; dependencies come from public indexes.

```powershell
python -m venv ft-venv
ft-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
ft-venv\Scripts\pip install .
```

### Dependencies

- `torch` (CUDA 13 build, `cp312`) — PyTorch official index: <https://download.pytorch.org/whl/cu130>
- `freetoken` — installed from this repository (`pip install .`); not published to PyPI yet.
- Optional prebuilt kernel-cache wheel (skips Triton JIT on the first request; must match CUDA 13.x) — attached to GitHub Releases for this repository.
- Optional: `imageio` (bundles a static `imageio-ffmpeg`) for mp4 container support on the vision path.

## Serve

```powershell
$env:FREETOKEN_PIN_BUDGET_GB = "120"   # ~half of physical RAM is the WDDM pin ceiling

# turbo4 + 1M KV + vision — longest-context profile
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 2048 `
  --kv-cache-dtype turbo4 --vision-on --moe-prefill-hit-d2d

# turbo4 + 512K KV + vision — larger resident expert cache (3072)
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 524288 --moe-cache-size 3072 `
  --kv-cache-dtype turbo4 --vision-on --moe-prefill-hit-d2d
```

- `--num-tokens` — paged KV pool size; 1048576 gives the 1M-token window (turbo4 keeps it under ~3 GiB), 524288 trades window for a bigger expert cache.
- `--moe-cache-size` — number of experts resident on-GPU (2048 / 3072 above).
- `--kv-cache-dtype` — see tier table above.
- `--vision-on` — loads the vision tower (~1 GiB bf16); omit for text-only serving.
- `--moe-prefill-hit-d2d` — serves prefill expert hits via device-to-device copies.

Then hit `POST /v1/chat/completions` as with any OpenAI-compatible server.

## Measured numbers

All figures below are measured on the reference rig (RTX 5090 32 GB + ~254 GiB RAM, Qwen3.8-Flash-Next-NVFP4, launched with `--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --vision-on --moe-prefill-hit-d2d`). Reproduce with a ThreadPoolExecutor ladder script, text legs at `max_tokens=96`, concurrency 1/2/4/6/8.

Concurrency ladder (text, `max_tokens=96`):

| concurrency | aggregate tok/s | per-stream tok/s | latency |
|---|---|---|---|
| 1 | 26.2 | 26.2 | — |
| 2 | 47.5 | ~24 | — |
| 4 | 66.9 | 17.2 | p50 ≈ 5.0s, p95 5.2s |
| 6 | 64.6 | ~11 | — |
| 8 | 68.8 | 9.9 | p95 climbs back |

The sweet spot is **concurrency 4**: the batching gain from 1→4 is real (+155%), while 4→8 just slices the same throughput finer (aggregate +3%, per-stream halved). The aggregate has a soft ceiling ≈65–77 tok/s set by PCIe bandwidth and per-step expert fetch volume — not compute — so raising it needs more bandwidth, not more concurrency. Adding `--max-running-requests 8` lets bs=6 fill the captured bs=8 graph and lifts the c6 aggregate to 76.9, at the cost of p95 rising to 7s (c8 tail 24.8s); use only for throughput-first scenarios.

MoE cache vs decode speed (`ft ctl cache rebuild`, no restart): slots 1024 → 2048 → 3072 give long-text steady-state decode ≈62 → 75 → 84 char/s (≈33 → 40 → 44 tok/s; short text ≈52), cumulatively **+36% with zero quality loss** — the cache stores exact weights, so every gain comes from hit rate. Past ~3072 the curve flattens (another +5–8%, but it eats into the KV pool); stop there.

Long context (turbo4 tier): needle retrieval 6/6 exact (322k/450k tokens × depth 30/60/90%, worst case depth-90% at 35.2s, the rest 5–12s, answers are verbatim random 4-digit numbers). Measured KV ≈6.9 KiB/token → the full 1M-token pool ≈6.8 GiB; TTFT at 901k context is 32.5s with prefill overlap (51s without). Steady-state decode speed is length-independent (same kernel; every length band lands in the same range).

Vision: with `--vision-on`, image requests cost only **+8ms** over text (cold TTFT medians: text 1518ms vs image 1526ms) — the ViT forward (weights ~0.84 GiB, GPU-side) is not the dominant cost; the ~1.5s floor is fixed MoE expert-cache cold-start overhead, which text pays too. A large image (19200 patches) peaks VRAM +984 MiB with first token at 2.9–4.2s. Repeated same-content requests refill from the mm RAM tier (warm TTFT below cold); different images get zero false hits (content keys 13/13 plus adversarial 174/174 green). Beyond 32768 patches the guard returns a clean 400 (`context_length_exceeded`); streaming mode carries it as an in-stream error frame.

## Tuning notes

- **Pin ceiling**: under WDDM the driver caps GPU-accessible pinned host memory at roughly half of physical RAM; locked (working-set) pages count against the same budget. Keep `FREETOKEN_PIN_BUDGET_GB` below the ceiling minus the locked-layer share, and raise it in steps after freeing other RAM consumers.
- **Locked ≠ pinned**: locked banks (head/tail layers) decode on the CPU executor; pinned banks stream to the GPU. More pinned = faster decode; locking is the fallback when pinned would exceed the ceiling.
- If startup OOMs on the last bank, lower `--num-tokens` first (turbo4 keeps the KV pool under ~3 GiB at 512k tokens), then the budget.

## Verification suite

`verification/` holds the regression tests that gate each stage (all green on the packaged state):

| script | covers |
|---|---|
| `p4_splice_test.py` | mm splice invariant (pad-span alignment, COW no-op) — 21 cases |
| `p4_finish_key_test.py` | finish-key + SWA regression legs |
| `p4_content_key_test.py` | content-hash prefix keys (same-image reuse / cross-image isolation) |
| `mm_prefill_chunk_test.py` | whole-segment mm prefill rollback |
| `r4_adv_probe.py` | adversarial fuzz: cross-span gaps, page ownership, `page_size ∈ {1,4,8}` × shapes × rounds |
| `kv_tier_test.py` | RAM-tier KV swap-in/out round-trip |
| `serve_video_smoke.py`, `vision_video_test.py`, `vision_p3_media_test.py` | end-to-end vision/media requests |

Run from the package root with the venv Python, e.g. `python verification/p4_splice_test.py`.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | cap for CUDA-pinned expert banks; exceeding it triggers split residency (locked CPU layers) | auto (~0.4–0.5 × RAM) |
| `FREETOKEN_LOAD_VISION` | `1` loads the vision tower (`--vision-on` sets it) | off |

## Provenance

Derived from upstream FreeToken commit `58f4b9e` (release `0.1.2+g816c324d0`). The Windows/quantization/mm changes are documented per-area in `CHANGELOG.md` and the round logs; the kernel cache wheel is unchanged from upstream. Upstream LICENSE (Apache-2.0) and copyright are preserved in `package/`.

The third-party checkpoint used with this build (Qwen3.8-Flash-Next-ABLITERATED-NVFP4) is not part of this repository; it is downloaded at runtime from its Hugging Face hub.
