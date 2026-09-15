# FreeToken — Windows single-GPU build

A patched branch of [FreeToken](https://github.com/FlashML-org/FreeToken) (base: `0.1.2+g816c324d0`, upstream main `58f4b9e`) tuned for **one consumer GPU + lots of system RAM** — e.g. RTX 5090 32 GB with ~250 GB RAM — serving large MoE checkpoints through an OpenAI-/Anthropic-compatible HTTP API.

Upstream is Apache-2.0; this branch keeps its LICENSE and copyright intact. See [Provenance](#provenance).

**Measured on one consumer rig** — single RTX 5090 (32 GB) + 256 GB DDR5 (4×64 GB) over PCIe 5.0, serving Qwen3.8-Flash-Next:

- **1M-token context window** via the turbo4 KV tier
- **3072 experts resident** in the on-GPU MoE cache
- **4-way concurrency as the sweet spot** — aggregate throughput peaks around 77 tokens/s at 4–8 concurrent streams; evaluated on this rig, 4-way sustains stable operation and covers most workloads
- **400 consecutive images** recognized end-to-end (content-keyed mm KV reuse)
- **40–50 tokens/s** single-stream decode

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
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model-path ./models/<checkpoint> `
  --kv-cache-dtype turbo4 --moe-cache-auto --vision-on
```

- `--kv-cache-dtype` — see tier table above.
- `--moe-cache-auto` — sizes the GPU-side expert LRU and the paged KV pool from remaining memory.
- `--vision-on` — loads the vision tower (~1 GiB bf16); omit for text-only serving.

Then hit `POST /v1/chat/completions` as with any OpenAI-compatible server.

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
