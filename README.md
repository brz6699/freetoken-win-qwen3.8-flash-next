# FreeToken — Windows single-GPU build

A patched branch of [FreeToken](https://github.com/FlashML-org/FreeToken) (base: `0.1.2+g816c324d0`, upstream main merged through `afd99cb`) tuned for **one consumer GPU + lots of system RAM** — e.g. RTX 5090 32 GB with ~250 GB RAM — serving large MoE checkpoints through an OpenAI-/Anthropic-compatible HTTP API.

Upstream is Apache-2.0; this branch keeps its LICENSE and copyright intact. See [Provenance](#provenance).

## Why this branch

Upstream FreeToken already targets MoE-offload inference, but its Windows path and KV-storage options were thin. The problems this branch solves, in one chain: tiered KV storage (turbo4 ≈ 4× smaller per token than bf16) frees GPU-side room for the MoE expert LRU; a larger expert cache means more hot experts stay on-GPU; and with vision enabled, the content-keyed mm tier keeps repeated images/videos from being re-prefilled, so one 32 GB card holds a large MoE, a 1M-token context, and a rich multimodal history at the same time. Concretely, this branch adds or stabilizes:

- **Windows compatibility** — working-set/pin-budget handling under WDDM, `ctypes`-based page locking (`VirtualLock`) replacing the POSIX-only `resource` module, graceful degradation paths for `expandable_segments` and the zmq/Proactor event-loop differences.
- **Split residency for expert banks** — when the CUDA pin budget is smaller than the expert weights, head/tail MoE layers stay OS-locked in RAM and decode on a multi-threaded CPU executor (AVX-512 BF16 + VNNI, NVFP4 W4A8), while the remaining banks are pinned for GPU streaming. Budget is tunable via `FREETOKEN_PIN_BUDGET_GB`.
- **Storage-side KV quantization for the paged pools** (QSA and MLA/DSA families): three tiers, measured on a round-trip against bf16 reference attention (max-cosine over sparse + split-k paths):

  | tier | bytes / token / KV layer | cosine vs bf16 | use when |
  |---|---|---|---|
  | `bf16` | 2048 | 1.0000 | default; smallest contexts |
  | `fp8_e4m3` | 1024 | 0.9994 | general balance point |
  | `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

  Storage is compressed; attention compute stays bf16. Indexer keys always stay bf16.
- **Multimodal (vision) support**: vision tower loads on demand; image requests reuse KV through content-hashed prefix keys (same image → KV reuse, different image → zero false hits), and the mm KV/state snapshots live in a capped RAM tier (`FREETOKEN_MM_TIER_MB`, keep at 4096) that swaps in/out against the GPU pool. Both swap directions are chunked (32768 tokens per chunk) so a restore never balloons a transient allocation past the tight VRAM headroom.
- **Leak-free page reclaim for tier restores** — a RAM-tier restore allocates the request's OWN fresh pages for the whole restored span, while the tree also keeps its copy of the matched prefix. Each cache handle now reports its own reclaim floor (`commit_floor()`): tree handles return `cached_len` (the borrowed prefix is the tree's), restore handles return 0 (the request owns the whole span, so the duplicated part is freed on finish). This fixed a steady per-request page leak (304 pages per tier restore, bounded by pool size) and is model-shape agnostic — every handle shape inherits the method from the base class, verified across the radix / hybrid / SWA / naive / tier handle family.
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

The verified combo of record for the reference box (RTX 5090, Qwen3.8-Flash-Next-NVFP4, agent-style traffic with heavy prefix reuse):

```powershell
$env:FREETOKEN_MM_TIER_MB = "4096"
ft-venv\Scripts\python.exe -m freetoken `
  --model-path E:\FreeToken\models\Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 3072 --kv-cache-dtype turbo4 `
  --moe-prefill-hit-d2d --host 0.0.0.0 --port 8001 --max-running-req 4
```

The two size knobs trade against each other; the working ladder (all at `--max-running-req 4`):

| context (`--num-tokens`) | MoE cache (`--moe-cache-size`) |
|---|---|
| 1048576 (1M) | 3072 (or 2048) |
| 524288 (512K) | 4096 |
| 262144 (256K) | 5120 |

At 1M context, `--moe-cache-size 4096` OOMs on this card; 3072 is the ceiling. Change one knob per restart and check stats before changing the next.

Flag notes:

- `--kv-cache-dtype turbo4` — see tier table above; it is what makes the 1M pool affordable (~7 GiB instead of ~28).
- `--moe-cache-size` — GPU-side expert LRU slots; size it with the ladder above for your context length.
- `--moe-prefill-hit-d2d` — serve prefill expert hits with device-to-device copies instead of re-fetching from host RAM.
- `--max-running-req 4` — matches the captured CUDA-graph buckets [1, 2, 4]; bs=8 capture is risky on this card because the remaining ~1 GiB VRAM headroom must also cover per-step intermediates, mm chunk swap-in temps, and vision encoder temps.

Then hit `POST /v1/chat/completions` as with any OpenAI-compatible server.

## What to expect at runtime

- **VRAM**: steady state ≈ 31.2–31.4 GiB of ~31.8 GiB usable. The ~1 GiB left is not slack — it is the working room for per-step intermediates, mm tier chunked swap-in temporaries, and vision encode temps. Don't stack more captured batch sizes on top of it.
- **Batch size**: agent traffic (a few concurrent sessions, staggered turns) mostly runs bs=1 with a short queue — the scheduler alternates prefill and decode steps, and queued requests join at the next prefill-step boundary, so bs bumps at those boundaries and is flat between them. This is workload-shaped, not an engine defect (vLLM/SGLang behave the same way with staggered arrivals). If you want bigger batches, have clients arrive together (parallel fan-out) rather than raising `--max-running-req`.
- **Prefix reuse**: with stable agent system prompts + histories, the cached ratio settles around 96–97%; warm requests are served from the tree/mm tier in seconds.

## Measured results

All numbers below are from live runs on the reference box (RTX 5090 32 GB, ~254 GiB RAM, Qwen3.8-Flash-Next-NVFP4, served with the combo above).

Decode speed by live batch size (aggregate completion tok/s):

| bs | aggregate tok/s | notes |
|---|---|---|
| 1 | ~43–50 | the steady state for staggered agent traffic |
| 2 | ~48 | |
| 4 | ~85 | practical ceiling with mm/vision reserves |
| 8 | ~111 | only from launches with more headroom; risky within the current reserves |

The jump from bs=1 to bs=4 is real; beyond that you are inside the VRAM reserve and CUDA-graph capture starts competing with runtime temporaries.

TTFT: cold requests follow roughly `tokens ÷ 3400` seconds (e.g. a ~16k-token first turn ≈ 5 s; the ~900k-token turn measured 32.5 s with prefill overlap on). Warm requests that hit the tree/mm tier land in seconds. Needle-in-haystack at 322k/450k-token contexts scored 6/6 exact (worst case 35.2 s at 90% depth, others 5–12 s); decode steady-state speed is length-independent.

MoE cache ladder (with `--num-tokens` held at the paired value): slots 1024 → 2048 → 3072 move long-context steady decode from ≈ 33 → 40 → 44 tok/s (short-context ≈ 52), cumulative **+36% with zero quality loss** — the cache holds exact weights, speed comes from hit rate. Beyond the paired value the curve flattens while costing KV-pool GiB; stop there.

Vision: an image request adds ~+8 ms over a cold text request — the ViT forward is not the cost; the base is MoE expert-cache cold start, shared with text. A large image (19200 patches) peaks +984 MiB VRAM with first token in 2.9–4.2 s. Repeated same-content requests are served from the mm RAM tier; different images never false-hit (content-key 13/13 + adversarial 174/174 legs green). Over the 32768-patch guard the API returns a clean 400 (`context_length_exceeded`), streaming mode carries it as an in-stream error frame.

Leak check: after the commit-floor fix, a 40-round × 3-request simulated agent soak (shared growing prefixes, chunked prefill, tier store/restore, eviction pressure, aborts — `leak_sim.py`) keeps every page accounted for after every operation, and live stats show KV usage stable across restarts.

## Tuning notes

- **Pin ceiling**: under WDDM the driver caps GPU-accessible pinned host memory at roughly half of physical RAM; locked (working-set) pages count against the same budget. Keep `FREETOKEN_PIN_BUDGET_GB` below the ceiling minus the locked-layer share, and raise it in steps after freeing other RAM consumers.
- **Locked ≠ pinned**: locked banks (head/tail layers) decode on the CPU executor; pinned banks stream to the GPU. More pinned = faster decode; locking is the fallback when pinned would exceed the ceiling.
- If startup OOMs on the last bank, lower `--num-tokens` first (turbo4 keeps the KV pool under ~7 GiB even at 1M tokens), then re-pick the MoE cache from the ladder, then the budget.
- mm tier sizing: `FREETOKEN_MM_TIER_MB=4096` caps the host-side snapshot store; it lives in system RAM (the ~250 GB side), not VRAM — the VRAM cost is only the chunked swap temporaries during restore.

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

Plus two accounting checks that run CPU-only against the live package tree: `leak_sim.py`-style soak (every page in exactly one of free/tree/running-row after every operation) and the handle-shape check for `commit_floor()` across all six cache-handle classes.

Run from the package root with the venv Python, e.g. `python verification/p4_splice_test.py`.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | cap for CUDA-pinned expert banks; exceeding it triggers split residency (locked CPU layers) | auto (~0.4–0.5 × RAM) |
| `FREETOKEN_MM_TIER_MB` | cap for the mm RAM tier (host-side multimodal snapshot store) | 4096 |
| `FREETOKEN_LOAD_VISION` | `1` loads the vision tower | off |

## Provenance

Derived from upstream FreeToken commit `58f4b9e` (release `0.1.2+g816c324d0`), with upstream main merged through `afd99cb` on top (the mm/MMItem + section-table-rope refactor, kept alongside this branch's KV-tier / mm-tier path). The Windows/quantization/mm changes are documented per-area in `CHANGELOG.md` and the round logs; the kernel cache wheel is unchanged from upstream. Upstream LICENSE (Apache-2.0) and copyright are preserved in `package/`.

The third-party checkpoint used with this build (Qwen3.8-Flash-Next-ABLITERATED-NVFP4) is not part of this repository; it is downloaded at runtime from its Hugging Face hub.
