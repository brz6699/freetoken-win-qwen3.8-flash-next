# FreeToken — Windows 单 GPU 构建

English version: [README.md](README.md).

基于 [FreeToken](https://github.com/FlashML-org/FreeToken)(基线 `0.1.2+g816c324d0`,上游 main `58f4b9e`)的补丁分支,针对 **一张消费级 GPU + 大容量系统内存** 调优——例如 RTX 5090 32 GiB 配 ~254 GiB RAM——通过 OpenAI-/Anthropic-兼容 HTTP API 服务大型 MoE checkpoint。上游 Apache-2.0 LICENSE 保留;详见 [Provenance](#provenance)。

本 README 列出本分支在上游 `58f4b9e` 之上**新增或改造**的内容,加上对上游已有 knob 的调优建议。每个功能点按同一格式:**(1) 在这套栈上观察到的问题**、**(2) 实现的功能设想**、**(3) 实测验证结果)。参考环境:RTX 5090 SM120 PCIe-bound + Qwen3.8-Flash-Next(-ABLITERATED)-NVFP4。

---

## 功能点

### 1. Verify 阶段统计排除(从 hot-expert LRU)

**(1)** MTP verify batch 携带 K+1-row 分布,与 decode-phase routing 不同。混入 hot-expert LRU 会晋升只有 verify 用得上的 expert,踢掉真实 decode working set。上游 recurrence-window admission 没区分 verify 与 decode 阶段。

**(2)** `MtpExpertBank` 与 `OffloadMoeCache` 链(`load_experts` / `ensure_experts` / `ensure_experts_hybrid` / `prefetch_decode_layer`)透传 `record_stats: bool` kwarg,由 scheduler 中 `ctx.batch.moe_force_decode` 决定取值。Verify admission 更新 hit/miss 计数,**不**晋升 LRU。Decode admission 两者都做。

**(3)** Expert residency 跟随真实 decode demand,不被 verify-phase 噪声拉偏。bs=8 mixed-decode aggregate ~187/~193 tok/s 跨 round;bs=4→bs=8 scaling ×1.85——PCIe BW ceiling 之内,无 LRU-thrash 凹陷。

---

### 2. turbo4 KV storage tier + TurboQuant packed-nibble rotspace

**(1)** bf16 paged KV 在 PCIe-bound 栈上 bs=1 长上下文吃太多 VRAM。已有的 fp8_e4m3 tier 在 1M-token context 下还留 ~50% headroom 没用。需要第四档,cosine ≥0.98 vs bf16,在 ~6.8 GiB pool 里装最长上下文。

**(2)** 三档分级,对 bf16 reference attention 测(max-cosine over sparse + split-k paths):

| tier | bytes / token / KV layer | cosine vs bf16 | use when |
|---|---|---|---|
| `bf16` | 2048 | 1.0000 | default;smallest contexts |
| `fp8_e4m3` | 1024 | 0.9994 | general balance point |
| `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

Storage 压缩;compute 保 bf16。Indexer keys 永远 bf16。`kvcache/turboquant.py` 实现 packed-nibble rotspace dequant + bf16 per-(token, head, slab) scales。`kvcache/dsv4_cost_model.py` 算每 token byte cost,含 scale overhead。Engine gate 限 turbo4 用在 QSA + MLA-latent families。

**(3)** bs=1 ~50k-token context 装进 ~3 GiB KV pool,verify peak 容得下不 OOM。Needle retrieval 6/6 exact 跨 322k/450k × depth {30%, 60%, 90%},worst-case depth-90% 在 35.2s,其余 5–12s。KV ≈6.9 KiB/token 实测 → full 1M-token pool ≈6.8 GiB。TTFT @ 901k context:prefill overlap 时 32.5s,否则 51s。稳态 decode 长度无关。

---

### 3. QSA sparse attention 扩展以适配 turbo4 packed-nibble rotspace

**(1)** 上游 `attention/qsa_sparse.py` 假定 bf16-packed KV layout 跑 block-top-k scatter/gather。加 turbo4 tier 要求 indexer 的 block_topk reads 与 attention-row writes 在不增加额外 dequant pass 的前提下处理 packed nibbles + per-token bf16 scales。

**(2)** `qsa_block_topk` Triton kernel 改为读 packed-nibble blocks + co-loaded bf16 scale row,scatter writes 走 `index_copy_` inplace,单次 launch 覆盖 K/V 两块 slab。上游文件 542 行 → ours 扩到 647 行(turbo4 branch + cost-model integrate `dsv4_cost_model.py`)。Compute path 不变,仍 bf16——只换 storage layout。

**(3)** bs=1 ~50k-token context 装进 ~3 GiB KV pool。Block_topk overhead vs bf16 在 measurement noise 之内;cosine vs bf16 reference 在 24 sanity legs 上稳定 0.9887。

---

### 4. BudgetSlider 运行时再平衡

**(1)** 上游 `engine/cache_budget.py` 只做启动期 sizing。固定 KV/MoE pools 在一套流量组合下 sizing,几小时后组合变化 headroom 浪费。Long-lived agent traffic 在 short high-rate burst 与 long idle stretches 之间交替;这套栈上重启不可接受。

**(2)** `engine/budget_slide.py` 加 `BudgetSlider` 类,在 idle transition 重分配 KV pages ↔ MoE slots。Asymmetric EMA:α 上升时高(burst 期间不要欠配),下降时低(load 回落时不要 pingpong)。`min_step_ratio` deadband 忽略阈值以下抖动;`demand_floor` 防 shrink-only 跌破下一请求 working set。`mha_pool.rebuild` / `OffloadMoeCache.rebuild` 切视图自 arena left end;base pointer 不动,CUDA graph 仍有效。只在 idle transition 触发——唯一同时没有 live page-index 引用也没有 in-flight graph replay 的时刻。Preserve-residency:已 pinned experts 不动。

**(3)** bs=4 mixed-decode aggregate 跨 round 稳 ~104 tok/s;bs=8 ~187/~193 tok/s;bs=4→bs=8 scaling ×1.85 within PCIe BW ceiling。Rebuild 节奏:burst 时每几条请求一次,idle ~每分钟一次。`budget_slide_test.py` 全绿。

---

### 5. mm content-key sha1 prefix cache + splice 不变量修复

**(1)** 重复 mm 请求重 prefill 同样图像像素;prefix match 落在 image_pad span 内时 splice 可能越界 COW buffer。Production crash regression 实测复现。

**(2)** Tokenizer worker 用图像像素负载 sha1 → content id;mm prefix matching/insertion 走 `mm_cache_key`(image_pad spans 替换为 content id)——同图复用 KV,异图零误命中。`match_req` 落在 pad span 内的 prefix fallback 到 `align_down(span_start, page_size)` 并丢弃 GDN snapshot(防 COW overrun);越过 last pad 的匹配原样放行。`qwen4_exp` / `gemma4` splice 在 `n_pads==0` 时为合法 no-op;中间态仍 raise。

**(3)** Content-key reuse 13/13;adversarial cross-image isolation 174/174 green。`p4_splice_test.py` 21 腿全绿,包含 production crash regression track@960 crossing middle-span → align_down=8 fallback、GDN dropped、955 pads all-in-row restored。Restart 复验:5 suites + 174 adversarial cases 全绿;三处改动仍在 packaged tree。

---

### 6. mm video path + dual-pad 拼写修复

**(1)** 不同 checkpoint 写 `image_token_id` 或 `video_token_id`(都在 vision_start/end 内)。PIL 没 ffmpeg backend 读不了 mp4 container。上游没识别两拼写,video legs 静默丢掉。

**(2)** Prepare 检测两拼写;展开统一写 `image_token_id`,model-side embed mask / cache key / scheduler 零改动;`vision_geometry` 透传 `video_token_id`。Video 部件接受帧列表(data:/http/local paths)或单个动画文件(GIF/WebP/APNG);≤8 帧等间隔采样;多帧走处理器 `videos=` 时间配对(回退逐帧堆叠),`t>1` grid;展开数取自真实 grid,仍只渲染一个 pad placeholder。mp4 fallback via `imageio`(bundled static `imageio-ffmpeg`):整段读入采样 ≤8 帧。超过 32768 patch guard 返回干净 400 `context_length_exceeded`,streaming 在流内承载 error frame。

**(3)** `serve_video_smoke.py` / `vision_video_test.py` / `vision_p3_media_test.py` 对比 HF oracle 全绿。Live serve(:8001)smoke:frame-list / GIF / image_url+video mixed legs 都对(静态 green ∉ animated {red, blue})。Reasoning leg prompt_tokens=238 measured;reasoning leg budget ≥512。Image vs text cold TTFT medians:text 1518ms vs image 1526ms(+8ms)。ViT forward 不是主导——~1.5s floor 是 fixed MoE expert-cache cold-start,text 同样付。Large image (19200 patches) peak VRAM +984 MiB,first-token latency 2.9–4.2s。

---

### 7. mm RAM tier swap-in/out

**(1)** 几小时后同 mm 内容到达时仍要付全 ViT + prefill from cold KV。Windows pinned-budget arena VRAM 有限,pure VRAM-resident mm cache 撑不住 hour-long gap。上游没有 mm-tier 机制。

**(2)** 完成的 mm 请求 stash prompt-segment KV + GDN state 进 host RAM。后续同 content-key 请求 restore rows 进新分配的 VRAM pages,直接续跑。`kvcache/mha_pool.py` / `qsa_pool.py`:`snapshot_rows` / `restore_rows` 打包 K/V compressed blocks + turbo4 dequant scales + QSA compressed-index rows(`slot // index_ratio`)到 CPU;restore 走 `index_copy_` inplace scatter(`[:, idx].copy_` 会先 gather 成临时量写不进去——踩过,inline doc)。`pending_ring` transient 不入 tier。`linear_state_pool.snapshot_state` / `restore_state` 整槽打包 conv+recurrent+slot_states。Tier storage LRU-bytes eviction,`FREETOKEN_MM_TIER_MB` budget(默认 1024)。Restore gate:content-key bit-match + boundary beats tree match + doesn't cut a pad run → allocate new pages + row refill + `_TierHandle`。Restore 接入 admission(budget gate 之后);`mm_gdn` 经 PendingReq→Req 透传;`_restore_linear_states` first prefill forward 写回。

**(3)** `kv_tier_test.py` 在真实 QSA+turbo4 pools 上全绿——row round-trip / byte accounting / same-key replacement / LRU eviction / pad-boundary GDN write-back。Vision two-suite regression green。Live serve(:8001)smoke:same-content repeat 走 restore 路径答对;image_url+video mixed legs 判定正确;prompt_tokens=238 measured。Warm TTFT 低于 cold TTFT 证实 tier hits。

---

## 上游已提供 knob 的调优建议

下列机制上游已有,本分支原样采纳并针对 PCIe-bound single-GPU 栈调参。

- **Shared cudaMemPool CUDA-graph ladder** — 上游 `engine/graph.py` 已 share cudaMemPool + graph handle across bs entries。bs-histogram ≥95% bs=1(`/v1/stats` `bs_hist`)时 `--cuda-graph-max-bs 1` 折叠为单 bs=1 capture;replay 向上 pad 到下一个 entry。bs=1-dominated Windows stacks 实测 bs=1 ladder 在 PyTorch allocator overhead 之后保留 ~7.5 GiB contiguous free。
- **bs-sized transport routing** — `hybrid_fetch_fraction`(0–1 ratio,init 读一次自 `ft bench bw`)取代固定 `hybrid_max_fetch` count。每 decode step fetch `fraction × misses` expert,PCIe 饱和,CPU GEMM 跨 bs=1…8 都 overlap。Profile 只在 engine init 读——decode 中途重测会抖 TTFT。
- **Recurrence-window MoE admission** — `expert_recency` recurrence window gate expert fetch on second sighting;首次 miss 走 CPU。上游 `moe/offload_cache.py` 已实现。与本分支 verify-phase stats exclusion 配合,让 hot LRU 跟踪真实 decode demand。
- **bs≥4 decode-step prefetch** — `_decode_hybrid` 在本层 GEMM dispatch 之前对 next_layer 调 `prefetch_decode_layer(next_layer, hint)`。bs<4 路由熵太高,prefetch-hit 概率压不过浪费 fetch 成本,gate 跳过。Tests:`tests/moe/test_decode_prefetch_gate.py`。
- **KV hot-window** — `FREETOKEN_HOT_WINDOW_TOKENS` cap `demand_tokens` signal,让 KV 涨到 soft ceiling 而不是纯 shrink-only。Opt-in:不设时严格保留 shrink-only。针对多轮 agent prompt:硬性 shrink-only floor 会把还在用的 prefix pages 踢掉。
- **Prefill/decode interleaving** — `_interleave_every`(默认 4)个连续 prefill chunk 强制插一个 decode step。计数单位 chunk 而非秒,保证 replay 确定性。Override:`prefill_interleave_every`。
- **Special-token ckpt** — `--enable-special-token-ckpt` 让 decode 在 tool-call opener token 处快照 GDN state。结构边界:之前(system prompt / few-shot / history)逐轮一致,之后漂移。Snapshot 每轮付一次,prefix reuse 每次 echo 命中。上游 args.py 已有同名 flag。

---

## Bench findings

两条互补 bench regime,各自 isolate 一个 operating point——彼此不能互相代替。两张表都在 exp 全部 mechanism default-on 状态下测得。

| regime | 测什么 | output length | prompt length |
|---|---|---|---|
| Short-output ladder | PCIe BW ceiling + per-step launch overhead;aggregate soft cap 与 prefill 无关 | `max_tokens=96` | short |
| Mixed-decode lc_mixed | steady-state agent workload,prefill overlap + warm prefix cache 都生效;aggregate 在 amortization 后 | per-stream ∈ {1k, 2k, 3k, 4k} | ~10k / ~50k shared-prefix |

同一硬件同一 serve profile(`--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --moe-prefill-hit-d2d`),lc_mixed aggregate 比 ladder aggregate 在同一 bs 下高一截,因为长 output 把 fixed per-step overheads(launch、cache rebuild check、scheduler tick)摊到更多 decoded tokens 上。

**Ladder**——`max_tokens=96`,text-only:

| concurrency | aggregate tok/s | per-stream tok/s | notes |
|---|---|---|---|
| 1 | 26.2 | 26.2 | |
| 2 | 47.5 | ~24 | |
| 4 | 66.9 | 17.2 | p50 ≈5.0s,p95 ≈5.2s — sweet spot |
| 6 | 64.6 | ~11 | |
| 8 | 68.8 | 9.9 | p95 climbs back |

Aggregate soft ceiling ≈65–77 tok/s——PCIe BW + per-step expert-fetch volume 决定,非 compute。`--max-running-requests 8` 让 bs=6 填 bs=8 graph → aggregate 升到 ~76.9,p95 ≈7s(c8 tail ≈25s)——throughput-first only。

**lc_mixed**——`bench_lc_mixed.py`,MTP off,`--cuda-graph-max-bs 1`,两 prompt tiers × bs ∈ {4, 8},各跑两轮:

| bs | prompt tier | aggregate tok/s | per-stream avg tok/s | TTFT cold → warm |
|---|---|---|---|---|
| 4 | ~10k | ~104 | ~25 | 3.2s → 2.4s |
| 4 | ~50k | ~104 | ~25 | 10.1s → 2.6s |
| 8 | ~10k | ~193 | ~24 | 46.6s → 46.8s |
| 8 | ~50k | ~187 | ~24 | 47.7s → 48.3s |

三点观察:per-stream decode ~25 tok/s 与长度无关直到 ~50k context(KV read BW 还不是瓶颈——weight streaming 主导);bs=4→bs=8 scaling ×1.85 within PCIe BW ceiling;bs=8 TTFT ~47s 反映 chunked-prefill queueing across 8 streams × XLONG prompts × ~8192-token chunks,非 decode slowdown。Round-to-round TTFT 下降证实 mm content-key path warm prefix cache hits 生效。

**MoE cache size vs decode speed**(`ft ctl cache rebuild` live resize,no restart)。Slots 1024 → 2048 → 3072:long-text steady-state decode ≈62 → 75 → 84 char/s(≈33 → 40 → 44 tok/s;short text ≈52)。Cumulative **+36% 零质量损失**——cache 存 exact weights,gains 纯靠 hit rate。Past ~3072 curve flatten(+5–8% 但吃 KV pool)。

---

## Serve profiles

Launcher `_exp_garble/run_exp.py` 是个 thin shim:prepend exp package dir 到 `sys.path`,让 `import freetoken` resolve 到 exp tree;CLI flags 透传不变。等价于 `ft serve` 跑同一 package。下面路径都是示意——换成你自己的 venv Python、launcher path、model dir。

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

这是 1M-token longest-context profile + b12x Triton backend(功能点 3)+ bs=4 cap。Budget slide enabled;encoder towers default-on;CUDA-graph ladder auto-tuned from VRAM。

bs=1-dominated tight profile(`/v1/stats` bs_hist ≥95% bs=1 时单独验证过):

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

- `--quant-backend moe.nvfp4=b12x` — NVFP4 MoE decode 选 b12x Triton backend(功能点 3)。不设时 auto-selector 在 SM80/99 上可能选 `marlin`,否则 fall through 到 torch `_scaled_mm`。SM120 + 这个 checkpoint 下 `b12x` 是对的;配合 `--moe-prefill-hit-d2d` 让 prefill hits 复用 GPU-side copies。支持 layer-level override:`linear=marlin,moe=b12x`。
- `--num-tokens` — paged KV pool size。turbo4 ≈6.9 KiB/token → full pool ≈6.8 GiB @1M,≈3.4 GiB @512k,≈0.4 GiB @65k。Pool 越小给 verify peak 留越多 contiguous VRAM headroom。
- `--moe-cache-size` — GPU 上驻留的 expert 数量。Steady-state decode 大致跟 hit rate 涨到 ~3072,然后 flatten。
- `--kv-cache-dtype` — 见功能点 2 tier table。
- `--max-running-req N` — scheduler concurrency cap。Argparse prefix-match `--max-running-requests`,两种写法都接受。Default 让 bs 涨到 ladder cap;`4` 让 bs=4 graph 保持热。
- `--text-model-only` — skip all encoder towers;它们的 VRAM 给 KV / expert caches。只想 skip certain kinds 用 `--mm-disable vision audio`。Vision / audio towers 默认 load,disabled 时 clean skip。上游已提供 flags。
- `--moe-prefill-hit-d2d` — prefill expert hits 走 device-to-device copies。上游已提供。
- `--cuda-graph-max-bs N` — CUDA-graph ladder cap;bs=1-dominated 设 1,共享 mempool 下 bs=4 burst slot 加 `4`。上游已提供 flag。
- `--moe-prefill-demand-tokens` — demand-prefill chunk-row threshold(功能点 5);默认 `64` 适配 cached-prefix incremental prefill。上游已提供 flag。
- `--mtp-expert-quant q4_0` — shrink the MTP draft expert bank to q4_0,让 verify peak fit inside tight VRAM。配合 `--mtp-device gpu` 才真启 MTP;本分支默认 off(见功能点 1 调优注)。
- `--disable-budget-slide` — opt-out of BudgetSlider(功能点 4),freeze KV/MoE pool sizes from startup。Workload single-model fixed-footprint、想让 CUDA-graph captures pin to stable addresses 时用。

## Requirements & install

Windows 11(或 WSL2)x64,NVIDIA GPU ≥24 GiB VRAM,推荐 ≥2× model-size RAM。Python 3.12(`cp312` wheels;kernel-cache wheel CUDA 13.x)。Source-only;deps 来自 public indexes。

```powershell
python -m venv ft-venv
ft-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
ft-venv\Scripts\pip install .
```

可选:kernel-cache wheel attached to GitHub Releases(skip first-request Triton JIT;matching CUDA 13.x)。可选:`imageio` bundled static `imageio-ffmpeg`,vision path mp4 container support。然后 `POST /v1/chat/completions` 跟 OpenAI-compatible server 一样。

## Verification suite

| script | covers |
|---|---|
| `p4_splice_test.py` | mm splice invariant (pad-span alignment, COW no-op) — 21 cases |
| `p4_finish_key_test.py` | finish-key + SWA regression legs |
| `p4_content_key_test.py` | content-hash prefix keys (reuse / isolation) |
| `mm_prefill_chunk_test.py` | whole-segment mm prefill rollback |
| `r4_adv_probe.py` | adversarial fuzz:`page_size ∈ {1,4,8}` × shapes × rounds + cross-span gaps + page ownership |
| `kv_tier_test.py` | RAM-tier KV swap-in/out round-trip |
| `budget_slide_test.py` | runtime KV/expert pool resize without restart |
| `serve_video_smoke.py`, `vision_video_test.py`, `vision_p3_media_test.py` | end-to-end vision/media requests |

Package root + venv Python:`python verification/p4_splice_test.py`。

## Environment variables

| variable | meaning | default |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | CUDA-pinned expert banks cap;超出触发 split residency(locked CPU layers) | auto (~0.4–0.5 × RAM) |
| `FREETOKEN_MM_TIER_MB` | mm RAM tier byte budget for swap-in/out eviction | 1024 |
| `FREETOKEN_HOT_WINDOW_TOKENS` | cap demand_tokens signal so KV grows to soft ceiling,非纯 shrink-only | unset(shrink-only preserved) |

## Provenance

<a name="provenance"></a>

Derived from upstream FreeToken commit `58f4b9e`(release `0.1.2+g816c324d0`)。Upstream PRs merged in:#377、#375、#343、#336、#332、#329、#319、#311、#310、#243、#257,加基础的 #1 / #2 / #3 / #4 / #5 / #6 / #7 / #25 / #75 / #117。Kernel-cache wheel unchanged from upstream。Upstream LICENSE(Apache-2.0)和 copyright preserved in `package/`。第三方 checkpoint Qwen3.8-Flash-Next-ABLITERATED-NVFP4 runtime 从其 HF hub 下载,不属于本 repo。
