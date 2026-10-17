# FreeToken — Windows 单 GPU 构建

English version: [README.md](README.md).

基于 [FreeToken](https://github.com/FlashML-org/FreeToken)(基线 `0.1.2+g816c324d0`,上游 main `58f4b9e`)的补丁分支,针对 **一张消费级 GPU + 大容量系统内存** 调优——例如 RTX 5090 32 GiB 配 ~254 GiB RAM——通过 OpenAI-/Anthropic-兼容 HTTP API 服务大型 MoE checkpoint。上游 Apache-2.0 LICENSE 保留;详见 [Provenance](#provenance)。

下面每个功能点按同一格式组织:**(1) 在这套栈上观察到的问题**、**(2) 实现的功能设想**、**(3) 实测验证结果)。所有数字参考环境:RTX 5090 SM120 PCIe-bound + Qwen3.8-Flash-Next(-ABLITERATED)-NVFP4。

---

## 功能点

### 1. 共享 cudaMemPool 的 CUDA-graph ladder + fixed-shape 折叠

**(1)** 上游每个 bs capture 各自持有 workspace + graph-private pool。bs ∈ {1,2,4,8} ladder 的死槽位 overhead 抢占 verify peak 需要的连续显存。`/v1/stats` bs-histogram 显示这套流量 ≥95% 是 bs=1,ladder 仍然在 capture bs=4/8 的图。

**(2)** `engine/graph.py` 所有 ladder entry 共享同一个 `cudaMemPool` + 一份按 `max_graph_bs` sized 的共享 `GraphCaptureBuffer`。ladder 各条目共享 workspace 而非各占一份,新增条目成本只是 topology metadata(~几 MB),不是再一份 workspace。Ladder = `[1, 2, 4] + range(8, max_bs+1, 8)`。`--cuda-graph-max-bs 1` 折叠为单 bs=1 capture,replay 向上 pad 到下一个 ladder entry。

**(3)** bs=1-dominated 流量:`--cuda-graph-max-bs 1` 在 PyTorch allocator overhead 之后保留 ~7.5 GiB 连续显存,MTP verify peak (bs=4 + `--mtp-draft-len 2` ≈3–4 GiB) 容得下。bs=4 mixed-decode bench:跨 round 稳态 ~104 tok/s aggregate。pad-to-next-ladder-entry 无质量损失。

---

### 2. Verify 阶段统计排除(从 hot-expert LRU)

**(1)** MTP verify batch 携带 K+1-row 分布,与 decode-phase routing 不同。混入 hot-expert LRU 会晋升只有 verify 用得上的 expert,踢掉真实 decode working set。

**(2)** `MtpExpertBank` 与 `OffloadMoeCache` 链(`load_experts` / `ensure_experts` / `ensure_experts_hybrid` / `prefetch_decode_layer`)透传 `record_stats: bool` kwarg,由 scheduler 中 `ctx.batch.moe_force_decode` 决定取值。Verify admission 更新 hit/miss 计数,**不**晋升 LRU。Decode admission 两者都做。

**(3)** Expert residency 跟随真实 decode demand,不被 verify-phase 噪声拉偏。bs=8 mixed-decode aggregate ~187/~193 tok/s 跨 round;bs=4→bs=8 scaling ×1.85——PCIe BW ceiling 之内,无 LRU-thrash 凹陷。

---

### 3. b12x Triton backend for NVFP4 MoE decode

**(1)** cuBLASLt 在 SM120 上对本模型的 NVFP4 MoE shape(`hidden=2560`、`num_experts=512/topk=10`、`moe_intermediate=640`)吃不满 PCIe BW;bs ∈ {1,4,8} decode 时 launch overhead per route 主导。

**(2)** `kernel/triton/nvfp4_fused_moe.py` 走 marlin-style:宽 int32 load(每 word 装 8 codes)、K reduction 延后、register-only `_split8` shuffle、fp32 accumulate、per-route grid `(num_routes, cdiv(N, BLOCK_N))`。Block-scale epilogue 折进 fp16 bits,global-scale 在 K-loop 之后施加 + arrival-counter split-K reduction 单次 launch。

**(3)** RTX 5090 实测 PCIe BW ceiling ~87%。Per-stream decode ~25 tok/s 与长度无关直到 ~50k context(KV read BW 还不是瓶颈——weight streaming 主导)。Aggregate bs=4 mixed-decode ~104 tok/s 跨 ~10k/~50k prompts。残差 ~13% 归因 FP4-dequant ALU。这条已经覆盖 DeepGEMM tiled cluster kernel 在我们 shape 范围内能提供的内容,所以不再引入 DeepGEMM。

---

### 4. Hybrid QSA sparse attention + turbo4 KV 分级存储

**(1)** bf16 paged KV 在 PCIe-bound 栈上 bs=1 长上下文吃太多 VRAM。需要 storage-side 压缩而不动 compute path。

**(2)** 三档分级,对 bf16 reference attention 测(max-cosine over sparse + split-k paths):

| tier | bytes / token / KV layer | cosine vs bf16 | use when |
|---|---|---|---|
| `bf16` | 2048 | 1.0000 | default;smallest contexts |
| `fp8_e4m3` | 1024 | 0.9994 | general balance point |
| `turbo4` | ~516 | 0.9887 | longest contexts (100k+ tokens) |

Storage 压缩;compute 保 bf16。Indexer keys 永远 bf16。QSA paged KV pool 在 turbo4 packed nibble rotspace 存;block top-k 走 Triton `qsa_block_topk`。Engine gate 限 turbo4 用在 QSA + MLA-latent families。

**(3)** bs=1 ~50k-token context 装进 ~3 GiB KV pool,verify-phase 不 OOM。Needle retrieval 6/6 exact 跨 322k/450k × depth {30%, 60%, 90%},worst-case depth-90% 在 35.2s,其余 5–12s。KV ≈6.9 KiB/token 实测 → full 1M-token pool ≈6.8 GiB。TTFT @ 901k context:prefill overlap 时 32.5s,否则 51s。稳态 decode 长度无关。

---

### 5. MoE prefill demand 阈值——按 chunk 行数 gate

**(1)** 整个 prompt 长度 gate expert-fetch 会在 cached-prefix incremental prefill 时过度 fetch:每步新增 tokens-per-step 很小但累计上下文很大。Whole-prompt gating 把每个 incremental chunk 当 cold-start 处理。

**(2)** `--moe-prefill-demand-tokens` 按 chunk 行数(`hidden_states.shape[0]`)gate,不按整个 prompt 长度。默认 `64`。Cached-prefix hits 跳过 demand-prefill 路径;cold start 一次性付。

**(3)** Cached-prefix warm hits 不重 fetch whole-prompt working set。bs=8 mixed-decode TTFT ~47s 反映 chunked-prefill queueing across 8 streams × XLONG prompts × ~8192-token chunks——不是 demand-gating overhead。Round-to-round TTFT 下降证实 warm prefix-cache 生效。

---

### 6. Round-7 运行时再平衡——八条独立 gate 的规则

**(1)** 固定 VRAM arena 必须在不重启 engine、不重载权重的前提下适应流量组合变化。每条上游 knob 单独看太粗;合在一起的规则需要各自独立可关,关一条不影响其他。

**(2)** 八条小规则,共享不变量:VRAM 启动一次性 allocate,之后只做 view reslicing + bookkeeping reset。

- **Recurrence-window MoE admission** — `ensure_experts_hybrid` 在 recurrence window 内第二次看到该 expert 时才 fetch 上 GPU,首次 miss 走 CPU 计算。Gate 每层 expert lookup 触发,在 miss 那一刻决定,而不是 periodic sweep。消除 cold prefill 阶段每个 expert 都短暂占 GPU slot 的 cache thrash。
- **bs-sized transport routing** — `hybrid_fetch_fraction`(0–1 ratio,init 读一次自 `ft bench bw`)取代固定 `hybrid_max_fetch` count。每 decode step fetch `fraction × misses` expert,PCIe 饱和,CPU GEMM 跨 bs=1…8 都 overlap。Profile 只在 engine init 读。
- **bs≥4 decode-step prefetch** — `_decode_hybrid` 在本层 GEMM dispatch 之前对 next_layer 调 `prefetch_decode_layer(next_layer, hint)`。bs<4 路由熵太高,prefetch-hit 概率压不过浪费 fetch 成本。测试 `tests/moe/test_decode_prefetch_gate.py`。
- **KV hot-window** — `FREETOKEN_HOT_WINDOW_TOKENS` cap budget-slider 看到的 `demand_tokens` 信号,让 KV 涨到 soft ceiling 而不是纯 shrink-only。Opt-in:不设时严格保留 shrink-only。针对多轮 agent prompt:硬性 shrink-only floor 会把还在用的 prefix pages 踢掉。
- **Prefill/decode interleaving** — `_interleave_every`(默认 4)个连续 prefill chunk 强制插一个 decode step。计数单位 chunk 而非秒,保证 replay 确定性。Override:`prefill_interleave_every`。
- **Asymmetric EMA shrink** — `BudgetSlider.propose_targets` 上行快下行慢:EMA α 上升时高(burst 期间不要欠配),下降时低(load 回落时不要 pingpong)。`min_step_ratio` deadband 忽略阈值以下抖动;`demand_floor` 防 shrink-only 跌破下一请求 working set。只在 idle transition 触发。
- **Special-token ckpt,工具调用复用** — `--enable-special-token-ckpt` 让 decode 在 tool-call opener token 处快照 GDN state。该 token 是结构边界——之前(system prompt / few-shot / history)逐轮一致,之后漂移。Snapshot 每轮付一次,prefix reuse 每次 echo 命中。替代旧的"ckpt every N tokens"启发式——后者大多数 ckpt 根本没匹配上。
- **BudgetSlider carrier** — `engine/budget_slide.py` 重新分配 KV pages ↔ MoE slots。`mha_pool.rebuild` / `OffloadMoeCache.rebuild` 切视图自 arena left end;base pointer 不动,CUDA graph 仍有效。Rebalance 只在 idle transition 触发——唯一同时没有 live page-index 引用也没有 in-flight graph replay 的时刻。Preserve-residency:已 pinned experts 不动。

**(3)** bs=4 mixed-decode aggregate 跨 round 稳 ~104 tok/s;bs=8 ~187/~193 tok/s;bs=4→bs=8 scaling ×1.85 within PCIe BW ceiling。配合 `--max-running-requests 8`,bs=6 填 bs=8 graph → aggregate 升到 ~76.9,p95 ≈7s(c8 tail ≈25s)。Rebuild 节奏:burst 时每几条请求一次,idle ~每分钟一次。`budget_slide_test.py` 全绿。

---

### 7. Windows 兼容层 + split residency + pin budget

**(1)** WDDM cap GPU-accessible pinned host memory ≈物理 RAM 一半;locked working-set pages 算同一 budget。上游 POSIX-only `resource` module、`expandable_segments`、Proactor/zmq loop 差异在 Windows 上没有 graceful degradation。Large MoE weights 默认 budget 装不下。

**(2)** `ctypes` `VirtualLock` 替代 POSIX-only `resource`;expandable_segments graceful degradation;Proactor loop adaptation;`FREETOKEN_PIN_BUDGET_GB` env knob(默认 auto ≈0.4–0.5 × RAM)。Ceiling 之下 head/tail MoE layers OS-lock CPU 走多线程 CPU executor(AVX-512 BF16 + VNNI, NVFP4 W4A8);剩余 banks pinned 流 GPU。Locked ≠ pinned:locked banks CPU-side decode,pinned GPU-side streaming。

**(3)** bs=1 ladder 稳态 ~14 GiB allocated / ~16 GiB contiguous free——verify peak 容得下。PyTorch 比 `nvidia-smi` 报少 ~3 GiB free(CUDA driver context + cuBLASLt workspace + cuDNN pools + CUDA-graph private pools 都在 PyTorch allocator visibility 之外)。`torch.cuda.empty_cache()` 只在启动期能 recover 这个 gap;runtime peak sizing 必须靠缩 steady-state allocated。启动期 last bank OOM:先降 `--num-tokens`(turbo4 ≈6.9 KiB/token),再降 budget。

---

### 8. mm content-key 前缀缓存 + splice 不变量修复(Round-4 / P4)

**(1)** 重复 mm 请求重 prefill 同样图像像素;prefix match 落在 image_pad span 内时 splice 可能越界 COW buffer。Production crash regression 实测复现。

**(2)** Tokenizer worker 用图像像素负载 sha1 → content id;mm prefix matching/insertion 走 `mm_cache_key`(image_pad spans 替换为 content id)——同图复用 KV,异图零误命中。`match_req` 落在 pad span 内的 prefix fallback 到 `align_down(span_start, page_size)` 并丢弃 GDN snapshot(防 COW overrun);越过 last pad 的匹配原样放行。`qwen4_exp` / `gemma4` splice 在 `n_pads==0` 时为合法 no-op;中间态仍 raise。

**(3)** Content-key reuse 13/13;adversarial cross-image isolation 174/174 green。`p4_splice_test.py` 21 腿全绿,包含 production crash regression track@960 crossing middle-span → align_down=8 fallback、GDN dropped、955 pads all-in-row restored。Restart 复验:5 suites + 174 adversarial cases 全绿;三处改动仍在 packaged tree。

---

### 9. mm video path + dual-pad 修复(Round-5)

**(1)** 不同 checkpoint 写 `image_token_id` 或 `video_token_id`(都在 vision_start/end 内)。PIL 没 ffmpeg backend 读不了 mp4 container。需要单一 serve 路径同时处理两种拼写而不动 model-side。

**(2)** Prepare 检测两拼写;展开统一写 `image_token_id`,model-side embed mask / cache key / scheduler 零改动;`vision_geometry` 透传 `video_token_id`。Video 部件接受帧列表(data:/http/local paths)或单个动画文件(GIF/WebP/APNG);≤8 帧等间隔采样;多帧走处理器 `videos=` 时间配对(回退逐帧堆叠),`t>1` grid;展开数取自真实 grid,仍只渲染一个 pad placeholder。`extract_media_parts` mirror template branch order。mp4 fallback via `imageio`(bundled static `imageio-ffmpeg`):整段读入采样 ≤8 帧。超过 32768 patch guard 返回干净 400 `context_length_exceeded`,streaming 在流内承载 error frame。

**(3)** `serve_video_smoke.py` / `vision_video_test.py` / `vision_p3_media_test.py` 对比 HF oracle 全绿。Live serve(:8001,vision default-on)smoke:frame-list / GIF / image_url+video mixed legs 都对(静态 green ∉ animated {red, blue})。Reasoning leg prompt_tokens=238 measured;reasoning leg budget ≥512。Large image (19200 patches) peak VRAM +984 MiB,first-token latency 2.9–4.2s。Image vs text cold TTFT medians:text 1518ms vs image 1526ms(+8ms)。ViT forward 不是主导——~1.5s floor 是 fixed MoE expert-cache cold-start,text 同样付。

---

### 10. mm RAM tier swap-in/out(Round-6)

**(1)** 几小时后同 mm 内容到达时仍要付全 ViT + prefill from cold KV。Pinned-budget arena 中没有跨长时间 gap 复用 prompt-segment KV 的手段。

**(2)** 完成的 mm 请求 stash prompt-segment KV + GDN state 进 host RAM。后续同 content-key 请求 restore rows 进新分配的 VRAM pages,直接续跑。`kvcache/mha_pool.py` / `qsa_pool.py`:`snapshot_rows` / `restore_rows` 打包 K/V compressed blocks + turbo4 dequant scales + QSA compressed-index rows(`slot // index_ratio`)到 CPU;restore 走 `index_copy_` inplace scatter(`[:, idx].copy_` 会先 gather 成临时量写不进去——踩过,inline doc)。`pending_ring` transient 不入 tier。`linear_state_pool.snapshot_state` / `restore_state` 整槽打包 conv+recurrent+slot_states。Tier storage LRU-bytes eviction,`FREETOKEN_MM_TIER_MB` budget(默认 1024)。Restore gate:content-key bit-match + boundary beats tree match + doesn't cut a pad run → allocate new pages + row refill + `_TierHandle`。Restore 接入 admission(budget gate 之后);`mm_gdn` 经 PendingReq→Req 透传;`_restore_linear_states` first prefill forward 写回。

**(3)** `kv_tier_test.py` 在真实 QSA+turbo4 pools 上全绿——row round-trip / byte accounting / same-key replacement / LRU eviction / pad-boundary GDN write-back。Vision two-suite regression green。Live serve(:8001)smoke:same-content repeat 走 restore 路径答对;image_url+video mixed legs 判定正确(static green ∉ animated {red, blue});prompt_tokens=238 measured。Warm TTFT 低于 cold TTFT 证实 tier hits。

---

## Bench findings

两条决定 serve-profile 的实测曲线。

**Concurrency sweet spot。** bs=4 在这块卡上 per-stream latency × aggregate 最优。`max_tokens=96`,text-only,`--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --moe-prefill-hit-d2d`:

| concurrency | aggregate tok/s | per-stream tok/s | notes |
|---|---|---|---|
| 1 | 26.2 | 26.2 | |
| 2 | 47.5 | ~24 | |
| 4 | 66.9 | 17.2 | p50 ≈5.0s,p95 ≈5.2s — sweet spot |
| 6 | 64.6 | ~11 | |
| 8 | 68.8 | 9.9 | p95 climbs back |

Aggregate soft ceiling ≈65–77 tok/s——PCIe BW + per-step expert-fetch volume 决定,非 compute。`--max-running-requests 8` 让 bs=6 填 bs=8 graph → aggregate 升到 ~76.9,p95 ≈7s(c8 tail ≈25s)——throughput-first only。

**MoE cache size vs decode speed。** `ft ctl cache rebuild` live resize,no restart。Slots 1024 → 2048 → 3072:long-text steady-state decode ≈62 → 75 → 84 char/s(≈33 → 40 → 44 tok/s;short text ≈52)。Cumulative **+36% 零质量损失**——cache 存 exact weights,gains 纯靠 hit rate。Past ~3072 curve flatten(+5–8% 但吃 KV pool)。

---

## Serve profiles

```powershell
# optional:仅当 expert weights 超过 pinned budget(WDDM 下约 RAM 一半)
$env:FREETOKEN_PIN_BUDGET_GB = "120"

# turbo4 + 1M KV — longest-context profile,encoder tower 默认加载
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 2048 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d

# turbo4 + 512K KV — larger resident expert cache
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 524288 --moe-cache-size 3072 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d

# bs=1-dominated tight profile — single-graph ladder,MTP off,q4_0 bank,text-only
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 65536 --moe-cache-size 1024 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d `
  --cuda-graph-max-bs 1 --moe-prefill-demand-tokens 64 `
  --mtp-expert-quant q4_0 --text-model-only
```

Flags worth understanding:

- `--num-tokens` — paged KV pool size。turbo4 ≈6.9 KiB/token → full pool ≈6.8 GiB @1M,≈3.4 GiB @512k,≈0.4 GiB @65k。Pool 越小给 verify peak 留越多 contiguous VRAM headroom。
- `--moe-cache-size` — GPU 上驻留的 expert 数量。Steady-state decode 大致跟 hit rate 涨到 ~3072,然后 flatten。
- `--kv-cache-dtype` — 见功能点 4 tier table。
- `--text-model-only` — skip all encoder towers;它们的 VRAM 给 KV / expert caches。只想 skip certain kinds 用 `--mm-disable vision audio`。Vision / audio towers 默认 load,disabled 时 clean skip。
- `--moe-prefill-hit-d2d` — prefill expert hits 走 device-to-device copies。
- `--cuda-graph-max-bs N` — CUDA-graph ladder cap;bs=1-dominated 设 1,共享 mempool 下 bs=4 burst slot 加 `4`。
- `--moe-prefill-demand-tokens` — demand-prefill chunk-row threshold;默认 `64` 适配 cached-prefix incremental prefill。
- `--mtp-draft-len` / `--mtp-device gpu` — opt-in。Acceptance ~0.70 with draft-len=2 在 multi-stream aggregate 净正,bs=1-dominated 净负(verify overhead ~70ms/step + redraft ~30ms > skipping forward passes savings)。本分支默认 off。

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

Derived from upstream FreeToken commit `58f4b9e`(release `0.1.2+g816c324d0`)。已 merge upstream PRs:#377(nightly engine manifests)、#375(issue templates)、#343(NVFP4 detection behind compressed-tensors)、#336(safetensors index shards)、#332(GLM-5.3-Flash)、#329(exact Triton top-k/top-p)、#319(in-repo triton router)、#311(PLE n-gram table disk stream)、#310(nightly kernel-cache wheel fix)、#243(`_scaled_mm` sm_89 stall avoidance,baseline commit),加上 #257(Qwen3.8-Flash-Next support)和基础的 #1 / #2 / #3 / #4 / #5 / #6 / #7 / #25 / #75 / #117。Kernel-cache wheel unchanged from upstream。Upstream LICENSE(Apache-2.0)和 copyright preserved in `package/`。第三方 checkpoint Qwen3.8-Flash-Next-ABLITERATED-NVFP4 runtime 从其 HF hub 下载,不属于本 repo。
