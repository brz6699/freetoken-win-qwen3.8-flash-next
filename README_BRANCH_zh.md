# FreeToken — Windows 单 GPU 构建

英文版见 [README.md](README.md)。

基于 [FreeToken](https://github.com/FlashML-org/FreeToken) 的补丁分支(基线:`0.1.2+g816c324d0`,上游 main `58f4b9e`),针对**单张消费级 GPU + 大容量系统内存**的场景调优——例如 RTX 5090 32 GB 搭配约 250 GB 内存——通过兼容 OpenAI / Anthropic 的 HTTP API 服务大型 MoE 模型。

上游采用 Apache-2.0 许可证;本分支完整保留其 LICENSE 与版权声明,见 [Provenance](#provenance)。

## 为什么做这个分支

上游 FreeToken 已经面向 MoE-offload 推理,但 Windows 路径和 KV 存储选项较薄弱。本分支是把这些缺口闭合并用真实流量测量的实验线。问题串成一条主线:KV 分级存储(turbo4 每 token 约为 bf16 的 1/4)腾出显存给 MoE 专家 LRU;专家缓存越大,热专家越能留在 GPU 上;多模态内容哈希前缀缓存让重复的图像/视频不再重新 prefill;CUDA-graph capture 围绕实测 batch 直方图塑形,避免死槽位的 workspace;MTP 自投机解码以可选方式接入,并把 verify 阶段的统计从热专家 LRU 中排除。具体来说,本分支新增或加固了:

- **Windows 兼容性** — WDDM 下的工作集 / 锁定内存预算处理、基于 `ctypes` 的页面锁定(`VirtualLock`)替代仅 POSIX 可用的 `resource` 模块,以及 `expandable_segments` 与 zmq/Proactor 事件循环差异的优雅降级路径。
- **专家权重组的分离驻留** — 当 CUDA 锁定内存预算小于专家权重总量时,头部/尾部 MoE 层以 OS-lock 方式常驻内存并在多线程 CPU 执行器上解码(AVX-512 BF16 + VNNI,NVFP4 W4A8),其余权重组则锁定在 pinned 内存中供 GPU 流式读取。预算可通过 `FREETOKEN_PIN_BUDGET_GB` 调节。
- **paged KV 池的存储侧量化**(QSA 与 MLA/DSA 系列):三档存储精度,基于对 bf16 参考注意力的往返测量(sparse + split-k 路径的最大余弦相似度):

  | 档位 | bytes / token / KV layer | cosine vs bf16 | 适用场景 |
  |---|---|---|---|
  | `bf16` | 2048 | 1.0000 | 默认;较小上下文 |
  | `fp8_e4m3` | 1024 | 0.9994 | 通用平衡点 |
  | `turbo4` | ~516 | 0.9887 | 超长上下文(100k+ tokens) |

  storage 压缩,attn compute 保持 bf16,indexer keys 始终 bf16。
- **默认开启的多模态支持** — vision / audio encoder tower 默认加载,除非显式禁用;图像请求按内容哈希复用 KV(同图→复用,异图→零误命中),mm KV 页面在 RAM tier 与 GPU pool 之间换入换出。`--text-model-only` 跳过所有 encoder tower 并把它们的 VRAM 还给 KV / expert cache;`--mm-disable vision`(或 `audio`)只跳过指定种类。
- **Qwen3.8-Flash-Next 服务画像** — QSA paged-KV pool 用上表的分级存储,NVFP4 routed experts 走 offload cache,mm 内容哈希前缀缓存端到端由 `verification/` 套件覆盖。

## 实验性机制

下面这套是为这条分支的硬件目标专门调过的若干组件,每一条对应 bench 中暴露的一个具体瓶颈。

**共享 cudaMemPool + fixed-shape ladder 的 CUDA-graph capture。** `engine/graph.py` 把所有 bs capture 到同一个 `cudaMemPool`,外加一份按 `max_graph_bs` sized 的共享 `GraphCaptureBuffer`。ladder 各条目共享 workspace 而不是各占一份,新增条目的成本只是几 MB topology metadata,不是再一份 workspace。ladder 由 `--cuda-graph-max-bs` 通过 `[1, 2, 4] + range(8, max_bs+1, 8)` 展开;对于 bs=1-dominated 流量(用 `/v1/stats` 的 `bs_hist` 验证),`--cuda-graph-max-bs 1` 把 ladder 收成单条 bs=1 capture,去掉死槽位的 overhead 而不损失质量。Replay 向上 pad 到下一个 ladder entry。

**Verify 阶段统计排除。** MTP verify batch 携带 K+1-row 分布,如果混入 decode-phase routing count 会污染热专家 LRU。`MtpExpertBank` 与 `OffloadMoeCache` 链(`load_experts` / `ensure_experts` / `ensure_experts_hybrid` / `prefetch_decode_layer`)透传一个 `record_stats: bool` kwarg,由 scheduler 中的 `ctx.batch.moe_force_decode` 决定取值。verify admission 仍更新 hit/miss 计数,但不参与 LRU promotion。结果:专家驻留反映真实 decode demand,不被 verify 阶段噪声拉偏。

**b12x Triton backend for NVFP4 MoE decode。** `kernel/triton/nvfp4_fused_moe.py` 走 marlin-style:宽 int32 load(每 word 装 8 codes)、K reduction 延后、register-only `_split8` shuffle、fp32 accumulate、per-route grid `(num_routes, cdiv(N, BLOCK_N))`。在 RTX 5090 上跑到 PCIe bandwidth ceiling 约 ~87%,残差来自 FP4-dequant ALU。block-scale epilogue 折进 fp16 bits,global-scale 在 K-loop 之后施加,arrival-counter split-K reduction 单次 launch。这条已经覆盖 DeepGEMM tiled cluster kernel 在我们 shape 范围内能提供的内容,所以不再额外引入 DeepGEMM。

**Hybrid QSA sparse attention + turbo4 KV。** QSA paged KV pool 用 turbo4 packed nibble rotspace 存储,compute 保持 bf16,indexer keys bf16。block top-k 走 Triton `qsa_block_topk`。这套组合让 bs=1 在 ~3 GiB KV pool 里扛住 ~50k-token 上下文,verify peak 时不会 OOM。

**Moe prefill demand 阈值。** `--moe-prefill-demand-tokens` 按 chunk 行数(`hidden_states.shape[0]`)而非整个 prompt 长度 gate expert-fetch。cached-prefix incremental prefill 时新 tokens-per-step 很小但累计上下文很大;阈值匹配 chunk size,所以命中 prefix cache 的增量走轻量路径,只有冷启动付一次 demand-prefill。默认 `64`。

**Budget slide:运行时 KV/expert rebalancing。** `engine/budget_slide.py` 根据当前流量重新分配 KV pool pages 与 MoE cache slots,不需要重启 engine 也不需要重新加载权重。实测触发节奏:burst 时每几条请求一次,idle 时大约每分钟一次。Rebuild 是 preserve-residency(已驻留的专家权重不动),只换 pool sizes。

## Round-7 —运行时再平衡 + pipeline overlap

在上游基础上叠的八条小规则,每条独立可测,合起来让一块固定大小的 VRAM arena 在不重新分配的前提下服务长寿命 agent 流量。共同不变量:启动一次性 allocate,之后只做视图切片和簿记重置。

- **Recurrence-window MoE admission。** `ensure_experts_hybrid` 在 recurrence window 内第二次看到该 expert 时才把它 fetch 到 GPU,首次 miss 走 CPU 计算。Gate 在每层 expert lookup 触发,所以 admission 在 miss 那一刻决定,而不是 periodic sweep。消除 cold prefill 阶段每个 expert 都短暂占一个 GPU slot 的 cache thrash。
- **bs-sized transport routing。** `hybrid_fetch_fraction`(0–1 比率,init 时由 `ft bench bw` 测一次)取代固定 `hybrid_max_fetch` 计数:每个 decode step fetch `fraction × misses` 个 expert,PCIe 饱和而 CPU GEMM 仍然 overlap。bs=1…8 都靠这一条。profile 只在 engine init 读一次——decode 中途重测会抖 TTFT。
- **bs≥4 decode-step prefetch。** `_decode_hybrid` 在本层 GEMM dispatch 之前对 next_layer 调 `prefetch_decode_layer(next_layer, hint)`,让下一层 PCIe 跟当前层 compute overlap。bs<4 路由熵太高,prefetch hit 概率压不过浪费 fetch 的成本,gate 跳过。见 `tests/moe/test_decode_prefetch_gate.py`。
- **KV hot-window。** `FREETOKEN_HOT_WINDOW_TOKENS` 给 budget-slider 看到的 `demand_tokens` 信号加 cap,让 KV 能涨到一个 soft ceiling,而不是纯 shrink-only。Opt-in:不设时严格保留 shrink-only 行为。针对多轮 agent prompt:硬性 shrink-only floor 会把还在用的 prefix pages 踢掉。
- **Prefill/decode interleaving。** `_interleave_every`(默认 4)个连续 prefill chunk 之后 scheduler 强制插一个 decode step。计数单位是 chunk 而非秒,保证 replay 确定性。override:`prefill_interleave_every`。
- **Asymmetric EMA shrink。** `budget_slide.propose_targets` 上行快下行慢:EMA 系数上升时高(burst 期间不要欠配),下降时低(load 回落时不要 pingpong)。`min_step_ratio` deadband 忽略阈值以下抖动;`demand_floor` 防止 shrink-only 跌破下一请求 working set。只在 idle transition 触发。
- **Special-token ckpt,工具调用复用。** `--enable-special-token-ckpt` 让 decode 在 tool-call opener token 处快照 GDN state。该 token 是结构边界——之前(system prompt / few-shot / history)逐轮一致,之后漂移。snapshot 每轮付一次成本,prefix reuse 每次 echo 命中。替代旧的"每 N token ckpt 一次"启发式——后者大多数 ckpt 根本没匹配上。
- **Budget slider(载体)。** `BudgetSlider` 在 idle 时重切 KV pages ↔ MoE slots。`mha_pool.rebuild` / `OffloadMoeCache.rebuild` 从 arena 左端重新切视图;base pointer 不动,CUDA graph 仍有效。Rebalance 只在 idle transition 触发——这是唯一同时没有 live page-index 引用也没有 in-flight graph replay 的时刻。

八条各自独立 gate,关掉或调某一条不影响其他。共享不变量就是 VRAM 启动一次性分配,之后只做 view reslice + bookkeeping reset。

## mm pipeline 三轮迭代

- **Round-4 / P4 — content-key 前缀缓存 + splice 不变量修复。** tokenizer worker 用图像像素负载的 sha1 生成 content id,mm 前缀匹配/插入走 `mm_cache_key`(image_pad 段替换为 content id)——同图复用 KV,异图零误命中。splice 不变量修复:模型要求 mm 请求的所有 image_pad 要么全部在本次 forward 的行内要么整段已被缓存(行内 0 个 pad)。`match_req` 对落在 pad 跨度内的前缀匹配回退到 `align_down(span_start, page_size)` 并丢弃 GDN 快照(防 COW 越界);越过最后一段 pad 的匹配原样放行。验证:`p4_splice_test.py` 21 腿 + `r4_adv_probe.py` 174 对抗。
- **Round-5 — 视频路径 + dual-pad 修复。** video 部件接受帧列表(data:/http/本地路径)或单个动画文件(GIF/WebP/APNG);≤8 帧等间隔采样;多帧走处理器 `videos=` 时间配对(回退逐帧堆叠),`t>1` grid;每个部件仍只渲染一个 pad 占位。checks 中两种 pad 拼写(`image_token_id` / `video_token_id`,均在 vision_start/end 内)都识别,展开统一写 `image_token_id`,模型侧 embed mask / cache key / scheduler 零改动。mp4 PIL 无 ffmpeg 后端时回退 `imageio`(bundled 静态 `imageio-ffmpeg`)整段读入后采样 ≤8 帧。超过 32768 patch 直接返回干净 400(`context_length_exceeded`),streaming 在流内承载 error frame。验证:`vision_video_test.py` / `vision_p3_media_test.py`。
- **Round-6 — mm RAM tier swap-in/out。** 已完成的 mm 请求把 prompt 段 KV + GDN 状态暂存主机内存;后续同内容键请求匹配时把行搬回新分配的 VRAM pages 直接续跑。实现:`snapshot_rows` / `restore_rows` 打包 K/V 压缩块 + turbo4 标尺 + QSA 压缩索引行(`slot // index_ratio`)到 CPU;`index_copy_` 原地散射(`[:, idx].copy_` 会先 gather 成临时量写不进去——踩过)。`pending_ring` 暂态不入 tier。Tier 存储按字节 LRU 淘汰,`FREETOKEN_MM_TIER_MB` budget 默认 1024。Restore gate:content key 逐位相等 + 边界优于树匹配 + 不切断 pad run。验证:`kv_tier_test.py` 在真实 QSA+turbo4 pool 上(行往返/字节核算/同键替换/LRU/pad 边界/GDN 写回)。

## 环境要求

- Windows 11(或 WSL2)x64,NVIDIA GPU ≥ 24 GB VRAM,推荐 ≥ 2× 模型大小的内存。
- Python 3.12(发布 wheel 为 `cp312`;kernel-cache wheel 针对 CUDA 13.x 构建)。

## 安装

源码安装,依赖来自公开索引源。

```powershell
python -m venv ft-venv
ft-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
ft-venv\Scripts\pip install .
```

### 依赖

- `torch`(CUDA 13 build,`cp312`)— PyTorch 官方索引:<https://download.pytorch.org/whl/cu130>
- `freetoken` — 由本仓库安装(`pip install .`);尚未发布到 PyPI。
- 可选预构建 kernel-cache wheel(跳过首个请求的 Triton JIT;必须匹配 CUDA 13.x)— 在本仓库 GitHub Releases 提供。
- 可选 `imageio`(bundled 静态 `imageio-ffmpeg`),为 vision 路径提供 mp4 container 支持。

## 启动服务

```powershell
# 可选:仅当专家权重总量超过 pinned budget(WDDM 下约为 RAM 的一半)时需要
$env:FREETOKEN_PIN_BUDGET_GB = "120"

# turbo4 + 1M KV — 最长上下文,encoder tower 默认加载
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 2048 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d

# turbo4 + 512K KV — 更大的常驻专家缓存
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 524288 --moe-cache-size 3072 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d

# bs=1-dominated 紧凑画像 — 单 graph ladder,MTP off,q4_0 bank,text-only
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model ./models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 65536 --moe-cache-size 1024 `
  --kv-cache-dtype turbo4 --moe-prefill-hit-d2d `
  --cuda-graph-max-bs 1 --moe-prefill-demand-tokens 64 `
  --mtp-expert-quant q4_0 --text-model-only
```

值得理解的几个 flag:

- `--num-tokens` — paged KV pool 大小。turbo4 ≈6.9 KiB/token → 1M ≈6.8 GiB,512k ≈3.4 GiB,65k ≈0.4 GiB。pool 越小,留给 verify peak 的 contiguous VRAM headroom 越多。
- `--moe-cache-size` — GPU 上常驻专家数。steady-state decode 大致随 hit rate 线性提升直到 ~3072,之后变平。
- `--kv-cache-dtype` — 见上面的分级表。
- `--text-model-only` — 跳过所有 encoder tower,它们的 VRAM 还给 KV / expert cache。只想跳过部分种类时改用 `--mm-disable vision audio`。
- `--moe-prefill-hit-d2d` — prefill 命中走 device-to-device copy。
- `--cuda-graph-max-bs N` — CUDA-graph ladder 上限;bs=1-dominated 用 `1`,共享 pool 下加 bs=4 burst slot 成本只几 MB 用 `4`,留空按 VRAM auto-tune。
- `--moe-prefill-demand-tokens` — demand-prefill 的 chunk-row 阈值;默认 64 适配 cached-prefix incremental prefill。

随后用任何 OpenAI 兼容客户端 POST `/v1/chat/completions` 即可。

## 实测数据

参考硬件:RTX 5090 32 GB + ~254 GiB RAM,Qwen3.8-Flash-Next(-ABLITERATED)-NVFP4。

**并发 ladder**(`max_tokens=96`,text-only,`--num-tokens 524288 --moe-cache-size 3072 --kv-cache-dtype turbo4 --moe-prefill-hit-d2d`):

| concurrency | aggregate tok/s | per-stream tok/s | 备注 |
|---|---|---|---|
| 1 | 26.2 | 26.2 | |
| 2 | 47.5 | ~24 | |
| 4 | 66.9 | 17.2 | p50 ≈ 5.0s,p95 5.2s — sweet spot |
| 6 | 64.6 | ~11 | |
| 8 | 68.8 | 9.9 | p95 反弹 |

Sweet spot 是 concurrency=4:从 1→4 batch 收益真实(+155%),从 4→8 只是把同一段 throughput 切得更细。Aggregate soft ceiling ≈65–77 tok/s 受 PCIe bandwidth 与每步 expert-fetch volume 限制,而不是算力。加 `--max-running-requests 8` 让 bs=6 借用捕获好的 bs=8 graph,c6 aggregate 抬到 76.9,代价 p95 ≈7s(c8 tail ≈25s)—只推荐给 throughput-first 场景。

**MoE cache 与 decode 速度**(`ft ctl cache rebuild`,不重启):slots 1024 → 2048 → 3072,long-text steady-state decode ≈62 → 75 → 84 char/s(≈33 → 40 → 44 tok/s;short text ≈52)。累计 **+36% 零质量损失**——cache 存的就是精确权重,增益完全来自 hit rate。~3072 之后曲线变平(再多 +5–8%,但开始吃 KV pool)。

**长上下文 + mixed-decode 长度**(`bench_lc_mixed.py`,prompt ~10k/~50k tokens 共享前缀,每条 stream 输出目标 ∈ {1024, 2048, 3072, 4096},MTP off,`--cuda-graph-max-bs 1`):

| bs | prompt tier | aggregate tok/s | per-stream avg tok/s | TTFT cold → warm |
|---|---|---|---|---|
| 4 | ~10k | ~104 | ~25 | 3.2s → 2.4s |
| 4 | ~50k | ~104 | ~25 | 10.1s → 2.6s |
| 8 | ~10k | ~193 | ~24 | 46.6s → 46.8s |
| 8 | ~50k | ~187 | ~24 | 47.7s → 48.3s |

三条观察:per-stream decode ~25 tok/s **与上下文长度无关**,直到 ~50k context 仍是这样(KV read bandwidth 还不是瓶颈,权重 streaming 主导);aggregate 从 bs=4→bs=8 大约 ×1.85,接近 PCIe bandwidth ceiling;bs=8 TTFT ~47s 来自 chunked-prefill 排队(8 streams × XLONG prompts × ~8192-token chunks),不是 decode 变慢。round-to-round TTFT 下降是 prefix cache 命中效果。

**Vision**(默认开启,vision tower bf16 ≈0.84 GiB):图像请求相对文本仅 **+8ms**(cold TTFT 中位数:文本 1518ms vs 图像 1526ms)。ViT forward 不是主要开销——~1.5s floor 是固定的 MoE expert-cache cold-start,文本同样付。大图(19200 patches)峰值 VRAM +984 MiB,首 token 2.9–4.2s。重复同内容请求从 mm RAM tier  refill;不同图像零误命中(content keys 13/13,adversarial 174/174 全绿)。超过 32768 patches 由 guard 返回干净的 400(`context_length_exceeded`),streaming 模式以流内 error frame 携带。

**长上下文检索**(turbo4 tier):needle 322k/450k × depth {30%, 60%, 90%} 共 6/6 exact,最差 depth-90% 用 35.2s,其余落在 5–12s。实测 KV ≈6.9 KiB/token → full 1M-token pool ≈6.8 GiB;TTFT @ 901k context 32.5s 含 prefill overlap(关掉 overlap 是 51s)。Steady-state decode 长度无关。

## 调优提示

- **Pin ceiling**:WDDM 下驱动将 GPU-accessible pinned host memory 上限设在大致物理内存一半;locked working-set pages 计入同一 budget。`FREETOKEN_PIN_BUDGET_GB` 应保持小于上限减去 locked-layer 份额,并且逐步上调,每次先释放其他 RAM 占用者。
- **Locked ≠ pinned**:locked bank(head/tail layers)在 CPU executor 上解码;pinned bank 流向 GPU。更多 pinned = decode 更快;locking 是 pinned 超过 ceiling 时的兜底。
- 启动 OOM 在最后一个 bank 时:先调小 `--num-tokens`(turbo4 ≈6.9 KiB/token),再调 budget。
- **Verify-phase peak sizing**:bs=4 + `--mtp-draft-len 2` 时 verify peak ≈3–4 GiB contiguous。配合共享 CUDA-graph mempool + bs=1 ladder + `--num-tokens 65536 --moe-cache-size 1024 --mtp-expert-quant q4_0`,steady-state ≈14 GiB allocated,≈16 GiB contiguous free — verify peak 容纳得很舒服。PyTorch 报告 free 比 `nvidia-smi` 少,是因为 CUDA driver context + cuBLASLt workspace + cuDNN pool + CUDA-graph private pool 都在 PyTorch allocator visibility 之外(这条 stack 上典型差 ~3 GiB)。`torch.cuda.empty_cache()` 只在启动时回收这一段;运行时 peak sizing 还得靠压低 steady-state allocated。
- **MTP 默认 off** — `--mtp-draft-len 2` 实测 acceptance ~0.70,在多流 aggregate 上净正向,bs=1-dominated 流量上净负向:verify overhead(~70ms/step)+ redraft(~30ms)超过跳过 forward pass 省下的时间。本分支默认关闭;在多流工作负载里接受单 token latency 略升以换 aggregate throughput 提升时再用 `--mtp-device gpu` 启用。

## Verification 套件

`verification/` 收录每层的回归测试(打包状态下全绿):

| script | covers |
|---|---|
| `p4_splice_test.py` | mm splice invariant(pad-span alignment,COW no-op)— 21 cases |
| `p4_finish_key_test.py` | finish-key + SWA 回归 |
| `p4_content_key_test.py` | content-hash prefix keys(同图复用 / 异图隔离) |
| `mm_prefill_chunk_test.py` | whole-segment mm prefill rollback |
| `r4_adv_probe.py` | adversarial fuzz:cross-span gaps,page ownership,`page_size ∈ {1,4,8}` × shapes × rounds |
| `kv_tier_test.py` | RAM-tier KV swap-in/out round-trip |
| `budget_slide_test.py` | KV/expert pool resize 不重启 |
| `serve_video_smoke.py`、`vision_video_test.py`、`vision_p3_media_test.py` | 端到端 vision/media 请求 |

在 package root 用 venv Python 跑,例如 `python verification/p4_splice_test.py`。

## 环境变量

| variable | meaning | default |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | CUDA-pinned expert banks 上限;超出触发分离驻留(locked CPU layers) | auto(~0.4–0.5 × RAM) |

Encoder toggle 走 CLI flag(`--text-model-only`、`--mm-disable ...`);vision / audio tower 默认加载,禁用时干净跳过。

## Provenance

派生自上游 FreeToken commit `58f4b9e`(release `0.1.2+g816c324d0`)。Windows / quantization / mm 改动按区域记录在 `CHANGELOG.md` 与 round logs;kernel cache wheel 与上游一致。上游 LICENSE(Apache-2.0)与版权声明保留于 `package/`。

第三方 checkpoint Qwen3.8-Flash-Next-ABLITERATED-NVFP4 不附在本仓库内,运行时从对应 Hugging Face hub 下载。
