# FreeToken — Windows 单 GPU 构建

英文版见 [README.md](README.md)。

基于 [FreeToken](https://github.com/FlashML-org/FreeToken) 的补丁分支（基线：`0.1.2+g816c324d0`，上游 main 合并至 `afd99cb`），针对**单张消费级 GPU + 大容量系统内存**的场景调优——例如 RTX 5090 32 GB 搭配约 250 GB 内存——通过兼容 OpenAI / Anthropic 的 HTTP API 服务大型 MoE 模型。

上游采用 Apache-2.0 许可证；本分支完整保留其 LICENSE 与版权声明，见 [Provenance](#provenance)。

## 为什么做这个分支

上游 FreeToken 已经面向 MoE-offload 推理，但 Windows 路径和 KV 存储选项较薄弱。本分支解决的问题可以串成一条主线：KV 分级存储（turbo4 每 token 约为 bf16 的 1/4）腾出显存给 MoE 专家缓存；专家缓存越大，热专家越能留在 GPU 上；开启视觉支持后，内容哈希前缀缓存 + mm 驻留区让重复的图像/视频不再重新 prefill——于是单张 32 GB 显卡可以同时跑大 MoE、百万级上下文和大量图像/视频历史。具体来说，本分支新增或加固了：

- **Windows 兼容性** — WDDM 下的工作集 / 锁定内存预算处理、基于 `ctypes` 的页面锁定（`VirtualLock`）替代仅 POSIX 可用的 `resource` 模块，以及 `expandable_segments` 与 zmq/Proactor 事件循环差异的优雅降级路径。
- **专家权重组的分离驻留** — 当 CUDA 锁定内存预算小于专家权重总量时，头部/尾部 MoE 层以 OS-lock 方式常驻内存并在多线程 CPU 执行器上解码（AVX-512 BF16 + VNNI，NVFP4 W4A8），其余权重组则锁定在 pinned 内存中供 GPU 流式读取。预算可通过 `FREETOKEN_PIN_BUDGET_GB` 调节。
- **paged KV 池的存储侧量化**（QSA 与 MLA/DSA 系列）：三档存储精度，基于对 bf16 参考注意力的往返测量（sparse + split-k 路径的最大余弦相似度）：

  | 档位 | 字节 / token / KV 层 | 与 bf16 的余弦相似度 | 适用场景 |
  |---|---|---|---|
  | `bf16` | 2048 | 1.0000 | 默认；较短上下文 |
  | `fp8_e4m3` | 1024 | 0.9994 | 一般平衡点 |
  | `turbo4` | ~516 | 0.9887 | 超长上下文（10 万 token 以上） |

  仅压缩存储，注意力计算保持 bf16。索引器 key 始终为 bf16。
- **KV 本体下沉宿主内存（`--kv-offload`）** — turbo4 之后每 token KV 已压到 ~7 KiB，但 1M 上下文仍需 ~6.8 GiB，和 expert cache、prefill 激活挤在同一张 32 GB 卡上（GLM 实测 boot 后 free 仅剩 ~3.3 GiB，首个 prefill 的激活峰值贴着边界）。而 KV 的访问是稀疏的：sparse indexer 每步打分只读压缩索引行，bulk 内容行只在被 top-k 点名的位置才被读——把低频读的 bulk 行也常驻显存是过度配置。`--kv-offload` 把 K/V body（paged 池的 bulk 行 + turbo4 dequant scale）落到 pinned 宿主内存，attend/gather kernel 经 UVA device pointer 直读 host 行、只在 top-k 位置跨 PCIe；显存侧只保留每步必读的索引层（QSA compressed-key slab、per-request pending ring、scratch 行、mrope 三轴位置表）。定价拆两本账（`kv_cost` 在 offload 时每 token 只计索引项、`host_bytes_per_token` 把 body 记进 host 账），于是同样的 `--num-tokens` 下显存侧 KV 占用降到 ~1/9（Qwen 7.0→0.78 KiB/token）、~1/5（GLM 3.4→0.7 KiB/token），省出的余量直接喂给 MoE 专家缓存。Windows 上 host tier 走 **register-direct**（mmap→commit→cudaHostRegister，与专家 banks 同机制）绕开 TCC 驱动对 `cudaHostAlloc` 的 fresh+registered 合并记账顶——否则专家 banks + PLE 表先注册 ~97 GiB 后，KV-tier 时点的 fresh 分配注定失败，且那次失败的 mapped 分配会连带后续小额显存分配在显存富余时 OOM（实测 conv_states 44 MiB vs 8.61 GiB free 复现）。§8（`README_BRANCH` 的 Stage D）有完整机制与实测。
- **多模态（视觉）支持**：视觉塔按需加载；图像请求通过内容哈希前缀 key 复用 KV（相同图像 → KV 复用，不同图像 → 零误命中），多模态的 KV / 状态快照存放在有上限的内存驻留区（`FREETOKEN_MM_TIER_MB`，建议保持 4096），与 GPU 池之间换入换出。两个方向都是分块的（每块 32768 token），保证恢复时不会出现撑爆紧张显存的大额临时分配。
- **驻留区恢复不再漏页** — 从内存驻留区恢复时，请求会为自己的一整段前缀申请全新的页，而缓存树里同时还留着自己的一份匹配副本。现在每种缓存句柄都会报告自己的回收底线（`commit_floor()`）：树句柄返回 `cached_len`（借自树的那段归树所有），恢复句柄返回 0（整段都是请求自己的，重复的那部分在结束时必须释放）。这修掉了每次驻留区恢复稳定漏 304 页的问题，并且与模型形态无关——六种缓存句柄全部从基类继承该方法，已逐一验证。
- **Qwen3.8-Flash-Next 服务配置** — QSA paged-KV 池配合上述分级存储、专家权重经 offload 缓存走 NVFP4 路线，以及由 `verification/` 套件端到端覆盖的多模态内容 key 前缀缓存。

## 环境要求

- Windows 11（或 WSL2）x64，NVIDIA GPU ≥ 24 GB 显存，建议内存 ≥ 模型体积的 2 倍。
- Python 3.12（`cp312`；可选 kernel-cache wheel 针对 CUDA 13.x 构建）。

## 安装

本仓库为纯源码分发，依赖从公开索引获取。

```powershell
python -m venv ft-venv
ft-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
ft-venv\Scripts\pip install .
```

### 依赖说明

- `torch`（CUDA 13 构建，`cp312`）— PyTorch 官方索引：<https://download.pytorch.org/whl/cu130>
- `freetoken` — 由本仓库安装（`pip install .`），暂未发布到 PyPI。
- 可选：预编译 kernel-cache wheel（跳过首次请求的 Triton JIT；须匹配 CUDA 13.x）— 在本仓库的 GitHub Releases 页下载。
- 可选：`imageio`（自带静态 `imageio-ffmpeg`），用于视觉路径的 mp4 容器支持。

## 启动服务

参考机（RTX 5090 + Qwen3.8-Flash-Next-NVFP4，agent 类流量、前缀复用率高）验证过的启动组合：

```powershell
$env:FREETOKEN_MM_TIER_MB = "4096"
ft-venv\Scripts\python.exe -m freetoken `
  --model-path E:\FreeToken\models\Qwen3.8-Flash-Next-ABLITERATED-NVFP4 `
  --num-tokens 1048576 --moe-cache-size 4096 --kv-cache-dtype turbo4 `
  --moe-prefill-hit-d2d --kv-offload --host 0.0.0.0 --port 8001 --max-running-req 4
```

（`--kv-offload` 把 KV bulk 行下沉宿主内存后，1M 档显存不再被 KV 吃满，专家缓存才能从 3072 上探到 4096；不加这个 flag 时 1M + 4096 会 OOM。）

上下文长度和专家缓存两个参数互相挤占，实测可用的搭配表（均为 `--max-running-req 4`）：

| 上下文（`--num-tokens`） | 专家缓存（`--moe-cache-size`） |
|---|---|
| 1048576（1M） | 3072（或 2048） |
| 524288（512K） | 4096 |
| 262144（256K） | 5120 |

1M 上下文下 `--moe-cache-size 4096` 会 OOM，3072 就是这张卡的顶。每次只改一个参数，重启后先看状态再改下一个。

参数说明：

- `--kv-cache-dtype turbo4` — 见上文档位表；1M 的 KV 池靠它压到约 7 GiB（bf16 要 ~28 GiB）。
- `--moe-cache-size` — GPU 侧专家缓存槽位数，按上面的表随上下文长度选取。
- `--moe-prefill-hit-d2d` — prefill 命中时直接在显卡内部拷贝，不再从内存重取。
- `--kv-offload` — K/V 本体（bulk 行 + turbo4 dequant scale）驻 pinned 宿主内存，显存侧只留索引层（每 token 只计 780 B 的 QSA 索引 shadow）。1M profile 靠它把显存让给 MoE；Windows 走 register-direct 绕开 TCC 记账顶。机制与实测见"为什么做这个分支"里的 kv-offload 条与 `README_BRANCH` §8。
- `--max-running-req 4` — 与已捕获的计算图尺寸 [1, 2, 4] 匹配；bs=8 的捕获在这张卡上有风险，因为剩余约 1 GiB 显存还要覆盖每步中间结果、驻留区分块换入的临时缓冲和视觉编码峰值。

之后按任意 OpenAI 兼容服务器的方式调用 `POST /v1/chat/completions` 即可。

## 运行时是什么样的

- **显存**：稳态约 31.2–31.4 GiB（可用约 31.8 GiB）。剩下的 ~1 GiB 不是富余，而是每步中间结果、驻留区分块换入临时缓冲、视觉编码峰值的工作空间，不要再往上面叠更大的批。
- **批量大小**：agent 类流量（少量并发会话、轮流发言）大多 bs=1 加一个短队列——调度器在 prefill 步和 decode 步之间交替，排队的请求在下一个 prefill 步边界加入，所以批只在边界处跳一下，中间保持平稳。这是流量形态决定的，不是引擎缺陷（vLLM/SGLang 面对错峰到达也是同样表现）。想要更大的批，让客户端同时到达（并行扇出），而不是调大 `--max-running-req`。
- **前缀复用**：稳定的 agent 系统提示 + 历史下，命中率稳在 96–97%；热请求从缓存树/驻留区回填，秒级返回。

## 实测数据

以下均为参考机实测（RTX 5090 32 GB + ~254 GiB 内存，Qwen3.8-Flash-Next-NVFP4，用上面的组合启动）。

解码速度随实际批大小（聚合完成 tok/s）：

| 批大小 | 聚合 tok/s | 说明 |
|---|---|---|
| 1 | ~43–50 | 错峰 agent 流量的稳态 |
| 2 | ~48 | |
| 4 | ~85 | 当前预留下的实际上限 |
| 8 | ~111 | 只在余量更大的启动里达到；当前预留下有风险 |

bs=1 到 bs=4 的增益是真实的；再往上就进入显存预留区，计算图捕获开始和运行时临时分配互相挤占。

TTFT：冷请求大致符合 `token 数 ÷ 3400` 秒（例如 ~16k token 的首轮 ≈ 5 s；~900k token 实测 32.5 s，开 prefill overlap）。命中缓存树/驻留区的请求秒级返回。needle 检索 322k/450k token × 深度 30/60/90% 共 6/6 全中（最差 90% 深度 35.2 s，其余 5–12 s）；解码稳态速度与上下文长度无关。

MoE 缓存阶梯（`--num-tokens` 保持配对值）：槽位 1024 → 2048 → 3072，长文稳态解码 ≈33 → 40 → 44 tok/s（短文 ≈52），累计 **+36%，质量零损失**——缓存存的是精确权重，增益全部来自命中率。超过配对值后曲线合拢（再挤一点收益，但要挤占 KV 池），到此为止。

KV offload（`--kv-offload`，1M 上下文）：Qwen 显存侧索引 shadow 780 B/token（≈0.72 GiB，vs 全价 ~6.8 GiB），body 下沉 pinned host 6.00 GiB + dscale 0.09 GiB；graph capture bs=[1,2,4] 完成，首个 bs=4 prefill 时 free 4.13 GiB——省出的余量正是要留给 MoE 专家和 prefill 激活。GLM 同档：boot 日志 `K + V = 0.69 GiB`（≈700 B/token，全价 ~3.4 GiB 的 ~1/5），host body 2.75 GiB。host tier 就位（total mapped ~102.8 GiB）后 graph capture 的大额显存分配照常成功，验证了 Windows 侧 register-direct 绕开 TCC 记账顶这条路径的正确性。

视觉：带图请求相对纯文本仅 +8ms 左右——大头不是 ViT 前向，而是 MoE 专家缓存冷启动的固定开销，文本同样承担。大图（19200 patch）显存峰值 +984 MiB，首 token 2.9–4.2s。同内容重复请求走内存驻留区回填，不同图像零假命中（内容键 13/13 + 对抗 174/174 全绿）。超 32768 patch 守卫返回干净的 400（`context_length_exceeded`），流式模式以流内 error 帧承载。

漏页检查：修复回收底线后，40 轮 × 3 请求的 agent 模拟浸泡测试（共享增长前缀、分块 prefill、驻留区存取、逐出压力、中断——`leak_sim.py`）在每一步操作后都能把每个页对上账，线上状态跨重启也保持平稳。

## 调优笔记

- **锁定上限**：WDDM 下驱动对 GPU 可访问的 pinned 宿主内存上限约为物理内存的一半；locked（工作集）页与 pinned 页共享同一预算。`FREETOKEN_PIN_BUDGET_GB` 应设为上限减去 locked 层占用之后的余量以下，并可在释放其他内存占用后逐步上调。
- **locked ≠ pinned**：locked 权重组（头/尾部层）在 CPU 执行器上解码；pinned 权重组向 GPU 流式供数。pinned 越多解码越快；当 pinned 会超出上限时，locked 是兜底。
- 若最后一个权重组启动时 OOM，先调小 `--num-tokens`（turbo4 下 1M token 的 KV 池也只有 ~7 GiB），再按配对表重选专家缓存，最后调预算。
- 驻留区大小：`FREETOKEN_MM_TIER_MB=4096` 只管系统内存侧的快照存储；它不占显存——显存成本只是恢复瞬间的分块临时缓冲。

## 验证套件

`verification/` 收录每个阶段把关的回归测试（打包状态下全部通过）：

| 脚本 | 覆盖内容 |
|---|---|
| `p4_splice_test.py` | mm splice 不变量（pad-span 对齐、COW no-op）— 21 用例 |
| `p4_finish_key_test.py` | finish-key + SWA 回归 |
| `p4_content_key_test.py` | 内容哈希前缀 key（同图复用 / 跨图隔离） |
| `mm_prefill_chunk_test.py` | 整段 mm prefill 回滚 |
| `r4_adv_probe.py` | 对抗性模糊测试：跨 span 间隙、页所有权，`page_size ∈ {1,4,8}` × 形状 × 轮次 |
| `kv_tier_test.py` | RAM 层级 KV 换入/换出往返 |
| `serve_video_smoke.py`、`vision_video_test.py`、`vision_p3_media_test.py` | 端到端视觉/媒体请求 |

另有两项仅 CPU 即可对包内代码运行的账务检查：浸泡测试（每步操作后每个页恰好属于 空闲/树/在跑请求 三者之一）和六种缓存句柄的 `commit_floor()` 形态检查。

在包根目录用 venv Python 运行，例如 `python verification/p4_splice_test.py`。

## 环境变量

| 变量 | 含义 | 默认值 |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | CUDA-pinned 专家组的预算上限；超出后触发分离驻留（locked CPU 层） | 自动（约为物理内存的 0.4–0.5 倍） |
| `FREETOKEN_MM_TIER_MB` | mm 内存驻留区上限（宿主侧多模态快照存储） | 4096 |
| `FREETOKEN_LOAD_VISION` | 置 `1` 时加载视觉塔 | 关闭 |
| `FREETOKEN_FP8_GPU_RESERVE_GB` | FP8 源 bank 预算里为 CUDA context + loader 暂存 + 基线余量留出的宿主内存（GB） | 24 |
| `FREETOKEN_DISABLE_GLUON` | 置 `1` 时禁用 Gluon SM120 NVFP4 kernel，回落到 Triton 路径（W4A4 关闭） | 关闭 |

## Provenance

派生自上游 FreeToken 提交 `58f4b9e`（发布 `0.1.2+g816c324d0`），并在其之上合并上游 main 至 `afd99cb`（mm/MMItem + 分段表 rope 重构，与本分支的 KV-tier / mm-tier 路径并存）。Windows / 量化 / 多模态改动按区域记录在 `CHANGELOG.md` 与各轮次日志中；kernel cache wheel 与上游一致。上游 LICENSE（Apache-2.0）与版权声明在 `package/` 中原样保留。

配套使用的第三方 checkpoint（Qwen3.8-Flash-Next-ABLITERATED-NVFP4）不包含在本仓库中，运行时从其 Hugging Face hub 下载。
