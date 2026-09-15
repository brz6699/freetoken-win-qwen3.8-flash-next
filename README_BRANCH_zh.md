# FreeToken — Windows 单 GPU 构建

英文版见 [README.md](README.md)。

基于 [FreeToken](https://github.com/FlashML-org/FreeToken) 的补丁分支（基线：`0.1.2+g816c324d0`，上游 main `58f4b9e`），针对**单张消费级 GPU + 大容量系统内存**的场景调优——例如 RTX 5090 32 GB 搭配约 250 GB 内存——通过兼容 OpenAI / Anthropic 的 HTTP API 服务大型 MoE 模型。

上游采用 Apache-2.0 许可证；本分支完整保留其 LICENSE 与版权声明，见 [Provenance](#provenance)。

**单机实测数据** — 单张 RTX 5090（32 GB）+ 256 GB DDR5（4×64 GB）、PCIe 5.0 通道，服务 Qwen3.8-Flash-Next：

- **100 万 token 上下文窗口**（turbo4 KV 档位）
- **3072 个专家常驻** GPU 内 MoE 缓存
- **4 路并发**稳定运行
- **连续识别 400 张图像**全流程通过（内容哈希 mm KV 复用）
- **单路解码 40–50 tokens/s**

## 为什么做这个分支

上游 FreeToken 已经面向 MoE-offload 推理，但 Windows 路径和 KV 存储选项较薄弱。本分支解决的问题可以串成一条主线：KV 分级存储（turbo4 每 token 约为 bf16 的 1/4）腾出显存给 MoE 专家缓存；专家缓存越大，热专家越能留在 GPU 上；开启 `--vision-on` 后，内容哈希前缀缓存 + mm RAM 层级让重复的图像/视频不再重新 prefill——于是单张 32 GB 显卡可以同时跑大 MoE、十万级以上上下文和大量图像/视频历史。具体来说，本分支新增或加固了：

- **Windows 兼容性** — WDDM 下的工作集 / 锁定内存预算处理、基于 `ctypes` 的页面锁定（`VirtualLock`）替代仅 POSIX 可用的 `resource` 模块，以及 `expandable_segments` 与 zmq/Proactor 事件循环差异的优雅降级路径。
- **专家权重组的分离驻留** — 当 CUDA 锁定内存预算小于专家权重总量时，头部/尾部 MoE 层以 OS-lock 方式常驻内存并在多线程 CPU 执行器上解码（AVX-512 BF16 + VNNI，NVFP4 W4A8），其余权重组则锁定在 pinned 内存中供 GPU 流式读取。预算可通过 `FREETOKEN_PIN_BUDGET_GB` 调节。
- **paged KV 池的存储侧量化**（QSA 与 MLA/DSA 系列）：三档存储精度，基于对 bf16 参考注意力的往返测量（sparse + split-k 路径的最大余弦相似度）：

  | 档位 | 字节 / token / KV 层 | 与 bf16 的余弦相似度 | 适用场景 |
  |---|---|---|---|
  | `bf16` | 2048 | 1.0000 | 默认；较短上下文 |
  | `fp8_e4m3` | 1024 | 0.9994 | 一般平衡点 |
  | `turbo4` | ~516 | 0.9887 | 超长上下文（10 万 token 以上） |

  仅压缩存储，注意力计算保持 bf16。索引器 key 始终为 bf16。
- **多模态（视觉）支持**，由显式的 `--vision-on` 开关控制：视觉塔按需加载；图像请求通过内容哈希前缀 key 复用 KV（相同图像 → KV 复用，不同图像 → 零误命中），多模态 KV 页存放在 RAM 层级，与 GPU 池之间换入换出。
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

```powershell
$env:FREETOKEN_PIN_BUDGET_GB = "120"   # WDDM 下锁定内存上限约为物理内存的一半
ft.exe serve --host 0.0.0.0 --port 8001 `
  --model-path ./models/<checkpoint> `
  --kv-cache-dtype turbo4 --moe-cache-auto --vision-on
```

- `--kv-cache-dtype` — 见上文档位表。
- `--moe-cache-auto` — 依据剩余内存自动设定 GPU 侧专家 LRU 与 paged KV 池大小。
- `--vision-on` — 加载视觉塔（约 1 GiB bf16）；纯文本服务可省略。

之后按任意 OpenAI 兼容服务器的方式调用 `POST /v1/chat/completions` 即可。

## 调优笔记

- **锁定上限**：WDDM 下驱动对 GPU 可访问的 pinned 宿主内存上限约为物理内存的一半；locked（工作集）页与 pinned 页共享同一预算。`FREETOKEN_PIN_BUDGET_GB` 应设为上限减去 locked 层占用之后的余量以下，并可在释放其他内存占用后逐步上调。
- **locked ≠ pinned**：locked 权重组（头/尾部层）在 CPU 执行器上解码；pinned 权重组向 GPU 流式供数。pinned 越多解码越快；当 pinned 会超出上限时，locked 是兜底。
- 若最后一个权重组启动时 OOM，先调小 `--num-tokens`（turbo4 在 512k token 下可将 KV 池压在约 3 GiB 以内），再调预算。

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

在包根目录用 venv Python 运行，例如 `python verification/p4_splice_test.py`。

## 环境变量

| 变量 | 含义 | 默认值 |
|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | CUDA-pinned 专家组的预算上限；超出后触发分离驻留（locked CPU 层） | 自动（约为物理内存的 0.4–0.5 倍） |
| `FREETOKEN_LOAD_VISION` | 置 `1` 时加载视觉塔（`--vision-on` 会设置它） | 关闭 |

## Provenance

派生自上游 FreeToken 提交 `58f4b9e`（发布 `0.1.2+g816c324d0`）。Windows / 量化 / 多模态改动按区域记录在 `CHANGELOG.md` 与各轮次日志中；kernel cache wheel 与上游一致。上游 LICENSE（Apache-2.0）与版权声明在 `package/` 中原样保留。

配套使用的第三方 checkpoint（Qwen3.8-Flash-Next-ABLITERATED-NVFP4）不包含在本仓库中，运行时从其 Hugging Face hub 下载。
