# 改造记录与设计说明

README 面向使用者（功能、安装、参数）；本文件面向维护者，按轮次记录改造思路、改动文件与验证过程。

版本: freetoken 0.1.3+gafd99cb (merged-src, 打包于 2026-09-21; 上一阶段 P4 = 内容哈希前缀缓存键 + 拼贴(splice)不变量修复, 见 Round-4~6)

## 本阶段功能 (P4)
1. 内容键缓存: tokenizer worker 用图像像素负载的 sha1 生成 content id, mm 请求的前缀
   匹配/插入均走 mm_cache_key (image_pad 段替换为内容 id) —— 同图复用 KV, 异图零误命中。
2. 拼贴不变量 (Round-4 修复, 生产崩溃回归): 模型要求 mm 请求的所有 image_pad 要么全部
   在本次 forward 的行内, 要么整段已被缓存 (行内 0 个 pad)。match_req 对落在 pad 跨度
   内的前缀匹配回退到 align_down(span_start, page_size) 并丢弃 GDN 快照 (防 COW 超进);
   越过最后一段 pad 的匹配原样放行 (P4 全量复用保留)。文本/无键 mm 路径逐位不变。
3. 配套放宽: qwen4_exp / gemma4 的 splice 在 n_pads==0 时为合法 no-op, 中间态仍 raise。

## 本阶段(第4轮)改动文件
- freetoken/scheduler/cache.py        (match_req: mm_span + _mm_bound 上限, 三个分支)
- freetoken/models/qwen4_exp/model.py (forward splice: scatter / raise / no-op)
- freetoken/models/gemma4/model.py    (_merge_multimodal: 同上放宽)

## 验证链 (全部通过, 复核于打包当日)
- p4_splice_test.py        21/21  (生产崩溃复现腿: 轨点960跨中段 -> 回退8, mamba丢弃, 955 pad全入行)
- p4_finish_key_test.py    ALL PASS (含 SWA 41 腿与回退-重验)
- p4_content_key_test.py   13/13 (同图复用 / 异图零误命中)
- mm_prefill_chunk_test.py 10/10 (mm 整段回滚)
- vision_p3_media_test.py  ALL PASS
- r4_adv_probe.py          174/174 对抗核查 (双跨段 gap/第2段/越过段, COW no-op 指纹,
                               页所有权零重复, ps in {1,4,8} x 3 形状 x 4 回合不变量模糊,
                               文本无上限 + 无键 mm 无复用)
- 重启后复验: 5 套件 + 174 项对抗全绿, 三处改动仍在安装包内。

## 复现方法 (Windows, CPU 即可)
在包根目录用安装后的 venv Python 运行: `python -X utf8 verification/p4_splice_test.py`
(其余套件同法; 脚本头部 sys.path 指向安装包 site-packages)

## 已知语义 (非缺陷, 前几轮已确认)
- 页对齐匹配: 匹配/插入边界按 page_size 向下取整 (预存在语义, 文本对照组证实与 P4 无关)。
- 探针指纹池为 fp32: 前缀和 > 2^24 时按 fp32 舍入 (比对需经 fp32 往返)。
- mm 永不参与 chunked prefill; GDN x64 轨点可落于 pad 跨段内 —— 正是本阶段修复的触发源。

## 清单
MANIFEST.sha256 为包内全部文件的逐文件 SHA256; zip 整体摘要见 SHA256SUMS.txt。

## 增补 (第5轮, 2026-09-13): 视频路径 + 双 pad 修复
1. 视频部件端到端: video 部件 = 帧列表(data:/http/本地路径)或单个动画文件(GIF/WebP/APNG);
   ≤8 帧等间隔采样; 多帧走处理器 videos= 时间配对(回退逐帧堆叠), t>1 grid; 每个部件仍只
   渲染一个 pad 占位, 展开数取自真实 grid; 32768 patch 预算覆盖整段视频。
2. 双 pad 修复: checkpoints 对 image/video 部件编码不同 pad (image_token_id / video_token_id,
   均包在 vision_start/end 内); prepare 检测两拼写, 展开统一写 image_token_id,
   模型侧 embed mask / cache key / scheduler 零改动。vision_geometry 透传 video_token_id。
3. extract_media_parts 镜像模板分支顺序 (image/image_url 键或 type==image; video 键或
   type==video), image_url 的 {"url":...} 自动解包。
4. mp4 容器: PIL 无 ffmpeg 后端时识别不了 mp4; fetch_video_frames 增加 imageio/ffmpeg
   回退 (imageio-ffmpeg 自带静态 ffmpeg), 整段读入后采样 ≤8 帧。
验证: vision_video_test.py / vision_p3_media_test.py 全绿 (HF oracle 逐位一致); live serve
(:8001, --vision-on) 冒烟过: 帧列表/GIF/image_url+video 混合/mp4 直传四种形态全部正确。

## Round-6 (2026-09-14): KV 换入换出 — mm RAM tier

目标: 重复图像/视频请求不重算媒体段。已完成的 mm 请求把 prompt 段 KV + GDN 状态暂存到主机内存
(RAM tier); 后续同内容键请求在匹配时把行搬回新分配的显存页, 直接续跑。

改动文件:
- kvcache/mha_pool.py / qsa_pool.py: snapshot_rows / restore_rows — K/V 压缩块 + turbo4 标尺 +
  QSA 压缩索引行 (slot//index_ratio) 按 token-slot 行打包到 CPU / 写回。index_copy_ 走原地散射;
  [:, idx].copy_ 会先 gather 成临时量写不进去 (踩过)。pending_ring 是暂态不入 tier。
- kvcache/linear_state_pool.py: snapshot_state / restore_state — conv+recurrent+slot_states
  整槽打包 / 写回。
- kvcache/base.py: MatchResult 增 gdn_host 字段。
- scheduler/cache.py: CacheManager mm tier — _tier_store(完成时; 边界取最近 ×CHUNK 轨道, KV/state
  长度严格配对否则跳过; LRU 按字节淘汰, FREETOKEN_MM_TIER_MB 默认 1024) / restore_mm(匹配时;
  内容键逐位相等且边界优于树匹配、不切断 pad run → 新页 + 行回填 + _TierHandle)。
- scheduler/prefill.py / scheduler.py / scheduler/utils.py / core.py: restore 接入 admission
  (预算门后), mm_gdn 经 PendingReq→Req 透传, _restore_linear_states 首个 prefill forward 写回。

验证: kv_tier_test.py 全绿 (CPU, QSA+turbo4 真实池: 行往返/字节核算/同键替换/LRU/pad 边界/GDN
写回); vision 两套回归全绿; live serve (:8001) 冒烟全绿 — 同内容重复请求走 restore 路径答对,
image_url+video 混合腿正确判定 (绿静图 ∉ {红,蓝} 动画), prompt_tokens=238。
注: reasoning 腿预算需 ≥512 (推理文本可超 500 token, content 在其后)。

## Round-7 (2026-09-21): 0.1.3 merged-src — 新模型家族 + mm 管线独立化 + 量化/MoE/权重流式框架

来源: freetoken-0.1.3+gafd99cb-merged-src-2026-09-21.zip，覆盖到 python/ 源码路径（含随包的
tvm_ffi_cpp_extension_py/，对应 apache-tvm-ffi==0.1.13.post3 的 C++ 扩展）。共约 165 个文件变更。

### 本阶段功能

1. 新模型家族
   - glm5_next (GLM-5.3-Flash): 完整支持 (attention/moe/mlp/kda/vision/weight)。线性注意力走
     KDA fused recurrent 内核 (kernel/fla/: fused_recurrent、kda、chunk_delta_h、solve_tril)；
     残差流采用 mHC — Manifold-Constrained Hyper-Connections (layers/mhc.py + Triton fused 内核
     kernel/triton/mhc.py)：hc_mult 条并行流由单次 fp32 GEMM 产生 pre/post/comb 三组混合映射，
     comb 经 softmax + Sinkhorn 近似双随机投影（"流形约束"：混合既不放大也不损失残差质量）。
     torch 实现为正确性基线，Triton 内核逐位对齐后接管 CUDA 路径。
   - qwen3_vl: 新增视觉语言模型支持 (config/model/vision/weight)。
   - minimax_m3、muse_glimmer: 各补 vision.py，接入多模态路径。

2. kpool DSA 索引 (attention/dsa_indexer_kpool.py + kernel/triton/kpool_compress.py)
   indexer K 缓存按 pool 粒度打分：每 index_kpool(=4) 个连续 token 折叠为一条 entry（逐通道
   softmax(gate+APE) 加权和），indexer 计算与 top-k 缩小 4 倍；选择 index_topk//kpool 个 pool 后
   展开回 token 行，请求尾部不满 pool 强制选入 (index_kpool_always_select_tail)，稀疏 MLA 内核用
   -1 gather-only 哨兵补齐宽度 (kpool-1) + select_k*kpool。

3. 多模态管线独立成包 (mm/: config、media、processor+processors、encoder_cache)
   调度侧新增 scheduler/mm.py：mm chunk 规划（gather 哪些 embedding 行、编码哪些 item），chunk
   边界在切断 image span 前回撤到 align 倍数——延续 Round-4 的 pad 不变量。新增启动参数:
   --mm-disable / --mm-embed-cache-device / --mm-encoder-weights / --mm-processor-kwargs /
   --image-min-tokens / --image-max-tokens / --allowed-media-domains /
   --allowed-local-media-path / --text-model-only。

4. 量化框架 (layers/quantization/)
   registry + scheme + backend 三层结构。linear: fp8_block / fp8_tensor / mxfp8 / nvfp4 /
   unquantized；moe: fp8_block / mxfp4 / mxfp8 / nvfp4 / unquantized；configs: compressed_tensors /
   fp8 / modelopt / mxfp4。启动参数 --quant-backend。

5. MoE 主机 bank 与权重流式
   - moe/expert_pieces.py: checkpoint 侧 build_expert_banks 的流式 piece 格式
     (bank_layer, e0, e1, 各投影含 _scale/_global 伴随量或已融合 gate_up)，由 expert kernel 的
     pack 决定 bank 布局；moe/legacy_format.py 兼容旧格式。参数 --moe-backend / --moe-strategy。
   - models/weight_stream.py: 块权重从 pinned host bank 经双 staging buffer 流式上传——block i
     计算时 copy stream 并行填充 block i+1。
   - qwen4_exp/ple_disk.py + kernel/csrc/ple_store (C++ 扩展): PLE 落盘后端，参数 --ple-backend。

6. API 层
   - responses_api: 输入项支持结构化 content（list 形态）；tool 输出含图像时，图像挂在随后的
     user 轮上（chat template 只渲染文本，与 Anthropic 路径行为一致）；GenerationError 透传
     HTTP 状态码。
   - anthropic_api / openai_api / tokenizer server 相应适配与精简。

### 与本仓库既有约定的衔接
- 版本号唯一来源仍是 python/freetoken/version.py (0.1.3)，pyproject 经 attr 指令读取。
- kernel cache wheel (freetoken-kernel-cache/) 保持独立打包，不随本 zip 变动。
