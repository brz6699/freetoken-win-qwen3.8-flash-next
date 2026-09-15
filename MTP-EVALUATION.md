# FreeToken Qwen3.8-Flash 的 MTP 补丁评估报告(重评版 v2)

> 评估对象与范围(重评说明)
> 头一版报告(下称 v1)针对的是 `python/freetoken/`(09-03 快照)那套较简实现。**此后 MTP 架构有迭代**,
> 权威实现已在前一版之后的 `freetoken-0.1.2+g816c324d0-windows-patched-src/freetoken/`(mtp.py 09‑07、scheduler.py 09‑07、
> weight.py 09‑07、model.py 09‑07、engine.py 09‑07)推进到 v2。**本报告通读并仅评估该 v2 快照树**,
> 并逐条回答"v1 报告的 ①~⑤ 在 v2 里解决了没有 / 仍存 / 新增"。评估范围:**仅静态评估,不修改代码**。
> 证据以「文件(v2 路径):行号」标注。
>
> 一句话总述(v2):这套 MTP 是一次**明显的迭代硬化**——v1 里"草案只看自己、靠 collapse/repeat、路由未对齐训练口径、
> GPU/host 内存不分摊、页表尾部泄漏、逐步多 host 同步"这些点 v2 基本都动了;它从"功能可用"演进成"参考实现(EAGLE /
> vLLM `Qwen4ExpMultiTokenPredictor`)对齐的工程版"。**剩下来仍需人盯的,是 v1 已指出的那两处对正确性的硬依赖
> (GDN 回滚等价性、`_last_residual` 共享时序),v2 用更细的簿记缓解但没从根上消除;外加若干 v2 新增的簿记口径必须定点回归。**

---

## 0. 评估基线:确认清楚你让我重读的是"哪棵树"

**先摆事实,避免按错版本写结论**:

- `python/freetoken/models/qwen4_exp/mtp.py`(09-03):v1 评估的较简实现。**不含** `write_ctx_kv / write_rows /
  attach_embed / gpu_bytes / fused_topk / _ctx_pending`,路由用裸 `torch.topk`,草案 residual 走
  `mtp_out.repeat(1,hc)`。逐字节检测:9 个月度符号在 python/ 里命中数 = 0。
- 快照 `...patched-src/freetoken/`(09-07):**v2 权威实现**,上述所有符号都在,且 scheduler 也更新。

→ 你的"架构有更新,重新读取评估"落在**快照树**。下文全部以 v2 快照为准;python/ 工作区那份是落后版(要用时再整体拷过来)。

---

## 1. v1 → v2 :这次"更新"到底改了什么(逐点)

| # | v1(09‑03,已评估) | v2(09‑07,本次评估) | 代码证(v2) |
|---|---|---|---|
| A | 草案第 j 格只 attend 自身;长程全靠"(collapse 后再 repeat 的)residual + token id" | 草案 **K/V 走 scratch 槽**,`batch.mtp_extra_locs` 把前序草案的真实槽位喂给后续草案 → **草案能看到前序草案的注意力上下文**(EAGLE 式);且 residual 链**不再 collapse/repeat**,`multi_hidden [T,hc*hidden]` 逐字续到下一位 | mtp.py:234-262,458-462,523-532;scheduler.py:1162-1186 |
| B | 路由:裸 `torch.topk(logits)`,再按"选中 logits 之和"renorm(其注释:跨零时爆炸 → 实测 acceptance=0 / mlp std~1e9) | 改为与主 MoE 同口径 `fused_topk`(全专家 softmax→topk→renorm);Windows 走 pure‑torch 回退,位一致 | mtp.py:134-149 |
| C | MoE 专家仅在 pinned host‑RAM,**gather 每草案把 ~100MB host→GPU 拉一次** | 新增 **GPU‑resident 模式(`bank.load(device=…)`)**;gather 变纯 on‑device 索引;OOM 回退 pinned host 路径;**`bank.gpu_bytes` 计入 GPU 缓存预算**(启动 auto‑size + 每次 rebuild fit‑check),池永远不为 resident bank OOM | mtp.py:72-104;weight.py:339-388;engine.py:326-347,521,894 |
| D | 草案是纯自回归近似,无参考实现锚定 | 明确对齐训练/推理参考:**vLLM `Qwen4ExpMultiTokenPredictor` "residual_linear_shared"** + **llm_base_proposer EAGLE‑shift 约定**("s 位置行消费 s+1 的 token 与 s 的 hidden,位置不变") | mtp.py:14-27,285-289,450-462,464-478 |
| E | 无 MTP context‑KV 维护;草案上下文不干净 | 主模型 hook `write_rows`/`write_ctx_kv`:以 EAGLE‑shift 给每个位置在 MTP 池里写**真实 fused 行 **(E(x_{t+1}),H_t) 的 K/V;prefill 行批内移位写、每 chunk 末行延迟补;decode 为单请求 eager;verify/回滚重跑用 `_ctx_write_off` 关掉、由**步末 `write_rows` 覆写被接受的真行**(L..L+m+1) | mtp.py:319-448;model.py:127-136;scheduler.py:1343-1357 |
| F | 每步 ~10+ 次 `.item()`/`to(“cpu”)` 多次清空队列的 host 往返 | **单次 batched host 转移**(一次 cat + `.tolist()`);草案链、token_pool 写入全 GPU‑resident;phase 计时显式落段(见 I) | scheduler.py:1166-1173,1228-1240,1341 |
| G | 无:verify 页表超量保留不回收 | 步末**释放未提交的页表整页尾部**(`keep_end=div_ceil(L+2+m)`,释放到 `verify_end=div_ceil(L+K+2)`),否则下一步 decode‑prep 会重分配已属块 → 覆写行 → **页泄漏、CacheManager.check_integrity 报错** | scheduler.py:1319-1336 |
| H | 无:对反复解码导致的 context 过期 | 步 0 先做 **MTP context‑KV 完整性检查**(slot 0..L-1 是否都是真行;`_ctx_upto`/`_ctx_pending` 记账);若被 graph‑replay 解码污染(无 hook)则重建:GDN 槽清零 + `allocate=False` 整前缀重跑,重算 S_L | scheduler.py:1125-1146 |
| I | 无 | **per‑phase wall‑clock 插桩**(前 8 步/进程,`MTP PHASE` 打印 rebuild/anchor/drafts/verify/accept/rollback/rewrite 各段毫秒),定位 host‑sync / batch‑rebuild 是否压在 line 外 | scheduler.py:1107-1123,1359-1370 |
| J | 无 | 全 head 内部 per‑stage **数值诊断 `_diag`**(embed/fusion/attn/mlp/logits 的 min/max/mean/std/l2,`NONFINITE` 标记)+首个 MTP 步打印 head vs model 的 top5 / overlap — 直指"0% acceptance"根因定位 | mtp.py:480-529;scheduler.py:1261-1303 |
| K | 无 | acceptance sanity 计数:`mtp_first_match`(d1 是否哪怕一次与主模型 greedy 一致)、`mtp_full_accept`(m=K 次数),区分"头系统性错"与"只是弱" | core.py:68-69;scheduler.py:1251-1258 |
| L | 无 GPU 预算分摊 | GPU‑resident 银行 `gpu_bytes` 参与**固定缓存项 + live rebuild fit‑check**(引擎预算自动绕开 resident bank) | engine.py:339-347,894 |
| M | verify 全权重层 host 流式 | verify 令 `vbatch.moe_force_decode=True`(取 decode 的按需顶层 MoE 挪动,避免逐层全路由 PCIe 回流) | scheduler.py:1203-1207 |

**对 v1 §2.3 的直接影响**:这是"架构做了实质重写"而非只改了参数——尤其 A/B/C/E/F/G/J/M 是对我之前列风险的针对性加固。

---

## 2. 重评分析 1:v2 的明显漏洞?(核心正确性已大改,风险面收窄但未归零)

### 2.1 v1 结论经 v2 复核后的迁移

| v1 原条目 | v1 定性 | v2 之后 | 依据(v2) |
|---|---|---|---|
| ① 触发过窄 batch.size==1 | 高(缺陷/取舍) | **仍是"有意单流 greedy"**(未解除):`_mtp_active` 仍 `batch.size==1 && is_decode && is_greedy`。但 v2 把它做成可观测(`_note_mtp_skip` 记录并指原因,见 §3.1)而非静默。**若并发/采样是你的用法,仍不会跑**——但这与开发者明说的一致,属取舍非缺陷。 | scheduler.py:977-985,1017-1042 |
| ② GDN(混合 radix)回滚等价性 | P0,须逐 token 回归 | **维持 P0(建议强度最高)**。v2 补了簿记细节(页表尾部脱离 G、`linear_slot_idx` vs `table_idx` 分离、rollback 后 `device_len`/`cached_len` 收尾、`moe_force_decode` 保持 verify/rerun 在同一 MOE 路径),但**"快照 S_L → 恢复 → 重跑到截断"这段是 per‑request 的破坏性状态结算,仍无等价证明**。开发者此前自己也把"逐 token 一致性回归"设为(若有)多路前置门槛 → 单路在放行前同理。 | scheduler.py:1098-1102,1305-1313,1319-1357 |
| ③ 共享 `_last_residual` 时序 | 中,脆弱 | **维持开放(未因 update 根治)**。v2 加 `_mtp_head is not None` 门(off 时零开销)与更细簿记,但 `self._last_residual = hidden` 仍是模型级可变属性、主 forward↔MTP 读仍是顺序耦合,依赖"单请求 decode + 全程 eager"不被打断。 | model.py:127-136;scheduler.py:1148-1152;engine.py:964(注释仍写明靠 eager/单请求规避) |
| ④ 草案条件化近似 | v1:只看自己 / collapse‑repeat | **已解决、并对齐参考实现**:草案现在能看到前序草案(scratch/mtp_extra_locs),residual 逐字续到下一格的 multi stream(无 collapse/repeat),融合/top‑mixer 的 shape 对训练口径。仍需对账的一环是 write_rows/forward 两套 fusion 是否位一致——见 x2。 | mtp.py:14-27,450-532 |
| ⑤ anchor/bonus 不计草案 / 跨步互斥 | 正确(设计确认) | **维持**。v2 计数公式、位置偏移(j 草案坐 L+1+j / fed token_pool[L+j])仍对——逐 token 复核无 off-by-one。 | scheduler.py:1246-1250 等 |
| (新增)页表尾部泄漏 | ❌ 未评估 → v2 自修 | v1 漏掉的一条隐性风险:v2 自己修掉了(**G**)。不新增遗留。 | scheduler.py:1319-1336 |

### 2.2 结论定性(v2)

**比 v1 成熟一个量级**:不再是"逐草案自回归 + collapse/repeat + 路由不对齐"的投机头,而是一个带 context‑KV、EAGLE 对齐、GPU‑resident 银行、预算分摊、诊断插桩和页表簿记的工程实现。**没有看到"必崩"级的明显漏洞。** 我仍未在静态层面证伪"GDN 回滚后逐 token 与纯 decode 完全一致",这里仍是 P0。

**但 v2 又引入了几处新的簿记/近似点,需随真机跑出来才能断言对**——见 §2.3 新观察。

### 2.3 v2 新增、且静态看不透的观察点(按评估价值排序)

- **x1 (P0,与 ② 同源) `write_rows` 与 `write_ctx_kv`(delayed last row / defer)簿记必须与 verify/rollback/rewrite 同步一致。** 这是 v2 新引入、最复杂的状态:prefill 批内移位、末行 deferred(`_ctx_pending`)、decode 消费 pending;verify 期关掉全局 `_ctx_write_off`,回滚重跑也在关闭态,最后 `write_rows` 在步末按 committed 结果覆写 L..L+m+1——本质是一组 **per‑request 的 context‑KV 一致性簿记**。任何一处(defer 时点 / 位置下标 / verify 被拒行是否晚于 rewrite 覆盖)差一步,下一位草案的上下文就读错——**吃错上下文不会报错,只会让 acceptance 悄悄掉**。必须:长文本、多 MTP 步、K 取 1 和 2 各跑,与 MTP‑off greedy 做逐 token 等价回归 + 在端上看 acceptance 基线,而非只信 DIAG 打印。
- 证实作者认真考虑过它的证据:步 0 的 `covered` 判定、`_ctx_dirty` 触发重建、`_ctx_upto==L-2` 这类簿记判断都在。但**覆盖正确性边界我仍不能静态板上钉钉**,故归到要真机背书。

- **x2 (P1‑中) 隐含"golden/oracle"假设:`write_rows` 必须和 `forward` 对同一 fused 行算出位一致的 K/V。** 两种代码都实现 fusion 前端(`write_rows` 用 `_fuse(attach embedding)` + unit‑weight combine;`forward` 里也有一套 `pre_fc_norm→fc_hidden→combine`)。二者只要在任一 norm/combine/权重细节上分叉,attention context 就位错。注释声称 bit‑consistent,但这是"两段代码必须永远保持同步"的隐式契约。**应跑一次 KV‑consistency 探针**:把某一被 draft 的位置用 `write_rows` 写出的 K/V,与对它 `forward` 一次应得的同位置 K/V 对比是否 bit‑equal——等价性与 ②/x1 一起在回归里覆盖。

- **x3 (P1‑低,新增)GPU‑resident 银行的分摊注入点只在「能起 engine 的路径」测得,但自动缓存重建的 fit 检查是**「如果银行已驻留,那么任何一次 pool 缩小若不含其 gpu_bytes 就会误把可释放的部分留作 KV」——需要一次「cache rebuild 时本就该缩池」触发路径来验,而不是只靠启动 auto‑size 覆盖。

- **x4 (P2,响应一致性,且为 v2 相对 v1 不变的表现点)** acceptance‑rate 的“分母=每步 K、含 anchor 0 断言、`drafts_total`/`accepted_total` 的累计口径同 v1”——这里**没有改 schema、没有补 `predicted` 口径**。下一位见 §3。

### 2.4 具体可复现验证(不修代码,只改为“评估后再跑”的清单)

**A. 主等价性(P0)**:同一批(长文本、>K、含终结符)下,MTP‑off greedy 生成的**逐 token id** 与 `--mtp`(K=1、K=2 各一次)完全相等。任一 diff → ②/x1 处有问题。
**B. KV‑consistency 探针(P1)**:对一个固定已接受前缀,单独调用 `write_rows` 产生的某格 K/V,与把它当 draft、跑 head `forward` 得到同位置 K/V,比较是否 bit‑equal(验证“两套 fusion 真的一致”,x2)。
**C. rebuild‑fit 探针(P1‑低,x3)**:手动触发一次 live cache rebuild、GPU‑resident 银行已占用时,确认 KV/moe 池没有因预算里漏扣 gpu_bytes 而把可回收内存当成缓存增长。
**D. 时序回归(P0 配套,③)**:在线路里让请求在“主 decode → MTP 读 residual”之间无其它 forward；若未来出现任何多路/采样/预填抢占,MTP 必须 fail‑closed(短路走普通 decode),而非静默读错 residual。

---

## 3. 重评分析 2:响应参数是否符合通用规范?(v2 基本无改,因此结论全保留 + 一个小补丁点)

### 3.1 无 mtp 参数(实测)在 v2 里的根因——依然不是“schema 不合规”

三层照旧成立:
1. `_mtp_report` 的**三个数全为 0 → 仍返回 None → 不注入**。计数仍只在 `_mtp_speculate` 里递增。(v2 openai_api.py:653-660、scheduler 计数位)
2. `_mtp_speculate` 仍只在 `_mtp_active`(`--mtp && is_decode && batch.size==1 && is_greedy && not aborted && can_decode`)时跑。(v2 scheduler.py:977-985)
3. 且 v2 新增了 `_note_mtp_skip`——**当 `enable_mtp` 但某 decode 不满足 MTP 条件(非 greedy、被 graph‑replay 无 hook、scratch 未就绪)时**,有**显式 skip 记账/日志**(scheduler.py:1017-1042)而非静默。→ 若你再测不到,现在能在日志里直接看到**为什么被 skip**(这是 v2 相比 v1 的可观测性进步)。

所以你要先做的排除顺序(v2):
- 有没有开 `--mtp`?(默认 False)
- 是不是 ≥2 并发 / 非 greedy → 现在看 **skip 日志** 落到哪条原因。
- 是不是该路径(cpp/offline)根本没接 `_mtp_report`。

### 3.2 命名/位置 vs llama‑server / vLLM(不变,仍不对齐)

`resp["mtp"] = {drafts, accepted, steps, acceptance_rate}`(openai_api.py:661-666)仍自造命名、放顶层非 `usage`、且 **`steps`/`mtp_phase_ms`/`first_match`/`full_accept` 这些 v2 新增的诊断只在 Req/stats/日志,没进响应**。→ 与 llama‑server `tokens_drafted / tokens_accepted / tokens_predicted` 那组**仍无法直接对账**。要把生成率/接受率给到下游并按通用口径量化,仍需 §4E。

### 3.3 口径仍是 moot(小建议可留着)
补丁给你的是 `accepted/drafts`,不含 `predicted`(每 MTP forward 实际产出 m+2)。这与你“想评估生成率/吞吐增益”的目标差一步。

---

## 4. 分级动作(重评后,按 v2 校准;不代改)

- **P0(两块,都是"真机跑出来才背书",不因 update 免除)**
  - ②/x1:逐 token MTP‑off vs MTP 等价回归(A);含 `--mtp-draft-len 1/2`、多步、长文本、终结断言。
  - ③:残差时序 fail‑closed 判定(D)。
- **P1**:x2 KV‑consistency 探针(B);x3 rebuild‑fit 探针(C);先跑真机 acceptance 基线按 §3.2 通用口径读数。
- **P2**:④ 残留对账(reference overlap 见 x2);§4E 响应归一。
- **维护层**:python/ 工作区(09‑03)落后快照(09‑07)一整轮;若要跑 v2,把 `freetoken-…-patched-src/freetoken/`(MTP 相关)同步回来并做 A 回归,再用 skip 日志快速判 why‑no‑mtp。

---

## 5. 附录

### 5.1 v2 关键行号(mtp 相关)
- `models/qwen4_exp/mtp.py`: GPU‑resident bank 72-104; fused_topk 路由 134-149; scratch/extra 注意力 234-262; context‑KV write_rows 335-366 与 write_ctx_kv 368-448; _fuse/_fuse_with 450-462; 参考 forward 464-532。
- `models/qwen4_exp/model.py`: _last_residual + _mtp_head context hook 127-136; 银行构造 162-170; GPU 预算 173-191,224-233。
- `scheduler/scheduler.py`: _mtp_active 977-985; _mtp_scratch_ready 990-1008; _note_mtp_skip 1017-1042; _mtp_draft_batch 引入 extra_locs 1045-1065; _mtp_speculate 1085-1376(含 context 完整性 1125-1146、draft chain 1161-1193、verify 1195-1223、acceptance 1225-1259、diag 1261-1303、rollback 1305-1317、页释放 1319-1336、rewrite 1343-1358、phase 1359-1370)。
- `engine/engine.py`: GPU‑resident 预算 326-347,521,894; _mtp_pool/scratch 366-390; forward_batch eager 958; mtp_forward 998。
- `core.py`: Req 统计 61-74; SamplingParams greedy 18-31。
- `server/openai_api.py`: 注入 229-231,365-370,454-456,508-510; _mtp_report 653-666。
- `server/api_server.py`: MTP finished 日志 263-270。

### 5.2 局限声明
- 全程**静态评估**,未在真实 CUDA/greedy 环境复跑;②、x1/x2/x3 属"要回归验证",非断言失败。
- 以快照(v2,09‑07)为判定对象;python/ 工作区为 v1(09‑03),两者已分立。
- 参考细节(vLLM/SGLang/llama.cpp 的命名与 schema)以各官方为准,此处用于"是否对齐通用观测",不构成精确语义承诺。

---

# 附篇:P4 阶段交付重新评估(2026‑09‑10,与 MTP 无关的独立改动)

> 对象:`FreeToken_P4_stage_v0.1.2-g816c324d0_2026-09-10.zip`(及解目录 `FreeToken_P4_stage_2026-09-10/`)。
> 定位:`VERSION_INFO.txt` 明确"P4 changes are pure Python",主题是**多模态图像请求的
> 内容哈希前缀缓存键 + 拼贴(splice)不变量修复**(Round‑4 生产崩溃回归)。**这不是 MTP 工作**;
> 上文 MTP 报告与本节相互独立。

## 7. 结构核对(已实测)

> **修订(2026‑09‑15 复核)**:本目录已由 **09‑14 的新 zip** 重新铺开(`FreeToken_P4_stage_v0.1.2-g816c324d0_2026-09-14.zip`),
> 内容较前两次评估**又推进到 Round‑6(mm RAM tier:KV 换入换出)**。当前实测:
> **404 个文件**(`MANIFEST.sha256` 列 401 条,3 个元文件未列入);notes 2718B→3950B→**5736B**(增补第5、6轮);
> 新增 `verification/kv_tier_test.py` 及第5轮的 `vision_video_test.py`/`serve_video_smoke.py`;
> 源码新增 `tokenizer/media.py`、`models/qwen4_exp/vision.py`、`kvcache/turboquant.py`,并多出整份 `dist-info/`。
> **完整性仍通过**:`sha256sum -c MANIFEST.sha256` → **401/401 OK,0 失败**。
> 结构/图像(Round‑4)见 §8–§9;视频(Round‑5)见 §10;**KV 换入换出(Round‑6)见 §12**。

## 8. P4 改了什么(经核对,与 notes 一致)

notes 声称只改 3 个文件;逐处核对成立:

1. **`scheduler/cache.py :: match_req`** — 引入 `mm_span` 与 `_mm_bound(cached_len)`:
   当 mm 请求带 `mm_cache_key` 时,匹配键 = 内容键(pad 段替换为负载 sha1);若匹配落在 `[span0, span1)` 内,
   把匹配 **回退到 `align_down(span0, page_size)`** 并(混合架构下)**丢弃捐出的 GDN 快照 `mamba_value`**;
   落在 span 之前(np 全在行内)或越过末尾(np 全已缓存)则原样放行。SWA / hybrid / 普通三条分支都接了 `_mm_bound`。
2. **`models/qwen4_exp/model.py :: forward`** — splice 计数不变量放宽为:
   `n_pads == mm_rows`(全入行→scatter)或 `n_pads == 0`(**已缓存整段→no‑op**);中间态 `raise`。
3. **`models/gemma4/model.py :: _merge_multimodal`** — 同样的放宽(`n_slots != 0` 才 raise,否则 return)。

**配套前提(已核对,不是漏洞)**:mm 请求被排除在 chunked prefill 之外(`prefill.py:178`,整段单 chunk 准入),
使"一次 forward 的行内 pad 要么全在、要么全被缓存"成立。

## 9. 评估结论

### 9.1 设计正确性(静态,证据充分)

- **不变量闭合**:match 侧 `_mm_bound` + insert 侧内容键 + splice 侧计数 + prefill 侧 mm 禁 chunk,四处对同一
  不变量(行内 pad 计数 ∈ {0, mm_rows})互相支撑,没有发现相互矛盾。
- **多图场景安全**:`mm_span=(首段首 pad, 末段末 pad+1)` 横跨全部 pad 段——中间若有多段 pad,任何落在 span 内的匹配
  都统一回退到**第一段之前**(清空所有段),不存在"清了一段留一段"的中间态 → split-pad 崩溃不能复现。
- **GDN 正确方向**:回退分支丢掉 `mamba_value`(不 COW 恢复),宁可重算前缀也不把过进状态恢复——是保守/安全的一侧。
- **键不可伪造**:`pads` 以 `input_ids[diff0]`(首个 pad id)为准收全,而 image_pad 是保留 id、文本无法产出 → 不会误判 span。

### 9.2 剩余关注点(需真机/端到端确认,非断言 bug)

- **c1(需回归)** 生产崩溃的触发源是 **GDN 的 ×64 chunk 提交把快照捐在中途 pad 上**(notes 与 linear.py 所述)。
  P4 让"匹配到中途 pad 节点"被回退,但**该中途节点仍被 insert 进树**(只是匹配时被 `_mm_bound` 拦住)。
  这是否在所有 page_size(1/4/8)与多轮同图下都只有"回退"而无副作用(例如反复回退导致的过度重算/RE‑PREFILL 放大),
  需端到端多轮同图跑测确认。
- **c2(需回归)** `_mm_bound` 的 cap 用**未 page 对齐的 `mm_span[0]` 再 align_down**;若 `mm_span[0]` 恰在页首,
  `align_down` 不移动,理论上仍落到 span 之前的整页边界——需按 `ps∈{1,4,8}` 验证边界(notes 的 r4_adv_probe 声称已覆盖
  `ps in {1,4,8} x 3 形状 x 4 回合`,但**我未能执行该套件**,见 9.3)。
- **c3(非缺陷)** keyless mm(离线预计算 embeds)不参与跨请求复用(匹配空前缀),保留旧排除;文本路径逐位不变。

### 9.3 关于验证链:我**未能实跑**,如实说明

- **为什么没跑成**:测试脚本把 `sys.path` 硬编码为 `G:\FreeToken\ft-venv\Lib\site-packages`(在你的机器上不存在);
  即便改指向包内 `package/`,链条缺 **`flashlib.kernels.slot_cache`**——它来自**独立**的
  `freetoken_kernel_cache` wheel(VERSION_INFO 注明 "unchanged, separate"),P4 zip 里没有;包内两个
  `.pyd` 是 **cp312**,而本机默认解释器是 **3.13**,也加载不了。→ 该套件是为**那台带完整内核缓存的 Windows 运行机**准备的。
- **我做了什么替代**:已实测 `sha256sum -c MANIFEST.sha256` 全绿;已**静态通读** `p4_splice_test.py` 等——
  它是货真价实的回归测试(驱动**真实** `CacheManager(hybrid_radix, page_size=8)` + **真实** `LinearStatePool`
  走完整引擎生命周期,复现的正是线上 CRC `10 pad vs 1000 soft rows`,并在每次 match 后断言不变量),**不是自证式玩具**。
  但"脚本设计看起来对"≠"我已验证它通过";**9.1/9.2 的结论以静态代码为准,验证链的执行仍待你在真机确认**。

### 9.4 建议动作

- **P0**:在有 `freetoken_kernel_cache`/`flashlib` 且 cp312 的机器上,按 notes 的复现命令跑全 6 套 + `r4_adv_probe`;
  特别复跑**同图多轮 turn‑2** 的原始崩溃场景(notes 的 p4_splice_test 第 1 腿)与 `ps∈{1,4,8}` 边界。
- **P1**:本地无法执行时,至少比对 `package/` 与你在运行机上现装 `site-packages/freetoken/` 的逐文件哈希
  (MANIFEST 已在,直接 `sha256sum -c` 即可判定"装的到底是不是这份 P4")。
- **P2**:若 mm 内容键要长期演进,建议把 9.2 c1/c2 写成常驻回归腿(多轮同图 × page_size 矩阵),而非一次性探针。

## 10. 第5轮增补评估(2026‑09‑13:视频路径 + 双 pad)

> 这是"大版本更新"的实质增量。核心新文件:`tokenizer/media.py`(视频/多模态 CPU 管线)、
> `models/qwen4_exp/vision.py`(视觉塔)、`kvcache/turboquant.py`;并新增两条验证脚本。

### 10.1 改了什么(经代码核对)

- **视频部件端到端**(`media.py`):`fetch_video_frames` 支持动画图(GIF/WebP/APNG,`n_frames`+`seek`)与
  **mp4 回退**(`_frames_via_imageio`,imageio/ffmpeg;非路径引用先落临时文件);`_sample_indices` ≤8 帧等间隔采样,
  **首尾必留**;多帧先试处理器 `videos=` 时间配对,失败回退 `_stack_frame_grids`(逐帧 images= 堆叠,要求各帧同格)。
- **双 pad**(notes 第2条):checkpoint 对 video 用 `video_token_id`;`MediaPipeline` 双拼写都识别为一个 vision span
  (`rest.eq(image)|rest.eq(video)`),展开**统一写 `image_token_id`** → 模型侧 embed mask / cache key / scheduler
  **零改动**(沿用单一 id)。`vision_geometry` 透传 `video_token_id` 给 `tokenize.py`。
- **模板分支镜像**:`extract_media_parts` 按模板顺序挑 image/image_url 与 video 键,`image_url` 的 `{"url":...}` 自动解包。
- **patch 预算**:`32768` 上限现在**按整段视频的所有帧合计**计(`t*h*w`),越界为客户端错误。

### 10.2 评估结论

**设计是自洽的**,几点值得肯定:双 pad 统一展开为单一 id 的做法,把"多一种 pad 拼写"这个复杂度**限制在 CPU 预处理层**,
模型/调度/缓存侧一行不动——这是正确的切面。视频帧采样、fallback 链、patch 预算的口径也都与既有不变量一致。

**但有几处要你特别确认(静态发现,非断言 bug):**

- **v1(应确认)单一 content id 覆盖整请求**:`_content_key` 把**所有** pad 位置统一替换为 `_content_id(payload_path)`,
  而 payload 是**整请求所有部件像素拼接的 sha1**。含义:多部件(A+B)请求的键 = 文本 + 一个"由 A+B 联合摘要得出的 id",
  **不是每个部件各自的 id**。后果是(a)正确性上安全——A+B 任一不同则 id 不同,不会误命中;
  (b)**复用粒度变粗**:同一张图出现在"图 A"与"图 A + 另一张图"的两种请求里,键不同、无法互相复用 KV;
  (c)**顺序相关**:A,B 与 B,A 得到不同 id → 不命中(安全向,但少复用)。这与 `media.py` 顶部 docstring
  "the i-th occurrence is replaced by the i-th image's token block" 的措辞**不完全一致**,建议核对是否即预期。
- **v2(应确认)视频的 32 MiB 原始体积上限**:`fetch_video_frames` 走 `_read_raw`,受 `_MAX_IMAGE_BYTES=32MiB`
  约束。notes 未提这一条对视频的适用;一个较大的 mp4/GIF 可能在解码**之前**就被拒。若视频放宽了 patch 预算(32768 覆盖整段),
  体积上限是否也该相应说明/调整,值得确认。
- **v3(需回归,承接上节 c1/c2)**:round‑5 让**多部件(image+video)、多 pad 段**成为常态,而 `match_req` 的
  `mm_span`/`_mm_bound` 多段推导逻辑正是 round‑4 引入、我在 §9.2 标为"待端到端确认"的部分。**新功能把旧风险的使用面放大**——
  上节 P0 的"多轮同图 × ps∈{1,4,8} 回归"现在还应覆盖**图+视频混合**的键/不变量。
- **v4(实现细节)**:`_frames_via_imageio` 用 `delete=False` 落临时文件并在 `finally` 清理,名字由
  `NamedTemporaryFile` 保证唯一,`src!=ref` 判断避免误删真实路径——**这一处实现是正确的**。

### 10.3 验证链(第5轮新增,均未在本机执行)

notes 称 `vision_video_test.py` / `vision_p3_media_test.py` 全绿(HF oracle 逐位一致),并做了 live‑serve(:8001)
四形态冒烟。与 §9.3 同因(缺 `flashlib` + cp312),**我无法在此机复跑**,结论以静态为准。

## 11. Round‑6 评估(2026‑09‑14:KV 换入换出 — mm RAM tier)

> 本轮目标:**重复的图像/视频请求不再重算媒体段**。已完成的 mm 请求把 prompt 段 KV + GDN 状态
> 驻留到主机内存(RAM tier);后续**同内容键**请求在匹配时把行搬回新分配的显存页,直接续跑。
> 改动 9 个文件:`kvcache/{mha_pool,qsa_pool,linear_state_pool,base}.py`、`scheduler/{cache,prefill,scheduler,utils}.py`、`core.py`。

### 12.1 机制(经代码核对)

- **store**(`cache.py::_tier_store`):请求完成时,取边界 `b` 处的行列快照(`pool.snapshot_rows`)并按需附 GDN 状态
  (`linear_state_pool.snapshot_state`);**KV 与 state 必须严格配对到同一 `b`**——优先取 `mamba_last_track_seqlen` 的冻结
  ping‑pong 槽,退而取"live 槽恰好结束在 prompt 末"的情形,**否则直接跳过**(注释明说"错配会让 GDN 递推过度推进")。
  按内容键(整数 digest)入 `OrderedDict`,字节计费,`FREETOKEN_MM_TIER_MB` 默认 1024,超限 **LRU 淘汰**。
- **restore**(`cache.py::restore_mm`):**digest 命中后还要 `torch.equal(ekey,key)` 全键比对**(digest 只作分桶,不作判定);
  门槛 `b > 树匹配`、`b <= input_len`、且过 `_mm_restore_ok`(**pad‑run 规则**:边界只可落在首段之前或末段之后,
  与 round‑4 `match_req` 同构);通过则**新分配页** `div_ceil(b,page_size)` 并 `restore_rows` 回填,返回 `_TierHandle`;
  GDN 经 `MatchResult.gdn_host` 由 `Scheduler._restore_linear_states` 在**首个 prefill forward 之前**写入 live 槽。
- **池往返**:`snapshot_rows` 取 packed K/V(含 turbo4 dscale)与 QSA cmp 索引行的**字节级**拷贝(免重量化);
  `restore_rows` 用 **`index_copy_`**(注释点明 `[:, idx].copy_` 会 gather 成临时量、写不进去——这是个真实的坑,已避开)。

### 12.2 评估结论

**设计是稳的**,几处关键点做得对:
- digest 分桶 + 全键 `torch.equal` 判定 → **弱哈希不会造成错误复用**(只造成 miss);
- KV/state **严格配对否则跳过** → 宁可不用也不过度推进 GDN;
- restore **新分配页 + 回填** → 与树页无别名,所有权清晰;`_mm_restore_ok` **复用 round‑4 的 pad‑run 不变量**,口径一致;
- `index_copy_` 的用法正确(避开 advanced‑indexing 陷阱);CPU 测试 `kv_tier_test.py` 也确实覆盖了
  池往返 / 字节核算 / 同键替换 / **边界落在 pad 段内被拒** / GDN 写回 / plain‑radix 句柄 / LRU,质量不错。

**发现一个需要处理的缺口(本次最高价值):**

- **t1(应修) `CacheManager.rebuild()` 不失效 `_mm_tier`。** 核对了 `rebuild`(cache.py:829)——它重指 `page_table`、
  重建 `free_slots`、**换一棵全新 prefix cache**、`reclaim_all_slots()` 回收 GDN 槽,**但完全没有清 `_mm_tier`**
  (`_mm_tier` 只在 `__init__`/`_tier_store`/`restore_mm` 出现)。后果:若一次**运行期 cache rebuild**
  (idle‑only,用户改 `num_pages` 等)改变了 KV pool 几何(`_kv_buffer` 的 `tokens` 维随页几何而变),
  其后一个**同内容键** mm 请求会走到 `restore_rows` —— 而 host `rows` 是**旧几何下打包**的,
  `index_copy_(1, idx, data["kv"])` 将遇到**形状不匹配**(大概率是响亮报错,而非静默错值;但至少是**可达的崩溃/异常路径**,
  且陈旧条目还在占 tier 字节)。
  - 建议:在 `rebuild()` 里 `self._mm_tier.clear(); self._mm_tier_bytes = 0`(最省事且正确),或把 pool 几何指纹纳入 tier 条目并在 restore 时校验。
  - 说明:`kv_tier_test.py` **未覆盖 rebuild 交互**,与"这是个真缺口"的判断一致。

**其余观察(静态,列为待确认):**
- **t2**:`_tier_digest` 仅用 `key[0]` 与 `numel()`(弱),虽由全键比对兜底,但若未来某路径漏掉 `torch.equal` 就会退化为误命中——
  现状安全,建议保留该全键校验为硬约定。
- **t3**:`restore_mm` 在 SWA 模型上直接返回树匹配(不载入 window tier),属**有意取舍**(注释明说),非缺陷。
- **t4**:tier 只在**进程内**驻留(主机内存),重启即失;notes 也未声称跨重启持久化——确认这是预期边界即可。

### 12.3 验证链(第6轮,均未在本机执行)

notes 称 `kv_tier_test.py` 全绿(CPU、QSA+turbo4 真实池:行往返/字节核算/同键替换/LRU/pad 边界/GDN 写回),
vision 两套回归与 live‑serve 冒烟全绿。**同 §9.3 因由,我无法在此机复跑。**

## 12. 附篇局限声明

- 本节为**静态评估 + 完整性校验**;所有验证套件**未执行**(原因见 9.3),故"通过/失败"以你在真机复跑为准。
- 评估对象为 P4/第5/6轮交付包的 Python 改动;MTP 相关结论见上文,两者不互相背书。
- 内容键的 payload 哈希(sha1)与 cmm 语义未逐位复核(超出本次范围),仅评估其在前缀缓存键/匹配上的用法。
- 每轮结论基于 `FreeToken_P4_stage_2026-09-10/` 目录**当时**的内容;本次为 09‑14 zip 解开态。该目录会随新版被覆盖,**每次评估前应先核对 VERSION_INFO 与 MANIFEST**。

