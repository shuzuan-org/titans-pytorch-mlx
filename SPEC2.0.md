# SPEC2.0

## 目标

在当前 `titans_lin` 分支上建立一套可复现的 **LoCoMo 对齐评测链路**，用于和 `../mem0/evaluation` 的结果做直接对比。

本阶段只围绕以下四个已确认方向展开：

1. 用已部署 mem0 服务做 **记忆测试**
2. 用 `../../process_kg3/mem0` 中实际配置的 MiniMax/OpenAI-compatible 接口做 **judge**
3. 在 H800 上用当前 `titans_lin` 跑一个 LoCoMo 可用训练模型
4. 用训练后的 `titans_lin` 在 LoCoMo 上跑评测，并和 mem0 做对比

开发产物统一落在：

- `usecase/locomobech/`

---

## 一、当前结论

### 1. LoCoMo 数据

当前可用官方数据为：

- `data/dataset/locomo/locomo10.json`

结构为：

- `conversation`
- `qa`
- `event_summary`
- `observation`
- `session_summary`

其中训练/评测最关键的是：

- `conversation`
- `qa`

### 2. 当前分支训练数据要求

当前 `titans_lin` 的 stage1 数据要求：

- 每条样本必须有至少一个明确问答
- 最终可落成：
  - `history_chunks`
  - `question_chunk`
  - `answer`

因此 LoCoMo 的接入方式应为：

- **一条 QA = 一条训练/评测样本**
- 共享同一对话历史
- 不能只拿 conversation 做无监督写入训练
- 为了和 mem0 严格对等，后续 `titans_lin` 的 LoCoMo 训练与测试也应考虑 **双视角拆分**：
  - A 视角样本
  - B 视角样本
- 即同一条 conversation + 同一组 QA，可以生成两套样本/测试会话，而不是只保留一个共享全知视角

补充说明：

- `locomo10.json` 中的 `conversation` 本身不带问答块
- 问答监督来自同一条样本顶层的 `qa` 列表
- 因此在 `titans_lin` 里训练时，必须把：
  - `conversation` 作为 history
  - `qa[i].question` 作为 `question_chunk`
  - `qa[i].answer` 作为 `answer`
- 也就是 **同一对话展开成多条 QA 样本**，而不是把整段 conversation 直接当一条训练样本

### 3. mem0 对齐目标

本次对齐对象以 `../mem0/evaluation` 为准，而不是 `../titans/scripts/eval_locomo10.py`。

mem0 当前评测链路关键文件：

- `../mem0/evaluation/run_experiments.py`
- `../mem0/evaluation/evals.py`
- `../mem0/evaluation/generate_scores.py`
- `../mem0/evaluation/metrics/llm_judge.py`
- `../mem0/evaluation/metrics/utils.py`

其特点：

1. 输入是 LoCoMo 问答结果 JSON
2. 评分包含：
   - `BLEU`
   - `F1`
   - `LLM Judge`
3. 输出按 category 汇总，并给 overall
4. `LLM Judge` 是 **二分类 CORRECT/WRONG**，不是 0-10 分制

---

## 二、评分模型方案

### 目标

确认两条链路的职责：

1. `http://58.211.6.130:10281/docs` 是否足够做 **记忆测试**
2. `../../process_kg3/mem0` 中实际使用的 MiniMax 接口是否足够做 **judge**

### 当前已确认信息（记忆服务）

该服务暴露的 OpenAPI 路径至少包含：

- `/api/v1/health`
- `/api/v1/users/{user_id}/exists`
- `/api/v1/users/{user_id}/imports/markdown`
- `/api/v1/users/{user_id}/imports/messages`
- `/api/v1/users/{user_id}/search`

这说明它当前就是：

- **memory ingestion / search 服务**

适合做 LoCoMo 中的记忆写入与检索测试。

### 当前判断（记忆服务）

**可以直接把 `http://58.211.6.130:10281` 用于 LoCoMo 记忆测试。**

具体可覆盖：

1. 用 `/imports/messages` 写入 conversation
2. 用 `/search` 对 question 做 memory retrieval
3. 用 `/exists` 做 user/session 检查

但它 **不能直接做 judge**，因为没有公开的 `chat.completions` 风格评分接口。

同时要注意：

- **不能把整段 LoCoMo conversation 原样一次性导入 mem0**
- 必须参考 `../mem0/evaluation/src/memzero/add.py` 做专门导入流程

原因：

1. mem0 原始评测是按 **conversation × speaker** 拆成两个 user
2. 再按 `session_n` 顺序导入
3. 每个 session 带 `session_n_date_time` 时间信息
4. 再按小 batch 逐步写入 memory

因此本项目的 LoCoMo 测试必须单独实现一套导入流程，而不是复用普通 `imports/messages` 的“整段直接导入”方式。

### 当前已确认信息（Judge / MiniMax）

在 `../../process_kg3/mem0/deployment/` 中已确认：

- `LLM_REMOTE_API_KEY=shuzuan2025-minimax`
- `LLM_REMOTE_MODEL=glm-4.7`
- `LLM_REMOTE_BASE_URL=https://mini.origintask.cn`

且实际调用方式为：

- `POST https://mini.origintask.cn/v1/chat/completions`
- `Authorization: Bearer shuzuan2025-minimax`

这是一条 **OpenAI-compatible chat completion** 链路。

### 当前判断（Judge / MiniMax）

**足够做 judge。**

也就是说后续可以直接做一个可插拔的 judge client：

- 默认按 OpenAI-compatible 接口调用
- 指向：`https://mini.origintask.cn`
- 模型：`glm-4.7`

如果该模型不支持 `response_format={"type":"json_object"}`，则退回普通文本解析 `CORRECT/WRONG`。

### 本阶段结论

1. `10281` 服务：负责记忆写入 / 检索测试
2. MiniMax OpenAI-compatible 接口：负责 judge

---

## 三、训练方案

### 训练目标

先验证：

- 当前 Titans memory 方案是否能在 LoCoMo-10 上形成可比较结果

### 数据构造方案

新增脚本：

- `scripts/build_locomo_stage1_data.py`

输出：

- `train.jsonl`
- `eval.jsonl`

规则：

1. 使用 `locomo10.json`
2. `conv 0-7` → train
3. `conv 8-9` → eval
4. 每个 QA 单独展开为一条样本
5. 每条样本都满足：
   - `history_chunks`
   - `question_chunk`
   - `answer`
6. 跳过 `category=5` 的 adversarial 样本（默认）

### 当前状态

该脚本已完成初版，可生成：

- `data/generated/locomo10_stage1_test/train.jsonl`
- `data/generated/locomo10_stage1_test/eval.jsonl`

本地验证结果：

- train: `1226` 条
- eval: `314` 条

每条样本都满足当前分支的训练要求：

- `history_chunks`
- `question_chunk`
- `answer`

---

## 四、评测方案（对齐 mem0）

### 目标

要实现的是：

- **测试集与 mem0 一致**
- **评分口径与 mem0/evaluation 一致**
- **结果汇总格式尽量一致**

### 对齐要求

#### 1. 测试集

使用：

- `data/dataset/locomo/locomo10.json`

#### 2. 测试流程

测试将分成两条并行链路：

##### A. mem0 测试链路

对每个 conversation：

1. 通过 `10281` 的 `/imports/messages` 导入 conversation
2. 对每个 QA 的 `question` 调 `/search`
3. 保存 mem0 retrieval / response 结果

##### B. titans_lin 测试链路

对每个 conversation：

1. 不直接使用一个共享全知视角
2. 参考 mem0 的双用户方案，构造：
   - `speaker_a_{conv_idx}` 视角
   - `speaker_b_{conv_idx}` 视角
3. 将同一段 conversation 分别写入两个视角会话
4. 对其中每个 QA 单独发起 query
5. 获取模型回答
6. 保存结果：
   - `question`
   - `answer`
   - `response`
   - `category`
   - `view`（a_view / b_view）

说明：
- 第一版可以先让同一 QA 在两个视角都测试
- 后续如有必要，再增加“问题归属视角”筛选

#### 3. 评分指标

与 `../mem0/evaluation/evals.py` 一致：

- `BLEU`
- `F1`
- `LLM Judge`

其中 `LLM Judge` 要对齐 `../mem0/evaluation/metrics/llm_judge.py`：

- judge 输出为：
  - `CORRECT`
  - `WRONG`
- 最终记为：
  - `1`
  - `0`

#### 4. 汇总格式

按 category 汇总：

- single-hop
- temporal
- multi-hop
- open-ended

并输出：

- `bleu_score`
- `f1_score`
- `llm_score`
- `count`
- `overall`

### 注意

当前仓库里的：

- `usecase/eval_locomo_stage1.py`

只是验证版，不算最终版。

最终版要放到：

- `usecase/locomobech/`

并拆成标准三段：

1. `predict`
2. `evaluate`
3. `aggregate`

---

## 五、H800 上的执行闭环

### 目标流程

在 H800 上完成以下闭环：

1. 准备 LoCoMo 数据
2. 训练当前 `titans_lin` 模型
3. 部署 titans 服务
4. 用 mem0 跑 LoCoMo memory test
5. 用训练后的 titans 跑 LoCoMo test
6. 用同一套 judge 和同一套汇总方法评分
7. 输出可与 mem0 对比的表格结果

### 期望目录

- 数据：
  - `data/dataset/locomo/locomo10.json`
  - `data/generated/locomo10_stage1/`
- checkpoint：
  - `checkpoints/locomo10_stage1_*`
- 结果：
  - `results/locomo_stage1_predictions.json`
  - `results/locomo_stage1_eval.json`
  - `results/locomo_stage1_scores.json`

---

## 六、最终开发计划

统一在：

- `usecase/locomobech/`

下创建对应脚本/说明。

### Step 1：导入数据到 mem0 中

目标：验证 mem0 记忆链路。

实施：

1. 新增 `usecase/locomobech/import_locomo_to_mem0.py`
2. 输入：`data/dataset/locomo/locomo10.json`
3. **不直接整段导入**，而是参考 `../mem0/evaluation/src/memzero/add.py`：
   - 每个 conversation 拆成两个用户：`speaker_a_{conv_idx}`、`speaker_b_{conv_idx}`
   - 按 `session_n` 顺序导入
   - 每个 session 带 `session_n_date_time`
   - 按小 batch 多次调用 `/imports/messages`
4. 使用固定 `user_id` / `conversation_id` 规则，便于重复测试
5. 导入后可用 `/exists` 与 `/search` 验证记忆是否生效

输出：

- 导入结果 JSON
- 失败样本日志

### Step 2：测试 mem0 评分

目标：得到 mem0 的 LoCoMo 测试结果。

实施：

1. 新增 `usecase/locomobech/run_mem0_locomo_test.py`
2. 对每个 QA：
   - 调用 mem0 `/search`
   - 将检索结果组织成回答输入/候选结果
   - 保存 `question/answer/response/category`
3. 新增 `usecase/locomobech/eval_locomo_scores.py`
4. 评分时对齐 `../mem0/evaluation`：
   - BLEU
   - F1
   - LLM Judge
5. Judge 使用 `../../process_kg3/mem0` 中已确认的 MiniMax OpenAI-compatible 配置：
   - `base_url=https://mini.origintask.cn`
   - `model=glm-4.7`
   - `api_key=shuzuan2025-minimax`

输出：

- `results/locomobech/mem0_predictions.json`
- `results/locomobech/mem0_eval.json`
- `results/locomobech/mem0_scores.json`

### 训练补充原则：titans 采用双视角

为了和 mem0 的 LoCoMo 方案严格对等，`titans_lin` 侧也采用双视角：

1. 同一条 conversation 拆成两个视角训练样本
2. 同一条 QA 可以生成：
   - A 视角样本
   - B 视角样本
3. 训练时不使用“单一共享全知 memory”作为唯一方案
4. 测试时输出：
   - a_view 指标
   - b_view 指标
   - overall 指标

### Step 3：准备 LoCoMo 数据并训练当前 titans_lin 模型

目标：在当前分支上训练一个 LoCoMo 可跑模型。

实施：

1. 使用 `scripts/build_locomo_stage1_data.py`
2. 输出目录固定为：
   - `data/generated/locomo10_stage1/`
3. 训练时不直接使用“原始长对话”作为单样本，而是：
   - 一条 `conversation`
   - 展开成多条 `qa`
   - 每个 `qa` 形成一条监督样本
4. 长对话通过两层限制处理：
   - 数据转换阶段：限制 `max-turns`
   - 训练阶段：`max_history_length / max_question_length / max_sequence_length`
5. 在 H800 上跑训练：
   - 使用 `scripts/train_stage1_distributed.py`
   - `--data-dir data/generated/locomo10_stage1`
   - checkpoint 输出到：
     - `checkpoints/locomo10_stage1_v2/`（初版默认）
6. 先只跑最小闭环训练，不追求最终最好分

输出：

- `checkpoints/locomo10_stage1_v2/final_model/stage1_state.pt`

### Step 4：用训练后的 titans_lin 模型跑 LoCoMo，并输出评分

目标：和 mem0 用同一测试集、同一评分方法进行对比。

实施：

1. 新增 `usecase/locomobech/run_titans_locomo_test.py`
2. 对每个 conversation：
   - 调用当前 stage1 服务写 memory
   - 对每个 QA 发起 query
   - 保存 `question/answer/response/category`
3. 复用 Step 2 的评分脚本
4. 复用同一 judge 模型和同一 category 汇总逻辑

输出：

- `results/locomobech/titans_predictions.json`
- `results/locomobech/titans_eval.json`
- `results/locomobech/titans_scores.json`

### Step 5：生成对比结果

目标：直接横向对比 mem0 与 titans。

实施：

1. 新增 `usecase/locomobech/compare_results.py`
2. 对比：
   - overall BLEU
   - overall F1
   - overall LLM score
   - per-category 指标
3. 输出 Markdown 或 JSON 报告

输出：

- `results/locomobech/compare_report.md`
- `results/locomobech/compare_report.json`

---

## 七、当前最小可实施版本

当前四点均已确认无原则性阻塞，可以开始开发。

最小闭环顺序就是：

1. `locomo10.json` 导入 mem0
2. mem0 跑 LoCoMo 测试并评分
3. `locomo10.json` 转 stage1 训练数据并训练 titans
4. titans 跑 LoCoMo 测试并评分
5. 输出 mem0 vs titans 对比结果

---

## 八、当前风险

1. `10281` 只能做 memory test，不能直接做 judge
2. `glm-4.7` 是否支持 `json_object` 需要代码里兼容降级方案
3. LoCoMo-10 数据量较小，更适合验证与对比，不适合支撑大规模主训练结论
4. mem0 `/search` 返回的是检索结果，不一定天然等于最终答案，需要在测试脚本中定义 response 生成策略

---

## 九、最终目标

最终交付应满足：

1. 可以用 mem0 服务完成 LoCoMo 记忆测试
2. 可以用 MiniMax judge 完成 LoCoMo 评分
3. 可以在 H800 上训练当前 Titans memory 版本
4. 可以在 H800 上跑 LoCoMo-10 全量预测
5. 输出与 `../mem0/evaluation` 对齐的对比结果表
6. 可以直接拿来和 mem0 的 LoCoMo 结果做横向对比

