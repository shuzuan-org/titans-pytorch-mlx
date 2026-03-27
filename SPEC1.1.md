# SPEC 1.1 - Stage1 测试期部署方案

## 1. 目标

本方案定义 `stage1` 在测试期的部署抽象，把当前训练态的单次 `forward` 拆成可独立调用的两类能力：

- `write_memory`：只写入 memory
- `chat_with_memory`：只读取 memory 并回答

当前重点是验证 memory API、session state 管理和最小可运行问答闭环，不做复杂服务治理。

## 2. 边界

### 2.1 当前包含

- 单机单模型 runtime
- 显式 `session_id -> session memory state` 管理
- memory state 序列化 / 反序列化
- 写入与问答分离
- 仅保存模型运行所需 state 与最小 metadata

### 2.2 当前不包含

- 不保存原始文本
- 不保存原始聊天历史
- 不保存 tokenizer 输出
- 不做多副本分布式一致性治理
- 不做审计日志 / event log

## 3. 复用现有能力

优先复用以下已有组件：

- `src/titans/stage1_models.py`
  - `FrozenBackboneAdapter.encode_tokens`
  - `FrozenBackboneAdapter.embed_input_ids`
  - `DenseTimelineMemory.init_state`
  - `DenseTimelineMemory.update`
  - `DenseTimelineMemory.retrieve`
  - `FrozenBackboneWithTimelineMemory._pool_hidden`

训练脚本 `scripts/train_stage1_distributed.py` 保持训练语义不变；部署侧通过新的 runtime 封装调用模型级公开方法。

## 4. 部署抽象

### 4.1 核心类

- `Stage1DeploymentRuntime`
  - 加载模型、checkpoint、session store
- `Stage1SessionState`
  - 表示单个 `session_id` 的 memory state
- `Stage1SessionStore`
  - 管理 `session_id -> state`
- `Stage1MemoryWriter`
  - 执行 write 请求
- `Stage1ChatGenerator`
  - 执行 query retrieve + generate

### 4.2 模型级公开方法

`src/titans/stage1_models.py` 需要暴露：

- `init_session_state(session_id)`
- `write_texts(state, texts)`
- `answer_query(state, query, generation_config)`
- `serialize_session_state(state)`
- `deserialize_session_state(payload)`

## 5. API 方案

### 5.1 写入接口

- 建议路由：`POST /v1/memory/write`
- 语义：只写入，不回答

输入：

- `session_id`
- `contents: string[]` 或 `content: string`
- 可选 `idempotency_key`

处理流程：

1. 读取或初始化 session state
2. tokenize 输入文本
3. `backbone.encode_tokens(...)`
4. `_pool_hidden(...)` 得到 `write_repr`
5. `memory.update(state, write_repr)`
6. 保存新 state
7. `memory_version += 1`

输出：

- `session_id`
- `memory_version`
- `num_items_written`

### 5.2 对话接口

- 建议路由：`POST /v1/chat/respond`
- 语义：只读取 memory 并回答，不写入

输入：

- `session_id`
- `query`
- 可选 generation 参数，如：
  - `max_new_tokens`
  - `temperature`
  - `do_sample`

处理流程：

1. 读取已有 session state
2. 编码 query 得到 `query_repr`
3. `memory.retrieve(state, query_repr)`
4. 将 `retrieved` 作为 prefix embedding 拼到 query 前
5. 调用 stage1 generation wrapper 生成回答

输出：

- `answer`
- 可选 debug：
  - `memory_version`
  - `retrieval_weights`

### 5.3 配套接口

- `DELETE /v1/sessions/{session_id}`
  - 清空 session memory
- `GET /v1/sessions/{session_id}`
  - 仅返回 metadata，不返回原文

## 6. Session State 结构

测试期只保留派生 state：

- `format_version`
- `session_id`
- `memory_version`
- `model_signature`
  - `backbone_name`
  - `hidden_size`
  - `memory_slots`
  - `config_hash`
- `memory_state`
  - `DenseTimelineMemory` 当前张量状态
- `stats`
  - `num_writes`
  - `num_queries`
  - `updated_at`

原则：**不保存原文，不保存完整历史，不保存 tokenizer 中间结果。**

## 7. 状态存储策略

第一版建议：

- 运行中用内存字典保存 session state
- 测试时可选 `torch.save / torch.load` snapshot
- snapshot 内容只包含 state payload

这样可以先验证 deployment API 是否合理，再决定是否引入外部 KV / DB。

## 8. 并发与一致性

- 同一 `session_id` 的 write 请求必须串行化
- chat 请求基于稳定快照，不在回答过程中修改 memory 内容
- `write` 成功后 `memory_version += 1`
- `chat` 默认不增加 `memory_version`
- chat 可以更新最小 metadata（如 `num_queries`、`updated_at`），但不改 `memory_state`

## 9. 关键实现差距与补齐项

当前训练代码缺少这些部署必需能力：

1. stage1 模型级 session / write / chat 公共方法
2. 独立 generation 路径
3. session state 序列化能力
4. 独立于训练脚本的 runtime 封装

建议补齐：

- `src/titans/stage1_models.py`
- `src/titans/stage1_runtime.py`

可选后续新增：

- `scripts/serve_stage1.py`

## 10. 验证方案

### 10.1 文档验证

需要确认本文档是否明确：

- 两个主接口
- session state 结构
- 不保存原文的范围
- 写读分离规则

### 10.2 单元验证

- 初始化空 session state
- 连续 write 两条信息后，state 发生变化
- 对同一 state 执行 chat，不修改 `memory_state`
- serialize -> deserialize 后，retrieve 结果保持一致或近似一致

### 10.3 集成验证

最小 smoke case：

1. `write("张三住在上海")`
2. `chat("张三住在哪？")`
3. 预期回答包含“上海”

更新覆盖 case：

1. `write("张三后来搬到北京")`
2. `chat("张三现在住在哪？")`
3. 用于验证更新后的检索/回答路径

## 11. 当前落地文件

- `SPEC1.1.md`
- `src/titans/stage1_models.py`
- `src/titans/stage1_runtime.py`

