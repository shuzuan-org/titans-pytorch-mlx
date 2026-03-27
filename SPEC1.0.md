# SPEC 1.0 - Frozen Qwen3.5-7B + Titans MAC Memory Training

## 1. 目标

本方案的目标是在 **Linux 训练服务器** 上，基于 **Qwen3.5-7B** 构建一个保留原始语言能力的长程记忆系统。

核心约束如下：

- 基座模型使用 **Qwen3.5-7B**
- **Qwen 参数全部冻结**，不做 LoRA，不做全参微调
- 训练对象仅限 **Titans memory 相关模块**
- 记忆机制采用 **Titans MAC (Memory as Context)** 思路
- 训练框架直接采用 **accelerate + DDP/FSDP**
- 本地代码开发后，推送到训练服务器 `~/lin/` 目录进行训练

本方案的重点不是改造 Qwen 本体，而是让 Titans memory 学会：

1. 在长上下文中识别值得写入的信息
2. 在后续问题出现时检索出有效记忆
3. 将检索出的记忆作为上下文提供给冻结的 Qwen

---

## 2. 非目标

当前版本明确不做以下内容：

- 不考虑 Mac / MLX 训练
- 不使用 LoRA
- 不训练 Qwen 任意层参数
- 不直接使用当前 `TitansMAC` 作为完整语言模型 backbone
- 不在第一阶段引入复杂多跳推理数据
- 不在第一阶段引入真实大规模通用预训练语料

说明：当前主用 backbone 是 `Qwen3.5-7B`，但代码结构必须从第一版开始支持后续切换到 `MiniMax`，因此“当前不训练 Qwen 参数”不等于“代码层只支持 Qwen”。

---

## 3. 总体架构

### 3.1 架构原则

当前仓库中的 `src/titans/models.py:240` `TitansMAC` 是一个完整的 Titans 语言模型实现，包含：

- token embedding
- MAC blocks
- lm head

该实现适合“Titans 作为主干模型”场景，但不适合直接作为“冻结 Qwen + 外挂 memory”场景。

因此，本方案采用 **Hybrid 架构**：

- **可切换的冻结 backbone** 作为语言主干，当前默认 `Qwen3.5-7B`
- **Titans memory 模块** 作为可训练外部记忆系统
- 按 MAC 思想将 memory 检索结果拼接为上下文，再送入 Qwen

后续应支持将 backbone 从 `Qwen3.5-7B` 切换到 `MiniMax`，且尽量不改动 memory 主逻辑与训练主流程。

### 3.2 拟新增核心模块

建议新增文件：

- `src/titans/frozen_backbone_mac.py`
- `src/titans/backbones/qwen.py`
- `src/titans/backbones/minimax.py`

其中主混合模型建议暂命名为：

- `FrozenBackboneWithTitansMAC`

其职责：

1. 加载指定 backbone
2. 冻结 backbone 全部参数
3. 持有 Titans memory 模块
4. 管理 chunk 级 memory state
5. 在每个 chunk 前构造 MAC 上下文
6. 调用 backbone 完成前向和 loss 计算

不同 backbone 的差异通过 adapter 处理，而不是散落在训练脚本和 memory 主逻辑中。

---

## 4. 可复用与不可复用部分

### 4.1 可直接复用

以下模块可优先复用：

- `src/titans/memory.py:65` `MemoryState`
- `src/titans/memory.py:183` `NeuralLongTermMemory`
- `PersistentMemory`（当前已被 `MACBlock` 使用）

这些模块已经具备：

- memory 检索接口
- memory 更新接口
- chunk 之间的状态传递能力

### 4.2 不直接复用

以下部分不应直接作为最终训练主干：

- `src/titans/models.py:240` `TitansMAC`
- `src/titans/models.py:126` `MACBlock` 中的 `SegmentedAttention`
- Titans 自身的 embedding/head

原因：

- 你的 backbone 是外部大模型，不是 Titans 自己的 Transformer
- attention/FFN 主体应由 Qwen 提供
- Titans 只负责 long-term memory 相关逻辑

---

## 5. 模型结构设计

### 5.1 核心思路

采用“冻结 backbone + 可训练 Titans memory”的结构：

```text
history chunks --> Titans memory write/update
question chunk --> Titans memory retrieve
[persistent tokens ; memory tokens ; question chunk] --> Frozen Backbone
Frozen Backbone --> answer logits --> loss
```

### 5.2 MAC 在本方案中的定义

此处 MAC 指 **Memory as Context**，不是把 Titans 原始 `MACBlock` 整块嵌入 Qwen，而是复用其核心思想：

1. 从 long-term memory 检索内容
2. 将检索结果作为上下文前缀
3. 与当前 chunk 一起送入 backbone

### 5.3 输入形式

推荐采用 `inputs_embeds` 方式送入 backbone，而不是将 memory 解码成离散 token。

原因：

- memory 检索结果本质上是连续表示
- 直接拼 embedding 更自然
- 不需要额外设计“memory 文本化”过程

因此，模型需要构造：

```text
[persistent embeddings] + [retrieved memory embeddings] + [chunk embeddings]
```

然后调用 backbone 前向。

---

## 6. 维度与对齐要求

这是本方案最关键的工程约束之一。

Titans memory 的 `config.dim` 必须与 backbone hidden size 对齐，或者通过一个桥接投影层对齐。

### 6.1 优先方案

优先采用：

- `TitansConfig.dim = backbone hidden_size`

这样可以避免多余投影层，减少不稳定因素。

### 6.2 备选方案

如果 Titans memory 维度与 backbone hidden size 不一致，则增加桥接层：

- `memory_to_backbone_proj`
- `backbone_to_memory_proj`（若更新时需要）

但第一版建议尽量避免双向桥接复杂度。

---

## 7. 训练参数范围

### 7.1 冻结部分

当前默认 backbone 为 Qwen3.5-7B，全部参数冻结。抽象层面要求是：所选 backbone 全部参数冻结。

Qwen3.5-7B 当前冻结范围：

- attention
- MLP
- embedding
- lm head
- norm

### 7.2 可训练部分

仅训练 Titans memory 相关参数，包括但不限于：

- `NeuralLongTermMemory`
  - `proj_k`
  - `proj_v`
  - `proj_q`
  - memory MLP 权重
  - gate 模块
  - `proj_out`
- `PersistentMemory`
- 必要的维度桥接层（如果存在）

---

## 8. 前向流程设计

### 8.1 输入组织

训练样本按 chunk 组织，而不是单条样本拼成一个超长字符串一次性送入。

建议样本结构：

- `history_chunks`: 前置上下文块
- `question_chunk`: 问题块
- `answer`: 标准答案

### 8.2 history chunk 阶段

对每个 `history_chunk`：

1. 取 chunk token embeddings / hidden 表示
2. 用当前 chunk 作为 memory query 做检索
3. 构造 `[persistent ; memory ; current chunk]`
4. 送入冻结 backbone
5. 提取用于 memory update 的表示
6. 更新 `MemoryState`

### 8.2.1 history backbone 模式默认值

当前默认采用：

- **`history_backbone_mode = full`**

即：

- 每个 `history_chunk` 都完整通过一次冻结 backbone
- 再使用 backbone 输出的 hidden states 参与 memory update

这样做的原因：

- 与最终 question 阶段的实际使用形态最一致
- history 写入时能利用更完整的语义表示
- 对 H800 + 多卡并行 + 冻结 7B backbone 场景，计算开销可接受
- 更适合作为第一版可靠基线

### 8.2.2 预留实验切换点

虽然默认值为 `full`，但仍建议保留可切换实验项：

- `history_backbone_mode = full`（默认）
- `history_backbone_mode = light`（后续对照实验）

其中：

- `full`：每个 history chunk 完整过 backbone
- `light`：仅在后续降本实验中探索，例如用更轻量的 history 编码方式替代完整 backbone

第一版实现应优先保证 `full` 路线稳定可跑。

### 8.3 question chunk 阶段

对 `question_chunk`：

1. 从 memory 检索相关内容
2. 构造 `[persistent ; memory ; question]`
3. 输入冻结 backbone
4. 对答案 token 计算 causal LM loss

### 8.4 第一版简化建议

第一版可以采用更简单的闭环：

- `history_chunks` 只负责 memory update
- `question_chunk` 才计算 loss

这样更适合先验证 memory 是否有效。

---

## 9. Memory update 信号来源

这是需要明确的一项设计决策。

### 9.1 推荐方案

使用 **backbone 处理当前 chunk 后的 hidden states** 作为 memory update 输入。

即：

```text
chunk --> Frozen Backbone --> hidden states --> Titans memory.update(...)
```

原因：

- hidden states 包含更强语义信息
- 更接近原始 MAC 中 “attention 输出再更新 memory” 的思想
- 比直接使用原始 embedding 更合理

### 9.2 默认实现建议

第一版默认采用：

- **使用 backbone 最后一层 hidden states** 作为 memory update 输入

理由：

- 语义最完整
- 实现最直接
- 最适合作为第一版基线

建议在实现中提供显式配置项，例如：

- `memory_update_source=last_hidden`

### 9.3 预留实验切换点

后续需要保留可切换方案，用于对比训练结果：

#### 方案 A：`last_hidden`（默认）
- 使用最后一层 hidden states
- 作为默认基线

#### 方案 B：`mid_hidden`
- 使用中间层 hidden states
- 目的：减少过强任务头语义偏置，观察是否更利于稳定写入

#### 方案 C：`fused_hidden`
- 对多层 hidden states 做加权或简单平均融合
- 目的：观察 multi-level features 是否更适合 memory 写入

### 9.4 不推荐方案

直接用 token embedding 更新 memory。

这样虽然实现简单，但语义压缩能力较弱。

---

## 10. 第一阶段数据集方案

### 10.1 第一阶段目标

第一阶段不追求复杂推理，主要验证：

1. memory 是否会写入关键信息
2. memory 是否能在长距离后检索出正确信息
3. 冻结 backbone 是否能利用 memory context 正确回答

### 10.2 推荐任务类型

第一阶段只做以下三类任务：

#### A. 单事实召回

示例：

- 历史中某处：`陈默的工号是 A17-9421`
- 末尾提问：`陈默的工号是什么？`

#### B. 多实体同属性干扰

示例：

- 多个人都有城市/工号/偏好
- 只问其中一个人的指定属性

#### C. 更新覆盖

示例：

- 先出现旧值：`李想住在杭州`
- 后出现新值：`后来李想搬到了苏州`
- 问：`李想现在住在哪里？`

### 10.3 第一阶段不做内容

第一阶段暂不做：

- 多跳复杂推理
- 摘要类任务
- 开放式长回答
- 真实大规模长文预训练

---

## 11. 第一阶段数据格式建议

建议新增脚本：

- `scripts/build_stage1_memory_data.py`

输出可采用 jsonl，每条样本形如：

```json
{
  "history_chunks": [
    "...",
    "..."
  ],
  "question_chunk": "...",
  "answer": "...",
  "meta": {
    "task_type": "single_kv",
    "num_entities": 12,
    "num_history_chunks": 8,
    "target_chunk_index": 5,
    "has_update": false
  }
}
```

这样做的理由：

- 与 MAC 的 chunk 处理流程一致
- 方便后续控制难度
- 方便分析 memory 检索是否命中

### 11.1 retrieval token 数量默认值与实验点

第一版建议给 memory retrieval 输出长度设置一个明确默认值。

默认建议：

- **`num_retrieved_memory_tokens = 16`**

理由：

- 比 8 个 token 有更高的信息承载能力
- 比 32 个 token 更节省显存与注意力成本
- 适合作为第一版中庸基线

建议把该项做成显式超参数，后续做 ablation：

- `8`
- `16`（默认）
- `32`
- 如有必要再测试 `64`

建议文档和训练日志中始终记录该值，因为它会直接影响：

- 总序列长度
- attention 开销
- memory 压缩强度
- 最终召回质量

---

## 12. 第一阶段 curriculum 建议

### Phase A

- 4~8 个 chunks
- 每 chunk 256~512 tokens
- 单事实
- 无更新

### Phase B

- 8~16 个 chunks
- 多实体同属性干扰
- query 放末尾

### Phase C

- 16~32 个 chunks
- 更新覆盖
- 相似实体干扰
- 部分 query 改写

原则：先让 memory 学会“稳定写读”，再增加噪音和长度。

---

## 13. 训练脚本与代码组织

建议新增以下文件：

- `src/titans/frozen_backbone_mac.py`
- `src/titans/backbones/base.py`
- `src/titans/backbones/qwen.py`
- `src/titans/backbones/minimax.py`
- `src/titans/data/stage1_memory_dataset.py`
- `scripts/build_stage1_memory_data.py`
- `scripts/train_frozen_backbone_mac.py`
- `configs/accelerate_ddp.yaml`
- `configs/accelerate_fsdp.yaml`

### 13.1 `src/titans/frozen_backbone_mac.py`

职责：

- 加载冻结 backbone
- 初始化 Titans memory
- 实现 chunk 级 memory read/write
- 构造 memory context prefix
- 输出训练 loss

### 13.2 `src/titans/backbones/base.py`

职责：

- 定义统一 backbone adapter 接口
- 屏蔽不同模型在 tokenizer、embedding、hidden states、loss 输入上的差异

设计原则：

- adapter 只做**薄封装**，不重写训练逻辑
- adapter 的核心职责是统一冻结 backbone 的输入输出边界
- 从概念上看，backbone 差异主要体现在 **hidden size** 与 **tokenizer/vocab**
- 但从工程实现上，还必须统一 `inputs_embeds` 前向、hidden states 提取、special tokens/pad 处理

建议最小接口如下：

#### 必须统一的接口

- `tokenize_batch(texts, ...)`
- `embed_input_ids(input_ids)`
- `forward_with_embeds(inputs_embeds, attention_mask, output_hidden_states=True)`
- `get_hidden_size()`
- `freeze_parameters()`

#### 建议统一的接口

- `get_pad_token_id()`
- `get_eos_token_id()`
- `get_vocab_size()`

#### 返回结构建议

建议 `forward_with_embeds(...)` 返回统一结构，至少包含：

- `logits`
- `last_hidden_state`
- `hidden_states`

这样训练主逻辑可以避免感知不同 backbone 的底层字段差异。

#### 为什么不只考虑 dim 和 vocab

虽然最核心的差异确实首先是：

- hidden dim
- tokenizer / vocab

但如果只按这两项设计，后续切换 backbone 时仍然容易在以下位置产生分支逻辑：

- `inputs_embeds` 是否稳定支持
- hidden states 如何开启与读取
- logits 从哪里取
- pad/eos 等 special token 如何处理

因此这些都应统一放在 adapter 层处理，而不是散落在训练主脚本中。

### 13.3 `src/titans/backbones/qwen.py`

职责：

- 实现 Qwen3.5-7B 的 adapter
- 适配 tokenizer / model / hidden states / inputs_embeds

### 13.4 `src/titans/backbones/minimax.py`

职责：

- 预留 MiniMax adapter
- 保证 future switch 时训练主逻辑不变

### 13.5 `scripts/train_frozen_backbone_mac.py`

职责：

- 解析训练参数
- 创建 dataset / dataloader
- 使用 `accelerate` 训练
- 支持 DDP / FSDP
- 保存 checkpoint

训练脚本中应通过参数选择 backbone，例如：

- `--backbone qwen`
- 后续支持 `--backbone minimax`

此外建议至少支持以下实验参数：

- `--memory-update-source {last_hidden,mid_hidden,fused_hidden}`
- `--num-retrieved-memory-tokens {8,16,32,64}`
- `--loss-mask-scope {answer_only,question_answer}`
- `--history-backbone-mode {full,light}`

---

## 14. 分布式训练方案

### 14.1 统一入口

训练统一使用：

```bash
accelerate launch --config_file <config> scripts/train_qwen35_mac.py ...
```

不再维护单独的 `torchrun` 版本脚本。

### 14.2 DDP / FSDP 策略

建议从代码结构上直接支持两种模式：

- DDP：用于快速跑通
- FSDP：用于更稳妥承载 7B 及未来更大模型

### 14.3 为什么需要 FSDP-ready

虽然当前 Qwen 冻结，但：

- 7B 本身显存占用仍高
- 后续可能切更大 backbone
- FSDP 能降低单卡承载压力

因此，当前就应把训练入口写成 FSDP 兼容。

---

## 15. checkpoint 策略

建议明确区分两类保存方式：

### 15.1 memory-only checkpoint

保存：

- Titans memory 参数
- persistent memory 参数
- 桥接层参数
- backbone adapter 配置
- 训练配置

适用于：

- 实验迭代
- 快速加载
- 与固定 Qwen backbone 配套复用

### 15.2 full training checkpoint

保存：

- accelerator/FSDP 恢复所需状态
- optimizer state
- scheduler state
- step / epoch 信息

适用于：

- 中断恢复训练

---

## 16. 服务器工作流

### 16.1 目录

建议服务器目录：

- `~/lin/titans_lin/`

### 16.2 conda 环境

建议新建环境：

- `titans_train`

### 16.3 训练流程

1. 本地在当前分支修改代码
2. 推送到远端仓库或同步到服务器目录
3. 在服务器 `~/lin/titans_lin/` 拉取最新代码
4. 激活 `titans_train`
5. 通过 `accelerate launch` 启动训练

---

## 17. 第一版实验目标

第一版不追求最终性能，先验证最小闭环：

1. 冻结 Qwen3.5-7B
2. Titans memory 可训练
3. chunk 化 history 能更新 memory
4. question 能从 memory 检索出有效上下文
5. 模型在合成数据上学会稳定召回答案

只要这五点打通，说明：

- 架构方向成立
- memory 与 frozen Qwen 的接口成立
- accelerate + DDP/FSDP 路线成立

---

## 18. 当前待确认问题

以下问题需要继续讨论并在后续版本明确：

1. Qwen3.5-7B 的具体 hidden size 与 Titans `dim` 如何对齐
2. backbone adapter 最小统一接口如何定义
3. memory retrieve 输出长度设为多少个 tokens 最合适
4. `history_backbone_mode=full` 作为默认方案时，light 模式是否还有保留价值
5. memory update 的输入取最后一层 hidden state，还是中间层 hidden state
6. question 阶段 loss 是否只监督 answer 区域
7. FSDP auto wrap policy 具体如何配置
8. checkpoint 默认保存 memory-only 还是双保存

### 18.1 当前默认建议

在没有实验结论前，第一版采用以下默认值：

- `memory_update_source = last_hidden`
- `num_retrieved_memory_tokens = 16`
- `loss_mask_scope = answer_only`

### 18.2 loss mask 默认方案与实验点

第一版默认建议：

- **只监督 answer 区域**，即 `loss_mask_scope = answer_only`

理由：

- 更符合“history/query 提供条件，answer 才是目标输出”的任务定义
- 避免 question token 对 loss 产生干扰
- 更适合作为 memory 贡献的基线评估方式

后续需要保留对比方案：

#### 方案 A：`answer_only`（默认）
- 只对 answer tokens 计算 loss

#### 方案 B：`question_answer`
- 对 question + answer 全部计算 causal LM loss
- 目的：观察更强语言建模约束是否有助于稳定训练

---

## 19. 当前结论

本方案的核心结论是：

- 不是直接训练 `TitansMAC`
- 而是基于 `TitansMAC` 的 **Memory as Context 思路**，实现一个 **Frozen Backbone + Titans memory hybrid**
- 当前 backbone 用冻结 Qwen3.5-7B
- 代码结构需要支持后续切换到 MiniMax
- trainable 部分只保留 Titans memory
- 分布式直接走 accelerate + DDP/FSDP
- 第一阶段先用合成 chunk 化记忆数据打通闭环

这是当前版本建议的落地方向。
