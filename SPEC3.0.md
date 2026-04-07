# SPEC3.0

## 目标

将当前 `titans_lin` 分支的 memory 实现从 `DenseTimelineMemory`（slot 矩阵）升级为 `NeuralLongTermMemory`（NLM，MLP 权重级 memory），对齐原始 Titans 论文的 MAC 方案。

底座模型（Qwen2.5-7B）不变，但后续可能加 LoRA 微调。

---

## 当前进展

### 已完成

1. NLM 模块移植完成（`src/titans/neural_memory.py`）
2. `DenseTimelineMemory` 已替换为 `NeuralLongTermMemory`
3. write 参数冻结（和原始 Titans 一致，NLM 内部 test-time learning）
4. retrieve 用 64 个 learnable query tokens（不再 pool 成 1 个 token）
5. prefix norm 缩放（让 prefix 和 token embedding 同量级）
6. 训练/服务/测试链路已适配

### 当前发现的问题

1. **loss 收敛但答案没变**
   - 原因：训练时 backbone 靠 teacher forcing 就能预测 answer tokens，不依赖 prefix
   - prefix 太小时（norm ≈ 1.5 vs token norm ≈ 134）被 backbone 完全忽略
   - 已通过 norm 缩放修复，待验证

2. **冻结 backbone + NLM prefix 的结构性矛盾**
   - backbone 在预训练时从未见过 prefix tokens
   - backbone 不知道怎么利用这些"外星向量"
   - write 参数冻结，写入质量取决于随机初始化
   - 即使 norm 缩放后，prefix 内容可能仍然对 backbone 无意义

3. **与 Prefix Tuning 的类比**
   - 当前方案本质上类似 Prefix Tuning
   - 但 Prefix Tuning 的 prefix 是固定可学习的
   - 而当前 prefix 每次随 history 变化，backbone 难以学到稳定模式

---

## 一、当前问题

### DenseTimelineMemory 的局限

1. memory = `[16, 3584]` 的 slot 矩阵，总共 0.22 MB
2. 写入方式 = attention-weighted additive update（在所有 slot 上叠加）
3. 不是"替换某个 slot"，也不是"梯度更新权重"
4. 写入越多，早期信息被冲掉
5. 对 `stage1_timeline_v2`（1~16 条短事实）刚好够用
6. 对 `locomo10`（120 条长对话）完全不够

### 原始 Titans MAC 的 memory

1. memory = `NeuralLongTermMemory` 内部的 `MemoryMLP`
2. 单层时，权重 = `[dim, dim]` = `[3584, 3584]` ≈ 98 MB
3. 写入方式 = 对 MLP 权重做梯度下降
4. 读取方式 = 把 query 过 MLP 得到检索结果
5. 有 momentum（动量）和 weight decay（遗忘）机制
6. 容量比当前方案大 ~450 倍

---

## 二、升级路线

### 路线 A（本次实施）：保持 MAC 思路，只换 memory 实现

#### 不变的部分

- 底座：Qwen2.5-7B，冻结，不改
- 推理方式：memory 检索结果作为 prefix embedding 拼到输入前
- 训练方式：冻结 backbone，只训练 memory 相关参数
- 服务接口：`serve_stage1.py` 的 HTTP API 不变
- 数据格式：`history_chunks / question_chunk / answer` 不变

#### 要改的部分

1. 把 `DenseTimelineMemory` 替换为 `NeuralLongTermMemory`
2. 写入逻辑：从 additive update 改为梯度下降更新 MLP 权重
3. 读取逻辑：从 attention over slots 改为 query 过 MLP
4. session state：从 `[slots, dim]` 矩阵变为 MLP weights + momentum
5. memory 初始化：从 `initial_memory` 参数变为小噪声初始化的 MLP 权重

#### 影响范围

- `src/titans/stage1_models.py`：核心改动
- `src/titans/stage1_runtime.py`：session state 序列化/反序列化
- `scripts/serve_stage1.py`：不需要改接口，但 checkpoint 格式会变
- `scripts/train_stage1_distributed.py`：可能需要调整 memory 相关超参
- `tests/test_stage1_runtime.py`：需要适配新 memory state 结构

#### 新增超参

- `memory_lr`：memory 写入时的学习率（theta_t）
- `memory_momentum`：动量系数（eta_t）
- `memory_decay`：遗忘/衰减系数（alpha_t）
- `num_memory_layers`：memory MLP 深度（1 = 线性，>=2 = 深层）

#### 预期收益

- memory 容量从 0.22 MB 提升到 ~98 MB
- 写入机制从"全局叠加"改为"梯度学习"
- 更自然的遗忘机制（weight decay）
- 能支撑 locomo10 级别的长对话记忆

#### 预期风险

- 训练速度会慢（每次写入需要计算梯度）
- GPU 内存占用会增加
- checkpoint 格式和旧版不兼容
- 需要调参：memory_lr / momentum / decay

---

### 路线 B（后续考虑）：走 MAL 注入路线

#### 核心区别

路线 A 是 memory 在 backbone **外面**，作为 prefix。
路线 B 是 memory 在 backbone **里面**，注入到 decoder layers。

#### 具体做法

参考 `../titans/src/titans/qwen35_injection.py`：

1. 把 Qwen 的每个 decoder layer 包装成 `QwenLayerWithMemory`
2. forward 顺序变成：
   - `hidden_states → NLM retrieve → residual add → 原始 Qwen layer`
3. 写入时：NLM 在处理 history tokens 时自动更新权重
4. 读取时：NLM 在处理 query tokens 时返回检索结果，直接加到 hidden states

#### 优点

- 和原始 Titans 论文最接近
- memory 和 attention 深度融合，信息流更自然
- 不需要显式的 prefix 拼接

#### 缺点

- 改动量大：需要修改 backbone 结构
- 调试复杂：memory 在每一层都参与计算
- 冻结 backbone 时可能有 grad 兼容问题
- 推理方式完全不同，train/serve/eval 全要改

#### 什么时候走路线 B

1. 路线 A 验证后，如果 prefix 方式的 memory 利用率不够好
2. 或者需要更深层的 memory-attention 融合
3. 或者要和原始 titans 分支的 MAL 结果做严格对比

---

## 三、路线 A 实施计划

### Phase 1：引入 NLM 模块

1. 从 `../titans/src/titans/memory.py` 引入：
   - `MemoryMLP`
   - `MemoryState`
   - `NeuralLongTermMemory`
2. 或者在 `src/titans/` 下新建 `neural_memory.py`，移植核心代码
3. 保留 `../titans/src/titans/config.py` 中相关配置

### Phase 2：替换 stage1_models.py 中的 memory

1. `FrozenBackboneWithTimelineMemory` 中：
   - 把 `DenseTimelineMemory` 替换为 `NeuralLongTermMemory`
   - 修改 `write_texts()`：从 additive update 改为梯度下降
   - 修改 `retrieve_from_state()`：从 attention over slots 改为 MLP forward
   - 修改 `init_session_state()`：初始化 MLP weights + momentum
   - 修改 `clone/serialize/deserialize_session_state()`：适配新 state 结构

### Phase 3：适配训练脚本

1. 新增超参：`memory_lr`, `memory_momentum`, `memory_decay`, `num_memory_layers`
2. `Stage1ModelConfig` 加对应字段
3. 训练脚本加对应 CLI 参数

### Phase 4：适配 runtime 和服务

1. `stage1_runtime.py`：session state 序列化格式变更
2. `serve_stage1.py`：接口不变，但内部 state 格式变了
3. checkpoint 格式：trainable_state_dict 内容会变

### Phase 5：测试验证

1. 先用 `stage1_timeline_v2` 验证能训练、能推理
2. 再用 `locomo10` 验证长对话记忆能力
3. 对比新旧方案的 loss 收敛和 eval 效果

---

## 四、关键技术参考

### 原始代码位置

- memory 核心：`../titans/src/titans/memory.py`
- 配置：`../titans/src/titans/config.py`
- MAC 模型：`../titans/src/titans/models.py` → `TitansMAC` / `MACBlock`
- MAL 注入：`../titans/src/titans/qwen35_injection.py`

### 原始 NLM 写入公式（论文）

```
M_t = (1 - alpha_t) * M_{t-1} + S_t
S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
loss(M; x) = ||M(k) - v||^2
```

其中：
- `alpha_t`：遗忘因子（weight decay）
- `eta_t`：动量系数
- `theta_t`：学习率
- `S_t`：surprise（动量累积的梯度）
- `M`：memory MLP
- `k, v`：从输入投影得到的 key 和 value

### 原始 NLM 读取

```
output = proj_out(M(proj_q(query)))
```

即：query 投影后过 memory MLP，再投影回来。

---

## 五、和 SPEC2.0 的关系

SPEC2.0 中的 LoCoMo benchmark pipeline 保持不变：
- mem0 导入/测试链路不受影响
- titans 测试链路接口不变（HTTP API 不变）
- 评分方式不变
- 只是 titans 内部的 memory 实现升级了

升级完成后，重新训练 + 重新跑 LoCoMo 测试即可。

---

## 六、下一步方案（待当前版本验证后实施）

### 背景

当前版本（NLM + 冻结 backbone + 64 query tokens prefix）存在一个结构性问题：

- backbone（Qwen2.5-7B）在预训练时从未见过 prefix tokens
- backbone 的 attention 不知道怎么利用这些 memory prefix
- 即使 norm 缩放后，prefix 内容对 backbone 可能仍然无意义
- 本质上类似 Prefix Tuning，但 backbone 完全冻结时效果有限

### 分步方案

#### 第一步：加 LoRA 微调 backbone

目标：让 backbone 学会"看到 prefix 时怎么反应"。

做法：
1. 在 Qwen2.5-7B 的 attention 层加 LoRA
2. LoRA 目标模块：`q_proj, k_proj, v_proj, o_proj`
3. LoRA rank: 16（和主分支一致）
4. 保持当前 MAC prefix 方式不变
5. NLM 结构不变

预期收益：
- backbone 不再完全冻结
- LoRA 让 backbone 学会利用 prefix tokens
- 改动最小，最快验证"backbone 冻结"是否是主要瓶颈

判断标准：
- 如果加 LoRA 后 memory 答案确实改变了 → 说明瓶颈在 backbone 冻结
- 如果加 LoRA 后仍然没变化 → 说明 prefix 方式本身有瓶颈，需要走第二步

参考：主分支 `scripts/train_memory_oracle.py` 中的 LoRA 配置。

#### 第二步：走路线 B（MAL 注入）

目标：把 NLM 注入到 backbone 的 decoder layers 内部，和主分支对齐。

做法：
1. 参考 `../titans/src/titans/qwen35_injection.py`
2. 把每个 Qwen decoder layer 包装成 `QwenLayerWithMemory`
3. forward 顺序：hidden_states → NLM retrieve → residual add → 原始 Qwen layer
4. 不再用 prefix 方式

预期收益：
- memory 信息通过 residual add 融合到 hidden states，backbone 不需要"理解 prefix"
- 和原始 Titans 论文最接近
- 和主分支的 MAL 方案完全对齐

判断标准：
- 和主分支的 LoCoMo 结果对比
- 和 mem0 对比

参考：
- 主分支方案 = Qwen3.5-0.8B + LoRA + NLM MAL 注入
- 本分支方案 = Qwen2.5-7B + LoRA + NLM MAL 注入

#### 第三步（可选）：方案 C，让 write 也参与训练

目标：解决 write 参数随机初始化不被训练的问题。

做法：
- 单层线性 memory 的更新公式可微分
- 不 detach NLMState
- 梯度从 loss → retrieve → W_new → write 公式 → proj_k, proj_v, gates 完整流回

适用条件：
- 第一步或第二步验证后，如果发现 write 质量仍然是瓶颈
- 只适用于 num_memory_layers=1

### 优先级（已确认）

1. 先验证当前版本（NLM + 冻结 backbone + 64 prefix + norm 缩放）
2. 加 LoRA（第一步）— 验证 backbone 能否学会利用 prefix
3. 方案 C：write 也参与训练（第二步）— 验证 write 质量对结果影响
4. 换成 MAL 注入（第三步）— 验证注入方式 vs prefix 方式的差异

每步只改一个变量，逐步定位瓶颈。

### 和主分支的对齐路线

```
当前 → 加 LoRA → MAL 注入 → 最终和主分支方案一致
```

主分支：Qwen3.5-0.8B + LoRA + NLM MAL
本分支目标：Qwen2.5-7B + LoRA + NLM MAL
