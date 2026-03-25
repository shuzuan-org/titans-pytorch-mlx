# MSA: Memory Sparse Attention 调研笔记

> **论文**：MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens
> **作者**：Yu Chen, Runkai Chen 等（Evermind / Shanda Group / 北京大学）
> **代码**：github.com/EverMind-AI/MSA
> **Backbone**：Qwen3-4B-Instruct-2507
> **定位**：端到端可微的 Latent State-Based Memory，介于 RAG 和参数记忆之间

---

## 1. 核心问题

Full attention 的 O(L²) 复杂度限制了 LLM 的有效上下文到 128K–1M。现有方案各有缺陷：

| 范式 | 代表 | 精度 | 可扩展性 | 端到端可微 | 灾难性遗忘 |
|------|------|------|---------|----------|----------|
| 参数记忆 | LoRA/CPT, **Titans** | High/Medium | No | — | Yes |
| 外部存储 | RAG, MemAgent | Medium | Yes (O(L)) | No | No |
| 隐状态记忆 | Linear Attn (RWKV, DeltaNet) | Low | Yes (O(L)) | No | Yes |
| **MSA (本文)** | — | **High** | **Yes (O(L))** | **Yes** | **No** |

MSA 的核心主张：**把检索融入 attention 机制本身**，用稀疏 attention + document-wise RoPE 实现近线性复杂度，同时保持端到端可微。

---

## 2. 架构设计

### 2.1 Sparse Attention Mechanism（§3.2.1）

不是 token-level 的稀疏 mask，而是 **document-level 的 top-k 选择**：

```
文档 d_i → backbone → K_i, V_i（标准 KV）
                     → K_i^R（Router K Projector，单独投影矩阵 W_KR）

chunk-mean pooling（P=64 tokens/chunk）→ K̄_i, V̄_i, K̄_i^R

Query → Q_q, K_q, V_q（标准）
      → Q_q^R（Router Q Projector，W_QR）

相关性：S_ij = max_chunk( mean_head( cos(Q_q^R, K̄_ij^R) ) )
文档分数：s_i = max_j S_ij
选择：I = Top-k({s_i}, k=16)

最终 attention：
  K_ctx = [K̄_selected ; K_q]
  V_ctx = [V̄_selected ; V_q]
  Output = Attention(Q_q, K_ctx, V_ctx)
```

**关键设计**：MSA routing **只作用于 upper layers**（后半层）。低层保持 Independent Document Processing（各文档独立过 self-attention），不做跨文档路由。原因：低层 hidden states 语义不够，routing 无效。

### 2.2 Parallel & Global RoPE（§3.2.2）

双 RoPE 策略解耦训练/推理长度：

- **文档侧**：每个文档 position ID 从 0 重置（document-wise RoPE）
- **Query 侧**：position ID 从 k 开始（k = 选中文档数），让 query 在位置上"接续"背景文档

效果：64K context 训练 → 100M 推理无退化。

### 2.3 训练

**Continual Pre-training**：158.95B tokens，17.9M queries，训练目标是 Generative Retrieval（自回归生成相关文档 ID）。

**Auxiliary Routing Loss**（公式 5）：对比学习，正样本=相关文档，负样本=不相关文档。

```
L_aux = -1/|P| Σ log( exp(s_i⁺/τ) / (exp(s_i⁺/τ) + Σ exp(s_i,j⁻/τ)) )
```

两阶段调度：
- Warmup：L = 0.1·L_LLM + L_aux（训路由器，lr=1e-4）
- 主阶段：L = L_LLM + 0.1·L_aux（训生成，lr=6e-6）

**Post-training**：两阶段 curriculum SFT（8K → 64K context）。

### 2.4 Memory Interleave（§3.5）

迭代式检索-生成，不是一次检索：

```
Query → 生成 doc IDs（数量自适应）→ 加载原文 → 拼入 context
     → 基于扩展 context 再生成 doc IDs → 加载 → ...
     → 判断够了 → 生成最终答案
```

训练时：每条多跳检索链拆成多个单步样本。

### 2.5 三级存储推理引擎（§3.4.2）

100M tokens / 2×A800 的关键工程：

| 存储层 | 内容 | 位置 | 100M 规模 |
|--------|------|------|----------|
| GPU VRAM | K̄^R（路由键）| 分布在多 GPU | ~56GB |
| CPU DRAM | K̄, V̄（内容 KV）| 主机内存 | ~169GB |
| 按需搬运 | 选中文档的 K̄, V̄ | 异步 GPU←CPU | top-16 文档 |

---

## 3. 实验结果

### 3.1 QA 任务（Table 2, 3）

9 个数据集，memory bank 277K–10M tokens，LLM judge 评分 0–5。

**同 backbone 对比（Qwen3-4B）**：

| 方法 | 均分 | vs MSA |
|------|------|--------|
| Standard RAG R@10 | 3.242 | -16.0% |
| RAG + Rerank R@10 | 3.372 | -11.5% |
| HippoRAG2 R@10 | 3.275 | -14.8% |
| **MSA @adaptive** | **3.760** | — |

**vs SOTA RAG（大模型 backbone）**：MSA 4B 在 9 个数据集中 4 个超过 KaLMv2+Qwen3-235B。

### 3.2 NIAH（Figure 4）

| 模型 | 32K | 128K | 512K | 1M |
|------|-----|------|------|-----|
| Qwen3-4B（backbone）| 0.95 | 0.99 | 0.42 | 0.25 |
| Qwen3-Next-80B-A3B | 1.00 | 1.00 | 0.88 | 0.81 |
| RL-MemoryAgent-14B | 0.98 | 0.97 | 0.95 | 0.93 |
| **MSA-4B** | **0.99** | **0.98** | **0.97** | **0.95** |

16K → 100M 退化 < 9%。

### 3.3 Ablation（Table 4，基于 MSA-S1）

| 变体 | 均分 | 下降 | 最严重影响 |
|------|------|------|----------|
| MSA-S2 (Full) | 3.976 | — | — |
| MSA-S1 (Full) | 3.694 | — (baseline) | — |
| w/o Memory Interleave | 3.497 | -5.3% | HotpotQA -19.2% |
| w/o Continual Pre-training | 2.537 | -31.3% | HotpotQA -43.1% |
| w/o Original Text | 2.325 | -37.1% | DuReader -46.2% |
| MSA-S2 vs S1 (curriculum) | +7.6% | — | MS MARCO +29.5% |

**最重要的 ablation 结论**：
- CPT 是基础，没有 CPT 路由器基本废掉（-31.3%）
- Original Text 不可省——光生成 doc ID 不够，必须加载原文才能精确回答（-37.1%）
- Memory Interleave 对多跳任务关键（HotpotQA -19.2%）
- Curriculum（8K→64K）对大 memory bank 外推关键（MS MARCO +29.5%）

---

## 4. 与本项目（Titans NLM / Memory Oracle）的关系

### 4.1 路线定位

| 维度 | MSA | 我们（NLM） |
|------|-----|-----------|
| 记忆载体 | 压缩 KV cache（外部） | 模型权重（内部） |
| 推理存储 | GPU routing keys + CPU KV | 零额外存储 |
| 在线学习 | 不支持（需离线预编码） | 支持（推理即学习） |
| 容量上限 | 100M+ tokens（靠存储） | 受限于权重容量 |
| 部署复杂度 | 需要三级存储基础设施 | 只需模型文件 |
| 灾难性遗忘 | 无（KV 不被覆盖） | 有（权重更新覆盖旧信息） |
| 跨文档推理 | 靠 Interleave 间接建模 | 权重矩阵隐式交叉引用 |

**核心差异**：MSA 是 "可微 RAG"——检索变成了可微的稀疏 attention，但知识仍在 KV cache 外部。NLM 是 "权重即记忆"——无检索，零额外存储，在线更新。

**不是竞争关系，是不同场景**：
- MSA：有明确文档库、需要精确检索、文档量巨大（百万级）
- NLM：知识内化后自主推理、持续在线学习、部署极简（无向量库/KV 基础设施）

### 4.2 MSA 对 Titans 的批评（Table 1）

MSA 给 Titans 的标签：Precision=Medium, Lifetime Memory=No, Catastrophic Forgetting=Yes。

这是合理的批评。我们在 LoCoMo 上的 F1=0.896 证明了**有限规模**下 NLM 有效，但尚未证明在大规模下仍然有效。应对策略：
- 不争 100M token 场景，明确定位于 "有限容量、零存储、在线学习"
- 灾难性遗忘 → Titans 的 weight decay (α_t) 设计上就是有意遗忘，问题是遗忘是否可控

### 4.3 可借鉴的技术点

#### 优先级 1：分层注入策略

MSA 经验（§3.2.1）：低层 hidden states 语义不足，routing 无效，只在 upper layers 做 MSA。

**对应 ablation**：当前 NLM 注入所有 28 层。可以试：
- 只注入后 14 层（upper half）
- 只注入后 7 层（top quarter）
- 预期：F1 持平或略降，但参数量减半，步速提升

**这是成本最低、信息量最大的 ablation。**

#### 优先级 2：选择性写入 gate

MSA 的 L_aux 对比 loss 训练路由器区分相关/不相关文档。NLM 当前依赖 surprise（梯度大小）隐式决定写入强度，没有显式 gate。

可以加轻量 gate network，用类似对比 loss 训练：
- 正样本：后续 query 会命中的信息 → gate 开
- 负样本：噪音/闲聊 → gate 关
- 天然配合 Stage 2 的 40-60% 噪音设计

#### 优先级 3：多轮回忆推理

MSA 的 Memory Interleave 对多跳 QA 提升 19.2%（HotpotQA）。

NLM 类比：当前 eval 是单次 forward 读写。可以在推理阶段做两轮：
1. 第一轮 forward → NLM 写入 → 初步回忆
2. 用初步回忆的线索作为新 query → 再过一次 NLM 读取 → 联想链

不需要改训练，只改推理流程。

### 4.4 不值得借鉴的

| 技术 | 原因 |
|------|------|
| 158.95B token CPT | 预算不允许，且我们不需要训路由器 |
| 三级存储引擎 | 纯工程，和 NLM 路线无关 |
| Document-wise RoPE | 我们的场景是连续对话流，不是独立文档库，position 重置可能破坏时序 |
| Top-k document selection | NLM 不做检索，直接从权重读取 |

### 4.5 NLM 的优势空间（MSA 做不到的）

1. **零存储部署**：NLM 只需模型文件，MSA 需要 GPU routing keys + CPU KV cache 基础设施
2. **真正在线学习**：MSA 的文档需要离线预编码（Stage 1），新文档需要重新 forward 并入库；NLM 在推理过程中持续学习
3. **隐式跨文档关联**：MSA 每次只选 top-16 文档，跨文档关系靠 Interleave 间接建模；NLM 所有信息在同一权重矩阵中，理论上可以自然交叉引用（需验证）
4. **MSA 自述局限**（§7）：对 "tightly coupled dependencies across multiple documents" 建模困难

---

## 5. 行动项

| 优先级 | 行动 | 预期收益 | 成本 |
|--------|------|---------|------|
| P0 | Ablation：NLM 只注入 upper layers（后14层 / 后7层）| 验证分层注入效果，可能减半参数 | 1h H800 |
| P1 | 设计 write gate + 对比 loss 原型 | 提高噪音环境下的写入选择性 | 需改 memory.py |
| P2 | 推理时两轮 forward 实验 | 多跳回忆能力 | 只改 eval 脚本 |
| P3 | LoCoMo 跨 session 关联题单独统计 F1 | 量化 NLM 的跨文档优势 | 只改 eval 分析 |

---

*调研日期：2026-03-25*
*原文 PDF：`docs/MSA__Memory_Sparse_Attention_for_Efficient_End_to_End_Memory_Model_Scaling_to_100M_Tokens.pdf`*
