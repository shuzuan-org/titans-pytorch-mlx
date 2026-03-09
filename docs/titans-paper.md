# Titans 论文与后续工作（L3 参考文档）

> 完整参考资料，按需查阅。决策结论 → L1/decisions.md。

---

## 原论文：Titans: Learning to Memorize at Test Time

**arXiv**: 2501.00663
**机构**: Google DeepMind
**发布**: 2024 年 12 月

### 核心机制

三层记忆设计：

| 记忆类型 | 实现 | 窗口 | 特点 |
|---------|------|------|------|
| 短期（注意力）| Self-Attention | 有限窗口 | 精确，O(N²) |
| 长期（神经记忆）| NeuralLongTermMemory | 无限 | 用梯度更新权重 |
| 持久（参数）| PersistentMemory | 全局 | 训练后固定 |

**Memory 更新规则**（核心公式）：
```
S_t = eta_t * S_{t-1} - theta_t * grad(||M(k_t) - v_t||²)
M_t = (1 - alpha_t) * M_{t-1} + S_t
```
- `alpha_t`：遗忘因子（data-dependent weight decay）
- `eta_t`：动量系数（surprise decay）
- `theta_t`：学习率（momentary surprise scale）

### 三种变体性能排名

**结论：MAL > MAC > MAG**

| 变体 | 全称 | 机制 | BABILong 2M | NeedleHaystack 16K |
|------|------|------|------------|-------------------|
| MAL | Memory as Layer | Memory → Attn → FFN（串行）| 最高 | 80%+ |
| MAC | Memory as Context | 拼接 memory 后做 Attention | 98.7% | 80%+ |
| MAG | Memory as Gate | Attn ∥ Memory，element-wise gate | 较低 | 较低 |
| 对比 | GPT-4 | — | ~88% | ~10%（跌落）|

### 模型配置参数（论文 Table）

| 参数 | 小模型 | 中模型 | 大模型 |
|------|-------|-------|-------|
| L_M（memory 层数）| 1 | 2 | 4 |
| N_p（persistent tokens）| 4 | 16 | 64 |
| window_size | 256 | 512 | 1024 |

**注**：论文的速度对比（MAG 20× faster than MAC）基于 L_M=1 的配置。

---

## 后续工作

### Titans Revisited (arXiv 2510.09551，2025-10)

**批判性复现报告**，主要发现：

1. **块大小被低估**：chunk_size / window_size 对最终性能的影响比原论文描述的大。
   大 chunk → 更好效果，但计算成本更高。
2. **优势幅度更小**：神经记忆确实有帮助，但在控制 chunk size 后，领先幅度比原论文小。
3. **实现细节敏感**：超参（memory_lr, memory_momentum）对结果影响显著。
4. **关键结论**（对本项目最重要）：

   > "memory updates alone are insufficient for meaningful test-time learning, possibly due to a mismatch between frozen backbone input projections and how memory evolves. Without joint adaptation, new information integration is limited."

   即 NLM 单独更新不够，必须与 backbone 联合适配——直接印证 LoRA 的必要性。

### MIRAS：统一理论框架（Google Research，2025-04 提出，2025-12 正式发布）

**论文**：It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization

**核心主张**：Transformer、Mamba、RetNet、DeltaNet 等所有序列模型本质上都在做**关联记忆优化**（Associative Memory Optimization）。

MIRAS 把任何序列模型分解为四个正交设计维度：

| 维度 | 含义 | 现有序列模型的对应 |
|------|------|-----------------|
| Memory Architecture | 记忆数据结构 | 向量（RNN）/ 矩阵（线性 Attn）/ 深度 MLP（Titans）|
| Attentional Bias | 目标函数 | L2（Titans）/ Dot-Product（Transformer）/ Huber / KL |
| Retention Gate | 遗忘机制 | weight decay / 正则化 / 遗忘门 |
| Memory Algorithm | 更新算法 | 梯度下降 / 闭合解 / 动量 |

**三个 MIRAS 变体**（均优于原始 Titans 基线）：

| 变体 | 目标函数 | 优势 |
|------|---------|------|
| YAAD | Huber Loss | 对异常值/噪声鲁棒，脏数据场景更优 |
| MONETA | 广义 Lp-norm | 更强记忆稳定性，长序列优势 |
| MEMORA | KL 散度 | 记忆状态保持概率分布，更受控 |

**对本项目的含义**：当前 NLM 用 MSE（L2）loss。YAAD（Huber）可能对异常 hidden state 更鲁棒，是未来改进候选。

### TPTT: Transforming Pretrained Transformers into Titans (arXiv 2506.17671，2025-06)

与本项目路线最接近的公开工作：

**方法**：
1. 加载预训练 Transformer（Llama-1B、Qwen2.5-1.5B、Mistral-7B 等）
2. 在每个 TransformerBlock 中插入 NeuralLongTermMemory（MAL 或 MAG 风格）
3. 冻结主干，只训练：memory 层（完整）+ attention/FFN 的 LoRA（rank=64, alpha=128）
4. 注入位置：q/k/v/o proj + gate_proj/down_proj

**结果**：Llama-1B Exact Match 提升 20%；效果接近从头训练，成本低 ~100×。

**与本项目差异**：
- TPTT 用 MAG（门控并行），本项目用 MAL（串行，前置 NLM）
- TPTT 无三阶段课程，直接在长文档任务上微调
- TPTT 已发布 pip 包：`pip install tptt`

### TTT / LaCT (arXiv 2407.04620 + 后续)

**TTT**：隐状态本身是一个 ML 模型（同类思路的独立工作）。
**LaCT**：改进 TTT 的块级并行化，大幅提升硬件利用率。

对本项目的意义：LaCT 的并行化思路可以解决串行 memory update 瓶颈，
但需要大规模重构，短期不在计划内。

### End-to-End TTT for Long Context (arXiv 2512.23675，2025-12)

**竞争方向**：训练时也做 meta-learning（不只是 test-time），使模型天生善于 test-time 更新。

**核心结果**：
- 3B 模型，128K context 下 **2.7× 快于全注意力**
- 性能随上下文长度扩展（Mamba2 / Gated DeltaNet 不能）
- E2E 端到端：训练时 meta-learning，推理时 next-token prediction 驱动更新

**竞争威胁评估**：若 E2E TTT 扩展到 instruction-following，"注入 NLM + LoRA"方案需重新评估性价比。当前本项目的差异化在于 Memory Oracle 任务设定（记忆质量，非语言建模），短期不受直接冲击。
