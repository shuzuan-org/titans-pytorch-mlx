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

### Titans Revisited (arXiv 2510.09551)

**批判性复现报告**，主要发现：

1. **块大小被低估**：chunk_size / window_size 对最终性能的影响比原论文描述的大。
   大 chunk → 更好效果，但计算成本更高。
2. **优势幅度更小**：神经记忆确实有帮助，但在控制 chunk size 后，领先幅度比原论文小。
3. **实现细节敏感**：超参（memory_lr, memory_momentum）对结果影响显著。

### TPTT: Test-time Training for Pre-trained Transformers (arXiv 2506.17671)

**关键思路**：用 LoRA 把预训练好的 vanilla Transformer 改造成 Titans 架构。

**方法**：
1. 加载预训练 Transformer（如 LLaMA, Mistral 等）
2. 在每个 TransformerBlock 中插入 NeuralLongTermMemory（MAL 或 MAG 风格）
3. 冻结主干大部分参数，只训练：memory 层（完整训练）+ attention/FFN 的 LoRA
4. 在长文档任务上继续训练 1K-5K steps

**结果**：效果与从头训练相当，成本低 ~100×。

**原论文 LoRA 配置**：rank=64, alpha=128，注入 q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj

### TTT / LaCT (arXiv 2407.04620 + 后续)

**TTT**：隐状态本身是一个 ML 模型（同类思路的独立工作）。
**LaCT**：改进 TTT 的块级并行化，大幅提升硬件利用率。

对本项目的意义：LaCT 的并行化思路可以解决 B3 瓶颈（串行 memory update），
但需要大规模重构，短期不在计划内。

### MIRAS（Google DeepMind 官方后续）

Titans 的官方改进框架，产出 YAAD、MONETA、MEMORA 等变体。
细节见官方论文（尚未检索完整 arXiv 号）。
