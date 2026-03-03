# Qwen3.5 基模适用性调研

## 调研动机

停止从头训后，考虑 TPTT 路线，需要选一个预训练基模注入 Titans memory。

## 关键发现

**Qwen3.5 不是纯 Transformer。**

架构：每 4 层 = 3× Gated DeltaNet + 1× Gated Attention。

Gated DeltaNet 已经做了线性递推 + 在线遗忘（delta rule），
与 Titans NeuralLongTermMemory 功能部分重叠。

此外：
- vocab_size 从 152,064（Qwen2.5）变为 248,320（Qwen3.5）
- 现有所有 tokenized .bin 数据对 Qwen3.5 全部不兼容

## 规格对比

| | Qwen2.5-7B | Qwen3.5-4B |
|--|--|--|
| 架构 | 纯 Transformer | DeltaNet + Attn 混合 |
| vocab | 152,064 | 248,320 |
| 原生上下文 | 32K | 262K |
| 与现有数据兼容 | ✅ | ❌ |
| Titans 注入干净度 | ✅ 清晰 | ⚠️ 重叠 |

## 结论

短期用 Qwen2.5-7B。Qwen3.5 等第一版跑通后再评估。
