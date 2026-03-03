# 基模架构规格（L3 参考文档）

> 具体数值参考，按需查阅。选型决策 → L1/decisions.md。

---

## Qwen2.5-7B（当前 TPTT 基模）

**HuggingFace**: `Qwen/Qwen2.5-7B`
**架构**：纯 Transformer（Decoder-only）
**许可**：Apache 2.0

| 参数 | 值 |
|------|-----|
| hidden_size | 3584 |
| num_hidden_layers | 28 |
| num_attention_heads | 28（Q）|
| num_key_value_heads | 4（KV，GQA）|
| head_dim | 128 |
| intermediate_size | 18944 |
| vocab_size | 152,064 |
| max_position_embeddings | 32,768 |
| rope_theta | 1,000,000 |
| FFN | SwiGLU（无 bias）|
| Norm | RMSNorm（pre-norm）|
| 参数总量 | ~7.6B |

**TPTT 注入点**：每个 `Qwen2DecoderLayer` 的 `self_attn` 前
（HuggingFace 代码：`modeling_qwen2.py`，`Qwen2DecoderLayer.forward()`）

**与当前 Titans 实现的维度对应**：
```
TitansConfig.dim = 3584          # = Qwen2.5-7B hidden_size
TitansConfig.num_heads = 28      # = num_attention_heads
TitansConfig.vocab_size = 152064 # = Qwen2.5-7B vocab_size
```

---

## Qwen3.5-4B（备选，当前不用）

**HuggingFace**: `Qwen/Qwen3.5-4B`
**架构**：混合（Gated DeltaNet + Gated Attention 交替）
**许可**：Apache 2.0

| 参数 | 值 |
|------|-----|
| hidden_size | 2560 |
| num_hidden_layers | 32（3×DeltaNet + 1×Attn 循环）|
| vocab_size | 248,320 |
| max_position_embeddings | 262,144 |
| rope_theta | 10,000,000 |
| head_dim | 256（Attn 层）|

**不选此模型的原因**：
1. vocab 不兼容（152K vs 248K），现有数据全部需要重新 tokenize
2. DeltaNet 层与 Titans memory 功能重叠，实验结论不干净

---

## Qwen3.5-9B（备选，扩大规模时考虑）

**HuggingFace**: `Qwen/Qwen3.5-9B`

| 参数 | 值 |
|------|-----|
| hidden_size | 3584 |
| num_hidden_layers | 40+（DeltaNet+Attn 混合）|
| vocab_size | 248,320 |
| max_position_embeddings | 262,144 |

**注**：hidden_size 与 Qwen2.5-7B 相同（3584），但层数更多，且混合架构。

---

## Tokenizer 兼容性矩阵

| Tokenizer | vocab_size | 兼容模型 |
|-----------|-----------|---------|
| Qwen2.5 | 152,064 | Qwen2.5 系列，现有 .bin 数据 |
| Qwen3.5 | 248,320 | Qwen3.5 系列（**与现有数据不兼容**）|

**注意**：使用 Qwen3.5 作为基模时，必须用 Qwen3.5 tokenizer 重新处理全部语料。
