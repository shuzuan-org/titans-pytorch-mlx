# 决策：转向 TPTT + Qwen2.5-7B

## 方案

在 Qwen2.5-7B 每个 `Qwen2DecoderLayer` 的 attention 前插入 NeuralLongTermMemory（MAL 风格）：

```
原：Attn → FFN
新：NeuralMemory → Attn → FFN
```

- LoRA rank=64，作用于 attention + FFN（主干不全冻结）
- NeuralLongTermMemory 完整训练，`num_memory_layers=1`
- 数据：现有 tokenized .bin 直接复用（Qwen2.5 tokenizer 匹配）
- 评估：BABILong 长文档问答，对比无 memory 的 Qwen2.5-7B baseline

## 选型理由

| 方案 | 预计耗时 | 可行性 |
|------|---------|-------|
| 7B MAG 从头训（当前配置）| 145 天 | ❌ |
| 7B MAL 从头训（1层 memory）| 35 天 | 勉强 |
| TPTT LoRA 微调 Qwen2.5-7B | 1-2 天 | ✅ |

TPTT 论文（arXiv 2506.17671）验证：LoRA 注入效果与从头训相当。

## 尚未决定

- 是否每层都注入，还是每隔 N 层注入一次
- LoRA 具体超参（rank/alpha）
- 微调数据配比（长文档 vs 通用语料）
