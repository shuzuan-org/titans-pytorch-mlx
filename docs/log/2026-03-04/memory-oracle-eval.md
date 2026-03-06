# Memory Oracle 训练+评估 — 2026-03-04

## 结论

Memory Oracle（Qwen2.5-7B + NLM×28 + LoRA r=16）完成三阶段训练，评估验证 NLM 记忆检索有效。

## 数据（最终，含 stop-token 修复）

| 指标 | OUT-DIST 12样本 | OUT-DIST 200样本（最终） |
|------|----------------|------------------------|
| F1 with NLM writes | 0.670 | **0.674** |
| F1 baseline (no writes) | 0.195 | **0.184** |
| **Δ (NLM 贡献)** | **+0.474** | **+0.489** |

- 200 样本评估为 held-out `oracle_eval.jsonl`，统计显著，结果稳定（每 20 步收敛）
- 初始评估 Δ+0.19，修复 stop-token 后提升到 Δ+0.49

- Checkpoint: `checkpoints/oracle_stage3/final/` (peft adapter, 39MB)
- 评估脚本: `scripts/final_eval3.py`
- 数据: 本地合成对话 10K/10K/5K（模板生成，40种变体）

## 踩过的坑

1. **训练/推理格式不一致**：training 用 `【写入】`/`【查询】`/`【记忆】`，旧 oracle API 用完全不同的模板 → 全部输出乱码。修复：`memory_oracle.py` 定义权威常量，training import 之。

2. **LoRA merge_and_unload 破坏模型**：Stage 3 在 40种合成模板上 loss→0（过拟合），merge 把错误权重写入 base model → 乱码。修复：`_load_weights` 改为 `PeftModel.from_pretrained`（不 merge）。

3. **init_state 用非零 MemoryMLP 权重**：random init 权重（std=0.02）导致 baseline（无 write）也有非零 NLM 残差。修复：`init_state` 改为 `torch.zeros_like(w)` 初始化。

4. **分步 write/read vs 全序列推理**：分步 multi-call 模式（write×N 然后 read）与训练的 single-pass 格式不一致（attention context 缺失）→ 乱码。修复：`MemoryOracle.write()` 只 buffer 消息，`read()` 拼完整序列一次 generate()。

5. **output 尾部假对话**：正确摘要后，模型继续生成假对话（无 `【写入/查询/记忆】` 标记）。修复：在 stop-token 列表中加入 `\n\n`，取第一段作为输出。

## 训练信息

- 平台：H800 GPU1（sglang conda env）
- 步速：~1.4s/step
- Stage 1: 5000 步，loss 0.0009→0（过拟合 500步）
- Stage 2: 10000 步，loss≈0
- Stage 3: 3000 步，loss 0.088→0.071（因 write token 也纳入 loss，不那么快过拟合）
