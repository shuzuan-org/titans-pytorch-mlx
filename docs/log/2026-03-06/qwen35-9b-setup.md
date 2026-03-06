# Qwen3.5-9B 环境准备 + 方向升级 — 2026-03-06

## 结论

Memory Oracle 基模切换到 Qwen3.5-9B；Qwen3.5 兼容性已验证；方向升级为终身个人记忆模型。
**9B Stage 1 评测（80样本）：F1≈0.000，Δ=0**，原因是 Qwen3.5 thinking 模式干扰；需要 Stage 2/3 训练才能压制 thinking 模式。

## 数据（Qwen2.5-7B oracle，200样本最终结果）

| 指标 | 值 |
|------|-----|
| F1 with NLM writes | 0.674 |
| F1 no-write baseline | 0.184 |
| **Δ (NLM 贡献)** | **+0.489** |

- 评估集：`data/oracle_eval.jsonl`（200 samples，held-out）
- 结果从第 20 个样本平稳收敛，统计可信

## Qwen3.5 兼容工作

**注入代码修改**（`src/titans/qwen35_injection.py`）：
- 新增 `_get_text_backbone()` 辅助函数，自动识别多模态/文本模型结构
- `Qwen3_5ForConditionalGeneration`：层路径 `model.language_model.model.layers`
- `Qwen3_5ForCausalLM`（`AutoModelForCausalLM` 加载方式）：层路径 `model.model.layers`
- hidden_size 从 `config.text_config.hidden_size` 读取（Qwen3.5 层级结构）

**Transformers 升级**：4.57.1 → 5.3.0，peft 0.18.1 兼容正常

**已验证**（H800 CPU 测试）：
- Qwen3.5-0.8B 注入 NLM×24 → forward pass OK，logits shape [1,6,248320]
- Qwen3.5-9B：NLM×32 注入 + LoRA r=16，forward 正常，37GB HBM

## Qwen3.5-9B Stage 1 训练 + 评测结果

**训练（GPU2，2026-03-06）**：
- Stage 1：5000步计划，实际跑到 step_2500（loss 在 step ~200 降至 0 = 过拟合，与 7B 相同）
- 步速：~162s/10steps（16s/step），比 7B 慢约 11倍（9B 参数 × 32层）
- Checkpoint：`checkpoints/oracle_9b/stage1/step_0002500.pt`（12GB）

**评测（80样本后终止）**：

| 模型 | F1(write) | F1(baseline) | Δ |
|------|-----------|-------------|---|
| Qwen2.5-7B Stage3（已知） | 0.674 | 0.184 | **+0.489** |
| Qwen3.5-9B Stage1（本次） | ≈0.000 | ≈0.000 | ≈0.000 |

**根因分析**：
1. **Qwen3.5 thinking 模式**：模型推理时默认生成 `<think>` token（ID=248068），被截断为空字符串
2. `/no_think` 前缀不稳定：部分样本生效，部分样本仍触发 thinking
3. **write 模式额外问题**：输入序列含多条 `【写入】` 条目，更长的 context 更容易触发 thinking
4. Stage 1 仅 2500步（过拟合）= 训练量不够，未能压制 base model 的 thinking 倾向

**下一步**：需要完整 Stage 2/3 训练（各 10000/3000步）来强化格式并压制 thinking 模式。

## 方向升级（2026-03-06）

从 "Memory Oracle Q&A 事实检索" → **终身个人记忆模型**：
- 不再只训练事实检索（F1 指标）
- 目标：跨 session 的个人理解，惊讶度驱动自动重要性过滤
- NLM state 按 user_id 持久化，提供个人化背景给闭源 LLM

**真实空白**（vs 竞品 mem0/MemOS/TTT-E2E）：
- 梯度写入 + 跨 session + 个人化，三者同时具备的方案目前没有
