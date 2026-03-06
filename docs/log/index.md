# 日志索引

每行一天，一句话结论。详细事件 → 对应日期目录。
**每次新增日志条目时必须同步更新此文件。**

| 日期 | 结论 |
|------|------|
| [2026-03-03](2026-03-03/) | 7B MAG 训练停止，转 TPTT；实现 tptt.py + tptt_train.py + tptt_babilong.py；smoke test 通过 |
| [2026-03-04](2026-03-04/) | Memory Oracle 三阶段训练完成；修复 5 个推理 bug（含 stop-token），NLM 检索贡献 Δ+0.49 F1（0.18→0.67），200样本 held-out 评估确认 |
| [2026-03-06](2026-03-06/) | Qwen3.5 兼容验证完成；transformers 5.3.0 升级；9B Stage1 训练 step_2500 完成；评测 F1≈0（thinking 模式干扰）；方向升级为终身个人记忆模型；重构 MemoryOracle；0.8B 7-GPU DDP 训练启动；修复 DDP 三处 bug（all_reduce 顺序、set_epoch、checkpoint 加载） |
