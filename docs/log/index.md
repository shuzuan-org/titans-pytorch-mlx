# 日志索引

每行一天，一句话结论。详细事件 → 对应日期目录。
**每次新增日志条目时必须同步更新此文件。**

| 日期 | 结论 |
|------|------|
| [2026-03-03](2026-03-03/) | 7B MAG 训练停止，转 TPTT；实现 tptt.py + tptt_train.py + tptt_babilong.py；smoke test 通过 |
| [2026-03-04](2026-03-04/) | Memory Oracle 三阶段训练完成；修复 5 个推理 bug（含 stop-token），NLM 检索贡献 Δ+0.49 F1（0.18→0.67），200样本 held-out 评估确认 |
| [2026-03-06](2026-03-06/) | Qwen3.5 兼容验证完成；transformers 5.3.0 升级；9B Stage1 训练 step_2500 完成；评测 F1≈0（thinking 模式干扰）；方向升级为终身个人记忆模型；重构 MemoryOracle；0.8B 7-GPU DDP 训练启动；修复 DDP 三处 bug（all_reduce 顺序、set_epoch、checkpoint 加载） |
| [2026-03-07](2026-03-07/) | 0.8B Oracle 多语言训练完成，LoCoMo F1=0.896（baseline 0.427，Δ+0.469）；Stage3 数据重设计（50-120轮、多实体并行、无时间标签）；EN stage2 数据意外删除（rm 未确认）；train_multilang.py 控制器崩溃（wait_for_data 语言级检查缺失） |
| [2026-03-08](2026-03-08/) | NLM 系统性诊断：KV cache 贡献 100%，NLM weights 贡献 0%（三路对比 + 语义测试 + 权重诊断）；4层叠加失效根因确认（init_state 1e-6、梯度 4e-5、batch非per-token、meta-params未训练）；架构方向待决（修NLM vs 外挂KV store） |
