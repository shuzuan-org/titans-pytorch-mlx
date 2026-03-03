# 7B MAG 训练停止：166s/step 根因分析

## 现象

8×H800 跑 7B MAG，单步耗时 ~166 秒。
已尝试 batch=2 + 手动梯度检查点重启，仍无法接受。

## 根因

`memory.py:_compute_gradients()`（L324-351）：

每步对每个 MAGBlock 运行一次 `torch.autograd.grad()`，
穿透 MemoryMLP：`Linear(3584→14336) → SiLU → Linear(14336→3584)`。

叠加因素：
- `num_memory_layers=2`：MLP 两层，autograd 图更深
- MAG 全序列处理：8192 个 (k,v) 对一次算梯度（MAC 是 512/chunk）
- 24 层串行，无法并行

结果：24 × autograd_through_14336dim_MLP = 瓶颈核心。

## 如果切换 num_memory_layers=1

线性 memory：`loss = ‖Wk - v‖²`，解析梯度 `dL/dW = 2(Wk-v)kᵀ`，无需 autograd。
预估加速 3-5×，约 40s/step，75K steps ≈ 35 天。

## 结论

从头训 7B 在当前架构配置下不可行。转向 TPTT 路线。
