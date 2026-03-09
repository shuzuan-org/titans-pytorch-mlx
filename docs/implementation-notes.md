# 代码实现细节（L3 参考文档）

> 代码结构和数据流的完整描述，按需查阅。
> 性能瓶颈分析 → L1/bottlenecks.md。

---

## NeuralLongTermMemory 完整数据流

```
输入 x: (B, S, D)
  │
  ├─ proj_k(x) → conv(可选) → SiLU → L2_normalize → keys  (B, S, D)
  ├─ proj_v(x) → conv(可选) → SiLU → L2_normalize → values (B, S, D)
  └─ proj_q(x) → conv(可选) → SiLU → L2_normalize → queries (B, S, D)
  │
  ├─ [Retrieve] memory.forward(queries) → retrieved (B, S, D)
  │
  ├─ [Gate compute]
  │   gate_decay(x) → alpha  (B, S, D)
  │   gate_lr(x) * memory_lr → theta (B, S, D)
  │   gate_momentum(x) * memory_momentum → eta (B, S, D)
  │   alpha_s, theta_s, eta_s = alpha.mean(), theta.mean(), eta.mean()  ← 标量化
  │
  ├─ [Gradient] _compute_gradients(keys, values) → grads: list[Tensor]
  │   ├─ 优先：HAS_CUDA_OPTIMIZATIONS → compute_memory_gradients_efficient()
  │   └─ fallback：torch.enable_grad() → memory.compute_loss() → autograd.grad()
  │
  ├─ [Update]
  │   new_momentum = eta_s * momentum - theta_s * grads
  │   new_weights = (1 - alpha_s) * weights + new_momentum
  │
  └─ proj_out(retrieved) → output (B, S, D)
     new_state = MemoryState(new_weights, new_momentum).detach()

返回: (output, new_state)
```

---

## MemoryMLP 结构

```python
# num_memory_layers=1（线性 memory）
Linear(D, D, bias=False)   # W ∈ R^{D×D}

# num_memory_layers=2（深层 memory，当前 7B 配置）
Linear(D, H, bias=False)   # H = D * memory_hidden_mult = 14336（7B时）
SiLU()
Linear(H, D, bias=False)

# num_memory_layers=N（通用）
Linear(D, H) → SiLU → [Linear(H,H) → SiLU] × (N-2) → Linear(H, D)
```

**初始化**：`nn.init.normal_(weight, std=config.init_std)`（init_std=0.02）

**线性 memory 的解析梯度**（可用于绕过 autograd）：
```
loss = ||Wk - v||² / N
dL/dW = 2/N * (Wk - v) ⊗ k   # outer product
```
当前代码未实现此优化，仍走 autograd 路径。

---

## MemoryState 数据结构

```python
@dataclass
class MemoryState:
    weights: list[torch.Tensor]   # len = num_memory_layers，每个 shape = [out, in]
    momentum: list[torch.Tensor]  # shape 同 weights，初始化为 zeros
```

**初始化**：`init_state()` 从 `memory.get_weights()` clone，不是 zero init。
这意味着初始 memory state 包含模型的训练权重，而非空白状态。

**跨 chunk/segment 传递**：
- TitansMAC：通过 `chunk` 循环显式传递 states
- TitansMAG/MAL：全序列一次性处理，无 chunk 循环，state 只在 batch 间传递
- 每步 `detach()` 截断梯度图，state 不参与主训练的反传

---

## 三种变体的 Block 结构

### MAGBlock（当前训练目标）

```
x → norm1 → [Attention（checkpoint）] → +residual → y_t
x → norm1 → [NeuralMemory（不 checkpoint）] → mem_out, new_state
output = y_t * mem_out            # Eq.28: element-wise gate
output → norm2 → [FFN（checkpoint）] → +residual
```

### MACBlock

```
x → memory.retrieve(x, state) → memory_tokens  # Eq.21
x → norm1 → Attention([persistent || memory_tokens || normed_x]) → +residual → y_t
y_t → memory.forward(y_t, state) → _, new_state  # Eq.24 更新 memory
y_t * memory.retrieve(y_t, new_state) → output    # Eq.25 最终输出
output → norm2 → FFN → +residual
```

### MALBlock

```
x → norm1 → NeuralMemory → +residual
x → norm2 → Attention(prefix=persistent) → +residual
x → norm3 → FFN → +residual
```

---

## FSDP 配置说明

`configs/fsdp_7b.yaml` 关键字段：
```yaml
fsdp_transformer_layer_cls_to_wrap: MAGBlock  # wrap 单元
fsdp_sharding_strategy: FULL_SHARD            # ZeRO-3 等价
fsdp_activation_checkpointing: false          # 禁用 FSDP 层面的 checkpoint
                                               # （MAGBlock 内部自己管理 checkpoint）
fsdp_use_orig_params: true                    # 兼容 LoRA 和 optimizer state
mixed_precision: bf16
```

**为什么 `fsdp_activation_checkpointing: false`**：
MAGBlock 自己对 attention + FFN 做了 `torch.utils.checkpoint`，
memory 不能 checkpoint（有状态），所以不能让 FSDP 统一管理 activation checkpoint。

---

## 数据格式

**BinaryTokenDataset**：
- 文件格式：`.bin`，uint32 little-endian，每个 int 是一个 token id
- 索引文件：`.idx`，记录每个 shard 的偏移
- 读取方式：mmap，零拷贝

**WeightedMixDataset**：
- 多数据源加权混合
- seed = base_seed + rank（保证各 GPU 数据独立）
- 支持 IterableDataset 接口

---

## NLM 工作原理：为什么能"记住"

### 写入：推理时梯度下降（test-time learning）

NLM 的核心创新：**MemoryMLP 的权重本身就是记忆载体**，推理时通过梯度更新改写权重。

每处理一段输入 token，NLM 执行：
```
1. x → proj_k → keys,  x → proj_v → values
2. 计算关联记忆损失：loss = ||M(k) - v||²
   （"如果输入 k，你能输出 v 吗？"）
3. 对 MemoryMLP 权重求梯度：grads = ∂loss/∂W
4. 带动量的权重更新：
   S_t = η·S_{t-1} - θ·grads    ← momentum
   M_t = (1-α)·M_{t-1} + S_t   ← weight decay（遗忘）
```

关键：这发生在 `torch.enable_grad()` 上下文中，**与主训练的反传完全独立**。
MemoryMLP 权重通过 `set_weights()` 直接改 `.data`，不进 optimizer。

### 读取：正向传播

```
q = proj_q(x) → SiLU → L2_normalize
output = proj_out(MemoryMLP(q))
```

MemoryMLP 的权重里已经存着之前写入的 k→v 关联，输入语义相近的 q 能召回对应的 v。

### 三个门控：数据依赖的学习率

```python
alpha = gate_decay(x)    # α：遗忘率 — 此刻忘多少旧的
theta = gate_lr(x)       # θ：学习率 — 此 token 写入深度
eta   = gate_momentum(x) # η：动量系数 — 延续多少上次"惊讶"
```

每个 token 自己决定写入强度，不是全局固定超参。
**当前实现简化**：将 (B, S, D) 的门控值 `.mean(dim=(0,1))` 聚合为 (D,) 的 per-dim 标量，
论文是 per-token per-dim 的顺序更新（每个 token 独立 S_t），理论上表达能力更强。

### 为什么还需要 LoRA

| 组件 | 更新时机 | 作用 |
|------|---------|------|
| MemoryMLP 权重 | 推理时（每个 chunk）| 存储/检索信息 |
| LoRA（Attn + FFN）| 训练时（监督学习）| 学会使用 NLM 输出 |
| Qwen3.5 base | 冻结 | 语言理解/生成能力 |

NLM 插在每个 decoder layer 前（MAL 风格）：
```
x → NLM → x + mem_out → Attention → FFN
```

原始 Qwen3.5 从未见过 `mem_out` 信号。没有 LoRA，Attention/FFN 无法辨别 NLM 输出中
有用的信息，会把它当噪声忽略。Titans Revisited 独立验证了这一点（2025-10）：

> "Without joint adaptation, new information integration is limited."

---

## NLM 有效性实测结论（2026-03-08）

### 实验：oracle.read() recall 来源分离

三路对比（固定 KV cache 内容，改变 NLM state）：

| 条件 | 正确数 | 解释 |
|------|-------|------|
| KV cache + NLM updated | 2/5 | baseline |
| KV cache only（NLM reset）| 2/5 | NLM 贡献 = 0 |
| 无 KV cache，NLM updated | 0/5 | NLM 单独 = 0 |

**结论**：当前 recall 完全来自 KV cache，NLM weights 贡献为零。

### 根因：4 层叠加失效

1. **init_state std=1e-6 → NLM 输出量级 ~1e-6，hidden state ~0.1**
   差 5 个数量级，训练期间模型"看不见"NLM，直接学会绕开。

2. **梯度极小（mean ~4e-5）**
   k 经 L2_normalize 每元素 ~0.014；MSE mean reduction 除以 N×D → 梯度极小；
   momentum 不足以积累到可见的权重变化。

3. **Batch 更新 ≠ 论文 per-token 顺序更新**
   整序列一次 batch step，decay 项 (1-α) 主导，权重反而**缩小至初始值的 0.28×**。

4. **门控 meta-params 从未训练**
   `gate_lr / gate_decay / gate_momentum` 的 `requires_grad=False`（固定在 sigmoid(0)=0.5）；
   学习率、遗忘率、动量均无法根据任务自适应。

### 可能出路

| 方向 | 代价 | 状态 |
|------|------|------|
| per-token 顺序更新 + 解冻 meta-params + init std↑ | 需重设计训练流程，NLM 循环慢 N× | 待评估 |
| `create_graph=True`（MAML 风格）使 meta-params 可微 | 3-5× 内存 | 待评估 |
| 改用外挂 KV store（接受结构性限制） | 放弃权重内化目标 | 待决策 |

---

## 可选优化模块

| 模块 | 功能 | 状态 |
|------|------|------|
| `cuda_optimizations.py` | batched_memory_update, compute_memory_gradients_efficient | 导入成功则自动使用，有 silent fallback |
| `triton_kernels.py` | RMSNorm, fused_add_rms_norm, silu_mul | Triton 可用时自动使用 |
| `flash_attention.py` | FlashAttention v2 wrapper | flash_attn 可用时自动使用 |
| `optimized_training.py` | 梯度累积优化 | 主脚本显式调用 |

**注意**：所有优化模块都有 silent fallback，若 import 失败不报错，直接走慢路径。
排查性能问题时，需手动检查 `HAS_*` 标志实际值。
