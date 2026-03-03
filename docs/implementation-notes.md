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

## 可选优化模块

| 模块 | 功能 | 状态 |
|------|------|------|
| `cuda_optimizations.py` | batched_memory_update, compute_memory_gradients_efficient | 导入成功则自动使用，有 silent fallback |
| `triton_kernels.py` | RMSNorm, fused_add_rms_norm, silu_mul | Triton 可用时自动使用 |
| `flash_attention.py` | FlashAttention v2 wrapper | flash_attn 可用时自动使用 |
| `optimized_training.py` | 梯度累积优化 | 主脚本显式调用 |

**注意**：所有优化模块都有 silent fallback，若 import 失败不报错，直接走慢路径。
排查性能问题时，需手动检查 `HAS_*` 标志实际值。
