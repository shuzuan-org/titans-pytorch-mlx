# SPEC 3.0 — Memory 扩容：余弦写入 + Top-K 稀疏 + 多 Token Prefix

## 背景

当前 `DenseTimelineMemory` 存在两个严重瓶颈：

1. **写入瓶颈** — 16 个槽位，点积匹配 + softmax 全局加权更新，导致：
   - 老槽位向量范数大，持续吸引新写入（马太效应）
   - 所有槽位每次都被污染，无法隔离不同概念

2. **读取瓶颈** — 检索时所有槽位加权求和压缩成 1 个 token，信息损失巨大：
   - 16 × 3584 维的记忆信息 → 1 × 3584 维的 prefix
   - backbone 只看到一个模糊的"记忆摘要"

## 优化目标

- 槽位数从 16 → **8000**（约 55MB @ bf16，可承受）
- 写入：余弦相似度 + Top-K 稀疏更新（K=4）
- 读取：Top-K 检索（K=16）→ 16 个独立 prefix token

---

## 当前数据流（改动前）

```
history_chunk → encode_tokens → pool → write_repr
    ↓
memory.update(state, write_repr):
    key = write_key(write_repr)                    # (B, H)
    scores = state @ key / sqrt(H)                 # (B, S) 点积
    weights = softmax(scores)                      # (B, S) 全局
    value = write_value(write_repr)                # (B, H)
    update = weights * value                       # (B, S, H) 每个槽都被更新
    next_state = state + gate * update

question → encode_tokens → pool → query_repr
    ↓
memory.retrieve(state, query_repr):
    query = query_proj(query_repr)                 # (B, H)
    scores = state @ query / sqrt(H)               # (B, S)
    weights = softmax(scores)                      # (B, S)
    retrieved = sum(weights * state)               # (B, H) 压缩成1个向量
    return output_proj(retrieved)                  # (B, H) → 1个prefix token
```

---

## 新数据流（改动后）

### Write（余弦 + Top-K 稀疏）

```
memory.update(state, write_repr):
    key = write_key(write_repr)                    # (B, H)
    # 余弦相似度替代点积
    state_norm = F.normalize(state, dim=-1)        # (B, S, H)
    key_norm = F.normalize(key, dim=-1)            # (B, H)
    scores = state_norm @ key_norm / temperature   # (B, S)
    # Top-K 稀疏：只更新最匹配的 K 个槽
    topk_scores, topk_indices = scores.topk(top_k_write, dim=-1)  # (B, K)
    weights = softmax(topk_scores)                 # (B, K)
    value = write_value(write_repr)                # (B, H)
    gate = sigmoid(write_gate(write_repr))         # (B, 1)
    # 只在 topk 槽位上做更新，其余严格冻结
    scatter_update(state, topk_indices, weights, value, gate)
    next_state = state  # 只有 K 个槽被改动
```

### Read（Top-K 多 Token 检索）

```
memory.retrieve(state, query_repr):
    query = query_proj(query_repr)                 # (B, H)
    # 同样用余弦
    state_norm = F.normalize(state, dim=-1)
    query_norm = F.normalize(query, dim=-1)
    scores = state_norm @ query_norm / temperature # (B, S)
    # Top-K 检索
    topk_scores, topk_indices = scores.topk(top_k_read, dim=-1)  # (B, K)
    weights = softmax(topk_scores)                 # (B, K)
    # 取出 K 个槽位向量，不做求和
    retrieved_slots = gather(state, topk_indices)  # (B, K, H)
    # 投影到 embedding 空间
    prefix_embeds = output_proj(retrieved_slots)   # (B, K, H)
    return prefix_embeds, weights                  # K 个 prefix token
```

### Forward 拼接

```
# 旧：1 个 prefix token
prefix_embeds = retrieved.unsqueeze(1)             # (B, 1, H)

# 新：K 个 prefix token
prefix_embeds = retrieved_slots                    # (B, K, H) 已经是多token
prefix_attention_mask = ones(B, K)                 # K 个位置
full_embeds = cat([prefix_embeds, token_embeds])   # (B, K+T, H)
full_labels = cat([-100]*K, labels)                # prefix 位置不算 loss
```

---

## 具体改动

### 1. `Stage1ModelConfig` 新增字段

```python
memory_slots: int = 8000          # 原 16
top_k_write: int = 4              # 写入稀疏度
top_k_read: int = 16              # 检索 token 数
memory_temperature: float = 0.1   # 余弦相似度温度
```

`num_retrieved_memory_tokens` 字段当前未使用，可以复用或废弃，建议用 `top_k_read` 替代语义更清晰。

### 2. `DenseTimelineMemory` 重写

```python
class DenseTimelineMemory(nn.Module):
    def __init__(self, hidden_size, memory_slots, top_k_write, top_k_read, temperature):
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.top_k_write = top_k_write
        self.top_k_read = top_k_read
        self.temperature = temperature

        # 初始化：L2 归一化，超球面均匀分布
        raw = torch.randn(memory_slots, hidden_size)
        self.initial_memory = nn.Parameter(F.normalize(raw, p=2, dim=-1))

        # Write 投影
        self.write_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_gate = nn.Linear(hidden_size, 1)

        # Read 投影
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(hidden_size)
```

#### update 方法

```python
def update(self, state, write_repr):
    # state: (B, S, H), write_repr: (B, H)
    key = self.write_key(write_repr)                         # (B, H)
    state_norm = F.normalize(state, dim=-1)
    key_norm = F.normalize(key, dim=-1)
    scores = torch.matmul(state_norm, key_norm.unsqueeze(-1)).squeeze(-1)  # (B, S)
    scores = scores / self.temperature

    topk_scores, topk_idx = scores.topk(self.top_k_write, dim=-1)  # (B, K)
    topk_weights = torch.softmax(topk_scores, dim=-1)               # (B, K)

    value = self.write_value(write_repr)                     # (B, H)
    gate = torch.sigmoid(self.write_gate(write_repr))        # (B, 1)

    # 构造稀疏更新
    update_vecs = topk_weights.unsqueeze(-1) * value.unsqueeze(1)  # (B, K, H)
    update_vecs = gate.unsqueeze(-1) * update_vecs                 # (B, K, H)

    # scatter 到对应槽位
    next_state = state.clone()
    topk_idx_expanded = topk_idx.unsqueeze(-1).expand_as(update_vecs)  # (B, K, H)
    next_state.scatter_add_(1, topk_idx_expanded, update_vecs)

    return next_state, gate.squeeze(-1)
```

#### retrieve 方法

```python
def retrieve(self, state, query_repr):
    # state: (B, S, H), query_repr: (B, H)
    query = self.query_proj(query_repr)                      # (B, H)
    state_norm = F.normalize(state, dim=-1)
    query_norm = F.normalize(query, dim=-1)
    scores = torch.matmul(state_norm, query_norm.unsqueeze(-1)).squeeze(-1)  # (B, S)
    scores = scores / self.temperature

    topk_scores, topk_idx = scores.topk(self.top_k_read, dim=-1)  # (B, K)
    topk_weights = torch.softmax(topk_scores, dim=-1)               # (B, K)

    # 取出 top-K 槽位
    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # (B, K, H)
    retrieved_slots = torch.gather(state, 1, topk_idx_expanded)     # (B, K, H)

    # 按权重加权（保留多 token）
    weighted_slots = topk_weights.unsqueeze(-1) * retrieved_slots   # (B, K, H)

    # 投影 + LayerNorm 对齐 backbone embedding 空间
    prefix_embeds = self.output_norm(self.output_proj(weighted_slots))  # (B, K, H)
    return prefix_embeds, topk_weights
```

### 3. `FrozenBackboneWithTimelineMemory` 适配

#### `__init__`

```python
self.memory = DenseTimelineMemory(
    hidden_size=self.backbone.hidden_size,
    memory_slots=config.memory_slots,
    top_k_write=config.top_k_write,
    top_k_read=config.top_k_read,
    temperature=config.memory_temperature,
)
```

#### `forward` — prefix 从 1 token → K token

```python
# retrieve 返回 (B, K, H) 而不是 (B, H)
retrieved, retrieval_weights = self.memory.retrieve(memory_state, question_repr)

token_embeds = self.backbone.embed_input_ids(input_ids)
prefix_embeds = retrieved.to(dtype=token_embeds.dtype)        # (B, K, H) 已经是多token
prefix_attention_mask = torch.ones(
    (batch_size, self.config.top_k_read),                     # K 个位置
    device=device, dtype=attention_mask.dtype,
)
full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
prefix_labels = torch.full(
    (batch_size, self.config.top_k_read),                     # K 个位置不算loss
    fill_value=-100, device=device, dtype=labels.dtype,
)
full_labels = torch.cat([prefix_labels, labels], dim=1)
```

#### `build_chat_inputs` — 推理路径同步

```python
# 同样改为多 token
prefix_embeds = retrieved.to(dtype=token_embeds.dtype)        # (1, K, H)
prefix_attention_mask = torch.ones(
    (1, self.config.top_k_read),
    device=attention_mask.device, dtype=attention_mask.dtype,
)
```

#### `retrieve_from_state` — 返回值形状变化

```python
# 原：retrieved (1, H) → 新：retrieved (1, K, H)
retrieved, retrieval_weights = self.memory.retrieve(state.memory_state, question_repr)
return retrieved, retrieval_weights, input_ids, attention_mask
```

### 4. `validate_session_state` — memory shape 校验更新

```python
# 原：(1, 16, H)
# 新：(1, 8000, H)
expected_shape = (1, self.config.memory_slots, self.backbone.hidden_size)
```

这个不需要改代码，因为已经用 `config.memory_slots`，自动适配。

### 5. 训练脚本 CLI 参数

```
--memory-slots 8000
--top-k-write 4
--top-k-read 16
--memory-temperature 0.1
```

---

## 参数量估算

Qwen2.5-7B hidden_size = 3584

| 组件 | 计算 | 参数量 |
|------|------|--------|
| initial_memory | 8000 × 3584 | 28.7M |
| write_key | 3584 × 3584 | 12.8M |
| write_value | 3584 × 3584 | 12.8M |
| write_gate | 3584 × 1 | 3.6K |
| query_proj | 3584 × 3584 | 12.8M |
| output_proj | 3584 × 3584 | 12.8M |
| **合计** | | **~80M** |

加上 LoRA ~60M，总可训练参数约 140M，可接受。

---

## 显存估算

- memory state: 8000 × 3584 × 2B (bf16) = **55MB/sample**
- batch_size=1 时 55MB，batch_size=4 时 220MB
- 余弦匹配中间变量：8000 维 scores，可忽略
- 相比 7B 模型本身 ~14GB，增量可控

---

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `src/titans/stage1_models.py` | Config 新增字段 + DenseTimelineMemory 重写 + forward/retrieve/build_chat_inputs 适配多 token prefix |
| `scripts/train_stage1_distributed.py` | 新增 CLI 参数 |
| `scripts/serve_stage1.py` | 新增 CLI 参数 |

---

## 向后兼容

- 旧 checkpoint（16 槽位）无法直接用于新模型（8000 槽位），需要重新训练
- 旧数据集完全兼容，不需要改
- serve API 接口不变，只是内部 prefix 从 1 token 变 16 token

---

## 验证方式

1. `--memory-slots 8000 --top-k-write 4 --top-k-read 16` 启动训练
2. 确认日志打印参数量约 80M（memory） + 60M（LoRA）= 140M
3. 训练 loss 正常下降
4. 推理时确认 retrieval_weights 形状为 (1, 16) 而不是 (1, 16) → 语义不同：旧的是 16 槽位的权重分布，新的是 top-16 的权重
5. 对比旧 16 槽位模型的回答准确率

---

## 已确认

- `memory_temperature` 初始值 **0.3**
- `output_proj` 后加 **LayerNorm** 对齐 backbone embedding 空间
- retrieve 取出的 K 个槽位 **乘以权重** 后再投影

## 后期优化点

- `memory_temperature` 可以改为 `nn.Parameter`（初始值 0.3~0.5），让模型自己学习最优温度
