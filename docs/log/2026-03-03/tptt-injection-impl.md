# TPTT 注入层 + 微调脚本实现

## 完成内容

三个文件：

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/titans/tptt.py` | 131 | 注入核心：QwenLayerWithMemory, inject_memory_into_qwen, reset_memory_states, get_trainable_params |
| `scripts/tptt_train.py` | 299 | 完整微调训练脚本（LoRA + memory，AdamW + cosine warmup）|
| `scripts/tptt_babilong.py` | ~280 | BABILong 评测（baseline vs TPTT，支持 checkpoint 加载）|
| `pyproject.toml` | +1 | train extras 加 `peft>=0.9.0` |

## 关键设计选择

**memory state 存在 module 属性**：`QwenLayerWithMemory.memory_state: MemoryState | None`。
每 batch 前调用 `reset_memory_states(model)`，推理每个新 document 前也需调用。
不用 `register_buffer` 是因为 state 是动态的、batch-dependent，不适合 buffer 语义。

**MemoryMLP 权重排除在 optimizer 之外**：
`get_trainable_params` 过滤 `.memory.memory.layers.*`，这些权重由 Titans 前向机制更新
（`_compute_gradients` + `set_weights`），放进 optimizer 只是浪费，不带来梯度信息。

**`use_conv=False`**：注入时关闭卷积。conv 参数增加显存，且 Qwen 自身有 RoPE 和 attention 的位置感知，无需额外 conv。

## 验证结果（本地 CPU，Qwen2.5-0.5B）

| # | 测试 | 结果 |
|---|------|------|
| 1 | inject + forward（真实 Qwen2.5-0.5B，无 peft）| PASS |
| 2 | inject + peft LoRA + forward | PASS |
| 3 | 数据加载 + memory + 训练循环 3 步（stub 模型）| PASS |
| 4 | tptt_train.py 端到端 3 步（真实 Qwen2.5-0.5B，fake shard）| PASS，3s 完成 |
| 5 | checkpoint save + peft load + forward | PASS |
| 6 | evaluate() 逻辑（tptt_babilong.py，无 BABILong 数据）| PASS |

## 修复的 Bug

**transformers 5.x API 变更**：`torch_dtype=` 改为 `dtype=`。
三个脚本均已修正：`tptt_train.py`, `tptt_babilong.py`, `qttt_babilong.py`。

**QwenLayerWithMemory.__getattr__ 两个坑**：
1. transformers 5.x Qwen2 forward 会访问 `decoder_layer.attention_type`，
   wrapper 没有这个属性 → 需加 `__getattr__` 代理到 `original_layer`
2. `nn.Module` 把子模块存在 `_modules` 而非 `__dict__`，
   `object.__getattribute__(self, "original_layer")` 会 fail →
   必须用 `self.__dict__.get("_modules", {}).get("original_layer")` 取 original_layer

**可训练参数**（Qwen2.5-0.5B + LoRA r=4）：
137M / 650M = 21%（memory 结构参数 + LoRA adapters）

## 下一步（H800 上）

```bash
# 安装 peft
pip install peft>=0.9.0

# Qwen2.5-7B 3 步训练验证
uv run python scripts/tptt_train.py \
  --model Qwen/Qwen2.5-7B \
  --data /data/tokens/ \
  --max-steps 20 --batch-size 2 \
  --output checkpoints/tptt_test

# BABILong baseline（确认指标起点）
uv run python scripts/tptt_babilong.py \
  --model Qwen/Qwen2.5-7B \
  --task qa1 --context-length 16k \
  --max-examples 50 --baseline-only
```

## H800 验证结果（2026-03-03）

### 环境
- Python 环境：`/home/shuzuan/miniconda3/envs/sglang`（torch 2.9.1+cu128）
- 模型：`/home/shuzuan/models/Qwen/Qwen2___5-7B`（ModelScope 下载，与 Qwen2.5-7B 等同）
- 数据：`/home/shuzuan/tokens/longwanjuan_zh`（2 shards，1.4B tokens）
- 追加依赖：`peft>=0.9.0`、`bitsandbytes>=0.49.2`（清华镜像）

### 内存配置（单卡 H800 80GB）
| 组件 | 显存 |
|------|------|
| Qwen2.5-7B (bfloat16) | 14.2 GB |
| + NLM×28 (dim=3584, 1层) | 19.5 GB (+5.4GB) |
| + LoRA r=16 | 19.7 GB |
| forward 峰值 (seq=2048) | 45.2 GB |
| 训练峰值（含 8-bit Adam 状态） | ~53 GB |

**结论：单卡 H800 (80GB) 可运行，峰值 53GB 有余量。**

### 步速（grad_accum=8, seq=2048, batch=1）
```
14.2 s/10steps → 1.42 s/step
effective tokens/step = 8 × 2048 = 16384
```

### 修复的 H800 特定问题
1. **OOM（batch=2/seq=4096）**：单卡装不下 7B+NLM+LoRA r=64+激活。改为 batch=1, seq=2048, LoRA r=16, 8-bit Adam。
2. **DataLoader num_workers=2 静默崩溃**：改为 num_workers=0。
3. **日志丢失**：logging.basicConfig 被 transformers 初始化抢占（no-op）。修复：加 `force=True, handlers=[StreamHandler(stdout)]`。

### 生产启动命令
```bash
nohup bash scripts/run_tptt_h800.sh 10000 /home/shuzuan/checkpoints/tptt_main \
  > /home/shuzuan/checkpoints/tptt_main/train.log 2>/home/shuzuan/checkpoints/tptt_main/stderr.log &
```
