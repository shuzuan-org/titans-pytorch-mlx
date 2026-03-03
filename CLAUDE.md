# Titans PyTorch/MLX 项目

> **L2 职责**：操作手册——命令、配置规格、checklist。团队共享，git-tracked。
> 架构直觉和决策历史 → L1（私有 memory/）。完整参考文档 → L3（docs/）。

---

## 项目目标

**训练一个把记忆刻进权重的模型，取代 RAG 体系。**

RAG 的本质缺陷：检索是外挂的，知识在模型外部，推理时需要两步（检索 + 生成），
上下文窗口有上限，且检索质量直接决定生成质量。

本项目的目标：用 Titans 的神经记忆机制，让模型在推理时通过梯度更新把新信息
直接写入自身权重（NeuralLongTermMemory），实现：

- **无检索**：知识在权重里，不需要向量库
- **在线更新**：推理过程中持续学习，不需要重新训练
- **无限上下文**：记忆容量不受 context window 限制，通过遗忘机制自动管理
- **可组合**：memory 模块可注入任意预训练 Transformer（TPTT 路线）

**当前路线**：在 Qwen2.5-7B 上注入 NeuralLongTermMemory（TPTT），
验证模型在 BABILong 等长文档任务上显著超过 RAG baseline。

---

## 训练服务器（H800 ST）

| 项目 | 值 |
|------|-----|
| IP | 111.6.70.85 |
| SSH 端口 | 101 |
| 用户 | shuzuan |
| 密码 | Free2024 |
| 主机名 | xc2-ubuntu1 |
| GPU | 8× NVIDIA H800 (80GB HBM，共 640GB) |
| CUDA | 12.8 |
| 内存 | 1.5TB |
| 磁盘 | 2.0TB `/dev/sda2`（已用 651GB） |
| Python | 3.10.12 |

```bash
# 接入命令（网络经腾讯云中转）
ssh -p 101 -o ServerAliveInterval=60 -o ServerAliveCountMax=5 shuzuan@111.6.70.85
```

---

## 包管理

```bash
uv run python          # 运行 Python（始终用 uv，不直接用 python）
pip install titans[train]  # H800 上安装训练依赖（含 accelerate）
```

---

## 数据流水线

```bash
# Step 1: Wikipedia XML bz2 → JSONL
python scripts/convert_wiki.py \
  --input /data/wiki/zhwiki-latest-pages-articles.xml.bz2 \
  --output /data/wiki/zh_wiki.jsonl

# Step 2: JSONL → tokenized .bin shards（Qwen2.5 tokenizer，vocab=152064）
python scripts/pretokenize_local.py \
  --sources /data/wiki/zh_wiki.jsonl /data/other/corpus.jsonl \
  --output /data/tokens/ \
  --tokenizer Qwen/Qwen2.5-7B \
  --shard-size 100000000 \
  --min-chars 100

# Step 3: 启动训练
bash scripts/run_pretrain_7b.sh
```

---

## 训练命令

```bash
# 7B MAG 预训练（8× H800，FSDP）
accelerate launch \
  --config_file configs/fsdp_7b.yaml \
  scripts/pretrain_distributed.py \
  --model mag \
  --config configs/model_7b.yaml \
  --data /data/tokens/ \
  --mix-weights configs/mix_weights_7b.yaml \
  --output checkpoints/mag_7b

# 单 GPU 调试（小模型）
uv run python scripts/pretrain_distributed.py \
  --model mag --dim 256 --num-layers 4 --batch-size 2 --max-steps 100
```

---

## 7B 模型规格（TitansMAG）

| 参数 | 值 |
|------|-----|
| dim | 3584 |
| num_heads | 28 |
| num_layers | 24 |
| vocab_size | 152,064（Qwen2.5）|
| seq_len | 8192 |
| ffn_mult | 3.5（≈ Qwen2.5 比例）|
| num_memory_layers | 2（瓶颈根因，见 L1）|
| memory_hidden_mult | 4.0（hidden=14336）|
| window_size | 512 |
| num_persistent_tokens | 16 |

**训练超参**：
| 参数 | 值 |
|------|-----|
| max_lr | 3e-4 |
| min_lr | 3e-5 |
| weight_decay | 0.1 |
| warmup_steps | 2000 |
| stable_steps | 65,500 |
| decay_steps | 7,500 |
| total_steps | 75,000 |
| grad_accum | 32 |
| batch/GPU | 2 |
| effective_batch | 4.2M tokens/step |

---

## TPTT 路线（当前方向）

在 Qwen2.5-7B 上注入 NeuralLongTermMemory，MAL 风格，LoRA 微调。

**注入位置**：每个 Qwen2.5 `Qwen2DecoderLayer` 内，attention 层前插入 memory 层
```
原：x → Attn → FFN
新：x → NeuralMemory(num_memory_layers=1) → Attn → FFN
```

**训练配置**：
| 参数 | 值 |
|------|-----|
| 基模 | `Qwen/Qwen2.5-7B` |
| LoRA rank | 64（作用于 Attn + FFN） |
| memory 层 | 完整训练，num_memory_layers=1 |
| 目标任务 | BABILong 风格长文档问答 |
| 数据 | 现有 tokenized .bin（无需重新 tokenize）|

---

## 性能速查

| 配置 | 步速（H800×8）| 备注 |
|------|------------|------|
| 7B MAG + 2-layer memory | ~166s/step | 已测，autograd 瓶颈 |
| 7B MAG + 1-layer memory | ~40s（推测）| 未测，解析梯度 |
| 7B MAL + 1-layer memory | ~45s（推测）| 未测 |
| TPTT LoRA 微调 | ~5s（推测）| 未测，大部分参数冻结 |

验证脚本（H800 上跑）：
```bash
uv run python -c "
import torch, time
from titans import TitansConfig, TitansMAG
cfg = TitansConfig(dim=512, num_heads=8, num_layers=2, num_memory_layers=2)
m = TitansMAG(cfg).cuda().bfloat16()
x = torch.randint(0, 1000, (2, 512)).cuda()
for _ in range(3): m(x)
torch.cuda.synchronize()
t = time.time()
for _ in range(10): m(x)
torch.cuda.synchronize()
print(f'2-layer: {(time.time()-t)/10:.3f}s/step')
"
```

---

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `src/titans/memory.py` | NeuralLongTermMemory, MemoryMLP（`_compute_gradients` L324）|
| `src/titans/models.py` | MAGBlock(L330), MACBlock, MALBlock |
| `src/titans/attention.py` | SlidingWindowAttention, SegmentedAttention |
| `src/titans/config.py` | TitansConfig dataclass |
| `src/titans/data.py` | BinaryTokenDataset, WeightedMixDataset |
| `src/titans/scheduler.py` | WSD scheduler |
| `scripts/pretrain_distributed.py` | 分布式训练主脚本 |
| `scripts/pretokenize_local.py` | tokenize 流水线 |
| `scripts/convert_wiki.py` | Wiki XML → JSONL |
| `configs/fsdp_7b.yaml` | FSDP 配置（wrap=MAGBlock, bf16, FULL_SHARD）|
| `configs/mix_weights_7b.yaml` | 数据混合权重 |
| `scripts/run_pretrain_7b.sh` | 7B 训练启动脚本 |

---

## 参考文档（L3）

| 文档 | 路径 |
|------|------|
| **日志索引** | `docs/log/index.md` |
| Titans 论文摘要 + 后续工作 | `docs/titans-paper.md` |
| Qwen2.5 / Qwen3.5 架构规格 | `docs/model-specs.md` |
| 代码实现细节（流程图、数据结构）| `docs/implementation-notes.md` |

### 日志使用方式

有值得记录的事件时：
1. 在 `docs/log/YYYY-MM-DD/` 下新建一个文件，文件名描述事件（kebab-case）
2. **必须**在 `docs/log/index.md` 增加或更新当天那行的结论

日志只记事实（发生了什么、实验数字、发现了什么）。
"为什么这样决定" → L1 `decisions.md`。"有什么性能问题" → L1 `bottlenecks.md`。
