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

**当前路线**：训练 Qwen3.5-0.8B Memory Oracle，三阶段课程训练，
验证在 LoCoMo 长对话记忆任务上显著超过 RAG/mem0 baseline。

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

## Memory Oracle 数据管线（当前）

### MiniMax API 配置

| 接口风格 | 变量 | 值 |
|---------|------|-----|
| Anthropic（主） | `ANTHROPIC_BASE_URL` | `https://mini.origintask.cn` |
| Anthropic（主） | `ANTHROPIC_AUTH_TOKEN` | `shuzuan2025-minimax` |
| OpenAI 兼容 | `base_url` | `https://mini.origintask.cn/v1` |
| OpenAI 兼容 | `api_key` | `shuzuan2025-minimax` |
| model | — | `MiniMax-Text-01` |

```bash
# 合成训练数据（MiniMax API，3 阶段）
# Stage 1：单 session，直接事实，无噪音
OPENAI_API_KEY=shuzuan2025-minimax python scripts/build_memory_data_v2.py \
  --stage 1 --n-samples 5000 --output data/oracle_stage1_v2.jsonl \
  --backend openai --openai-model MiniMax-Text-01 \
  --openai-base-url https://mini.origintask.cn/v1 \
  --concurrency 128 --lang zh

# Stage 2：多 session，40-60% 噪音，跨轮综合查询
OPENAI_API_KEY=shuzuan2025-minimax python scripts/build_memory_data_v2.py \
  --stage 2 --n-samples 10000 --output data/oracle_stage2_v2.jsonl \
  --backend openai --openai-model MiniMax-Text-01 \
  --openai-base-url https://mini.origintask.cn/v1 \
  --concurrency 128 --lang zh

# Stage 3：同实体多次更新，含 importance 标注
OPENAI_API_KEY=shuzuan2025-minimax python scripts/build_memory_data_v2.py \
  --stage 3 --n-samples 3000 --output data/oracle_stage3_v2.jsonl \
  --backend openai --openai-model MiniMax-Text-01 \
  --openai-base-url https://mini.origintask.cn/v1 \
  --concurrency 128 --lang zh

# 三阶段训练（8× H800，torchrun 启动）
# ⚠️  必须同时加 --multi-gpu，否则 8 进程全堆 GPU 0 → 立即 OOM
# ⚠️  batch_size 不能太小（<4/卡）：batch=2+seq=512 → 矩阵规模 (2,1,1024) 太小，步速 26s！
# Stage1 最佳：batch=16/卡, seq=256 → 2.08s/step, 47GB/卡
# Stage2/3 最佳：batch=8/卡, seq=512 → 3.5s/step, 49GB/卡
# Stage2 max_steps=2500（等效 35 epochs × 4500条/64 per step）
# Stage3 max_steps=2750（等效 35 epochs × 4000条/64 per step）
# nohup 启动必须在命令内 cd 到项目目录，torchrun 用完整路径

# nohup 方式（推荐，防止 SSH 断线）：
# nohup bash -c "cd /home/shuzuan/prj/titans-pytorch-mlx && \
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   /home/shuzuan/miniconda3/envs/sglang/bin/torchrun --nproc_per_node=8 --master_port=29510 \
#   scripts/train_memory_oracle.py --stage N ... --multi-gpu" > /tmp/stageN.log 2>&1 &

/home/shuzuan/miniconda3/envs/sglang/bin/torchrun --nproc_per_node=8 \
  scripts/train_memory_oracle.py --stage 1 \
  --data data/oracle_stage1.jsonl \
  --model Qwen/Qwen3.5-0.8B --output checkpoints/oracle_stage1 \
  --multi-gpu --batch-size 16 --grad-accum 1 --seq-len 256

/home/shuzuan/miniconda3/envs/sglang/bin/torchrun --nproc_per_node=8 \
  scripts/train_memory_oracle.py --stage 2 \
  --data data/oracle_stage2_v2.jsonl \
  --resume checkpoints/oracle_stage1 --output checkpoints/oracle_stage2 \
  --multi-gpu --batch-size 8 --grad-accum 1 --seq-len 512 --max-steps 2500

/home/shuzuan/miniconda3/envs/sglang/bin/torchrun --nproc_per_node=8 \
  scripts/train_memory_oracle.py --stage 3 \
  --data data/oracle_stage3_v2.jsonl \
  --resume checkpoints/oracle_stage2 --output checkpoints/oracle_stage3 \
  --multi-gpu --batch-size 8 --grad-accum 1 --seq-len 512 --max-steps 2750
```

---

## MAG 预训练管线（归档，非当前重点）

```bash
# Wiki XML → tokenized .bin shards（Qwen2.5 tokenizer）
python scripts/convert_wiki.py \
  --input /data/wiki/zhwiki-latest-pages-articles.xml.bz2 \
  --output /data/wiki/zh_wiki.jsonl

python scripts/pretokenize_local.py \
  --sources /data/wiki/zh_wiki.jsonl \
  --output /data/tokens/ \
  --tokenizer Qwen/Qwen2.5-7B \
  --shard-size 100000000

# 7B MAG 预训练（8× H800，FSDP）
bash scripts/run_pretrain_7b.sh
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

## Memory Oracle 路线（当前方向）

Qwen3.5-0.8B + NeuralLongTermMemory（qwen35_injection.py），三阶段课程训练。

**注入位置**：每个 Qwen3.5 `Qwen3DecoderLayer` 前，MAL 风格
```
原：x → Attn → FFN
新：x → NeuralMemory(num_memory_layers=1) → x + mem_out → Attn → FFN
```

**训练配置**：
| 参数 | 值 |
|------|-----|
| 基模 | `Qwen/Qwen3.5-0.8B` |
| LoRA rank | 16（作用于 Attn + FFN） |
| memory 层 | num_memory_layers=1（线性），Titans 机制更新 |
| 目标任务 | LoCoMo 长对话记忆（Write→Read F1） |
| 数据 | build_memory_data_v2.py 合成，12 语种，3 阶段 |
| 评测 | eval_locomo.py（200 题 F1，baseline 0.427，已达 0.896）|

**已弃用**：TPTT 路线（Qwen2.5-7B + BABILong），见 `scripts/tptt_train.py`。

---

## 性能速查

**Memory Oracle（当前）**：

| 配置 | 步速 | 硬件 | 备注 |
|------|------|------|------|
| 0.8B + NLM×28 + LoRA r=16 | 1.42 s/step | H800×1 | 已测（grad_accum=8, seq=2048, batch=1）|
| 0.8B + NLM×28 + LoRA r=16 峰值显存 | 53 GB | H800×1 | 已测 |

**MAG 预训练（归档）**：

| 配置 | 步速 | 硬件 | 备注 |
|------|------|------|------|
| 7B MAG + 2-layer memory | ~166s/step | H800×8 | 已测，autograd 瓶颈 |
| 7B MAG + 1-layer memory | ~40s（推测）| H800×8 | 未测 |

---

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `src/titans/memory.py` | NeuralLongTermMemory, MemoryMLP（核心记忆模块）|
| `src/titans/qwen35_injection.py` | Qwen3.5 注入，freeze/unfreeze，get/set NLM states |
| `src/titans/config.py` | TitansConfig dataclass |
| `scripts/train_memory_oracle.py` | Memory Oracle 三阶段训练主脚本 |
| `scripts/train_multilang.py` | 自包含多语言训练流水线（生成→合并→训练→评测）|
| `scripts/build_memory_data_v2.py` | 数据合成（MiniMax API，12 语种，3 阶段）|
| `scripts/eval_locomo.py` | LoCoMo 评测（主评测脚本）|
| `src/titans/models.py` | MAGBlock, MACBlock, MALBlock（MAG 预训练，非当前重点）|
| `src/titans/tptt.py` | Qwen2.5 注入（TPTT 路线，已弃用）|

---

## Ablation 实验规范（从 autoresearch 方法论吸收）

**核心原则**：固定时间预算，不固定 step。架构不同步速不同，只有时间可比。

### 本项目 ablation 配置

| 项目 | 规格 |
|------|------|
| 统一起点 | `checkpoints/oracle_stage2/final`（固定，不能换）|
| 训练预算 | **1 小时**（H800 单卡，grad_accum=8，seq=2048）|
| 评测集 | LoCoMo 前 50 题（固定子集，一旦选定不能换）|
| 评测指标 | **write F1**（单一指标）|
| 结果记录 | `experiments/results.tsv`（每次追加）|

### results.tsv 格式

```
exp_id  config_delta          F1_50q  status   note
baseline  —                   0.xxx   keep     stage2/final zero-shot
E_A  no_NLM + LoRA r=16      0.xxx   keep/discard  纯LoRA ablation
E_B  NLM + LoRA r=16         0.xxx   keep/discard  NLM净贡献验证
```

### 实验纪律

- **单变量原则**：每次只改一个配置（有/无 NLM，不同 alpha，不同 LoRA rank）
- **eval 子集锁定**：50 题子集索引存 `experiments/eval_subset.json`，所有实验共用
- **先跑最关键的**：E_A（无NLM）vs E_B（有NLM）是最高优先级，回答 NLM 是否有净贡献

### program.md 概念（研究假设文件）

维护 `experiments/program.md`，内容：当前最优配置 + 下一批待测变量 + 成功标准。
实验迭代的是配置，不是代码。

---

## autoresearch 子环境（H800，归档）

GPT 小模型语言建模实验，已完成 E1（NLM 注入退化 +0.014 bpb，discard）。
结论：裸语言建模不是验证 NLM 的合适场景。

| 项目 | 值 |
|------|-----|
| 路径 | `/home/shuzuan/prj/autoresearch` |
| 分支 | `autoresearch/mar8` |
| uv | `~/.local/bin/uv` |
| 结果 | `/home/shuzuan/prj/autoresearch/results.tsv` |

---

## 参考文档（L3）

| 文档 | 路径 |
|------|------|
| **日志索引** | `docs/log/index.md` |
| Titans 论文摘要 + 后续工作 | `docs/titans-paper.md` |
| Qwen2.5 / Qwen3.5 架构规格 | `docs/model-specs.md` |
| 代码实现细节（流程图、数据结构）| `docs/implementation-notes.md` |
| 相关工作调研（AgeMem / Titans 生态 / autoresearch）| `docs/survey-related-work.md` |

### 日志使用方式

```bash
bash scripts/journal-new.sh <描述>   # 创建带模板的日志条目
bash scripts/memory-lint.sh          # 检查记忆系统健康度
```

有值得记录的事件时：
1. `bash scripts/journal-new.sh <描述>` 创建条目（自动生成模板）
2. 在"发现"章节标注 **[确认]**（实验/代码验证）或 **[推测]**（逻辑推断）
3. **必须**在 `docs/log/index.md` 更新当天那行的结论

日志只记事实（发生了什么、实验数字、发现了什么）。
"为什么这样决定" → L1 `decisions.md`。"有什么性能问题" → L1 `bottlenecks.md`。
