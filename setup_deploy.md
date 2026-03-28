# Stage1 部署与启动说明

本文档说明当前 stage1 最小训练路径的：

- 环境配置
- 服务器目录约定
- 启动命令
- 训练脚本入参

当前新增核心文件：

- `src/titans/stage1_data.py`
- `src/titans/stage1_models.py`
- `scripts/train_stage1_distributed.py`

当前已部署到服务器目录：

- `~/lin/titans_lin/`

---

## 1. 环境配置

根据当前分支约束：

- 训练服务器目录：`~/lin/titans_lin/`
- conda 环境：`titans_train`
- 分布式启动方式：`accelerate launch`

### 1.1 激活环境

```bash
conda activate titans_train
cd ~/lin/titans_lin
```

### 1.2 安装依赖

如果服务器环境还没补齐训练依赖，可执行：

```bash
pip install -e .
pip install accelerate transformers datasets wandb
```

或直接：

```bash
pip install -e '.[train]'
```

### 1.3 服务器本地模型

当前服务器上已存在可直接用于 stage1 测试的本地 Qwen2.5-7B 模型：

```text
/home/shuzuan/models/Qwen/Qwen2.5-7B
```

建议在服务器 smoke test、DDP/FSDP 联调时优先使用该本地路径，避免因外网不可达导致 `from_pretrained` 拉取失败。

也就是说，服务器上启动训练时，优先写成：

```bash
--backbone-name /home/shuzuan/models/Qwen/Qwen2.5-7B
```

---

## 2. 数据位置

默认读取：

```text
data/generated/stage1/
├── train.jsonl
├── eval_structured.jsonl
└── eval_itinerary.jsonl
```

默认参数下无需改数据路径。

---

## 3. 启动命令

### 3.1 单卡启动

```bash
PYTHONPATH=src accelerate launch scripts/train_stage1_distributed.py \
  --data-dir data/generated/stage1 \
  --checkpoint-dir checkpoints/stage1 \
  --backbone qwen \
  --backbone-name /home/shuzuan/models/Qwen/Qwen2.5-7B \
  --history-backbone-mode full \
  --memory-update-source last_hidden \
  --num-retrieved-memory-tokens 16 \
  --loss-mask-scope answer_only \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 1 \
  --lr 1e-4 \
  --epochs 1
```

### 3.2 多卡 DDP 启动

```bash
PYTHONPATH=src accelerate launch \
  --config_file configs/accelerate_ddp.yaml \
  scripts/train_stage1_distributed.py \
  --data-dir data/generated/stage1 \
  --checkpoint-dir checkpoints/stage1 \
  --backbone qwen \
  --backbone-name /home/shuzuan/models/Qwen/Qwen2.5-7B \
  --history-backbone-mode full \
  --memory-update-source last_hidden \
  --num-retrieved-memory-tokens 16 \
  --loss-mask-scope answer_only \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 1 \
  --lr 1e-4 \
  --epochs 1
```

### 3.3 FSDP 启动

```bash
PYTHONPATH=src accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  scripts/train_stage1_distributed.py \
  --data-dir data/generated/stage1 \
  --checkpoint-dir checkpoints/stage1 \
  --backbone qwen \
  --backbone-name /home/shuzuan/models/Qwen/Qwen2.5-7B \
  --history-backbone-mode full \
  --memory-update-source last_hidden \
  --num-retrieved-memory-tokens 16 \
  --loss-mask-scope answer_only \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 1 \
  --lr 1e-4 \
  --epochs 1
```

---

## 4. 默认实现约定

当前默认行为：

- `history_backbone_mode=full`
- `memory_update_source=last_hidden`
- `num_retrieved_memory_tokens=16`
- `loss_mask_scope=answer_only`

语义说明：

- history chunk 全量过 frozen backbone
- 使用最后一层 hidden states 更新 Titans memory
- question 阶段检索 memory token 作为前缀上下文
- loss 只对 answer token 生效，其他位置为 `-100`

---

## 5. 主要入参说明

### 5.1 数据与输出

- `--data-dir`：stage1 数据目录，默认 `data/generated/stage1`
- `--checkpoint-dir`：checkpoint 输出目录，默认 `checkpoints/stage1`
- `--resume`：从已有 checkpoint 恢复

### 5.2 backbone 相关

- `--backbone`：当前仅支持 `qwen`
- `--backbone-name`：HF 模型名或本地模型路径
- `--torch-dtype`：`auto` / `fp16` / `bf16` / `float32`
- `--attn-implementation`：可选 attention 实现
- `--trust-remote-code`：是否允许 remote code

### 5.3 memory 相关

- `--history-backbone-mode`：当前实现为 `full`
- `--memory-update-source`：当前实现为 `last_hidden`
- `--num-retrieved-memory-tokens`：检索出的 memory token 数量，默认 `16`
- `--loss-mask-scope`：当前实现为 `answer_only`
- `--num-persistent-tokens`：persistent token 数量
- `--num-memory-layers`：Titans memory MLP 层数
- `--memory-hidden-mult`：memory hidden dim 倍数
- `--memory-lr`：memory update learning rate 系数
- `--memory-momentum`：memory momentum 系数
- `--memory-decay`：memory decay 系数
- `--memory-dropout`：memory dropout
- `--use-memory-conv`：是否启用 memory 中的 depthwise conv

### 5.4 长度控制

- `--max-history-length`：每个 history chunk 的最大 token 长度
- `--max-question-length`：question prompt 最大长度
- `--max-sequence-length`：`question + answer` 最大长度

### 5.5 训练相关

- `--epochs`：训练轮数
- `--max-steps`：最大训练 step，`-1` 表示按 epoch 推导
- `--batch-size`：训练 batch size
- `--eval-batch-size`：评估 batch size
- `--grad-accum`：梯度累积步数
- `--lr`：优化器学习率
- `--weight-decay`：权重衰减
- `--grad-clip`：梯度裁剪
- `--warmup-ratio`：warmup 比例
- `--precision`：`none` / `fp16` / `bf16`
- `--gradient-checkpointing`：启用 backbone gradient checkpointing

### 5.6 日志与保存

- `--save-every`：每多少 step 保存 checkpoint
- `--eval-every`：每多少 step 做一次评估
- `--log-every`：每多少 step 打日志
- `--wandb`：启用 wandb
- `--wandb-project`：wandb project 名称
- `--wandb-run-name`：wandb run 名称

---

## 6. checkpoint 说明

训练脚本会在 `checkpoint-dir` 下保存：

- `best_model/`
- `step_xxx/`
- `final_model/`

每个 checkpoint 至少包含：

- accelerate 状态
- `trainer_state.json`
- `stage1_state.pt`

其中 `stage1_state.pt` 包含：

- 训练配置
- 模型配置
- `global_step`
- `epoch`
- 总参数量 / 可训练参数量
- memory-only 相关权重快照

---

## 7. 当前注意事项

第一版目标是先打通最小闭环，因此当前实现重点是：

- frozen backbone + Titans memory
- stage1 数据可直接训练

---

## 8. Stage1 服务部署与启动

当前仓库已补充最小 stage1 服务入口：

- `scripts/serve_stage1.py`

它是 `src/titans/stage1_runtime.py` 的 thin wrapper，直接复用：

- `Stage1DeploymentRuntime.write_memory`
- `Stage1DeploymentRuntime.chat_with_memory`
- `Stage1DeploymentRuntime.get_session_metadata`
- `Stage1DeploymentRuntime.delete_session`

当前只覆盖最小 smoke / demo 能力，不引入额外服务治理。

### 8.1 服务依赖

当前服务脚本只使用 Python 标准库 HTTP server，不额外引入 FastAPI / Uvicorn。

因此依赖仍以模型运行依赖为主：

```bash
pip install -e .
pip install -e '.[train]'
```

至少需要：

- `torch`
- `transformers`

### 8.2 模型与 checkpoint 路径要求

服务启动前需要准备：

1. `--backbone-name`
   - 可以是 HF 模型名
   - 也可以是本地模型路径
2. `--checkpoint-path`（可选）
   - 如果不传，则只加载 backbone + 当前代码里的 memory 模块初始权重
   - 如果传入，则会加载训练得到的 memory 相关权重

服务器上建议优先使用本地 backbone 路径：

```text
/home/shuzuan/models/Qwen/Qwen2.5-7B
```

### 8.3 本地单机启动命令

默认只绑定 localhost：

```bash
PYTHONPATH=src python3 scripts/serve_stage1.py \
  --host 127.0.0.1 \
  --port 8000 \
  --backbone-name /path/to/Qwen2.5-7B \
  --checkpoint-path checkpoints/stage1_timeline_v2/final_model/stage1_state.pt
```

如果只是做基础验证，也可以先不传 checkpoint：

```bash
PYTHONPATH=src python3 scripts/serve_stage1.py \
  --host 127.0.0.1 \
  --port 8000 \
  --backbone-name /path/to/Qwen2.5-7B
```

### 8.4 远程服务器启动命令

按当前分支约束，建议在服务器上执行：

```bash
cd ~/lin/titans_lin
conda activate titans_train
PYTHONPATH=src python3 scripts/serve_stage1.py \
  --host 127.0.0.1 \
  --port 8000 \
  --backbone-name /home/shuzuan/models/Qwen/Qwen2.5-7B \
  --checkpoint-path checkpoints/stage1_timeline_v2/final_model/stage1_state.pt
```

如果 checkpoint 文件名不同，按实际路径替换即可。

### 8.5 HTTP 接口

当前服务提供 4 个接口：

- `POST /v1/memory/write`
- `POST /v1/chat/respond`
- `GET /v1/sessions/{session_id}`
- `DELETE /v1/sessions/{session_id}`

#### 1) 写入 memory

```bash
curl -X POST http://127.0.0.1:8000/v1/memory/write \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "contents": ["梁安航目前在深圳工作。"]
  }'
```

#### 2) 带 memory 回答

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/respond \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "query": "梁安航目前在哪个城市工作？",
    "max_new_tokens": 64,
    "temperature": 0.0,
    "include_debug": true
  }'
```

#### 3) 查看 session metadata

```bash
curl http://127.0.0.1:8000/v1/sessions/demo-1
```

只返回 metadata，不返回原始文本。

#### 4) 删除 session

```bash
curl -X DELETE http://127.0.0.1:8000/v1/sessions/demo-1
```

### 8.6 session 清理方式

当前 session state 保存在服务进程内存中。

清理方式：

1. 对单个 session：
   - `DELETE /v1/sessions/{session_id}`
2. 对全部 session：
   - 重启服务进程

当前阶段不引入外部 KV / DB。

---

## 9. 本地 usecase 脚本

当前仓库已补充最小本地示例：

- `usecase/run_stage1_eval_case.py`

默认读取：

```text
data/generated/stage1_timeline_v2_sample/eval.jsonl
```

默认使用其中一条样本字段：

- `history_chunks`
- `question_chunk`
- `answer`

脚本流程：

1. 从 eval 数据读取一条样本
2. 把 `history_chunks` 写入 memory
3. 对 `question_chunk` 做一次带记忆回答
4. 删除 session，清空 memory state
5. 对同一个问题再做一次无记忆回答
6. 打印：
   - Gold / 标准答案
   - With memory
   - Without memory

### 9.1 运行方式

```bash
PYTHONPATH=src python3 usecase/run_stage1_eval_case.py \
  --data-file data/generated/stage1_timeline_v2_sample/eval.jsonl \
  --index 0 \
  --backbone-name /path/to/Qwen2.5-7B \
  --checkpoint-path checkpoints/stage1_timeline_v2/final_model/stage1_state.pt
```

如果只是本地基础验证，也可以先不传 checkpoint。

---

## 10. 本地 smoke 验证步骤

本地环境较弱时，只要求做到“基本可跑、能做基础验证”。

建议顺序：

### 10.1 跑 runtime 单测

```bash
python3 -m pytest tests/test_stage1_runtime.py
```

### 10.2 跑本地 usecase

```bash
PYTHONPATH=src python3 usecase/run_stage1_eval_case.py \
  --backbone-name /path/to/Qwen2.5-7B
```

确认终端能打印：

- `Gold / 标准答案`
- `With memory`
- `Without memory`

### 10.3 跑本地 HTTP smoke

先启动服务：

```bash
PYTHONPATH=src python3 scripts/serve_stage1.py \
  --host 127.0.0.1 \
  --port 8000 \
  --backbone-name /path/to/Qwen2.5-7B
```

然后依次验证：

1. `POST /v1/memory/write`
2. `POST /v1/chat/respond`
3. `GET /v1/sessions/{session_id}`
4. `DELETE /v1/sessions/{session_id}`

---

## 11. 远程服务器测试建议步骤

本地完成基础改动后，再到服务器做真实模型测试。

建议顺序：

1. 同步代码到 `~/lin/titans_lin/`
2. `conda activate titans_train`
3. 确认 checkpoint 路径与本地模型路径
4. 启动 `scripts/serve_stage1.py`
5. 先跑一次 `usecase/run_stage1_eval_case.py`
6. 再用 `curl` 验证 4 个 HTTP 接口
7. 记录：
   - 使用的 checkpoint
   - 使用的 backbone 路径
   - 样本 index
   - 返回结果是否符合预期

如果服务器侧有临时修复，记得同步回本地分支。

当前脚本偏向最小可用实现，后续可继续补：

- 更完整的 backbone adapter 抽象
- 更细粒度的评估指标
- 更严格的 resume / checkpoint 元信息
- 更完整的 FSDP 配置示例

