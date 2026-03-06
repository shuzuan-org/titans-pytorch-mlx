#!/bin/bash
# Qwen3.5-0.8B Memory Oracle — 英文从零训练 (Stage 1-3)
# 从基模型出发，全程用英文合成数据，markers=[WRITE]/[QUERY]/[MEMORY]
#
# 用法：
#   nohup bash scripts/train_08b_en.sh > logs/train_08b_en.log 2>&1 &

set -eo pipefail

MODEL=/home/shuzuan/models/Qwen/Qwen3___5-0.8B
CKPT=/home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_08b_en
LOGDIR=/home/shuzuan/prj/titans-pytorch-mlx/logs
DATA=/home/shuzuan/prj/titans-pytorch-mlx/data/memory_en
SCRIPT=/home/shuzuan/prj/titans-pytorch-mlx/scripts/train_memory_oracle.py

mkdir -p "$LOGDIR" "$CKPT"

TORCHRUN=/home/shuzuan/miniconda3/envs/sglang/bin/torchrun
GPUS=0,1,2,3,4,5,6
PORT=29503

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — 单轮对话热身 (5000 steps)
# 目标：学会 [WRITE]/[QUERY]/[MEMORY] 格式，基础记忆读写
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Stage 1 (English warmup, 7-GPU DDP) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 1 \
    --lang en \
    --data "$DATA/stage1.jsonl" \
    --model "$MODEL" \
    --output "$CKPT/stage1" \
    --max-steps 5000 \
    --save-steps 1000 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 1e-4 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_en_s1.log"

echo "=== Stage 1 完成，开始 Stage 2 ==="

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — 多轮对话 + 噪声 (10000 steps)
# 目标：跨 session 记忆积累，抗噪声干扰
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Stage 2 (English multi-session, 7-GPU DDP) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 2 \
    --lang en \
    --data "$DATA/stage2.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage1/final" \
    --output "$CKPT/stage2" \
    --max-steps 10000 \
    --save-steps 2000 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 5e-5 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_en_s2.log"

echo "=== Stage 2 完成，开始 Stage 3 ==="

# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — 实体更新精调 (3000 steps)
# 目标：理解信息更新（如"we moved"/"changed jobs"），精确覆盖旧记忆
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Stage 3 (English entity update, 7-GPU DDP) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 3 \
    --lang en \
    --data "$DATA/stage3.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage2/final" \
    --output "$CKPT/stage3" \
    --max-steps 3000 \
    --save-steps 600 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 2e-5 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_en_s3.log"

echo "=== 英文 Stage 1-3 全部完成 ==="
echo "Checkpoint: $CKPT/stage3/final"
