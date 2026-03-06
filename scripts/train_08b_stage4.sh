#!/bin/bash
# Qwen3.5-0.8B Memory Oracle Stage 4 — 英文 LoCoMo 迁移微调
# 从 Stage 3 checkpoint 继续，用英文对话数据迁移到英文域
#
# 用法：
#   nohup bash scripts/train_08b_stage4.sh > logs/train_08b_stage4.log 2>&1 &

set -eo pipefail

MODEL=/home/shuzuan/models/Qwen/Qwen3___5-0.8B
CKPT=/home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_08b
LOGDIR=/home/shuzuan/prj/titans-pytorch-mlx/logs
DATA=/home/shuzuan/prj/titans-pytorch-mlx/data
SCRIPT=/home/shuzuan/prj/titans-pytorch-mlx/scripts/train_memory_oracle.py

mkdir -p "$LOGDIR" "$CKPT"

TORCHRUN=/home/shuzuan/miniconda3/envs/sglang/bin/torchrun
# 用 GPU6（避开评测用的 GPU7）
GPUS=0,1,2,3,4,5,6
PORT=29502

echo "=== Stage 4 (English LoCoMo fine-tune, 7-GPU DDP) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 2 \
    --data "$DATA/oracle_en.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage3/final" \
    --output "$CKPT/stage4" \
    --max-steps 2000 \
    --save-steps 400 \
    --batch-size 1 \
    --grad-accum 2 \
    --lr 5e-6 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_stage4.log"

echo "=== Stage 4 完成 ==="
