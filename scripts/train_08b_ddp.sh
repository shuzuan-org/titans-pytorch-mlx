#!/bin/bash
# Qwen3.5-0.8B Memory Oracle Stage 1+2+3 — 7-GPU DDP
# GPU0-6: 训练, GPU7: 空闲（可用于监控 eval）
#
# 用法：
#   nohup bash scripts/train_08b_ddp.sh > logs/train_08b_ddp.log 2>&1 &

set -eo pipefail

MODEL=/home/shuzuan/models/Qwen/Qwen3___5-0.8B
CKPT=/home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_08b
LOGDIR=/home/shuzuan/prj/titans-pytorch-mlx/logs
DATA=/home/shuzuan/prj/titans-pytorch-mlx/data
SCRIPT=/home/shuzuan/prj/titans-pytorch-mlx/scripts/train_memory_oracle.py

mkdir -p "$LOGDIR" "$CKPT"

# 7 卡并行，grad_accum=2 → effective_batch = 7×1×2 = 14 ≈ 原单卡 16
TORCHRUN=/home/shuzuan/miniconda3/envs/sglang/bin/torchrun
GPUS=0,1,2,3,4,5,6
PORT=29501

echo "=== Stage 1 (7-GPU DDP, 0.8B) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 1 \
    --data "$DATA/oracle_1.jsonl" \
    --model "$MODEL" \
    --output "$CKPT/stage1" \
    --max-steps 500 \
    --save-steps 100 \
    --batch-size 1 \
    --grad-accum 2 \
    --lr 2e-4 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_stage1.log"

echo "=== Stage 2 (7-GPU DDP, 0.8B) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 2 \
    --data "$DATA/oracle_2.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage1/final" \
    --output "$CKPT/stage2" \
    --max-steps 3000 \
    --save-steps 500 \
    --batch-size 1 \
    --grad-accum 2 \
    --lr 5e-5 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_stage2.log"

echo "=== Stage 3 (7-GPU DDP, 0.8B) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 \
    --master_port=$PORT \
    "$SCRIPT" \
    --stage 3 \
    --data "$DATA/oracle_3.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage2/final" \
    --output "$CKPT/stage3" \
    --max-steps 1000 \
    --save-steps 200 \
    --batch-size 1 \
    --grad-accum 2 \
    --lr 1e-5 \
    --lora-rank 16 \
    --num-memory-layers 1 \
    --multi-gpu \
    2>&1 | tee "$LOGDIR/train_08b_stage3.log"

echo "=== 全部完成 ==="
