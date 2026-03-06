#!/bin/bash
# Qwen3.5-9B Memory Oracle Stage 2/3 — 7-GPU DDP 训练
# GPU0-6: 训练, GPU7: eval 监控
#
# 用法：bash scripts/train_9b_ddp.sh [stage2|stage3|all]
#
# Stage2 预计 ~6h（7卡加速），Stage3 ~2h

set -e

MODEL=/home/shuzuan/models/Qwen/Qwen3___5-9B
CKPT=/home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_9b
LOGDIR=/home/shuzuan/prj/titans-pytorch-mlx/logs
DATA_DIR=/home/shuzuan/prj/titans-pytorch-mlx/data
SCRIPT=/home/shuzuan/prj/titans-pytorch-mlx/scripts/train_memory_oracle.py

mkdir -p "$LOGDIR"

STAGE=${1:-all}

# ── Stage 2 ────────────────────────────────────────────────────────────────
run_stage2() {
    echo "=== Stage 2 (7-GPU DDP) ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
        --nproc_per_node=7 \
        --master_port=29500 \
        "$SCRIPT" \
        --stage 2 \
        --data "$DATA_DIR/oracle_2.jsonl" \
        --model "$MODEL" \
        --resume "$CKPT/stage1" \
        --output "$CKPT/stage2" \
        --max-steps 10000 \
        --batch-size 1 \
        --grad-accum 2 \
        --lr 5e-5 \
        --lora-rank 16 \
        --num-memory-layers 1 \
        --multi-gpu \
        2>&1 | tee "$LOGDIR/train_9b_stage2.log"
    echo "Stage 2 done."
}

# ── Stage 3 ────────────────────────────────────────────────────────────────
run_stage3() {
    echo "=== Stage 3 (7-GPU DDP) ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
        --nproc_per_node=7 \
        --master_port=29500 \
        "$SCRIPT" \
        --stage 3 \
        --data "$DATA_DIR/oracle_3.jsonl" \
        --model "$MODEL" \
        --resume "$CKPT/stage2/final" \
        --output "$CKPT/stage3" \
        --max-steps 3000 \
        --batch-size 1 \
        --grad-accum 2 \
        --lr 1e-5 \
        --lora-rank 16 \
        --num-memory-layers 1 \
        --multi-gpu \
        2>&1 | tee "$LOGDIR/train_9b_stage3.log"
    echo "Stage 3 done."
}

# ── 运行 ───────────────────────────────────────────────────────────────────
case "$STAGE" in
    stage2) run_stage2 ;;
    stage3) run_stage3 ;;
    all)
        run_stage2
        run_stage3
        ;;
    *)
        echo "Usage: $0 [stage2|stage3|all]"
        exit 1
        ;;
esac

echo "=== 全部完成 ==="
