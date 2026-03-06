#!/bin/bash
# TPTT fine-tuning on H800 (single GPU, verified config)
# Usage: bash scripts/run_tptt_h800.sh [--steps N] [--output DIR]
cd /home/shuzuan/prj/titans-pytorch-mlx
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

STEPS=${1:-10000}
OUTPUT=${2:-/home/shuzuan/checkpoints/tptt_qwen25_7b}
# 消耗已用的位置参数，避免 "$@" 把 $1/$2 重复传给 Python
[ $# -ge 2 ] && shift 2 || shift $#

/home/shuzuan/miniconda3/envs/sglang/bin/python scripts/tptt_train.py   --model /home/shuzuan/models/Qwen/Qwen2___5-7B   --data /home/shuzuan/tokens/longwanjuan_zh   --max-steps $STEPS   --batch-size 1   --grad-accum 8   --seq-len 2048   --lora-rank 16   --lora-alpha 32   --num-workers 0   --save-steps 500   --output $OUTPUT   --device cuda   --dtype bfloat16   --use-8bit-adam   "$@"
