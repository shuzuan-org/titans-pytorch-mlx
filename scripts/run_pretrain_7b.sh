#!/bin/bash
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
#
# Launch 7B TitansMAG pretraining on 8x H800 GPUs.
# Total batch: 2 seq × 8 GPU × 32 grad_accum = 512 seq × 8192 tokens ≈ 4.2M tokens/step
# Schedule: 75,000 steps × 4.2M tokens/step = 315B tokens

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

accelerate launch \
  --config_file configs/fsdp_7b.yaml \
  scripts/pretrain_distributed.py \
  --model mag \
  --dim 3584 \
  --num-heads 28 \
  --num-layers 24 \
  --vocab-size 152064 \
  --seq-len 8192 \
  --batch-size 2 \
  --grad-accum 32 \
  --schedule wsd \
  --max-lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-steps 2000 \
  --stable-steps 65500 \
  --decay-steps 7500 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --bin-data configs/mix_weights_7b.yaml \
  --checkpoint-dir /home/shuzuan/checkpoints/titans_7b \
  --save-every 1000 \
  --eval-every 500 \
  --log-every 10 \
  --wandb \
  --wandb-project titans-7b-300B \
  "$@"
