#!/bin/bash
# 英文数据 + 训练完整流水线
# Stage 1 正在并行生成中（GPU0-6），本脚本轮询等待完成后自动继续
#
# 用法：nohup bash scripts/run_en_pipeline.sh > logs/run_en_pipeline.log 2>&1 &

set -eo pipefail

PY=/home/shuzuan/miniconda3/envs/sglang/bin/python3
TORCHRUN=/home/shuzuan/miniconda3/envs/sglang/bin/torchrun
SCRIPT=/home/shuzuan/prj/titans-pytorch-mlx/scripts/build_memory_data_v2.py
TRAIN=/home/shuzuan/prj/titans-pytorch-mlx/scripts/train_memory_oracle.py
MODEL=/home/shuzuan/models/Qwen/Qwen3___5-0.8B
DATA=/home/shuzuan/prj/titans-pytorch-mlx/data/memory_en
CKPT=/home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_08b_en
LOGS=/home/shuzuan/prj/titans-pytorch-mlx/logs
mkdir -p "$DATA" "$CKPT" "$LOGS"

PORT=29503
GPUS=0,1,2,3,4,5,6
S1_TARGET=5000
S2_TARGET=10000
S3_TARGET=3000

log() { echo "[$(date '+%H:%M:%S')] $*"; }

count_lines() {
    local total=0
    for f in "$@"; do
        [ -f "$f" ] && total=$((total + $(wc -l < "$f")))
    done
    echo $total
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 等待 Stage 1 完成（轮询，每 3 分钟检查一次）
# ──────────────────────────────────────────────────────────────────────────────
log "=== Phase 1: 等待 Stage 1 完成（目标 $S1_TARGET 条）==="
while true; do
    total=$(count_lines "$DATA/stage1.jsonl" "$DATA"/stage1_part*.jsonl 2>/dev/null)
    running=$(pgrep -c -f "build_memory_data_v2.*stage1" 2>/dev/null || echo 0)
    log "  Stage 1: $total 条 | 运行中进程: $running"
    if [ "$total" -ge "$S1_TARGET" ] || [ "$running" -eq 0 ]; then
        log "  Stage 1 条件满足，继续。"
        break
    fi
    sleep 180
done

# 停掉仍在跑的 Stage 1 进程（已够数）
pkill -f "build_memory_data_v2.*stage1" 2>/dev/null || true
sleep 2

# 合并 part*.jsonl → stage1.jsonl
log "合并 Stage 1 数据..."
for i in 1 2 3 4 5 6; do
    PART="$DATA/stage1_part${i}.jsonl"
    [ -f "$PART" ] && cat "$PART" >> "$DATA/stage1.jsonl" && rm "$PART"
done
log "Stage 1 total: $(wc -l < "$DATA/stage1.jsonl") samples"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Stage 2 数据生成（7 GPU 并行）
# ──────────────────────────────────────────────────────────────────────────────
log "=== Phase 2: Stage 2 数据生成 (7 GPU × 1500 条) ==="
PIDS=()
for GPU in 0 1 2 3 4 5 6; do
    nohup $PY $SCRIPT \
        --stage 2 --lang en --n-samples 1500 \
        --output "$DATA/stage2_part${GPU}.jsonl" \
        --backend local --model $MODEL --device cuda:${GPU} \
        > "$LOGS/build_en_s2_part${GPU}.log" 2>&1 &
    PIDS+=($!)
done
log "Stage 2 进程: ${PIDS[*]}"

for PID in "${PIDS[@]}"; do wait "$PID" 2>/dev/null || true; done

log "合并 Stage 2..."
for i in 0 1 2 3 4 5 6; do
    PART="$DATA/stage2_part${i}.jsonl"
    [ -f "$PART" ] && cat "$PART" >> "$DATA/stage2.jsonl" && rm "$PART"
done
log "Stage 2 total: $(wc -l < "$DATA/stage2.jsonl") samples"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Stage 3 数据生成（7 GPU 并行）
# ──────────────────────────────────────────────────────────────────────────────
log "=== Phase 3: Stage 3 数据生成 (7 GPU × 450 条) ==="
PIDS=()
for GPU in 0 1 2 3 4 5 6; do
    nohup $PY $SCRIPT \
        --stage 3 --lang en --n-samples 450 \
        --output "$DATA/stage3_part${GPU}.jsonl" \
        --backend local --model $MODEL --device cuda:${GPU} \
        > "$LOGS/build_en_s3_part${GPU}.log" 2>&1 &
    PIDS+=($!)
done
log "Stage 3 进程: ${PIDS[*]}"

for PID in "${PIDS[@]}"; do wait "$PID" 2>/dev/null || true; done

log "合并 Stage 3..."
for i in 0 1 2 3 4 5 6; do
    PART="$DATA/stage3_part${i}.jsonl"
    [ -f "$PART" ] && cat "$PART" >> "$DATA/stage3.jsonl" && rm "$PART"
done
log "Stage 3 total: $(wc -l < "$DATA/stage3.jsonl") samples"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: 三阶段训练
# ──────────────────────────────────────────────────────────────────────────────
log "=== Phase 4: Stage 1 训练 (5000 steps) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 --master_port=$PORT \
    "$TRAIN" \
    --stage 1 --lang en \
    --data "$DATA/stage1.jsonl" \
    --model "$MODEL" \
    --output "$CKPT/stage1" \
    --max-steps 5000 --save-steps 1000 \
    --batch-size 1 --grad-accum 4 \
    --lr 1e-4 --lora-rank 16 --num-memory-layers 1 --multi-gpu \
    2>&1 | tee "$LOGS/train_08b_en_s1.log"

log "=== Phase 5: Stage 2 训练 (10000 steps) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 --master_port=$PORT \
    "$TRAIN" \
    --stage 2 --lang en \
    --data "$DATA/stage2.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage1/final" \
    --output "$CKPT/stage2" \
    --max-steps 10000 --save-steps 2000 \
    --batch-size 1 --grad-accum 4 \
    --lr 5e-5 --lora-rank 16 --num-memory-layers 1 --multi-gpu \
    2>&1 | tee "$LOGS/train_08b_en_s2.log"

log "=== Phase 6: Stage 3 训练 (3000 steps) ==="
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN \
    --nproc_per_node=7 --master_port=$PORT \
    "$TRAIN" \
    --stage 3 --lang en \
    --data "$DATA/stage3.jsonl" \
    --model "$MODEL" \
    --resume "$CKPT/stage2/final" \
    --output "$CKPT/stage3" \
    --max-steps 3000 --save-steps 600 \
    --batch-size 1 --grad-accum 4 \
    --lr 2e-5 --lora-rank 16 --num-memory-layers 1 --multi-gpu \
    2>&1 | tee "$LOGS/train_08b_en_s3.log"

log "=== 全流水线完成 ==="
log "Checkpoint: $CKPT/stage3/final"
