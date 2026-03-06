#!/bin/bash
# 生成英文三阶段训练数据并合并
# 在 Stage 1 并行生成完成后运行此脚本继续生成 Stage 2/3
#
# 用法：nohup bash scripts/build_en_data_all.sh > logs/build_en_data_all.log 2>&1 &

set -eo pipefail

PY=/home/shuzuan/miniconda3/envs/sglang/bin/python3
SCRIPT=/home/shuzuan/prj/titans-pytorch-mlx/scripts/build_memory_data_v2.py
MODEL=/home/shuzuan/models/Qwen/Qwen3___5-0.8B
DATA=/home/shuzuan/prj/titans-pytorch-mlx/data/memory_en
LOGS=/home/shuzuan/prj/titans-pytorch-mlx/logs
mkdir -p "$DATA" "$LOGS"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: 等待 Stage 1 并行生成完成，然后合并
# ──────────────────────────────────────────────────────────────────────────────
echo "=== 等待 Stage 1 并行生成完成 ==="
wait $(pgrep -f "stage1_part" || true)

echo "=== 合并 Stage 1 数据 ==="
# part0 已在 stage1.jsonl，合并 part1-6 到同一文件
for i in 1 2 3 4 5 6; do
    PART="$DATA/stage1_part${i}.jsonl"
    if [ -f "$PART" ]; then
        cat "$PART" >> "$DATA/stage1.jsonl"
        echo "  merged: $PART ($(wc -l < "$PART") lines)"
    fi
done
echo "Stage 1 total: $(wc -l < "$DATA/stage1.jsonl") samples"

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Stage 2 并行生成（10000 条，7 GPU × 1500 条）
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Stage 2 并行生成 (7 GPU × 1500 条) ==="
PIDS=()
for GPU in 0 1 2 3 4 5 6; do
    nohup $PY $SCRIPT \
        --stage 2 --lang en --n-samples 1500 \
        --output "$DATA/stage2_part${GPU}.jsonl" \
        --backend local --model $MODEL --device cuda:${GPU} \
        > "$LOGS/build_en_s2_part${GPU}.log" 2>&1 &
    PIDS+=($!)
    echo "  Started Stage2 part${GPU} on cuda:${GPU} PID=${PIDS[-1]}"
done

echo "等待 Stage 2 生成完成..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "=== 合并 Stage 2 数据 ==="
for i in 0 1 2 3 4 5 6; do
    PART="$DATA/stage2_part${i}.jsonl"
    if [ -f "$PART" ]; then
        cat "$PART" >> "$DATA/stage2.jsonl"
    fi
done
echo "Stage 2 total: $(wc -l < "$DATA/stage2.jsonl") samples"

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Stage 3 并行生成（3000 条，7 GPU × 450 条）
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Stage 3 并行生成 (7 GPU × 450 条) ==="
PIDS=()
for GPU in 0 1 2 3 4 5 6; do
    nohup $PY $SCRIPT \
        --stage 3 --lang en --n-samples 450 \
        --output "$DATA/stage3_part${GPU}.jsonl" \
        --backend local --model $MODEL --device cuda:${GPU} \
        > "$LOGS/build_en_s3_part${GPU}.log" 2>&1 &
    PIDS+=($!)
    echo "  Started Stage3 part${GPU} on cuda:${GPU} PID=${PIDS[-1]}"
done

echo "等待 Stage 3 生成完成..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "=== 合并 Stage 3 数据 ==="
for i in 0 1 2 3 4 5 6; do
    PART="$DATA/stage3_part${i}.jsonl"
    if [ -f "$PART" ]; then
        cat "$PART" >> "$DATA/stage3.jsonl"
    fi
done
echo "Stage 3 total: $(wc -l < "$DATA/stage3.jsonl") samples"

# ──────────────────────────────────────────────────────────────────────────────
# 验证所有数据
# ──────────────────────────────────────────────────────────────────────────────
echo "=== 验证数据质量 ==="
for STAGE in 1 2 3; do
    $PY $SCRIPT --validate --output "$DATA/stage${STAGE}.jsonl" --lang en
done

echo "=== 英文三阶段数据生成全部完成 ==="
echo "  Stage 1: $(wc -l < "$DATA/stage1.jsonl") samples"
echo "  Stage 2: $(wc -l < "$DATA/stage2.jsonl") samples"
echo "  Stage 3: $(wc -l < "$DATA/stage3.jsonl") samples"
echo ""
echo "下一步：bash scripts/train_08b_en.sh"
