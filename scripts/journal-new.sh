#!/bin/bash
# journal-new.sh — 创建 docs/log/ 日志条目
# 用法: bash scripts/journal-new.sh <描述>
# 示例: bash scripts/journal-new.sh stage2-training-results

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "用法: $0 <描述>"
    echo "示例: $0 stage2-training-results"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE=$(date +%Y-%m-%d)
DESC=$(echo "$1" | tr ' /\\'"'"'"' '-' | sed 's/--*/-/g; s/^-//; s/-$//' | cut -c1-64)
DIR="$PROJECT_ROOT/docs/log/${DATE}"
FILE="$DIR/${DESC}.md"

if [ -f "$FILE" ]; then
    echo "已存在: $FILE"
    exit 0
fi

mkdir -p "$DIR"

cat > "$FILE" << EOF
# ${DESC}

创建: $(date '+%Y-%m-%d %H:%M')
状态: 进行中

## 目标

(这次探索/实验要解决什么问题)

## 过程

(按时间顺序记录操作和观察)

## 发现

标注规则: [确认] = 实验/日志/代码验证 | [推测] = 逻辑推断未验证

## 结论

投影到: (完成后填写：写入 MEMORY.md 的哪个块 / 更新了哪些 L2 文档)
EOF

# 提示更新 index.md
INDEX="$PROJECT_ROOT/docs/log/index.md"
if [ -f "$INDEX" ]; then
    echo ""
    echo "✅ $FILE"
    echo "📝 记得在 $INDEX 更新当天那行的结论"
else
    echo "✅ $FILE"
fi
