#!/bin/bash
# memory-lint.sh — 检查项目记忆系统健康度
# 用法: bash scripts/memory-lint.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="$HOME/.claude/projects/$(echo "$PROJECT_ROOT" | tr '/' '-')/memory"
MEMORY="$MEMORY_DIR/MEMORY.md"

echo "=== Titans 记忆系统健康检查 ==="
echo ""

# 1. L1 行数
if [ ! -f "$MEMORY" ]; then
    echo "❌ MEMORY.md 不存在: $MEMORY"
    exit 1
fi
lines=$(wc -l < "$MEMORY")
echo "📏 L1 行数: $lines / 200"
if [ "$lines" -gt 180 ]; then
    echo "   ⚠️  接近上限，淘汰最冷条目"
elif [ "$lines" -gt 150 ]; then
    echo "   📊 空间适中"
else
    echo "   ✅ 空间充足（剩余 $((200 - lines)) 行）"
fi

# 2. L2 行数
L2="$PROJECT_ROOT/CLAUDE.md"
echo ""
if [ -f "$L2" ]; then
    l2lines=$(wc -l < "$L2")
    echo "📏 L2 行数: $l2lines"
    if [ "$l2lines" -gt 280 ]; then
        echo "   ⚠️  过长，考虑下沉 L3"
    else
        echo "   ✅ 正常"
    fi
fi

# 3. 模式库条数
echo ""
echo "📚 模式库:"
pattern_count=$(awk '/^## 模式库/{found=1; next} /^## /{found=0} found && /^\*\*/' "$MEMORY" | wc -l)
echo "   共 $pattern_count 个模式"

# 4. 偏差条数
echo ""
echo "🧠 项目特有偏差:"
bias_count=$(awk '/^## 项目特有偏差/{found=1; next} /^## /{found=0} found && /^\*\*/' "$MEMORY" | wc -l)
echo "   共 $bias_count 条"

# 5. 导航链接检查
# 格式约定：导航块全部使用 [label](path)
#   - L1 私有文件以 memory/ 开头 → 解析到 $MEMORY_DIR/
#   - 其余路径 → 解析到 $PROJECT_ROOT/
echo ""
echo "🔗 导航链接:"
broken=0
while IFS= read -r link; do
    [ -z "$link" ] && continue
    if [[ "$link" == memory/* ]]; then
        check_path="$MEMORY_DIR/${link#memory/}"
    else
        check_path="$PROJECT_ROOT/$link"
    fi
    if [ -e "$check_path" ]; then
        echo "   ✅ $link"
    else
        echo "   ❌ 断链: $link"
        broken=$((broken + 1))
    fi
done < <(
    awk '/^## 导航/{found=1; next} /^## /{found=0} found' "$MEMORY" | \
    grep -oP '(?<=\]\()[^)]+' | grep -v '^$' || true
)
[ "$broken" -eq 0 ] && echo "   全部有效"

# 6. Journal 状态
echo ""
echo "📓 Journal (docs/log/):"
log_dir="$PROJECT_ROOT/docs/log"
if [ -d "$log_dir" ]; then
    day_count=$(find "$log_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$day_count" -eq 0 ]; then
        echo "   (空)"
    else
        echo "   最近 5 天:"
        find "$log_dir" -mindepth 1 -maxdepth 1 -type d -printf "   %f\n" | sort -r | head -5
        [ "$day_count" -gt 5 ] && echo "   ... 共 $day_count 天"

        # aging check：进行中 > 3 天
        echo ""
        echo "   🕒 Aging（进行中 > 3天）:"
        now=$(date +%s)
        aging_found=0
        while IFS= read -r f; do
            mtime=$(stat -c %Y "$f")
            age_days=$(( (now - mtime) / 86400 ))
            if [ "$age_days" -ge 3 ]; then
                echo "      ⚠️  $(basename "$(dirname "$f")")/$(basename "$f") (${age_days}d)"
                aging_found=1
            fi
        done < <(find "$log_dir" -name "*.md" -exec grep -l "状态: 进行中" {} + 2>/dev/null || true)
        [ "$aging_found" -eq 0 ] && echo "      ✅ 全部新鲜"
    fi

    # index.md 同步检查
    echo ""
    INDEX="$log_dir/index.md"
    if [ -f "$INDEX" ]; then
        # 找有 journal 条目但 index.md 没有对应日期的天
        unsynced=0
        while IFS= read -r day; do
            date_str=$(basename "$day")
            if ! grep -q "$date_str" "$INDEX" 2>/dev/null; then
                echo "   ⚠️  index.md 缺少 $date_str 的条目"
                unsynced=1
            fi
        done < <(find "$log_dir" -mindepth 1 -maxdepth 1 -type d | sort -r | head -10)
        [ "$unsynced" -eq 0 ] && echo "   ✅ index.md 与 journal 同步"
    else
        echo "   ⚠️  docs/log/index.md 不存在"
    fi
else
    echo "   ⚠️  docs/log/ 不存在"
fi

# 7. [确认]/[推测] 标注统计
echo ""
echo "🏷  标注统计（journal 最近 7 天）:"
confirm_count=0
guess_count=0
cutoff=$(date -d "7 days ago" +%Y-%m-%d)
while IFS= read -r f; do
    day=$(basename "$(dirname "$f")")
    [[ "$day" < "$cutoff" ]] && continue
    c=$(grep -c "\[确认\]" "$f" 2>/dev/null) || c=0
    g=$(grep -c "\[推测\]" "$f" 2>/dev/null) || g=0
    confirm_count=$((confirm_count + c))
    guess_count=$((guess_count + g))
done < <(find "$log_dir" -name "*.md" 2>/dev/null || true)
echo "   [确认] $confirm_count  [推测] $guess_count"
total=$((confirm_count + guess_count))
if [ "$total" -eq 0 ]; then
    echo "   ⚠️  近 7 天 journal 无标注，请在发现章节加 [确认]/[推测]"
fi

echo ""
echo "=== 完成 ==="
