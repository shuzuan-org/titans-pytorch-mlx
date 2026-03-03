#!/bin/bash
# Memory system health check — covers all layers including L1 private files.
# Run manually: bash scripts/check_memory.sh

FAIL=0
WARN=0

red()    { echo -e "\033[31m❌  $*\033[0m"; }
yellow() { echo -e "\033[33m⚠️   $*\033[0m"; }
green()  { echo -e "\033[32m✓   $*\033[0m"; }

# Derive L1 path from the current git root — works on any machine or username.
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$GIT_ROOT" ]; then
    red "Not inside a git repository — run from the project root"
    exit 1
fi
L1_KEY=$(echo "$GIT_ROOT" | sed 's|/|-|g')
L1_DIR="$HOME/.claude/projects/${L1_KEY}/memory"

echo "── L1 (private memory: $L1_DIR) ──────────────"

# MEMORY.md must stay under 200 lines (auto-inject truncates there)
if [ -f "$L1_DIR/MEMORY.md" ]; then
    LINES=$(wc -l < "$L1_DIR/MEMORY.md")
    if [ "$LINES" -gt 200 ]; then
        red "MEMORY.md is ${LINES} lines — auto-inject truncates at 200"
        FAIL=1
    else
        green "MEMORY.md: ${LINES}/200 lines"
    fi
else
    red "MEMORY.md not found at $L1_DIR"
    FAIL=1
fi

# decisions.md must not contain "未实施" in decision bodies
if [ -f "$L1_DIR/decisions.md" ]; then
    if grep -v '^[>*#]' "$L1_DIR/decisions.md" | grep -q '未实施'; then
        red "decisions.md contains '未实施' in a decision body — unimplemented plans belong in MEMORY.md"
        FAIL=1
    else
        green "decisions.md: no unimplemented plans"
    fi
fi

# MEMORY.md must not directly reference docs/log/ (L1 → L2 only, not L1 → L3)
if [ -f "$L1_DIR/MEMORY.md" ]; then
    if grep -q 'docs/log/' "$L1_DIR/MEMORY.md"; then
        yellow "MEMORY.md has direct L3 log references — route through L2 instead"
        WARN=1
    fi
fi

echo ""
echo "── L2 (CLAUDE.md) ────────────────────────────────────"

if [ -f "CLAUDE.md" ]; then
    LINES=$(wc -l < "CLAUDE.md")
    if [ "$LINES" -gt 300 ]; then
        yellow "CLAUDE.md is ${LINES} lines — consider moving reference material to docs/"
        WARN=1
    else
        green "CLAUDE.md: ${LINES} lines"
    fi

    if ! grep -q 'docs/log/index.md' CLAUDE.md; then
        red "CLAUDE.md does not reference docs/log/index.md"
        FAIL=1
    else
        green "CLAUDE.md → docs/log/index.md reference present"
    fi
else
    red "CLAUDE.md not found — run from project root"
    FAIL=1
fi

echo ""
echo "── L3 (docs/log) ─────────────────────────────────────"

INDEX="docs/log/index.md"

if [ ! -f "$INDEX" ]; then
    red "docs/log/index.md missing"
    FAIL=1
else
    MISSING=0
    for dir in docs/log/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
        [ -d "$dir" ] || continue
        entry_date=$(basename "$dir")
        if ! grep -q "$entry_date" "$INDEX"; then
            red "log/$entry_date/ exists but missing from index.md"
            MISSING=1
            FAIL=1
        fi
    done
    [ $MISSING -eq 0 ] && green "log/index.md covers all date directories"

    ENTRY_COUNT=$(find docs/log -name '*.md' ! -name 'index.md' | wc -l)
    DATE_COUNT=$(find docs/log -mindepth 1 -maxdepth 1 -type d | wc -l)
    green "log: ${DATE_COUNT} days, ${ENTRY_COUNT} event files"
fi

echo ""
echo "──────────────────────────────────────────────────────"
if [ $FAIL -eq 1 ]; then
    red "Health check FAILED"
    exit 1
elif [ $WARN -eq 1 ]; then
    yellow "Health check passed with warnings"
    exit 0
else
    green "All checks passed"
    exit 0
fi
