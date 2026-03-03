#!/bin/bash
# Called by Claude Code PostToolUse hook after every Write/Edit.
# Warns if a log entry was written but today's date is not yet in index.md.
#
# Claude Code passes the file path via CLAUDE_TOOL_INPUT_FILE_PATH.
# Exits 0 always (this is advisory, not a hard gate).

FILE="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

# Only act on files inside docs/log/YYYY-MM-DD/
if echo "$FILE" | grep -qE 'docs/log/[0-9]{4}-[0-9]{2}-[0-9]{2}/.+'; then
    entry_date=$(echo "$FILE" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
    index="docs/log/index.md"

    if [ -n "$entry_date" ] && ! grep -qF "$entry_date" "$index" 2>/dev/null; then
        echo "⚠️  MEMORY: $index has no entry for $entry_date — update it before committing" >&2
    fi
fi

exit 0
