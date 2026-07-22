#!/usr/bin/env bash
# PostToolUse hook guard — keep the derived memory FTS DB fresh on every write.
#
# Fires after each Write/Edit. Reads the tool-call JSON on stdin, and ONLY when
# the edited file is a memory SOURCE (docs/memory/ or docs/daily/) does it rebuild
# docs/memory/memory.db. Every other edit pays just a jq+grep and exits — no cost.
#
# The DB is STRICTLY DERIVED and gitignored (see build_memory_db.py); this hook
# never writes memory content, it only regenerates the index from the markdown.
# It must never break the turn: all failure paths swallow errors and exit 0.
#
# Wired from .claude/settings.local.json (PostToolUse / matcher "Write|Edit").
# Interpreter is python3 because the hook runs in the Linux/WSL Claude Code env;
# the "python3.11" in the README refers to the separate Windows training env.

# repo root = two levels up from this script (tools/memory_loop/ -> repo root),
# so the build targets the right repo regardless of the hook's cwd.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# PostToolUse delivers the tool call as JSON on stdin; pull the target path.
path="$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)"

case "$path" in
  */docs/memory/*|docs/memory/*|*/docs/daily/*|docs/daily/*)
    # idempotent full rebuild (~2.5s). Never surface an error to the turn.
    python3 "$repo/tools/memory_loop/build_memory_db.py" --quiet >/dev/null 2>&1 || true
    ;;
esac

exit 0
