#!/usr/bin/env python3
"""Read ONE inbox.jsonl line on stdin, print the session-facing event line.
Exists as a real file because inline `python3 -c` inside the bash stream script
hit quoting hell (\" inside single quotes -> SyntaxError) and silently ate
messages on 2026-07-24. Exit 0 only on successful emission — the stream script
advances the consumed counter ONLY on success."""
import json
import sys

try:
    d = json.loads(sys.stdin.readline())
    files = " files=" + ",".join(d["files"]) if d.get("files") else ""
    print("TELEGRAM_MSG:" + d["text"] + files, flush=True)
except Exception as e:  # noqa: BLE001 — any failure must NOT consume the message
    print("emit failed: " + repr(e), file=sys.stderr)
    sys.exit(1)
