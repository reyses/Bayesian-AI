#!/usr/bin/env python3
"""Session-side inbox watcher (replaces the direct poll_one.py loop).

Blocks until state/inbox.jsonl (fed by the tg-ingress daemon) has a line this
session hasn't consumed, prints it as NEW_MESSAGE:<text> (+ FILES: line when
attachments came with it), advances state/consumed.txt, exits 0. Run as a
Claude Code background task: exit re-wakes Claude in the live session.

Never touches the Telegram API — the daemon owns getUpdates exclusively.
If the session dies, messages simply accumulate in the inbox (nothing lost)
and watchdog.py alerts the phone.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
INBOX = STATE / "inbox.jsonl"
CONSUMED_F = STATE / "consumed.txt"

def consumed():
    try:
        return int(CONSUMED_F.read_text().strip())
    except Exception:
        return 0

def main():
    STATE.mkdir(exist_ok=True)
    # v2 (2026-07-24): consumption is ATOMIC (flock) and MONOTONIC — the counter
    # was found rewound to 2/80, causing full-inbox replays to the session.
    # Racing watcher instances now serialize on the lock, and a stale writer can
    # never move the counter backwards.
    import fcntl
    lock_path = STATE / "consumed.lock"
    while True:
        try:
            with open(lock_path, "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                lines = INBOX.read_text(encoding="utf-8").splitlines() if INBOX.exists() else []
                n = consumed()
                if len(lines) > n:
                    entry = json.loads(lines[n])
                    nxt = n + 1
                    if nxt > n:                       # monotonic guard
                        CONSUMED_F.write_text(str(nxt))
                    fcntl.flock(lk, fcntl.LOCK_UN)
                    print(f"NEW_MESSAGE:{entry.get('text','')}")
                    if entry.get("files"):
                        print("FILES:" + ";".join(entry["files"]))
                    sys.exit(0)
                fcntl.flock(lk, fcntl.LOCK_UN)
        except Exception as e:
            print(f"watch error: {e!r}", file=sys.stderr)
        time.sleep(1)

if __name__ == "__main__":
    main()
