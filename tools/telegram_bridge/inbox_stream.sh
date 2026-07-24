#!/bin/bash
# Persistent inbox consumer for a live Claude session (2026-07-24).
# Replaces one-shot wait_inbox.py arming: runs as a session-lifetime Monitor,
# emits each new inbox message as an event, and advances consumed.txt under
# flock. Never needs re-arming — the re-arm-discipline failure mode (two
# bridge "collapses" on 07-24) is designed out.
# wait_inbox.py remains for one-shot use in scripts; do not run both at once.
STATE="$(dirname "$0")/state"
INBOX="$STATE/inbox.jsonl"
LOCK="$STATE/consumed.lock"
CFILE="$STATE/consumed.txt"

while true; do
  N=$(wc -l < "$INBOX" 2>/dev/null || echo 0)
  C=$(cat "$CFILE" 2>/dev/null || echo 0)
  if [ "$N" -gt "$C" ]; then
    # emit then advance, one message at a time, atomically vs other consumers
    (
      flock 9
      C2=$(cat "$CFILE" 2>/dev/null || echo 0)   # re-read under lock
      if [ "$N" -gt "$C2" ]; then
        sed -n "$((C2+1))p" "$INBOX" | python3 -c '
import sys, json
d = json.loads(sys.stdin.readline())
files = " files=" + ",".join(d["files"]) if d.get("files") else ""
print(f"TELEGRAM_MSG:{d[\"text\"]}{files}", flush=True)'
        echo $((C2+1)) > "$CFILE"
      fi
    ) 9>"$LOCK"
  else
    sleep 2
  fi
done
