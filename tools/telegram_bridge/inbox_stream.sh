#!/bin/bash
# Persistent inbox consumer for a live Claude session (v2, 2026-07-24).
# v1 postmortem: inline python with escaped quotes SyntaxError'd on every
# message while the counter still advanced — messages consumed into the void.
# v2: emission via emit_msg.py (a real file, no quoting), and the counter
# advances ONLY when emission succeeds. If emission keeps failing, the message
# stays pending and the tg-watchdog alerts the owner instead of silence.
HERE="$(dirname "$0")"
STATE="$HERE/state"
INBOX="$STATE/inbox.jsonl"
LOCK="$STATE/consumed.lock"
CFILE="$STATE/consumed.txt"

while true; do
  N=$(wc -l < "$INBOX" 2>/dev/null || echo 0)
  C=$(cat "$CFILE" 2>/dev/null || echo 0)
  if [ "$N" -gt "$C" ]; then
    (
      flock 9
      C2=$(cat "$CFILE" 2>/dev/null || echo 0)
      if [ "$N" -gt "$C2" ]; then
        if sed -n "$((C2+1))p" "$INBOX" | python3 "$HERE/emit_msg.py"; then
          echo $((C2+1)) > "$CFILE"
        else
          sleep 10   # emission failed: do NOT consume; retry after backoff
        fi
      fi
    ) 9>"$LOCK"
  else
    sleep 2
  fi
done
