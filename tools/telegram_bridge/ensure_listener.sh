#!/usr/bin/env bash
# Stop-hook: the session may not go idle while the Telegram channel is deaf.
#
# WHY (owner, 2026-08-02: "we need to fix the constant disconnect from the main
# session" / "telegram was off again"): four separate disconnects in two days,
# every one the same root cause — the one-shot wait_inbox listener died (turn
# forgot to re-arm, pkill self-match, operator agent's loop lapsed) and nothing
# noticed, because watchdog.py only alerts on UNCONSUMED mail, not on a missing
# listener. This hook closes the gap at the structural level: every time the
# main session tries to finish a turn, it verifies a listener exists.
#
# Exit 0  -> allow stop  (a wait_inbox is running, or the bridge is paused)
# Exit 2  -> block stop; stderr tells Claude exactly what to do.
#
# Pause switch (owner at the IDE, TG deliberately idle):
#   touch tools/telegram_bridge/state/bridge_paused   # hook stands down
#   rm    tools/telegram_bridge/state/bridge_paused   # hook re-engages

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$HERE/state/bridge_paused" ]; then
    exit 0
fi

if pgrep -f "wait_inbox[.]py" >/dev/null 2>&1; then
    exit 0
fi

echo "TELEGRAM LISTENER IS DOWN (Stop hook tools/telegram_bridge/ensure_listener.sh). The owner is on Telegram and cannot reach this session. Before finishing this turn: launch tools/telegram_bridge/wait_inbox.py with the Bash tool and run_in_background:true (never shell '&', exactly one instance). If the bridge is intentionally idle, create tools/telegram_bridge/state/bridge_paused instead." >&2
exit 2
