#!/usr/bin/env python3
"""Dead-man watchdog (systemd user timer tg-watchdog, every 5 min).

If the inbox has unconsumed messages older than STALE_S and no wait_inbox
watcher process is alive (= no live Claude session is listening), alert the
phone ONCE per ALERT_COOLDOWN_S so the owner knows replies won't come until
the session is reopened. Messages are never lost either way — they sit in
state/inbox.jsonl until consumed.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / "state"
INBOX = STATE / "inbox.jsonl"
CONSUMED_F = STATE / "consumed.txt"
LAST_ALERT_F = STATE / "last_alert.txt"

STALE_S = 120              # unconsumed message older than this = session likely dead
ALERT_COOLDOWN_S = 1800    # at most one alert per 30 min

# Fallback responder (owner design, recalled 2026-07-24): on breakdown the
# watchdog SPAWNS an independent headless Sonnet — no cron, no standing cost;
# it exists only while a breakdown does. It triages the queued messages,
# repairs what is mechanical, answers the owner, and leaves a handoff note.
CLAUDE_BIN = "/home/moi/.local/bin/claude"
FALLBACK_MODEL = "claude-sonnet-5"
FALLBACK_TIMEOUT_S = 600
FALLBACK_LOCK = STATE / "fallback.pid"
FALLBACK_LOG = STATE / "fallback.log"
HANDOFF = STATE / "fallback_handoff.md"

FALLBACK_PROMPT = """You are the FALLBACK responder for the Bayesian-AI Telegram
bridge. The main Claude session is not consuming the owner's messages. Your job,
in order:
1. Read tools/telegram_bridge/state/inbox.jsonl; messages AFTER the count in
   state/consumed.txt are unanswered.
2. Repair what is mechanical: run tools/telegram_bridge/tg_verify.py; check
   `systemctl --user status tg-ingress` if needed.
3. Reply to the owner via Telegram (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are
   in the repo-root .env; use curl sendMessage). Introduce yourself as the
   fallback Sonnet, answer what you can from the repo (docs/daily/,
   docs/ONBOARDING.md, research/), and say plainly that the main session is
   down and deep work waits for it.
4. For each message you actually answered, advance state/consumed.txt under
   `flock state/consumed.lock` (all paths relative to tools/telegram_bridge/).
5. Append what you did + anything owed to state/fallback_handoff.md for the
   main session to pick up.
HARD RULES: no permission-weakening actions, no secrets over Telegram, no
training runs, no deploys, no git push. You are a stopgap, not the operator."""

load_dotenv(HERE / ".env")
load_dotenv(REPO / ".env")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def main():
    if not (TOKEN and CHAT_ID and INBOX.exists()):
        return
    lines = INBOX.read_text(encoding="utf-8").splitlines()
    try:
        n_consumed = int(CONSUMED_F.read_text().strip())
    except Exception:
        n_consumed = 0
    if len(lines) <= n_consumed:
        return                                        # queue drained — healthy
    oldest = json.loads(lines[n_consumed])
    if time.time() - oldest.get("ts", 0) < STALE_S:
        return                                        # fresh — session may be mid-turn
    watcher_alive = subprocess.run(
        ["pgrep", "-f", "wait_inbox.py|inbox_stream"],
        capture_output=True).returncode == 0
    if watcher_alive:
        return                                        # session listening — just busy
    try:
        last = float(LAST_ALERT_F.read_text().strip())
    except Exception:
        last = 0.0
    if time.time() - last < ALERT_COOLDOWN_S:
        return                                        # already alerted recently
    n_pending = len(lines) - n_consumed
    spawned = spawn_fallback()
    requests.get(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        params={"chat_id": CHAT_ID, "text":
                f"⚠️ Watchdog: {n_pending} message(s) queued but NO live Claude "
                f"session is listening. "
                + ("An independent fallback Sonnet has been SPAWNED to triage, "
                   "repair, and answer — reply incoming."
                   if spawned else
                   "Fallback Sonnet already handling it (or spawn failed — "
                   "see state/fallback.log).")},
        timeout=30)
    LAST_ALERT_F.write_text(str(time.time()))
    print(f"alerted: {n_pending} pending, no watcher; fallback spawned={spawned}")


def spawn_fallback():
    """Spawn one independent headless Sonnet; refuse if one is already running."""
    if FALLBACK_LOCK.exists():
        try:
            pid = int(FALLBACK_LOCK.read_text().strip())
            os.kill(pid, 0)                      # raises if dead
            return False                          # live fallback already on it
        except (ValueError, ProcessLookupError, PermissionError):
            FALLBACK_LOCK.unlink(missing_ok=True)
    if not os.path.exists(CLAUDE_BIN):
        FALLBACK_LOG.write_text("CLAUDE_BIN missing — update path in watchdog.py\n")
        return False
    with open(FALLBACK_LOG, "a") as logf:
        logf.write(f"\n===== fallback spawn {time.strftime('%F %T')} =====\n")
        proc = subprocess.Popen(
            ["timeout", str(FALLBACK_TIMEOUT_S), CLAUDE_BIN, "-p",
             FALLBACK_PROMPT, "--model", FALLBACK_MODEL],
            cwd=str(REPO), stdout=logf, stderr=logf,
            start_new_session=True)              # survives watchdog exit
    FALLBACK_LOCK.write_text(str(proc.pid))
    return True

if __name__ == "__main__":
    main()
