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
        ["pgrep", "-f", "wait_inbox.py"], capture_output=True).returncode == 0
    if watcher_alive:
        return                                        # session listening — just busy
    try:
        last = float(LAST_ALERT_F.read_text().strip())
    except Exception:
        last = 0.0
    if time.time() - last < ALERT_COOLDOWN_S:
        return                                        # already alerted recently
    n_pending = len(lines) - n_consumed
    requests.get(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        params={"chat_id": CHAT_ID, "text":
                f"⚠️ Watchdog: {n_pending} message(s) queued but NO live Claude "
                f"session is listening. They are safe on disk and will be "
                f"delivered when a session re-arms the watcher. To get answers "
                f"now: open the session, or run bridge.py."},
        timeout=30)
    LAST_ALERT_F.write_text(str(time.time()))
    print(f"alerted: {n_pending} pending, no watcher")

if __name__ == "__main__":
    main()
