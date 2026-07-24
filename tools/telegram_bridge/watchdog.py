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
        ATTEMPT_F.unlink(missing_ok=True)             # breakdown over — reset
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


GEMINI_BIN = "/home/moi/.npm-global/bin/gemini"
ATTEMPT_F = STATE / "fallback_attempt.txt"

# Second-provider prompt (owner 2026-07-24: "fallback on antigravity cli so it
# is failproof"). Deliberately NARROWER than the Claude prompt: acknowledge +
# triage only, no repairs — gemini -p has no project-permission layer, so the
# prompt is the only scope control. Provider order: Claude first (permissioned,
# full repair), Gemini second (acknowledge-only), template alert always.
GEMINI_PROMPT = """You are an emergency responder for this repo's Telegram
bridge; the primary AI is unreachable. Do ONLY this: (1) read
tools/telegram_bridge/state/inbox.jsonl — messages after the count in
tools/telegram_bridge/state/consumed.txt are unanswered; (2) send the owner ONE
Telegram message via curl using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from
./.env: introduce yourself as the Gemini emergency fallback, list their
unanswered messages back to them, and say the primary AI is down so answers
wait for it. (3) Append a note of what you sent to
tools/telegram_bridge/state/fallback_handoff.md. Do NOT modify anything else,
do NOT advance consumed.txt, do NOT run repairs."""


def _alive(pidfile):
    try:
        os.kill(int(pidfile.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
        return False


def spawn_fallback():
    """Failover chain: Claude Sonnet -> Gemini CLI -> (caller's template alert).
    One live fallback at a time; provider escalates per breakdown episode and
    resets when the queue drains (main() only calls this while pending>0)."""
    if FALLBACK_LOCK.exists() and _alive(FALLBACK_LOCK):
        return False                              # live fallback already on it
    FALLBACK_LOCK.unlink(missing_ok=True)
    try:
        attempt = int(ATTEMPT_F.read_text().strip())
    except Exception:
        attempt = 0
    providers = []
    if os.path.exists(CLAUDE_BIN):
        providers.append(("claude", ["timeout", str(FALLBACK_TIMEOUT_S),
                                     CLAUDE_BIN, "-p", FALLBACK_PROMPT,
                                     "--model", FALLBACK_MODEL]))
    if os.path.exists(GEMINI_BIN) and (
            os.environ.get("GEMINI_API_KEY")
            or (Path.home() / ".gemini" / "oauth_creds.json").exists()):
        providers.append(("gemini", ["timeout", str(FALLBACK_TIMEOUT_S),
                                     GEMINI_BIN, "-p", GEMINI_PROMPT,
                                     "--approval-mode", "yolo"]))
    if attempt >= len(providers):
        return False                              # chain exhausted -> template alert
    name, cmd = providers[attempt]
    with open(FALLBACK_LOG, "a") as logf:
        logf.write(f"\n===== fallback spawn [{name}] {time.strftime('%F %T')} =====\n")
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=logf, stderr=logf,
                                start_new_session=True)  # survives watchdog exit
    FALLBACK_LOCK.write_text(str(proc.pid))
    ATTEMPT_F.write_text(str(attempt + 1))
    return True

if __name__ == "__main__":
    main()
