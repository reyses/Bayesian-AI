#!/usr/bin/env python3
"""Owner-quiet nudge (systemd user timer tg-nudge, every 30 min).

Owner design (2026-07-24): between NUDGE_WINDOW hours, if the owner has been
quiet for QUIET_S while something is PENDING ON THEM, send one "what's wrong /
still waiting on X" ping. A quick Sonnet composes the ping with live context;
a plain template is the fallback so the nudge never silently fails.

The session keeps state/pending_for_owner.json current:
    {"items": ["ratify entry-head freeze", ...], "since": <epoch>}
Empty items (or missing file) = nothing pending = never nudge. The session
MUST clear items once the owner answers, or this will nag about stale asks.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / "state"
PENDING_F = STATE / "pending_for_owner.json"
INBOX = STATE / "inbox.jsonl"
LAST_NUDGE_F = STATE / "last_nudge.txt"

NUDGE_WINDOW = (6, 22)     # local hours [start, end): owner's waking hours
QUIET_S = 3 * 3600         # owner silent this long -> eligible
NUDGE_COOLDOWN_S = 6 * 3600  # at most one nudge per 6h
SONNET_TIMEOUT_S = 120
CLAUDE_BIN = "/home/moi/.local/bin/claude"

load_dotenv(HERE / ".env")
load_dotenv(REPO / ".env")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def last_owner_activity():
    try:
        lines = INBOX.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[-1])["ts"] if lines else 0
    except Exception:
        return 0


def compose_with_sonnet(items):
    prompt = (
        "Compose a SHORT (max 4 lines) friendly Telegram check-in to the owner "
        "of this trading-research repo. They have been quiet a few hours while "
        f"these decisions wait on them: {items}. Ask if everything is OK and "
        "restate the pending items in one compact line each. No preamble, no "
        "markdown headers — output ONLY the message text.")
    try:
        r = subprocess.run(
            ["timeout", str(SONNET_TIMEOUT_S), CLAUDE_BIN, "-p", prompt,
             "--model", "claude-sonnet-5", "--output-format", "json"],
            cwd=str(REPO), capture_output=True, text=True,
            timeout=SONNET_TIMEOUT_S + 10)
        txt = json.loads(r.stdout).get("result", "").strip()
        if 0 < len(txt) < 1500:
            return txt
    except Exception:
        pass
    return None


def inbound_deep_check():
    """Owner design (2026-07-24): a nudge with no reply is ambiguous — busy
    owner, or owner DID reply into a broken receive leg. Check + self-heal the
    three detectable inbound killers; return list of anomalies found/fixed."""
    fixed = []
    # (a) webhook set on the token kills getUpdates entirely
    try:
        info = requests.get(f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo",
                            timeout=30).json().get("result", {})
        if info.get("url"):
            requests.get(f"https://api.telegram.org/bot{TOKEN}/deleteWebhook",
                         timeout=30)
            fixed.append(f"a webhook was set ({info['url'][:40]}) — deleted it")
    except Exception:
        pass
    # (b) competing poller stealing updates (daemon writes marker on 409)
    conflict_f = STATE / "conflict409.txt"
    try:
        if time.time() - float(conflict_f.read_text().strip()) < 1800:
            subprocess.run(["pkill", "-f", "telegram_bridge/bridge.py"])
            subprocess.run(["pkill", "-f", "telegram_mcp.py"])
            subprocess.run(["systemctl", "--user", "restart", "tg-ingress.service"])
            fixed.append("another poller was stealing messages (409) — killed "
                         "stale pollers, restarted ingress")
    except (FileNotFoundError, ValueError):
        pass
    # (c) daemon not polling at all
    hb = STATE / "heartbeat.txt"
    if not hb.exists() or time.time() - hb.stat().st_mtime > 180:
        subprocess.run(["systemctl", "--user", "restart", "tg-ingress.service"])
        fixed.append("daemon was not polling — restarted it")
    return fixed


def post_nudge_escalation():
    """If the last nudge got no owner reply within one timer cycle, verify the
    owner CAN reply (inbound path), self-heal, and follow up once."""
    esc_f = STATE / "nudge_escalated.txt"
    try:
        last_nudge = float(LAST_NUDGE_F.read_text().strip())
    except Exception:
        return
    if last_owner_activity() > last_nudge:
        esc_f.unlink(missing_ok=True)          # owner replied — cycle closed
        return
    if time.time() - last_nudge < 1800:
        return                                  # give them one cycle to answer
    try:
        if float(esc_f.read_text().strip()) >= last_nudge:
            return                              # already escalated this nudge
    except Exception:
        pass
    fixed = inbound_deep_check()
    if fixed:
        text = ("🔧 No reply to my last check-in, so I verified the RECEIVE "
                "path and found+fixed:\n" + "\n".join(f"• {f}" for f in fixed)
                + "\nIf you replied earlier, it may have been lost — please resend.")
    else:
        text = ("ℹ️ No reply to my check-in — receive path verified CLEAN "
                "(polling live, no webhook, no competing poller). If you DID "
                "reply and your message has no 👍 on it, send /health; if even "
                "that gets no answer, the loop is down and I'm wrong. "
                "Otherwise ignore me — the pending items keep.")
    if TOKEN and CHAT_ID:
        requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                     params={"chat_id": CHAT_ID, "text": text}, timeout=30)
    esc_f.write_text(str(time.time()))
    print(f"post-nudge escalation: fixed={fixed or 'none, path clean'}")


def main():
    h = time.localtime().tm_hour
    if not (NUDGE_WINDOW[0] <= h < NUDGE_WINDOW[1]):
        return
    post_nudge_escalation()
    try:
        pending = json.loads(PENDING_F.read_text())
        items = pending.get("items") or []
    except Exception:
        return
    if not items:
        return
    if time.time() - last_owner_activity() < QUIET_S:
        return
    try:
        if time.time() - float(LAST_NUDGE_F.read_text().strip()) < NUDGE_COOLDOWN_S:
            return
    except Exception:
        pass
    text = compose_with_sonnet(items) or (
        "👋 Quiet for a while — everything OK? Still waiting on you for:\n"
        + "\n".join(f"• {i}" for i in items)
        + "\n(/health for loop status; reply here anytime)")
    if TOKEN and CHAT_ID:
        requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                     params={"chat_id": CHAT_ID, "text": text}, timeout=30)
        LAST_NUDGE_F.write_text(str(time.time()))
        print(f"nudged: {len(items)} pending items")


if __name__ == "__main__":
    main()
