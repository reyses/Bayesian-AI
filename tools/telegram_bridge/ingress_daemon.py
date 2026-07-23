#!/usr/bin/env python3
"""Always-on Telegram ingress daemon (systemd user service tg-ingress).

THE ONLY process allowed to call getUpdates (two pollers on one bot token steal
each other's updates via the offset — that's why the fallback is a disk queue,
not a second poller). Captures every message from TELEGRAM_CHAT_ID to
state/inbox.jsonl and downloads attachments to state/downloads/ (the
phone->repo half of bidirectional file sharing; repo->phone is sendDocument).

Survives session death: messages queue on disk until a consumer reads them.
Consumers: tools/telegram_bridge/wait_inbox.py (live Claude session watcher).
Watchdog: tools/telegram_bridge/watchdog.py alerts if the queue goes stale.

Offset is persisted (state/tg_offset.txt) so restarts never replay or drop.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / "state"
DOWNLOADS = STATE / "downloads"
INBOX = STATE / "inbox.jsonl"
OFFSET_F = STATE / "tg_offset.txt"

load_dotenv(HERE / ".env")
load_dotenv(REPO / ".env")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API = f"https://api.telegram.org/bot{TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{TOKEN}"

def load_offset():
    try:
        return int(OFFSET_F.read_text().strip())
    except Exception:
        return None

def save_offset(v):
    OFFSET_F.write_text(str(v))

def ack_received(msg):
    """Owner rule (2026-07-22): ack every captured message so the phone knows it
    was received. Cleanest form = emoji reaction (adds NO chat line); fallback
    to a minimal '.' text if reactions are unavailable."""
    try:
        r = requests.post(f"{API}/setMessageReaction", json={
            "chat_id": msg["chat"]["id"], "message_id": msg["message_id"],
            "reaction": [{"type": "emoji", "emoji": "👍"}]}, timeout=10)
        if r.json().get("ok"):
            return
    except Exception:
        pass
    try:
        requests.get(f"{API}/sendMessage",
                     params={"chat_id": CHAT_ID, "text": "."}, timeout=10)
    except Exception:
        pass

def handle_command(text, msg):
    """Owner remote-control commands (2026-07-22): handled by the daemon itself,
    never queued to the session. /wake opens VS Code with the repo on the
    desktop (GUI env injected — the daemon runs headless under systemd)."""
    cmd = text.strip().lower()
    if cmd in ("/wake", "/vscode", "/open"):
        env = dict(os.environ,
                   DISPLAY=":0", WAYLAND_DISPLAY="wayland-0",
                   XDG_RUNTIME_DIR=f"/run/user/{os.getuid()}")
        try:
            import subprocess
            subprocess.Popen(["snap", "run", "code", str(REPO)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            reply = ("🖥️ VS Code launching with the repo on the desktop. "
                     "Fully remote instead? vscode.dev/tunnel/rxmoi-bayesian")
        except Exception as e:
            reply = f"wake failed: {e!r} — use vscode.dev/tunnel/rxmoi-bayesian"
        try:
            requests.get(f"{API}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": reply}, timeout=10)
        except Exception:
            pass
        return True
    return False

def download_attachment(msg):
    """Download document/photo/voice to state/downloads/; return local paths."""
    paths = []
    candidates = []
    if "document" in msg:
        candidates.append((msg["document"]["file_id"], msg["document"].get("file_name", "document")))
    if "photo" in msg:                      # largest size is last
        candidates.append((msg["photo"][-1]["file_id"], f"photo_{msg['message_id']}.jpg"))
    if "voice" in msg:
        candidates.append((msg["voice"]["file_id"], f"voice_{msg['message_id']}.ogg"))
    for file_id, name in candidates:
        try:
            info = requests.get(f"{API}/getFile", params={"file_id": file_id}, timeout=30).json()
            fp = info["result"]["file_path"]
            data = requests.get(f"{FILE_API}/{fp}", timeout=120).content
            out = DOWNLOADS / f"{int(time.time())}_{name}"
            out.write_bytes(data)
            paths.append(str(out))
        except Exception as e:
            print(f"attachment download failed: {e!r}", file=sys.stderr)
    return paths

def main():
    if not TOKEN:
        print("No TELEGRAM_BOT_TOKEN", file=sys.stderr)
        sys.exit(1)
    STATE.mkdir(exist_ok=True)
    DOWNLOADS.mkdir(exist_ok=True)
    offset = load_offset()
    print(f"tg-ingress up; offset={offset}", flush=True)
    while True:
        try:
            params = {"timeout": 50}
            if offset is not None:
                params["offset"] = offset
            res = requests.get(f"{API}/getUpdates", params=params, timeout=60).json()
            if not res.get("ok"):
                time.sleep(3)
                continue
            for upd in res["result"]:
                offset = upd["update_id"] + 1
                save_offset(offset)
                msg = upd.get("message") or {}
                sender = str(msg.get("chat", {}).get("id", ""))
                if CHAT_ID and sender != CHAT_ID:
                    print(f"ignored sender={sender}", file=sys.stderr, flush=True)
                    continue
                text = msg.get("text") or msg.get("caption") or ""
                if text and handle_command(text, msg):
                    ack_received(msg)
                    continue                     # command handled; not queued
                files = download_attachment(msg)
                if not text and not files:
                    continue
                entry = {"ts": int(time.time()), "update_id": upd["update_id"],
                         "text": text, "files": files}
                with INBOX.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
                ack_received(msg)
                print(f"queued: {text[:60]!r} files={len(files)}", flush=True)
        except Exception as e:
            print(f"poll error (retrying): {e!r}", file=sys.stderr, flush=True)
            time.sleep(3)

if __name__ == "__main__":
    main()
