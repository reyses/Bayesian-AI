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
    if cmd in ("/health", "/status"):
        # On-demand full-loop diagnostic, answered by the daemon itself so it
        # works even when no Claude session is alive (owner request 2026-07-24
        # after the dead-consumer incident).
        import subprocess
        try:
            n_inbox = sum(1 for _ in open(STATE / "inbox.jsonl", encoding="utf-8"))
        except Exception:
            n_inbox = 0
        try:
            n_consumed = int((STATE / "consumed.txt").read_text().strip())
        except Exception:
            n_consumed = 0
        watcher = subprocess.run(["pgrep", "-f", "wait_inbox.py"],
                                 capture_output=True).returncode == 0
        try:
            hb_age = int(time.time() - (STATE / "heartbeat.txt").stat().st_mtime)
        except Exception:
            hb_age = -1
        pending = max(0, n_inbox - n_consumed)
        verdict = ("✅ loop healthy" if watcher and pending == 0
                   else "⚠️ messages waiting, session listening" if watcher
                   else "🔴 NO session listening — replies will stall")
        reply = (f"{verdict}\ndaemon: alive (this reply proves polling)\n"
                 f"heartbeat age: {hb_age}s\nqueue: {pending} pending "
                 f"({n_consumed}/{n_inbox} consumed)\n"
                 f"session watcher: {'armed' if watcher else 'ABSENT'}")
        try:
            requests.get(f"{API}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": reply}, timeout=10)
        except Exception:
            pass
        return True
    if cmd in ("/fix", "/restart"):
        # Owner one-button repair (2026-07-24): mechanically repair every leg
        # the daemon can reach, report what it did. The one leg it cannot
        # revive is a dead Claude session — for that it states so plainly.
        import subprocess
        did = []
        r = subprocess.run(["systemctl", "--user", "restart",
                            "tg-verify.timer", "tg-watchdog.timer"],
                           capture_output=True, text=True)
        did.append("timers: " + ("restarted" if r.returncode == 0 else f"FAILED {r.stderr[:80]}"))
        watcher = subprocess.run(["pgrep", "-f", "wait_inbox|inbox_stream"],
                                 capture_output=True).returncode == 0
        try:
            n_inbox = sum(1 for _ in open(STATE / "inbox.jsonl", encoding="utf-8"))
            n_consumed = int((STATE / "consumed.txt").read_text().strip())
        except Exception:
            n_inbox = n_consumed = 0
        pending = max(0, n_inbox - n_consumed)
        if watcher:
            did.append(f"session consumer: alive ({pending} pending will be delivered)")
        else:
            # dead session -> spawn the independent fallback Sonnet right now
            # (owner one-button repair; same spawner the watchdog uses)
            try:
                import watchdog as wd
                spawned = wd.spawn_fallback()
                did.append(f"session consumer: DEAD ({pending} pending) — "
                           + ("independent fallback Sonnet SPAWNED; it will "
                              "answer here shortly."
                              if spawned else
                              "fallback Sonnet already running on it."))
            except Exception as e:
                did.append(f"session consumer: DEAD; fallback spawn failed "
                           f"({e!r:.60}) — /wake to open VS Code instead.")
        if cmd == "/restart":
            did.append("daemon: restarting itself now (systemd brings it back in ~10s)")
        reply = "🔧 /fix report:\n" + "\n".join(f"• {d}" for d in did)
        try:
            requests.get(f"{API}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": reply}, timeout=10)
        except Exception:
            pass
        if cmd == "/restart":
            # exit non-zero; Restart=always relaunches with persisted offset
            os._exit(1)
        return True
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
            if not info.get("ok"):
                # Bot API getFile hard-caps at 20MB — surface the failure loudly
                # instead of silently dropping the message (owner report 2026-07-23).
                paths.append(f"DOWNLOAD_FAILED({name}): {info.get('description','?')}")
                continue
            fp = info["result"]["file_path"]
            data = requests.get(f"{FILE_API}/{fp}", timeout=120).content
            out = DOWNLOADS / f"{int(time.time())}_{name}"
            out.write_bytes(data)
            paths.append(str(out))
        except Exception as e:
            print(f"attachment download failed: {e!r}", file=sys.stderr)
            paths.append(f"DOWNLOAD_FAILED({name}): {e!r:.80}")
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
            # liveness heartbeat: tg-verify alerts+restarts if this goes stale,
            # catching a daemon that is alive-as-a-process but not polling.
            (STATE / "heartbeat.txt").touch()
            if res.get("error_code") == 409:
                # another poller is stealing this token's updates — inbound is
                # BROKEN even though we look alive. Marker lets tg_nudge/verify
                # detect + self-heal (kill stale pollers) instead of guessing.
                (STATE / "conflict409.txt").write_text(str(time.time()))
                print("409 conflict: another getUpdates poller on this token",
                      file=sys.stderr)
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
                has_attachment = any(k in msg for k in ("document", "photo", "voice",
                                                        "video", "audio", "sticker"))
                if not text and not files and not has_attachment:
                    continue
                if has_attachment and not files:
                    # unsupported attachment type — queue a note, never silent-drop
                    files = [f"UNSUPPORTED_ATTACHMENT: keys={[k for k in msg if k not in ('chat','from','date','message_id')]}"]
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
