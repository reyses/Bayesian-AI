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
    parts = text.strip().split(None, 1)
    verb = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if verb in ("/cli", "/agy"):
        # One-shot AI answer from the phone: /cli <q> -> Sonnet, /agy <q> ->
        # Antigravity. Runs DETACHED via run_cli.py (a blocking call here
        # would stall the poll loop); that process replies to Telegram itself.
        import subprocess
        reply = None
        if not arg:
            reply = f"usage: {verb} <question>"
        else:
            provider = "sonnet" if verb == "/cli" else "agy"
            try:
                subprocess.Popen(
                    [sys.executable, str(HERE / "run_cli.py"), provider, arg],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                reply = f"⏳ {provider} working on it — answer follows here."
            except OSError as e:
                reply = f"spawn failed: {e!r:.80}"
        try:
            requests.get(f"{API}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": reply}, timeout=10)
        except Exception:
            pass
        return True
    if verb in ("/memq", "/memaudit", "/memstats"):
        # Teacher memory-bank query + audit from the phone (owner 2026-07-24).
        # Read-only by construction (sqlite URI mode=ro): these commands can
        # never create, mutate, or lock the bank a live run is writing to.
        import sqlite3
        db = str(REPO / "research/dojo_forge/gate_state/teacher_memory.db")
        ledger = REPO / "research/dojo_forge/gate_state/memory_ledger.jsonl"
        reply = ""
        if verb == "/memq":
            if not os.path.exists(db):
                reply = "memory bank empty — the memory pilot hasn't run yet."
            elif not arg:
                reply = "usage: /memq <search terms>  (FTS over qwen's memos)"
            else:
                try:
                    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    rows = con.execute(
                        "SELECT m.day, m.minute, m.text FROM memos_fts "
                        "JOIN memos m ON m.id = memos_fts.rowid "
                        "WHERE memos_fts MATCH ? "
                        "ORDER BY bm25(memos_fts) ASC, m.id ASC LIMIT 5",
                        (arg,)).fetchall()
                    con.close()
                    reply = ("\n———\n".join(
                        f"[{d} m{m}] {t[:400]}" for d, m, t in rows)
                        or f"no memos match '{arg}'")
                except sqlite3.Error as e:
                    reply = f"memq error: {e}"
        elif verb == "/memaudit":
            if not ledger.exists():
                reply = "ledger empty — no memory events recorded yet."
            else:
                try:
                    n = max(1, min(int(arg), 30)) if arg.strip().isdigit() else 8
                    events = ledger.read_text(encoding="utf-8").splitlines()[-n:]
                    out = []
                    for ln in events:
                        try:
                            e = json.loads(ln)
                        except ValueError:
                            continue
                        ev = e.get("event", "?")
                        if ev == "write_admitted":
                            out.append(f"✍️ WRITE {e.get('day')} m{e.get('minute')} "
                                       f"id={e.get('memo_id')} tags={e.get('tags')}")
                        elif ev == "write_rejected":
                            out.append(f"🚫 REJECTED {e.get('day')} m{e.get('minute')} "
                                       f"({e.get('reason')})")
                        elif ev == "retrieve":
                            out.append(f"🔍 RETRIEVE ep-day={e.get('episode_day')} "
                                       f"q='{str(e.get('query'))[:60]}' "
                                       f"granted={e.get('granted_ids')}")
                        else:
                            out.append(f"• {ev}: {str(e)[:90]}")
                    reply = "\n".join(out) or "ledger unparseable"
                except OSError as e:
                    reply = f"memaudit error: {e}"
        else:  # /memstats
            if not os.path.exists(db):
                reply = "memory bank empty — the memory pilot hasn't run yet."
            else:
                try:
                    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    total = con.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
                    per_day = con.execute(
                        "SELECT day, COUNT(*) FROM memos GROUP BY day "
                        "ORDER BY day").fetchall()
                    con.close()
                    counts = {}
                    if ledger.exists():
                        for ln in ledger.read_text(encoding="utf-8").splitlines():
                            try:
                                ev = json.loads(ln).get("event", "?")
                            except ValueError:
                                continue
                            counts[ev] = counts.get(ev, 0) + 1
                    reply = (f"memos: {total} total\n"
                             + "\n".join(f"  {d}: {c}" for d, c in per_day)
                             + "\nledger: "
                             + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
                except sqlite3.Error as e:
                    reply = f"memstats error: {e}"
        try:
            requests.get(f"{API}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": reply[:4000]},
                         timeout=10)
        except Exception:
            pass
        return True
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

        def active(unit):
            return subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", unit]
            ).returncode == 0

        did = []
        # diagnose -> fix -> verify, per leg (owner 2026-07-24: /fix must
        # SHOW the diagnosis, not just claim repairs)
        for unit in ("tg-verify.timer", "tg-watchdog.timer",
                     "ag-sentinel.timer"):
            before = active(unit)
            if not before:
                subprocess.run(["systemctl", "--user", "restart", unit],
                               capture_output=True)
                after = active(unit)
                did.append(f"{unit}: was DOWN -> "
                           + ("restarted OK" if after else "RESTART FAILED"))
            else:
                did.append(f"{unit}: healthy")
        try:
            hb_age = int(time.time() - (STATE / "heartbeat.txt").stat().st_mtime)
            did.append(f"poller heartbeat: {hb_age}s ago"
                       + ("" if hb_age < 180 else " (STALE)"))
        except OSError:
            did.append("poller heartbeat: MISSING")
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
            _ = drill_broken_reference  # PHASE-2 DRILL: intentional NameError
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
