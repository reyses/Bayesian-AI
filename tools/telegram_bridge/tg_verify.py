#!/usr/bin/env python3
"""Third-party loop verifier (systemd user timer tg-verify, every 2 min).

Independent of both the ingress daemon and the session watchdog. Verifies the
DAEMON side of the loop mechanically and SELF-HEALS where it safely can:

  1. tg-ingress service active?           -> if not: restart it, alert phone.
  2. daemon actually polling? (heartbeat) -> if stale >HEARTBEAT_STALE_S:
                                             restart service, alert phone.
Every check appends one verdict line to state/health.log so there is a
diagnostic record ("was telegram closed accidentally?") the owner can read.

Consumer-side staleness (messages queued, no session listening) stays with
watchdog.py — one owner-alert per failure class, no double alarms.

Restarting tg-ingress is safe: offset is persisted, messages are never lost,
and getUpdates single-poller discipline is preserved (same service instance).
"""
import os
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / "state"
HEARTBEAT = STATE / "heartbeat.txt"
HEALTH_LOG = STATE / "health.log"
LAST_VERIFY_ALERT = STATE / "last_verify_alert.txt"

HEARTBEAT_STALE_S = 180    # daemon long-polls at 50s; 3 missed cycles = dead
ALERT_COOLDOWN_S = 1800    # at most one phone alert per 30 min per this leg

load_dotenv(HERE / ".env")
load_dotenv(REPO / ".env")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def log(line):
    with open(HEALTH_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


def alert(text):
    try:
        last = float(LAST_VERIFY_ALERT.read_text().strip())
    except Exception:
        last = 0.0
    if time.time() - last < ALERT_COOLDOWN_S:
        return
    if TOKEN and CHAT_ID:
        try:
            requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                         params={"chat_id": CHAT_ID, "text": text}, timeout=30)
            LAST_VERIFY_ALERT.write_text(str(time.time()))
        except Exception as e:
            log(f"ALERT-SEND-FAILED {e!r}")
    # desktop side too, in case the phone leg is what died
    try:
        subprocess.run(["notify-send", "-u", "critical", "tg-verify", text],
                       env=dict(os.environ, DISPLAY=":0",
                                DBUS_SESSION_BUS_ADDRESS=
                                f"unix:path=/run/user/{os.getuid()}/bus"),
                       timeout=10)
    except Exception:
        pass


def service_active():
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "tg-ingress.service"]
    ).returncode == 0


def service_uptime_s():
    out = subprocess.run(
        ["systemctl", "--user", "show", "tg-ingress.service",
         "--property=ActiveEnterTimestampMonotonic", "--value"],
        capture_output=True, text=True).stdout.strip()
    try:
        started = int(out) / 1e6
        with open("/proc/uptime") as f:
            now = float(f.read().split()[0])
        return now - started
    except Exception:
        return 1e9


REPAIR_LOCK = STATE / "code_repair.pid"
CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")

REPAIR_PROMPT = (
    "INCIDENT: the Telegram ingress daemon (tools/telegram_bridge/"
    "ingress_daemon.py, systemd user unit tg-ingress.service) is DOWN and a "
    "plain restart did NOT bring it back — likely a crash-loop from a code "
    "error. Diagnose: `journalctl --user -u tg-ingress.service -n 60 "
    "--no-pager` and read the traceback. Fix the bug in ingress_daemon.py "
    "(smallest correct change), then `systemctl --user restart "
    "tg-ingress.service`, wait ~15s, verify active AND that journal shows "
    "polling (no new traceback). Then report what was broken and what you "
    "changed via Telegram sendMessage using TELEGRAM_BOT_TOKEN and "
    "TELEGRAM_CHAT_ID from the repo-root .env (curl). Touch NOTHING outside "
    "tools/telegram_bridge/.")


def spawn_code_repair():
    """Restart didn't heal -> the daemon code itself is likely broken; spawn
    the repair-armed Sonnet (Edit scoped to the bridge dir). One at a time."""
    try:
        os.kill(int(REPAIR_LOCK.read_text().strip()), 0)
        return False
    except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
        pass
    with open(STATE / "code_repair.log", "a") as logf:
        logf.write(f"\n===== repair spawn {time.strftime('%F %T')} =====\n")
        proc = subprocess.Popen(
            ["timeout", "600", CLAUDE_BIN, "-p", REPAIR_PROMPT,
             "--model", "claude-sonnet-5", "--allowedTools",
             "Bash(systemctl:*),Bash(journalctl:*),Bash(curl:*),"
             "Bash(python3:*),Bash(pgrep:*),Bash(cat:*),Bash(tail:*),"
             "Bash(head:*),Bash(ls:*),Read,Grep,Glob,"
             "Edit(tools/telegram_bridge/**),Write(tools/telegram_bridge/**)"],
            cwd=str(REPO), stdout=logf, stderr=logf, start_new_session=True)
    REPAIR_LOCK.write_text(str(proc.pid))
    return True


def main():
    STATE.mkdir(exist_ok=True)
    if not service_active():
        subprocess.run(["systemctl", "--user", "restart", "tg-ingress.service"])
        time.sleep(5)
        state = "RESTARTED-OK" if service_active() else "RESTART-FAILED"
        log(f"ingress INACTIVE -> {state}")
        if state == "RESTART-FAILED":
            spawned = spawn_code_repair()
            log(f"code-repair sonnet spawned={spawned}")
            alert("🔴 tg-verify: daemon DOWN and restart FAILED — likely a "
                  "code error. A repair-armed Sonnet has been spawned to "
                  "diagnose the crash-loop, fix the code, and report here.")
        else:
            alert(f"🔴 tg-verify: ingress daemon was DOWN (accidental close?) — "
                  f"{state.lower()}. See state/health.log.")
        return
    if service_uptime_s() < HEARTBEAT_STALE_S:
        log("ok (startup grace — daemon recently started, first long-poll pending)")
        return
    hb_age = (time.time() - HEARTBEAT.stat().st_mtime) if HEARTBEAT.exists() else 1e9
    if hb_age > HEARTBEAT_STALE_S:
        subprocess.run(["systemctl", "--user", "restart", "tg-ingress.service"])
        log(f"ingress heartbeat stale ({int(hb_age)}s) -> restarted")
        alert(f"🟠 tg-verify: daemon alive but NOT POLLING "
              f"(heartbeat {int(hb_age)}s old) — restarted it.")
        return
    log(f"ok (heartbeat {int(hb_age)}s)")


if __name__ == "__main__":
    main()
