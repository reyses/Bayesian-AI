"""
Claude Code <-> Telegram bridge (tools/telegram_bridge/bridge.py)

Talk to Claude Code from your phone. Unlike the AG bridge (telegram_mcp.py),
this does NOT paste keystrokes into a focused window — each incoming Telegram
message runs a HEADLESS Claude Code turn (`claude -p --resume <session>`)
in the repo, and the reply is sent back to Telegram. The desktop can be
locked; nothing needs focus.

Session model: one persistent bridge conversation (session id stored on
disk) so context carries across messages. `/new` starts a fresh one.

Security:
  - Only the configured TELEGRAM_CHAT_ID is served; all other senders are
    ignored and logged.
  - Claude runs headless under the PROJECT's permission settings — anything
    not allowlisted is denied non-interactively (Claude will say so rather
    than do it). No --dangerously-skip-permissions, ever.

Setup (once):
  1. On your phone: message @BotFather -> /newbot -> pick a name (e.g.
     "Claude Fable Bridge") -> copy the token.
  2. Put it in tools/telegram_bridge/.env  (gitignored):
         CLAUDE_TG_BOT_TOKEN=123456:ABC...
         CLAUDE_TG_CHAT_ID=<your chat id — same value as TELEGRAM_CHAT_ID
                            in the repo-root .env>
  3. Run:  python3.11 tools/telegram_bridge/bridge.py
     (or run_bridge.bat; for auto-start use Task Scheduler -> run at logon)

Commands from the phone:
  /new     start a fresh conversation
  /status  bridge + session status
  anything else = a message to Claude
"""
import os
import sys
import json
import time
import glob
import subprocess
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
STATE_DIR = HERE / "state"
STATE_DIR.mkdir(exist_ok=True)
SESSION_FILE = STATE_DIR / "session_id.txt"
OFFSET_FILE = STATE_DIR / "offset.txt"
DOWNLOAD_DIR = REPO_ROOT / "scratch" / "claude_tg_downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG = STATE_DIR / "bridge.log"

POLL_TIMEOUT_S = 50          # Telegram long-poll timeout
CLAUDE_TIMEOUT_S = 900       # per-turn wall clock for a headless Claude run
TG_CHUNK = 4000              # Telegram hard limit is 4096 chars/message
MODEL = os.environ.get("CLAUDE_TG_MODEL", "")   # empty = project default


def log(*a):
    line = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_env():
    envf = HERE / ".env"
    if envf.exists():
        for ln in envf.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    tok = os.environ.get("CLAUDE_TG_BOT_TOKEN")
    cid = os.environ.get("CLAUDE_TG_CHAT_ID")
    if not tok or not cid:
        sys.exit("Set CLAUDE_TG_BOT_TOKEN and CLAUDE_TG_CHAT_ID in "
                 f"{envf} (see module docstring).")
    return tok, cid


def discover_claude() -> str:
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    pats = [
        os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude-code", "*", "claude.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Claude", "claude-code", "*", "claude.exe"),
    ]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p))
    if not cands:
        sys.exit("claude.exe not found; set CLAUDE_BIN")
    cands.sort(key=lambda p: os.path.basename(os.path.dirname(p)))
    return cands[-1]


class Bridge:
    def __init__(self):
        self.token, self.chat_id = load_env()
        self.api = f"https://api.telegram.org/bot{self.token}"
        self.claude = discover_claude()
        self.session_id = SESSION_FILE.read_text().strip() if SESSION_FILE.exists() else ""
        self.offset = int(OFFSET_FILE.read_text()) if OFFSET_FILE.exists() else None

    # ---- Telegram I/O -----------------------------------------------------
    def send(self, text: str):
        for i in range(0, max(len(text), 1), TG_CHUNK):
            try:
                requests.post(f"{self.api}/sendMessage",
                              json={"chat_id": self.chat_id,
                                    "text": text[i:i + TG_CHUNK] or "(empty reply)"},
                              timeout=30)
            except requests.RequestException as e:
                log("send failed:", e)

    def typing(self):
        try:
            requests.post(f"{self.api}/sendChatAction",
                          json={"chat_id": self.chat_id, "action": "typing"}, timeout=10)
        except requests.RequestException:
            pass

    def download(self, file_id: str, name_hint: str) -> str:
        try:
            r = requests.get(f"{self.api}/getFile", params={"file_id": file_id},
                             timeout=30).json()
            fp = r["result"]["file_path"]
            data = requests.get(
                f"https://api.telegram.org/file/bot{self.token}/{fp}", timeout=120).content
            ext = fp.rsplit(".", 1)[-1] if "." in fp else "bin"
            out = DOWNLOAD_DIR / f"tg_{int(time.time())}_{name_hint or 'file'}.{ext}"
            out.write_bytes(data)
            return str(out)
        except (requests.RequestException, KeyError) as e:
            log("download failed:", e)
            return ""

    # ---- Claude -----------------------------------------------------------
    def ask_claude(self, prompt: str) -> str:
        cmd = [self.claude, "-p", prompt, "--output-format", "json"]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        if MODEL:
            cmd += ["--model", MODEL]
        try:
            r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=CLAUDE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"(Claude timed out after {CLAUDE_TIMEOUT_S}s — the turn may still land in the session; /status to check)"
        if r.returncode != 0:
            # --resume of a stale/compacted session can fail; retry fresh once
            if self.session_id:
                log("resume failed rc", r.returncode, "— retrying with a fresh session")
                self.session_id = ""
                return self.ask_claude(prompt)
            return f"(claude error rc={r.returncode}: {(r.stderr or '')[:500]})"
        try:
            payload = json.loads(r.stdout)
            sid = payload.get("session_id", "")
            if sid:
                self.session_id = sid
                SESSION_FILE.write_text(sid)
            return payload.get("result", "") or "(no text in reply)"
        except json.JSONDecodeError:
            return r.stdout[:TG_CHUNK] or "(unparseable reply)"

    # ---- main loop --------------------------------------------------------
    def handle(self, msg: dict):
        text = (msg.get("text") or "").strip()
        cap = (msg.get("caption") or "").strip()
        if text == "/new":
            self.session_id = ""
            if SESSION_FILE.exists():
                SESSION_FILE.unlink()
            self.send("Fresh conversation started.")
            return
        if text == "/status":
            self.send(f"Bridge alive. session={self.session_id or '(none yet)'} "
                      f"repo={REPO_ROOT.name} model={MODEL or 'project default'}")
            return

        prompt = text
        if "photo" in msg:
            p = self.download(msg["photo"][-1]["file_id"], "photo")
            prompt = f"[User sent a photo from their phone, saved at: {p}]" + \
                     (f"\n{cap}" if cap else "")
        elif "document" in msg:
            doc = msg["document"]
            p = self.download(doc["file_id"], Path(doc.get("file_name", "doc")).stem)
            prompt = f"[User sent a file from their phone, saved at: {p}]" + \
                     (f"\n{cap}" if cap else "")
        if not prompt:
            return
        log("Q:", prompt[:120])
        self.typing()
        reply = self.ask_claude(prompt)
        log("A:", reply[:120])
        self.send(reply)

    def run(self):
        log(f"bridge up | claude={self.claude} | repo={REPO_ROOT}")
        # drop backlog so old messages don't replay into Claude on restart
        if self.offset is None:
            try:
                r = requests.get(f"{self.api}/getUpdates", timeout=15).json()
                if r.get("ok") and r["result"]:
                    self.offset = r["result"][-1]["update_id"] + 1
            except requests.RequestException:
                pass
        while True:
            try:
                params = {"timeout": POLL_TIMEOUT_S}
                if self.offset is not None:
                    params["offset"] = self.offset
                r = requests.get(f"{self.api}/getUpdates", params=params,
                                 timeout=POLL_TIMEOUT_S + 10).json()
                if not r.get("ok"):
                    if r.get("error_code") == 409:
                        log("409: another poller holds this token — retrying in 30s")
                        time.sleep(30)
                    continue
                for upd in r["result"]:
                    self.offset = upd["update_id"] + 1
                    OFFSET_FILE.write_text(str(self.offset))
                    m = upd.get("message") or {}
                    sender = str((m.get("chat") or {}).get("id", ""))
                    if sender != str(self.chat_id):
                        log("ignored message from unauthorized chat", sender)
                        continue
                    self.handle(m)
            except requests.RequestException as e:
                log("poll error:", e)
                time.sleep(5)
            except KeyboardInterrupt:
                log("bridge stopped")
                return


if __name__ == "__main__":
    Bridge().run()
