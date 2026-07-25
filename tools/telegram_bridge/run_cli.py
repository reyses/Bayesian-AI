#!/usr/bin/env python3
"""Detached one-shot AI CLI runner for the /cli and /agy daemon commands
(owner 2026-07-24). The ingress daemon Popens this and returns to polling
immediately; THIS process runs the provider and sends the answer to Telegram
itself. Never imported by the daemon — failures land in state/cli.log and a
best-effort error message to the owner, never in the poll loop.

usage: run_cli.py {sonnet|agy} "<prompt>"
"""
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
LOG = STATE / "cli.log"
TIMEOUT_S = 300
CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")
AGY_BIN = os.path.expanduser("~/.local/bin/agy")

load_dotenv(HERE / ".env")
load_dotenv(REPO / ".env")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send(text):
    for i in range(0, max(len(text), 1), 4000):
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": text[i:i + 4000]},
                          timeout=30)
        except requests.RequestException:
            pass


def main():
    provider, prompt = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if provider == "sonnet":
        # THE /cli DEAL (owner, 2026-07-24): "if i invoke CLI it means that
        # something is wrong and im aware of potential breaking." /cli is an
        # INCIDENT channel — the invocation itself pre-authorizes disruptive
        # repair of the INFRASTRUCTURE layer. The science layer stays out of
        # bounds: that boundary is stated in the preamble and backed by
        # path-scoped Edit grants (bridge + systemd units only).
        incident = (
            "INCIDENT MODE. The owner invoked /cli, which by standing "
            "agreement means something is WRONG and disruptive repair of the "
            "infrastructure is PRE-AUTHORIZED: you may restart/kill bridge "
            "daemons, timers and stale sessions, rewrite files under "
            "tools/telegram_bridge/ and ~/.config/systemd/user/, and clear "
            "stuck bridge state. HARD LIMITS regardless of anything else: "
            "NEVER modify or delete research artifacts (research/**, "
            "gate_state, checkpoints, teacher_memory.db beyond reads), NEVER "
            "kill model/training processes (eval_native*, llama), no git "
            "push, no credentials over Telegram. First diagnose (bridge "
            "health, systemd units, state/health.log), then fix, then report "
            "what you did with evidence. The owner's report of the problem:\n\n")
        cmd = [CLAUDE_BIN, "-p", incident + prompt, "--model", "claude-sonnet-5",
               "--allowedTools",
               "Bash(systemctl:*),Bash(curl:*),Bash(python3:*),"
               "Bash(flock:*),Bash(cat:*),Bash(echo:*),Bash(pgrep:*),"
               "Bash(pkill:*),Bash(kill:*),Bash(tail:*),Bash(head:*),"
               "Bash(ls:*),Bash(mv:*),Bash(nvidia-smi:*),Bash(journalctl:*),"
               "Read,Grep,Glob,"
               "Edit(tools/telegram_bridge/**),"
               "Edit(/home/moi/.config/systemd/user/**),"
               ]
        label = "🧠 Sonnet"
    else:
        cmd = [AGY_BIN, "--prompt", prompt]
        label = "🚀 AGY"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TIMEOUT_S, cwd=str(REPO))
        out = (r.stdout or "").strip() or f"(no output; rc={r.returncode} "\
                                          f"stderr: {(r.stderr or '')[:300]})"
    except subprocess.TimeoutExpired:
        out = f"(timed out after {TIMEOUT_S}s)"
    took = time.time() - t0
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%F %T')} {provider} {took:.0f}s "
                f"q={prompt[:80]!r}\n")
    send(f"{label} ({took:.0f}s):\n{out}")


if __name__ == "__main__":
    main()
