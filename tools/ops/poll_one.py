"""Single-shot Telegram long-poller for the in-session chat loop.

Blocks on getUpdates until ONE text message arrives from the configured
TELEGRAM_CHAT_ID, prints `NEW_MESSAGE:{text}` and exits 0. Run as a Claude Code
background task: its exit re-wakes Claude IN THE CURRENT SESSION with the
message in the task output — that's the phone -> chat-window channel.
Claude replies via the Bot API (sendMessage) and relaunches this poller.

.env resolution: tools/ops/.env first, then repo-root .env (the live copy —
this script lived at repo root until 2026-07-22, so root is the canonical one).
Security: messages from any chat id other than TELEGRAM_CHAT_ID are ignored
(logged to stderr), matching the bridge.py policy.
"""
import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
load_dotenv(HERE / ".env")
load_dotenv(REPO / ".env")          # does not override already-set keys
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TOKEN:
    print("No token")
    sys.exit(1)

def main():
    offset = None
    # clear backlog
    try:
        init_res = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=10).json()
        if init_res.get("ok") and init_res["result"]:
            offset = init_res["result"][-1]["update_id"] + 1
    except:
        pass

    while True:
        try:
            req_url = f"https://api.telegram.org/bot{TOKEN}/getUpdates?timeout=30"
            if offset is not None:
                req_url += f"&offset={offset}"

            res = requests.get(req_url, timeout=35).json()
            if res.get("ok") and res["result"]:
                for update in res["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    sender = str(msg.get("chat", {}).get("id", ""))
                    if text and CHAT_ID and sender != CHAT_ID:
                        print(f"IGNORED sender={sender}", file=sys.stderr)
                        continue
                    if text:
                        print(f"NEW_MESSAGE:{text}")
                        sys.exit(0)
        except Exception:
            time.sleep(2)

if __name__ == "__main__":
    main()
