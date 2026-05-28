import os
import sys
import time
import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Set the TELEGRAM_BOT_TOKEN environment variable.")
URL = f"https://api.telegram.org/bot{TOKEN}"
INBOX_FILE = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/telegram_inbox.txt"

def poll_telegram():
    offset = None
    print("[INFO] Telegram Listener is LIVE. Writing messages to file...", flush=True)
    while True:
        try:
            req_url = f"{URL}/getUpdates?timeout=30"
            if offset is not None:
                req_url += f"&offset={offset}"
            
            res = requests.get(req_url, timeout=35).json()
            if res.get("ok"):
                for update in res["result"]:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        chat_id = update["message"]["chat"]["id"]
                        text = update["message"]["text"]
                        
                        # Write to file for the agent to parse on wake-up
                        with open(INBOX_FILE, "a") as f:
                            f.write(f"{text}\n")
                        print(f"[RECEIVED] {text}", flush=True)
        except Exception as e:
            time.sleep(2)

if __name__ == '__main__':
    poll_telegram()
