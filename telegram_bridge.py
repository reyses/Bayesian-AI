import os
import sys
import time
import threading
import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Set the TELEGRAM_BOT_TOKEN environment variable.")
URL = f"https://api.telegram.org/bot{TOKEN}"

def send_message(chat_id, text):
    res = requests.post(f"{URL}/sendMessage", json={"chat_id": chat_id, "text": text})
    if res.status_code != 200:
        print(f"[ERROR] Failed to send to {chat_id}: {res.text}", flush=True)

def poll_telegram():
    offset = None
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
                        print(f"TELEGRAM_INBOX|{chat_id}|{text}", flush=True)
        except Exception as e:
            time.sleep(2)

def stdin_reader():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.strip()
            if line.startswith("TELEGRAM_OUTBOX|"):
                parts = line.split("|", 2)
                if len(parts) == 3:
                    send_message(int(parts[1]), parts[2])
        except Exception as e:
            pass

if __name__ == '__main__':
    print("[INFO] Telegram Bridge is LIVE. Waiting for messages...", flush=True)
    t1 = threading.Thread(target=poll_telegram, daemon=True)
    t2 = threading.Thread(target=stdin_reader, daemon=True)
    t1.start()
    t2.start()
    # Keep main thread alive
    while True:
        time.sleep(1)
