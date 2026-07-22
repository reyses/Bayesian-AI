import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

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
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    if text:
                        print(f"NEW_MESSAGE:{text}")
                        sys.exit(0)
        except Exception as e:
            time.sleep(2)

if __name__ == "__main__":
    main()
