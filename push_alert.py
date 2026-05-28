import os
import sys
import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TOKEN or not CHAT_ID:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")

def push_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    res = requests.post(url, json=payload)
    if res.status_code != 200:
        print(f"[ERROR] Failed to push message: {res.text}")
    else:
        print(f"[SUCCESS] Push notification sent.")

def push_image(image_path, caption=""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    data = {"chat_id": CHAT_ID, "caption": caption}
    with open(image_path, "rb") as image_file:
        files = {"photo": image_file}
        res = requests.post(url, data=data, files=files)
    if res.status_code != 200:
        print(f"[ERROR] Failed to push image: {res.text}")
    else:
        print(f"[SUCCESS] Image sent.")

def push_document(doc_path, caption=""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    data = {"chat_id": CHAT_ID, "caption": caption}
    with open(doc_path, "rb") as doc_file:
        files = {"document": doc_file}
        res = requests.post(url, data=data, files=files)
    if res.status_code != 200:
        print(f"[ERROR] Failed to push document: {res.text}")
    else:
        print(f"[SUCCESS] Document sent.")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        push_message(sys.argv[1])
    elif len(sys.argv) == 3:
        if sys.argv[2].endswith(('.png', '.jpg', '.jpeg')):
            push_image(sys.argv[2], caption=sys.argv[1])
        else:
            push_document(sys.argv[2], caption=sys.argv[1])
