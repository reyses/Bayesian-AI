import asyncio
import os
import sys
import time
import threading
import subprocess
import requests
import pyperclip
import pyautogui
import re
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("TelegramNotifier")

from dotenv import load_dotenv
# Resolve .env against THIS script's directory, not the caller's cwd —
# the IDE typically spawns MCP servers with an unrelated working dir.
load_dotenv(Path(__file__).resolve().parent / ".env")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TOKEN or not CHAT_ID:
    print("CRITICAL ERROR: Tokens missing. Please either:", file=sys.stderr)
    print("1) Hardcode TOKEN='...' and CHAT_ID='...' directly into telegram_mcp.py", file=sys.stderr)
    print("2) Create a .env file with TELEGRAM_BOT_TOKEN=... and TELEGRAM_CHAT_ID=...", file=sys.stderr)
    raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")

DOWNLOAD_DIR = Path(r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\scratch\telegram_downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
def send_telegram_alert(message: str) -> str:
    """Send an autonomous push notification to the user's Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return "Push notification sent successfully."
    except Exception as e:
        return f"Failed to send notification: {str(e)}"

@mcp.tool()
def send_telegram_media(filepath: str, caption: str = "") -> str:
    """Send a local file (image or document) to the user's Telegram.

    Relative paths are resolved against the directory containing this
    script (the repo root), so the agent can pass `oos_chart.png`
    regardless of the MCP server's current working directory.
    """
    if not os.path.isabs(filepath):
        resolved = str(Path(__file__).resolve().parent / filepath)
        print(f"[send_telegram_media] '{filepath}' -> '{resolved}'", file=sys.stderr, flush=True)
        filepath = resolved
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
    
    file_ext = filepath.lower().split('.')[-1]
    is_photo = file_ext in ['png', 'jpg', 'jpeg', 'gif']
    
    endpoint = "sendPhoto" if is_photo else "sendDocument"
    url = f"https://api.telegram.org/bot{TOKEN}/{endpoint}"
    
    # Multipart form data
    file_field = "photo" if is_photo else "document"
    data = {"chat_id": CHAT_ID, "caption": caption}
    
    try:
        with open(filepath, 'rb') as f:
            files = {file_field: f}
            res = requests.post(url, data=data, files=files)
        res.raise_for_status()
        return f"Media {endpoint} sent successfully."
    except Exception as e:
        return f"Failed to send media: {str(e)}"

@mcp.tool()
def run_autostats(arg: str) -> str:
    """Run the autostats script with the given argument."""
    script_path = str(DOWNLOAD_DIR.parent / "autostats.py")
    try:
        res = subprocess.run(["python", script_path, str(arg)], capture_output=True, text=True, check=True)
        return f"autostats completed:\n{res.stdout}"
    except subprocess.CalledProcessError as e:
        return f"autostats failed (exit {e.returncode}):\n{e.stderr}"
    except Exception as e:
        return f"Failed to run autostats: {str(e)}"

@mcp.tool()
def run_autoplot(arg: str) -> str:
    """Run the autoplot script with the given argument."""
    script_path = str(DOWNLOAD_DIR.parent / "autoplot.py")
    try:
        res = subprocess.run(["python", script_path, str(arg)], capture_output=True, text=True, check=True)
        return f"autoplot completed:\n{res.stdout}"
    except subprocess.CalledProcessError as e:
        return f"autoplot failed (exit {e.returncode}):\n{e.stderr}"
    except Exception as e:
        return f"Failed to run autoplot: {str(e)}"

def inject_prompt(message):
    """Uses Pyperclip and PyAutoGUI to paste the message instantly."""
    print(f"Injecting message: {message}", file=sys.stderr)
    
    # Copy message to OS clipboard
    pyperclip.copy(message)
    
    # Paste instantly using Ctrl+V
    pyautogui.hotkey('ctrl', 'v')
    
    # Give a tiny delay for the UI to register the paste
    time.sleep(0.1)
    
    # Press Enter to send it to the agent
    pyautogui.press('enter')

def download_telegram_file(file_id: str, ext: str = "") -> str:
    """Downloads a file from Telegram given its file_id and returns the local path."""
    try:
        # 1. Get file path from Telegram
        url = f"https://api.telegram.org/bot{TOKEN}/getFile?file_id={file_id}"
        res = requests.get(url).json()
        if not res.get("ok"):
            return ""
        
        file_path_tg = res["result"]["file_path"]
        
        # 2. Download the file
        download_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path_tg}"
        file_data = requests.get(download_url).content
        
        # 3. Save locally
        filename = f"tg_{int(time.time())}_{file_id[:8]}"
        if ext:
            filename += f".{ext}"
        elif '.' in file_path_tg:
            filename += f".{file_path_tg.split('.')[-1]}"
            
        local_path = DOWNLOAD_DIR / filename
        with open(local_path, "wb") as f:
            f.write(file_data)
            
        return str(local_path.resolve())
    except Exception as e:
        print(f"Failed to download file: {e}", file=sys.stderr)
        return ""

def poll_telegram():
    """Background loop that listens for messages and injects them."""
    offset = None
    print("[INFO] Telegram Bridge is LIVE in MCP. Polling and injecting directly...", file=sys.stderr, flush=True)
    
    # Give the IDE 5 seconds to finish spinning up the MCP server before starting PyAutoGUI
    time.sleep(5)
    
    while True:
        try:
            req_url = f"https://api.telegram.org/bot{TOKEN}/getUpdates?timeout=30"
            if offset is not None:
                req_url += f"&offset={offset}"
            
            res = requests.get(req_url, timeout=35).json()
            if res.get("ok"):
                for update in res["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    
                    text = msg.get("text", "")
                    caption = msg.get("caption", "")
                    
                    local_filepath = ""
                    
                    # Handle Photo
                    if "photo" in msg:
                        # Get highest res photo (last item in array)
                        photo = msg["photo"][-1]
                        local_filepath = download_telegram_file(photo["file_id"], ext="jpg")
                        
                    # Handle Document
                    elif "document" in msg:
                        doc = msg["document"]
                        ext = doc.get("file_name", "").split(".")[-1] if "." in doc.get("file_name", "") else ""
                        local_filepath = download_telegram_file(doc["file_id"], ext=ext)
                    
                    # Construct Injection String
                    if local_filepath:
                        print(f"[RECEIVED MEDIA] Saved to {local_filepath}. Typing...", file=sys.stderr, flush=True)
                        time.sleep(2)
                        
                        inject_str = f"Telegram: [User sent a file saved at {local_filepath}]"
                        if caption:
                            inject_str += f"\nCaption: {caption}"
                        inject_prompt(inject_str)
                        
                    elif text:
                        match_stats = re.match(r"(?i)^/?autostats\((.*?)\)$", text.strip())
                        match_plot = re.match(r"(?i)^/?autoplot\((.*?)\)$", text.strip())
                        
                        if match_stats:
                            arg = match_stats.group(1)
                            print(f"[COMMAND] Intercepted autostats({arg})", file=sys.stderr, flush=True)
                            script_path = str(DOWNLOAD_DIR.parent / "autostats.py")
                            subprocess.Popen(["python", script_path, arg], cwd=str(DOWNLOAD_DIR.parent))
                            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": f"Triggered autostats({arg})"})
                        elif match_plot:
                            arg = match_plot.group(1)
                            print(f"[COMMAND] Intercepted autoplot({arg})", file=sys.stderr, flush=True)
                            script_path = str(DOWNLOAD_DIR.parent / "autoplot.py")
                            subprocess.Popen(["python", script_path, arg], cwd=str(DOWNLOAD_DIR.parent))
                            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": f"Triggered autoplot({arg})"})
                        else:
                            print(f"[RECEIVED TEXT] {text}. Typing...", file=sys.stderr, flush=True)
                            time.sleep(2)
                            inject_prompt("Telegram: " + text)
                        
        except Exception as e:
            time.sleep(2)

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    
    # Start the polling loop in a daemon thread so it runs in the background
    # and dies automatically when the MCP server shuts down.
    listener_thread = threading.Thread(target=poll_telegram, daemon=True)
    listener_thread.start()
    
    # Run the MCP server
    mcp.run()
