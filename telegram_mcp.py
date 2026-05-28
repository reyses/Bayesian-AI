import asyncio
import os
import requests
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("TelegramNotifier")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TOKEN or not CHAT_ID:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")

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

if __name__ == "__main__":
    mcp.run()
