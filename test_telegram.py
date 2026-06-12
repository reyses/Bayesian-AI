import sys
import os
repo_root = "/home/reyse/Bayesian-AI"
sys.path.append(repo_root)

try:
    from telegram_mcp import send_telegram_alert
    send_telegram_alert('🤖 Hello from the ai-node VM! Telegram integration is working perfectly.')
    print('SUCCESS: Telegram alert sent.')
except Exception as e:
    print('ERROR:', e)
