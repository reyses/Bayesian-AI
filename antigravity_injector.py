import time
import os
import ctypes
import pyautogui

def get_latest_telegram_message(log_path):
    """Reads the last line of the telegram_inbox.txt."""
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip()
    except Exception as e:
        print(f"Error reading inbox: {e}")
    return None

import pyperclip

def inject_prompt(message):
    """Uses Pyperclip and PyAutoGUI to paste the message instantly."""
    print(f"Injecting message: {message}")
    
    # Copy message to OS clipboard
    pyperclip.copy(message)
    
    # Paste instantly using Ctrl+V
    pyautogui.hotkey('ctrl', 'v')
    
    # Give a tiny delay for the UI to register the paste
    time.sleep(0.1)
    
    # Press Enter to send it to the agent
    pyautogui.press('enter')

def main():
    print("Antigravity Auto-Prompt Injector Started")
    print("Waiting for new Telegram messages...")
    
    inbox_path = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\telegram_inbox.txt"
    last_processed_msg = get_latest_telegram_message(inbox_path)
    
    while True:
        current_msg = get_latest_telegram_message(inbox_path)
        
        # If we have a new message that we haven't processed yet
        if current_msg and current_msg != last_processed_msg:
            # We assume the user has the Antigravity window focused, or they just sent the message
            # For robustness, we could use pywinauto to force-focus the window, 
            # but pyautogui will just type wherever the cursor currently is.
            
            print("New message detected! Typing in 2 seconds...")
            time.sleep(2) # Brief delay to ensure UI is ready
            
            inject_prompt("Telegram: " + current_msg)
            last_processed_msg = current_msg
            
        time.sleep(1) # Poll every 1 second

if __name__ == "__main__":
    # Safety feature: Move mouse to corner of screen to abort PyAutoGUI
    pyautogui.FAILSAFE = True
    main()
