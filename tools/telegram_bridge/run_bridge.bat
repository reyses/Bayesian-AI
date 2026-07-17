@echo off
rem Claude Code <-> Telegram bridge launcher.
rem For auto-start: Task Scheduler -> Create Task -> Trigger "At log on"
rem -> Action: start this .bat (run whether user is logged on or not).
cd /d "%~dp0..\.."
python3.11 tools\telegram_bridge\bridge.py
pause
