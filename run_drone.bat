@echo off
title Drone Worker Initializer
echo ==========================================================
echo    BAYESIAN-AI DRONE NODE INITIALIZATION
echo ==========================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ and check "Add to PATH".
    pause
    exit /b
)

REM Ensure dependencies are installed
echo [INFO] Ensuring required libraries are installed...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install requests numpy pandas torch scikit-learn seaborn matplotlib group-lasso psutil >nul 2>&1

set /p MOTHERSHIP_IP="Enter the IP address of the Mothership (e.g. 192.168.1.50): "

echo.
echo [INFO] Launching Drone Worker connected to %MOTHERSHIP_IP%...
echo.

python "research/Regression segments/drone_worker.py" --host %MOTHERSHIP_IP%

pause
