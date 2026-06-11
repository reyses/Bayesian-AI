@echo off
chcp 65001 > nul
title Bayesian-AI Telemetry Stream
echo Connecting to VM to stream telemetry...
gcloud.cmd compute ssh ai-node --project=bayesian-ai --zone=us-central1-a --command="cd /home/reyse/Bayesian-AI && python3 core_v2/telemetry/cli.py"
pause
