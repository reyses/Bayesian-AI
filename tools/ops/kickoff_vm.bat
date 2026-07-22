@echo off
title Autonomous Drone Kickoff
echo ========================================================
echo   Kicking off Autonomous Year Pipeline on ai-node VM
echo ========================================================
echo.

echo [1/3] Uploading the runner script and latest code to the VM...
call gcloud.cmd compute scp "vm_runner.sh" "ai-node:/home/reyse/Bayesian-AI/" --project=bayesian-ai --zone=us-central1-a
call gcloud.cmd compute scp "research/Regression segments/run_year.py" "ai-node:/home/reyse/Bayesian-AI/research/Regression segments/" --project=bayesian-ai --zone=us-central1-a
call gcloud.cmd compute scp "research/Regression segments/stage1_speed_pass.py" "ai-node:/home/reyse/Bayesian-AI/research/Regression segments/" --project=bayesian-ai --zone=us-central1-a
call gcloud.cmd compute scp "research/Regression segments/stage2_parallel_chaos.py" "ai-node:/home/reyse/Bayesian-AI/research/Regression segments/" --project=bayesian-ai --zone=us-central1-a
call gcloud.cmd compute scp "research/Regression segments/tiering.py" "ai-node:/home/reyse/Bayesian-AI/research/Regression segments/" --project=bayesian-ai --zone=us-central1-a
call gcloud.cmd compute scp "scratch/vm_telegram_mcp.py" "ai-node:/home/reyse/Bayesian-AI/telegram_mcp.py" --project=bayesian-ai --zone=us-central1-a

echo [2/3] Making the script executable and launching in the background...
call gcloud.cmd compute ssh ai-node --project=bayesian-ai --zone=us-central1-a --command="chmod +x /home/reyse/Bayesian-AI/vm_runner.sh && nohup /home/reyse/Bayesian-AI/vm_runner.sh > /home/reyse/Bayesian-AI/run.log 2>&1 &"

echo.
echo [3/3] SUCCESS!
echo The job is now running autonomously in the background on the VM.
echo It will process the full year, ZIP the final stage2_year_segments.json,
echo leave the ZIP on the VM. When it's done, just run "download_results.bat" to fetch it to your PC!
echo.
echo You can safely close your laptop or exit this window!
pause
