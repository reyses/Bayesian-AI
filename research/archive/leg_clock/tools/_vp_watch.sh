#!/usr/bin/env bash
cd /mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI
for i in $(seq 1 19); do
  if ! pgrep -f vp_study.py >/dev/null; then echo VP_PROCESS_ENDED; break; fi
  if grep -q 'Written to' vp_study.log 2>/dev/null; then echo VP_DONE; break; fi
  sleep 30
done
echo '--- log tail ---'
tail -5 vp_study.log 2>/dev/null
