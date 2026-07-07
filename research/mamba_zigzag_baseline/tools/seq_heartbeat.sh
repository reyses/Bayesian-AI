#!/usr/bin/env bash
# Heartbeat/completion watcher for the seq training run. Polls the log; exits
# early on completion / process death / error, else after ~9 min (heartbeat).
# Re-arm each time it fires. Prints recent epochs + latest checkpoint.
cd /mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI
LOG=seq_train.log
for i in $(seq 1 18); do
  if ! pgrep -f train_mamba_rl_seq >/dev/null; then echo PROCESS_GONE; break; fi
  if grep -q 'Training complete' "$LOG" 2>/dev/null; then echo TRAINING_COMPLETE; break; fi
  if grep -qE 'Traceback|CUDA out of|RuntimeError' "$LOG" 2>/dev/null; then echo POSSIBLE_ERROR; break; fi
  sleep 30
done
echo '=== heartbeat ==='
grep -E 'Epoch [0-9]+ \| Reward' "$LOG" | tail -5
echo '--- latest ckpt ---'
ls -t mamba_rl_seq_checkpoint_ep*.pth 2>/dev/null | head -1
