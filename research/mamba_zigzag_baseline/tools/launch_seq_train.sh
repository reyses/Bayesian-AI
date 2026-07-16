#!/usr/bin/env bash
# Bulletproof detached launcher for the seq trainer. The redirect happens
# INSIDE the script (exec) so the log is captured regardless of how the
# parent invokes this. Run via:
#   wsl bash -c "cd <repo> && setsid bash research/.../launch_seq_train.sh <N> </dev/null >/dev/null 2>&1 & disown; echo started"
set -u
N="${1:-50}"
cd /mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI
exec $HOME/venvs/bayesian-ai/bin/python -u \
    research/mamba_zigzag_baseline/pipeline/train_mamba_rl_seq.py \
    --num_episodes "$N" --seed 42 > seq_train.log 2>&1
