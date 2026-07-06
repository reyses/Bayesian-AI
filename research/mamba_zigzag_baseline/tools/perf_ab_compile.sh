#!/usr/bin/env bash
# Interleaved A/B: eager (A) vs --compile (B) on the CURRENT train_mamba_rl.py.
# Usage: bash perf_ab_compile.sh <out.txt> <reps> [max_steps] [start_idx]
# Appends to out.txt so it can be run in chunks.
set -u
OUT="$1"; REPS="$2"; MAXSTEPS="${3:-2300}"; START="${4:-1}"
F=research/mamba_zigzag_baseline/pipeline/train_mamba_rl.py
END=$((START + REPS - 1))
for i in $(seq "$START" "$END"); do
  for V in A B; do
    FLAG=""
    if [ "$V" = B ]; then FLAG="--compile"; fi
    R=$(.venv_wsl/bin/python "$F" --num_episodes 1 --days 2024_02_20 --seed 42 \
        --max-steps "$MAXSTEPS" --no-checkpoint $FLAG 2>&1 | grep -o 'bars/sec = [0-9.]*')
    echo "$V$i: ${R:-RUN_FAILED}" | tee -a "$OUT"
  done
done
