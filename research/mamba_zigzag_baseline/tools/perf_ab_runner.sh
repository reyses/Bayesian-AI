#!/usr/bin/env bash
# Interleaved A/B speed test for train_mamba_rl.py variants (run from repo root in WSL).
# Usage: bash research/mamba_zigzag_baseline/tools/perf_ab_runner.sh <A.py> <B.py> <out.txt> [reps] [max_steps]
# Copies each variant over the pipeline file in ABAB order, runs a fixed-seed
# 1-day capped run, greps bars/sec. Restores B at the end.
set -u
A_SRC="$1"; B_SRC="$2"; OUT="$3"; REPS="${4:-4}"; MAXSTEPS="${5:-1800}"
F=research/mamba_zigzag_baseline/pipeline/train_mamba_rl.py
: > "$OUT"
for i in $(seq 1 "$REPS"); do
  for V in A B; do
    if [ "$V" = A ]; then cp "$A_SRC" "$F"; else cp "$B_SRC" "$F"; fi
    R=$($HOME/venvs/bayesian-ai/bin/python "$F" --num_episodes 1 --days 2024_02_20 --seed 42 \
        --max-steps "$MAXSTEPS" --no-checkpoint 2>&1 | grep -o 'bars/sec = [0-9.]*')
    echo "$V$i: ${R:-RUN_FAILED}" | tee -a "$OUT"
  done
done
cp "$B_SRC" "$F"
