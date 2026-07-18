#!/usr/bin/env bash
# Week validation driver — runs B2T / B2C / RunC stage-1 over 4 days x {real,Fourier},
# SEQUENTIALLY (one at a time: no GPU contention, no OOM), idempotent (skips completed runs),
# then prints the 3-model flip-timing contrast with day-block bootstrap CIs.
#
# Day 1 (2024_02_20) is already done (B2Tmap/B2Cmap/RUNCmap) and is reused by the analysis.
# Safe to re-run: anything whose log already shows "[MAIN] wrote" is skipped.
#
# RUN:  bash research/fspace_experiment/run_week.sh
set -u
cd "$(dirname "$0")/../../.."

declare -A FROOT=( [WK]=FEATURES_RUN_B2T [WKBC]=FEATURES_RUN_B2C [WKRC]=FEATURES_RUN_C2024 )
DAYS=(2024_02_21 2024_02_22 2024_02_23 2024_02_26)

for pfx in WK WKBC WKRC; do
  for D in "${DAYS[@]}"; do
    for FR in REAL FOUR; do
      if [ "$FR" = "REAL" ]; then day="$D"; else day="${D}_FOUR"; fi
      log="logs/stage1_${pfx}_${D}_${FR}.log"
      if grep -q "MAIN] wrote" "$log" 2>/dev/null; then
        echo "[skip] ${pfx}_${D}_${FR} (already complete)"; continue
      fi
      echo "[run ] ${pfx}_${D}_${FR}  (features=${FROOT[$pfx]})"
      python research/fspace_cadence/pipeline/stage1_speed_pass.py \
        --day "$day" --tf 1s --run_name "${pfx}_${D}_${FR}" \
        --features_root "DATA/ATLAS/${FROOT[$pfx]}" > "$log" 2>&1
      if grep -q "MAIN] wrote" "$log" 2>/dev/null; then echo "       done"; else echo "       FAILED (see $log)"; fi
    done
  done
done

echo "=== WEEK ANALYSIS ==="
python research/fspace_cadence/tools/week_analysis.py
