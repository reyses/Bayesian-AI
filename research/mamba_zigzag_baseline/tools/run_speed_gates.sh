#!/bin/bash
# Speed-pass gates for train_mamba_rl_seq.py --compile_act.
# Run from the repo root INSIDE WSL (GPU must be free).
# Gate 1: fp32 logit-path parity, eager vs compiled (1e-6 on losses, actions identical)
# Gate 2: bitwise self-determinism of the compiled path (two identical runs)
# Gate 3: bf16 full-epoch perf A/B (eager baseline from the smoke: 248-254 bars/s)
set -e
PY=/home/reyses/venvs/bayesian-ai/bin/python
T=research/mamba_zigzag_baseline/pipeline/train_mamba_rl_seq.py
CK=research/mamba_zigzag_baseline/tools/compile_parity_check.py
R=research/mamba_zigzag_baseline/reports
DAYS=2024_02_20,2024_05_15,2024_08_13,2024_11_12
COMMON="--days $DAYS --num_episodes 1 --no-checkpoint --seed 0 --init_from checkpoints/mamba_warmstart.pth"

echo "== gate 1a: eager fp32, 2000 bars =="
$PY $T $COMMON --max-steps 2000 --no_autocast --loss-dump $R/parity_eager.npz
echo "== gate 1b: compiled fp32, 2000 bars =="
$PY $T $COMMON --max-steps 2000 --no_autocast --compile_act --loss-dump $R/parity_compiled.npz
echo "== gate 1c: compare (tol 1e-6, actions exact) =="
$PY $CK $R/parity_eager.npz $R/parity_compiled.npz --tol 1e-6
echo "== gate 2: compiled rerun -> bitwise self-determinism =="
$PY $T $COMMON --max-steps 2000 --no_autocast --compile_act --loss-dump $R/parity_compiled2.npz
$PY $CK $R/parity_compiled.npz $R/parity_compiled2.npz --bitwise
echo "== gate 3: compiled bf16 full-epoch perf ([PERF] vs eager 248-254 bars/s) =="
$PY $T $COMMON --smoke_metrics --compile_act
echo "ALL GATES DONE"
