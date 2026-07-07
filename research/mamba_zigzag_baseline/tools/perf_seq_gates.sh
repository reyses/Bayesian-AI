#!/usr/bin/env bash
# Seq-trainer gates: self-determinism (2 identical seeded runs + diff) then a
# longer speed run. Run from repo root inside WSL.
set -u
P=research/mamba_zigzag_baseline
PY=.venv_wsl/bin/python
$PY $P/pipeline/train_mamba_rl_seq.py --num_episodes 1 --days 2024_02_20 --seed 42 \
    --max-steps 2300 --no-checkpoint --loss-dump $P/reports/perf/parity_seq_a.npz 2>&1 | grep -E 'PERF|Error|Traceback'
$PY $P/pipeline/train_mamba_rl_seq.py --num_episodes 1 --days 2024_02_20 --seed 42 \
    --max-steps 2300 --no-checkpoint --loss-dump $P/reports/perf/parity_seq_b.npz 2>&1 | grep -E 'PERF|Error|Traceback'
$PY $P/tools/perf_parity_diff.py $P/reports/perf/parity_seq_a.npz $P/reports/perf/parity_seq_b.npz
$PY $P/pipeline/train_mamba_rl_seq.py --num_episodes 1 --days 2024_02_20 --seed 42 \
    --max-steps 6000 --no-checkpoint 2>&1 | grep -E 'PERF|Error|Traceback'
