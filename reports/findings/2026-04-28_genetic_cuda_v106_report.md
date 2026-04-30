# CUDA Genetic optimization report — ZigzagRunner_v106

Generated: 2026-04-28 22:15

## Setup

- Atlas: `DATA/ATLAS`
- Train: through `2025-12-31` (314,716 1m bars)
- Holdout: from `2026-01-01` (76,798 1m bars)
- Optimizer: numba.cuda kernel + manual rand/1/bin DE on GPU
- Search dim: 6, popsize_mult: 30 (actual P=180), maxiter: 30, seed: 42
- DE wall time: 3.4 sec
- Best train PnL (float32 sim): $+7,017.68

## Top-5 candidates (CPU float64 verification + holdout)

