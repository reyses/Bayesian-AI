# Bayesian-AI Tools Index

## Data Sourcing & Pipeline
- `tools/sourcing/convert_nt8_csv_to_parquet.py`: Converts RAW NT8 CSV dumps into ATLAS_NT8 schema with session-day partitioning and contract roll stitching.
- `DATA/pipeline/build_timeframes.py`: Aggregates 1s and 1m ATLAS parquets into coarser timeframes (5s, 15s, 5m, 1h, etc.) and validates their parity.
- `DATA/pipeline/databento_to_atlas.py`: Downloads and processes raw databento MBP-1 tick/bar data into the ATLAS schema.

## Perf / Profiling (Mamba RL)
- `research/mamba_zigzag_baseline/tools/perf_step_breakdown.py`: Sync-bracketed per-component ms/bar attribution of the RL training step (env vs forwards vs backward vs syncs); works where CUPTI is dead (WSL2).
- `research/mamba_zigzag_baseline/tools/perf_parity_diff.py`: Diffs two `--loss-dump` .npz parity dumps (first action divergence, max |loss delta| over identical-action prefix).
- `research/mamba_zigzag_baseline/tools/perf_mamba_ssm_probe.py`: Probes mamba-ssm fused kernels — import/ABI health, step() autograd verdict (silent None grads), L=1 vs L=500 fused/pure speed.
- `train_mamba_rl.py` flags: `--seed --max-steps --no-checkpoint --compile --profile-dir --loss-dump --perf-warmup` (inert by default; bars/sec printed at end; compile is opt-in — fails the 1e-4 loss-parity gate at ~1.5e-3).
