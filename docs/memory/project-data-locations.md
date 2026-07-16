---
name: project-data-locations
description: "Where data and the WSL venv live after the 2026-07-16 disk reorganization — raw dumps on D:, venv on WSL ext4, ATLAS in-repo"
metadata: 
  node_type: memory
  type: project
  originSessionId: 49f1ab8b-f170-41ec-955f-86beb538417f
---

**Data/venv locations after the 2026-07-16 reorganization (post 0-bytes-free incident):**
- **WSL venv**: `/home/reyses/venvs/bayesian-ai` (ext4). The old `.venv_wsl` inside the
  OneDrive repo is DELETED. All launchers (`run_sweeps*.sh`, leg_clock `_*.sh`,
  mamba_zigzag_baseline tools, `train_mamba_rl{,_seq}.py` WSL respawn) point at the new
  path. Wheel matrix preserved by byte-copying site-packages (mamba_ssm 2.2.4 +
  causal_conv1d 1.5.0.post8, torch 2.6.0+cu124, py3.12); freeze snapshot at
  `docs/reference/venv_freeze_2026-07-16.txt`.
- **Raw source dumps**: `D:\Bayesian-AI-data\` — `DATA/RAW` (Databento GLBX dumps),
  `DATA/RAW_NT8` (NT8 CSV exports), `archive/DATA` (old pipeline exports),
  `Desktop_RAW` (the databento_to_atlas ingest source, formerly `Desktop/RAW`).
  Path constants updated in `config/settings.py`, `DATA/pipeline/databento_to_atlas.py`,
  `tools/sourcing/convert_nt8_csv_to_parquet.py`, order_flow_ablation builders.
- **ATLAS stays in-repo** under the [[project-atlas-keep-policy]] (OHLCV 1s→1D + SFE
  feature stores only). `artifacts/` and `checkpoints/` were emptied 2026-07-16
  (Moises-approved, incl. research_A segment .pth) — treat their contents as ephemeral.

**Why:** C: hit 0 bytes free 2026-07-15; OneDrive was syncing 6.7 GB of CUDA libs and
~7 GB of raw dumps for no benefit; venv on /mnt/c is a measured Mamba perf drag.

**How to apply:** new raw data goes to `D:\Bayesian-AI-data\`; never create venvs or
multi-GB regenerable artifacts inside the OneDrive repo; when a script can't find raw
data, check these D: paths before assuming data loss.
