# VRAM Grid Scramble Contamination Report

Date: 2026-06-11

## Overview
A bug was discovered in `core_v2/FPS/forward_pass_system_vram.py` and the VRAM pipeline path inside `train_gpu_research_A.py` where the CNN grid was blindly sliced using `reshape(-1, 8, 23)` instead of passing through the correct `assemble_v2_grid` helper. 

This scrambled the channel meanings for any convolutional nets trained via this path.

## Contaminated Checkpoints
The following checkpoints in `training/rl_engine/checkpoints/` were trained using the scrambled grids and are marked as suspect:

- `research_A_segment_1.pth`
- `research_A_segment_1_latest_epoch.pth`
- `research_A_segment_2_latest_epoch.pth`

## Resolution
- Deduplicated `VRAMForwardPassSystem` to inherit cleanly from `ForwardPassSystem`.
- Updated `train_gpu_research_A.py` to use `assemble_v2_grid(v2_matrix)` to ensure parity with the native CPU evaluation engine.
- Re-training should proceed from fresh initialization or prior non-VRAM checkpoints.
