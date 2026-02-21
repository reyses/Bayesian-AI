# JULES Implementation Details

This document details the implementation of the following instructions:
1.  **Snowflake Clusters** (JULES_SNOWFLAKE_CLUSTERS.md)
2.  **Fractal DNA Tree** (JULES_FRACTAL_DNA_TREE.md)
3.  **Golden Path Oracle** (JULES_GOLDEN_PATH_ORACLE.md)
4.  **PID Optimizer** (JULES_PID_OPTIMIZER.md)
5.  **Dynamic Exits** (JULES_DYNAMIC_EXIT.md)

## 1. Snowflake Clusters
**Goal:** Split the pattern clustering space into distinct LONG and SHORT branches to improve purity and eliminate direction gates.

**Changes:**
- **`training/fractal_clustering.py`:**
    - Modified `FractalClusteringEngine` to support split branches (`_fit_branch`).
    - Renamed original `create_templates` to `_fit_branch` and created a new `create_templates` that splits patterns by oracle direction (`z_score` based).
    - `PatternTemplate` now includes a `direction` field ('LONG' or 'SHORT').
    - Scalers are now split: `_long_scaler` and `_short_scaler`.
- **`training/orchestrator.py`:**
    - Updated Phase 3 saving logic to save `pattern_library_long.pkl`, `pattern_library_short.pkl`, and their respective scalers.
    - Updated `run_forward_pass` (Phase 4) loading logic to load these split files (with backward compatibility).
    - Updated **Gate 1** matching logic: candidates are routed to the LONG or SHORT branch based on their `z_score`.
    - Simplified the **Direction Gate**: direction is now inferred from the branch (or `z_score`), removing complex heuristics.

## 2. Fractal DNA Tree
**Goal:** Create a hierarchical "DNA" of market context (1h -> 15m -> 5m -> 15s) for every pattern to improve prediction accuracy.

**Changes:**
- **`training/fractal_dna_tree.py`:** Created new module containing `TreeNode`, `PatternDNA`, and `FractalDNATree` classes.
- **`training/orchestrator.py`:**
    - Integrated DNA Tree building at the end of Phase 3 (`FractalDNATree.fit` on manifest).
    - Integrated DNA matching in `run_forward_pass` (Gate 1).
    - Added `pattern_dna` logging to `oracle_trade_records` and `decision_matrix_records`.
    - Added a "TOP 10 DNA PATHS" section to the Phase 4 report.

## 3. Golden Path Oracle
**Goal:** Calculate the *true* achievable ideal profit by simulating a sequential (non-overlapping) trading path, rather than a theoretical parallel sum.

**Changes:**
- **`training/orchestrator.py`:**
    - Added `_compute_golden_path_ideal` function.
    - Implemented **Phase 2** logic: uses `dna_expectancy` (if available) as the weighting factor for the golden path, prioritizing high-quality fractal contexts over raw MFE.
    - Updated `PROFIT GAP ANALYSIS` in the report to use the golden path ideal.
    - Added "Parallel-all-signals upper bound" as an informational metric.
    - Added "Delta capture rate" (Actual PnL / Golden Path Ideal) as a key KPI.

## 4. PID Optimizer
**Goal:** Replace the inefficient Latin Hypercube DOE with a Bayesian Optuna optimizer for PID parameters (kp, ki, kd), while analytically deriving TP/SL/Trail from oracle stats.

**Changes:**
- **`training/doe_parameter_generator.py`:**
    - Refactored to use `optuna` (TPE sampler).
    - Removed legacy generation methods (`generate_latin_hypercube_set`, etc.).
    - Added `optimize_pid` method.
- **`training/orchestrator_worker.py`:**
    - Added `_analytical_exits` to compute TP/SL/Trail from template oracle stats (MFE/MAE).
    - Rewrote `_optimize_template_task` to use `_analytical_exits` (fixed) and `generator.optimize_pid` (search).
    - Updated `requirements.txt` to include `optuna`.

## 5. Dynamic Exits
**Goal:** Adjust trailing stops dynamically based on real-time belief network conviction.

**Changes:**
- **`training/timeframe_belief_network.py`:** `get_exit_signal` method was verified as implemented.
- **`training/wave_rider.py`:** `update_trail` method was verified as updated to accept `exit_signal` and adjust trails.
- **`training/orchestrator.py`:** Integration in the simulation loop was verified.

## Verification
- Dependencies (`optuna`, `pandas`, `torch`, etc.) installed.
- Syntax checks passed for modified files.
- Logic flow verified against instructions.
