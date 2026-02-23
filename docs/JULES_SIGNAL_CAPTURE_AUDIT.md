# Jules Signal Capture Audit

> **Objective:** Document the end-to-end signal capture pipeline, from raw fractal discovery to final trade execution, and detail the audit mechanisms used to track missed opportunities (False Negatives) and execution quality.

---

## 1. Fractal Discovery & Oracle Labeling

The signal capture process begins with the **Fractal Discovery Agent** (`training/fractal_discovery_agent.py`), which scans historical data using a top-down hierarchical approach.

### Top-Down Scan
1.  **Macro Level**: Starts at the largest timeframe (e.g., 4H, 1H) to identify major structural patterns (`ROCHE_SNAP`, `STRUCTURAL_DRIVE`).
2.  **Drill-Down**: For each macro pattern found, the agent defines a time window to scan at the next smaller timeframe (e.g., 15m -> 5m -> 1m -> 15s).
3.  **Leaf Level**: The finest resolution (typically 15s or 1s) is where actionable trade signals are generated.

### The Oracle (Ground Truth)
For every detected pattern, the agent consults an internal "Oracle" (`_consult_oracle`) to determine the *true future outcome* of that pattern. This labeling is crucial for training and audit but is **never used for live trading decisions**.

-   **Lookahead**: The Oracle looks ahead `N` bars (defined by `ORACLE_LOOKAHEAD_BARS`).
-   **Classification**: It calculates the Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
-   **Labels**:
    -   `MEGA_LONG` / `MEGA_SHORT`: High conviction move (Reward:Risk > `ORACLE_HOME_RUN_RATIO`).
    -   `SCALP_LONG` / `SCALP_SHORT`: Moderate move (Reward:Risk > `ORACLE_SCALP_RATIO`).
    -   `NOISE`: Price did not move significantly or stopped out immediately.

These labels serve as the "Ground Truth" against which the system's decisions are audited.

---

## 2. The Entry Gatekeeper Pipeline

Once a raw pattern is discovered, it must pass a series of **Gates** in the **Orchestrator** (`training/orchestrator.py`) to become a valid trade signal.

### Gate 0: Headroom & Structural Rules
*   **Purpose**: Filter out patterns in unfavorable market conditions or "Nightmare Fields".
*   **Logic**:
    -   **Rule 1**: Must be a recognized pattern type.
    -   **Rule 2**: `z_score` must be significant (> 0.5 sigma).
    -   **Rule 3**: Approach Zone (0.5 - 2.0 sigma) requires strong trend confirmation (`ADX`, `Hurst`).
    -   **Rule 4**: Extreme Zone (> 2.0 sigma) requires "Headroom" (macro context must not be exhausted).
*   **Audit Log**: Rejected candidates are logged with reason `gate0`, `gate0_noise`, `gate0_r3_struct`, etc.

### Gate 1: Clustering Match (Pattern Recognition)
*   **Purpose**: Ensure the pattern matches a known, profitable template in the **Pattern Library**.
*   **Logic**:
    -   Calculates the Euclidean distance between the pattern's 14D feature vector and the nearest cluster centroid.
    -   **Threshold**: `dist < 4.5` (previously 3.0).
*   **Audit Log**: Rejected candidates are logged with reason `gate1` (No Match).

### Gate 2: Bayesian Brain (Probability)
*   **Purpose**: Use historical performance to probability-weight the signal.
*   **Logic**:
    -   Queries the **Quantum Bayesian Brain** for the `probability` of the matched template.
    -   **Threshold**: `probability > 0.05` (exploration mode) or higher for production.
*   **Audit Log**: Rejected candidates are logged with reason `gate2`.

### Gate 3: Fractal Conviction (Belief Network)
*   **Purpose**: Ensure multi-timeframe alignment.
*   **Logic**:
    -   Queries the **Timeframe Belief Network** (10 workers from 1h down to 1s).
    -   **Threshold**: Global conviction must meet `MIN_CONVICTION` (typically 0.65).
*   **Audit Log**: Rejected candidates are logged with reason `gate3`.

---

## 3. The Exit Logic

Trades are managed by the **WaveRider** (`training/wave_rider.py`) and **Orchestrator Worker** (`training/orchestrator_worker.py`), using physics-based gates and dynamic belief updates.

### Physics-Based Gates
1.  **Fourier Gate (Minimum Hold)**:
    -   Prevents premature exits due to noise.
    -   **Logic**: Must hold for at least `period / 2` unless `PnL < -0.5 * StopLoss`.
2.  **Laplace Gate (Urgent Exit)**:
    -   Detects kinetic exhaustion.
    -   **Logic**: If `PnL > 0` and `damping > 0.8` (momentum decay), exit immediately.

### Dynamic Exits (Belief-Based)
The **Timeframe Belief Network** continuously monitors the trade:
-   **Tighten Trail**: If conviction drops or wave matures -> standard trail or tighter.
-   **Widen Trail**: If conviction is high and wave is fresh -> widen trail to capture trend.
-   **Urgent Flip**: If belief direction flips -> close immediately (`urgent_flip`).

---

## 4. Audit Artifacts

The system generates detailed CSV logs in `checkpoints/` (and sharded in `run_logs/`) to enable "Profit Gap Analysis".

### `oracle_trade_log.csv` (Traded Signals)
Contains every trade **actually taken** by the system.
-   **Key Fields**: `entry_time`, `exit_time`, `actual_pnl`, `oracle_label` (Truth), `capture_rate`.
-   **Purpose**: Measure execution quality and capture efficiency.

### `fn_oracle_log.csv` (Missed Opportunities / False Negatives)
Contains every signal that was **rejected by a gate** but was labeled `MEGA` or `SCALP` by the Oracle (i.e., a real missed move).
-   **Key Fields**: `oracle_label`, `fn_potential_pnl` (money left on table), `gate_blocked` (which gate killed it), `workers` (snapshot of belief network).
-   **Purpose**: Identify which gate is too strict. If `workers` agreed with the direction but `gate1` blocked it, we have a "Model Gap" (missing template).

### `signal_log.csv` (Decision Matrix)
Contains **every candidate signal** evaluated, regardless of outcome.
-   **Key Fields**: `gate` (passed/rejected reason), `oracle_pnl` (potential), `micro_z`, `macro_z`.
-   **Purpose**: Full funnel analysis of the decision pipeline.

---

## 5. Profit Gap Analysis (The Golden Path)

The system calculates the **Ideal Profit** (Golden Path) to quantify performance:
1.  **Ideal Profit**: Sum of `oracle_potential_pnl` for all valid signals in a sequential, non-overlapping timeline.
2.  **Actual Profit**: Sum of `actual_pnl` from traded signals.
3.  **Leakage Buckets**:
    -   **Missed (Gate Blocked)**: Value of valid signals rejected by gates (`fn_oracle_log`).
    -   **Wrong Direction**: Value lost taking a trade opposite to the Oracle label.
    -   **Exit Leakage**: Difference between `oracle_potential_pnl` (MFE) and `actual_pnl` on winning trades (leaving money on the table).

**Goal**: Minimize the gap between Actual Profit and Ideal Profit by tuning gates and improving exit logic.
