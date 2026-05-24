# Quant Engineering Audit Report — Bayesian-AI Yield Optimization

**Date**: May 23, 2026
**Subject**: Codebase Audit and Yield Optimization Strategy

## Executive Summary

After a comprehensive audit of the Bayesian-AI quant architecture (`core_v2`, `live`, and `docs`), the system currently suffers from **structural unprofitability** in a live environment despite showing positive theoretical OOS results. The fundamental bottleneck is that the **average OOS yield per trade ($4.87)** is lower than the **modeled live friction ($6.00 per leg)**. 

To achieve a higher per-hour yield, the system must transition from high-frequency, low-conviction scalping to higher-conviction, duration-based holds using LightGBM filters and the V2 feature set. Furthermore, critical latency and hardcoded data structure bugs must be patched immediately before deploying capital.

---

## 1. Execution & Architectural Bottlenecks (Critical Path)

### 1.1 Structural Friction vs. Yield Collapse
- **Issue**: The `docs/ROADMAP.md` indicates an OOS yield of $4.87 per trade (4,644 trades = $22.6K). However, `live/l5_decider.py` correctly defines `FRICTION_PER_LEG = 6.0` ($4 commission + $2 slippage). 
- **Impact**: The strategy is operating with negative mathematical expectation in live markets. Every trade executed mathematically loses ~$1.13 net of fees. 
- **Fix**: The system must *drastically reduce trade frequency* and *increase conviction*. The strategy should exclusively execute setups where expected MFE (Maximum Favorable Excursion) > $15 to clear the friction hurdle.

### 1.2 The "B9" 25-Second Latency Trap
- **Issue**: The L5 Hybrid pipeline (`L5_HYBRID_PIPELINE_SPEC.md`) uses a B9 K=5 model to decide sizing (pyramid vs. cut) **25 seconds** (5 bars × 5s) after entry.
- **Impact**: 25 seconds is an eternity in NQ/MNQ order book physics. Waiting 25 seconds to size up misses the highest-velocity portion of the pivot breakout. Conversely, waiting 25 seconds to cut a toxic trade subjects the portfolio to unnecessary Maximum Adverse Excursion (MAE).
- **Fix**: Retrain B9 on K=1 or K=2 (5 to 10 seconds). The sizing decision must be synchronous with structural validation, ideally using L2/L3 order book imbalance rather than delayed price trajectories.

### 1.3 C#/Python Bridge Latency & V2 Streaming
- **Issue**: As noted in the L5 spec, V2 feature computation is currently missing from the NT8 C# strategy (`ZigzagRunnerHybrid_v1.0.0-RC.cs`). The Python sidecar either sends empty feature sets or relies on slow parquet disk lookups (`DATA/ATLAS_NT8/FEATURES_5s_v2/`).
- **Impact**: Lookup-based feature extraction in live trading guarantees severe slippage, pushing the already precarious $4.87 gross yield into deeper negative territory.
- **Fix**: Implement real-time, streaming computation of the 184D V2 feature vector directly in the NT8 C# layer or via a zero-latency memory-mapped file (mmap) from Python.

### 1.4 Hardcoded Tick and Dollar Values (Systemic Risk)
- **Issue**: Files like `live/engine_v2.py` and `live/l5_decider.py` hardcode global constants like `TICK = 0.25`, `TV = 0.50`, and `DOLLAR_PER_POINT = 2.0` (MNQ parameters).
- **Impact**: If the engine is attached to NQ (Tick Value = $5.00) or ES (Tick Value = $12.50), internal PnL accounting, B9 threshold logic (e.g., `B9_CUT_THRESHOLD_NORMAL = 5.0`), and risk management will fail silently, leading to catastrophic mis-sizing.
- **Fix**: As requested in `ARCHITECTURE.md`, refactor all internal engine representations to use **Ticks-only accounting**. Dollar values must only be computed at the boundary (e.g., UI Dashboard, trade logger) using the specific asset's `TickSize` and `TickValue` from the NT8 instrument profile.

---

## 2. Algorithmic & Modeling Bottlenecks

### 2.1 CNN Collapse and the LightGBM Pivot
- **Issue**: The CNN pivot classifier completely collapsed on V2 features (AUC 0.51 - random guessing).
- **Impact**: The current engine is trading "blind" relative to the structural state, leading to the massive over-trading problem (4,644 trades) noted above.
- **Fix**: Prioritize the deployment of `JULES_PIVOT_LGBM_V2.md`. LightGBM is inherently better suited for tabular, non-spatial 184D financial data. **Do not ship unless the LightGBM Test AUC exceeds 0.68.** An AUC of <0.68 will fail to generate the >$15/trade conviction required to beat live friction.

### 2.2 SFE Drift Between V1 and V2
- **Issue**: The V1 to V2 migration revealed a significant SFE (Statistical Field Engine) drift in `z_se` (Pearson 0.87, median shift 0.47σ). 
- **Impact**: Legacy tier engine thresholds (e.g., `ROCHE = 2.0`) cause 5-minute NMP entries to drop by 88% on V2 features. 
- **Fix**: Recalibrate all z_score and velocity-based thresholds immediately using the cumulative distribution function (CDF) of the new V2 features to match the exact historical fire-rates of V1.

---

## 3. Recommended Roadmap for Maximum Yield

To transition this system into a high-yield, production-ready state, execute the following steps in strict order:

1. **Purge Hardcoded Values (Immediate):**
   - Refactor `engine_v2.py` and `l5_decider.py` to ingest `TickSize` and `TickValue` dynamically via `NT8Client`. Convert all internal threshold logic (like B9's $5.0 cut threshold) into tick-space logic (`20 ticks`).

2. **Deploy the LightGBM V2 Filter (1-2 Days):**
   - Execute the `train_pivot_lgbm_v2.py` script. Optimize for a probability threshold that yields the top quintile of expected PnL. The goal is to slash the 4,644 trade count by 80%, retaining only the ~900 trades that average >$20 per trade.

3. **Streamline the C#/Python Bridge (2-3 Days):**
   - Rewrite the NT8/Python IPC. V2 features must be streamed via memory-mapped files or fast TCP. Drop the delayed lookup protocol.

4. **Retune the B9 Sizer (Post-LGBM):**
   - Once LGBM is active, retrain B9 on the newly filtered subset of trades. Reduce the K-horizon from 5 bars (25s) to 2 bars (10s) to aggressively pyramid early momentum and cut adverse chop before slippage accumulates. 

**Conclusion:** The codebase is mature conceptually, but the current iteration treats live friction as theoretical rather than physical. By moving to ticks-only accounting, replacing the failed CNN with a high-threshold LightGBM filter, and fixing the 25-second sizing delay, the system will pivot from a high-frequency bleeder to a low-frequency, high-yield extractor.

---

## 4. `core` vs `core_v2` Deprecation Audit

### 4.1 Current Status
The legacy `core` folder **cannot be completely deprecated yet**. 

According to `docs/JULES_v1_to_v2_migration.md`, the migration from V1 (`core`) to V2 (`core_v2`) is paused at **Phase 0b**. While `core_v2` successfully implements the 184D feature vector and passes all lookahead audits, the broader system is still deeply entangled with V1 features.

### 4.2 Ongoing Dependencies
A sweep of the codebase reveals **132 direct import dependencies** on `core.*` spread across the architecture:
- **`live/` (31 imports):** Files like `engine_v2.py`, `live_engine.py`, and `l5_decider.py` still import `core.features.FEATURE_NAMES`, `core.ledger`, and `core.statistical_field_engine`.
- **`training/` (27 imports):** The primary training loop `nightmare_blended.py` and all CNNs (`cnn_flip.py`, `cnn_hold.py`, `cnn_risk.py`) are strictly wired to the 79D/91D V1 feature vector from `core`.
- **`tools/` (74 imports):** The vast majority of EDA (Exploratory Data Analysis), simulation, and diagnostic charting tools are still coupled to V1 concepts.

### 4.3 Blocking Migration Phases
To safely delete the `core` directory, the following phases from the migration roadmap must be approved and executed:
1. **Phase 1 (Tier Engine Rewire):** `training/nightmare_blended.py` must be refactored to read V2 columns. Furthermore, because V2's `z_se` calculation drifted from V1, legacy entry thresholds (e.g., `ROCHE = 2.0`) must be recalibrated. (Currently, keeping literal V1 thresholds on V2 features drops 5-minute entries by 88%).
2. **Phase 2 (Live Engine Migration):** `live/engine_v2.py` must swap its reading path from `DATA/ATLAS_LIVE/FEATURES_5s` to the `_v2` directory.
3. **Phase 3 (CNN/Model Retraining):** The existing neural networks must be retrained to accept the 184D V2 native input, or alternatively, use the `core_v2.v1_compat` shim.

### 4.4 `core_v2` is NOT Self-Contained
A structural review of the `core_v2` directory confirms that it is **not self-contained**. It was built as a partial overlay rather than a complete rewrite, meaning it relies heavily on legacy modules remaining in `core`:
- **Shared Missing Dependencies:** The `core_v2/exits/` module (comprising 11 files like `band_exit.py`, `take_profit.py`, `giveback.py`) continuously imports classes from `core.exit_engine` and `core.trading_config`. These modules do not exist in `core_v2`.
- **Cross-Contamination:** Even files that *were* ported to V2 still mistakenly import their V1 counterparts. For example, `core_v2/ledger.py` explicitly calls `from core.features import N_FEATURES` and `from core.engine_signals import ...`, breaking self-containment.

**Recommendation:** Do not delete the `core` directory. Approve and execute Phase 1 of `JULES_v1_to_v2_migration.md` first, re-baseline the OOS simulation to confirm we still sit near the -$164/day IS anchor, and slowly migrate the live logic. To ever fully deprecate `core`, `core_v2` must either reimplement or cleanly migrate missing modules like `exit_engine.py` and `trading_config.py`.