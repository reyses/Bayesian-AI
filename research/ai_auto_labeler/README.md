# AI Auto Labeler

This research folder contains the pipeline and tools used to algorithmically generate the "Golden Dataset" of perfect hindsight trades, designed to mimic the manual visual labeling process.

## Architecture

The system mimics the human visual workflow of identifying structural swings and picking the optimal entry and exit points for a trade.

### 1. `pipeline/ai_labeler.py`
This is the core engine that processes a single day of 1-second `ATLAS` data and generates a JSON file of perfect trades.
- **Data Source**: Reads 1-second `.parquet` OHLCV data from `DATA/ATLAS/`.
- **Swing Identification**: Uses a centered cubic spline (`UnivariateSpline(s=len*30)`) to generate a smoothed geometric representation of the day's price action (the "orange line" in `cusp_marker.py`).
- **Wiggle Filtering**: Identifies all local minima and maxima (turns) in the cubic spline. It then filters out any turns that have a structural prominence of less than `7.0` points to eliminate noise ("wiggles").
- **Trade Optimization**: For each valid turn:
  - Finds the absolute extreme price (highest high or lowest low) within a 5-minute window of the turn to act as the **0-MAE Entry**.
  - Scans forward 60 minutes to find the absolute extreme price (Max-MFE) before the trade crosses a trailing stop threshold (or hits the end of the session). This acts as the **Perfect Exit**.
- **Output**: Saves the optimal trades (entry timestamp, exit timestamp, entry price, exit price, PnL) to `DATA/ai_cusp_picks/ai_picks_YYYY-MM-DD_multi.json`.

### 2. `pipeline/run_all_months.py`
A batch processing script designed to run the `ai_labeler.py` engine across the entire historical dataset.
- Iterates over 27 months of trading days (from 2024-01 to 2026-03).
- Launches `ai_labeler.py` as a subprocess for each day.
- Aggregates execution statistics and progress.

### 3. `tools/inspect_losses.py`
A diagnostic tool used to audit the output of the auto-labeler.
- Reads all the generated JSON files in `DATA/ai_cusp_picks/`.
- Identifies and aggregates any trades that resulted in a negative PnL ("losses").
- Provides a statistical breakdown of why these losses occurred (usually due to edge cases like the market closing abruptly, or 7-point structural moves that failed to hit an arbitrary 10-point winning threshold before reversing).

### 4. `pipeline/ai_labeler_v2.py` (structural launch labeling)
Rewrite of the labeling logic per the manual workflow. Fixes v1's **fixed 60-bar prominence** (which
truncated slow-forming trends and mismatched the 7pt-filter / 10pt-win criteria).
- **Cubic**: centered cubic (`cubic_utils.find_raw_turns`, N=20 — same as `cusp_marker`) → turns + curvature.
- **Entry (flat-zone best bar)**: the flat zone = the contiguous span where the smoothed price stays
  within `FLAT_BAND_PTS` of the turn (a broad rounded hump). Snap the entry to the real **1s** extreme in
  that span (low=LONG / high=SHORT) → 0-MAE entry. (Sharp turns already sit on their extreme.)
- **Confirm (uncapped forward walk)**: walk forward from the entry, **no 60-bar cap**; skip sub-`TREND_PTS`
  wiggles; continue until a ≥`TREND_PTS` favorable move confirms the trend. Session ends first → no trade.
- **Reversal → flagged**: if price breaks back past the entry (adverse > `REVERSAL_TOL_PTS`) before
  +`TREND_PTS`, the region is written to `DATA/ai_cusp_picks/flagged/` for human inspection (should not
  happen if the entry is the true extreme — so it's a QC signal, not a silent drop).
- **Exit (mirror)**: the trend ends when it retraces ≥`TREND_PTS` from its running peak (a real opposing
  move) or the session ends; snap the exit to the 1s best bar in that peak's flat zone → best exit.
- **Constants** (named): `CUBIC_N=20`, `TREND_PTS=7.0`, `FLAT_BAND_PTS=3.0`, `REVERSAL_TOL_PTS=1.0`.
  `TREND_PTS`/`FLAT_BAND_PTS` are the next candidates to make **regime-adaptive** (via the
  `research/recovery_dynamics` amplitude envelope — the "normal swing" breathes 30-50% by regime).
- Run: `python research/ai_auto_labeler/pipeline/ai_labeler_v2.py --day 2024_03_04` (or `--month 2024_03`).
- First test (2024-03-04): 56 trades, all ≥7pt positive (median 19pt, max 64), MAE~0, 1 reversal flagged.

## Interaction with Other Scripts

The output of this pipeline (`DATA/ai_cusp_picks/`) is designed to be loaded directly into the main visualization and manual labeling UI (`tools/viz/cusp_marker.py`). 
By passing the `--load-ai` flag to `cusp_marker.py`, the AI's algorithmically generated trades are loaded as read-only overlays, drawn with exact entry-to-exit background spans, allowing the user to visually inspect and verify the AI's logic against their own manual intuition.
