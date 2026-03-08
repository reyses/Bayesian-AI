# Standalone Research Tool — Developer Guide
> For Claude Code: how to add new analysis modules to `standalone_research.py`

## Overview
`standalone_research.py` is a 6K-line research harness that:
- Loads ATLAS bar data across configurable timeframes
- Runs lettered analysis modules (A through R, extensible)
- Generates plots to `tools/plots/standalone/`
- Saves text report to `tools/standalone_report.txt`
- Each module generates its OWN data from raw bars — no dependency on trade logs

## CLI
```bash
python tools/standalone_research.py --data DATA/ATLAS_1WEEK --base-tf 15m
python tools/standalone_research.py --data DATA/ATLAS --start S          # skip to Analysis S
python tools/standalone_research.py --data DATA/ATLAS --start S --full   # include 16D pipeline
```

Key flags:
- `--data PATH` — ATLAS data directory (default: DATA/ATLAS)
- `--base-tf 15m` — base timeframe for analysis
- `--months 2025_01 2025_02` — specific months to load
- `--context-days 21` — lookback window for I-MR computation
- `--analysis-days 7` — how many days to analyze (use 120 for large samples)
- `--start X` — skip to analysis letter X (uppercase)
- `--full` — enable full 16D fractal pipeline (after letter analyses)
- `--cache file.npz` — save/load feature matrix cache
- `--top N` — top N factors to show in screening

## Data Available to All Modules
After initial load (before any analysis letter), these are in scope:

```python
base_df        # DataFrame: ATLAS bars for base TF (e.g. 15m)
price_imr      # Dict: I-MR computation results (close, MR, UCL, LCL, regimes)
regime_ids     # Array: regime ID per bar
regime_meta    # List[dict]: per-regime stats (start, end, direction, etc.)
bar_indices    # Array: bar indices with oracle labels
mfes           # Array: MFE per bar (oracle forward move)
maes           # Array: MAE per bar (oracle adverse move)
directions     # Array: oracle direction per bar (+1/-1)
PLOTS_DIR      # str: output directory for plots
_start_at      # str: which letter to start from (uppercase)
args            # argparse namespace (all CLI flags)
```

After `--full` gate (step 7+):
```python
all_dfs        # Dict[tf_label -> DataFrame]: all 14 TF DataFrames
all_physics    # Dict[tf_label -> list]: ThreeBodyQuantumState per bar per TF
stacked_16d    # ndarray: 16D feature matrix stacked across TFs
```

## How to Add a New Analysis Module

### Step 1: Find the insertion point
Analyses go BEFORE the `if not args.full:` gate (line ~4764).
Insert after the last `else: print("[SKIP] Analysis R")` block.

### Step 2: Follow the pattern
```python
    # =====================================================================
    if _start_at <= 'S':
        print(f"\n{'='*70}")
        print(f"  ANALYSIS S: YOUR TITLE HERE")
        print(f"{'='*70}")

        # Your analysis code here
        # Use base_df, regime_meta, mfes, directions, etc.
        # Generate data FROM the loaded bars — don't read trade logs

        # Save plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        # ... plot ...
        fig.savefig(os.path.join(PLOTS_DIR, 'analysis_s_your_name.png'), dpi=150)
        plt.close(fig)
        print(f"  [saved] analysis_s_your_name.png")

    else:
        print(f"  [SKIP] Analysis S (--start {_start_at})")
```

### Step 3: Key rules
1. **Self-contained data**: Each module generates its own data from `base_df` /
   `all_physics` / `regime_meta`. Don't read external CSVs.
2. **Plot to PLOTS_DIR**: `os.path.join(PLOTS_DIR, 'analysis_X_name.png')`
3. **Print results**: All print() goes to both stdout and `_report_buf` (tee'd)
4. **Use `Agg` backend**: matplotlib must use non-interactive backend
5. **Letter ordering**: Letters must be sequential. After R comes S, T, U, etc.
6. **Guard with `_start_at`**: `if _start_at <= 'S':` so `--start T` skips it
7. **Error handling**: Wrap risky code in try/except, print error, continue

### Step 4: If module needs multi-TF data
If your analysis needs physics from multiple TFs, it must go AFTER the
`if not args.full:` gate (inside the full 16D pipeline section, line ~4775+).
There, `all_dfs` and `all_physics` are available.

Alternatively, load specific TFs yourself:
```python
from tools.research.data import load_atlas_tf, compute_tf_physics
df_1h = load_atlas_tf(args.data, '1h', months=args.months)
physics_1h = compute_tf_physics(df_1h)
```

## Subpackage: tools/research/
Helper modules used by standalone_research:

| File | Purpose |
|------|---------|
| `data.py` | TF_HIERARCHY, load_atlas_tf(), compute_tf_physics(), extract_16d() |
| `imr.py` | compute_price_imr(), detect_regimes(), compute_regime_oracle() |
| `screening.py` | Factor screening, regression R2, pad_to_fixed_depth() |
| `seeds.py` | SeedPrimitiveLibrary, inflection detection, adaptive splitting |
| `plots.py` | All plot functions, resolve_plots_dir(), PLOTS_DIR global |

## Output Locations
- Plots: `tools/plots/standalone/` (or `tools/plots/standalone_{data}_{days}d/`)
- Report: `tools/standalone_report.txt` (or inside PLOTS_DIR as `research_report.txt`)
- Both created automatically by the harness

## Existing Analyses (A-R)
| Letter | Topic | Key Output |
|--------|-------|------------|
| A | Price I-MR + regime detection | Regime chart, UCL/LCL plot |
| B | Regime oracle (MFE/MAE per regime) | Direction accuracy by regime |
| C | Screening: which 16D factors predict MFE? | Factor ranking table |
| D | Factor interaction (2-way) | Interaction heatmap |
| E | Regression: R2 by factor subset | Best predictor combos |
| F | Temporal stability: factor R2 over time | Rolling R2 chart |
| G | Regime clustering (DBSCAN on regime features) | Cluster scatter |
| H | Direction prediction (logistic) per regime type | Accuracy by type |
| I | Seed classification (20 primitives) | Shape gallery |
| J | Adaptive R2 sub-types + IQR quality gate | R2 distribution |
| K | Direction prediction (full model, 70.6% acc) | Feature importance |
| L | Seed shape transitions | Transition matrix |
| M | Price model fitting (92% R2) | Price vs predicted |
| N | Multi-TF seed alignment | TF agreement chart |
| O | Magnitude prediction | Predicted vs actual magnitude |
| P | Combined direction+magnitude model | Joint model performance |
| Q | Signed magnitude histogram + 192D profiles | Histogram + profile pairs |
| R | CNN pattern detection (Conv1D, 7 classes) | Confusion matrix |
| S | Exit trend guard — band conflict study | Band conflict chart |
| T | Partial bar robustness (live simulation) | Complete vs partial accuracy, per-TF staleness |
| U | Expected move confidence interval (dp/dt psychohistory) | CI coverage, direction from P50, ambition ratio, counter-trend context |
