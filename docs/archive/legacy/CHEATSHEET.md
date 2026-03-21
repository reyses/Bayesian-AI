# Bayesian-AI Command Cheatsheet

## Training
```bash
python training/trainer.py --fresh                      # Full pipeline (wipe + IS + OOS + Replay)
python training/trainer.py --forward-pass               # IS + OOS + Replay (keep checkpoints)
python training/trainer.py --forward-pass --skip-oos    # IS only
python training/trainer.py --oos                        # OOS only (reuse library)
python training/trainer.py --train-only                 # Phases 2-3 only
python training/trainer.py --strategy-report            # Phase 6 only
python training/trainer.py --data DATA/ATLAS_1DAY       # Fast validation (~3s)
python training/trainer.py --data DATA/ATLAS_1WEEK      # Screening (~30s)
```

## Live Trading
```bash
python -m live.launcher --dry-run                       # Paper trade (NT8 connected)
python -m live.launcher --dry-run --no-gui              # Headless paper trade
python -m live.launcher                                 # REAL MONEY (careful!)
python -m live.launcher --dry-run --ping-pong           # Continuous wave-riding
python -m live.launcher --dry-run --long-only           # Long bias only
python -m live.launcher --dry-run --short-only          # Short bias only
python -m live.launcher --dry-run --anchor-tf 1m        # Change anchor TF
python -m live.launcher --dry-run --max-daily-loss 100  # Tighter loss limit
python -m live.launcher --replay-only                   # Replay only + parity report (no NT8)
```

## Tools
```bash
python tools/session_overlay.py --data DATA/ATLAS_OOS --trades checkpoints/oos_trade_log.csv
python tools/session_overlay.py --data DATA/ATLAS --trades checkpoints/oracle_trade_log.csv
python tools/analyze_gates.py                           # Gate threshold analysis
python tools/analyze_gates.py --apply                   # Write thresholds to JSON
python tools/run_analytics.py                           # Re-run analytics only
python tools/standalone_research.py --data DATA/ATLAS_1WEEK  # Research harness
python tools/standalone_research.py --data DATA/ATLAS_1WEEK --start 5  # Skip modules
python tools/checkpoint_viewer.py                       # Inspect pattern library
python tools/pattern_map.py                             # Signal funnel viz
python tools/trade_visualizer.py                        # Trade overlay on waveform
python tools/nt8_to_parquet.py INPUT OUTPUT             # NT8 export -> ATLAS
```

## Data Paths
```
DATA/ATLAS/          # IS: 12 months (Jan-Dec 2025), 14 TFs
DATA/ATLAS_OOS/      # OOS: 2 months (Jan-Feb 2026)
DATA/ATLAS_1DAY/     # Fast validation (Jan 2, 2025)
DATA/ATLAS_1WEEK/    # Screening (Jan 2-10, 2025)
```

## Reports (after forward pass)
```
reports/is_report.txt          # IS summary
reports/oos_report.txt         # OOS summary
checkpoints/trade_analytics.txt    # IS analytics (t-test, ANOVA, OLS)
checkpoints/oos_analytics.txt      # OOS analytics
reports/run_history.csv        # All runs comparison
reports/live/parity_report_*.txt   # Phase 7: OOS vs Replay parity
```

## Key Checkpoints
```
checkpoints/oracle_trade_log.csv   # IS trade log
checkpoints/oos_trade_log.csv      # OOS trade log
checkpoints/tuning.json            # Hot-reloadable tuning params
checkpoints/gate_thresholds.json   # Gate thresholds
```

## NT8 Bridge
```
Indicator file: docs/NT8_BayesianBridge.cs
Deploy to:      C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Indicators\
Default port:   5199
```
