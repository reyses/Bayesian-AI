# tools/_labeling — Human & Auto Labeling

> Virtual folder. Files live in `tools/` root.

## Human-in-the-loop (interactive GUIs)
| Tool | Purpose |
|---|---|
| **[`trade_marker.py`](../trade_marker.py)** | Click START / click END on price chart to mark trades. Auto-detects direction. Saves `DATA/regime_seeds/seeds_YYYY-MM-DD_multi.json` with MFE/MAE/duration |
| [`peak_marker.py`](../peak_marker.py) | Single-click peak marking on price chart |
| [`dmi_peak_marker.py`](../dmi_peak_marker.py) | Peak marking on DMI/Volume chart (not price) |
| [`regime_labeler.py`](../regime_labeler.py) | Step through I-MR regime segments, Y/N per segment |
| [`draw_levels.py`](../draw_levels.py) | Interactive horizontal S/R lines on 1m chart |
| [`swing_inspector.py`](../swing_inspector.py) | Grade continuous swing groups (~10 trades per snapshot) |
| [`seed_inspector.py`](../seed_inspector.py) | Step through auto I-MR seeds, accept/reject each |

## Auto-seed generators
| Tool | Purpose |
|---|---|
| [`auto_swing_marker.py`](../auto_swing_marker.py) | ZigZag-based swing detector calibrated from human seeds |
| [`auto_levels.py`](../auto_levels.py) | Auto S/R detection (rules from 31 weeks of hand-drawn levels) |
| [`auto_seeds_day_chart.py`](../auto_seeds_day_chart.py) | Chart auto seeds vs physics peaks for a day |
| [`pivot_seed_scanner.py`](../pivot_seed_scanner.py) | Oracle pivot scanner (REAL reversals vs FAKEOUTS) |
| [`pivot_seed_scanner_mtf.py`](../pivot_seed_scanner_mtf.py) | MTF: 1s pivots + 1m exhaustion + 15s state |
| [`build_peak_seeds.py`](../build_peak_seeds.py) | Convert pivot scanner CSV → auto-swing seed JSON |
| [`imr_to_seeds.py`](../imr_to_seeds.py) | I-MR regimes → seed JSON (calibrated on human seeds) |
| [`generate_training_labels.py`](../generate_training_labels.py) | Forward-PnL labels at 5/15/30/60/180 bars |
