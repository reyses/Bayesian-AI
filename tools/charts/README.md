# tools/_charts — Visualizations

> Virtual folder. Files in `tools/` root.

## Trade overlays
| Tool | Purpose |
|---|---|
| [`trade_visualizer.py`](../trade_visualizer.py) | Price waveform + entry/exit markers |
| [`session_overlay.py`](../session_overlay.py) | 1h candlesticks + adaptive Fibs + 1m trades |
| [`pattern_map.py`](../pattern_map.py) | Detected patterns overlaid on price waveform |

## DMI-focused
| Tool | Purpose |
|---|---|
| [`dmi_session_chart.py`](../dmi_session_chart.py) | DMI session chart |
| [`dmi_swing_plot.py`](../dmi_swing_plot.py) | DMI swing plot |
| [`dmi_imr_chart.py`](../dmi_imr_chart.py) | DMI I-MR chart (SPC-style) |
| [`dmi_peak_overlay.py`](../dmi_peak_overlay.py) | Human peaks overlaid on DMI + Volume |
| [`dmi_se_overlay.py`](../dmi_se_overlay.py) | DMI + SE bands + Volume overlay |

## Strategy-comparison / RM (in _rm_pivot)
See `_rm_pivot/README.md` for `chart_rm_trades.py`, `chart_strategy_comparison.py`, `chart_regression_z.py`, `chart_1s_trades.py`.
