# tools/_regret — Regret & Post-Run Analysis

> Virtual folder. Files in `tools/` root.

## Generic regret / review
| Tool | Purpose |
|---|---|
| [`regret_analysis.py`](../regret_analysis.py) | Post-run: how much each exit reason left on the table |
| [`trade_review.py`](../trade_review.py) | Comprehensive post-run trade review |
| [`analyze_pnl.py`](../analyze_pnl.py) | Quick PnL analysis on trades CSV |
| [`analyze_gates.py`](../analyze_gates.py) | Oracle-driven optimal gate thresholds |

## Distribution / mode
| Tool | Purpose |
|---|---|
| [`daily_hourly_pnl.py`](../daily_hourly_pnl.py) | Daily mode/median/mean + hourly contribution |
| [`pnl_mode_buckets.py`](../pnl_mode_buckets.py) | 10 tick-aligned PnL ranges, find mode |
| [`pnl_tier_distribution.py`](../pnl_tier_distribution.py) | Per-tier + per-day aggregates + histogram |
| [`hourly_oos_report.py`](../hourly_oos_report.py) | PnL by hour for live-trading comparison |
| [`sunday_hourly_eda.py`](../sunday_hourly_eda.py) | Which Sunday hours are profitable |

## Pathology / concentration
| Tool | Purpose |
|---|---|
| [`giveback_analysis.py`](../giveback_analysis.py) | After peak PnL, bars until break-even |
| [`peak_capture_regret.py`](../peak_capture_regret.py) | How much of 20-bar peak do we capture? |
| [`pattern_regret_report.py`](../pattern_regret_report.py) | Regret by pattern assignment |
| [`pareto_loss_concentration.py`](../pareto_loss_concentration.py) | 80/20 — which trades dominate losses? |
| [`winner_maxout_loser_rehab.py`](../winner_maxout_loser_rehab.py) | Trail stop winners + flip losers |
| [`big_loss_entry_signature.py`](../big_loss_entry_signature.py) | Entry features predicting catastrophic losses |
| [`big_loss_physics.py`](../big_loss_physics.py) | When catastrophic losers tip their hand (MAE / running PnL) |
| [`loser_cliff_eda.py`](../loser_cliff_eda.py) | Natural "dead" timescale where loser peak stalls |
| [`loser_physics.py`](../loser_physics.py) | When flipping direction would rescue losers |
| [`measure_bad_trade_holds.py`](../measure_bad_trade_holds.py) | Winners vs losers hold-time cohort |
