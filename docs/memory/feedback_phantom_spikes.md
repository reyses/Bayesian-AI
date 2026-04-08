---
name: NT8 Phantom Spikes — Fake Edge Warning
description: NT8 exported data contains phantom spikes that create artificial z_se extremes. All data must come from Databento.
type: feedback
---

NT8 exported data contains phantom price spikes that don't exist in real market data.
These spikes create artificial z_se extremes → easy reversion trades → inflated backtest PnL.

**Evidence (2026-04-03)**:
- NT8 data: nightmare ticker = +$4,350 over 29 days
- Clean Databento data: nightmare ticker = -$2,427 over 29 days
- TradeCNN $1,609/day OOS was on NT8 data → FAKE

**Why:** User discovered this during clean data rebuild. Template system found 4,287 patterns on NT8 data but only 1,381 on clean data — the 2,906 difference was phantom spikes.
**How to apply:** NEVER use NT8 exported data for backtesting or training. Only Databento. If someone references old baselines from NT8 data, flag them as unreliable.
