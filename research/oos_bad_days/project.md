# Project: Lift the OOS Bad Days

DMAIC research project. Opened 2026-05-21 (autonomous run). Status:
**ANALYZE complete — converged on a negative; IMPROVE needs a user decision.**

## Define
Reduce the frequency and severity of negative-P&L OOS days for the L5 zigzag
stack. Sealed OOS = 51 days (2026-03-19..05-18). User framing (2026-05-21):
"lift the OOS bad days", measured "hour by hour, like a manufacturing site".

## Measure
- **14/51 OOS days negative** (−$3,570 total; 8 worst days = 85% of the
  bad-day loss; worst −$571). `tools/oos_bad_day_characterize.py`.
- **The deployed B-stack already shaves bad days +$175/day** (95% CI
  [+$98, +$269], significant); good-day delta −$9 (not sig). The stack's whole
  edge IS bad-day protection — the non-significant headline +$42/day hid this.
- Hourly decomposition (`tools/oos_hourly_characterize.py`): the strategy is a
  cash-session operation — ET hours 09–15 produce ≈100% of the $/day; the
  overnight block (16:00–08:00 ET) nets ≈$0 churn. Bad days = the RTH session
  failing to trend (chopping); damage spread across hours, no clock pattern.

## Analyze
Every day-level / session-level lever tested and FAILED (IS-discover /
OOS-confirm):
- **Hour-of-day skipping** — not significant on $/day; negative hours are
  near-zero churn, and which are negative is not IS→OOS stable.
- **Intraday cumulative-P&L session-stop** — fails like the per-trade stop:
  39% IS / 81% OOS recovery; on OOS made negative days 14→20, worst day
  −$571→−$844, −$79/day CI [−$154,−$22] SIGNIFICANT loss.
  `tools/oos_intraday_stop_analysis.py`.
- **Bad-day clustering** — daily P&L is iid day-to-day: lag-1 autocorr IS
  +0.06 (perm p=0.24) / OOS −0.01 (p=0.96). Prior-day P&L is not a signal.
  `tools/oos_bad_day_autocorr.py`.
- **Prior work**: DRS session-start macro prediction failed OOS; the per-trade
  drawdown stop was rejected (76% of −$100 legs recover).
- One real regularity: the chop regime (mean leg amplitude) IS autocorrelated
  day-to-day (IS +0.275 / OOS +0.485) — but it does not propagate into bad-day
  clustering, and B10 already exploits the vol regime.

## Improve
No day/session-level intervention beats the deployed B-stack. Chop is not
predictable ahead of time at the day level. The bad days are the irreducible
cost of a trend zigzag meeting choppy markets — and they are already the
B-stack's main job. Candidate avenues with a real prior (require a user
decision — not run autonomously):
1. **Causal ATR study** — a wider zigzag whipsaws less in chop (the user's
   original ATR interest). The FLAT ATR sweep was oracle-contaminated; this
   needs a causal streaming forward pass. Any ATR/B9 retrain should pick its
   levels from this, not the FLAT sweep.
2. **Strengthen B9** — per-leg continuous amplitude sizing is the only action
   template that has ever beaten the R-trigger (the L5 paradigm boundary).
Available now, no model: **RTH-only** (flat outside ET 09–15) — cuts daily
P&L std ~15% and ~3 OOS bad days, ≈$/day-neutral (not significant) — a
defensible pure risk control, not a fix.

## Control
Pending — depends on the Improve decision.
