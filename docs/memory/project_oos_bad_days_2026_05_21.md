---
name: oos-bad-days-2026-05-21
description: The OOS bad days resist every day/session-level fix — 4 levers ruled out. The deployed B-stack (+$175/day on bad days) is the validated mitigation.
metadata:
  type: project
---

The "lift the OOS bad days" research (2026-05-21 autonomous run; DMAIC project
`research/oos_bad_days/project.md`) converged on a strong negative.

**The fact.** The OOS bad days (14/51 days negative, −$3,570, 8 worst = 85% of
the loss) are NOT preventable by any day-level, session-level, or prior-day
signal. Bad days = the RTH cash session (ET 09–15, ≈100% of the $/day) failing
to trend — chopping. Four levers ruled out, rigorously (IS-discover /
OOS-confirm):
- hour-of-day skipping — not significant on $/day;
- intraday cumulative-P&L session-stop — fails like the per-trade stop (81%
  OOS recovery; −$79/day CI [−154,−22] SIGNIFICANT loss; negative days 14→20);
- bad-day clustering — daily P&L is iid day-to-day (lag-1 IS +0.06 p=0.24 /
  OOS −0.01 p=0.96); prior-day P&L is not a signal;
- (prior work) DRS session-start macro prediction failed OOS; per-trade
  drawdown stop rejected (76% recover).

**What works:** the deployed B-stack (B7/B9/B10) shaves bad days **+$175/day,
CI [+$98, +$269], significant** (the day-level MEASURE); good-day delta −$9
(not sig). The bad days are already the B-stack's main job. The
non-significant headline +$42/day hid this asymmetric protection.

**Why:** chop is not predictable ahead of time at the day level. One real
regularity — the chop regime (leg amplitude) IS autocorrelated day-to-day (IS
+0.275 / OOS +0.485) — does not propagate into bad-day clustering, and B10
already exploits the vol regime.

**How to apply.** Do NOT re-propose a day-level bad-day predictor, an intraday
P&L stop, a per-trade drawdown stop, or a prior-day-outcome filter — all
rejected with OOS evidence. The residual bad days are the irreducible cost of a
trend zigzag in choppy markets. The only avenues with a real prior: (1) a
CAUSAL ATR study — a wider zigzag whipsaws less in chop, needs a causal
streaming pass (the FLAT sweep is contaminated, see
[[flat-pipeline-cross-param]]); (2) strengthening B9 — per-leg continuous
amplitude sizing, the only action template that has beaten the R-trigger.
RTH-only is a defensible pure risk control (~15% less daily std, ~3 fewer bad
days, ≈$/day-neutral) but not a fix. Tools: `oos_bad_day_characterize.py`,
`oos_hourly_characterize.py`, `oos_intraday_stop_analysis.py`,
`oos_bad_day_autocorr.py`.
