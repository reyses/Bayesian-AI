# P_hold during-trade logistic — HONEST DOUBLE NULL (both kill-points fired)
**Doc:** 089 · **Date:** 2026-07-16 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executor:** Opus worker (ladder trial #4). Design: Moises — run the entry logistic
DURING the trade on the full F-space, open-ended exit, "confidence should decay
that we are in a buy as it turns into a sell." Reviewer verification: FULL/BASE
AUC + fixed-5m capture reproduced from phold_rows.parquet EXACTLY.

## 1. What ran
13,292 train / 24,083 test engagements (entry-P ≥ frozen 2024 p90 = 0.760),
990k during-trade rows at 1-min cadence; target = active label still agrees with
entry direction. FULL model = 409 V2 feature cols + context; BASELINE = 4 context
features only (elapsed, drift_so_far, entry_P, trail_vol). Train 2024 / test 25-26.

## 2. The theory's SHAPE is confirmed...
Decay curves separate exactly as predicted: engagements whose label held stay at
P_hold ≈ 0.70-0.76 for an hour; engagements whose label flipped decay 0.61 → 0.42.
The confidence DOES drain out of a buy as it becomes a sell.

## 3. ...but BOTH kill-points fired
- **KILL-POINT A (F-space value): FIRED.** FULL OOS AUC 0.647 vs BASELINE 0.689 —
  the full 409-dim F-space is WORSE than "am I up + how long + entry conviction +
  local vol" (delta −0.042, CI [−0.054,−0.031]). In the first 10 minutes the
  baseline wins by −0.10-0.11 AUC (BASE hits 0.74-0.77 there). FULL only edges
  ahead after minute 40 (+0.017, noise-level). FULL's calibration is also
  overconfident at both ends (pred 0.15 → obs 0.36; pred 0.94 → obs 0.75).
- **KILL-POINT B (open-ended exit): FIRED.** Exit-on-P_hold-decay captures
  MEDIAN −2.75 to −3.00 pts vs fixed-5m hold +1.75. Cause: the 0.5-crossing is a
  LAGGING confirm — median +3.0 min AFTER the label flip (only 27% of crossings
  are early warnings). Exiting on it locks in the adverse excursion.

## 4. What the null teaches (three load-bearing findings)
1. **The exit is where the money is — precisely measured now.** Oracle exit
   (label end): median +27.5 pts, capture ratio 0.23. Fixed 5m: +1.75, ratio
   0.014. **The exit gap is ~16× the current harvest.** Everything the entry
   machinery earned is a toe in the water of this number.
2. **Trivial during-trade state is STRONG.** The 4-feature baseline P_hold is a
   free, causal, honest hold-state (AUC 0.74-0.77 in the first 10 min). It goes
   straight into the Mamba's input state at zero additional cost.
3. **A static snapshot cannot time the turn.** The full field state at minute τ
   knows less about the flip than the trade's own path — because the entry P
   already spent the field's information, and the flip is a PATH/sequence event.
   This null is close to a proof that the exit is a sequential-model problem —
   i.e., the Mamba's actual job, not a logistic's. (It also re-confirms the
   graveyard: every static exit rule has now lost, including a 409-feature one.)
4. Curio for later: FULL's top coefficients are dominated by the L5 level-
   distance family at 1h/4h + multi-TF vwap/price-mean — level geometry, Moises'
   VP-zone instinct — but multicollinear in a losing linear model; park it as a
   candidate feature FAMILY for the sequence model, not a finding.

## 5. Ladder trial #4 verdict
PASS on discipline (honest nulls, kill-points respected, numbers reproduce
exactly). One process defect now twice observed: workers launching their heavy
run in background and stopping early — future specs say RUN SYNCHRONOUSLY.

## 6. State / next
The stage-1 evidence file is complete: entry P (calibrated, converts to points),
hold-state baseline (free, strong early), exit = open sequential problem with a
16× measured ceiling. Next artifact: **Mamba handoff spec** — state = per-stream
fire vector + pooled entry-P + the 4-feature P_hold baseline + path; objective =
close the fixed-5m → oracle capture gap. Design doc for Moises' review first.
Artifacts: reports/phold_exit_model.md, phold_rows.parquet, phold_run.log,
tools/phold_exit_model.py.
