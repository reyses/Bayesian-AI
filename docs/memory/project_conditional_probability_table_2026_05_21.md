---
name: conditional-probability-table-2026-05-21
description: The conditional probability table — a 4-entry diagnostic if-this-then-that table over zigzag-leg events, built 2026-05-21; diagnostic asset, not yet in production
metadata:
  type: project
---

A diagnostic research thread: an empirical conditional-probability table that
answers "if-this-then-that" questions about zigzag-leg events.

**Why:** User reframe 2026-05-21 — "I don't want to catch more, I want to be
able to DIAGNOSE that." The goal is not a new trading model but a probability
table — the ATR zigzag pinpoints events (chop, fakeout, leg age), the table
attaches probability estimates. A local-LLM (Llama) layer to chain the
if-this-then-that reasoning was discussed and back-burnered.

**How to apply:** This is a DIAGNOSTIC asset — it has NOT changed production
code. When a future session proposes a leg-structure trading rule, check the
table first; several "obvious" edges are already measured and turn out weak.

Four entries (reports in `reports/findings/oos_bad_days/2026-05-21_*`, tools
at `tools/`):
- Entry 1 `conditional_probability_table.py` — chop begets chop: after 3
  consecutive low-range legs P(next low) = 65% vs 33% base. Chop is sticky.
- Entry 2 `leg_chop_survival.py` — early chop forecasts a ~2x longer leg
  (fixed-early-window; v1 was confounded).
- Entry 3 `trend_continuation.py` — after a chop/fakeout the preceding trend
  continues only ~53-56% vs ~50-52% base (~+4pp). WEAK. The mechanical
  counter-trend entry is slightly wrong, but the directional edge is too
  small for a B1-7 gate to fix profitably (B1-B6 augmentation already tested
  at -$76/day; direction is not the lever).
- Entry 4 `leg_age_hazard.py` — leg-death hazard is HUMP-shaped: peaks at a
  ~5-min danger window, then declines. NOT monotone exhaustion — a pure
  time-stop is unsupported; only a danger-window-aware check is backed. The
  steep early arm is partly the min_bars=36 (3-min minimum leg) floor.

Method discipline for this thread: [[zigzag-conditional-table-confounds]]
(five confounds caught and fixed during the arc).

Candidate next entry (entry 5): amplitude-expended survival — P(leg
continues) given it has already moved P points, the price-distance analog of
entry 4's time hazard. Grew out of [[oos-bad-days-2026-05-21]] but is its own
thread (diagnose, not lift bad days).
