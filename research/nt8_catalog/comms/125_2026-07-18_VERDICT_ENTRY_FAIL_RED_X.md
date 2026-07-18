# VERDICT — Entry-fail Red X (doc 122): letter-PASS, spirit-THIN; the lane closes honestly
**Doc:** 125 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** FINAL
Sealed 2024-fit / single-shot on the 23,378-engagement test population
(reviewer external-consistency check: population count, class mix, and the
49.7% fail rate all match the independently-built powered-frontier numbers).

## 1. THE finding — entry P is ANTI-predictive on terminal economics
P-only terminal-good AUC on test = **0.4961** (below coin flip). Moises'
diagnosis is fully vindicated: P was trained on direction-agreement and
carries ZERO terminal-outcome signal — marginally negative. Every deployment
assumption that "higher P = better trade outcome" is wrong at the terminal
horizon.

## 2. But the cure is not at the entry snapshot
Full model (tier, det, leg geometry, lambda-hat, tod, vol, P): test AUC
0.5135; increment over P-only +0.0174 = noise by the house bar. The three
pre-registered points all clear their CIs (letter-PASS) but the real
yardstick is vs BASE: **+2.1pp good-rate (43.2->45.3%) at the cost of 65% of
volume**. Even keeping 5% of trades only reaches 46.9%. The ~50% fail regime
is NOT escapable at entry with existing features.

## 3. Feature autopsy
- **lambda-hat: AUC 0.500 flat** on terminal outcomes — it predicts
  direction-alignment (doc 084) but nothing about where the trade ENDS.
- Leg geometry, sig_with_leg, pivot_age: ~0.497-0.500. Nothing.
- The thin signal lives in **nmp9_tier + det identity** (which kind of setup
  fired): RIDEMOM goods 0.515 vs **CASCADE goods 0.247** (CASCADE entries end
  bad 3:1 — small-N caveat, quantile-trap risk, note-not-promote).

## 4. RULING
The entry-fail lane is **CLOSED for snapshot features** — same law, third
appearance: turns live in paths (089-092), cuts don-t beat holding (107), and
now terminal outcomes are not knowable at entry (this doc). The fail problem
routes where everything else routed: the RIDE/path side. The forge genome and
any future Mamba inherit nmp9_tier + det as context channels (already in the
spec) — not as filters.
GRAVEYARD ENTRY: entry-time terminal filtering on existing features — max
+2.1pp good-rate at -65% volume; P anti-predictive (0.496); lambda-hat flat.
