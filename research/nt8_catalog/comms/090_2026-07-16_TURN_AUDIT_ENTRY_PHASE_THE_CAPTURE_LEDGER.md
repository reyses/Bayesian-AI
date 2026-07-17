# Turn-detection audit + entry phase — the capture ledger vs the 50-80% target
**Doc:** 090 · **Date:** 2026-07-16 · **Author:** Claude · **Status:** FINAL
**Trigger:** Moises — "we know where the flips exist, inspect directly what signals
we have on them and how accurate we are at detecting them" + the target: "if we
catch 80% of the leg (10% lost at start, 10% at exit) it's golden; even 50% is
worthwhile."

## 1. Turn-detection audit (tools/turn_detection_audit.py; ~40k interior label
## turns, test 2025+26; chance-anchored)
- Chance precision at ±2 min = **0.43** (turns are dense: 43% of RTH sits within
  2 min of one).
- **Every one of the 34 evaluated streams has precision BELOW chance**
  (0.10-0.25). Best direction-recall: RENKO-24 0.30 (on a 123k-fire hose),
  SAR-23 0.13, TUNNEL 0.12, ROUND 0.12, DOW 0.12 — all with sub-chance precision.
- Verdict: the corpus is a MID-LEG engine. It is structurally quiet at the turn.
  We own no turn-timing instrument at the ±2-min scale.

## 2. Entry phase (top-decile fires inside same-direction labels, N=31,220)
- **Median entry phase 0.64** — the leg is ~2/3 spent at entry. Mode 0.95
  (the most common entry is at the leg's END — the momentum tautology's bill).
- Only **3%** of entries land in the first 10% of the leg (Moises' start budget);
  11% in the first quarter.

## 3. The ledger vs the target
| stage | measured | target budget |
|---|---|---|
| entry loss | ~64% of leg forfeited (median) | ≤10% |
| exit instrument | none above chance at ±2m | ≤10% |
| realized capture | 5m hold 1.4%; oracle-exit ceiling from THESE entries 23% | 50-80% |
**Even a perfect exit cannot reach the 50% floor from current entries.** The
50-80% zone requires entries near the turn — and since the labels CHAIN, the
turn that ends leg k IS the entry of leg k+1: exit-timing and entry-timing are
ONE instrument. The program's binding constraint is a turn detector accurate to
~±1-2 min. Prior art warning: the 2026-07-07 probe-turn signal failed the house
bar (null-anchored gap 0.036 < 0.05) — this is the hardest problem here.

## 4. Back-to-back structure (Moises' question)
Top-decile fires arrive in same-direction BURSTS (median 0.8 min between
same-direction fires); only 16% of consecutive pairs are direction flips
(~16 flips/day; flip-gap median 4 min, p75 15). Episodes (burst-collapsed) are
the natural swing-detection unit — a follow-up if needed.

## 5. Lanes launched (Moises: "we have options")
1. **Old-school brackets** (Sonnet worker, running): SL×TP×TMAX grid on the
   calibrated entries; sealed 2024-selection readout; graveyard caveat noted —
   prior stop verdicts were on the OLD system's entries, never on this population.
2. **Turn Catalog** (Opus worker, running): mine the 463 articles for
   turn-timing concepts, verbatim citations, dossier-style — the exit research
   done "just like the articles."
3. **Sequential model (Mamba)**: spec pending; lanes 1-2 become its baselines.
