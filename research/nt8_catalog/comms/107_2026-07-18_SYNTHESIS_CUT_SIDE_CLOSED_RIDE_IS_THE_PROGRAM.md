# SYNTHESIS — the cut side is CLOSED at full power; the ride side IS the program
**Doc:** 107 · **Date:** 2026-07-18 (dawn) · **Author:** Claude (reviewer) · **Status:** FINAL
Night charter (Moises): "cut losers fast, let winners ride — be creative,
without Mamba." Four experiments later, the creative answer is that the adage
is HALF wrong for this system — and proving it cheaply just saved the
expensive half of the roadmap.

## 1. CORRECTION OF RECORD (supersedes doc 100's headline)
Doc 100's "+17.7 t/ep dumb-stop edge" was a **class-balance artifact**. The
198-episode set was 1:1 wrong:good (sized for LLM-fleet cost); the natural
tape is 49.7% wrong / 43.2% good / 7.1% dead-band. At full power (N=23,378
engagements, 282 test days, CIs ~7× tighter — population verified against the
doc-102 histogram exactly):

| policy | net vs never-bail (t/ep) | 95% CI |
|---|---|---|
| stop X=8..48 | −6.9 … −3.4 | ALL straddle/below 0 |
| stop+re-entry (frozen 103) | −4.50 | [−9.55, +0.86] |
| stop+veto (frozen 105) | −0.72 | [−1.53, +0.08] (≈never-bail, vetoes everything) |
| **never-bail** | **0 = the frontier** | — |

**No cut policy beats holding. Every point estimate is negative.** The stop
pays +107…+140 on wrongs and bleeds −212…−278 on dipped goods — and dipped
goods are 25% of the tape (58.5% of all goods; MORE common at natural mix,
not less).

## 2. The night's ledger (every mechanism, measured)
1. **Dumb stops** (all X): negative at natural mix. Graveyard rule #1
   ("cutting a loser loses at every level") re-confirmed at maximum power.
2. **Moises' re-entry** (103): the mechanism WORKS where predicted (halves
   the dipped-knife bleed) but re-buys bouncing losers; net negative.
3. **Distilled path-veto** (105): zero out-of-sample discrimination at the
   trigger (precision = base rate). The 10%-vs-54% discrimination the blind
   LLM agents showed lives in NON-path channels (fires/z/tier) — a real v2
   hypothesis, but moot given §1: even a PERFECT veto only converges the stop
   back to never-bail (the veto's −0.72 ≈ 0 is that convergence, observed).
4. **Blind LLM bail** (100): loses to stops on the balanced set — and the
   stops lose to holding on the real one.

## 3. The law (proposed graveyard entry)
**On top-decile combiner entries, losers cut themselves.** The entry quality
+ the label structure mean adverse excursions are mostly survivable dips;
damage-control mechanisms pay their toll on the 43% of good trades and
cannot earn it back on the wrongs. "Cut losers fast" is already implemented
by the ENTRY layer (not firing) and the R-trigger (reversal exit); every
additional cut overlay is net-negative. **The entire harvestable edge is in
"let winners ride"** — the one significant lever in every experiment this
week (+19.5 blind winners edge; B9 +$66/day CI[+41,+94]; ride family aligned
0.76-0.85).

## 4. Consequences (immediate, cheap)
1. **PRODUCTION_RUN_SPEC §6 rewritten**: cut-head = NONE (never-bail
   baseline). Mamba's exit head trains on ride-length ONLY. This SHRINKS the
   curriculum (one head, one objective) — directly easing Moises' GPU-cost
   concern: the expensive run now has a smaller, sharper job.
2. **Interim-NT8 implication**: the mechanical manager needs NO cut logic —
   entry (combiner P) + R-trigger + B9 sizing is the complete system;
   simplest possible NinjaScript surface (favors architecture A further).
3. **Wrong-dir dojo assets retained**: the sandbox, the natural-mix
   population, and the powered-frontier tool are the standing harness for
   ANY future cut proposal — the bar is now "beat never-bail at N=23k," and
   the graveyard says don't try without a new information source.

## 5. Provenance
103 (re-entry FAIL) → 105 (veto FAIL + under-power discovery) → 106 (powered
frontier, the decisive table) — all sealed, frozen-params-only evaluation,
day-block CIs, reviewer-verified chain, committed. Reports:
reports/wrongdir/{stop_reenter,veto_logistic,powered_frontier}.md.
