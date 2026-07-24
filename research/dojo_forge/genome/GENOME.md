# GENOME v1  (anchor-TF = 1m; provisional fork defaults 2026-07-24 — owner review pending)
[G1.0] DEFAULT = HOLD. Never-bail beat every cut policy at N=23,378. Exit only on positive reversal evidence — never on drawdown or giveback alone.
[G1.1] IF adverse excursion on a clean-entry trade THEN HOLD — on top-decile entries losers cut themselves; the dip is usually survivable.
[G1.2] IF giveback/retrace AND anchor-TF trend intact (velocity sign persists, band position holds) THEN HOLD — even on a large giveback.
[G1.3] IF the trade is in profit and retraces THEN HOLD — cutting-and-banking loses; the heavy right tail pays for the giveback toll.
[G1.4] Accelerating loss ALONE is not an exit — usually a survivable dip. Exit only if it coincides with confirmed structural reversal (G1.8).
[G1.5] Ignore 5s-level wiggles: 5s is substrate noise, not signal. Anchor every exit decision on 15s/1m/5m structure.
[G1.6] A single-frame turn signal is not a reliable exit — turns live in paths. Require multi-bar, multi-TF confirmation before exiting.
[G1.7] Winners are captured by DURATION, not timing. While the anchor-TF trend persists, holding one more bar dominates exiting.
[G1.8] EXIT on confirmed structural reversal: the anchor-TF (1m) breaks prior swing structure against your position — a break, not a pullback — or its velocity reverses sign AND holds across multiple bars (a durable flip, never one bar).
```

### Token estimate of the injected block
Rule text (10 lines + `# GENOME v1  (anchor-TF = 1m; provisional fork defaults 2026-07-24 — owner review pending)` header, incl. newlines) = **1,395 characters**
(measured via `wc -c`). Heuristic: **chars/4 (prose-dominant** — only a handful of
numeric tokens like `N=23,378`, `15s/1m/5m`) ≈ **~349 tokens**. Under the 400
target, well under the 600 cap. (For reference, the numeric-dense chars/1.65
heuristic would give ~845; it does not apply here — these lines are English
sentences, not number grids.)

---

# WHAT WE KNOW — the numbers (briefing; clean-provenance only)
**Exclusion discipline (mandatory read).** Everything below is measured on tape
INDEPENDENT of the gen-0 forge curriculum. Three tempting sources are DELIBERATELY
EXCLUDED because they are circular — measured on the very ai_cusp/gen-0 episodes the
teacher is being graded on: (1) the `tiered_effectiveness` pts/ep and capture ratios,
including the oracle-vs-5m exit gap and the ~87-pt oracle ceiling; (2) the
`teacher_why` interrogation; (3) every gen-0 truth-file oracle statistic, including
"median oracle exit ≈ minute 15." They may be RIGHT — but citing them to justify
rules that are then graded on the same episodes is exactly the self-deception
`RIDE_EDGE_GATE_SPEC.md §0` exists to prevent. If you want those numbers, MEASURE
them on a clean held-out slab first.

## B.1 — THE EDGE: it's ride-length, not entry timing or exits
The direction/entry problem is essentially solved, and it is NOT where the money is.
The calibrated combiner reaches pooled OOS AUC 0.676 across 55 streams
[DISTILLED/N=1.07M·OOS-25+26]; its top decile converts to +3.86 pts/trade @5m ($7.72),
CI[+2.48,+5.06], +3.26 net of friction [econ_conversion/top-decile·OOS]. But a
direction classifier ALONE is not a live strategy — AUC 0.864 yet every TP/SL grid
loses OOS, info ceiling ~83% on V2 entry features [MEMORY§4/V2-OOS]. And it is not in
the exits either: a blind LLM exit does NOT beat a dumb 5m hold overall (delta +3.9,
CI[−1.0,+9.3], not significant) [doc098/200eps·nonce-audited].

What DOES separate is the RIDE-vs-FADE split. The ride family aligns with truth at
0.76–0.85 (FREIGHT 0.854, RIDEMOM 0.810, RIDEAGAINST 0.789, RIDECALM 0.781); the
pure-fade family is ANTI-aligned at 0.17–0.30 (KILLSHOT 0.172, FADEMOM 0.206) — the
naive fade is 71–83% WRONG on direction, and invertible [DISTILLED/OOS-25+26]. The
one place an LLM exit measurably helped was on winners: +19.5 pts, CI[+8.3,+32.1],
significant — while it LOST −9.6 on wrong-side trades [doc098/200eps]. Translation:
the harvestable alpha is letting winners run, not calling tops. And the edge is
durable — combiner shelf-life median 37 weeks on 8-week windows, 57 on 16-week, and
right-censored so the true life is longer [overfit_decay/OOS].

## B.2 — WHAT KILLS YOU: cutting (early exits)
This is the expensive lesson, measured at maximum power. On top-decile entries the
cut side is CLOSED: at N=23,378 engagements over 282 test days, EVERY dumb stop
X=8..48 nets −6.9…−3.4 t/ep with all CIs ≤0; stop+re-entry −4.50 [−9.55,+0.86]; and
even a perfect path-veto only converges back toward never-bail (−0.72 [−1.53,+0.08])
[doc107/N=23,378·282d]. Never-bail IS the frontier. Mechanically, a stop banks
+107…+140 on the wrongs but bleeds −212…−278 on dipped goods — and dipped goods are
25% of the tape, 58.5% of all good trades [doc107]. The natural mix is 49.7% wrong /
43.2% good / 7.1% dead-band; the earlier "dumb stops win" headline was a 1:1
class-balance artifact [doc107/vs-doc100].

Every overlay repeats the pattern. Per-trade fixed stop ≈ −$31/day; "cut at −$100"
rejected because 76% of −$100 legs recover [MEMORY§4/L5-OOS]. Intraday session-P&L
stop −$79/day, CI[−154,−22], significant LOSS — 81% of stopped OOS days recover
[MEMORY§4/L5-OOS]. Vol-adaptive exit thresholds −$112/day OOS [MEMORY§4]. Pyramid
attenuation (C15) had AUC 0.883 yet EVERY recall budget lost, −$29…−$155/day
[MEMORY§4]. Cut-and-bank a winner LOSES at every level: hold−cut EV is positive
everywhere because the heavy right tail pays for the ~1R giveback toll [MEMORY§4].
The ONE structurally-optimal binary exit is the R-trigger (reversal exit), which
recovers ~1R off the low — that is precisely WHY fixed-dollar overlays lose: they
fire before the reversal does [MEMORY§4/§5].

## B.3 — TIMING FACTS
Honest state: **the exit-timing distribution is NOT cleanly measured — measure it,
do not assume it.** The only oracle-exit timing we have lives in the gen-0 truth
files and is CIRCULAR (excluded above). What IS clean: entries land at leg phase
~0.64 and the oracle-exit ceiling on the label-turn population is only ~23% vs the
50–80% target — the ±1–2 min turn is THE binding constraint [docs089-092/label-turn·OOS].
On giveback behavior the numbers are hard and clean: 76% of −$100 legs recover
[MEMORY§4], 81% of stopped OOS days recover [MEMORY§4], and 58.5% of good trades dip
adversely before paying [doc107]. The predictive window is SHORT — combiner edge is
real at 1–5 min and its CIs blow out past 15 min [econ_conversion/OOS]. Key nuance:
a short PREDICTION horizon is not a short HOLD horizon. The signal predicts a few
minutes ahead; the winner itself runs far longer. Do not exit just because the signal
went quiet.

## B.4 — SIGNAL RELIABILITY
Two hard rules from measurement. First, **5s is substrate, not signal** — inherently
noise; anchor every read at 15s/1m/5m [MEMORY§4]. Second, **turns live in paths, not
snapshots.** 46 static turn detectors plus a 409-dim F-space snapshot were scored on
±2-min turn detection and NONE beat the 0.43 density null: best real precision 0.31
(TURN-CLIMAX), best coverage 0.30 (RENKO) at precision 0.17, EXIT-KMDR leads by only
−0.2 min [docs089-092/label-turn·OOS]. The full 409-dim vector even LOST to 4 trivial
context features (AUC 0.638 vs 0.685, below the 0.05 house bar) [phold_exit_model/OOS].
So a single-frame "turn detected" is unreliable by construction — the turn is a
sequential/path object, not a snapshot.

Which streams earned their keep (all OOS, with CIs): B9, the during-trade
remaining-amplitude regressor, +$66/day @K=5, CI[+41,+94] (K=10 +$31, K=30 +$15, both
now significant) [MEMORY§5/51d-OOS]. B10 vol-regime sizer +$69/day, CI[+7,+144], OOS
AUC 0.949 [MEMORY§5]. The B-stack bad-day shave +$175/day, CI[+98,+269] — the real
bad-day mitigation [MEMORY§5]. As raw direction separators, OHLC-01 (AUC 0.841, N=619)
and PROP-TURN-P (0.689, N=131k) lead [DISTILLED/OOS]. Graveyard: ATR-09 is a pure
inverter (0.500), ORB-02 (0.436) was a lookahead artifact, RSI-06 is dead (0.515)
[DISTILLED, doc045].

## B.5 — REGIME FACTS
The one clean, counter-intuitive regime fact: **zigzag WANTS volatility.** The B10
sizer's action is INVERTED from intuition — boost size 1.3× when P(high-vol)≥0.5, cap
to 0.7× when P(low-vol)≥0.7 — and it composes multiplicatively with B9, OOS AUC 0.949,
+$69/day CI[+7,+144] [MEMORY§5]. The λ̂ trend-persistence term carries a SELECTIVE
edge: +0.070 alignment in the CALM regime, negligible +0.016 in momentum — most of
the ride-alignment gain comes from the λ̂ flip, not the raw term [DISTILLED,doc084/OOS].
Adding λ̂ flips 59.6% of anti-aligned fade fires and lifts label agreement 0.26→0.54,
+28pp OOS [doc084/OOS]. Beyond vol-regime and λ̂, regime-level exit/skip fixes have
NOT paid: hour-of-day skip not significant, no useful day-clustering [MEMORY§4].
