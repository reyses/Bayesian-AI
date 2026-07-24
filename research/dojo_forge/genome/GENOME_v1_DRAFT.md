# GENOME v1 — DRAFT: the Trader's Handbook (for owner ratification)

**Status:** DRAFT. Does NOT replace `genome/GENOME.md`. Awaiting owner sign-off.
**Date:** 2026-07-23 · **Author:** Claude (distiller) · supersedes the 3 gen-0 seeds.

Two parts. **PART A** is the executable rule set (the injected block). **PART B** is
the "what we know" briefing — the numbers you would give a human trader before their
first session, clean-provenance only. Owner's 3 rule-forks remain open (now 4).

---

# PART A — THE RULES

## Why this draft exists
Gen-0 (3 naive seeds) exited too early: median exit minute 7 vs oracle 15, and
LOST to never-bail — 26.3 vs 38.9 pts/ep, delta −12.6 CI[−22.0, −3.4] SIGNIFICANT
(`reports/tiered_effectiveness_2026-07-23.md`). The G0.3 interrogation
(`reports/teacher_why_2026-07-23.md`) showed the model already *reasons* toward
HOLD when reasoning is enabled — every one of the 10 worst premature exits, when
replayed with the reasoning bypass removed, resolved to HOLD via G0.3. So the
failure is NOT that the model wants to exit; it is that `p_exit>0.5` fires without
the reasoning gate. **These rules therefore sharpen WHEN to exit and make EXIT the
rare, evidence-gated branch — they add zero exit eagerness.** The gate rewards
ride-length capture vs a 5m hold (`RIDE_EDGE_GATE_SPEC.md` Amendment v2.1 §1), so
every rule aims at holding longer without cutting survivable dips.

---

## INJECTED BLOCK (this — and only this — goes in every frame's system prompt)

```
# GENOME v1
[G1.0] DEFAULT = HOLD. Never-bail beat every cut policy at N=23,378. Exit only on positive reversal evidence — never on drawdown or giveback alone.
[G1.1] IF adverse excursion on a clean-entry trade THEN HOLD — on top-decile entries losers cut themselves; the dip is usually survivable.
[G1.2] IF giveback/retrace AND anchor-TF trend intact (velocity sign persists, band position holds) THEN HOLD — even on a large giveback.
[G1.3] IF the trade is in profit and retraces THEN HOLD — cutting-and-banking loses; the heavy right tail pays for the giveback toll.
[G1.4] Accelerating loss ALONE is not an exit — usually a survivable dip. Exit only if it coincides with confirmed structural reversal (G1.8).
[G1.5] Ignore 5s-level wiggles: 5s is substrate noise, not signal. Anchor every exit decision on 15s/1m/5m structure.
[G1.6] A single-frame turn signal is not a reliable exit — turns live in paths. Require multi-bar, multi-TF confirmation before exiting.
[G1.7] Winners are captured by DURATION, not timing. While the anchor-TF trend persists, holding one more bar dominates exiting.
[G1.8] EXIT on confirmed structural reversal: anchor-TF (1m/5m) breaks prior swing structure against your position — a break, not a pullback.
[G1.9] EXIT on a durable regime flip: anchor-TF velocity reverses sign AND holds across bars — momentum turning to reversion, not one bar.
```

### Token estimate of the injected block
Rule text (10 lines + `# GENOME v1` header, incl. newlines) = **1,395 characters**
(measured via `wc -c`). Heuristic: **chars/4 (prose-dominant** — only a handful of
numeric tokens like `N=23,378`, `15s/1m/5m`) ≈ **~349 tokens**. Under the 400
target, well under the 600 cap. (For reference, the numeric-dense chars/1.65
heuristic would give ~845; it does not apply here — these lines are English
sentences, not number grids.)

---

## PROVENANCE (footnotes — NOT injected; owner audit only)
CLEAN = measured on real held-out tape independent of the gen-0 curriculum.
CIRCULAR = derived from the ai_cusp / gen-0 curriculum episodes themselves →
INADMISSIBLE as a rule basis; used only as motivation where noted.

| Rule | Source | Population | Flag |
|---|---|---|---|
| G1.0 | doc-107 SYNTHESIS; DISTILLED.md | N=23,378 engagements, 282 test days, natural-mix top-decile combiner entries, OOS 2025+26, day-block CIs | **CLEAN** |
| G1.1 | doc-107 §3 "the law"; MEMORY §4.4 | same 282-day powered frontier; "cut/bail a loser LOSES at every drawdown level" | **CLEAN** |
| G1.2 | amends seed G0.3; doc-107 §1 (dipped goods = 58.5% of goods, recover) | 282-day frontier. NOTE: the teacher_why interrogation that *motivated* keeping G0.3 is CIRCULAR (gen-0 episodes) — used as motivation only; the rule's numeric basis is doc-107 CLEAN | **CLEAN** (motivation CIRCULAR, excluded) |
| G1.3 | MEMORY §4.3 (cut-and-bank a winner LOSES; hold−cut EV positive at every level; giveback toll ~1R) | L5 OOS 51-day + B-stack validation | **CLEAN** |
| G1.4 | amends seed G0.2; doc-107 §1 + MEMORY §4 (cutting accelerating losers loses — dipped goods recover) | 282-day frontier | **CLEAN** (neuters a dangerous naive seed) |
| G1.5 | MEMORY §4 ("5s level is inherently noise — substrate not predictor; anchor at 15s/1m/5m") | V2 architecture finding | **CLEAN** |
| G1.6 | turn_detection_audit; MEMORY §5 (docs 089–092): 46 static detectors + 409-dim snapshot fail the 0.43 chance null; turns are sequential/path objects | train 2024 / test 2025+26, ±2min label-turn | **CLEAN** |
| G1.7 | doc-107 §3 (ride is the only significant edge lever); RIDE_EDGE_GATE §1 (metric = ride capture vs 5m-hold) | 282-day frontier + gate spec. NOTE: the 38.9-vs-26.3 magnitude is CIRCULAR (curriculum-measured); only the DIRECTION (never-bail > early-exit) is cited, and it replicates doc-107 CLEAN | **CLEAN** (magnitude CIRCULAR, excluded) |
| G1.8 | MEMORY §4/§5 (R-trigger reversal exit = the ONLY structurally-optimal binary exit; recovers ~1R off the low) | L5 OOS | **CLEAN** |
| G1.9 | DERIVED — owner named "regime flip" as a warranted exit. NOT independently measured as an exit trigger; reasoned extension of R-trigger (G1.8) using frame velocity/reversion channels | none — no clean population | **UNVALIDATED** (see open Q3) |

### Seed disposition
- G0.1 (adverse excursion → HOLD) → **kept + strengthened** as G1.1.
- G0.2 (multi-family + accelerating loss → EXIT) → **neutered** into G1.4. As
  written, G0.2 is the seed most likely to *cause* the premature-exit failure:
  "accelerating loss" is exactly the survivable dip doc-107 says never to cut.
  G1.4 keeps the multi-family/reversal spirit but forbids exiting on the loss alone.
- G0.3 (giveback + trend intact → HOLD) → **kept + operationalized** as G1.2
  (defines "trend intact" in frame terms: velocity-sign persistence + band position).

---

# PART B — WHAT WE KNOW: the numbers
*A first-session briefing built ONLY from clean-provenance measured facts. Every
number carries an inline `[source/population]` tag. Read the exclusion note first.*

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

---

## COMBINED TOKEN ESTIMATE (Part A + Part B)
Stated mix, with an honest cross-check (the char heuristics and a word-count estimate
disagree, so a range is reported rather than a false-precision point).

- **Part A injected block** (10 rules + header): 1,395 chars → chars/4 ≈ **349 tokens**.
- **Part B body** (exclusion note + B.1–B.5): **7,179 chars, 1,051 words** (measured
  via `wc`).
  - Pure prose baseline: chars/4 ≈ **1,795 tokens**.
  - Requested 70/30 blend (0.70·chars/4 + 0.30·chars/1.65): 1,256 + 1,305 ≈ **2,561
    tokens** — but this OVER-counts here. The chars/1.65 divisor is calibrated for
    solid number grids (tables of figures); Part B is English sentences with embedded
    numbers, which tokenize far closer to prose.
  - Word-count cross-check (1,051 words × ~1.33 tokens/word) ≈ **~1,400 tokens** — the
    most reliable estimate for mixed prose, and it sits BELOW even the pure-prose char
    baseline, confirming the 1.65 blend is inflated for this content.
  - **Best estimate Part B: ~1,400–1,800 tokens.**
- **TOTAL Part A + Part B ≈ 349 + (1,400–1,800) ≈ ~1,750–2,150 tokens.** Centered on
  the ~2,000 target and inside the 2,400 hard cap under every method; only the
  known-inflated 1.65-blend (~2,910 total) would exceed the cap, and it does not apply
  to sentence-form text. Part B alone lands in / just above the requested 1,200–1,600
  band by the word-count method.

---

## OPEN QUESTIONS (rule-level calls only the owner can make)

1. **Retire G0.2 entirely, or keep the neutered G1.4?** G1.4 still lets the model
   exit on "structural reversal," which under-specified could re-open the
   too-early-exit door. Safer alternative: delete the accelerating-loss concept
   outright and rely only on G1.8 (structural reversal) + G1.9. Do you want a
   loss-based exit clause in the genome at all?

2. **Is G1.9 (regime flip) admissible?** It has NO clean measured basis — it is a
   reasoned extension, and it may just be a second name for G1.8's structural
   reversal, doubling the exit surface with no evidence. Options: (a) keep as-is,
   (b) fold into G1.8 and drop G1.9, (c) hold it out until measured. My
   recommendation: (b) or (c) — one evidence-backed exit trigger beats two, one
   of which is a guess.

3. **How should "anchor TF" be pinned?** G1.2/G1.7/G1.8/G1.9 all say "anchor-TF"
   but do not fix which TF. MEMORY says anchor at 15s/1m/5m and 5s is noise. Do
   you want the genome to name a SPECIFIC anchor (e.g., 1m for structure, 5m for
   regime), or leave "anchor-TF" abstract so the model reads it per frame? A fixed
   anchor is more auditable; an abstract one is more flexible but less falsifiable.

4. **Handbook injection: every frame vs anchor-only?** Two options with a real
   cost/reliability trade-off. (a) Inject the FULL handbook (Part A + Part B, ~1,781
   tokens) into every frame's system prompt — the numbers are always in-context and
   the model can't "forget" why a rule exists, but you pay ~1,781 tokens/frame across
   thousands of frames × generations (the dominant recurring ctx cost). (b) Inject
   only Part A (~349 tokens/frame) and expose Part B once at episode/window start,
   trusting the model to RETAIN the numbers across the context window — 5× cheaper,
   but retention is unverified and degrades as the window fills. My lean: (b) with a
   measured retention check (probe the model mid-window on 2–3 Part-B facts before
   committing), or a hybrid — Part A always, plus a 1-line numeric anchor per rule
   (e.g. append "[never-bail frontier N=23,378]" to G1.0). Which cost are you willing
   to pay: tokens every frame, or a retention risk you must test?
