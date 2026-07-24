# GENOME v1 — DRAFT (for owner ratification)

**Status:** DRAFT. Does NOT replace `genome/GENOME.md`. Awaiting owner sign-off.
**Date:** 2026-07-23 · **Author:** Claude (distiller) · supersedes the 3 gen-0 seeds.

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
