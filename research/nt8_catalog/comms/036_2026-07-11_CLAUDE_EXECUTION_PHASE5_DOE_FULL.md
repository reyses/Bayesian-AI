# Execution Report — Phase-5 DOE across ALL proposals (executed by CLAUDE)
**Doc:** 036 · **Date:** 2026-07-11 · **Author:** Claude (executor; AG out of usage) · **Status:** FINAL
**Supersedes** the ATR-09 lead in doc 035 (that lead FAILED replication — see §3).

## 1. Critical bug found FIRST (voids all prior Phase-5 numbers)
The entry-feature extraction indexed the FULL-session feature parquet (row 0 = 17:00 CT)
with `event_idx` that is RTH-relative (row 0 = 08:30 CT) for 22/24 dossiers. It read
**overnight features for daytime trades**. Proof (ATR-09 event #0, 2024-01-23):
`event_idx=4098` -> feature bar 23:15 the PRIOR night, `z_se=-0.38`; the correct RTH bar
is 14:11, `z_se=+1.46`. Every Phase-5 number AG produced AND my doc-035 null were
computed on misaligned overnight noise. New `tools/ag_phase5_doe.py` detects each
dossier's index convention (rth / full-session / exclude-brick) and aligns correctly.
Artifact: `tools/ag_phase5_doe.py`, evidence run pasted in commit.

## 2. DOE scope (leakage-free entry PhE-only, day-block bootstrap, 2024->2025)
Ran 22 dossiers. Excluded: RENKO-24 (brick-index space), ORDERFLOW-14 (2025/2026 only),
ADX-08 & SCALP-18 (thin, <30/yr). Forward-direction VALID branches (N>=30, >=20 days,
day-block CI excludes 0, |mode|>=2pts): **3** — ATR-09 INVERT, RSI-06 INVERT, SEASON-12 ACT.
Multiple-comparisons context: ~40 branch-tests, ~2 false valids expected by chance.
So forward-only "valid" is a CANDIDATE flag, not a finding. Full table:
`reports/AG_cat_00_PHASE5_DOE.md`.

## 3. Year-SWAP replication (train 2025 -> test 2024) — the decisive test
A real edge survives both directions; a lottery artifact does not.
| Candidate | forward (24->25) | swap (25->24) | verdict |
|---|---|---|---|
| **ATR-09 INVERT** | +10.3 CI[+2.0,+16.7] VALID | +1.0 CI[-2.3,+4.2] ns; features 13->1 (unstable) | **FAILS replication — not robust** |
| **RSI-06 INVERT** | +97.2 CI[+0.8,+235.6] (fat tail) | -3.4 CI[-59.5,+47.5] ns | **lottery artifact — dead** |
| **SEASON-12 ACT** | +94.1 CI[+64.7,+125.8] VALID | +59.5 CI[+43.6,+76.4] VALID | replicates BOTH ways |
| PIVOT-16 INV / ROUND-05 ACT | clean but N<30 | underpowered both ways | underpowered leads |

## 4. Honest bottom line
- **No clean, robust, both-year F-space entry discriminator exists among the catalog
  proposals.** The high-N dossiers (DOW-19 33k, SAR-23 33k, TUNNEL-20 32k, ZONE-21 3k)
  are decisive nulls — they have the power to find an edge and show none.
- **ATR-09 INVERT (my doc-035 lead) is RETRACTED.** It was strong one-directionally but
  fails the year-swap and its feature selection is unstable. The replication rule caught
  a one-year artifact — exactly its job.
- **SEASON-12 ACT** is the only both-way survivor, BUT it is the gap-fill dossier with
  heavy right-skew (EV +59..+94 vs mode +6..+24 = mean is 4-15x the mode). The "F-space
  discriminator" is proxying GAP SIZE; the EV is carried by rare large gap-fills = the
  outlier-day trap (graveyard). FLAG for a median-based / capped-magnitude retest before
  it is believed; do NOT bank it.
- Consistent with the program's honest floor: raw catalog signals are noise, and F-space
  conditioning at the **5s single-bar entry snapshot** does not cleanly rescue them.

## 5. What this is NOT (door left open, honestly)
Features here = the 5s single-bar entry snapshot (52 dims incl. z_se/hurst/lambda), NOT
the full multi-TF telescoping ladder (doc 017), and NOT the PhXit/PhPost descriptive
"conversion" analysis. A fuller ladder could change the verdict — but the burden of proof
is now high: 22 concepts, both-year replication, and only a skew-suspect survivor.

## 6. Still open for AG
B1 depth-leakage re-derivation (15 dossiers, doc-034 mods), B2 OHLC-01 anchor, B3 RSI-06
1948-pt trace, full multi-TF ladder, SEASON-12 median/capped retest. Note the alignment
fix here (`ag_phase5_doe.py`) should be back-ported to any script that joins events to
feature parquets.

Artifacts: `tools/ag_phase5_doe.py`, `reports/AG_cat_00_PHASE5_DOE.md`,
cache in `tools/_doe_cache/`. Committed + pushed this turn.
