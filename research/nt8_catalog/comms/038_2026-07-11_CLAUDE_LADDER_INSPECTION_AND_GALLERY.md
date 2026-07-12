# Ladder Inspection + Full Visual Gallery (Claude-executed)
**Doc:** 038 · **Date:** 2026-07-11 · **Author:** Claude (executor) · **Status:** FINAL

## 1. Multi-TF telescoping ladder (doc-017) — built and tested on ATR-09
Prior extractions used only `L*_5s`. `FEATURES_5s_v2` actually holds L1-L5 for EVERY
timeframe (5s..1D) on the 5s grid, so the true telescoping ladder IS buildable:
`tools/ag_phase5_ladder.py` — tiers 5s/15s/1m/5m/15m, each looking back N bars at its
own resolution (918 features). Forward split (train early days, test later).

**Result: the ladder did NOT improve differentiation.**
- It DID select the higher-TF λ/ldist structure (`L4_5m_lambda_hat_12`,
  `L4_15s_lambda_t_21`, `L4_1m_lambda_se_12`, `L5_*_ldist_*` across TFs) — the multi-TF
  NMP context is real and picked up.
- But the INVERT ride branch STILL carries the same 4 disasters (−182, −214, −223), and
  overall got slightly WORSE than the 5s approach-ladder (918 feats / 369 events = overfit).
- **Ceiling finding**: entry F-space, however rich, CANNOT separate the catastrophic
  rides from the +12 cluster. The tail is not predictable from entry state -> it must be
  handled by an EXIT rule (disaster stop), not entry selection. This closes the exit
  question: R-trigger alone can't cap a −223; the disaster stop is load-bearing.

## 2. Full visual gallery — 20 tests, for tonight's inspection
`reports/assets/phase5_gallery/` — one plot per dossier of the 2025 forward discriminator
branches (ACT/SKIP/INVERT), mode + median + mean marked (mode-first). Index with reads:
`reports/assets/phase5_gallery/INDEX.md`. Grouping:
- **Structure (tight cluster off 0)**: ATR-09 (INV, +tail-risk), PIVOT-16, ROUND-05,
  ORB-02 — the last three underpowered but clean.
- **Lottery (mean far from mode)**: SEASON-12, RSI-06, SQZ-04, CROSS-11.
- **Null (branches ≈ SKIP; high-N decisive)**: DOW-19, SAR-23, TUNNEL-20, ZONE-21,
  MACD-07, VWMA-10, VWAP-03, VP-01, OHLC-01, HNS-22, VA-13, FIB-17.
- Excluded: ADX-08/SCALP-18 (thin), ORDERFLOW-14 (year coverage), RENKO-24 (index space).

## 3. Honest standing conclusion (all proposals, visualized)
- The only dossier with a tight, forward-holding branch cluster AND enough N is ATR-09
  (INVERT ride), and its edge is capped by an entry-invisible catastrophic tail.
- PIVOT-16 / ROUND-05 show the same clean-cluster shape but lack power (need more events).
- The rest are lottery (outlier-carried) or null.
- No entry-F-space discriminator cleanly separates good from bad across the catalog; the
  richer ladder confirmed the ceiling rather than breaking it.

Artifacts: `tools/ag_phase5_ladder.py`, `tools/ag_phase5_gallery.py`,
`reports/assets/phase5_gallery/*` (20 PNGs + INDEX.md). Commit pending (classifier outage).
