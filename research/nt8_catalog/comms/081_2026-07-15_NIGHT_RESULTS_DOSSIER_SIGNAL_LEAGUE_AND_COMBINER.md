# NIGHT RESULTS — 12-stream dossier signal league + pooled combiner
**Doc:** 081 · **Date:** 2026-07-15 (overnight) · **Author:** Claude (executor, autonomous per Moises directive) · **Status:** FINAL

Directive being executed: *"continue in autonomous mode, and follow this same approach on
all the dossiers, see if we can extract more signal"* + the funnel Moises stated mid-run:
**extract causal signals aligned with the AI labels → mix all signals to boost → hand the
completed signal to the Mamba for trade management.** This doc is stages 1+2.

## 1. What ran
`tools/dossier_signal_pipeline.py` — 12 article-faithful causal signal streams over ALL
604 5s days (576 with AI labels), continuous windows (doc 073, no cold start), triggers
RTH-gated. Per stream: shared causal features {pivot_age_min, sig_with_leg, value, tod,
sig_with_leg×age}, target = direction agreement with the ACTIVE AI label, logistic
train-2024 → test-2025+26, day-block bootstrap CIs. Every trigger condition cites its
verified source (Batch A/B detector or audited legacy deep-dive) in the generator
docstring. Documented deviations only: one-shot latches removed on VWAP-03/ROUND-05
(one fire per |z|-excursion / per prime cycle); ORB-02, CROSS-11, VWMA-10 keep
first-only because the scan-break IS the rule (doc 070).
Skipped rather than fabricated (doc 080 §skip): MACD-07, RSI-06, FIB-17, SQZ-04, SAR-23,
HNS-22, VP-01, VA-13, ZONE-21, SCALP-18, ORDERFLOW-14, RENKO-24.

## 2. League table (OOS = 2025+26; baseline 0.50; full md: `reports/dossier_signal_league.md`)
```
ZIGZAG     N=  4852 OOS-AUC 0.556 base 0.96 || low 0.95 | mid 0.96 | high 0.97
ORB-02     N=   539 OOS-AUC 0.436 base 0.97 || flat ~0.97
SEASON-12  N=   521 OOS-AUC 0.618 base 0.48 || low 0.40 | mid 0.37 | high 0.66
VWAP-03    N= 29577 OOS-AUC 0.604 base 0.41 || low 0.31 | mid 0.40 | high 0.50
OHLC-01    N=   619 OOS-AUC 0.841 base 0.48 || low 0.07 | mid 0.59 | high 0.77
PIVOT-16   N=   324 OOS-AUC 0.939 base 0.05 || low 0.00 | mid 0.00 | high 0.15
ROUND-05   N= 44332 OOS-AUC 0.623 base 0.63 || low 0.52 | mid 0.63 | high 0.75
CROSS-11   N=   504 OOS-AUC 0.616 base 0.66 || low 0.55 | mid 0.65 | high 0.76
VWMA-10    N=   540 OOS-AUC 0.714 base 0.63 || low 0.36 | mid 0.76 | high 0.78
DOW-19     N= 36842 OOS-AUC 0.610 base 0.38 || low 0.28 | mid 0.38 | high 0.49
TUNNEL-20  N= 35228 OOS-AUC 0.604 base 0.59 || low 0.49 | mid 0.59 | high 0.68
ATR-09     N=   882 OOS-AUC 0.500 base 0.01 || flat 0.01
```
(ADX-08 from doc 079 completes the roster: OOS AUC 0.660, terciles 0.39→0.74.)

### Reading it — three signal families
1. **Genuine separators** (balanced base, big tercile spread — the stage-0 workhorses):
   **OHLC-01 is the night's star**: AUC 0.841 at ~1 fire/day, BOTH tails actionable
   (low tercile 0.07 [0.03,0.13] → inverted = 93% right; high 0.77 [0.69,0.85]).
   VWMA-10 (0.714, 0.36→0.78), SEASON-12 (0.618, 0.40→0.66), CROSS-11 (0.616),
   ROUND-05 (0.623 with high tercile 0.75 [0.73,0.76] on N=9,706), TUNNEL-20,
   VWAP-03 (low tercile 0.31 → inverted 69% on ~5k fires), DOW-19 (low 0.28 →
   inverted 72% on 6.5k fires). All spreads clear the ≥0.10 signal bar with
   non-overlapping day-block CIs.
2. **Momentum tautologies** (base ~0.96-0.97 or ~0.01-0.05): ZIGZAG and ORB-02 fire
   mid-leg and inherit the label's direction; PIVOT-16 ("bounce at S1/R1") and ATR-09
   ("fade the extreme") are the same thing MIRRORED — they fade mid-leg, so the label
   is running against them 95-99% of the time. Their direction content is nearly
   deterministic; their VALUE is timing (see §3). PIVOT-16/ATR-09 as INVERTERS are
   candidates for continuation signals, not bounce signals — the article's premise is
   backwards on MNQ 5s, consistent with the catalog's Phase-4 zero-edge verdict.
3. **Dead**: ORB-02 adds nothing beyond its 0.97 base (AUC 0.436 — the logistic can't
   separate inside the tautology); ATR-09 flat 0.01 (pure inverter, no gradation).

## 3. ZIGZAG timing split (`reports/zigzag_phase_in_label.md`)
The 0.96 agreement is real but timing-partitioned (phase = position inside the label):
```
phase (0.0,0.1]  N= 143 agree 0.64  med mins-left 52.7
phase (0.1,0.25] N= 677 agree 0.94  med mins-left 39.9
phase (0.25,0.5] N=1386 agree 0.97  med mins-left 21.2
phase (0.5,0.75] N=1152 agree 0.97  med mins-left  8.2
phase (0.75,1.0] N=1493 agree 0.99  med mins-left  1.4   <- MODE (0.97): rear-view
```
Median fire = phase 0.55 with 10.9 min left. **43% of confirms arrive in phase 0.1-0.5,
agreeing 0.94-0.97 with 21-40 minutes of label remaining** — that cohort is the causal
turn-clock. Stable per year: agree 0.96/0.96/0.97 in 2024/25/26.

## 4. Combiner preview — the MIX (`reports/combiner_preview.md`)
Pooled logistic over all 154,760 fires (68,104 train 2024 / 86,656 test 2025+26),
features = shared causal set + per-stream one-hot + **consensus** (same-direction
co-fires from other streams within ±3 min).

- **Pooled OOS AUC 0.687** (test base 0.533).
- **Decile calibration is monotone and honest OOS** — predicted vs observed:
  0.28→0.25, 0.35→0.36, 0.44→0.43, ..., 0.72→0.73, 0.83→0.80 (all day-block CIs ±0.02).
  The model's P(right) MEANS what it says on held-out years.
- **Actionable tails**: bottom decile observed 0.25 [0.23,0.26] → INVERT = 75% right;
  top decile 0.80 [0.79,0.82] → ACT = 80% right. That is ~17,300 OOS fires/2yrs in the
  two tails at 75-80% directional accuracy vs the 0.50 coin baseline — 5-6× the ≥0.10
  signal bar.
- Biggest standardized coefs: sig_with_leg +0.454, is_ZIGZAG +0.481, is_ATR09 −0.462,
  is_DOW19 −0.258, inter −0.172 (the doc-078 inversion carrier again), consensus +0.097.
- Raw consensus effect is modest and humped: 0 co-fires 0.49 → 3-5 co-fires 0.56 →
  6+ drops to 0.51 (many streams firing together = chop crowding, not conviction).

## 5. Honest caveats (do not oversell)
1. **P(right-about-label) ≠ P(profit).** Tautology streams inflate the tails: "ZIGZAG
   fired" alone buys P≈0.96, but MODE of its timing is the label's last minutes. The
   economic layer must condition on timing (leg age is IN the feature set, so the
   combiner partially handles this — but P&L conversion is unproven).
2. Dense streams (ROUND/DOW/TUNNEL/VWAP ≈ 146k of 155k rows) dominate the pooled
   deciles; the per-stream league is the per-article verdict, the combiner is the mixer.
3. Fires are not independent events (co-firing rows share moments); day-block CIs
   mitigate, pseudo-replication remains for trade-level claims.
4. 2026 is a partial year inside "test".
5. Signal rows parquets (12 files, 3.6MB, `reports/signal_rows_*.parquet`) are
   gitignored — regenerate with `python research/nt8_catalog/tools/dossier_signal_pipeline.py`.

## 6. Incidents (full disclosure)
- **Disk hit 0 bytes free** mid-run (C: full; OneDrive local content 244GB + WSL 34GB).
  First run crashed at stream 7 (exit 120, "not enough space"). Freed 3.6GB via
  `npm cache clean --force` + `pip cache purge` (official cache cleans only; a broader
  temp sweep was denied by permissions and NOT worked around). 4.8GB free at finish.
  **Flag for Moises: the machine needs a real disk plan** — tonight's margin is thin.
- League md was split across two runs by the crash; rebuilt from saved row parquets via
  `tools/league_merge_from_rows.py` (same data → same logistic; CIs re-drawn, point
  estimates replicated both runs).
- VWAP-03 multi-fire initially produced 2,127 fires/day (my one-shot-latch removal made
  it fire on every downtick while |z|>2). Fixed to prime-on-crossing (one fire per
  excursion) BEFORE the full run; documented in the generator docstring.

## 7. Next steps proposed (for Moises' morning review — nothing executed)
1. **Economic conversion**: P-tail fires → forward drift at label-relevant horizons
   (no stops, doc-046 style) to turn P(right) into $/fire; then the honest-floor framing.
2. **Phase-conditioned zigzag**: split by causal leg-age at fire (the 43% early cohort).
3. **Overfit-decay shelf-life** (doc 075 standard) on the combiner: train sliding
   2-month windows, measure weeks-to-<70% of peak AUC.
4. **Mamba handoff spec**: state-vector fusion (per-stream time-since-fire × direction
   × P) as RL input features — the "full power + trade management" stage.
5. ADX-08 rows into the pooled combiner (currently 12 streams + ADX separate).
