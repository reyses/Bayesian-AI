# Article-Accuracy Audit of the NT8 Catalog (Doc ID: AUDIT-ACC-01)
**Date:** 2026-07-11 · **Auditor:** Claude (Fable 5)
**Scope:** Do the 4 pillar synthesis files and the 18 test dossiers accurately reflect
what the 463 raw articles actually claim? (Frequency + magnitude framing per the
amended Master Validation Protocol §5 — nulls removed for the exploration stage.)
**Method:** Every checkable quantitative claim in `reports/01–04_*.md` was grepped back
to `raw_articles/`; every dossier script's event definition was read and compared to
its source article. This complements (does not replace) `AUDIT_REPORT.md` (2026-07-10,
folder discipline) and the per-dossier `REVIEW_AG.md` / `REVIEW_FABLE5.md` files.

---

## 1. Pillar claims NOT found in the articles (imported numbers)

| # | Pillar claim | What the article actually says | Source |
|---|---|---|---|
| 1 | 01 §2: VA rotation has an **"80% probability"** of crossing to the other side | Qualitative only: "rotations between the extremes are more likely than strong directional moves." No probability anywhere in the corpus. The 80% figure is imported Market-Profile folklore. | `how-to-trade-with-volume-profile-part-1.md:150-162` |
| 2 | 01 §5: footprint imbalance = **"3x (3:1 ratio)"** | "significantly larger… percentage threshold you can customize." The only 3:1 in the corpus is a risk:reward ratio in a different article. | `footprint-charts-guide.md:70-74`; `how-to-trade-liquidity-traps-in-futures.md:103` |
| 3 | 03 §7: **"Statistical ATR fading (90% rule)"** | The source stat is **72.44%** of days respect the daily ATR — measured on **YM, one 6-month window**, with an explicit warning that such stats drift and must be re-run. No 90% figure exists. | `the-statistical-analysis-of-trading-patterns.md:43-50` |
| 4 | 03 §6: squeeze forms around the **"21-period EMA"** | Bollinger basis = **20-period SMA**. | `bollinger-bands-explained-a-futures-traders-guide.md:29-31` |
| 5 | 02 §3: Elliott tunnels = **"34-EMA High and Low"** | The Wavy Tunnel article names no MA periods at all (grep "34" = 0 hits). | `find-market-patterns-using-elliott-wave-theory.md` |
| 6 | 02 §9: Renko brick "(e.g., 4 ticks)" | "a set number of ticks" — no example size. Minor (illustrative). | `renko-charts-in-futures-trading.md:27` |

**Verified accurate** (spot-checked, no discrepancy): ADX>25 threshold; golden cross =
50/200 SMA; VA = ~70% of session volume; P/b/D/B profile shapes incl. short-covering /
long-liquidation reads; trapped traders & absorption definitions; floor-pivot formulas
(PP=(H+L+C)/3, S1/R1); fib 23.6/38.2/50/61.8; H&S 3-stage volume divergence; virgin
supply/demand zones = "2–5 candles tight base + sharp departure, first test only"
(near-verbatim); 30-min ORB as a common window; seasonality 5/15/30-year lookbacks;
scalp stack = VWAP + 9/20 EMA + RSI + cumulative delta; overtrading checklist =
ATR/RSI/VWAP; APZ double-smoothed-EMA + percentage bands; mean-reversion 5-step with
20-SMA baseline; **VWAP z ±2 / "95% within 2 SD" IS article-backed**
(`z-score-futures-trading-strategy.md:59-67`).

---

## 2. Dossiers whose EVENT DEFINITION contradicts or invents the article claim

### 2.1 SEASON-12 — sweep "Survivor" built on an invented rule ⚠ HIGHEST IMPACT
Script: long at open Mon/Tue, short at open Thu/Fri, hold to EOD.
**No article claims weekday directional bias.** The corpus's actual weekday content:
- Gap-FILL rates by weekday (NQ, 08/2020–02/2021: Tue gaps fill ~70%, Mon gaps-down 64% do NOT fill) — `the-statistical-analysis-of-trading-patterns.md`.
- Volume/range by weekday (no direction).
- Dedicated seasonality articles are **calendar-year** 5/15/30-yr patterns, not weekday.
The Mon/Tue-bull / Thu/Fri-bear assignment has zero article provenance → it is a
researcher degree of freedom, and its dossier EVs are all CI-includes-0 anyway
(best cell 2025 Thu/Fri +50.5 [-3.2, +104.8]). The 2026-07-09 "Seasonality Survivor"
verdict should be downgraded to UNTESTED-AS-WRITTEN; the article-faithful test is the
**weekday gap-fill** event (frequency ~1/day, well-defined magnitude), which was never run.

### 2.2 ROUND-05 — tests the OPPOSITE direction of the article claim
Script: fade/bounce at 00/50 levels, mean-revert target SMA20, 10pt stop.
Article + own pillar (03 §8): round numbers hold **stop clusters**; a breach triggers
**cascade acceleration = continuation**, not a bounce (`how-to-trade-liquidity-traps-in-futures.md:39`).
The dossier's negative EVs (2025 bearish bounce −3.77 [−6.41, −1.00] sig) are therefore
*weak confirmation of the article* (fading the level loses) mislabeled as a failed test.
The article-faithful event = breach + acceleration (continuation), untested.

### 2.3 ADX-08 — the ">25 gate" is vacuous as implemented
`adx_proxy = (168-bar high-low range) / ATR(168) × 100`. That is not ADX (Wilder DMI);
its scale is hundreds-to-thousands, so `>25` is ~always true. Confirmed degenerate by
its own frequency: **exactly 1 bullish + 1 bearish event per day** (N=258 on 258 days
2024; 227/227 in 2025) — the test degenerates to "first SMA20 cross of the day, both
directions, every day." The article's ADX>25 trend gate was never actually tested.
Fix available in-repo: FIB-17's `compute_adx` is a proper DMI-based ADX — reuse it.

### 2.4 ATR-09 — three-way mismatch (label ≠ article ≠ code)
(a) "daily ATR" = **yesterday's single-day range**, not an average true range (no
multi-day averaging). (b) Trigger = price at **open ± 1.0×range** (open-anchored);
the article measures **today's high-low range vs ATR** (range-anchored) — a different
event. (c) The "90% rule" title matches neither the article's 72.44% nor the code's
1.0× trigger. EVs all CI-includes-0; conclusion unreliable either way.

### 2.5 CROSS-11 — sweep "Inversion" is a different object than the article's claim
Article: golden/death cross = **50-DAY / 200-DAY SMA on daily charts** (macro regime,
`identifying-trend-with-moving-averages.md:37`). Script: 50/200 **minute-equivalent**
intraday crosses (600/2400 five-second bars). Testing an intraday adaptation is
legitimate exploration, but the result cannot be attributed to the article's concept.
The "Death Cross inversion" rests on ONE cell: 2025 EV +35.4, CI [0.03, 75.59] —
barely excluding zero across ~70 setup-year cells in the sweep. Per protocol §6 this
is at most an INVERSION-CANDIDATE flag; as a headline "Inversion" it is overclaimed.

### 2.6 VWAP-03 — omits the article's mandated entry confirmation
`z-score-futures-trading-strategy.md` (entry rules): "**Wait for Z-score to begin
turning back toward zero before entering—confirmation that momentum is shifting**",
plus "check higher-timeframe direction before fading extreme readings."
Script enters **immediately at first ±2σ band touch**, no turn confirmation, no trend
filter. Also the article computes z with a **rolling lookback std (10/20/30 bars)**;
the script uses cumulative session VWAP variance. Knife-catch entries plausibly
explain the dossier's signature: high raw WR (0.60–0.77) with negative mean EV.
The article-faithful test (touch → z-turn → enter) is untested.

### 2.7 SCALP-18 — half the article's stack is missing, and N is unusable
Article stack: VWAP + **9/20 EMA alignment** + RSI + **cumulative delta confirmation**.
Script: VWAP side + price≤EMA20 + RSI40/60 — no 9-EMA, no delta. Frequency check:
N = 4–20 events/YEAR → no conclusion at any magnitude. Not the article's setup.

### 2.8 ORDERFLOW-14 — "Trapped Traders" definition diluted
Article/pillar: heavy buy volume **at the absolute high** of a bar that closes lower.
Script: bar `delta>0 and close<open` — far weaker condition. Also delta exists only for
one 6-month block (2024 N=0), so the "Dual-Year Validated" header is wrong for this
dossier. (Data itself is genuine bid/ask delta — good.)

### 2.9 Minor / acceptable adaptations (documented, no action needed)
- **SQZ-04**: implements TTM-style BB-inside-Keltner. Article is qualitative ("bands
  tighten noticeably"), so this is a fair operationalization — but it contradicts the
  pillar's own spec (bandwidth < 20th percentile). Synthesis and test disagree with
  each other, not with the article.
- **FIB-17**: "+ADX>25 confluence" is a synthesis construct; articles present fib
  levels and ADX separately. Label it an adaptation, not an article claim.
- **MACD-07**: 12/26 correctly scaled to 1-min equivalent; zero-line exit is bespoke.
- **VWMA-10**: same-period VWMA/SMA cross matches the article's general rule (its
  example used 15/20).
- **RSI-06**: note both alleged "RSI Bearish inversion" cells are CI-includes-0 in the
  dossier itself (2024 −12.4 [−39.2, +16.3]; 2025 −8.6 [−67.3, +42.8]). The 2026-07-09
  "Inversion" label is not supported by the dossier's own numbers.
- **VP-01 / VA-13 / PIVOT-16 / OHLC-01 / ORB-02**: math faithful (VA 70% around POC,
  pivot formulas, 08:30 CT open). VP-01/OHLC-01/ORB-02 carry existing FABLE-5 reviews.

---

## 3. What survives the audit, in frequency + magnitude terms
Under article-faithful(ish) rules, **nothing in the sweep is significant-positive in
both years**. The only claims that repeat across years are **negative-EV** ones:
- FIB-17 bearish pullback: sig negative BOTH years (−11.7 / −11.1).
- VA-13 bullish rotation: sig negative BOTH years (−2.9 / −7.5).
- ORDERFLOW-14 delta divergence: sig negative in both windows tested.
Per protocol §6 these are INVERSION-CANDIDATE flags for the discrimination stage —
the honest summary of the 2026-07-09 sweep is "several stable negative responses, no
stable positive ones," not "2 Survivors + 2 Inversions."

## 4. Protocol-consistency notes after the null removal (user edit, 2026-07-11)
- §5 now runs pure empirical counting (frequency + magnitude). Consistent with the
  exploration framing; the §7 "50% random-walk arithmetic reference" for symmetric
  barriers is correctly retained (it is arithmetic, not a null run).
- **Stale text to reconcile**: `AG_cat_00_INDEX.md` Execution Rules still mandate
  "Matched + phantom nulls at the per-signal layer" and its table carries a
  "P(resp) vs null" column; the AG Phase-3B directives likely reference nulls too.
- Pre-existing (2026-07-10 audit) still open: every INDEX row's "what it measures"
  says "VWAP Touch" (copy-paste), and the flat tools/ vs tests/ split.

## 5. Second-level joint logistic regression (`ag_joint_bayes_model.py`) — REVIEWED, headline INVALID

The 2026-07-10 audit took "+26.30 pp lift in the top posterior decile (82,102 events)"
at face value. Code review shows the lift is an artifact:

1. **The pooled label mixes two incompatible outcome definitions.** Directional rows
   (VWAP/APZ/Candle/MA) win only if a +2σ target is hit before a −2σ stop (~50% base
   rate by symmetry). Squeeze rows are `mode='volatility'`: they "win" if price moves
   2σ in **either** direction within 60×5s bars — with σ = std of single 5s diffs, a
   ~5-minute random walk excursion is ~√60·σ ≈ 7.7σ, so this label is ≈always true.
   The first-level index already knew this: `Squeeze_State … P(resp) 1.00 (vs 1.00)`.
2. **The model therefore learns the label TYPE, not confluence.** `sqz_state` coef
   2.5682 (odds 13.0) vs ~0.04 for everything else; the +26.3 pp "top tier"
   (N≈21.7k) is essentially the squeeze-triggered rows, which are defined to win.
   This is outcome-definition leakage through a feature, not a tradable edge.
3. **Everything is in-sample.** `fit(X,y)` then `predict_proba` on the same rows;
   2024 only. No 2025, violating both the repo OOS rule and protocol §7 (both years).
   Even in-sample, the extreme tier is miscalibrated (predicted 0.93, actual 0.73).
4. **82,102 rows ≠ 82,102 events.** One row is emitted PER TRIGGER at the same
   1m timestamp with identical context and overlapping 5-minute outcome paths →
   heavy pseudo-replication (see the CI effective-N rule); no day-block CI anywhere.
5. **Base rate 0.6072 is meaningless** — it averages a ~0.5-base population with a
   ~1.0-base population. The verdict criterion ("top tier > +10pp over base = tradable
   edge") compares against this pooled number.
6. **σ standard mismatch**: uses std of 5s diffs, not the protocol §7 trailing 1m
   regression-residual sigma — magnitudes not comparable with the dossiers.
7. Inputs inherit the first-level infidelities (§2): VWAP with no z-turn
   confirmation, minute-scale MA cross, candle shapes (already twice-dead).

**Fix for the re-run:** drop or separately model the volatility-mode rows (one label
definition per model); one row per timestamp (or cluster-robust by timestamp/day);
train 2024 → evaluate 2025 with day-block bootstrap CI on the top-tier lift; report
per-trigger base rates; use the §7 sigma standard.

## 6. Augmentation layer (`tests/*/augmentation/`) — NOT covered by the MVP, and its DOE model runs on RANDOM features

Every dossier carries an `augmentation/` subfolder (`augmentation_protocol.md`,
`followup_proposals.md`, `ag_logistic_model.py`, `fspace_doe_report.md`, plots).
Audit findings:

1. **The Master Validation Protocol does not document this stage at all.** The MVP's
   Test-Dossier architecture (§2) enumerates the binder contents as script + OQ traces
   + GDP report + assets, and its lifecycle stops at PQ (§5) → flags (§6). The
   augmentation layer — its 4-phase temporal-mapping protocol, the per-dossier DOE
   logistic regression, the F-space feature-selection reports, and how any of it
   feeds back into verdicts — exists only as per-dossier template files with no
   MVP section defining its gates, acceptance criteria, or GDP doc requirements.
   (Noted at user request, 2026-07-11.)
2. **`ag_logistic_model.py` fits its logistic regression to random noise.** The
   feature matrix is literally `X = np.random.randn(len(df), 5)` (identical copy in
   all 18 dossiers) — a placeholder that was never wired to real F-space features.
   All the surrounding rigor (magnitude weighting, stratified 5-fold CV "OOS Guard",
   decile tiers) is applied to noise; the tier tables at the top of every
   `fspace_doe_report.md` are meaningless, and their AUCs ≈ 0.48–0.52 confirm it.
   Any tier delta in those tables (e.g., ADX-08 bottom tier +8.45 pp) is pure
   binning luck on noise posteriors.
3. **`fspace_doe_report.md` files are corrupted by append-duplication.** The real
   F-space "ML Feature Extraction & Selection" blocks (4,644-dim fractal slice,
   PyTorch stepwise) were appended by a separate tool 0–4× per dossier (ADX-08 has
   the same block 4 times with tiny run-to-run diffs; CROSS-11/FIB-17/ORDERFLOW-14
   have none). Coverage is inconsistent and the writer clobbers/append behavior is
   mixed (`ag_logistic_model.py` writes mode 'w'; the appender does not dedupe).
4. **The real F-space selection results are below the house signal bar.** 15
   features selected from 4,644 candidates on ~967 samples, McFadden pseudo-R²
   0.0467 (< the 0.05 noise threshold), no held-out year, and stepwise forward
   selection over 4.6k dims on <1k samples is maximal selection bias. These
   feature lists should be treated as hypotheses only.
5. Housekeeping: OHLC-01, ORB-02 and ORDERFLOW-14 have no `events.parquet`, so the
   augmentation step cannot run for them at all.

**Fix:** add an "Augmentation (post-PQ exploration)" section to the MVP (inputs =
`events.parquet`; artifacts; explicitly non-verdict-bearing), wire real F-space
features into `ag_logistic_model.py` (or delete it in favor of the PyTorch pipeline),
make the appender idempotent, and regenerate the 18 reports.

## 7. Recommended re-runs (article-faithful, one change each)
1. **SEASON-12b**: weekday **gap-fill** event (the actual article stat) — gap at 08:30
   CT vs prior close, did it fill by EOD, by weekday. ~250 events/yr.
2. **ROUND-05b**: round-number **breach-continuation** (the actual article claim),
   magnitude = post-breach follow-through to resolution.
3. **ADX-08b**: rerun with FIB-17's real DMI ADX; keep everything else fixed.
4. **VWAP-03b**: add the z-turn confirmation bar + rolling-lookback z per the article.
5. **ATR-09b**: trigger on today's range filled ≥ X% of a true 14-day daily ATR
   (X sweepable at fixed values; report per-X, no cherry-pick).
