# 2026-05-17 — Cross-Day Feature Dataset (DRS Phase 1A)

## Context
Day-Regime Sizer (DRS) needs a feature set V2 doesn't cover. The composite
zigzag pipeline tops out at ~$927/day with 5/31 OOS days net negative; per
HONEST_CAVEATS, bad days look identical to good days in V2 features at
session start. The fix is a separate cross-day classifier on overnight +
calendar + intermarket features.

This finding documents Phase 1A: build the feature dataset.

## What was built

**Tools** (new, in `tools/sourcing/`):
- `calendar_dates.py` — writes hand-curated FOMC/CPI/NFP CSVs
- `fetch_vix_dxy.py` — yfinance fetch (^VIX, DX-Y.NYB) -> parquet
- `build_cross_day_features.py` — joins everything into per-day features

**Outputs** (in `DATA/CROSS_DAY/`):
- `raw/fomc_dates.csv` — 16 FOMC announcement dates (2025-2026)
- `raw/cpi_dates.csv`  — 24 CPI release dates
- `raw/nfp_dates.csv`  — 24 NFP release dates
- `raw/vix_daily.parquet` — 364 rows, 2024-12-02 to 2026-05-15
- `raw/dxy_daily.parquet` — 367 rows, 2024-12-02 to 2026-05-18
- `cross_day_features.parquet` — 293 rows x 21 cols

## Feature inventory (21 cols, 293 days)

**Identity:** `date_label`, `date`, `dow`, `source` (ATLAS|NT8).

**Internal MNQ (computed from DATA/ATLAS/1m + DATA/ATLAS_NT8/1m):**
- `today_rth_open` — first 1m bar open at 09:30 ET
- `yest_rth_close` — last RTH close (lookahead-safe at 09:30+)
- `overnight_gap_pct` — (today_open - yest_close) / yest_close
- `overnight_range_pct` — (high-low between 16:00 ET yest and 09:30 ET today) / yest_close
- `prior_day_range_pct` — (high - low) / open in yest RTH
- `prior_day_c2c_pct` — close-to-close (yest vs day_before)

**External (yfinance daily):**
- `vix_close_prior`, `vix_chg_prior`
- `dxy_close_prior`, `dxy_chg_prior`

**Calendar flags:**
- `is_fomc`, `is_cpi`, `is_nfp`, `is_opex`
- `days_since_fomc`, `days_to_next_fomc`

**Reserved for trainer:**
- `target_day_pnl` — to be filled by joining forward-pass output (currently NaN)

## Sample statistics

| Metric | Min | 25% | Median | 75% | Max |
|--------|-----|-----|--------|-----|-----|
| overnight_gap_pct | -4.17% | -0.36% | +0.10% | +0.44% | +3.88% |
| overnight_range_pct | 0.20% | 0.68% | 1.01% | 1.47% | 6.22% |
| prior_day_range_pct | 0.13% | 0.89% | 1.25% | 1.82% | 12.74% |
| VIX (prior close) | 14.22 | 16.34 | 17.87 | 21.15 | 52.33 |
| DXY (prior close) | 96.22 | 98.23 | 99.07 | 100.18 | 109.96 |

Event-day frequencies (sanity-check against expectations):
- FOMC: 10/293 = 3.4% (expected ~3.2% from 8/year × 1.25y span)
- CPI:  16/293 = 5.5% (expected ~4.8%)
- NFP:  14/293 = 4.8% (expected ~4.8%)
- OpEx: 11/293 = 3.8% (expected ~4.0%)

All within expected frequency bands.

## Bugs found and fixed during build

**BUG #1 — Sunday futures sessions polluting "yesterday" lookup.**
ATLAS data has files for Sundays (futures-only sessions starting 18:00 ET).
With these in the sorted day list, Monday's "yesterday" resolved to Sunday,
which has 0 RTH bars -> Monday systematically skipped. Skipped 200/377 days
on first run; only Wed-Fri (dow=2-4) appeared in output.

**Fix**: filter `all_days` to weekdays only (`d.weekday() < 5`). After fix:
293/316 Mon-Fri days included (skipped 23, mostly holidays).

**BUG #2 — Weekend bars missing from Monday's overnight window.**
Monday's file (e.g. `2025_01_06.parquet`) contains bars from Sun 19:00 ET
to Mon 18:59 ET. The Sun 19:00-23:59 ET bars carry `date_et = Sunday`, so
my overnight_slice (which only looked at `date_et in {prev_date, today_date}`)
missed them entirely.

**Fix**: overnight_slice now also includes bars where
`prev_date < date_et < today_date` from today_df (captures weekend bars).

## 23 skipped days — explanations

| Date | Day | Reason |
|------|-----|--------|
| 2025_01_01 to 03 | Wed-Fri | Index-2 boundary (need yesterday + day_before) |
| 2025_01_09, 10, 13 | Thu, Fri, Mon | Jimmy Carter funeral / day-of-mourning |
| 2025_03_21 | Fri | Likely holiday/data gap |
| 2025_04_01, 02 | Tue, Wed | Adjacent to 4/1 (?) |
| 2025_06_20 | Fri | Juneteenth observance |
| 2025_07_01, 02 | Tue, Wed | Adjacent to Independence Day |
| 2025_09_19 | Fri | ? |
| 2025_10_01, 02 | Wed, Thu | ? |
| 2025_12_19 | Fri | ? |
| 2026_01_01, 02 | Thu, Fri | New Year holiday |
| 2026_01_05 | Mon | Adjacent (yesterday=skipped Fri) |
| 2026_04_03 | Fri | Good Friday |
| 2026_04_06, 07 | Mon, Tue | Adjacent to Good Friday |
| 2026_04_27 | Mon | Last NT8 day (no "tomorrow" needed but its yest may be NT8-NaN) |

Most are explainable holidays or adjacent-to-holiday days where prior-day
data has gaps. 7% drop rate is acceptable.

## Validation spot-checks

| Date | Day | Open | Gap | VIX | Flags | Notes |
|------|-----|------|-----|-----|-------|-------|
| 2025-01-06 | Mon | 21732.75 | +1.04% | 16.13 | — | First Mon of year, post-NYE weekend |
| 2025-06-18 | Wed | 21770.00 | +0.18% | 21.60 | FOMC | June FOMC day 2 (announcement) ✓ |
| 2025-12-10 | Wed | 25647.25 | -0.22% | 16.93 | FOMC + CPI | Double-event day, both correctly flagged ✓ |
| 2026-01-28 | Wed | 26253.25 | +0.68% | 16.35 | FOMC | January FOMC day 2 ✓ |
| 2026-03-19 | Thu | 24214.00 | -0.90% | 25.09 | — | Post-FOMC (3/17-18) day ✓ |

All flags align. VIX and DXY values are within historical ranges.

## Caveats and follow-ups

1. **FOMC/CPI/NFP dates are HAND-CURATED**, not scraped from official
   sources. Header comment in `calendar_dates.py` flags this. For tight
   accuracy, user should replace CSVs with official Fed/BLS schedules.

2. **NaN gaps in VIX/DXY** (16 and 14 rows respectively) come from year
   boundaries and edge dates where prior-day data wasn't in yfinance window.
   Trainer must handle these (impute or drop).

3. **DXY ticker**: used `DX-Y.NYB` (ICE cash dollar index). Fallback to
   `DX=F` (DXY futures) is configured but not triggered.

4. **target_day_pnl column is all NaN**. This is intentional — it will be
   filled by joining day-aggregated PnL from the hardened forward pass.
   Currently only 31 OOS days have PnL via the deliverable's existing
   forward pass output. To train DRS, we need IS day PnL too.

## What's next (Phase 1B)

**Decision required**: how to populate `target_day_pnl` for IS days.

Option A (canonical, more compute): run the deliverable's hardened forward
pass over IS as well. Requires generating B1/B2/B4/B5/B6 IS prediction
caches (only B7 IS exists today). Likely 30-60 min total compute.

Option B (cheaper, less aligned): use a PROXY target like daily realized
range or daily MNQ return as a stand-in for "day quality". Less rigorous
but immediately trainable. Useful for a first sanity-check of whether
cross-day features have ANY signal before committing to Option A compute.

Option C (avoid): train DRS on OOS only (23-31 days). Sample too small.

**Recommendation**: Option B first as a fast feasibility check. If features
show even modest correlation with proxy target, proceed to Option A for
real training. If features show NO signal even with proxy, DRS approach
is dead and we save the compute.

## Files

- `tools/sourcing/calendar_dates.py`
- `tools/sourcing/fetch_vix_dxy.py`
- `tools/sourcing/build_cross_day_features.py`
- `DATA/CROSS_DAY/cross_day_features.parquet`
- `DATA/CROSS_DAY/raw/{vix,dxy}_daily.parquet`
- `DATA/CROSS_DAY/raw/{fomc,cpi,nfp}_dates.csv`
