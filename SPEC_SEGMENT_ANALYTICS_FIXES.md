# SPEC: Regression Segments — Analytics Correctness Fixes

**Scope:** `research/Regression segments/` — analyze_tier_pnl.py, phase_d_mapping.py, stage1_speed_pass.py, stage2_parallel_chaos.py, mothership_server.py
**Priority order:** Fix 1 → 2 → 3 → 4 before re-running any tier-vs-PnL diagnostics. Fixes 5–7 are independent.
**Invariant:** No behavioral change to segment *generation* outputs except where noted (Fix 4 normalizes a convention; Fix 6 is perf-only and must be tier-identical). All diagnostic scripts must produce identical results on a NaN-free day before/after Fixes 1 & 4.

---

## Fix 1 (P0) — analyze_tier_pnl.py: use raw bar coordinates

**Problem:** Trade `entry_bar`/`exit_bar` from `oos_trade_data.json` metadata are raw OHLCV bar indices. Segment `start_idx`/`end_idx` are post-NaN-drop coordinates. Any day with dropped rows shifts all segments → silent trade-to-tier misassignment.

**Change:** Everywhere the script compares trade bars to segment bounds, use:

```python
s_idx = s.get('raw_start_idx', s['start_idx'])
e_idx = s.get('raw_end_idx', s['end_idx'])
```

Three sites: the `exit_bar == entry_bar` point-lookup, and the overlap computation (`overlap_start`/`overlap_end` both use raw bounds).

**Guard:** If a loaded segment file lacks `raw_start_idx` on its first segment, print a warning naming the file ("segments predate raw-index fix — coordinates may be shifted; re-run stage2 for this day") and continue with the fallback.

**Acceptance:** On a day with zero NaN-dropped rows, output is byte-identical to current. On a day with dropped rows, trade counts per tier change and a unit check confirms a trade at raw bar B maps to the segment whose raw range covers B.

---

## Fix 2 (P0) — analyze_tier_pnl.py: exclude tier-99 sentinel from all stats

**Problem:** `s.get('volatility_tier', 99)` flows 99 into the weighted tier average and into Pearson/Spearman. One UNPROCESSED_CHAOS overlap corrupts a trade's tier (e.g., 50% T2 + 50% T99 → "tier 50.5") and destroys the correlation diagnostic.

**Change:**
1. Define `VALID_TIER_MAX = 9` at module top.
2. In both the point-lookup and the overlap loop: skip segments where `volatility_tier` is missing, non-numeric, or `> VALID_TIER_MAX`. They contribute neither to `weighted_tier_sum` nor `overlap_sum`.
3. Track skipped coverage: accumulate `unclassified_overlap_len` per trade. Add a `pct_unclassified` column to the output CSV.
4. If `pct_unclassified > 0.5` for a trade, set its tier to `None` and exclude it from the summary table and correlations. Report the count of excluded trades.
5. Keep the tier journey string but render skipped spans as `T?(<pct>%)` so the chronology stays honest.

**Acceptance:** No value > 9 appears in the `tier` column. Correlations computed only over classified trades. A synthetic test trade overlapping 50/50 T2/UNPROCESSED yields tier 2.0 with `pct_unclassified = 0.5`.

---

## Fix 3 (P1) — phase_d_mapping.py: unreachable EXTREME bin

**Problem:** Tiers run 1–9. Binning `<=3 / <=6 / <=9 / else` puts tier 9 (PURE_CHAOS) in HIGH; EXTREME never fires.

**Change (apply at BOTH binning sites — trade mapping and per-bar prob mapping; extract one helper):**

```python
def vol_bucket(tier_raw):
    if not isinstance(tier_raw, (int, float)):
        return str(tier_raw) if tier_raw in ('UNCLASSIFIED',) else 'UNCLASSIFIED'
    t = int(tier_raw)
    if t <= 3:  return 'LOW'
    if t <= 6:  return 'MEDIUM'
    if t <= 8:  return 'HIGH'
    if t == 9:  return 'EXTREME'
    return 'UNCLASSIFIED'   # 99 sentinel and anything else
```

Note this also fixes the 99-sentinel leak in phase_d (currently `isinstance(99, int)` → bucket EXTREME would be wrong; route to UNCLASSIFIED).

**Acceptance:** Tier 9 trades land in EXTREME; tier 99 lands in UNCLASSIFIED; tiers 7–8 in HIGH.

---

## Fix 4 (P1) — Unify boundary convention: half-open `[start, end)`

**Problem:** analyze_tier_pnl uses `start <= bar < end`; phase_d uses `s <= bar <= e` (double-counts boundary bars, and with raw_end_idx defined as first-bar-after, leaks one bar into the next segment).

**Change:**
1. **Declare the convention** in project.md Control section: all segment ranges are half-open `[start_idx, end_idx)` and `[raw_start_idx, raw_end_idx)`; `raw_end_idx` is the raw index of the first bar AFTER the segment.
2. phase_d_mapping.py trade loop: `if s_idx <= entry_bar < e_idx:` (strict upper bound).
3. phase_d_mapping.py per-bar prob slice: `p_end = min(len(probs), e_idx - start_idx)` (drop the `+ 1`).
4. Verify stage2's raw-index backfill already matches (it does: `raw_indices[e]` = first bar after). Stage1's `raw_end_idx` likewise. No generation change needed — document only.

**Acceptance:** Sum of per-segment bar counts over a fully-classified day == total bar count exactly (no double-counted boundary bars). A trade entering exactly on a boundary maps to exactly one segment.

---

## Fix 5 (P2) — mothership_server.py: validate `day` path component

**Problem:** `day` from the URL flows into filesystem paths on `/download/<day>` and `/submit/<day>` → path traversal (e.g., `..%2f..`).

**Change:** At the top of both handlers:

```python
import re
DAY_RE = re.compile(r'^\d{4}_\d{2}_\d{2}$')
...
if not DAY_RE.match(day):
    return jsonify({"error": "invalid day format"}), 400
```

Also apply to drone-side: drone_worker passes day into paths/subprocess args — reject non-matching day in the drone before acting on a job.

**Acceptance:** `GET /download/..%2f..%2fboot.ini` → 400. Valid day unchanged.

---

## Fix 6 (P2, perf) — stage1_speed_pass.py: strided hunt-forward

**Problem:** The MESSY branch calls full `evaluate_block` (GroupLasso + ElasticNetCV) at every +1 bar offset, up to 500×. Dominant cost on choppy days.

**Change:** Replace the linear `for k in range(1, max_hunt)` scan with stride-then-refine:

```python
STRIDE = 5
found_coarse = None
for k in range(STRIDE, max_hunt, STRIDE):
    tier_k, *_ = evaluate_block(seed_start + k, SEED_BARS, error_band, ...)
    if tier_k in (1, 2):
        found_coarse = k
        break

if found_coarse is None:
    found_k = max(1, max_hunt)
else:
    # refine backward within the last stride window to find the earliest clean offset
    found_k = found_coarse
    for k in range(found_coarse - 1, max(found_coarse - STRIDE, 0), -1):
        tier_k, *_ = evaluate_block(seed_start + k, SEED_BARS, error_band, ...)
        if tier_k in (1, 2):
            found_k = k
        else:
            break
```

Worst-case error vs exact scan: zero — backward refine recovers the earliest clean offset within the stride. Cost: ~max_hunt/5 + STRIDE evals vs max_hunt.

**Caveat to record in project.md:** chaos-block boundaries are unchanged (refine is exact), but anyone increasing STRIDE without the backward refine WILL shift boundaries → tiers are path-dependent on E; keep refine mandatory.

**Acceptance:** On one reference day, `stage1_segments_<day>.json` is identical (segment boundaries, statuses, tiers) to pre-change output; wall-clock reduced. If identical-output check fails, abort and report.

---

## Fix 7 (P2, hygiene) — Deduplicate tier-classification logic

**Problem:** Three near-copies: `categorize_segment` (stage1, tiers 1–3), inline tier logic in stage1's `batched_ols_scan_pytorch`, and `categorize_chaos_segment` (stage2, tiers 1–9). Drift risk; stage1's inline copy already has a subtly different branch order.

**Change:**
1. New module `research/Regression segments/tiering.py` with one function:

```python
def classify_tier(residuals: np.ndarray, E: float, max_tier: int = 8) -> int:
    """Tier t passes if max_res <= (1.0 + 0.5*t)*E and
    max_consecutive(residuals > (0.5 + 0.5*t)*E) < 3. Returns max_tier+1 if none pass."""
    for t in range(1, max_tier + 1):
        hi = (1.0 + 0.5 * t) * E
        lo = (0.5 + 0.5 * t) * E
        if residuals.max() <= hi and max_consecutive(residuals > lo) < 3:
            return t
    return max_tier + 1
```

2. Stage1 call sites: `classify_tier(res, E, max_tier=2)` → returns 1, 2, or 3 (3 = "not pristine"), preserving current stage1 semantics. Stage2: `classify_tier(res, E, max_tier=8)` → 1–9.
3. Move `max_consecutive` into tiering.py; both stages import it.
4. Add `research/Regression segments/test_tiering.py`: parametrized cases pinning current stage2 behavior (boundary residual exactly at threshold, consecutive-outlier guard at 2 vs 3, all-zero residuals → tier 1, huge residuals → 9), plus a stage1-equivalence check against the old `categorize_segment` on random residual vectors.

**Acceptance:** test_tiering.py green; stage1 and stage2 outputs on the reference day unchanged vs pre-refactor.

---

## Execution order & validation gate

1. Fixes 1–4 (diagnostics + convention). Re-run `analyze_tier_pnl.py` and `phase_d_mapping.py`; archive old reports to `reports/findings/pre_fix/` — prior tier-vs-PnL conclusions are suspect and must not be cited.
2. Fix 7, then Fix 6 (refactor before perf change so the identical-output check runs against shared code).
3. Fix 5 anytime.
4. Gate: one reference day must produce byte-identical stage1/stage2 JSON across Fixes 6–7. Diagnostics (Fixes 1–4) are *expected* to change on days with NaN drops or tier-99 overlap — record deltas in a cycle file `cycle_01.md`.
