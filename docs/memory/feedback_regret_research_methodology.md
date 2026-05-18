---
name: regret-research-methodology
description: The technical research method used across the regret-oracle arc — sediment 1D → pair → triplet → k-way, pivot target if signal weak, stratify before escalating k, configurable defaults, lookahead audit on every selector-usable axis. Distilled from the 2026-05-14..16 sessions.
metadata:
  type: feedback
---

This is the **technical workflow** that produced the regret-oracle direction
findings. Apply the same sequence to new sub-questions in this arc.

## The escalation ladder

For ANY new "does feature X predict direction/magnitude" question:

### GLOBAL (all data)

1. **1D regression per feature** (linear + quadratic + Spearman).
   Tool pattern: `tools/regret_feature_regression.py`.
   Catches monotone and U-shape relationships. Don't skip — many features
   have U-shape that linear-only sees as zero.

2. **Per-feature quantile table** (5 bins, per-cell stats).
   Tool pattern: `tools/regret_feature_table.py`.
   Per cell: n, mode_$, median_$, mean_$ + 95% bootstrap CI, noise_floor_$.
   Mode-vs-noise + CI-excludes-0 are the two gates for "real edge."

3. **Pair stratification + pair regression**.
   Tools: `regret_pair_clusters.py` (5×5 joint), `regret_pair_regression.py`
   (additive + interaction). Looking for: which feature × feature
   combinations stratify edge most.

4. **Triplet, only if pairs surface promising joints**.
   Don't blindly escalate. 3-way interaction term adds ~0 R² typically.

5. **k=4 / k=5 ONLY with reduced bins** (per [[kway-r2-saturation]] —
   3 bins at k=4, 2 at k=5). 4-way+ interactions add ~0.

### STRATIFIED (subset by primary feature first, then sediment within stratum)

When global signal saturates around R²~0.35, switch to stratified analysis.
**Mirror the global ladder within each stratum** — don't skip the 1D
within-stratum step:

6. **Stratified 1D regression** (per user 2026-05-16 — confirmed sound).
   For each stratum × each feature: 1D regression on signed_mfe.
   Surfaces **stratum-conditional features** — features that look weak
   globally but are strong inside specific strata (e.g., `z_1m` R²=0.05
   globally but R²=0.25 within RTH AM).
   Output: per (feature × stratum) row with R², slope, slope-per-σ,
   quadratic R², Spearman ρ. Add a `conditional_lift = R²_max_stratum −
   R²_global` column — sort by this to surface the "weak globally /
   strong in one stratum" set.

7. **Stratified pair regression** (the strongest move when global signal
   is noisy). Tool: `regret_stratified.py`. Subset by one feature
   (bar_range, tod_minutes, regime_2d) THEN run pair analysis within each
   stratum. Often beats unstratified k=4/5 with simpler model.

8. **Stratified triplet, only if stratified pairs surface joints**.

**Each level within stratum mirrors a level globally.** Don't skip the
1D-within-stratum step.

## Methodological levers

When direction/magnitude signal is weak on a target, try in this order:

1. **Pivot the target.** mfe_dollars → signed_mfe was decisive (see
   [[signed-mfe-pivot]]). For ANY direction work, target = signed_mfe.

2. **Stratify the data.** Heterogeneous data hides subgroup signals
   ("shaft from seeds"). Stratifying by bar_range or tod_minutes
   beats blind feature escalation.

3. **Add trajectory information.** Single-point features at entry
   collapse trade dynamics. Layer 3 (see [[bayesian-archetypes-pending]])
   uses the full N-D trajectory.

## Mandatory metrics (CLAUDE.md protocol)

Every per-cell or per-cluster $/trade stat MUST include:

- **mode** (histogram bin $2 for $/trade, $25 for $/day)
- **mean** with 95% bootstrap CI (4,000 resamples, percentile method)
- median (also useful)
- Direction stats: pct_long + Wilson 95% CI on pct_long
- Compare mode to noise floor where applicable

**Trade WR** = (∑profit/|∑loss|) − 1 (PF-based, NOT count-based).
**Day WR** = winning_days / total_active_days (count-based).
$/day or $/trade claims WITHOUT 95% CI + significance statement are
forbidden per CLAUDE.md.

## Lookahead audit

Before stratifying or matching on any axis, ASK:

- Is this feature knowable at entry time?
- Is it knowable using only past data?
- If it requires forward MFE or end-of-day stats → LOOKAHEAD.

Lookahead axes I've hit:
- `duration_bucket` (from `time_to_mfe_min` — forward)
- `regime_2d` (from end-of-day stats per the 2026-05-11 caveat)
- `mfe_dollars`, `signed_mfe`, `mfe_velocity` (targets, not features)
- `exit_*` state vector (mirror of entry, at exit bar)
- Centered-window oracle detection (uses future bars to find extrema)

Lookahead-clean axes:
- `tod_minutes` (wall-clock at entry)
- `bar_range`, `volume` at entry bar
- All entry-time state-vector z/dist/slope/fan features
- `full_window`, `available_fwd_min` (pure session geometry)
- All V2 features at entry bar

Direction-callable findings on LOOKAHEAD axes are descriptive only; on
CLEAN axes they're selector-usable. Distinguish in every report.

## Tool construction patterns

Every research tool:

- Takes `--input` (CSV) and `--out-dir`, `--name` for output naming
- Outputs to `reports/findings/regret_oracle/<filename>_<name>.csv`
  (per CLAUDE.md: "Tool outputs to file, not just stdout")
- Has CLI parameters for ALL thresholds (5%, K bins, min_n, etc.) —
  configurable from day 1 (per CLAUDE.md "no magic numbers")
- Stdout prints: header, progress (every N% for long runs), summary
  rankings (top 10-25 by relevant metric), caveats
- Saved to `tools/regret_*.py` (NOT throwaway inline code)
- Reuses helpers from sibling tools where possible (e.g., the
  bootstrap-CI and hist-mode functions)

For long runs (k>=4 cluster runs ~5-10 min):
- Run via `run_in_background=true`
- Print progress every ~10% with `flush=True`
- Or use file-based progress logging

## Reporting

After ANY meaningful research run, produce three artifacts:

1. **Findings doc** at `reports/findings/regret_oracle/<date>_<topic>.md`
2. **Daily journal entry** at `docs/daily/<date>.md`
3. **INDEX entry** in `docs/daily/INDEX.md` (prepend; one-line dense)

Per CLAUDE.md hard rules. Don't skip.

## Caveats that apply to every finding

- IS-only findings are descriptive, not actionable. 2026 OOS validation
  is mandatory before claiming edge (per MEMORY hard rule).
- Multi-comparison risk: ~400k+ cells tested across the k-way analyses.
  Some top cells will be spurious. Sign-stability OOS check required.
- No $/day claim without 95% CI + significance statement.

See also [[user-collaboration-protocol]], [[signed-mfe-pivot]],
[[kway-r2-saturation]], [[regret-six-layer-architecture]],
[[bayesian-archetypes-pending]].
