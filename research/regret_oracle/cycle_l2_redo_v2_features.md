# Cycle: L2 redo on full V2 features

Started 2026-05-16. Builds on the 2026-05-16 morning k-way analysis findings
(R² saturated ~0.35 on ~19-feature subset).

## Why

The L2 direction-discrimination work used only ~19 features from the
daisy-chain CSV's state vector. The codebase has ~190+ V2 features per bar
at `DATA/ATLAS/FEATURES_5s_v2/L{1,2,3}_{5s..1D}/`. Question: is R²=0.35 the
true ceiling or an artifact of the incomplete feature subset?

Answering this is a prerequisite for L3 priority. If full-feature L2 still
saturates around 0.35, the trajectory-based L3 build is justified. If R²
jumps significantly, L3 might be redundant and a simpler L2-based selector
could suffice.

## Protocol — additions to the standard L2 sediment ladder

Per the [methodology memory](../../memory/feedback_regret_research_methodology.md)
and user's 2026-05-16 confirmation:

1. **Join V2 features to daisy-chain trades at the entry bar** (single bar
   per trade for now; per-bar trajectory deferred for L3).
2. **Prune redundant features** via correlation clusters (V2 features are
   the same concept at multiple TFs — highly correlated). Target ~30-50
   non-redundant features.
3. **Global ladder** on pruned features:
   - 1D regression per feature (signed_mfe target)
   - Pair clustering + regression
   - Triplet (only if pairs surface joints)
4. **Stratified ladder** by primary stratifier (bar_range, tod_minutes):
   - **Stratified 1D regression** per feature per stratum (NEW step per
     user 2026-05-16 — surfaces stratum-conditional features)
   - Stratified pair clustering + regression
   - Stratified triplet only if needed

## Outputs

All under `reports/findings/regret_oracle/`:

- `daisy_with_v2_features_IS_full.csv` or `.parquet` — joined data
- `feature_prune_IS_full.csv` — correlation clusters + selected representatives
- `kway_2_v2_full_clusters_IS_full_daisy_signed.csv` etc. (re-run on full pool)
- `stratified_1d_<stratifier>_IS_full_daisy_signed.csv` (NEW output)
- Findings doc: `reports/findings/regret_oracle/2026-05-16_l2_redo_v2_features.md`

## Tools

- `tools/regret_join_v2_features.py` — NEW (P1 of L3 build, reused here)
- `tools/regret_feature_prune.py` — NEW (correlation clusters + 1D R² rank)
- `tools/regret_kway.py` — extend to support custom feature list from CSV columns
- `tools/regret_stratified.py` — extend to support `--mode 1d` for stratified 1D regression

## Decision criteria

| Outcome | Implication |
|---|---|
| R²_max stays ~0.35 | Ceiling is structural; L3 trajectory approach justified |
| R²_max jumps to 0.45 | Hidden capacity; L3 design might simplify; consider ensemble |
| R²_max jumps to 0.55+ | L2 alone might suffice; L3 build is optional polish |

## Caveats per CLAUDE.md / MEMORY

- IS-only findings — OOS validation on 2026 required before any claim
- Multi-comparison: pruning to ~40 features still gives ~780 pairs / ~10k
  triplets; some top cells will be spurious
- Strat 1D adds N × K = ~160 more regressions per stratifier — adds
  conditional-feature surface; same multi-comparison caveat
