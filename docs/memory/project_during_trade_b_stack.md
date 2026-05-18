---
name: project-during-trade-b-stack
description: Re-conceiving B-models as during-trade state estimators rather than entry-time snapshots. L5 execution-layer paradigm enabled by the trade trajectory dataset built 2026-05-17.
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User insight 2026-05-17 (after building IS trade-trajectory dataset):
"a lot of B should be made during trades not at entry"

This re-conceives the entire B-model stack. Every existing B-model
(B1-B8) is currently a SNAPSHOT-AT-ENTRY predictor. The architectural
shift: each could be retargeted as a TIME-VARYING state estimator
that updates as the trade unfolds.

**Why this is structurally correct**: data processing inequality.
P(outcome | entry features + trajectory to K) >= P(outcome | entry features alone).
More information can only help (or be ignored).

**During-trade analogs of existing B-models**:

| Pre-entry B | During-trade analog | Operational action |
|---|---|---|
| B1 pivot-imminent | "is opposite pivot forming?" given K held | Early exit (B-cut/B9) |
| B2 fakeout | "am I being faked out NOW?" | Cut + flag direction |
| B4 pivot-region | "is opposite pivot-zone forming around me?" | Tighten exit |
| B5 leg-phase | "where in MY leg am I now?" | Time exit / pyramid timing |
| B6 directional pivot | "next pivot direction from HERE" | Cut or flip |
| B7 leg sizer | REMAINING amplitude from here (B10 candidate) | Pyramid, partial exit |
| B8 hour-risk | 60min risk update at K | Position cap adjust |

Each produces a different operational signal — cut, flip, pyramid, tighten,
hold. The full L5 execution layer is a multi-channel in-flight monitor,
not just one model.

**Enabler**: `reports/findings/regret_oracle/trade_trajectory_IS.parquet`
(74,976 rows = 17,748 legs x 5 K horizons {5, 10, 30, 60, 120} in 5s units).
Built 2026-05-17 from `is_hardened_legs.csv` (17,767 legs / 275 days,
$690/day flat IS baseline).

**Why:** During-trade B-models reduce variance of selector decisions and
unlock execution actions (cut/flip/pyramid) that snapshot-at-entry models
cannot access. The trajectory dataset enables this; before it existed,
training data was unavailable. The morning's trail-tightening failure
(2026-05-17 early) used composite signals at entry, NOT during-trade
features — different paradigm.

**How to apply**:
- L5 execution layer in the regret-oracle 6-layer architecture
- Adds streaming feature requirement: V2 features at entry+K, K+1, ...
  for K in {5, 10, 30, 60, 120} bars (5s units) while position is open
- Requires hybrid NT8 + Python sidecar with per-K state queries
  (not just R-trigger event triggers)
- Each during-trade B must walk-forward validate within IS, OOS held out

**Risk to manage**: more snapshots per leg = more degrees of freedom =
more overfitting potential. With 17k legs x 5 K horizons x 184 V2 features,
spurious patterns are easy to fit. Walk-forward + bootstrap CI mandatory
on every retargeted B. Per [[feedback_quantile_selection_overfit]] —
same risk class.

**Sequenced execution plan**:
1. Walk-forward CI on B-cut (B9, V2-only, target exit_pnl<-$50). If CI
   positive, paradigm is validated on ONE task.
2. Retarget B7 to remaining-amplitude (call it B10). Same trajectory
   dataset, different target. Second proof point.
3. If 1+2 succeed: systematically retarget B1/B5/B6 to during-trade.
   ~2 weeks of work for full L5 execution layer.
4. If step 1 fails (CI crosses zero): kill paradigm, don't waste 2 weeks.

**Current state (2026-05-17 evening)**: diagnostic on V2-only at K=30-60
shows AUC 0.89-0.94 with naive +$43-53/day on one val fold for B-cut.
NO CI yet. Walk-forward is the next step.

Related: [[project_zigzag_calibration]] (Python pipeline calibration),
[[feedback_quantile_selection_overfit]] (overfit risk), [[user_collaboration_protocol]]
(don't pivot mid-experiment — finish B9 walk-forward before retargeting).
