---
name: l5-sim-deploy-2026-05-19
description: "L5 stack built + wired into engine_v2 for Phase-1 SIM deploy. Phase-1 1c OOS lift +$42/day NOT SIGNIFICANT (CI [-$39,+$118]). Deploy to SIM only, not real money."
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

L5 stack (V2 features + B7/B9/B10) is wired into `live/engine_v2.py`
behind opt-in `--engine-mode l5` flag. Default `blended` mode unchanged.

**Why:** User went to sleep 2026-05-18 directing autonomous build for SIM
deploy 2026-05-19. Constraint: 1 contract/position max ($100 equity
buffer per additional contract). Phase-1 collapses B7/B9/B10 sizing
surfaces to filters: B7 = binary skip; B9 = binary CUT at K=5; B10 =
risk regime modulating B7+B9 thresholds (NOT skip-day per user's
"tighter limits not full skip" guidance).

**How to apply:**
  - Production money deployment GATED. OOS Phase-1 delta +$42/day NOT
    statistically significant vs FLAT 1c (CI crosses zero). Do NOT
    promote to real money until SIM data refines thresholds.
  - SIM deployment IS appropriate -- zero financial risk, gives
    real-time data on V2 streaming + zigzag pivot timing + B7/B9
    inference latency.
  - Launch: `python -m live.engine_v2 --engine-mode l5 --mock` first
    (mock replay validates), then `--engine-mode l5` for real SIM.
  - Pre-flight: MUST run `tools/sourcing/build_cross_day_features.py`
    daily before launch (B10 day-mode needs today's row); then
    `python tools/preflight_check.py` (7 checks).
  - If L5 misbehaves on SIM: drop the `--engine-mode l5` flag to fall
    back to existing BlendedEngine. Zero rollback cost.

**Key files** (paths -- check via `git status` to find rename/move):
  - `training/live_feature_engine_v2.py` -- streaming 185D V2 vector
    on-demand via `get_v2_vector(ts)`. Subclass of LiveFeatureEngine.
    300/300 parity vs batch on 2026_05_06 at 1e-6 tol.
  - `live/l5_decider.py` -- evaluate(state) -> DecisionBatch with
    zigzag state machine + B7/B9/B10 inference.
  - `tools/forward_pass_1contract.py` -- IS-calibrated Phase-1 OOS
    forward pass. Locked thresholds: B7 skip>=1.90, B9 cut<+5
    (normal); +0.2/+10 tighter on cautious mode.
  - `tools/preflight_check.py` -- pre-session SIM checklist.
  - `docs/Active/LIVE_L5_ARCHITECTURE.md` -- thin-wrapper architecture.

**OOS gating numbers** (51 days, IS-calibrated thresholds locked,
2026-05-19 sleep-run):
  FLAT 1c:          $+454/day CI [+$261, +$664]
  Phase-1 stack:    $+496/day CI [+$311, +$693]
  Delta vs FLAT:    $+42/day  CI [-$39, +$118]  NOT SIG

**UPDATE 2026-05-19 LATE-MORNING -- Mock validated against forward pass:**
  - 5-day mock May 11-15: $+3,582 vs OOS Phase-1 forecast $+3,716
    (gap $-27/day = 3.6%). Engine is decision-identical.
  - 20/20 EXACT entry+exit price match on May 11 spot-check.
  - Streaming pivot detector now ports the FULL detect_swings logic
    (min_bars=36, ATR median of 42 TRs, 5-day pre-warmup); per-day
    pivot capture 98% week-aggregate.
  - --pivot-source=replay mode added for regression testing.

**Related:** [[live-l5-architecture-thin-wrapper]],
[[user-collaboration-protocol]], [[oos-only-for-nn-validation]]
