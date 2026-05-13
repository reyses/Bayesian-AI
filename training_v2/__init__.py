"""training_v2 — V2-native training pipeline.

Reads V2 layered features (185D = L0 + 8 TFs × 23) directly via
core_v2.features.load_features(). No V1 conversion, no compat cache,
no v1_compat shim.

Architecture (informed by 2026-04 to 2026-05 EDA stack):
  - ticker.py       : per-5s state from V2 parquets + raw OHLCV + regime label
  - engine.py       : bar loop, position state, dispatch to strategies/exits
  - strategies/     : entry rules (one file per rule, "first signal wins")
  - exits.py        : exit rules (z_se reversal, swing_noise spike, regime flip, ...)
  - regime_router.py: day-level eligibility (regime_2d → which strategies are armed)
  - ledger.py       : position state + closed-trade list
  - run.py          : CLI entry

CNN is used as filter+entry (Phase 6). Deterministic strategies fire first;
CNN gates take/skip and can also generate its own entries when no rule fires.
"""
