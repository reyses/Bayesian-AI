"""training_iso_v2 — V2-native ISOLATED tier pipeline.

Same V2 substrate as training_v2/ (V2 ticker, V2 features by name, no V1
conversion, no caches, no shims) but with ISOLATION semantics: each tier
runs in its OWN engine instance with its OWN ledger. All tier engines see
the same bar stream in parallel; one tier's open position does NOT block
another tier's entry. Total $/day = sum across N parallel tier accounts.

This lets us measure each tier's standalone EV cleanly — no first-signal-
wins arbitration, no chains, no cross-tier interaction confound.

The legacy 9 ExNMP tiers are ported to V2-native form here (no V1 indices,
no 91D vector, all features read by V2 column name):

  CASCADE       : NMP + multi-TF wick rejection + 1h velocity aligned
  KILL_SHOT     : NMP + multi-TF wick rejection (z-extreme + wick filter)
  FREIGHT_TRAIN : extreme 1m velocity in compressed swing_noise regime
  FADE_AGAINST  : NMP fade with 1h velocity opposing the fade direction
  RIDE_AGAINST  : NMP-flipped + 1h velocity opposing (regime-driven flip)
  RIDE_MOMENTUM : NMP-flipped + high 1m velocity
  RIDE_CALM     : NMP-flipped + low 1m velocity
  FADE_MOMENTUM : NMP fade + high 1m velocity (freight-train flavor)
  FADE_CALM     : NMP fade + low 1m velocity (base NMP)

Wick math is pure OHLCV, not V1-specific — it carries over to V2 unchanged
(see wicks.py). Multi-TF OHLCV (5m, 15m, 1h) added to V2Ticker for tiers
that need wick rejection at higher timeframes.

Key files:
  - ticker.py            : V2Ticker extended with multi-TF OHLCV for wicks
  - state.py             : BarState extended with ohlcv_5m/15m/1h
  - wicks.py             : directional wick math (upper/lower wick ratios)
  - iso_orchestrator.py  : runs N engines in parallel on one ticker
  - run_iso.py           : CLI for iso forward pass
  - strategies/          : V2-native ports of the legacy 9 tiers
"""
