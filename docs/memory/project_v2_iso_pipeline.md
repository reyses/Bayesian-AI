---
name: V2-native iso pipeline (training_iso_v2/)
description: 2026-05-05 — V2-native isolated tier pipeline with the 9 legacy ExNMP tiers ported to V2 column names + an OU-aware reversion-decay exit rule. Replaces misnamed `training_iso V2/`.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

`training_iso_v2/` (no space, clean name) is a V2-native ISOLATED tier pipeline.
Same V2 substrate as `training_v2/` (V2Ticker, V2 features by name, no V1
conversion, no caches) but each tier runs in its OWN engine with its OWN
ledger — N parallel single-strategy engines on one bar stream.

Replaces `training_iso V2/` (with space), which was misnamed: 9 of 11 .py
files imported from `core.features` / `core.statistical_field_engine` /
`training.sfe_ticker` (V1). The new folder is fully V2-pure.

## Architecture additions over training_v2

- **`iso_orchestrator.py`**: runs N engine instances in parallel on one
  ticker. Each tick is dispatched to all engines simultaneously. Each
  engine has its own ledger; one engine's open position does NOT block
  another's entry. Total $/day = sum across N tier accounts.
- **`ticker.py`**: extended V2Ticker. Loads 5m / 15m / 1h OHLCV in addition
  to 5s/1m. Provides `state.ohlcv_5m / ohlcv_15m / ohlcv_1h` as the most
  recent CLOSED higher-TF bar (lookahead-free via `searchsorted(... ts -
  period, side='right') - 1`).
- **`state.py`**: BarState extended with `ohlcv_5m`, `ohlcv_15m`, `ohlcv_1h`.
- **`wicks.py`**: pure-OHLCV directional wick math
  (`upper_wick_ratio`, `lower_wick_ratio`, `directional_wick`). Pure market
  data — V2-pure because OHLCV isn't V1-specific.

## 9 V2-native tier ports

| tier | trigger (V2-native) |
|---|---|
| FADE_CALM | NMP seed + `\|L2_1m_price_velocity_w\| < 5.0` |
| FADE_MOMENTUM | NMP seed + `\|L2_1m_price_velocity_w\| >= 5.0` |
| RIDE_CALM | NMP seed + (regime, dir) in DEFAULT_FLIP_CELLS + `\|vel_1m\| < 5.0`, FLIP direction |
| RIDE_MOMENTUM | NMP seed + flip cell + `\|vel_1m\| >= 5.0`, FLIP |
| FADE_AGAINST | NMP seed + `sign(L2_1h_price_velocity_w)` opposes fade direction |
| RIDE_AGAINST | NMP seed + 1h opposing, FLIP direction |
| KILL_SHOT | NMP seed + `directional_wick(5m)>=0.50 + directional_wick(15m)>=0.45` |
| CASCADE | NMP seed + multi-TF wick + 1h velocity ALIGNED with fade |
| FREIGHT_TRAIN | `\|vel_1m\|>=10` + `swing_noise_1m<=100` + `hurst_1m<=0.5`, FADE the velocity |

NMP seed (V2): `\|L3_1m_z_se_15\|>=1.8 + L3_1m_reversion_prob_15>=0.55`.
Direction: `z>0 -> short; z<0 -> long`.

All thresholds are V1-style defaults; expect to recalibrate against V2
unit distributions on full IS (FADE_* tiers worked on smoke; KILL_SHOT /
CASCADE / FREIGHT_TRAIN had 0 entries — wick/velocity bars in V2 may need
different cutoffs).

## OUReversionDecay exit (new in iso pipeline)

Background: NMP entry depends on `reversion_prob_w`, which is the OU
first-passage probability from `z_se` — given current OU calibration
(theta, sigma), probability the price returns to band within the window.
A high entry rprob means the decay rate was strong enough to predict
reversion.

The OU calibration UPDATES each bar. If during a trade the current rprob
DROPS materially below its entry value, the OU decay rate weakened —
the mean-reversion thesis is dying.

Rule: `OUReversionDecay(tf='1m', decay_factor=0.6)` exits when
`current_rprob <= entry_rprob × 0.6` (configurable). Skipped for RIDE
tiers (the thesis isn't reversion).

Wired through `_entry_extras` in `run_iso.py` which captures
`entry_reversion_prob` at trade open into `position.extras`.

## Smoke test (2025_06_15)

13 total trades across 9 tiers, +$325.50 (single day). FADE_CALM, FADE_MOMENTUM,
FADE_AGAINST all 100% WR (small samples). RIDE_CALM and RIDE_AGAINST losing
(expected — flip cells have +/-18% asymmetry but per-trade is mixed). KILL_SHOT,
CASCADE, FREIGHT_TRAIN had 0 entries — thresholds likely need V2-unit recalibration.

## Run commands

```
# Full IS — all 9 tiers in parallel
python -m training_iso_v2.run_iso --is

# OOS
python -m training_iso_v2.run_iso --oos

# Subset of tiers
python -m training_iso_v2.run_iso --is --tiers KILL_SHOT,CASCADE,FREIGHT_TRAIN

# With production thresholds
python -m training_iso_v2.run_iso --is \
    --thresholds training_v2/output/thresholds_prod.json

# Single day smoke
python -m training_iso_v2.run_iso --smoke
```

## Open work

- Calibrate KILL_SHOT/CASCADE wick thresholds for V2 (legacy used 0.83/0.77 V1;
  V2 wicks computed from raw OHLCV may need different cutoffs)
- Calibrate FREIGHT_TRAIN extreme_velocity (legacy 100 V1, current 10 V2 default)
- Run iso pipeline on full IS + OOS, compare per-tier $/day vs legacy 2026-04-18
  iso baseline
- Investigate OUReversionDecay impact: run with vs without, see if OU exit
  reduces round-trip losses (the 64% of losers that had peak >$20 then gave back)
- Migrate the V1 `training_iso V2/` (with space) to deprecation; this is the
  successor
