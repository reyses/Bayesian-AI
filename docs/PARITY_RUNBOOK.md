# Parity Validation Runbook

## What parity proves

The engine has THREE layers of computation. Parity means each layer in
live mode produces the same result as the same layer in backtest mode
on the same input bars.

| Layer | What it does | Parity test |
|-------|-------------|-------------|
| Features | SFE + windowing → 91D vector per bar | `FEATURES_LIVE_5s` vs `FEATURES_NT8_5s` |
| Decisions | Tier classification + entry/exit logic | engine `v2_trades_*.csv` vs baseline trades |
| Execution | NT8 actually fills what we send | engine `v2_trades_*.csv` vs `nt8_trades_*.csv` |

All three must pass before live trading is meaningful.

## Pre-session checklist

Before starting the engine for a parity test session:

1. **Bridge updated**
   - `docs/NT8_BayesianBridge.cs` copied to `Documents/NinjaTrader 8/bin/Custom/Indicators/`
   - Recompiled in NinjaScript Editor (no errors in Output window)
   - Indicator added to a 5s MNQ chart on Sim101 account

2. **Clean state**
   - NT8 position is flat (manually flatten any leftover trades)
   - `live/state/checkpoint.json` is current (auto-saved by previous run)

3. **Fresh data**
   - Dump latest history via BayesianHistoryDumper indicator
   - `python tools/convert_nt8_atlas.py --start 2026-03-20`
   - `python training/build_dataset.py --atlas DATA/ATLAS_NT8 --resolution 5s`

4. **Verify the build**
   - Last line: `Checkpoint saved: DATA/ATLAS_NT8/checkpoint.json`
   - 1h_z_se: var=X.XX **ALIVE** (not DEAD)

## Run the engine

```bash
python -m live.engine_v2
```

Watch for:
- **STEP 3 WARMUP**: bar counts per TF, checkpoint age, last bar timestamp
- **STEP 5 CATCH-UP**: how many bars from last dump to now
- **STEP 6 SYNC VERIFIED**: lag should be < 10s once a fresh bar arrives
- **STEP 7 TRADING**: dashboard health bar should show **● TRADING** in green

## During the session

Watch the dashboard health bar:
- **● TRADING** (green) = normal, bars flowing ~12/min
- **● CATCH_UP** (blue) = processing replay, no orders
- **● BROKER_DISCONNECTED** (red) = NT8 lost broker, no orders
- **● STALE** (red) = no bars for 60+ seconds
- Activity field shows: `direction tier +Nch  |  TRD/CHN side entry→exit ±$N`

If anything turns red:
1. Don't panic — the engine blocks orders automatically
2. Check NT8: connection status, account, popups
3. When NT8 recovers, dashboard turns green automatically

## Pre-shutdown

Click **SAVE NOW** button on dashboard before closing. Confirms all
buffers (ledger, features, checkpoint) are flushed to disk.

Then click the X to close the window. Engine shuts down within 1s,
flushes again, releases GPU/RAM.

## Post-session validation

The single command:
```bash
python tools/parity_validate.py
```

Auto-detects today's date. Outputs:
```
================================================================
PARITY VALIDATION — 2026_04_14
================================================================

OVERALL: PASS

LAYER 1: FEATURE PARITY (live vs build_dataset)  [PASS]
  Live bars:    4,800
  Base bars:    14,556
  Overlap:      4,263 bars
  Cells exact:  387,933 / 387,933 (100.00%)
  Features OK:  91 / 91

LAYER 2: EXECUTION PARITY (engine vs NT8)  [PASS]
  Engine decisions:     12
  Engine fills logged:  12
  NT8 fills logged:     12
  Matched by order_id:  12

VERDICT: ALL PARITIES PASSED — engine is faithful
```

Saves to `reports/findings/parity_validate_YYYY_MM_DD.txt`.

Exit codes:
- 0 = PASS (both layers match)
- 1 = WARN (mismatches but not catastrophic)
- 2 = FAIL (significant divergence)

## What divergences mean

### Layer 1 (Feature Parity) fails
- **0% match** → wiring bug in LiveFeatureEngine (window slicing, bar feed)
- **Some features perfect, others off** → SFE state contamination (duplicate bars, replay issues)
- **All features identical except acceleration at bar 0** → prev_velocities not loaded from checkpoint

### Layer 2 (Execution Parity) fails
- **Engine fills not in NT8** → orders sent but not filled (rejection, panic disconnect, manual cancel)
- **NT8 fills not in engine** → manual orders, FLATTEN button without engine trigger
- **Slippage diffs** → market orders filling at different prices than the bar close (normal, but log magnitude)

## After a successful parity pass

Once all three layers PASS for a clean session:
1. Engine is proven faithful to backtest
2. Now safe to add post-parity items from todo (guard bands, exit improvements, etc.)
3. Each new physics change requires re-running parity to confirm it doesn't break anything

## Workflow summary

```
Maintenance window (5pm ET):
  1. Dump history in NT8
  2. python tools/convert_nt8_atlas.py --start 2026-03-20
  3. python training/build_dataset.py --atlas DATA/ATLAS_NT8 --resolution 5s

Market open:
  4. python -m live.engine_v2
  5. Monitor dashboard health bar
  6. Click SAVE NOW + close window when done

Post-session:
  7. Dump history in NT8 (now includes today's bars)
  8. python tools/convert_nt8_atlas.py --start 2026-03-20
  9. python training/build_dataset.py --atlas DATA/ATLAS_NT8 --resolution 5s
  10. python tools/parity_validate.py
```

If step 10 returns PASS, the engine is faithful and you can iterate on
physics improvements. If WARN or FAIL, fix the divergence before
adding anything new.
