# L5 Hybrid Pipeline Spec — B9 During-Trade Sizer Deployment

**Status**: Draft — 2026-05-17
**Validated**: B9 K=5 model, OOS +$67/day, CI [+$32, +$106], 14.1% lift over $475 baseline
**Failed prerequisites**: B9 (binary cut, BOTH targets) — do NOT deploy

## What B9 K=5 does

At entry+25s (= 5 bars × 5s), predicts `remaining_pnl_usd = exit_pnl_usd - pnl_usd_so_far` for the open position. Applies continuous sizing rule:

```
predicted_remaining   action                 size_factor
  > +$50              pyramid (scale up)      1.5
  in [+$10, +$50]     hold full               1.0
  in [-$10, +$10]     uncertain, hold         1.0
  in [-$50, -$10]     reduce 50%              0.5
  < -$50              cut completely          0.0
```

Realized P&L per leg with action:
```
realized = pnl_usd_at_K + size_factor * (exit_pnl_usd - pnl_usd_at_K)
```

## Integration into the hybrid stack

```
NT8 ZigzagRunnerNative (C# strategy)            Python sidecar
       │                                              │
       ├─ R-trigger fire detected ─────────────┐      │
       │                                       └────→  B7 leg-sizer (V2 at R-trigger bar)
       │                                       ┌────  size = gbm_ev(pred_amp_R)
       ├─ market order placed at returned size ┘      │
       │                                              │
       │ [position open, T+0 to T+25s]                │
       │                                              │
       ├─ at T+25s (timer-driven) ─────────────┐      │
       │ snapshot V2 + trajectory features      └────→  B9 K=5 model
       │ send SIZE_QUERY                        ┌────  pred_remaining = HGB.predict(features)
       │                                        │     size_factor = size_from_pred(pred)
       │                                        │     return ACTION
       │                                        ┘
       │   if ACTION == CUT:        ExitPosition immediately at market
       │   if ACTION == REDUCE_50:  Close half position
       │   if ACTION == PYRAMID:    Add 50% to position
       │   else:                    Hold (no action)
       │                                              │
       │ [position continues]                         │
       │                                              │
       ├─ next R-trigger fire ────────────────────────→ (exit at R-trigger as before)
       │                                              │
```

## IPC Protocol Extension (BayesianBridge.cs)

Current messages (per BayesianBridge.cs v7.0.0):
- `PLACE_ORDER` (Python → NT8)
- `CLOSE_POSITION` (Python → NT8)
- `CANCEL_ORDER` (Python → NT8)
- `FILL` / `ORDER_STATUS` / `POSITION` (NT8 → Python)
- `BAR` / `PARTIAL_BAR` (NT8 → Python)

NEW messages required for L5:
- `SIZE_QUERY` (NT8 → Python): sent at T+25s after fill. Payload: `{position_id, entry_ts, entry_price, leg_dir, current_ts, current_price, position_qty}`
- `SIZE_ACTION` (Python → NT8): response. Payload: `{position_id, action: HOLD|REDUCE_50|CUT|PYRAMID, reason}`
- `REDUCE_POSITION` (Python → NT8): reduce position by N%. Already mappable to existing partial-close logic.

## C# Strategy Side Changes (ZigzagRunnerNative_v1.0.1-RC)

Inside `ZigzagRunnerNative`, add:

```csharp
private DateTime sizeQueryDueAt = DateTime.MinValue;
private bool sizeQuerySent = false;

// In OnExecutionUpdate, when entry fills:
if (execution.MarketPosition != MarketPosition.Flat)
{
    sizeQueryDueAt = execution.Time.AddSeconds(25);
    sizeQuerySent = false;
}

// In OnBarUpdate (primary series), check if size query is due:
if (Position.MarketPosition != MarketPosition.Flat
    && Time[0] >= sizeQueryDueAt
    && !sizeQuerySent)
{
    SendSizeQueryToPython();
    sizeQuerySent = true;
}
```

The 25-second delay matches B9 K=5 calibration (= 5 × 5s bars).

## Python Sidecar Changes (live/L5_during_trade.py — TO BUILD)

```python
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path('reports/findings/regret_oracle')

class L5_During_Trade:
    def __init__(self):
        # Load B9 K=5 production model
        with open(MODEL_DIR / 'b9_remaining_amplitude_K5.pkl', 'rb') as f:
            self.b10 = pickle.load(f)

    def size_from_pred(self, pred_remaining):
        if pred_remaining > 50:   return 1.5
        if pred_remaining > 10:   return 1.0
        if pred_remaining > -10:  return 1.0
        if pred_remaining > -50:  return 0.5
        return 0.0

    def handle_size_query(self, query):
        # query: dict with position info from NT8
        # Need to compute trajectory features at T+25s
        features = self.compute_trajectory_features(query)
        X = np.array([features[c] for c in self.b10['feat_cols']]).reshape(1, -1)
        pred = float(self.b10['model'].predict(X)[0])
        size_factor = self.size_from_pred(pred)
        action = self.size_factor_to_action(size_factor)
        return {
            'position_id': query['position_id'],
            'action': action,
            'pred_remaining_usd': pred,
            'size_factor': size_factor,
        }
```

## Validation gates BEFORE live deployment

1. **NT8 sim parity**: run ZigzagRunnerNative_v1.0.0-RC on NT8 Playback over 30 OOS days. Compare to `composite_forward_pass_hardened.csv` flat baseline ($475/day). Trade timestamps must match within ±5s.

2. **B9 inference parity**: run B9 K=5 inference on a known IS day via the new Python sidecar. Outputs must match `b9_remaining_amplitude_walk_forward.csv` predictions for that day within 1e-3.

3. **Hybrid end-to-end on sim**: NT8 Playback + Python sidecar + B9. Verify SIZE_QUERY messages fire at T+25s on every fill. Verify SIZE_ACTION responses applied correctly (reduce 50% of contracts, etc.).

4. **30-day sim run**: NT8 Sim101 + hybrid for 30 days. Compare to flat-sized R-trigger baseline. Expect +$67/day median improvement, CI overlap with [+$32, +$106].

5. **GO/NO-GO**: if 30-day sim shows positive CI on delta, promote to live at 0.25× position size. Scale to 1.0× over 4 weeks if metrics hold.

## What NOT to do

- ❌ Deploy C11 (binary cut) — failed both IS walk-forward and OOS implicit
- ❌ Skip the 30-day sim — anti-doom-cascade rule, no shortcut to live
- ❌ Retune B9 thresholds based on OOS results — that contaminates the test
- ❌ Trust the +$67 point estimate. The CI [+$32, +$106] is the operating range; plan for the low end.

## Cost analysis (per CLAUDE.md $/day-lift framing)

- Honest floor: +$32/day (lower CI bound)
- Point estimate: +$67/day
- Headline ceiling: +$106/day (upper CI bound)
- Annualized at floor: ~$8,000/year extra on the $120k/year baseline
- Annualized at point: ~$17,000/year
- vs cost of building: ~3-4 days of work already invested + ongoing model monitoring

## Next steps (post-validation)

If 30-day sim confirms the lift:

1. **B9 K=60** as second action gate (5 minutes in) — already trained, walk-forward sig 2/4. Add as secondary check.
2. **B11 retarget direction classifier** at K bars (different head: "is the leg about to flip?")
3. **B12 fakeout-during-trade** — retarget B2 to mid-trade
4. **L6 validation layer**: monitor B9 drift over rolling 30-day windows in production. Retrain quarterly.

## Files referenced

- Models: [reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl](reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl)
- Walk-forward: [reports/findings/regret_oracle/b9_remaining_amplitude_walk_forward.csv](reports/findings/regret_oracle/b9_remaining_amplitude_walk_forward.csv)
- OOS test: [reports/findings/regret_oracle/b9_OOS_singleshot_results.txt](reports/findings/regret_oracle/b9_OOS_singleshot_results.txt)
- Failed B9 (do not deploy): [reports/findings/regret_oracle/c11_bad_trade_summary.txt](reports/findings/regret_oracle/c11_bad_trade_summary.txt), [c11_v2_cut_saves_summary.txt](reports/findings/regret_oracle/c11_v2_cut_saves_summary.txt)
- NT8 strategy: [docs/nt8/ZigzagRunnerNative_v1.0.0-RC.cs](docs/nt8/ZigzagRunnerNative_v1.0.0-RC.cs)
- IPC ref: [docs/nt8/BayesianBridge.cs](docs/nt8/BayesianBridge.cs)
- Trajectory build: [tools/build_trade_trajectory_dataset.py](tools/build_trade_trajectory_dataset.py)
- B9 trainer: [tools/train_b9_remaining_amplitude.py](tools/train_b9_remaining_amplitude.py)
- OOS test: [tools/build_oos_trajectory_and_b9_test.py](tools/build_oos_trajectory_and_b9_test.py)

---

## v1.0.0-RC implementation status (2026-05-18)

### Components built

| File | Purpose | Status |
|---|---|---|
| `live/L5_sidecar.py` | Python TCP server loading B7+B9+B10, handles JSON messages | **DONE**, offline-replay verified |
| `docs/nt8/ZigzagRunnerHybrid_v1.0.0-RC.cs` | NT8 strategy with IPC hooks (DAY_OPEN, ENTRY_QUERY, SIZE_QUERY, POSITION_CLOSED) | **DONE** (skeleton — pivot detection logic delegated to ZigzagRunnerNative_v1.0.0-RC) |

### Sidecar protocol implementation

Messages and responses verified in `live/L5_sidecar.py`:
- `DAY_OPEN`: returns b10_mult (1.3 boost / 0.7 cap / 1.0 hold)
- `ENTRY_QUERY`: returns contracts = round(b7_size × b10_mult)
- `SIZE_QUERY`: returns action ∈ {HOLD, REDUCE_50, CUT, PYRAMID}
- `POSITION_CLOSED`: logged for monitoring

### Forward-pass-validated $/day expectation

| Scenario | OOS Δ/day | CI |
|---|---|---|
| Full stack B7+B9+B10 vs flat | +$672 | [+$426, +$939] |
| At 30% live-vs-sim gap | +$470 | +$224 floor |
| At 50% gap | +$336 | +$114 floor |

Even pessimistic 50% gap keeps CI lower bound positive.

### Pre-deployment validation steps

1. **Sidecar smoke test**:
   ```
   python -m live.L5_sidecar --offline-replay
   ```
   Should write `reports/findings/regret_oracle/l5_offline_replay.csv` with
   2,926 rows of B9 decisions. Run completed 2026-05-18.

2. **TCP server test** (with manual client):
   ```
   python -m live.L5_sidecar --port 5200
   ```
   Then from another shell, send DAY_OPEN with a real date, expect b10_mult
   response.

3. **NT8 compile**: copy `ZigzagRunnerHybrid_v1.0.0-RC.cs` to
   `Documents\NinjaTrader 8\bin\Custom\Strategies\` (gated by user approval),
   F5 compile.

4. **NT8 sim parity check**: apply ZigzagRunnerHybrid on MNQ 06-26 SIM
   account. Verify:
   - Output window shows "L5 sidecar connected" on State.Configure
   - "DAY_OPEN ... b10_mult=X" at session start
   - "ENTRY_QUERY pos_... contracts=N" at each R-trigger fire
   - "SIZE_QUERY pos_...: action=HOLD/REDUCE/CUT/PYRAMID" T+25s after fill

5. **30-day sim run**: NT8 Strategy Analyzer with Playback over 30 days,
   compare to backtested forward_pass_full_stack OOS results. Per-day P&L
   should match within ±20%.

6. **GO/NO-GO**: if 30-day sim CI on delta vs flat > 0, promote to live
   at 0.25× position size. Ramp to 1.0× over 4 weeks if metrics hold.

### Known gaps / TODOs

1. **V2 feature computation in NT8**: the hybrid strategy currently sends
   EMPTY `v2_features` in IPC messages. For full functionality, V2 features
   (~184 cols) must be computed in NT8 (C# port of core_v2/features.py) OR
   pre-computed by a separate Python data feed.
   - Workaround: sidecar can READ V2 features from `DATA/ATLAS_NT8/FEATURES_5s_v2/`
     parquet by timestamp lookup, but that requires historical data parity.
   - For LIVE: V2 computation must be streaming, not lookup.

2. **Pivot detection logic in hybrid file**: the v1.0.0-RC skeleton omits
   the 300-line pivot state machine from ZigzagRunnerNative. Production
   build should either inherit (C# inheritance) or copy that logic
   verbatim into the hybrid file.

3. **Friction modeling**: forward-pass-stack approximates mid-trade-action
   friction by 0 extra cost. Real friction for B9 CUT/REDUCE/PYRAMID adds
   ~$6/action. Slippage stress test showed +$38/day floor at $10/action;
   production should monitor actual friction vs assumption.

4. **No persistent state**: sidecar restarts lose `_day_cache` and
   `_positions`. Production should serialize to disk or use a small DB.

5. **Position ID handling**: simple timestamp-based ID may collide on
   simultaneous rapid pivots. Production should use NT8's internal
   order tag or a UUID.

