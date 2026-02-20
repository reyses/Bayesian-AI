# Jules Task: Add 5s & 1s Belief Network Workers + Gate 1 Worker Bypass

## Context

The `TimeframeBeliefNetwork` currently has 8 workers (1h → 15s). The ATLAS already
builds `5s/` and `1s/` parquet files in `DATA/ATLAS/` and `DATA/ATLAS_OOS/`. The
forward pass runs at 15s resolution (bar_i increments once per 15s bar).

This task adds two sub-15s workers and a Gate 1 bypass so the belief network can
generate trades even when no cluster template matches.

---

## Files to Modify

1. `training/timeframe_belief_network.py`
2. `training/orchestrator.py`

---

## Part 1 — Sub-15s Workers in `timeframe_belief_network.py`

### 1a. Class constants (lines ~263-269)

**Before:**
```python
TIMEFRAMES_SECONDS = [3600, 1800, 900, 300, 180, 60, 30, 15]
TF_WEIGHTS         = [4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
MIN_CONVICTION     = 0.48
MIN_ACTIVE_LEVELS  = 3
DEFAULT_DECISION_TF = 300
_TF_LABELS = {3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
              180:'3m',  60:'1m',   30:'30s',   15:'15s'}
```

**After:**
```python
TIMEFRAMES_SECONDS = [3600, 1800, 900, 300, 180, 60, 30, 15, 5, 1]
TF_WEIGHTS         = [4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.15]
MIN_CONVICTION     = 0.48
MIN_ACTIVE_LEVELS  = 3
DEFAULT_DECISION_TF = 300
_TF_LABELS = {3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
              180:'3m',  60:'1m',   30:'30s',   15:'15s',
              5:'5s',    1:'1s'}
```

### 1b. `TimeframeWorker.__init__` — add sub-15s flag (lines ~132-145)

The `bars_per_update = tf_secs // 15` calculation gives 0 for tf_secs < 15, which
breaks the tick index. Add a flag to use a different indexing strategy.

**Add to `__init__` after `self.tf_seconds = tf_seconds`:**
```python
# For sub-15s workers (5s, 1s): the forward pass runs at 15s resolution.
# These workers still update every 15s bar, but they index into a finer
# states list by taking the LAST sub-bar within each 15s period.
self.is_sub15s = tf_seconds < 15
if self.is_sub15s:
    self.sub_bars_per_15s = 15 // tf_seconds   # 3 for 5s, 15 for 1s
    self.bars_per_update  = 1                   # update every 15s bar
else:
    self.bars_per_update = max(1, tf_seconds // 15)
```

### 1c. `TimeframeWorker.tick()` — sub-15s index mapping (lines ~161-166)

**Before:**
```python
tf_bar_idx = bar_i // self.bars_per_update

if tf_bar_idx == self._last_tf_bar_idx:
    return False   # TF bar unchanged -- belief still valid
if not self._states or tf_bar_idx >= len(self._states):
    return False   # No state yet (warmup period)
```

**After:**
```python
if self.is_sub15s:
    # Pick the LAST sub-bar within the current 15s bar to get the
    # most up-to-date fine-grained state at each 15s tick.
    #   5s: bar_i=0 -> state_idx=2,  bar_i=1 -> state_idx=5, ...
    #   1s: bar_i=0 -> state_idx=14, bar_i=1 -> state_idx=29, ...
    tf_bar_idx = bar_i * self.sub_bars_per_15s + (self.sub_bars_per_15s - 1)
else:
    tf_bar_idx = bar_i // self.bars_per_update

if tf_bar_idx == self._last_tf_bar_idx:
    return False   # TF bar unchanged -- belief still valid
if not self._states or tf_bar_idx >= len(self._states):
    return False   # No state yet (warmup period)
```

### 1d. `TimeframeBeliefNetwork.__init__` — pass is_leaf to 5s worker too

The 5s worker should be treated as a leaf (top-K matching) since it's also a
very fine-grained bar. Update the worker creation:

**Before:**
```python
self.workers: Dict[int, TimeframeWorker] = {
    tf: TimeframeWorker(tf, is_leaf=(tf == 15))
    for tf in self.TIMEFRAMES_SECONDS
}
```

**After:**
```python
self.workers: Dict[int, TimeframeWorker] = {
    tf: TimeframeWorker(tf, is_leaf=(tf <= 15))
    for tf in self.TIMEFRAMES_SECONDS
}
```

### 1e. `TimeframeBeliefNetwork.prepare_day()` — accept df_5s, df_1s

**Before (signature):**
```python
def prepare_day(self, df_15s: pd.DataFrame, states_15s: list = None):
```

**After (signature):**
```python
def prepare_day(self, df_15s: pd.DataFrame, states_15s: list = None,
                df_5s: pd.DataFrame = None, df_1s: pd.DataFrame = None):
```

**Inside the loop, add handling for sub-15s workers before the existing `for tf_secs` loop:**
```python
# Sub-15s workers: use external fine-grained DataFrames (cannot resample from 15s)
_sub15s_data = {5: df_5s, 1: df_1s}

# Supply 15s states directly if available
if states_15s is not None:
    self.workers[15].prepare(states_15s)

for tf_secs in self.TIMEFRAMES_SECONDS:
    if tf_secs == 15:
        if states_15s is None:
            try:
                s = self.engine.batch_compute_states(df_15s, use_cuda=True)
                self.workers[15].prepare(s)
            except Exception as e:
                logger.warning(f"TBN: 15s state compute failed: {e}")
                self.workers[15].prepare([])
        continue

    if tf_secs < 15:
        # Sub-15s: use the supplied fine-grained DataFrame
        df_fine = _sub15s_data.get(tf_secs)
        if df_fine is None or df_fine.empty:
            self.workers[tf_secs].prepare([])
            continue
        try:
            states = self.engine.batch_compute_states(df_fine, use_cuda=True)
            self.workers[tf_secs].prepare(states)
        except Exception as e:
            logger.warning(f"TBN: TF={tf_secs}s state compute failed: {e}")
            self.workers[tf_secs].prepare([])
        continue

    # Existing logic for tf_secs >= 30 (resample from df_15s)
    try:
        ...  # existing code unchanged
```

---

## Part 2 — Load 5s/1s Data Per Day in `orchestrator.py`

### 2a. After loading `df_15s` and before calling `prepare_day()` (lines ~518-537)

**Before:**
```python
# Belief network: Task 1 for all 8 TF workers
try:
    _states_15s = self.engine.batch_compute_states(df_15s, use_cuda=True)
    belief_network.prepare_day(df_15s, states_15s=_states_15s)
except Exception as _bn_err:
    _states_15s = []
    belief_network.prepare_day(df_15s, states_15s=[])
```

**After:**
```python
# Load sub-15s ATLAS files for 5s and 1s workers (same directory structure)
_atlas_root = os.path.dirname(os.path.dirname(day_file))  # e.g. DATA/ATLAS_OOS
def _load_sub15s(tf_label: str) -> pd.DataFrame:
    path = os.path.join(_atlas_root, tf_label, f"{day_date}.parquet")
    if not os.path.exists(path):
        return None
    try:
        _df = pd.read_parquet(path)
        if 'timestamp' in _df.columns and not np.issubdtype(_df['timestamp'].dtype, np.number):
            _df['timestamp'] = _df['timestamp'].apply(lambda x: x.timestamp())
        return _df
    except Exception:
        return None

_df_5s = _load_sub15s('5s')
_df_1s  = _load_sub15s('1s')

# Belief network: Task 1 for all 10 TF workers (1h → 1s)
try:
    _states_15s = self.engine.batch_compute_states(df_15s, use_cuda=True)
    belief_network.prepare_day(df_15s, states_15s=_states_15s,
                               df_5s=_df_5s, df_1s=_df_1s)
except Exception as _bn_err:
    _states_15s = []
    belief_network.prepare_day(df_15s, states_15s=[],
                               df_5s=_df_5s, df_1s=_df_1s)
```

**Note:** `day_file` is already a path like `DATA/ATLAS_OOS/15s/20251201.parquet`.
`os.path.dirname(os.path.dirname(day_file))` gives `DATA/ATLAS_OOS`. Verify this
works; if not, adjust the path derivation accordingly.

---

## Part 3 — Gate 1 Worker Bypass in `orchestrator.py`

### 3a. After the Gate 1 rejection block (lines ~809-811)

This is the block that currently does `skip_dist += 1; continue`.

Add a `_worker_bypass_candidate` alongside the existing `best_candidate` / `best_dist`
selection. After evaluating all candidates in the inner loop, if `best_candidate` is
None but there's a high-conviction belief, execute a worker-only trade.

**Add a new constant near the other gate constants (e.g. near MIN_CONVICTION):**
```python
_WORKER_BYPASS_MIN_CONVICTION = 0.65  # minimum conviction to trade without a cluster match
```

**Inside the candidate evaluation loop, modify the Gate 1 block:**
```python
# Current code:
else:
    skip_dist += 1
    _candidate_gate[id(p)] = 'gate1'

# Replace with:
else:
    # Worker bypass: allow trade without cluster match if belief conviction is strong.
    # FN analysis showed 85-100% worker accuracy for no-match signals.
    _bypass_belief = belief_network.get_belief()
    if (_bypass_belief is not None
            and _bypass_belief.conviction >= _WORKER_BYPASS_MIN_CONVICTION
            and best_candidate is None):
        # Use this candidate as the bypass trade using worker-derived params.
        # Mark it so the FIRE block uses worker-only setup instead of template.
        _worker_bypass_candidate = p
        _worker_bypass_belief    = _bypass_belief
    else:
        skip_dist += 1
        _candidate_gate[id(p)] = 'gate1'
```

**Before the `if best_candidate:` block, initialise the bypass vars at the top of the bar's candidate loop:**
```python
_worker_bypass_candidate = None
_worker_bypass_belief    = None
```

### 3b. After `if best_candidate:` block — add worker bypass FIRE path

```python
elif _worker_bypass_candidate is not None and self.wave_rider.position is None:
    # ── Worker-bypass trade (no cluster match, conviction-driven) ──
    p    = _worker_bypass_candidate
    bel  = _worker_bypass_belief
    side = bel.direction   # 'long' or 'short'

    # TP from decision-TF worker's OLS prediction (ticks), fallback 20 ticks
    _bp_tp_ticks = max(8, int(round(bel.predicted_mfe))) if bel.predicted_mfe > 2.0 else 20

    # SL from sigma_fractal of the live bar state
    _live_s   = p.state
    _sigma_pt = getattr(_live_s, 'sigma_fractal', 0.0)
    _bp_sl_ticks = max(4, int(round(_sigma_pt / self.asset.tick_size * 1.5)))

    # Equity risk gate (same as regular trades)
    if _equity_enabled:
        _max_loss_usd = _bp_sl_ticks * self.asset.tick_size * self.asset.point_value
        if _max_loss_usd > running_equity * _MAX_RISK_FRACTION:
            skip_dist += 1   # account risk too large, skip
        else:
            # FIRE worker-bypass trade
            self.wave_rider.open_position(
                entry_price=price,
                side=side,
                state=None,
                stop_distance_ticks=_bp_sl_ticks,
                take_profit_ticks=_bp_tp_ticks,
            )
            current_position_open = True
            active_entry_price    = price
            active_entry_time     = ts_raw
            active_side           = side
            active_template_id    = -1   # sentinel: worker-only trade

            _dmi_at_entry = round(
                getattr(p.state, 'dmi_plus',  0.0)
              - getattr(p.state, 'dmi_minus', 0.0), 2)

            pending_oracle = {
                'template_id':      -1,
                'direction':        'LONG' if side == 'long' else 'SHORT',
                'entry_price':      price,
                'entry_time':       ts,
                'entry_depth':      getattr(p, 'depth', 6),
                'oracle_label':     0,
                'oracle_label_name': 'WORKER_BYPASS',
                'oracle_mfe':       0.0,
                'oracle_mae':       0.0,
                'long_bias':        0.0,
                'short_bias':       0.0,
                'dmi_diff':         _dmi_at_entry,
                'belief_active_levels': bel.active_levels,
                'belief_conviction':    round(bel.conviction, 4),
                'wave_maturity':        round(bel.wave_maturity, 4),
                'decision_wave_maturity': round(bel.decision_wave_maturity, 4),
                'entry_workers':    __import__('json').dumps(
                                        belief_network.get_worker_snapshot()),
            }
    else:
        # No equity tracking: fire unconditionally
        self.wave_rider.open_position(
            entry_price=price,
            side=side,
            state=None,
            stop_distance_ticks=_bp_sl_ticks,
            take_profit_ticks=_bp_tp_ticks,
        )
        current_position_open = True
        active_entry_price    = price
        active_entry_time     = ts_raw
        active_side           = side
        active_template_id    = -1

        pending_oracle = {
            'template_id':      -1,
            'direction':        'LONG' if side == 'long' else 'SHORT',
            'entry_price':      price,
            'entry_time':       ts,
            'entry_depth':      getattr(p, 'depth', 6),
            'oracle_label':     0,
            'oracle_label_name': 'WORKER_BYPASS',
            'oracle_mfe':       0.0,
            'oracle_mae':       0.0,
            'long_bias':        0.0,
            'short_bias':       0.0,
            'dmi_diff':         _dmi_at_entry,
            'belief_active_levels': bel.active_levels,
            'belief_conviction':    round(bel.conviction, 4),
            'wave_maturity':        round(bel.wave_maturity, 4),
            'decision_wave_maturity': round(bel.decision_wave_maturity, 4),
            'entry_workers':    __import__('json').dumps(
                                    belief_network.get_worker_snapshot()),
        }
```

---

## Verification Checklist

After implementation, run:
```
python training/orchestrator.py --oos --data DATA/ATLAS_OOS --account-size 500 --no-dashboard
```

Check `checkpoints/oos_report.txt` for:
1. **10 workers** firing — conviction values should now reflect 5s and 1s signals
2. **Worker agreement table** should show entries for `5s` and `1s` rows
3. **Gate 3 skips** > 0 (conviction gate active with 10 workers)
4. **`oracle_label_name = WORKER_BYPASS`** trades in `oos_trade_log.csv` — count should
   be roughly the FN no-match count (was 52 in previous run)
5. **No crash** when 5s or 1s ATLAS files are missing for a day (graceful fallback to
   empty worker states)

---

## Notes

- The `_load_sub15s` helper silently returns `None` when a file doesn't exist for a day,
  so days missing fine-grained ATLAS data degrade gracefully (sub-15s workers get empty
  states and never fire on those days).
- `active_template_id = -1` sentinel: any report logic that looks up `pattern_library[-1]`
  will KeyError — ensure `params` lookups in the FIRE block are guarded with
  `if active_template_id != -1`.
- The equity-gated and non-equity-gated bypass paths share the same `pending_oracle`
  structure — consolidate the duplicate block if it exceeds 20 lines.
- `wave_rider.open_position(state=None, ...)` — verify `WaveRider.open_position` accepts
  `state=None` without crash; it's used for logging only.
