# Jules Spec: Partial Bar Feed from NT8

## Summary

NT8 sends partial (forming) bar OHLCV for all higher TFs whenever a lower TF
bar completes, and on throttled ticks for sub-minute TFs. Python receives these
and re-ticks TBN workers with fresh partial states — no aggregation needed on
the Python side.

**Research backing**: Analysis W (`reports/research/W_partial_bar_u/`) confirmed
k-NN direction prediction is robust to partial bars (+0.7pp lift). Deploying
this gives live trading fresher TBN beliefs across all timeframes.

## Part 1: NT8 Bridge — Partial Bar Messages

### File: `docs/NT8_BayesianBridge.cs`

**1a. On completed bar → send partials for all higher TFs**

In `OnBarUpdate()`, after the existing `SendRawJson(json)` at line 234,
add a loop that snapshots the **current forming bar** (`[0]` index) for every
data series with a larger period than the one that just closed:

```csharp
// After sending completed BAR, send PARTIAL_BAR for all higher TFs
for (int hi = idx + 1; hi < _barPeriodSecs.Length; hi++)
{
    if (CurrentBars[hi] < 1) continue;
    string partial = "{"
        + Q("type") + ":" + Q("PARTIAL_BAR") + ","
        + Q("instrument") + ":" + Q(Instrument.FullName) + ","
        + Q("tf") + ":" + Q(_barLabels[hi]) + ","
        + Q("bar_period_s") + ":" + _barPeriodSecs[hi] + ","
        + Q("timestamp") + ":" + D2S(ToUnixSeconds(Times[hi][0])) + ","
        + Q("open") + ":" + D2S(Opens[hi][0]) + ","
        + Q("high") + ":" + D2S(Highs[hi][0]) + ","
        + Q("low") + ":" + D2S(Lows[hi][0]) + ","
        + Q("close") + ":" + D2S(Closes[hi][0]) + ","
        + Q("volume") + ":" + D2S(Volumes[hi][0])
        + "}";
    SendRawJson(partial);
}
```

Key points:
- Uses `[0]` index = current forming bar (not `[1]` = last completed)
- Only sends for series with HIGHER period than the triggering bar
- Data series are ordered by period (ascending) in `_barPeriodSecs`
- Do NOT buffer PARTIAL_BAR in `_allBars` (no history replay for partials)

**1b. OnMarketData → throttled sub-minute partials**

Add `OnMarketData()` override for tick-level updates to sub-minute forming bars:

```csharp
private DateTime _lastPartialSend = DateTime.MinValue;
private const int PARTIAL_THROTTLE_MS = 250;

protected override void OnMarketData(MarketDataEventArgs e)
{
    if (e.MarketDataType != MarketDataType.Last) return;  // trades only
    if (_client == null) return;

    DateTime now = DateTime.UtcNow;
    if ((now - _lastPartialSend).TotalMilliseconds < PARTIAL_THROTTLE_MS)
        return;
    _lastPartialSend = now;

    // Send partial bars for all sub-minute series (period < 60s)
    for (int i = 0; i < _barPeriodSecs.Length; i++)
    {
        if (_barPeriodSecs[i] >= 60) break;  // sorted ascending, stop at 1m+
        if (CurrentBars[i] < 1) continue;
        string partial = "{"
            + Q("type") + ":" + Q("PARTIAL_BAR") + ","
            + Q("instrument") + ":" + Q(Instrument.FullName) + ","
            + Q("tf") + ":" + Q(_barLabels[i]) + ","
            + Q("bar_period_s") + ":" + _barPeriodSecs[i] + ","
            + Q("timestamp") + ":" + D2S(ToUnixSeconds(Times[i][0])) + ","
            + Q("open") + ":" + D2S(Opens[i][0]) + ","
            + Q("high") + ":" + D2S(Highs[i][0]) + ","
            + Q("low") + ":" + D2S(Lows[i][0]) + ","
            + Q("close") + ":" + D2S(Closes[i][0]) + ","
            + Q("volume") + ":" + D2S(Volumes[i][0])
            + "}";
        SendRawJson(partial);
    }
}
```

Key points:
- Throttled to 250ms (4 updates/sec) to avoid flooding
- Only fires on Last (trade) events, not bid/ask
- Only sub-minute series (< 60s period)
- Sorted ascending means we can `break` at first >= 60

**1c. Version bump**

Update header and `BRIDGE_VERSION` const to `6.5.0 — {date} {time}`.

## Part 2: Python Live Engine — Handle PARTIAL_BAR

### File: `live/live_engine.py`

**2a. Message handler**

In the message dispatch (where `BAR` messages are handled), add a case for
`PARTIAL_BAR`:

```python
elif msg_type == 'PARTIAL_BAR':
    self._on_partial_bar(msg)
```

**2b. New method `_on_partial_bar(self, msg)`**

```python
def _on_partial_bar(self, msg: dict):
    """Handle partial (forming) bar from NT8 — update TBN worker state."""
    bar_period = int(msg['bar_period_s'])
    if not self._belief_network:
        return

    bar_data = {
        'timestamp': float(msg['timestamp']),
        'open':      float(msg['open']),
        'high':      float(msg['high']),
        'low':       float(msg['low']),
        'close':     float(msg['close']),
        'volume':    float(msg['volume']),
    }

    self._belief_network.update_partial(bar_period, bar_data)
```

## Part 3: TBN — Partial Bar Worker Update

### File: `core/timeframe_belief_network.py`

**3a. New method on TimeframeBeliefNetwork**

```python
def update_partial(self, tf_seconds: int, bar_data: dict):
    """
    Update a worker with a partial (forming) bar.
    Called from live engine when NT8 sends PARTIAL_BAR.
    """
    if tf_seconds not in self.workers:
        return

    worker = self.workers[tf_seconds]

    # Build a minimal DataFrame for batch_compute_states
    import pandas as pd
    df = pd.DataFrame([bar_data])
    if 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'], unit='s')

    try:
        states = self.engine.batch_compute_states(df, use_cuda=True)
        if states:
            worker.update_partial_state(states[-1])
    except Exception as e:
        logger.debug(f"TBN partial update TF={tf_seconds}s failed: {e}")
```

**3b. New method on _TFWorker**

```python
def update_partial_state(self, state_raw):
    """
    Replace the current state with a partial bar state and re-analyze.
    Does NOT advance _last_tf_bar_idx — the completed bar will still
    trigger a normal tick when it arrives.
    """
    state = state_raw['state'] if isinstance(state_raw, dict) and 'state' in state_raw else state_raw

    # Re-run analysis with fresh partial state (reuse last bar index)
    tf_bar_idx = max(self._last_tf_bar_idx, 0)
    self._analyze(state, tf_bar_idx, ...)
```

**Important**: `update_partial_state` should NOT update `_last_tf_bar_idx`.
When the actual completed bar arrives, the normal `tick()` path still fires
and overwrites with clean completed data. Partials are interim updates only.

**3c. Pattern library access**

`_analyze()` needs `pattern_library, scaler, valid_tids, centroids_scaled`.
These are already available on the parent TBN. Options:
- Store refs on worker at `prepare()` time
- Pass TBN ref to worker
- Call `_analyze()` from TBN level instead of worker level

Cleanest: store refs on worker during `tick_all()` first call, then
`update_partial_state()` can reuse them. Add fields:

```python
# In _TFWorker.__init__:
self._cached_pl = None
self._cached_scaler = None
self._cached_vids = None
self._cached_centroids = None

# In tick() after successful _analyze():
self._cached_pl = pattern_library  # etc.
```

## Part 4: Forward Pass Support (Optional)

The forward pass uses ATLAS data with complete bars. To test partial bar
enrichment in backtesting:

- In `trainer.py` forward pass loop, when processing bar_i, for each TF
  with `bars_per_update > 1`, resample the child bars seen so far into a
  partial parent bar and call `update_partial()`.
- This simulates what NT8 would do in live.
- **Mark as optional** — can validate live-only first.

## Testing

1. **Bridge**: Add indicator to sim chart, verify PARTIAL_BAR messages in
   `docs/nt8_logs/` (they should appear between BAR messages)
2. **Python**: Run `python -m live.launcher --dry-run`, verify TBN workers
   show more frequent belief updates in logs
3. **Comparison**: Run live with/without partial bars, compare direction
   accuracy and trade quality

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `docs/NT8_BayesianBridge.cs` | Add PARTIAL_BAR in OnBarUpdate + OnMarketData | ~40 |
| `live/live_engine.py` | Handle PARTIAL_BAR message + `_on_partial_bar()` | ~20 |
| `core/timeframe_belief_network.py` | `update_partial()` + `_TFWorker.update_partial_state()` | ~40 |
| Total | | ~100 |
