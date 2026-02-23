# Continuous State Telemetry (CST) — Exit Architecture

> **Thesis:** The Oracle maps a non-linear probability wave. The Worker manages it
> with static, linear geometry (fixed price thresholds, fixed clocks). Continuous
> State Telemetry shifts the exit criteria from **Spatial Decay** (price hitting a
> line) to **Structural Decay** (the 16D matrix falling apart).

---

## The Problem: Information Asymmetry & The -1R Bleed

The 16D feature vector at entry (z-score, ADX, Hurst, coherence, momentum, parent
context, tf alignment, etc.) describes a **momentary alignment** — a physics
configuration that historically preceded profitable moves. Market entropy immediately
begins degrading this alignment.

When higher-timeframe structures decay — 5m ADX collapses, 1h regression mean
shifts, coherence breaks — the physical mass carrying the trade vaporizes. The trade
is **mathematically dead in the 16D space**. But the Worker lacks visibility into
these higher dimensions and blindly holds until price physically bleeds to the static
stop-loss.

**Result:** Full -1R losses on trades that possessed 0.0 Expected Value minutes
before the spatial stop was hit. The money is lost in the **bleed zone** — the
interval between structural death and spatial death.

---

## The Mechanism: Centroid Tether

At Gate 1, the system matches the pattern's 16D feature vector to the nearest
template centroid (Euclidean distance < 4.5). That centroid **is** the structural
anchor — the trade exists because the 16D state was close to that point in feature
space.

The Centroid Tether continuously asks: **are you still close?**

### Per-Template Basin Statistics (computed in Phase 2)

During clustering, every member pattern has a distance to its centroid. Store two
floats per template:

```
basin_mean  = mean(distances of all member patterns to centroid)
basin_std   = std(distances of all member patterns to centroid)
```

These define the template's **basin of attraction** — the region of 16D space where
the template has historically demonstrated an edge.

### Runtime Tether Check (every tick)

```python
# At entry (store once):
entry_centroid = centroids_scaled[matched_template_id]
entry_basin_mean = template.basin_mean
entry_basin_std  = template.basin_std

# Every tick:
current_vector = scaler.transform(build_16d_vector(current_state))
distance = np.linalg.norm(current_vector - entry_centroid)
tether_sigma = (distance - entry_basin_mean) / entry_basin_std

if tether_sigma > 2.0:
    structural_integrity = 0   # DYNAMIC ABORT
elif tether_sigma > 1.0:
    structural_integrity = 1   # TETHER WARNING (exit at bar close)
else:
    structural_integrity = 2   # INTACT (edge alive)
```

**Cost:** One `np.linalg.norm` on a 16D vector per tick. Effectively zero.

---

## The Dynamic Abort

If the real-time state drifts >2 sigma outside the entry centroid's basin, the
telemetry drops to 0. The Worker triggers an **immediate market-order abort,
regardless of current PnL.**

This is not a PnL decision. The trade could be +8 ticks or -3 ticks. **Irrelevant.**
The edge that generated the position no longer exists. Holding is gambling.

Specific triggers that cause >2-sigma drift:

| 16D Dimension | Structural Event | Effect |
|---------------|-----------------|--------|
| `parent_ctx` | Parent context flips Trend -> Chop | Discontinuous jump |
| `self_adx` | ADX collapses (25 -> 12) | Trend strength gone |
| `osc_coh`, `tf_alignment` | Timeframes decouple | Fractal alignment breaks |
| `self_hurst` | Persistent -> Anti-persistent (0.65 -> 0.42) | Memory structure inverts |
| `parent_z`, `parent_dmi_diff` | Macro regime shift | Higher-TF mass vaporizes |

Any one can push the vector >2-sigma. Euclidean distance aggregates all 16 dimensions
into a single check.

### Execution Rules

1. **No Fourier override.** The minimum-hold gate does NOT block a structural abort.
   If the matrix dies 2 bars after entry, you exit 2 bars after entry.
2. **No PnL gate.** Positive PnL does not justify holding a dead matrix.
   A profitable trade with zero structural integrity is a coin flip from here.
3. **Market order.** Not "tighten the trail." Not "exit at bar close."
   `structural_integrity == 0` -> exit at market price. Now.

---

## Exit Hierarchy (replaces current trail-based system)

| Priority | Layer | Trigger | Action |
|----------|-------|---------|--------|
| 1 | **Dynamic Abort** | distance > basin_mean + 2*basin_std | Immediate market exit |
| 2 | **Centroid Tether** | distance > basin_mean + 1*basin_std | Exit at next bar close |
| 3 | **Urgent Flip** | Belief network direction reversal | Immediate exit (existing) |
| 4 | **Spatial Stop** | Price hits -1R | Catastrophic insurance only |

The spatial -1R stop almost never fires. Structural exits catch decay **minutes
before** price bleeds to the stop. The bleed zone collapses from minutes to seconds.

---

## Left Tail Truncation

This architecture truncates the left tail of the PnL distribution. It cuts toxic
flow at fractional losses the exact second the mathematical edge dissolves, rather
than waiting for localized noise to hit the rigid spatial stop-loss.

```
Current loss distribution:
    -1.0R  ████████████████████   (full stop hits — bleed-to-death)
    -0.7R  ███████                (late trail catches)
    -0.3R  ██                     (urgent flips — rare)

CST loss distribution:
    -1.0R  ██                     (gaps only — catastrophic insurance)
    -0.7R  ███                    (rare fast structural decay)
    -0.3R  ████████████████       (abort at edge death)
    -0.1R  ████████████           (abort within ticks of entry)
```

Average loss shrinks. Win rate may not change. **But losers lose less.**

With ~859 losers (35% of 2,454 trades): if average loss moves from -1R to -0.4R,
system PnL transforms without changing a single entry rule.

---

## Oracle Training Addition

### Pre-compute `structural_integrity[]` alongside MFE/MAE

In the oracle lookahead loop (`_consult_oracle`), for each bar after entry:

```python
for i in range(1, ORACLE_LOOKAHEAD_BARS):
    future_state = states[entry_bar + i]
    future_vector = scaler.transform(build_16d_vector(future_state))
    distance = np.linalg.norm(future_vector - entry_centroid)
    sigma = (distance - template.basin_mean) / template.basin_std
    structural_integrity[i] = 0 if sigma > 2.0 else (1 if sigma > 1.0 else 2)
```

This produces a per-bar Boolean/int array alongside the existing price array.

### Trainable Outputs

1. **Optimal tether radius**: Scan thresholds, find where edge transitions from
   positive to zero EV. May differ from 2-sigma — let the data decide.
2. **Bleed cost quantification**: For each historical trade, compute
   `PnL(integrity_break) - PnL(stop_hit)`. Sum across all trades = headline KPI:
   **"CST would have saved $X."**
3. **Per-template tether half-life**: Some templates have tight basins (fast decay,
   exit early). Others have wide basins (state stays coherent, ride longer). The
   radius becomes per-template, not global.

---

## Implementation Files

| File | Change | Scope |
|------|--------|-------|
| `training/fractal_clustering.py` | Compute `basin_mean`, `basin_std` per template | Phase 2 |
| `training/fractal_discovery_agent.py` | Add `structural_integrity[]` to oracle loop | Phase 1 |
| `training/wave_rider.py` | Add `check_structural_integrity()` method | Runtime |
| `training/orchestrator.py` | Wire tether check into forward pass tick loop | Runtime |
| `core/quantum_field_engine.py` | Add `build_16d_vector(state)` utility if needed | Utility |

### Data Stored Per Template (2 new floats)

```python
@dataclass
class PatternTemplate:
    ...
    basin_mean: float = 0.0    # mean distance of members to centroid
    basin_std:  float = 1.0    # std of member distances to centroid
```

### New Columns in `oracle_trade_log.csv`

- `structural_integrity_at_exit`: Was the matrix alive (2), warning (1), or dead (0)?
- `bar_of_structural_death`: First bar where integrity dropped to 0 (None if never)
- `bleed_bars`: bars between structural death and actual exit
- `bleed_cost`: PnL difference between structural death and actual exit

---

## Role Separation: Oracle vs Workers

The Oracle and Workers serve fundamentally different roles:

- **Oracle** = teacher. Uses lookahead to label what *was* profitable, trains
  templates, computes basin statistics. Builds the curriculum. Never trades.
- **Workers** = executors. Do their own 16D evaluation in real-time with zero
  lookahead. Match centroids, monitor structural integrity via CST, abort when
  the matrix decays. Same math as the oracle — without seeing the future.

The templates give workers **pattern recognition** (the prior probability). CST
gives them **reality checking** (is the structure still holding right now?). The
template is the map. The worker is the navigator. CST is the windshield.

### Why Sub-Resolution TFs (15s/5s/1s) Don't Trade

Workers need compute budget to evaluate the 16D vector, measure centroid distance,
and decide. At 30s you have 30 seconds between bars — enough for full CST eval.
At 15s the worker becomes the fastest structural **sensor** (feeding the tether),
but too tight to be the decision timeframe. At 5s/1s, bars arrive faster than
meaningful 16D evaluation — they provide raw wick data only.

| Layer | Role | CST Function |
|-------|------|-------------|
| 30m - 30s | Trade + Decide | Full 16D eval, centroid match, CST tether |
| 15s | Fastest sensor | Highest-resolution structural telemetry feed |
| 5s / 1s | Wick detection | Raw price extremes for stop piercing (inner loop) |

---

## Augmented Playbook (Star Schema)

The base playbook (templates, centroids, basin stats) is frozen after training.
Workers build their own **augmented layer** on top — a flat star-schema lookup
updated during the session gap.

### Schema: Star (flat, one key, one hop)

```python
# augmented_playbook[template_id] -> dict
augmented[template_id] = {
    'recent_wr':        0.55,     # rolling win rate (last N trades)
    'recent_avg_pnl':   12.3,     # rolling avg PnL
    'n_recent_trades':  7,        # sample size
    'tether_radius':    2.0,      # sigma multiplier (may tighten/widen)
    'regime_tag':       'trend',  # current market regime
    'active':           True,     # not suspended
}
```

**Why star, not snowflake:** The codebase is still stabilizing. Snowflake adds
hierarchical traversal and propagation complexity. One bad propagation suspends
an entire branch. Star is debuggable — each template's record is independent.
Snowflake can come later when the foundation earns it.

### Session Gap Update Cycle

NQ US session ends ~4:00 PM ET. The dead zone before Asian open is the natural
heartbeat for augmented playbook updates:

```
US Session:     Trade with base + augmented playbook
    |
4:00 PM ET:     Session closes
    |
Analysis Gap:   - Score today's trades (PnL, CST bleed cost, tether accuracy)
                - Update augmented playbook per template
                - Flag templates going cold, boost templates running hot
                - Write augmented_playbook.json
    |
Next Session:   Load augmented_playbook.json, trade with updated layer
```

The base playbook is the curriculum. The augmented playbook is the worker's field
journal — updated every session gap, reflecting what actually happened vs. what
the oracle predicted.

---

## Testing: Target Day Mode

Instead of 10-month consecutive runs (4 hours, diluted signal), use deterministic
single-day tests for focused diagnostics:

```bash
python training/orchestrator.py --data DATA --target-day 20250314
```

Pick a day. Study every trade. Rerun it. Pick another. Build intuition about what
the system does in specific market conditions — trending, choppy, FOMC, etc.

Each day is an independent trial. Results are immediately interpretable:

```
Day 2025-03-14 (trending):  4 trades, +$380, 75% WR
Day 2025-06-02 (choppy):    2 trades, -$90,  0% WR
Day 2025-08-19 (FOMC):      0 trades (gates blocked all)
Day 2025-04-28 (mean-rev):  6 trades, +$520, 83% WR
```

This tells you more in 4 minutes than the full run tells you in 4 hours. You see
which regime the edge works in, not just the blended average.

---

## Verification Plan

1. **Phase 2 audit**: After clustering, print basin stats per template.
   Expect `basin_mean ~ 2-4`, `basin_std ~ 0.5-1.5` (normalized 16D space).
2. **Oracle dry run**: Compute `structural_integrity[]` for all historical trades.
   Generate histogram of `bar_of_structural_death` — expect peak at 3-8 bars for
   losers, confirming the bleed zone exists.
3. **Bleed cost report**: Sum `bleed_cost` across all losers. This is the dollar
   value CST captures. Target: >50% reduction in average loss magnitude.
4. **Target day validation**: Run `--target-day` on 5-10 diverse days (trend, chop,
   FOMC, mean-reversion). Verify CST abort fires on structurally dead trades.
   Compare per-day PnL with and without CST.
5. **Forward pass with CST**: Run full simulation. Compare PnL distribution tails.
   The left tail should visibly truncate.
