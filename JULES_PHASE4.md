# Jules Task: Star Schema Discovery + Phase 4 Forward Pass + Phase 5 Strategy Selection

## Overview

The pipeline currently ends at Phase 3 (playbook generation). This task adds:
- **Task A**: Star schema in discovery — full parent chain per pattern
- **Task B**: Phase 4 — forward pass executing the playbook with fractal zoom-in
- **Task C**: Phase 5 — strategy ranking and risk scoring

These are sequential: A modifies discovery, B uses the star schema for execution, C analyzes results.

---

## Task A: Star Schema in Discovery

### Problem
`PatternEvent` only stores the immediate parent (`parent_type`, `parent_tf`). When executing in Phase 4, we need the **full ancestry chain** to know: "this 15s entry was inside a 1m structural drive, which was inside a 5m Roche snap, which was inside a 15m Roche snap, which was inside a 1h structural drive, which was inside a 4h Roche snap."

Without the chain, we can't distinguish between a 15s signal backed by 4H structure vs one backed by nothing.

### File: `training/fractal_discovery_agent.py`

#### Step 1: Add `parent_chain` to PatternEvent

```python
@dataclass
class PatternEvent:
    pattern_type: str       # 'ROCHE_SNAP' or 'STRUCTURAL_DRIVE'
    timestamp: float
    price: float
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    file_source: str
    idx: int
    state: ThreeBodyQuantumState
    timeframe: str = '15s'
    depth: int = 0
    parent_type: str = ''
    parent_tf: str = ''
    window_data: Optional[pd.DataFrame] = None
    parent_chain: Optional[list] = None  # NEW: list of dicts, one per ancestor
```

Each dict in `parent_chain`:
```python
{
    'tf': '4h',                    # timeframe
    'type': 'ROCHE_SNAP',         # pattern type
    'z': 2.5,                     # z_score at that level
    'mom': 1.4,                   # momentum at that level
    'coh': 0.85,                  # coherence at that level
    'timestamp': 1706832000.0     # when the parent triggered
}
```

Ordered from immediate parent (index 0) to root ancestor (last index).

#### Step 2: Propagate chain through `scan_atlas_topdown()`

In `scan_atlas_topdown()`, each level creates `parent_windows` for the next level. Currently these are `(start, end, parent_type, parent_tf)` tuples.

**Change the window tuple** to include the chain:
```python
# Old:
parent_windows.append((start, end, pattern.pattern_type, pattern.timeframe))

# New:
chain_entry = {
    'tf': pattern.timeframe,
    'type': pattern.pattern_type,
    'z': pattern.z_score,
    'mom': pattern.momentum,
    'coh': pattern.coherence,
    'timestamp': pattern.timestamp
}
# Prepend this pattern's info to its own chain
full_chain = [chain_entry] + (pattern.parent_chain or [])
parent_windows.append((start, end, pattern.pattern_type, pattern.timeframe, full_chain))
```

#### Step 3: Pass chain into `_batch_scan_windowed()`

Update `_batch_scan_windowed()` to accept and propagate the chain:

- The `parent_windows` parameter now contains 5-tuples instead of 4-tuples
- When creating `PatternEvent` objects, set `parent_chain` from the window's chain

```python
# In _batch_scan_windowed, when creating PatternEvent:
PatternEvent(
    ...,
    parent_type=p_type,
    parent_tf=p_tf,
    parent_chain=p_chain,  # from the parent window tuple
    window_data=window_slice
)
```

#### Step 4: Level 0 (macro) patterns have empty chain

For `_batch_scan_full()` (Level 0, e.g., 1D), patterns have no parents:
```python
PatternEvent(..., parent_chain=[])
```

#### Step 5: Update `_merge_windows()`

The merge function currently handles 4-tuples. Update to handle 5-tuples. When merging overlapping windows, keep the chain from the dominant (longer) window.

### Verification for Task A

```bash
python -c "
from training.fractal_discovery_agent import FractalDiscoveryAgent
agent = FractalDiscoveryAgent()
manifest = agent.scan_atlas_topdown('DATA/ATLAS')
# Check a deep pattern has a chain
deep = [p for p in manifest if p.depth >= 5]
if deep:
    p = deep[0]
    print(f'Pattern: depth={p.depth}, tf={p.timeframe}')
    print(f'Chain length: {len(p.parent_chain)}')
    for ancestor in p.parent_chain:
        print(f'  {ancestor[\"tf\"]} {ancestor[\"type\"]} z={ancestor[\"z\"]:.2f}')
    print('PASS')
"
```

Expected: A depth-5 pattern should have a chain of length 5 (or up to 5), each entry showing tf/type/z from ancestor levels.

---

## Task A.2: Update Clustering Feature Vector

### File: `training/fractal_clustering.py`

The feature vector currently uses 7 dimensions. With the parent chain, we add **condensed ancestry features**.

#### Approach: Fixed-width ancestry encoding

Since chains vary in length (depth 0 has 0 ancestors, depth 8 has 8), encode a fixed-width summary:

```python
# In create_templates(), when building features per pattern:

# Existing 7D features
base = [abs(z), abs(v), abs(m), c, tf_scale, depth, parent_ctx]

# NEW: ancestry summary (4 additional features)
chain = getattr(p, 'parent_chain', None) or []

if chain:
    # Immediate parent physics
    parent_z = abs(chain[0].get('z', 0.0))
    parent_mom = abs(chain[0].get('mom', 0.0))

    # Root ancestor physics (macro context)
    root = chain[-1]
    root_z = abs(root.get('z', 0.0))
    root_is_roche = 1.0 if root.get('type') == 'ROCHE_SNAP' else 0.0
else:
    parent_z = 0.0
    parent_mom = 0.0
    root_z = 0.0
    root_is_roche = 0.0

features.append(base + [parent_z, parent_mom, root_z, root_is_roche])
```

This makes the feature vector **11D** instead of 7D. The clustering will naturally separate patterns with different ancestry contexts.

**IMPORTANT**: Also update `refine_clusters()` to use the same 11D vector when re-computing centroids for fissioned sub-templates.

### Verification

Run the full pipeline after Task A + A.2:
```bash
python training/orchestrator.py --fresh --no-dashboard --iterations 50
```

Expected: Discovery produces 2634 patterns (same count), but templates may differ slightly because the 11D clustering separates by ancestry.

---

## Task B: Phase 4 — Forward Pass with Fractal Zoom

### Concept

Phase 4 replays the full year day-by-day. For each day:
1. Scan from macro (4H) down to micro (15s), just like discovery
2. At each zoom level, check if the pattern matches a library template
3. Only fire when the full cascade aligns AND brain approves
4. WaveRider manages position at 15s resolution

### File: `training/orchestrator.py`

#### Step 1: Save scaler in Phase 2.5

After `clustering_engine.create_templates(manifest)`, save the scaler:
```python
import pickle
scaler_path = os.path.join(self.checkpoint_dir, 'clustering_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(clustering_engine.scaler, f)
```

#### Step 2: Add `--forward-pass` CLI flag

In `main()`, add:
```python
parser.add_argument('--forward-pass', action='store_true', help="Run Phase 4 forward pass using existing playbook")
```

When `--forward-pass` is set, skip Phases 2/2.5/3, load the library, and call `run_forward_pass()`.

#### Step 3: Add `run_forward_pass()` method

```python
def run_forward_pass(self, data_source: str):
    """
    Phase 4: Forward pass — replay full year using playbook.
    Scans fractal cascade per day, matches templates, trades via WaveRider.
    Brain learns from outcomes.
    """
```

**Inner logic:**

1. **Load prerequisites**:
   - `pattern_library` from `checkpoints/pattern_library.pkl`
   - `clustering_scaler` from `checkpoints/clustering_scaler.pkl`
   - Build centroid matrix: numpy array of all library centroids (N_templates x 11D)
   - Build params lookup: `{template_id: params_dict}`

2. **Build centroid index for fast matching**:
   ```python
   template_ids = list(pattern_library.keys())
   centroids = np.array([pattern_library[tid]['centroid'] for tid in template_ids])
   # Scale centroids using the saved scaler
   centroids_scaled = scaler.transform(centroids)
   ```

3. **Iterate days** (load ATLAS 15s directory, group files by date):
   For each day:

   a. **Fractal cascade scan** — reuse `FractalDiscoveryAgent` but with a key difference:
      - Instead of collecting ALL patterns, we're looking for **actionable cascades**
      - Start with 4H bars → compute states → find structure_ok
      - For each 4H trigger: zoom into 1H within that time window
      - Continue: 15m → 5m → 1m → 15s
      - At each level, build the parent_chain as we descend

   b. **At entry level (15s)**: For each candidate pattern with its chain:
      - Extract 11D feature vector (same as clustering)
      - Scale using saved scaler
      - Find nearest centroid: `distances = np.linalg.norm(centroids_scaled - feature_scaled, axis=1)`
      - `nearest_idx = np.argmin(distances)`
      - `match_distance = distances[nearest_idx]`
      - `template_id = template_ids[nearest_idx]`

      - **Gate 1**: `match_distance < 3.0` (configurable threshold — reject if no close template)
      - **Gate 2**: `brain.should_fire(template_id)` (probability + confidence gate)
      - **Gate 3**: `wave_rider.position is None` (no overlapping trades)

      - If all gates pass:
        ```python
        params = pattern_library[template_id]['params']
        side = 'short' if state.z_score > 0 else 'long'  # from cascade/structure logic
        wave_rider.open_position(
            entry_price=price,
            side=side,
            state=state,
            stop_distance_ticks=params.get('stop_loss_ticks', 15)
        )
        ```

   c. **Position management**: After the cascade scan, iterate remaining 15s bars:
      ```python
      for bar in remaining_15s_bars:
          result = wave_rider.update_trail(bar.price, bar.state, bar.timestamp)
          if result['should_exit']:
              outcome = TradeOutcome(
                  state=entry_state,
                  entry_price=position_entry_price,
                  exit_price=result['exit_price'],
                  pnl=result['pnl'],
                  result='WIN' if result['pnl'] > 0 else 'LOSS',
                  timestamp=bar.timestamp,
                  exit_reason=result['exit_reason'],
                  entry_time=position_entry_time,
                  exit_time=bar.timestamp,
                  duration=bar.timestamp - position_entry_time,
                  direction=position_side.upper(),
                  template_id=matched_template_id
              )
              brain.update(outcome)
              day_trades.append(outcome)
      ```

   d. **End of day**:
      - Force-close any open position (TIME exit)
      - Run `batch_regret_analyzer.batch_analyze_day(day_trades, day_15s_data)`
      - Print day summary: trades, wins, losses, PnL, brain stats
      - Checkpoint brain state

4. **Final output**:
   ```
   === Phase 4 Forward Pass Complete ===
   Days: 250
   Total trades: 1,847
   Win rate: 52.3%
   Total PnL: $12,450.00
   Sharpe: 1.42
   Max drawdown: -$2,100.00
   Templates fired: 45/69
   Brain coverage: 38 templates with >30 trades
   ```

### Key Implementation Notes

- **Reuse `QuantumFieldEngine`** for state computation at each TF level — same GPU engine
- **Reuse `_batch_scan_windowed` logic** from discovery agent for the cascade — but instead of saving all patterns, filter for actionable ones only
- **The cascade scan can be simplified**: You don't need to scan ALL bars at each TF. Only scan within the time windows of the parent trigger. This is exactly what discovery already does.
- **Consider creating a `FractalCascadeScanner` helper** that wraps the discovery agent's logic but returns actionable signals instead of a full manifest

### Data Access

Each TF level reads from `DATA/ATLAS/{tf}/` directories. The discovery agent already handles loading parquet files by date. Reuse `_load_parquet_file()` and the ThreadPool I/O pattern.

---

## Task C: Phase 5 — Strategy Selection & Risk Scoring

### Concept

After Phase 4, the brain has win/loss records per template_id. Phase 5 analyzes this to produce:
1. **Strategy rankings** — which templates are profitable
2. **Risk scores** — which templates are dangerous
3. **Production playbook** — filtered set of approved strategies

### File: `training/orchestrator.py`

#### Add `run_strategy_selection()` method

Called after `run_forward_pass()` or separately via `--strategy-report` flag.

```python
def run_strategy_selection(self):
    """
    Phase 5: Analyze brain data + regret history to rank strategies.
    """
```

**Logic:**

1. **Collect per-template stats from brain**:
   ```python
   for template_id in pattern_library:
       data = brain.table.get(template_id, {'wins': 0, 'losses': 0, 'total': 0})
       prob = brain.get_probability(template_id)
       conf = brain.get_confidence(template_id)

       # Also compute from trade_history:
       template_trades = [t for t in brain.trade_history if t.template_id == template_id]
       pnls = [t.pnl for t in template_trades]

       if len(pnls) > 1:
           sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
           max_dd = compute_max_drawdown(pnls)  # cumulative PnL drawdown
           avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
           avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
       else:
           sharpe = 0
           max_dd = 0
           avg_win = 0
           avg_loss = 0
   ```

2. **Classify templates into tiers**:
   ```
   TIER 1 (Production): prob > 0.55, conf > 0.30, sharpe > 0.5, total > 30
   TIER 2 (Promising):  prob > 0.50, conf > 0.20, total > 15
   TIER 3 (Unproven):   total < 15 (insufficient data)
   TIER 4 (Toxic):      prob < 0.45 AND total > 20 (proven losers)
   ```

3. **Risk scoring per template**:
   ```python
   risk_score = (
       0.3 * (1 - win_rate) +          # Loss frequency
       0.3 * (abs(avg_loss) / (avg_win + 1e-6)) +  # Loss magnitude vs wins
       0.2 * (max_consecutive_losses / 10.0) +       # Streak risk
       0.2 * (abs(max_dd) / (total_pnl + 1e-6))     # Drawdown severity
   )
   ```

4. **Ancestry analysis** — which macro contexts produce best results:
   ```python
   # Group templates by their root ancestor type
   roche_backed = [t for t in tier1 if library[t]['centroid'][-1] == 1.0]  # root_is_roche
   struct_backed = [t for t in tier1 if library[t]['centroid'][-1] == 0.0]

   print(f"Roche-backed strategies: {len(roche_backed)} templates, avg Sharpe: ...")
   print(f"Structure-backed strategies: {len(struct_backed)} templates, avg Sharpe: ...")
   ```

5. **Output report**:
   ```
   === Phase 5: Strategy Selection ===

   TIER 1 — PRODUCTION READY (12 templates)
   ┌──────────┬────────┬──────┬───────┬────────┬──────────┬──────────┐
   │ Template │ Trades │ Win% │ Sharpe│ Tot PnL│ Max DD   │ Risk     │
   ├──────────┼────────┼──────┼───────┼────────┼──────────┼──────────┤
   │ T-150    │ 87     │ 58%  │ 1.82  │ $4,200 │ -$380    │ LOW      │
   │ T-72     │ 52     │ 55%  │ 1.34  │ $2,100 │ -$520    │ LOW      │
   │ ...      │        │      │       │        │          │          │
   └──────────┴────────┴──────┴───────┴────────┴──────────┴──────────┘

   TIER 4 — TOXIC (8 templates, EXCLUDED from production)
   ┌──────────┬────────┬──────┬───────┬────────┬──────────┬──────────┐
   │ T-91     │ 34     │ 32%  │ -0.8  │ -$1,800│ -$1,200  │ HIGH     │
   │ ...      │        │      │       │        │          │          │
   └──────────┴────────┴──────┴───────┴────────┴──────────┴──────────┘

   ANCESTRY ANALYSIS:
     4H Roche → lower TF entry: 8/12 Tier 1 templates (67%)
     4H Structural → lower TF entry: 4/12 Tier 1 templates (33%)
     Best macro context: 4H Roche Snap (avg Sharpe 1.6)

   PRODUCTION PLAYBOOK: 12 templates saved to checkpoints/production_playbook.pkl
   ```

6. **Save production playbook**:
   ```python
   production = {
       tid: {
           'centroid': library[tid]['centroid'],
           'params': library[tid]['params'],
           'win_rate': prob,
           'sharpe': sharpe,
           'risk_score': risk,
           'tier': 1
       }
       for tid in tier1_ids
   }
   pickle.dump(production, open('checkpoints/production_playbook.pkl', 'wb'))
   ```

### CLI

```bash
# Full pipeline: discovery → clustering → optimization → forward pass → strategy selection
python training/orchestrator.py --fresh --no-dashboard --iterations 50 --forward-pass --strategy-report

# Just forward pass + report (using existing playbook)
python training/orchestrator.py --forward-pass --strategy-report

# Just report (using existing brain data)
python training/orchestrator.py --strategy-report
```

---

## File Summary

| File | Action |
|------|--------|
| `training/fractal_discovery_agent.py` | Add `parent_chain` to PatternEvent, propagate through drill-down, update `_merge_windows()` |
| `training/fractal_clustering.py` | Expand feature vector from 7D to 11D (add ancestry features) |
| `training/orchestrator.py` | Save scaler in Phase 2.5, add `run_forward_pass()` (Phase 4), add `run_strategy_selection()` (Phase 5), CLI flags |

## Execution Order

1. **Task A first** — modifies discovery + clustering. Run `--fresh` to rebuild with star schema.
2. **Task B** — Phase 4 forward pass. Requires Task A's output (library with 11D centroids).
3. **Task C** — Phase 5 strategy selection. Requires Task B's output (brain with trade history).

## Key Design Decisions

- **11D feature vector** — adds 4 ancestry features without blowing up dimensionality
- **Fixed ancestry encoding** — parent_z, parent_mom, root_z, root_is_roche covers 80% of information
- **Centroid matching threshold** — 3.0 in scaled space (configurable). Too tight = no trades, too loose = bad matches
- **Brain gates execution** — pessimistic prior (9% win rate) means early days are cautious, later days are informed
- **One position at a time** — WaveRider enforces this naturally
- **Regret analysis per day** — BatchRegretAnalyzer feeds into Phase 5 risk scoring
- **Production playbook** — only Tier 1 templates survive for live trading
