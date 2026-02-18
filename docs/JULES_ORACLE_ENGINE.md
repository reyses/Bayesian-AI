# JULES TASK: Oracle-Fact Engine & Star Schema Analytics

## PRIORITY: CRITICAL

## OBJECTIVE
Transform the system from "Passive Observer" to "Active Judge."

- **The Fact (Source)**: Every discovered pattern is immediately judged by an Oracle (Look-Ahead) to establish its Ground Truth outcome.
- **The Dimension (Cluster)**: Templates aggregate these Facts to calculate Win Rate, Risk (Variance), and Next-State Transitions.
- **The Blindfold**: Clustering remains pure (Physics only), but the resulting Template carries the accumulated wisdom of its members.
- **The Audit**: The Simulator compares the Strategy's decision against the Pattern's hidden Marker.

---

## RULE: NO MAGIC NUMBERS

**Every numeric threshold, window size, ratio, or multiplier MUST be defined as a named constant at the top of the file or in a config dict.** No bare literals in logic.

Bad:
```python
if max_up > 5 * 0.25:        # What is 5? What is 0.25?
    if max_up > 3.0 * max_down:  # Why 3.0?
```

Good:
```python
# --- Oracle Configuration ---
ORACLE_MIN_MOVE_TICKS = 5          # Minimum move in ticks to qualify as non-noise
ORACLE_HOME_RUN_RATIO = 3.0        # MFE/MAE ratio for "Mega" classification
ORACLE_SCALP_RATIO = 1.2           # MFE/MAE ratio for "Scalp" classification

if max_up > ORACLE_MIN_MOVE_TICKS * tick_size:
    if max_up > ORACLE_HOME_RUN_RATIO * max_down:
```

This applies to ALL code in this task — oracle thresholds, lookahead windows, classification boundaries, transition probabilities, everything.

---

## STEP 1: CONFIGURATION CONSTANTS

Create `config/oracle_config.py`:

```python
"""
Oracle Engine Configuration
All thresholds and parameters in one place.
"""

# Lookahead windows per timeframe (in bars)
# Chosen to represent ~1-2 "meaningful move" durations at each resolution
ORACLE_LOOKAHEAD_BARS = {
    '1s':  300,   # 5 minutes
    '5s':  120,   # 10 minutes
    '15s':  60,   # 15 minutes
    '1m':   60,   # 1 hour
    '5m':   24,   # 2 hours
    '15m':  16,   # 4 hours
    '1h':    8,   # 8 hours (1 trading day)
    '4h':    6,   # 24 hours
    '1D':    5,   # 5 trading days
    '1W':    4,   # 4 weeks
}

# Classification thresholds
ORACLE_MIN_MOVE_TICKS = 5          # Min move in ticks to be non-noise
ORACLE_HOME_RUN_RATIO = 3.0        # MFE/MAE ratio for Mega classification (+/-2)
ORACLE_SCALP_RATIO = 1.2           # MFE/MAE ratio for Scalp classification (+/-1)

# Marker values (semantic names for clarity)
MARKER_MEGA_LONG = 2
MARKER_SCALP_LONG = 1
MARKER_NOISE = 0
MARKER_SCALP_SHORT = -1
MARKER_MEGA_SHORT = -2

# Template intelligence thresholds
TEMPLATE_MIN_MEMBERS_FOR_STATS = 5      # Need at least N patterns to compute stats
TEMPLATE_TOXIC_RISK_THRESHOLD = 0.70    # risk_score above this = toxic
TEMPLATE_HIGH_WIN_RATE = 0.55           # Above this = promising

# Transition matrix
TRANSITION_MIN_SEQUENCE_GAP_BARS = 1    # Min bars between events to count as transition
TRANSITION_MAX_SEQUENCE_GAP_BARS = 100  # Max bars to look for next event
```

---

## STEP 2: UPGRADE DATA SCHEMA

### File: `core/pattern_utils.py` (or wherever PatternEvent lives)

Add oracle fields to PatternEvent:

```python
@dataclass
class PatternEvent:
    # ... existing physics fields ...
    pattern_id: str = ""                    # Unique ID (f"{timestamp}_{bar_index}")

    # THE ORACLE FACT (Hidden from Clustering, Visible to Audit)
    oracle_marker: int = 0                  # MARKER_* constants from oracle_config
    oracle_meta: Dict = field(default_factory=dict)  # {'mfe': float, 'mae': float, 'lookahead_bars': int}
```

Add intelligence fields to PatternTemplate:

```python
@dataclass
class PatternTemplate:
    # ... existing fields (centroid, member_count, etc.) ...

    # THE STAR SCHEMA DIMENSION (Aggregated from member oracle markers)
    # 1. Performance Stats
    stats_win_rate: float = 0.0             # Fraction of members with |oracle_marker| >= 1
    stats_expectancy: float = 0.0           # Mean (mfe - mae) across members
    stats_mega_rate: float = 0.0            # Fraction of members with |oracle_marker| == 2

    # 2. Risk Profile
    risk_variance: float = 0.0              # StdDev of member MFE values
    risk_score: float = 0.0                 # 0.0 (Safe) to 1.0 (Toxic)

    # 3. Direction Bias
    long_bias: float = 0.0                  # Fraction of positive markers (1,2) vs total non-noise
    short_bias: float = 0.0                 # Fraction of negative markers (-1,-2) vs total non-noise

    # 4. Navigation (Markov Transition Map)
    transition_map: Dict[int, float] = field(default_factory=dict)  # {target_template_id: probability}
```

---

## STEP 3: IMPLEMENT THE ORACLE

### File: `training/fractal_discovery_agent.py`

Add `_consult_oracle` method and wire it into pattern scanning.

```python
from config.oracle_config import (
    ORACLE_LOOKAHEAD_BARS, ORACLE_MIN_MOVE_TICKS,
    ORACLE_HOME_RUN_RATIO, ORACLE_SCALP_RATIO,
    MARKER_MEGA_LONG, MARKER_SCALP_LONG, MARKER_NOISE,
    MARKER_SCALP_SHORT, MARKER_MEGA_SHORT
)

class FractalDiscoveryAgent:

    def _consult_oracle(self, df, bar_index, timeframe, tick_size):
        """
        Judge a pattern using future price data.

        Looks ahead N bars (timeframe-dependent) and classifies the outcome
        based on the ratio of max favorable vs max adverse excursion.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            bar_index: Index of the pattern bar
            timeframe: String timeframe key (e.g., '15m', '1h')
            tick_size: Asset tick size for min-move calculation

        Returns:
            (marker: int, meta: dict)
            marker: One of MARKER_* constants
            meta: {'mfe': float, 'mae': float, 'lookahead_bars': int}
        """
        lookahead = ORACLE_LOOKAHEAD_BARS.get(timeframe, 60)

        if bar_index + lookahead >= len(df):
            return MARKER_NOISE, {}

        entry_price = df.iloc[bar_index]['close']
        future_slice = df.iloc[bar_index + 1 : bar_index + 1 + lookahead]

        max_up = float(future_slice['high'].max() - entry_price)
        max_down = float(entry_price - future_slice['low'].min())

        min_move = ORACLE_MIN_MOVE_TICKS * tick_size
        marker = MARKER_NOISE

        # Long-side classification (price went up more than down)
        if max_up > min_move:
            if max_down > 0 and max_up > ORACLE_HOME_RUN_RATIO * max_down:
                marker = MARKER_MEGA_LONG
            elif max_down > 0 and max_up > ORACLE_SCALP_RATIO * max_down:
                marker = MARKER_SCALP_LONG
            elif max_down == 0:
                marker = MARKER_MEGA_LONG  # No adverse move at all

        # Short-side classification (price went down more than up)
        if max_down > min_move:
            if max_up > 0 and max_down > ORACLE_HOME_RUN_RATIO * max_up:
                marker = MARKER_MEGA_SHORT
            elif max_up > 0 and max_down > ORACLE_SCALP_RATIO * max_up:
                marker = MARKER_SCALP_SHORT
            elif max_up == 0:
                marker = MARKER_MEGA_SHORT

        meta = {
            'mfe': max_up,
            'mae': max_down,
            'lookahead_bars': lookahead
        }

        return marker, meta
```

### Wire into scan loop:

Wherever patterns are created in the discovery loop, call the oracle:

```python
    # After detecting a pattern at bar_index in df for timeframe tf:
    marker, meta = self._consult_oracle(df, bar_index, tf, self.tick_size)

    event = PatternEvent(
        pattern_id=f"{timestamp}_{bar_index}",
        # ... existing fields ...
        oracle_marker=marker,
        oracle_meta=meta,
    )
```

**IMPORTANT**: `self.tick_size` must be set from the asset config. If the discovery agent doesn't have it, pass it through from the orchestrator (MNQ tick_size = 0.25).

---

## STEP 4: BLIND CLUSTERING + INTELLIGENCE AGGREGATION

### File: `training/fractal_clustering.py`

Clustering uses ONLY physics features (existing behavior — do NOT change this).

After clustering, aggregate oracle intelligence per template:

```python
from config.oracle_config import (
    TEMPLATE_MIN_MEMBERS_FOR_STATS,
    MARKER_MEGA_LONG, MARKER_SCALP_LONG,
    MARKER_SCALP_SHORT, MARKER_MEGA_SHORT, MARKER_NOISE
)

def _aggregate_oracle_intelligence(self, template, patterns):
    """
    Post-clustering: compute template-level stats from member oracle markers.
    Called AFTER clustering is complete. Does NOT influence cluster assignment.
    """
    markers = [p.oracle_marker for p in patterns if hasattr(p, 'oracle_marker')]

    if len(markers) < TEMPLATE_MIN_MEMBERS_FOR_STATS:
        return  # Not enough data

    # 1. Win Rate (any non-noise outcome)
    wins = sum(1 for m in markers if abs(m) >= 1)
    template.stats_win_rate = wins / len(markers)

    # 2. Mega Rate (home runs)
    megas = sum(1 for m in markers if abs(m) == 2)
    template.stats_mega_rate = megas / len(markers)

    # 3. Expectancy (mean MFE - MAE from oracle_meta)
    mfe_values = []
    mae_values = []
    for p in patterns:
        meta = getattr(p, 'oracle_meta', {})
        if 'mfe' in meta and 'mae' in meta:
            mfe_values.append(meta['mfe'])
            mae_values.append(meta['mae'])

    if mfe_values:
        template.stats_expectancy = np.mean(mfe_values) - np.mean(mae_values)
        template.risk_variance = float(np.std(mfe_values))

    # 4. Risk Score (0 = safe, 1 = toxic)
    # High variance + low win rate = toxic
    if template.stats_win_rate > 0:
        # Coefficient of variation normalized to [0,1]
        cv = template.risk_variance / (np.mean(mfe_values) + 1e-9) if mfe_values else 1.0
        template.risk_score = min(1.0, cv * (1.0 - template.stats_win_rate))
    else:
        template.risk_score = 1.0

    # 5. Direction Bias
    non_noise = [m for m in markers if m != MARKER_NOISE]
    if non_noise:
        longs = sum(1 for m in non_noise if m > 0)
        shorts = sum(1 for m in non_noise if m < 0)
        total_nn = len(non_noise)
        template.long_bias = longs / total_nn
        template.short_bias = shorts / total_nn
```

Call this after `create_templates()` returns:

```python
for template in final_templates:
    member_patterns = [patterns[i] for i in template.member_indices]
    self._aggregate_oracle_intelligence(template, member_patterns)
```

---

## STEP 5: TRANSITION MATRIX (Markov Map)

### File: `training/fractal_clustering.py`

After all templates have members assigned, compute inter-template transitions:

```python
from config.oracle_config import (
    TRANSITION_MIN_SEQUENCE_GAP_BARS,
    TRANSITION_MAX_SEQUENCE_GAP_BARS
)

def _build_transition_matrix(self, templates, all_patterns):
    """
    For each template, count how often its members are followed by
    members of other templates (sorted by time).

    This creates a Markov transition map: P(next_template | current_template).
    """
    # Sort all patterns globally by timestamp
    sorted_patterns = sorted(all_patterns, key=lambda p: p.timestamp)

    # Build pattern -> template_id lookup
    pattern_to_template = {}
    for template in templates:
        for p in template.patterns:
            pattern_to_template[id(p)] = template.template_id

    # Count transitions
    for i in range(len(sorted_patterns) - 1):
        curr = sorted_patterns[i]
        curr_tid = pattern_to_template.get(id(curr))
        if curr_tid is None:
            continue

        # Find next pattern within gap window
        for j in range(i + 1, len(sorted_patterns)):
            nxt = sorted_patterns[j]
            gap = nxt.timestamp - curr.timestamp
            if gap < TRANSITION_MIN_SEQUENCE_GAP_BARS:
                continue
            if gap > TRANSITION_MAX_SEQUENCE_GAP_BARS:
                break

            nxt_tid = pattern_to_template.get(id(nxt))
            if nxt_tid is not None and nxt_tid != curr_tid:
                # Record transition
                template_map = next(t for t in templates if t.template_id == curr_tid)
                if nxt_tid not in template_map.transition_map:
                    template_map.transition_map[nxt_tid] = 0
                template_map.transition_map[nxt_tid] += 1
                break  # Only count first transition

    # Normalize to probabilities
    for template in templates:
        total = sum(template.transition_map.values())
        if total > 0:
            template.transition_map = {
                k: v / total for k, v in template.transition_map.items()
            }
```

---

## STEP 6: THE AUDIT GATE

### File: `training/orchestrator_worker.py` (or wherever trades are evaluated)

After a trade completes, compare the agent's decision with the oracle:

```python
def _audit_trade(self, outcome, pattern):
    """
    Compare strategy decision against oracle ground truth.

    Returns dict with audit metrics:
        - oracle_match: bool (did strategy agree with oracle?)
        - oracle_marker: int (what the oracle said)
        - classification: str (TP, FP, TN, FN)
    """
    oracle_says = getattr(pattern, 'oracle_marker', MARKER_NOISE)

    # True Positive: Agent traded in oracle's direction and oracle was right
    # False Positive: Agent traded but oracle said noise or opposite
    # True Negative: Agent skipped and oracle said noise
    # False Negative: Agent skipped but oracle said profitable

    if outcome is not None:
        # Agent traded
        agent_long = outcome.direction == 'LONG'
        oracle_long = oracle_says > 0
        oracle_short = oracle_says < 0

        if agent_long and oracle_long:
            classification = 'TP'
        elif not agent_long and oracle_short:
            classification = 'TP'
        elif oracle_says == MARKER_NOISE:
            classification = 'FP_NOISE'   # Traded noise
        else:
            classification = 'FP_WRONG'   # Traded wrong direction
    else:
        # Agent skipped
        if oracle_says == MARKER_NOISE:
            classification = 'TN'
        else:
            classification = 'FN'  # Missed a real move

    return {
        'oracle_match': classification == 'TP',
        'oracle_marker': oracle_says,
        'classification': classification
    }
```

---

## STEP 7: ORACLE REPORT

Add an oracle summary to the Phase 4/5 reports:

```
ORACLE AUDIT SUMMARY
----------------------------------------
  True Positive (correct trade):     145 (42%)
  False Positive - Noise:             89 (26%)
  False Positive - Wrong Dir:         67 (19%)
  False Negative (missed move):       45 (13%)

  Precision: 54%  (TP / (TP + FP))
  Recall: 76%     (TP / (TP + FN))

  Template Oracle Alignment:
    T-144@15m: 78% aligned (32 trades, 25 TP)  <-- BEST
    T-90@1h:   23% aligned (40 trades, 9 TP)   <-- WORST
```

---

## FILES TO CREATE
1. `config/oracle_config.py` (~40 lines) — All constants

## FILES TO MODIFY
1. `core/pattern_utils.py` (or PatternEvent location) — Add oracle_marker, oracle_meta fields
2. `training/fractal_discovery_agent.py` — Add _consult_oracle, wire into scan loop
3. `training/fractal_clustering.py` — Add _aggregate_oracle_intelligence, _build_transition_matrix
4. `training/orchestrator_worker.py` — Add _audit_trade
5. `training/orchestrator.py` — Add oracle report section to Phase 4/5 output

## FILES TO NOT TOUCH
- `core/quantum_field_engine.py` — Physics engine unchanged
- `core/cuda_physics.py` — CUDA kernels unchanged
- `training/monte_carlo_engine.py` — MC sweep is independent
- `training/anova_analyzer.py` — ANOVA is independent

## VERIFICATION
1. Run `python training/orchestrator.py --fresh` — oracle markers should appear in discovery output
2. After clustering, templates should show stats_win_rate, risk_score, long_bias, short_bias
3. Oracle report should show TP/FP/TN/FN breakdown
4. Clustering assignments should be IDENTICAL with and without oracle (blindfold test)
