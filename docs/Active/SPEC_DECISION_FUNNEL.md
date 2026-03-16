# Decision Funnel — Temporal Candidate Narrowing

**Date**: 2026-03-15
**Status**: SPEC — research phase
**Priority**: HIGH (addresses RPN 648 — score competition ignores quality)

---

## 1. Problem Statement

The system evaluates all candidates on a single bar and picks the closest template match.
There is no temporal dimension to entry decisions. A candidate that appeared 10 minutes ago
and survived changing market conditions is treated identically to one that appeared just now.

**Current flow:**
```
Bar arrives → 7 candidates → gate cascade → score (distance) → 1 winner → ENTER
```

**No memory of previous bars.** No narrowing over time. No survival signal.

## 2. Core Concept

**The Decision Funnel** tracks candidate signals across bars. Survival duration under
evolving market conditions is a quality signal that no snapshot evaluation can replicate.

```
t-10m: 400 candidates pass initial screening across all TFs
t-7m:  200 survive — some regimes collapsed, workers flipped
t-3m:   40 survive — convergence is clear, direction forming
t-1m:    5 survive — conviction confirmed, momentum aligned
t-0:     1 trade  — the most validated candidate
```

**Key insight**: the act of surviving the funnel IS the quality signal. A candidate
that persisted through 40 bars of market evolution has been stress-tested by the
market itself. No backward lookback can reconstruct this.

## 3. What is a "Candidate" Over Time?

A candidate is NOT a specific pattern event (those are per-bar). A candidate is a
**template × direction × depth** tuple that persists when:

1. The template's centroid is still within gate1_dist of the current bar's features
2. The direction is still supported by the TBN (conviction above threshold)
3. The z-score regime hasn't collapsed (still in actionable zone)
4. No exit-level signal has invalidated the thesis (regime decay, DI crossover)

When any of these fail, the candidate is eliminated from the funnel. When all four
hold across multiple bars, the candidate is a **funnel survivor**.

## 4. Data Structure

```python
@dataclass
class FunnelCandidate:
    template_id: int
    direction: str              # 'long' | 'short'
    depth: int
    first_seen_bar: int         # bar index when candidate entered funnel
    bars_survived: int          # count of bars still valid
    conviction_path: list       # conviction at each bar [0.55, 0.58, 0.61, ...]
    workers_aligned_path: list  # count of aligned workers per bar
    z_score_path: list          # z-score evolution
    adx_path: list              # ADX strength evolution
    eliminated_bar: int = None  # bar index when eliminated (None = still active)
    elimination_reason: str = '' # why it died

@dataclass
class DecisionFunnel:
    active: Dict[tuple, FunnelCandidate]   # (tid, direction, depth) → candidate
    history: List[FunnelCandidate]         # eliminated candidates (for analysis)

    # Funnel metrics (per bar)
    candidates_at_bar: List[int]           # [400, 350, 200, 100, 40, 5]
    narrowing_rate: float                  # how fast the funnel contracts
```

## 5. Funnel Operations

### 5a. On Each Bar (pre-entry evaluation)

```python
def on_bar(self, bar_idx, bar_state, belief_network):
    """Update funnel: add new candidates, eliminate stale ones."""

    # 1. Generate new candidates from current bar's features
    new_candidates = self._detect_candidates(bar_state)
    for c in new_candidates:
        key = (c.template_id, c.direction, c.depth)
        if key not in self.active:
            self.active[key] = FunnelCandidate(
                template_id=c.template_id,
                direction=c.direction,
                depth=c.depth,
                first_seen_bar=bar_idx,
                bars_survived=0,
                conviction_path=[],
                workers_aligned_path=[],
                z_score_path=[],
                adx_path=[],
            )

    # 2. Validate all active candidates against current state
    belief = belief_network.get_belief()
    for key, fc in list(self.active.items()):
        alive, reason = self._validate(fc, bar_state, belief)
        if alive:
            fc.bars_survived += 1
            fc.conviction_path.append(belief.conviction if belief else 0)
            fc.workers_aligned_path.append(self._count_aligned(belief, fc.direction))
            fc.z_score_path.append(abs(bar_state.z_score))
            fc.adx_path.append(getattr(bar_state, 'adx_strength', 0))
        else:
            fc.eliminated_bar = bar_idx
            fc.elimination_reason = reason
            self.history.append(fc)
            del self.active[key]

    # 3. Record funnel width for narrowing rate
    self.candidates_at_bar.append(len(self.active))
```

### 5b. Entry Decision (when a pattern fires)

```python
def get_mature_candidates(self, min_survival_bars=8):
    """Return candidates that have survived long enough to trade."""
    mature = []
    for fc in self.active.values():
        if fc.bars_survived < min_survival_bars:
            continue

        # Conviction must be rising (not flat or falling)
        if len(fc.conviction_path) >= 3:
            conv_trend = fc.conviction_path[-1] - fc.conviction_path[-3]
            if conv_trend <= 0:
                continue

        # Workers must be converging (more aligned over time)
        if len(fc.workers_aligned_path) >= 3:
            align_trend = fc.workers_aligned_path[-1] - fc.workers_aligned_path[-3]
            if align_trend < 0:
                continue

        mature.append(fc)

    return mature
```

### 5c. Funnel-Aware Scoring

```python
def funnel_score(fc: FunnelCandidate, gate_result: GateResult) -> float:
    """Score that combines template match quality with funnel survival."""

    # Survival duration: log scale (diminishing returns after 20 bars)
    survival_score = min(1.0, fc.bars_survived / 20.0)

    # Conviction trajectory: rising = good, flat/falling = bad
    conv_trend = np.polyfit(range(len(fc.conviction_path)), fc.conviction_path, 1)[0]
    conv_score = np.clip(conv_trend * 10, -1, 1)  # normalize

    # Worker convergence: more workers aligning over time
    if len(fc.workers_aligned_path) >= 3:
        align_delta = fc.workers_aligned_path[-1] - fc.workers_aligned_path[0]
        align_score = np.clip(align_delta / 5, -1, 1)
    else:
        align_score = 0

    # ADX trend: rising = trending market, falling = collapsing
    if len(fc.adx_path) >= 3:
        adx_trend = fc.adx_path[-1] - fc.adx_path[0]
        adx_score = np.clip(adx_trend / 10, -1, 1)
    else:
        adx_score = 0

    # Combine with template distance (lower = better, invert)
    distance_score = 1.0 - min(1.0, gate_result.dist / 3.0)

    return (
        survival_score * 0.25 +
        conv_score * 0.25 +
        align_score * 0.15 +
        adx_score * 0.10 +
        distance_score * 0.25
    )
```

## 6. Integration with Existing System

The funnel does NOT replace the gate cascade. It sits BEFORE it:

```
Bar arrives
  │
  ├─ FUNNEL UPDATE: add new candidates, validate existing, eliminate stale
  │
  ├─ PATTERN FIRES: does a gate-passing signal exist on this bar?
  │   └─ YES: check if it matches a mature funnel candidate
  │       ├─ MATCH: use funnel score (survival + conviction trend + distance)
  │       └─ NO MATCH: use current score (distance only) — penalized
  │
  ├─ GATE CASCADE: quality/depth/distance/brain
  ├─ COMPETITION: funnel-aware scoring
  └─ ENTRY
```

**New signal with no funnel history**: still traded, but penalized in score.
The system doesn't require funnel maturity — it just rewards it.

## 7. Validation Plan

### Phase 1: Offline Analysis (standalone research)

Use existing IS signal_log + I-MR auto seeds to simulate the funnel retroactively:

1. Load IS 15s data bar-by-bar
2. At each bar, compute which templates would pass gate 0-1 (feature extraction + distance)
3. Track template×direction persistence across bars
4. When a REAL trade occurred (from signal_log), check:
   - Was this template×direction in the funnel? For how many bars?
   - Was conviction rising or falling in the prior bars?
   - What was the funnel width (how many competitors survived)?
5. Correlate funnel metrics with trade outcome (PnL, WR)

**Hypothesis**: trades where the template×direction survived 8+ bars in the funnel
have higher WR and $/trade than trades where it appeared on the entry bar.

### Phase 2: I-MR Seed Validation

Use auto seeds from `tools/imr_regime_segments.py` as ground truth:
- I-MR regimes are structural moves lasting 10-300 minutes
- If a funnel candidate aligns with an I-MR regime start, the funnel correctly
  identified a developing move
- If a funnel candidate appears mid-regime or at regime end, it's late

### Phase 3: Integration

If Phase 1 confirms the hypothesis:
1. Add `DecisionFunnel` to `ExecutionEngine`
2. Modify `_finalize_entry()` to use funnel-aware scoring
3. Add funnel metrics to signal_log for ongoing monitoring

## 8. Research Tool Specification

```
tools/funnel_research.py

Usage:
    python tools/funnel_research.py --data DATA/ATLAS_1WEEK
    python tools/funnel_research.py --data DATA/ATLAS --month 2025_03
    python tools/funnel_research.py --seeds DATA/regime_seeds/imr_auto/imr_seeds_all_*.json

Output:
    reports/findings/funnel_research_YYYYMMDD.txt

Analyses:
    1. Template persistence: how many bars does a template×direction survive?
    2. Conviction trajectory at entry: was it rising for funnel survivors?
    3. Funnel width at entry: narrow funnel = validated signal?
    4. Correlation: funnel_score vs trade PnL/WR
    5. I-MR alignment: do funnel survivors coincide with regime starts?
```

## 9. Expected Outcomes

### Optimistic
- Funnel survivors show 80%+ WR vs 66% for snapshot entries
- Conviction trend is the strongest predictor of trade quality
- The 2-5 minute sweet spot trades are overwhelmingly funnel survivors
- The <30s noise trades are overwhelmingly non-survivors (appeared at t-0)

### Pessimistic
- Template×direction persistence is random (no correlation with outcome)
- Conviction fluctuates too fast for trend detection at 15s resolution
- The funnel adds latency without improving quality (late entries)
- Markets are efficient enough that survival = the move already happened

### Null Result
- Funnel metrics correlate in IS but not OOS (overfit to template library)
- In this case, the lookback 6D geometry (price shape) may be sufficient

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Funnel adds entry latency (miss fast moves) | Fewer trades, lower total PnL | Minimum survival = 4 bars (1 min), not 40 |
| State management bugs | Wrong funnel state → wrong entries | Unit test funnel with synthetic data |
| Overfit to IS template persistence patterns | OOS funnel doesn't narrow same way | Validate on ATLAS_OOS separately |
| Compute cost (re-evaluate all templates per bar) | Slower forward pass | Cache feature extraction, only recompute distance |

---

## 11. Relationship to Existing Features

| Feature | What it captures | Funnel adds |
|---------|-----------------|-------------|
| Lookback 6D | Price path shape (10 bars) | Market EVENT evolution (40 bars) |
| Wave maturity | Pattern exhaustion at entry | Candidate persistence over time |
| Worker agreement | Snapshot consensus | Convergence TRAJECTORY |
| Conviction | Snapshot belief strength | Belief MOMENTUM (rising/falling) |
| Score competition | Best template distance | Best VALIDATED candidate |
