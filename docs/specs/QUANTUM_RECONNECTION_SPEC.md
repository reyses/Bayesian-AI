# Quantum Reconnection Phase 1 — Instrumentation + Soft Thresholds
> Spec version: 1.0 | Date: 2026-03-16 | Status: PROPOSED

## 1. Objective

Replace the binary gate cascade in `ExecutionEngine._gate_check()` (line 648)
and `_finalize_entry()` (line 836) with a soft probabilistic scoring system that
uses the 9 orphaned quantum state variables already computed every bar on GPU.

Current state: `distance < 3.0 AND conviction > 0.48 AND momentum aligned -> TRADE`
Target state:  `P(success) = weighted combination of quantum fields -> TRADE if P > threshold`

Phase 1 scope: instrument all fields, convert 4 hard gates to soft multipliers,
wire `P(success)` into score competition. Does NOT replace `_gate_check` entirely
(that is Phase 5).

---

## 2. Variables to Wire

### 2.1 Wave Function Probabilities

| Variable | Computed | Stored | Currently Read |
|----------|----------|--------|----------------|
| `P_at_center` | `statistical_field_engine.py:476` (`prob0[i]`) | `MarketState.P_at_center` (`market_state.py:69`) | Nowhere |
| `P_near_upper` | `statistical_field_engine.py:477` (`prob1[i]`) | `MarketState.P_near_upper` (`market_state.py:70`) | Nowhere |
| `P_near_lower` | `statistical_field_engine.py:478` (`prob2[i]`) | `MarketState.P_near_lower` (`market_state.py:71`) | Nowhere |

**Read target**: `execution_engine.py` — accessed via `cand.state.P_at_center` etc.
in the new `_compute_p_success()` method.

### 2.2 Entropy (Chaos Measure)

| Variable | Computed | Stored | Currently Read |
|----------|----------|--------|----------------|
| `entropy` | `statistical_field_engine.py:479` | `MarketState.entropy` | Nowhere |
| `entropy_normalized` | `statistical_field_engine.py:480` | `MarketState.entropy_normalized` | `feature_extraction.py:31` (feat[15] passthrough only) |

**Read target**: `execution_engine.py` — `(1 - entropy_normalized)` as order measure
in `P(success)`.

### 2.3 Reversion Probability (Tunnel Probability)

| Variable | Computed | Stored | Currently Read |
|----------|----------|--------|----------------|
| `reversion_probability` | `statistical_field_engine.py:488` | `MarketState.reversion_probability` | `execution_engine.py:760` (hard gate only) |

**Change**: Currently used as hard block (`< reversion_prob_min` -> skip).
Convert to continuous weight in `P(success)`.

### 2.4 Coherence (Oscillation Entropy Normalized)

| Variable | Computed | Stored | Currently Read |
|----------|----------|--------|----------------|
| `oscillation_entropy_normalized` | `statistical_field_engine.py:386,502` | `MarketState.oscillation_entropy_normalized` | `feature_extraction.py:36` (feat[15]), `timeframe_belief_network.py:179` |

This IS the coherence field — renamed from quantum terminology. Value in (0,1]
where 1.0 = perfect oscillation alignment, 0.0 = maximum disorder.

**Read target**: `execution_engine.py` — direct multiplier in score competition
(Step 2 below).

### 2.5 Additional Orphans (instrument only in Phase 1)

| Variable | Computed | Stored | Phase 1 Action |
|----------|----------|--------|----------------|
| `breakout_probability` | `SFE:401` | `MarketState` | Log to signal_log. No scoring use yet. |
| `reversion_potential` | `SFE:403` | `MarketState` | Log to signal_log. No scoring use yet. |
| `prob_weight_center/upper/lower` | `SFE:473-475` | `MarketState` (complex) | No action — these are sqrt(P_i), redundant with P_i. |

---

## 3. Scoring Formula

### 3.1 Original Design Intent (from QUANTUM_DESIGN_INTENT.md)

```
ENTER when:
  max(P_at_center, P_near_upper, P_near_lower) > 0.75   -- wave function collapsed
  AND reversion_probability > 0.60                        -- tunnel clear
  AND entropy LOW                                         -- chaos resolved
```

No hard boolean gates. Continuous spectrum of expected value.

### 3.2 Proposed: P(success)

```python
def _compute_p_success(self, state, conviction, dir_prob_spread) -> float:
    """Probabilistic trade quality score. Returns P in [0, 1]."""

    P_wave     = max(state.P_at_center, state.P_near_upper, state.P_near_lower)
    P_tunnel   = state.reversion_probability
    P_order    = 1.0 - state.entropy_normalized       # low entropy = high order
    P_coherence = state.oscillation_entropy_normalized  # cross-TF alignment
    P_conviction = conviction                           # TBN path conviction [0,1]
    P_brain    = dir_prob_spread                        # |P(long) - P(short)| from brain
    P_momentum = sigmoid(F_momentum * side_sign)        # directional alignment [0,1]

    P_success = (
        W_WAVE       * P_wave          # 0.20 — wave function collapse
      + W_TUNNEL     * P_tunnel        # 0.15 — reversion probability
      + W_ENTROPY    * P_order         # 0.15 — chaos resolution
      + W_CONVICTION * P_conviction    # 0.15 — TBN path agreement
      + W_BRAIN      * P_brain         # 0.10 — historical template WR
      + W_MOMENTUM   * P_momentum      # 0.10 — momentum alignment
      + W_TEMPLATE   * exp(-dist²/2)   # 0.10 — cluster proximity
      + W_COHERENCE  * P_coherence     # 0.05 — oscillation coherence
    )
    return P_success
```

### 3.3 Weights (TradingConfig fields)

| Weight | Value | Config Field | Rationale |
|--------|-------|-------------|-----------|
| `W_WAVE` | 0.20 | `p_success_w_wave` | Core quantum signal — dominant probability |
| `W_TUNNEL` | 0.15 | `p_success_w_tunnel` | Path clearance to target |
| `W_ENTROPY` | 0.15 | `p_success_w_entropy` | Chaos filter — low entropy = tradeable |
| `W_CONVICTION` | 0.15 | `p_success_w_conviction` | Multi-TF path agreement |
| `W_BRAIN` | 0.10 | `p_success_w_brain` | Bayesian learning signal |
| `W_MOMENTUM` | 0.10 | `p_success_w_momentum` | Direction confidence |
| `W_TEMPLATE` | 0.10 | `p_success_w_template` | Pattern match quality |
| `W_COHERENCE` | 0.05 | `p_success_w_coherence` | Oscillation alignment bonus |
| **Sum** | **1.00** | | Weights must sum to 1.0 |

### 3.4 Thresholds

| Threshold | Value | Config Field | Meaning |
|-----------|-------|-------------|---------|
| `P_SUCCESS_TRADE` | 0.55 | `p_success_min` | Minimum P(success) to allow entry |
| `P_SUCCESS_HIGH` | 0.75 | `p_success_high` | High-confidence — allow larger sizing |

### 3.5 Sigmoid Helper

```python
def _sigmoid(x: float, scale: float = 5.0) -> float:
    return 1.0 / (1.0 + math.exp(-scale * x))
```

`scale=5.0` gives sigmoid(0.2)=0.73, sigmoid(-0.2)=0.27 — reasonable discrimination
around zero. Tune from data.

---

## 4. Soft Threshold Conversions

### 4.0 Gate 0 — Hurst (line 750)

**Current**: Hard block if `hurst_exponent < hurst_min` (default 0.40).

**Soft replacement**: Probability multiplier.
```python
# Replace hard block with continuous penalty
if hurst < 0.30:
    P_hurst_mult = 0.25    # severe penalty, nearly blocked
elif hurst < hurst_min:
    P_hurst_mult = 0.25 + 0.75 * (hurst - 0.30) / (hurst_min - 0.30)  # linear ramp
else:
    P_hurst_mult = 1.0     # no penalty
# Apply: P_success *= P_hurst_mult
```

**What this changes**: Trades in the 0.30-0.40 hurst range are no longer hard-blocked.
They receive a 25-100% penalty multiplier, allowing them if other signals are strong.

**Risk**: Low-hurst trades historically lose more. Mitigated by the combined P(success)
threshold still blocking weak setups. Monitor hurst-bucketed WR in IS report.

### 4.1 Gate 2 — Brain Reject (line 798)

**Current**: Hard block if `brain.should_fire(tid) == False` (binary).

**Soft replacement**: Template WR as continuous weight.
```python
# Replace binary should_fire with continuous probability
brain_stats = self.brain.get_stats(tid)
if brain_stats and brain_stats['total'] >= 5:
    P_brain = brain_stats['wins'] / brain_stats['total']
else:
    P_brain = 0.50  # uninformative prior

# Hard floor: still block if P_brain < 0.30 (clearly unprofitable template)
if P_brain < 0.30:
    return fail('gate2')  # keep hard block for garbage templates

# Otherwise: P_brain feeds into P(success) via W_BRAIN
```

**What this changes**: Templates with 30-50% WR are no longer hard-blocked. They
contribute a weak P_brain signal. Only sub-30% templates are rejected.

**Risk**: Marginal templates sneak through. Mitigated by P(success) threshold.
Track gate2 rejection rate in IS report to verify not flooding with bad templates.

### 4.2 Gate 3 — Conviction (line 852)

**Current**: Hard block if `conviction < 0.48` (MIN_CONVICTION).

**Soft replacement**: Continuous weight, lower soft floor.
```python
# Replace binary is_confident with continuous contribution
# conviction already in [0, 1] — use directly as P_conviction in formula
# Soft floor: trades with conviction < 0.30 are blocked (noise territory)
CONVICTION_SOFT_FLOOR = 0.30  # config field: p_success_conviction_floor

if conviction < CONVICTION_SOFT_FLOOR:
    return fail('gate3')  # hard block below noise floor

# Above floor: conviction feeds directly into P(success) via W_CONVICTION
# No threshold needed — low conviction = low P(success) = natural rejection
```

**What this changes**: Trades with conviction 0.30-0.48 are no longer hard-blocked.
A trade with conviction=0.35 but strong wave function + low entropy CAN pass.

**Risk**: Low-conviction trades are often wrong-direction. Mitigated by momentum
gate still penalizing misaligned entries. Monitor conviction-bucketed WR.

### 4.3 Gate 4 — Momentum Alignment (line 876)

**Current**: Hard block if `sign(F_momentum) != sign(trade_direction)`.

**Soft replacement**: Sigmoid penalty multiplier.
```python
# Replace binary sign check with sigmoid
F_mom = getattr(state, 'F_momentum', 0.0)
side_sign = 1.0 if side == 'long' else -1.0
P_momentum = _sigmoid(F_mom * side_sign)  # [0, 1]

# Penalty: misaligned momentum reduces P(success) but doesn't hard-block
# sigmoid(-0.5) = 0.076 — effectively blocked by P(success) threshold
# sigmoid(+0.5) = 0.924 — nearly full contribution
# Near-zero momentum: sigmoid(0) = 0.5 — neutral

# No hard block needed — sigmoid naturally penalizes misalignment
# But keep 0.6x multiplier as safety net for strong misalignment
if P_momentum < 0.20:
    P_success *= 0.60  # additional penalty for clearly wrong direction
```

**What this changes**: Trades with weak momentum misalignment (F_momentum slightly
negative vs direction) are no longer hard-blocked. Strong misalignment still
effectively blocked by sigmoid driving P_momentum near zero.

**Risk**: This is the gate with the strongest empirical signal (WR drops from 88%
to 45% on misalignment). Keep the 0.60x penalty as insurance. If OOS WR drops,
revert to hard block first.

### 4.4 Gates That Stay Hard

| Gate | Reason |
|------|--------|
| Gate 1 (template distance < 3.0) | Physical meaning — beyond 3.0 is a different pattern entirely |
| Gate 0.5 (depth filter/blacklist) | Structural — depth is discrete, not probabilistic |
| Gate 0 (regime compatibility) | Keep hard — wrong regime = guaranteed loss |
| Gate 0 (session filter) | Keep hard — overnight noise is structural |
| Gate 2.5 (TF confluence) | Keep hard — multi-TF disagreement is regime signal |
| FDMI fakeout block | Keep hard — State A is a binary regime condition |

---

## 5. Validation Criteria

### 5.1 IS Forward Pass

| Metric | Baseline (V7.0.0) | Acceptable Range | Failure Threshold |
|--------|-------------------|------------------|-------------------|
| Win Rate | 96.6% | 91.6% - 100% (within 5%) | < 91.6% |
| PnL | $39,736 | > $30,000 | < $25,000 |
| $/Trade | $12.05 | > $9.00 | < $7.00 |
| Trade Count | 3,298 | 2,000 - 4,000 | < 1,500 or > 5,000 |
| Max DD | — | < $150 | > $200 |

### 5.2 OOS Forward Pass

| Metric | Baseline | Acceptable Range | Notes |
|--------|----------|------------------|-------|
| Win Rate | 100% (compressed) | > 95% | Coherence validated +15% gap — expect improvement |
| PnL | $8,200 | > $6,000 | |
| $/Trade | $12.75 | > $10.00 | |

### 5.3 Key Diagnostics

- **P(success) distribution**: histogram of P(success) for winning vs losing trades.
  Expect clear separation (winners cluster > 0.65, losers < 0.55).
- **Component contribution**: per-component breakdown showing which weights drive
  winning trades. If W_WAVE and W_TUNNEL dominate, the quantum reconnection is working.
- **Gate migration report**: for each softened gate, count how many previously-blocked
  trades now pass AND their WR. If new-pass trades lose at > 60%, the soft gate is
  too permissive.
- **Hurst-bucketed WR**: trades with hurst < 0.40 that now pass — must have WR > 80%.
- **Conviction-bucketed WR**: trades with conviction 0.30-0.48 — must have WR > 85%.

---

## 6. Implementation Steps

### Step 1: Log quantum fields in signal_log (OBSERVATION ONLY)

**Files**: `training/forward_pass.py` (or wherever signal_log CSV is written)

Add columns to signal_log CSV:
- `P_at_center`, `P_near_upper`, `P_near_lower`
- `entropy`, `entropy_normalized`
- `reversion_probability`
- `coherence` (= `oscillation_entropy_normalized`)
- `breakout_probability`, `reversion_potential`

Source: `cand.state.<field>` — all fields already on `MarketState`.

**Validation**: Run IS forward pass, verify columns populated, spot-check ranges.
No scoring changes. No risk.

### Step 2: Add coherence as soft multiplier in score competition

**File**: `core/execution_engine.py`, method `_gate_check()` (line 817-827)

```python
# After score computation, apply coherence multiplier
coherence = getattr(state, 'oscillation_entropy_normalized', 0.5)
score *= (1.0 + 0.5 * (1.0 - coherence))  # low coherence = penalty (higher score = worse)
```

**Validation**: IS forward pass. Expect trade count to drop slightly (low-coherence
trades lose score competition). WR should stay flat or improve.

### Step 3: Convert conviction from hard threshold to continuous weight

**File**: `core/execution_engine.py`, method `_finalize_entry()` (lines 852-865)

Replace:
```python
if not getattr(_belief, 'is_confident', True):
    return HOLD  # gate3
```

With:
```python
if _belief.conviction < CONVICTION_SOFT_FLOOR:  # 0.30
    return HOLD  # gate3 — below noise floor
# Otherwise: conviction contributes to P(success) in Step 4
```

**File**: `core/timeframe_belief_network.py` — add `CONVICTION_SOFT_FLOOR = 0.30`
as class constant alongside existing `MIN_CONVICTION = 0.48`.

**Validation**: IS forward pass. Expect 5-15% more trades passing gate3. Monitor
WR of new trades (conviction 0.30-0.48 bucket).

### Step 4: Wire P(success) formula into _gate_check

**File**: `core/execution_engine.py`

Add method `_compute_p_success(self, state, conviction, dir_prob_spread, dist, side)`.
Add 8 weight fields + 2 threshold fields to `TradingConfig`.

Integration point: after all hard gates pass and score is computed (line 817),
compute `P(success)` and use as score:
```python
p_success = self._compute_p_success(state, conviction, brain_spread, dist, side)
if p_success < self.config.p_success_min:
    return fail('p_success_low')
score = -p_success  # higher P = better score (lower is better in competition)
```

Note: conviction and side are not available in `_gate_check` — they come from
`_finalize_entry`. This means P(success) must be computed in `_finalize_entry`
as a post-direction gate, or the architecture must change to compute direction
before gating. **Recommend**: compute partial P(success) in `_gate_check` (without
conviction/momentum), compute full P(success) in `_finalize_entry` as final gate.

**Validation**: IS + OOS forward pass. Compare against baseline. This is the
critical step — run both and check all metrics in Section 5.

### Step 5: A/B test quantum scoring vs current scoring

**File**: `core/execution_engine.py`

Add config flag `quantum_scoring_enabled: bool = False` to `TradingConfig`.

When enabled: use `_compute_p_success()` for scoring and soft gates.
When disabled: use current hard gates + distance-based scoring.

Run both IS + OOS with each mode. Compare:
1. WR, PnL, $/trade, trade count
2. P(success) distribution overlap
3. Which trades differ between modes (the "migration set")
4. WR of migration set specifically

**Decision criteria**: Ship quantum scoring if OOS WR >= baseline AND migration
set WR > 70%. Otherwise, tune weights and re-test.

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `core/execution_engine.py` | `_compute_p_success()`, soft gate logic, config reads |
| `core/timeframe_belief_network.py` | `CONVICTION_SOFT_FLOOR` constant |
| `core/config.py` (or wherever TradingConfig lives) | 8 weight fields, 2 thresholds, 1 flag |
| `training/forward_pass.py` | Signal log columns for quantum fields |

## 8. Rollback Plan

Every change is behind `quantum_scoring_enabled` flag (Step 5). If anything
regresses, set flag to `False` and the system reverts to current hard gates
with zero code changes.

Phase 1 does not delete any gate logic. Hard gates remain in code, controlled
by the flag. Full gate removal is Phase 5 (separate spec).
