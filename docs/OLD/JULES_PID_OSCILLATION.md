# JULES TASK: Sub-Minute PID Oscillation — Correct Architecture

## Visual Reference
The user provided a /MNQ 5s chart with Standard Error Bands at 1σ/2σ/3σ/4σ.
These ARE the quantum field sigma levels (center = L1_STABLE, 1σ ≈ inner Roche, 2σ ≈ outer Roche).
The chart shows price contained in a ~9-point range over 13 minutes, bouncing rhythmically
between the 1σ and 2σ inner bands while DMI stays low (~20–26).
Each bounce off a band is a micro ROCHE_SNAP but price NEVER escapes — it just oscillates.
THIS is PID oscillation: the Standard Error Band channel is the physical container.

## Architecture (IMPORTANT — different from naive first draft)

The user's direction has three parts:
1. **"Computed as a component of the cluster"** — `term_pid` and `oscillation_coherence` are
   engine state fields that get included in the cluster feature vector (making it 16D).
   Clusters will naturally learn to separate PID-regime patterns from non-PID patterns
   without being told to — the geometry does it.

2. **"Filtered out as a pattern"** — there is NO `PID_OSCILLATION` pattern type.
   The fractal discovery agent continues emitting only `ROCHE_SNAP` and `STRUCTURAL_DRIVE`
   events. PID metrics are carried in the state attached to those events, not as a new event type.

3. **"Computed and deciphered in its own way"** — a dedicated `PIDOscillationAnalyzer` class
   (new file: `training/pid_oscillation_analyzer.py`) watches the raw 15s state stream,
   detects the PID regime, and produces its own `PIDSignal` objects with their own
   entry/exit logic. This runs **in parallel to** the fractal clustering pipeline,
   not through it.

## Files to Modify / Create

---

### 1. `core/quantum_field_engine.py` — Compute term_pid + oscillation_coherence

In `batch_compute_states()`, after the main physics arrays are ready (z_scores, velocity, etc.)
and **before** the results loop, add:

```python
# ─── PID Control Force ──────────────────────────────────────────────────────
# Models the algorithmic market-maker control force acting on price at each bar.
# P = proportional to current deviation from equilibrium (z_score)
# I = integral of accumulated deviation (cumulative bias)
# D = derivative = rate of change of z_score (dampening)
pid_kp = params.get('pid_kp', DEFAULT_PID_KP)   # 0.5
pid_ki = params.get('pid_ki', DEFAULT_PID_KI)   # 0.1
pid_kd = params.get('pid_kd', DEFAULT_PID_KD)   # 0.2

pid_p       = pid_kp * z_scores
pid_i       = pid_ki * np.clip(np.cumsum(z_scores), -10.0, 10.0)
pid_d       = np.zeros_like(z_scores)
pid_d[1:]   = pid_kd * np.diff(z_scores)
term_pid_arr = pid_p + pid_i + pid_d   # shape: (n,)

# ─── Oscillation Coherence ──────────────────────────────────────────────────
# Rolling std of z_score over a short window.  Low std = tight periodic
# oscillation (PID regime).  High std = chaotic / trending.
# Inverted and normalised to (0, 1] so 1 = perfectly tight oscillation.
_ow = min(5, rp)
osc_std = np.full(n, np.nan)
for _i in range(_ow - 1, n):
    osc_std[_i] = np.std(z_scores[_i - _ow + 1 : _i + 1])
osc_std[:_ow - 1] = osc_std[_ow - 1]
oscillation_coherence_arr = 1.0 / (1.0 + osc_std)   # (0, 1]
```

In the results loop, replace `term_pid=0.0` with:
```python
term_pid             = float(term_pid_arr[i]),
oscillation_coherence= float(oscillation_coherence_arr[i]),
```

---

### 2. `core/three_body_state.py` — Add oscillation_coherence field

In the `# ═══ NIGHTMARE FIELD EQUATION COMPONENTS ═══` section, after `term_pid`:

```python
oscillation_coherence: float = 0.0   # 1=tight periodic PID oscillation, 0=noisy/trending
```

Also add to `__hash__` and `__eq__` as appropriate (follow existing float field pattern).
Also add to the default constructor call near line 324.

---

### 3. `training/fractal_clustering.py` — Expand feature vector to 16D

In `extract_features()`, after reading `self_adx`, `self_hurst`, `self_dmi_diff` from state,
also read:

```python
self_pid       = getattr(state, 'term_pid', 0.0)
self_osc_coh   = getattr(state, 'oscillation_coherence', 0.0)
```

Add them as the last two dimensions in the return list (dims 15 and 16):

```python
return [abs(z), v_feat, m_feat, c, tf_scale, depth, parent_ctx,
        self_adx, self_hurst, self_dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
        self_pid, self_osc_coh]   # NEW: 15=PID force, 16=oscillation coherence
```

Update the docstring from "14D" to "16D".

**NOTE**: Because the feature dimensions changed, all existing checkpoint data
(pattern_library, scaler) must be rebuilt. Run with `--fresh` after this change.

---

### 4. `training/pid_oscillation_analyzer.py` — NEW dedicated analyzer

This class watches the raw 15s state stream and deciphers the PID regime independently
of the fractal clustering pipeline.

```python
"""
PID Oscillation Analyzer — Sub-Minute Band Flip Detector

Watches the 15s quantum state stream. When the market enters a PID-controlled
oscillation regime (price bouncing between Standard Error Bands with low DMI),
identifies the band-touch flip points and emits PIDSignal objects.

Trade logic (Hawaiian Surfer at sub-minute scale):
  - Enter at band touch (price at 1σ or 2σ, oscillation confirmed)
  - Direction: toward center (L1_STABLE) if entering at outer band,
               toward outer band if entering after center cross
  - Exit: at opposite band or at center (depending on entry zone)
  - Stop: outside the band that was touched at entry
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

# Regime detection thresholds
PID_MIN_FORCE       = 0.3    # |term_pid| must exceed this
PID_MIN_OSC_COH     = 0.5    # oscillation_coherence threshold
PID_MAX_Z_ENTER     = 2.0    # don't enter if z >= 2.0 (nightmare field)
PID_MIN_BASE_COH    = 0.4    # base quantum coherence minimum
PID_MAX_ADX         = 30.0   # DMI low = PID regime (visual shows 20-26)
PID_MIN_REGIME_BARS = 3      # must see N consecutive PID bars before entering

# ── TENSION classification thresholds ───────────────────────────────────────
# A PID signal is classified TENSION (dangerous-but-profitable) when any of:
#   1. z_score near outer Roche (>= 1.5σ) — PID fighting possible breakout
#   2. term_pid very large (>= 1.0) — control force maxed out, system under strain
#   3. escape_probability elevated (>= 0.25) — quantum field says breakout is real
#   4. oscillation_coherence falling while regime persists — control degrading
# TENSION signals are logged in shadow but flagged separately.
# They are NEVER enabled for live trading until a dedicated analysis sprint.
PID_TENSION_Z_MIN        = 1.5    # z >= this → approaching outer Roche → TENSION
PID_TENSION_FORCE_MAX    = 1.0    # |term_pid| >= this → maxed-out control → TENSION
PID_TENSION_ESCAPE_MIN   = 0.25   # escape_probability >= this → TENSION
PID_TENSION_COH_DROP     = 0.15   # osc_coh dropped >= this vs 3-bar avg → TENSION


@dataclass
class PIDSignal:
    timestamp:     object       # bar timestamp
    direction:     str          # 'LONG' | 'SHORT'
    entry_price:   float
    target_price:  float        # center band or opposite sigma band
    stop_price:    float        # outside the touched band
    z_score:       float        # entry z_score
    band_touched:  str          # '1sig' | '2sig'
    regime_bars:   int          # how many consecutive PID bars before this signal
    osc_coherence: float        # oscillation_coherence at entry
    term_pid:      float        # PID control force at entry
    pid_class:     str          # 'STABLE' | 'TENSION'
    tension_reason: str         # '' | 'outer_roche' | 'maxed_force' | 'escape_risk' | 'coh_drop'


class PIDOscillationAnalyzer:
    def __init__(self, sigma_per_bar: float = None):
        """
        sigma_per_bar: the current day's sigma (from engine center mass computation).
        Updated per-day via reset().
        """
        self._sigma      = sigma_per_bar or 1.0
        self._regime_n   = 0     # consecutive PID bars seen
        self._signals    = []

    def reset(self, sigma: float):
        """Call at start of each day with the day's regression sigma."""
        self._sigma    = sigma
        self._regime_n = 0
        self._signals  = []

    def tick(self, state) -> Optional[PIDSignal]:
        """
        Feed one ThreeBodyQuantumState. Returns a PIDSignal if an entry is triggered,
        else None. Call once per 15s bar during the forward pass.
        Signal is classified as STABLE or TENSION for separate shadow analysis.
        """
        force    = abs(getattr(state, 'term_pid', 0.0))
        osc_coh  = getattr(state, 'oscillation_coherence', 0.0)
        base_coh = getattr(state, 'coherence', 0.0)
        adx      = getattr(state, 'adx_strength', 100.0)
        z        = state.z_score
        escape   = getattr(state, 'escape_probability', 0.0)

        # Check if this bar is in PID regime
        in_pid = (force    >= PID_MIN_FORCE
              and osc_coh  >= PID_MIN_OSC_COH
              and base_coh >= PID_MIN_BASE_COH
              and adx      <= PID_MAX_ADX
              and abs(z)   < PID_MAX_Z_ENTER)

        if in_pid:
            self._osc_coh_history.append(osc_coh)
            if len(self._osc_coh_history) > 3:
                self._osc_coh_history.pop(0)
            self._regime_n += 1
        else:
            self._regime_n = 0
            self._osc_coh_history.clear()
            return None

        if self._regime_n < PID_MIN_REGIME_BARS:
            return None   # not enough consecutive PID bars yet

        # Identify band touch and direction
        sigma     = self._sigma
        price     = state.particle_position
        center    = price - z * sigma   # L1_STABLE approximation

        if z <= -1.0:
            direction    = 'LONG'
            target_price = center
            stop_price   = price - 0.5 * sigma
            band_touched = '1sig' if abs(z) < 2.0 else '2sig'
        elif z >= 1.0:
            direction    = 'SHORT'
            target_price = center
            stop_price   = price + 0.5 * sigma
            band_touched = '1sig' if abs(z) < 2.0 else '2sig'
        else:
            return None   # near center, no directional edge

        # ── TENSION classification ───────────────────────────────────────────
        # Dangerous-but-profitable: high reward but misfire risk is large.
        # These are logged separately and NEVER enabled for live trading
        # until a dedicated analysis sprint.
        tension_reason = ''
        if abs(z) >= PID_TENSION_Z_MIN:
            tension_reason = 'outer_roche'      # approaching outer Roche limit
        elif force >= PID_TENSION_FORCE_MAX:
            tension_reason = 'maxed_force'      # PID control maxed out
        elif escape >= PID_TENSION_ESCAPE_MIN:
            tension_reason = 'escape_risk'      # quantum field says breakout is real
        elif (len(self._osc_coh_history) >= 3
              and (self._osc_coh_history[0] - osc_coh) >= PID_TENSION_COH_DROP):
            tension_reason = 'coh_drop'         # coherence degrading mid-regime

        pid_class = 'TENSION' if tension_reason else 'STABLE'

        return PIDSignal(
            timestamp     = state.timestamp,
            direction     = direction,
            entry_price   = price,
            target_price  = target_price,
            stop_price    = stop_price,
            z_score       = z,
            band_touched  = band_touched,
            regime_bars   = self._regime_n,
            osc_coherence = osc_coh,
            term_pid      = getattr(state, 'term_pid', 0.0),
            pid_class     = pid_class,
            tension_reason= tension_reason,
        )

    # Also add to __init__:
    # self._osc_coh_history = []   # rolling 3-bar osc_coh for TENSION coh_drop check

    @property
    def signals(self) -> List[PIDSignal]:
        return list(self._signals)
```

---

### 5. `training/orchestrator.py` — Wire PIDOscillationAnalyzer into forward pass

**Import** at top of file:
```python
from training.pid_oscillation_analyzer import PIDOscillationAnalyzer, PIDSignal
```

**Instantiate** before the day loop:
```python
pid_analyzer = PIDOscillationAnalyzer()
```

**Reset** at start of each day (after sigma is known from day states):
```python
day_sigma = np.mean([s['state'].sigma_fractal for s in day_states if s['state'].sigma_fractal > 0]) or 1.0
pid_analyzer.reset(sigma=day_sigma)
```

**Tick** once per 15s bar alongside the main candidate evaluation:
```python
pid_signal = pid_analyzer.tick(state=current_state, oracle_marker=current_oracle_marker)
# PID signals are SHADOW MODE ONLY — never passed to wave rider.
# Sub-minute profit/risk is unfavorable: tight ticks mean one misfire
# erodes many wins.  Log the signal and let the oracle tell us what
# WOULD have happened.  Enable live trading only after data analysis
# confirms the regime is robust.
```

**No position is ever opened from a PID signal.**
All PID analysis is write-to-CSV only.

**Logging**: Separate CSV `checkpoints/pid_oracle_log.csv`, columns:
```
timestamp, direction, entry_price, target_price, stop_price, z_score,
band_touched, regime_bars, osc_coherence, term_pid,
pid_class,       # 'STABLE' | 'TENSION'
tension_reason,  # '' | 'outer_roche' | 'maxed_force' | 'escape_risk' | 'coh_drop'
oracle_label, oracle_label_name, oracle_mfe, oracle_mae,
would_have_hit_target,   # bool: did price reach target_price before stop_price?
would_have_hit_stop,     # bool: did price reach stop_price first?
theoretical_pnl          # dollars if the trade had been taken (point_value applied)
```

The `would_have_hit_*` fields are filled per-signal in the orchestrator using a
bar-by-bar scan of the remaining day data (same principle as `_audit_trade`):

```python
def _audit_pid_signal(sig: PIDSignal, day_bars: pd.DataFrame,
                      bar_idx: int, point_value: float) -> dict:
    """
    Scan forward from bar_idx to see if price hit target or stop first.
    Uses high/low of each subsequent bar to check touch.
    Lookahead cap: 40 bars (10 minutes at 15s) — PID oscillations are fast.
    """
    lookahead  = min(40, len(day_bars) - bar_idx - 1)
    hit_target = False
    hit_stop   = False
    for i in range(bar_idx + 1, bar_idx + 1 + lookahead):
        hi = day_bars.iloc[i]['high']
        lo = day_bars.iloc[i]['low']
        if sig.direction == 'LONG':
            if lo <= sig.stop_price:
                hit_stop = True; break
            if hi >= sig.target_price:
                hit_target = True; break
        else:  # SHORT
            if hi >= sig.stop_price:
                hit_stop = True; break
            if lo <= sig.target_price:
                hit_target = True; break

    if hit_target:
        theo_pnl = abs(sig.target_price - sig.entry_price) * point_value
    elif hit_stop:
        theo_pnl = -abs(sig.stop_price - sig.entry_price) * point_value
    else:
        # Neither hit within 10 min — use last bar's close vs entry
        last_close = day_bars.iloc[min(bar_idx + lookahead, len(day_bars)-1)]['close']
        diff = (last_close - sig.entry_price) if sig.direction == 'LONG' \
               else (sig.entry_price - last_close)
        theo_pnl = diff * point_value

    return {
        'would_have_hit_target': hit_target,
        'would_have_hit_stop':   hit_stop,
        'theoretical_pnl':       round(theo_pnl, 2),
    }
```

This is called in the orchestrator's bar loop immediately after `pid_analyzer.tick()` returns
a signal, using the already-loaded `day_df` (15s bars for that day) and `bar_idx`.

---

## What Changes in the Cluster (Feature Impact)

The 2 new dimensions (`self_pid`, `self_osc_coh`) let the clustering naturally separate:
- **PID regime ROCHE_SNAPs** (high osc_coh, significant term_pid, low ADX)
  → these will cluster into templates with predictable low-sigma, high win-rate characteristics
- **Non-PID ROCHE_SNAPs** (chaotic, trending, large term_pid swing)
  → separate cluster region; higher variance, lower win-rate

This means `_exception_tids` will naturally start to contain PID-regime templates
(because they have low `regression_sigma_ticks`), and the existing data-override in Gate 0
will admit PID-regime fractal patterns without special-casing.

## FN Accounting — Critical Separation

PID-regime bars must NOT appear in the fractal FN oracle log.
Currently every bar with an oracle-positive pattern that was not traded becomes an FN record.
If a bar is in PID regime and the fractal pipeline skips its pattern (Gate 0 Rule 3 snap,
noise zone, etc.), it would incorrectly inflate Gate 0's FN count — but the fractal rules
were correct to skip it; the PID analyzer is responsible for that bar, not the fractal pipeline.

Implementation:

**In `orchestrator.py`, during FN audit** (both the "competed" and "no_match" loops):
```python
# Skip bars already handled by the PID analyzer
_state = getattr(p, 'state', None)
_is_pid_bar = (
    _state is not None
    and abs(getattr(_state, 'term_pid', 0.0)) >= PID_MIN_FORCE_AUDIT
    and getattr(_state, 'oscillation_coherence', 0.0) >= PID_MIN_OSC_COH_AUDIT
    and getattr(_state, 'adx_strength', 100.0) <= PID_MAX_ADX_AUDIT
)
if _is_pid_bar:
    pid_fn_count += 1   # counted separately
    continue            # do NOT add to fn_oracle_records
```

Where `PID_MIN_FORCE_AUDIT = 0.3`, `PID_MIN_OSC_COH_AUDIT = 0.5`, `PID_MAX_ADX_AUDIT = 30.0`
match the analyzer's detection thresholds.

The PID analyzer keeps its own oracle audit (separate CSV: `checkpoints/pid_oracle_log.csv`)
that shows:
- How many PID signals fired
- Win rate and avg PnL of PID trades
- How many PID-regime bars had oracle-positive labels but no PID signal triggered
  (these are the PID-specific FNs, separate from the fractal FNs)

**Report section additions** in `phase4_report.txt`:
```
  PID OSCILLATION SUMMARY  (shadow mode — no live trades)
    Bars in PID regime:          NNN
    Signals fired:               NNN  (excluded from fractal FN count)
      STABLE signals:            NNN  (safe oscillation)
      TENSION signals:           NNN  (dangerous-but-profitable)
        outer_roche:             NNN  (z >= 1.5, near outer Roche)
        maxed_force:             NNN  (|term_pid| >= 1.0)
        escape_risk:             NNN  (escape_probability >= 0.25)
        coh_drop:                NNN  (coherence degrading mid-regime)

    STABLE  theoretical WR: NN.N%   avg_pnl: $NNN   max_streak_loss: N
    TENSION theoretical WR: NN.N%   avg_pnl: $NNN   max_streak_loss: N
      (TENSION: high reward but variance -- study before ever enabling)
```

## What is NOT changed
- `fractal_discovery_agent.py` — no new pattern type, no new code
- Gate 0 rules — unaffected (PID signals bypass the gate entirely)
- Hawaiian Surfer (macro) — separate sprint
- DOE parameter space — `pid_kp/ki/kd` already tunable

## Shadow Mode — What to Study from pid_oracle_log.csv

PID signals are logged but NEVER traded until the data justifies it.
After first `--fresh` run, analyse the shadow log for:

1. **Win rate by band touched**: Are 2σ touches more reliable than 1σ?
   (expect 2σ to be cleaner — closer to outer Roche)

2. **Win rate by regime_bars**: Does longer confirmed regime → better edge?
   (more regime_bars = more established oscillation = higher predictability)

3. **Win rate by time of day**: Pre-market vs open vs lunch vs close?
   (PID regimes may dominate at lunch and overnight when volume is thin)

4. **Theoretical avg_pnl vs std(pnl)**: Sharpe of PID signals.
   If std(pnl) >> avg_pnl → not ready to trade (one misfire = many wins erased).

5. **Consecutive loss streaks**: Maximum drawdown from a run of stop-outs.
   This is the misfire erosion problem — only enable trading if max streak < 3.

**STABLE signals** — enable live trading when:
- Win rate ≥ 58%
- Theoretical Sharpe > 1.5
- Max consecutive losses ≤ 3 in any single day
- At least 300 STABLE shadow signals analysed

**TENSION signals** — separate go-live decision, MUCH higher bar:
- Win rate ≥ 65%  (higher reward must justify the variance)
- Theoretical Sharpe > 2.0
- Max consecutive losses ≤ 2 in any single day (tight — one streak can erase the day)
- At least 300 TENSION shadow signals AND a dedicated analysis sprint to understand
  WHICH tension_reason is worth trading and which is permanently too dangerous
- Each tension_reason goes live independently (e.g. `outer_roche` may be fine,
  `coh_drop` may never be worth trading)

Until both thresholds are independently cleared: shadow mode only for both classes.

## Run Order After Implementation
1. `--fresh` (rebuilds scaler + pattern_library with 16D features)
2. Examine `checkpoints/pid_oracle_log.csv`
3. Check FN report — PID-regime bars excluded from fractal Gate 0 count
4. Tune `PID_MIN_REGIME_BARS`, `PID_MIN_OSC_COH`, `PID_MAX_ADX` if shadow WR is low
5. Only wire live trading after shadow analysis passes all 5 criteria above
