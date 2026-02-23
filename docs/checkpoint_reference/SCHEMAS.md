# Checkpoint Object Schemas

Reference for all data structures Jules encounters when modifying `training/orchestrator.py`.
These objects are loaded from `.pkl` files (not tracked in git) but their STRUCTURE is
documented here so Jules can write correct diagnostic code.

---

## PatternTemplate (from `training/fractal_clustering.py`)

```python
@dataclass
class PatternTemplate:
    template_id: int                          # Unique cluster ID
    centroid: np.ndarray                      # 16D feature vector (raw scale)
    member_count: int                         # Patterns in cluster
    patterns: List[PatternEvent]
    physics_variance: float                   # Z-score std dev
    transition_map: Dict[int, float] = {}     # {next_tid: probability}
    transition_probs: Dict[int, float] = {}
    expected_value: float = 0.0
    outcome_variance: float = 0.0
    avg_drawdown: float = 0.0
    risk_score: float = 0.0
    risk_variance: float = 0.0
    stats_win_rate: float = 0.0
    stats_expectancy: float = 0.0
    stats_mega_rate: float = 0.0
    long_bias: float = 0.0                    # Fraction LONG oracle markers
    short_bias: float = 0.0                   # Fraction SHORT oracle markers
    parent_cluster_id: Optional[int] = None
    mean_mfe_ticks: float = 0.0               # Mean max-favorable-excursion (ticks)
    mean_mae_ticks: float = 0.0               # Mean max-adverse-excursion (ticks)
    p75_mfe_ticks: float = 0.0                # 75th-pct MFE
    p25_mae_ticks: float = 0.0                # 25th-pct MAE
    regression_sigma_ticks: float = 0.0
    mfe_coeff: Optional[List[float]] = None   # 14D OLS weights for MFE prediction
    mfe_intercept: float = 0.0
    dir_coeff: Optional[List[float]] = None   # 14D logistic weights for P(LONG)
    dir_intercept: float = 0.0
    semantic_name: str = "Unknown"            # e.g. "[15m] Roche+Shock Trend"
```

## pattern_library (Dict[int, Dict])

Each value mirrors PatternTemplate fields:
```python
pattern_library[template_id] = {
    'centroid': np.ndarray,       # 16D
    'params': Dict,               # {stop_loss_ticks, take_profit_ticks, trailing_stop_ticks}
    'member_count': int,
    'transition_map': Dict,
    'expected_value': float,
    'outcome_variance': float,
    'avg_drawdown': float,
    'risk_score': float,
    'long_bias': float,
    'short_bias': float,
    'stats_win_rate': float,
    'semantic_name': str,
    'mean_mfe_ticks': float,
    'mean_mae_ticks': float,
    'p75_mfe_ticks': float,
    'p25_mae_ticks': float,
    'risk_variance': float,
    'regression_sigma_ticks': float,
    'mfe_coeff': Optional[List[float]],
    'mfe_intercept': float,
    'dir_coeff': Optional[List[float]],
    'dir_intercept': float,
}
```

## 16D Feature Vector (exact order)

```python
features = [
    abs(z_score),                             # [0]
    np.log1p(abs(velocity)),                  # [1]
    np.log1p(abs(momentum)),                  # [2]
    coherence,                                # [3]
    np.log2(max(1, tf_seconds)),              # [4]
    float(depth),                             # [5]
    1.0 if parent_type == 'ROCHE_SNAP' else 0.0,  # [6]
    self_adx / 100.0,                         # [7]
    self_hurst,                               # [8]
    (self_dmi_plus - self_dmi_minus) / 100.0, # [9]
    abs(parent_z),                            # [10]
    (parent_dmi_plus - parent_dmi_minus) / 100.0,  # [11]
    1.0 if root_is_roche else 0.0,            # [12]
    self_dir * root_dir,                      # [13] TF alignment
    self_pid,                                 # [14]
    self_osc_coherence,                       # [15]
]
```

## clustering_scaler

`sklearn.preprocessing.StandardScaler` fitted on the 16D features.
Attributes: `.mean_` (16,), `.scale_` (16,), `.var_` (16,).

## template_tiers (Dict[int, int])

`{template_id: tier}` where tier is 1 (best) to 4 (worst).

## TradeOutcome (from `core/bayesian_brain.py`)

```python
@dataclass
class TradeOutcome:
    state: Union[str, int]         # template_id
    entry_price: float
    exit_price: float
    pnl: float
    result: str                    # 'WIN' or 'LOSS'
    timestamp: float
    exit_reason: str
    entry_time: float = 0.0
    exit_time: float = 0.0
    duration: float = 0.0
    direction: str = 'LONG'
    template_id: Optional[int] = None
```

## BayesianBrain (from `core/bayesian_brain.py`)

```python
brain.table[template_id] = {
    'wins': int, 'losses': int, 'total': int
}
# P(Win) = (wins + 1) / (total + 11)  # Beta(1,10) prior
# Confidence = min(total / 100, 1.0)

brain.should_fire(template_id, min_prob=0.05, min_conf=0.0) -> bool
brain.update(outcome: TradeOutcome) -> None
```

## Position (from `training/wave_rider.py`)

```python
@dataclass
class Position:
    entry_price: float
    entry_time: float
    side: str                      # 'long' or 'short'
    stop_loss: float               # Absolute price
    high_water_mark: float
    entry_layer_state: ThreeBodyQuantumState
    template_id: Optional[int] = None
    profit_target: Optional[float] = None
    trailing_stop_ticks: Optional[int] = None
    trail_activation_ticks: Optional[int] = None
    original_trail_ticks: Optional[int] = None
    last_adjustment_reason: Optional[str] = None
    breakeven_locked: bool = False
    breakeven_level: Optional[float] = None
    entry_dmi_inverse: bool = False
    bars_in_trade: int = 0
```

## PendingReview (from `training/wave_rider.py`)

```python
@dataclass
class PendingReview:
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    review_end_time: float
    side: str
    exit_reason: str
```

## BeliefState (from `training/timeframe_belief_network.py`)

```python
@dataclass
class BeliefState:
    direction: str                 # 'long' or 'short'
    conviction: float              # [0, 1]
    predicted_mfe: float           # ticks
    active_levels: int
    wave_maturity: float = 0.0
    decision_wave_maturity: float = 0.0
    tf_beliefs: Dict[int, WorkerBelief]
```

## WorkerBelief

```python
@dataclass
class WorkerBelief:
    tf_seconds: int
    dir_prob: float                # P(LONG) [0, 1]
    pred_mfe: float
    template_id: int
    tf_bar_idx: int
    conviction: float              # |dir_prob - 0.5| * 2
    wave_maturity: float = 0.0
    z_score: float = 0.0
    momentum: float = 0.0
```

Worker snapshot JSON (saved in oracle logs):
```json
{"1h": {"d": 0.47, "c": 0.06, "m": 0.31, "mfe": 277.9, "z": 0.11}, ...}
```

## ThreeBodyQuantumState (key fields for forward pass)

```python
state.z_score            # Normalized distance (sigma units)
state.particle_velocity  # Momentum
state.momentum_strength  # Normalized KE
state.coherence          # Wave coherence [0, 1]
state.hurst_exponent     # [0, 1]
state.adx_strength       # [0, 100]
state.dmi_plus           # [0, 100]
state.dmi_minus          # [0, 100]
state.tunnel_probability # P(revert to center)
state.escape_probability
state.lagrange_zone      # 'L1_STABLE', 'L2_ROCHE', 'L3_ROCHE'
state.pattern_type       # 'NONE', 'COMPRESSION', 'WEDGE', 'BREAKDOWN'
state.F_reversion        # Tidal force
state.F_momentum         # Kinetic energy
state.term_pid           # PID control force
state.oscillation_coherence
state.timestamp
```

---

## CSV Column Reference

### oracle_trade_log.csv
```
template_id, playbook, direction, entry_price, entry_time, entry_depth,
root_tf, max_hold_bars, oracle_label, oracle_label_name, oracle_mfe,
oracle_mae, long_bias, short_bias, dmi_diff, belief_active_levels,
belief_conviction, wave_maturity, decision_wave_maturity, entry_workers,
exit_price, exit_time, hold_bars, exit_reason, actual_pnl,
oracle_potential_pnl, capture_rate, result, exit_workers,
exit_conviction, exit_wave_maturity, exit_signal_reason, exit_decay_score
```

### fn_oracle_log.csv
```
timestamp, depth, oracle_label, oracle_label_name, oracle_dir,
fn_potential_pnl, reason, gate_blocked, workers
```

### signal_log_YYYY_QN.csv (decision matrix)
```
timestamp, date, status, pattern_type, micro_z, macro_z,
distance_to_cluster, conviction, template_id, tier, playbook,
trade_direction, trade_result, trade_pnl, exit_reason,
exit_signal_reason, exit_conviction, exit_wave_maturity
```
