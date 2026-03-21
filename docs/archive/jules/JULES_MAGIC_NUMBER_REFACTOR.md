# Magic Number Refactor — Derived Parameters from Base Metrics

## Principle
Every numeric threshold must either:
1. **Derive from data** (ATR, template stats, tick_size, volatility)
2. **Live in a single config class** with a name and docstring
3. **Be a structural constant** (TF definitions, maintenance times, numerical stability)

No inline literals in logic. If a number controls behavior, it has a name and a source.

## Phase 1: Config Class (`core/trading_config.py`)

Create a single `TradingConfig` dataclass that holds ALL tunable parameters.
Instantiated once, passed to ExitEngine, ExecutionEngine, and TBN.

```python
@dataclass
class TradingConfig:
    """All tunable parameters in one place. No magic numbers in logic."""

    # === Base metrics (from market/asset) ===
    tick_size: float = 0.25
    point_value: float = 2.0

    # === Derived at init from ATR/volatility (set by caller) ===
    atr_ticks: float = 0.0          # current ATR in ticks — set before each run
    noise_floor_ticks: float = 0.0  # median swing noise — set from data

    # === Exit: Giveback ===
    giveback_pct: float = 0.10                # fraction of peak to tolerate giving back
    giveback_min_mfe_mult: float = 1.0        # multiplier on noise_floor → min MFE to arm
    giveback_aggressive_mult: float = 2.0     # peak >= N× noise → aggressive tier
    giveback_aggressive_pct: float = 0.40     # aggressive tier threshold
    giveback_slow_flip_reduction: float = 0.15  # tighten by this when 30m flips
    giveback_slow_flip_floor: float = 0.25    # floor after slow flip tightening

    # === Exit: Envelope ===
    envelope_halflife_bars: float = 40.0
    envelope_floor_pct: float = 0.15
    envelope_min_bars: int = 8
    envelope_force_boost: float = 1.3   # aligned net_force → boost decay
    envelope_force_penalty: float = 0.7 # adverse net_force → shrink decay
    envelope_early_suppress_pct: float = 0.5  # suppress if < N% of halflife
    envelope_floor_trigger_pct: float = 0.3   # exit if unrealized < N% of level

    # === Exit: Self-tuning ===
    tune_window: int = 30
    tune_hl_min: float = 8.0
    tune_hl_max: float = 30.0
    tune_gb_min: float = 0.10
    tune_gb_max: float = 0.90
    tune_too_early_capture: float = 0.20   # capture < N% = too early
    tune_too_early_mfe: float = 8.0        # peak < N ticks = too early
    tune_too_late_giveback: float = 0.50   # gave back > N% = too late
    tune_growth_rate: float = 1.10         # halflife grows by 10% per cycle
    tune_shrink_step: float = 0.05         # giveback shrinks by 5pp per cycle
    tune_revert_rate: float = 0.10         # 10% step toward baseline

    # === Exit: Breakeven ===
    be_activation_ticks: float = 4.0    # TODO: derive from noise_floor
    be_buffer_ticks: float = 1.0

    # === Exit: Watchdog ===
    watchdog_tick_threshold: float = 8.0  # TODO: derive from ATR
    watchdog_bar_threshold: int = 5
    watchdog_worker_threshold: int = 5
    watchdog_mfe_floor_ticks: float = 2.0
    watchdog_mfe_progress_pct: float = 0.5

    # === Exit: Timescale ===
    timescale_tighten_mult: float = 1.5   # tighten at N× avg_mfe_bar
    timescale_urgent_mult: float = 2.5    # urgent at N× p75_mfe_bar
    max_hold_parent_bars: int = 5         # N parent TF bars → max hold
    max_hold_min_bars: int = 20           # absolute floor

    # === TBN: Physics blend ===
    physics_accel_weight: float = 0.5     # acceleration weight in momentum signal
    physics_ml_blend: float = 0.5         # weight of physics vs ML in dir_prob

    # === TBN: Wave maturity composite ===
    wave_pattern_weight: float = 0.4
    wave_band_weight: float = 0.3
    wave_reversion_weight: float = 0.3
    wave_band_sigma_max: float = 3.0      # z-score at which band = fully mature

    # === TBN: Conviction & path ===
    min_conviction: float = 0.48
    min_active_levels: int = 3
    conviction_scale: float = 2.0         # abs(dir_prob - 0.5) × N

    # === TBN: Band confluence ===
    band_min_active: int = 3
    band_majority_mult: float = 2.0       # support > resistance × N
    band_trigger_threshold: float = 0.5
    band_normalize_sigma: float = 3.0
    band_blend_threshold: float = 0.3     # blend only when strength > N
    band_blend_weight: float = 0.4        # max influence on direction

    # === TBN: Pace blend ===
    pace_max_weight: float = 0.30         # max influence on direction
    pace_weight_scale: float = 0.15       # scale: |blend| × N
    pace_ahead_long: float = 0.80         # target P(dir) when ahead
    pace_ahead_short: float = 0.20
    pace_behind_long: float = 0.35        # target P(dir) when behind
    pace_behind_short: float = 0.65
    pace_tighten_threshold: float = 0.3   # pace < N → behind
    pace_widen_threshold: float = 1.5     # pace > N → ahead
    pace_time_gate: float = 0.3           # only active after N% time elapsed

    # === TBN: Trade health fusion ===
    health_pace_weight: float = 0.6
    health_decay_weight: float = 0.4

    # === TBN: Exit signal thresholds ===
    tighten_wave_maturity: float = 0.85
    widen_wave_maturity: float = 0.30
    slow_flip_long_threshold: float = 0.45   # 30m dir_prob below → flip
    slow_flip_short_threshold: float = 0.55  # 30m dir_prob above → flip
    band_exit_strength_min: float = 0.5      # band signal strength gate

    # === TBN: Decay cascade ===
    decay_alpha_max: float = 3.0
    decay_alpha_min: float = 1.0
    decay_theta_exit: float = 1.5
    decay_progress_cap: float = 2.0

    # === Execution: Pattern quality ===
    noise_z_threshold: float = 0.5        # below = noise zone
    approach_z_threshold: float = 2.0     # above = extreme zone
    headroom_z_max: float = 3.0           # parent z < N = has headroom
    nightmare_z: float = 3.0              # micro z > N + no headroom = skip

    # === Execution: Gate thresholds ===
    gate1_dist: float = 4.5              # template match distance
    hurst_min: float = 0.5
    reversion_prob_min: float = 0.40
    momentum_override_ratio: float = 1.0
    brain_min_prob: float = 0.05
    adx_trend_confirmation: float = 25.0
    hurst_trend_confirmation: float = 0.6

    # === Execution: Direction cascade ===
    bias_threshold: float = 0.55
    dmi_threshold: float = 0.0
    worker_bypass_conviction: float = 0.65
    momentum_accel_weight: float = 0.5    # F_net weight in momentum signal
    momentum_trigger: float = 0.5         # threshold to trigger momentum dir
    momentum_conviction_cap: float = 0.15
    momentum_conviction_coeff: float = 0.05
    signed_mfe_conviction_cap: float = 0.30
    signed_mfe_conviction_coeff: float = 0.10
    brain_winrate_margin: float = 0.10    # WR difference to trigger brain dir
    template_bias_min_sum: float = 0.10
    band_cascade_conviction: float = 0.55
    dmi_long_prob: float = 0.55
    dmi_short_prob: float = 0.45
    velocity_long_prob: float = 0.52
    velocity_short_prob: float = 0.48

    # === Execution: SL/TP sizing multipliers ===
    sl_p25_mae_mult: float = 3.0
    sl_mean_mae_mult: float = 2.0
    sl_default_ticks: float = 20.0
    sl_min_ticks: float = 4.0
    tp_min_ticks: float = 4.0
    tp_default_ticks: float = 50.0
    trail_sigma_mult: float = 1.1
    trail_mae_mult: float = 1.1
    trail_default_ticks: float = 10.0
    trail_min_ticks: float = 2.0
    trail_activation_mae_mult: float = 0.3
    atr_sl_mult: float = 3.0             # ATR floor for SL
    atr_tp_mult: float = 5.0             # ATR floor for TP
    significance_threshold: float = 2.0   # MAE/MFE/sigma must exceed to use

    # === Execution: OLS TP bounds ===
    ols_lower_pct: float = 0.20           # prediction must be > N% of anchor
    ols_upper_pct: float = 5.0            # prediction must be < N× anchor
    brain_tp_adjust_pct: float = 0.50     # max brain E[PnL] adjustment

    # === Execution: Live bias ===
    live_bias_min_trades: int = 5
    live_bias_winrate_min: float = 0.60
    live_bias_margin: float = 0.15
```

## Phase 2: Derive from Data (TODOs)

These parameters currently use fixed values but SHOULD derive from runtime data:

| Parameter | Current | Should derive from |
|-----------|---------|-------------------|
| `be_activation_ticks` | 4 | `noise_floor_ticks` (median swing noise) |
| `watchdog_tick_threshold` | 8 | `atr_ticks * 1.0` |
| `giveback_min_mfe_mult` | 16 ticks | `noise_floor_ticks * 1.0` |
| `sl_min_ticks` | 4 | `max(2, noise_floor_ticks * 0.5)` |
| `significance_threshold` | 2.0 ticks | `noise_floor_ticks * 0.25` |

For Phase 2, `TradingConfig.derive_from_data(atr, noise_floor)` method computes these.

## Phase 3: Replace Inline Literals

For each file, replace every magic number with `self.config.<param_name>`.

### execution_engine.py (~85 replacements)
- Constructor: accept `config: TradingConfig`
- All gate thresholds → `self.config.gate1_dist`, `self.config.hurst_min`, etc.
- Direction cascade probabilities → `self.config.dmi_long_prob`, etc.
- SL/TP sizing → `self.config.sl_p25_mae_mult`, etc.

### exit_engine.py (~57 replacements)
- Constructor: accept `config: TradingConfig`
- All envelope params → `self.config.envelope_*`
- Giveback → `self.config.giveback_*`
- Self-tuning → `self.config.tune_*`
- Watchdog → `self.config.watchdog_*`

### timeframe_belief_network.py (~61 replacements)
- Constructor: accept `config: TradingConfig`
- All class constants → config fields
- Physics blend → `self.config.physics_*`
- Pace blend → `self.config.pace_*`
- Wave maturity → `self.config.wave_*`
- Band confluence → `self.config.band_*`
- Decay cascade → `self.config.decay_*`

## Phase 4: Wire Config Through Pipeline

- `trainer.py`: create `TradingConfig()`, pass to EE, ExitEngine, TBN
- `live_engine.py`: same wiring
- Config can be serialized to JSON for hot-reload in live

## Execution Order

1. Create `core/trading_config.py` with the dataclass
2. Wire into `exit_engine.py` (smallest file, fewest dependencies)
3. Wire into `timeframe_belief_network.py`
4. Wire into `execution_engine.py` (most replacements)
5. Wire through `trainer.py` and `live_engine.py` constructors
6. Validate: `--forward-pass --data DATA/ATLAS_1DAY` (results must be identical)

## Structural Constants (DO NOT move to config)
- TF definitions: `TIMEFRAMES_SECONDS`, `TF_WEIGHTS`, `_TF_LABELS`
- Maintenance window: `16:45`, `18:00` ET
- Numerical stability: `1e-7`, `1e-9`, sigmoid clamp `±20`
- Rounding precision: `round(..., 3)`
- Division-by-zero guards: `max(1, ...)`, `max(0.1, ...)`
