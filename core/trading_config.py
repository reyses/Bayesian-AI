"""Central config for all tunable trading parameters. No magic numbers in logic."""
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """All tunable parameters in one place.

    Instantiated once per run, passed to ExitEngine, ExecutionEngine, TBN.
    """

    # === Base metrics (from market/asset) ===
    tick_size: float = 0.25
    point_value: float = 2.0

    # === Derived at init from ATR/volatility (set by caller) ===
    atr_ticks: float = 0.0
    noise_floor_ticks: float = 0.0

    # === Exit: Giveback ===
    giveback_pct: float = 0.10
    giveback_min_mfe_mult: float = 1.0
    giveback_aggressive_mult: float = 2.0
    giveback_aggressive_pct: float = 0.40
    giveback_slow_flip_reduction: float = 0.15
    giveback_slow_flip_floor: float = 0.25

    # === Exit: Envelope ===
    envelope_halflife_bars: float = 40.0
    envelope_floor_pct: float = 0.15
    envelope_min_bars: int = 8
    envelope_force_boost: float = 1.3
    envelope_force_penalty: float = 0.7
    envelope_early_suppress_pct: float = 0.5
    envelope_floor_trigger_pct: float = 0.3

    # === Exit: Self-tuning ===
    tune_window: int = 30
    tune_hl_min: float = 8.0
    tune_hl_max: float = 30.0
    tune_gb_min: float = 0.10
    tune_gb_max: float = 0.90
    tune_too_early_capture: float = 0.20
    tune_too_early_mfe: float = 8.0
    tune_too_late_giveback: float = 0.50
    tune_growth_rate: float = 1.10
    tune_shrink_step: float = 0.05
    tune_revert_rate: float = 0.10

    # === Exit: Breakeven ===
    be_activation_ticks: float = 4.0
    be_buffer_ticks: float = 1.0

    # === Exit: Watchdog ===
    watchdog_tick_threshold: float = 8.0
    watchdog_bar_threshold: int = 5
    watchdog_worker_threshold: int = 5
    watchdog_mfe_floor_ticks: float = 2.0
    watchdog_mfe_progress_pct: float = 0.5

    # === Exit: Timescale ===
    timescale_tighten_mult: float = 1.5
    timescale_urgent_mult: float = 2.5
    max_hold_parent_bars: int = 5
    max_hold_min_bars: int = 20

    # === Exit: General ===
    giveback_min_mfe_ticks: float = 16.0
    giveback_anchor_patience_pct: float = 0.3

    # === Exit: Envelope internals ===
    envelope_template_hl_divisor: float = 5.0
    envelope_template_hl_floor: float = 8.0
    envelope_peak_min_ticks: float = 4.0
    envelope_giveback_hl_floor: float = 0.5
    envelope_band_coeff: float = 0.5
    envelope_band_mult_min: float = 0.5
    envelope_band_mult_max: float = 1.5
    envelope_anchor_patience_max: float = 2.0
    envelope_hl_mult_floor: float = 0.3

    # === Exit: Band urgent ===
    band_urgent_min_strength: float = 0.6
    band_urgent_trigger_strength: float = 0.7
    band_urgent_loss_ticks: float = 2.0

    # === TBN: Physics blend ===
    physics_accel_weight: float = 0.5
    physics_ml_blend: float = 0.5

    # === TBN: Wave maturity composite ===
    wave_pattern_weight: float = 0.4
    wave_band_weight: float = 0.3
    wave_reversion_weight: float = 0.3
    wave_band_sigma_max: float = 3.0

    # === TBN: Conviction & path ===
    min_conviction: float = 0.48
    min_active_levels: int = 3
    conviction_scale: float = 2.0

    # === TBN: Band confluence ===
    band_min_active: int = 3
    band_majority_mult: float = 2.0
    band_trigger_threshold: float = 0.5
    band_normalize_sigma: float = 3.0
    band_blend_threshold: float = 0.3
    band_blend_weight: float = 0.4

    # === TBN: Pace blend ===
    pace_max_weight: float = 0.30
    pace_weight_scale: float = 0.15
    pace_ahead_long: float = 0.80
    pace_ahead_short: float = 0.20
    pace_behind_long: float = 0.35
    pace_behind_short: float = 0.65
    pace_tighten_threshold: float = 0.3
    pace_widen_threshold: float = 1.5
    pace_time_gate: float = 0.3

    # === TBN: Trade health fusion ===
    health_pace_weight: float = 0.6
    health_decay_weight: float = 0.4

    # === TBN: Exit signal thresholds ===
    tighten_wave_maturity: float = 0.85
    widen_wave_maturity: float = 0.30
    slow_flip_long_threshold: float = 0.45
    slow_flip_short_threshold: float = 0.55
    band_exit_strength_min: float = 0.5

    # === TBN: Decay cascade ===
    decay_alpha_max: float = 3.0
    decay_alpha_min: float = 1.0
    decay_theta_exit: float = 1.5
    decay_progress_cap: float = 2.0

    # === Execution: Pattern quality ===
    noise_z_threshold: float = 0.5
    approach_z_threshold: float = 2.0
    headroom_z_max: float = 3.0
    nightmare_z: float = 3.0

    # === Execution: Gate thresholds ===
    gate1_dist: float = 4.5
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
    momentum_accel_weight: float = 0.5
    momentum_trigger: float = 0.5
    momentum_conviction_cap: float = 0.15
    momentum_conviction_coeff: float = 0.05
    signed_mfe_conviction_cap: float = 0.30
    signed_mfe_conviction_coeff: float = 0.10
    brain_winrate_margin: float = 0.10
    template_bias_min_sum: float = 0.10
    band_cascade_conviction: float = 0.55
    dmi_long_prob: float = 0.55
    dmi_short_prob: float = 0.45
    velocity_long_prob: float = 0.52
    velocity_short_prob: float = 0.48

    # === Execution: SL/TP sizing ===
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
    atr_sl_mult: float = 3.0
    atr_tp_mult: float = 5.0
    significance_threshold: float = 2.0

    # === Execution: OLS TP bounds ===
    ols_lower_pct: float = 0.20
    ols_upper_pct: float = 5.0
    brain_tp_adjust_pct: float = 0.50

    # === Execution: Live bias ===
    live_bias_min_trades: int = 5
    live_bias_winrate_min: float = 0.60
    live_bias_margin: float = 0.15
