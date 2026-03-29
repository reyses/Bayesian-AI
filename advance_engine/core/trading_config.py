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

    # === Exit: Wrong Direction (never profitable) ===
    wrong_dir_min_bars: int = 8        # 2 min at 15s  -- time to develop before checking
    wrong_dir_adverse_ticks: float = 10.0  # $5 MNQ  -- adverse threshold to cut

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
    envelope_adx_slope_boost: float = 0.05   # rising ADX slows decay (hl *= 1 + slope * this)
    envelope_adx_slope_penalty: float = 0.1  # falling ADX speeds decay (hl *= 1 + slope * this)

    # === Exit: Giveback shape blend ===
    giveback_shape_blend: float = 0.70       # 70% shape threshold + 30% tier (when shape available)

    # === Exit: Belief flip ===
    belief_flip_di_gap: float = 5.0          # DI gap threshold for crossover exit (87% accurate at ≥5)
    belief_flip_min_bars: int = 3            # minimum bars before DI crossover allowed

    # === Exit: Band urgent ===
    band_urgent_min_strength: float = 0.6
    band_urgent_trigger_strength: float = 0.7
    band_urgent_loss_ticks: float = 0.0  # 0 = fire on any thesis invalidation (even in profit)

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

    # === Peak entry gate thresholds ===
    peak_adx_chop_threshold: float = 0.0    # disabled -- ADX filter adds no value (research 2026-03-19)
    # Research (2026-03-19): ADX filter has minimal impact on PF (1.97 at all thresholds).
    # ADX<15 blocked $1,888 profitable trades. ADX<10 blocked 20 trades ($7.80/tr).
    # Lowered to 2 -- only blocks completely flat markets with zero directional movement.
    peak_fake_vol_threshold: float = 2.5    # log1p(volume) above this = flow still active (fake)
    peak_fake_fm_threshold: float = 3.0     # log1p(F_momentum) above this = momentum building (fake)
    peak_sensor_min_oppose: int = 2         # block when N+ of 3 1m sensors oppose direction
    peak_cooldown_bars: int = 6             # bars to wait after peak exit before next entry

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
    sl_min_ticks: float = 2.0   # absolute minimum floor ($0.50 MNQ  -- near breakeven)
    sl_max_ticks: float = 200.0  # absolute maximum cap ($100 for MNQ)  -- prevents runaway from unscaled TF stats
    sl_mfe_ratio: float = 1.0   # SL capped at 1× expected MFE (risk = expected reward, not more)
    sl_tolerance_mult: float = 1.0  # multiplier on p95 MAE tolerance interval (>1 = wider, <1 = tighter)
    sl_tolerance_k: float = 5.0    # fallback: mean + k*std when p95 unavailable (5σ ≈ 99.99994%)
    tp_min_ticks: float = 4.0
    tp_default_ticks: float = 50.0
    trail_sigma_mult: float = 1.1
    trail_mae_mult: float = 1.1
    trail_default_ticks: float = 10.0
    trail_min_ticks: float = 2.0
    trail_activation_mae_mult: float = 0.3
    # Trailing stop activation: wait for MFE to reach this fraction of expected profit
    trail_activation_pct: float = 0.80       # activate at 80% of p75_mfe
    trail_activation_floor: float = 20.0     # minimum activation threshold ($5 MNQ)
    trail_activation_ceiling: float = 400.0  # maximum activation threshold ($100 MNQ)
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

    # === Fractal DMI (dual-timeframe trend gating) ===
    # State A: Fakeout filter  -- block trend entries when macro has no energy
    fdmi_fakeout_micro_z: float = 2.5       # micro Z breakout threshold
    fdmi_fakeout_macro_adx: float = 20.0    # macro ADX below this = no energy
    # State B: Wave rider  -- enter on micro pullback realigning with macro trend
    fdmi_trend_macro_adx: float = 25.0      # macro ADX above this = strong trend
    # State C: Fractal exhaustion  -- exit when micro energy spikes and dies at macro wall
    fdmi_exhaust_micro_adx: float = 45.0    # micro ADX overextended threshold
    fdmi_exhaust_macro_z: float = 2.5       # macro Z band wall (2.5-3.0 SE)
    # TF pairing
    fdmi_macro_tf: int = 60        # macro TF in seconds (60 = 1m)
    fdmi_micro_tf: int = 5         # micro TF in seconds (5 = 5s)

    # === C&E Matrix: Entry  -- Momentum Ignition (enhanced Wave Rider) ===
    ce_momentum_micro_z_max: float = 0.5       # micro Z pullback upper bound (fair value)
    ce_momentum_micro_z_min: float = -1.5      # micro Z pullback lower bound (1-sigma dip)

    # === C&E Matrix: Entry  -- Structural Reversion (Rubber Band / OU process) ===
    ce_reversion_macro_adx: float = 20.0       # macro ADX below this = mean-reverting regime
    ce_reversion_micro_z: float = 3.0          # micro Z extreme for reversion entry (3-sigma)

    # === C&E Matrix: Exit  -- Death Hook (Liquidity Absorption) ===
    ce_death_hook_micro_adx: float = 40.0      # micro ADX overextended threshold
    ce_death_hook_macro_z: float = 2.0         # macro Z band wall (2-sigma)

    # === C&E Matrix: Exit  -- Regime Decay (Sand Trap) ===
    ce_regime_decay_adx: float = 20.0          # macro ADX collapse threshold
    ce_regime_decay_di_cross: bool = True       # also exit on DI crossover against trade

    # === C&E Matrix: Exit  -- Survival Probability (Time-Stop) ===
    ce_survival_min_bars: int = 10             # minimum bars before time-stop activates
    ce_survival_target_pct: float = 0.50       # must achieve 50% of TP target in time
    ce_survival_z_var_max: float = 0.20        # Z variance below this = flatlining
    ce_survival_lookback: int = 5              # bars to compute Z variance over

    # === Quant: Hurst-based regime (replaces/supplements ADX) ===
    hurst_persistent_threshold: float = 0.55   # H > this = persistent trend (Hawkes)
    hurst_mean_revert_threshold: float = 0.45  # H < this = anti-persistent (OU)
    hurst_regime_exit: float = 0.50            # H drops below this = trend memory dead

    # === Quant: Volume Delta confirmation ===
    volume_delta_min: float = 0.0              # minimum abs(delta) to count as confirmed

    # === Quant: Bayesian ePnL exit ===
    epnl_exit_min_obs: int = 3                 # minimum brain observations before ePnL active
    epnl_exit_threshold: float = 0.0           # exit when ePnL drops below this

    # === Quant: Tidal Wave (adverse volatility expansion) ===
    tidal_wave_se_expansion_pct: float = 0.20  # SE must expand by 20% in lookback
    tidal_wave_lookback: int = 3               # bars to measure SE expansion over

    # === Entry: Regime-Aware Gate 0 (improvement A) ===
    regime_strong_adx: float = 25.0            # ADX above = strong trend
    regime_developing_adx: float = 20.0        # ADX above + rising = developing
    regime_exhaust_slope: float = -2.0         # ADX slope below = exhausting
    regime_range_adx: float = 20.0             # ADX below + H<0.45 = range

    # === Entry: Multi-TF Confluence Gate 2.5 (improvement C) ===
    tf_confluence_min: float = 0.40            # block when < 40% TFs agree on direction
    tf_confluence_enabled: bool = True

    # === Entry: Time-of-Day Session Filter (improvement F) ===
    session_filter_enabled: bool = True
    overnight_z_min: float = 1.5               # minimum z for overnight entries
    overnight_dist_mult: float = 0.7           # tighter template match overnight

    # === Entry: Volatility-Normalized Sizing (improvement B) ===
    vol_sizing_enabled: bool = True
    vol_sl_strong_trend: float = 1.5           # SL = ATR × this in strong trend
    vol_sl_developing: float = 2.0
    vol_sl_range: float = 2.5
    vol_sl_default: float = 2.0
    vol_tp_strong_trend: float = 5.0           # TP = ATR × this in strong trend
    vol_tp_developing: float = 3.0
    vol_tp_range: float = 2.0
    vol_tp_default: float = 3.0

    # === Entry: Confidence-Weighted Direction (improvement G) ===
    dir_voting_enabled: bool = True
    dir_min_vote_score: float = 0.08           # minimum net score to enter (0 = any signal)
    dir_smfe_weight: float = 3.0               # signed MFE regression weight
    dir_logistic_weight: float = 2.5           # per-cluster logistic weight
    dir_brain_weight: float = 1.5              # brain direction WR weight
    dir_template_weight: float = 1.0           # template bias weight
    dir_band_weight: float = 2.0               # multi-TF band confluence weight
    dir_dmi_weight: float = 1.5                # DMI trend weight
    dir_velocity_weight: float = 0.5           # velocity fallback weight
    dir_fdmi_weight: float = 4.0               # FDMI ignition/reversion weight (highest)
