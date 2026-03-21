"""
Unified Exit Engine  -- cascade orchestrator
============================================
Owns: position state, MFE tracking, sub-bar wick resolution, cascade ordering.
Delegates each exit check to its standalone module in core/exits/.

Usage:
    exit_eng = ExitEngine(mode='training', tick_size=0.25, tick_value=0.50)
    pos = exit_eng.open_position(side, entry_price, bar_idx, tid, lib_entry)
    result = exit_eng.evaluate(pos, bar_high, bar_low, bar_close, bar_idx, ...)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class ExitAction(Enum):
    HOLD = 'hold'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    TRAIL_STOP = 'trail_stop'
    MAX_HOLD = 'max_hold'
    ENVELOPE_DECAY = 'envelope_decay'
    BAND_URGENT = 'band_urgent'
    WATCHDOG = 'watchdog'
    BREAKEVEN_LOCK = 'breakeven_lock'
    PEAK_GIVEBACK = 'peak_giveback'
    BROWNIAN_GIVEBACK = 'brownian_giveback'  # Brownian motion reversal detection
    MAINTENANCE_FLAT = 'maintenance_flat'
    # C&E Matrix structural exits
    DEATH_HOOK = 'death_hook'           # liquidity absorption at macro wall
    REGIME_DECAY = 'regime_decay'       # macro trend collapsed / DI reversal
    SURVIVAL_STOP = 'survival_stop'     # Bayesian ePnL expired
    SURVIVAL_FLATLINE = 'survival_flatline'  # structural flatline (z-var + under target)
    BELIEF_FLIP = 'belief_flip'         # TBN consensus flipped or DI crossover
    TIDAL_WAVE = 'tidal_wave'           # adverse volatility expansion
    V_REVERSAL = 'v_reversal'           # 4 bars off peak + profit + breakeven locked
    PEAK_STATE_EXIT = 'peak_state_exit'  # inverted entry: opposite direction entry conditions met


@dataclass
class ExitResult:
    action: ExitAction
    exit_price: float
    reason: str
    pnl_ticks: float = 0.0
    bars_held: int = 0
    trail_level: float = 0.0
    envelope_level: float = 0.0
    band_zone: str = ''
    band_action: str = ''  # 'tighten', 'widen', 'urgent', 'none'


@dataclass
class PositionState:
    """Unified position state  -- used by ExitEngine and exit modules."""
    side: str                    # 'long' or 'short'
    entry_price: float
    entry_bar_index: int
    template_id: int
    tick_size: float = 0.25
    tick_value: float = 0.50     # MNQ: $0.50 per tick

    # Template-specific parameters (from pattern_library)
    sl_ticks: float = 0.0
    tp_ticks: float = 0.0
    trail_activation_ticks: float = 0.0
    trailing_stop_ticks: float = 0.0
    max_hold_bars: int = 120

    # Dynamic state (updated each bar)
    peak_favorable: float = 0.0
    bars_held: int = 0
    bars_since_peak: int = 0        # bars since last new MFE  -- V-reversal detector
    peak_volume: float = 0.0          # highest volume seen during trade (for vol exhaustion)
    breakeven_locked: bool = False
    envelope_active: bool = False
    envelope_level: float = 0.0

    # Position lifecycle
    entry_time: float = 0.0
    stop_loss: float = 0.0          # absolute price level
    profit_target: float = 0.0      # absolute price level (0 = none)
    envelope_T0: float = 0.0
    envelope_halflife: float = 20.0

    # 30m worker flip: once fired, stays True for remainder of trade
    slow_flip_active: bool = False

    # Discovery TF for TF-aware exits (regime_decay checks this TF's DMI)
    discovery_tf_seconds: float = 15.0

    # DMI direction confirmation counter  -- uses discovery TF's DMI.
    # Counts consecutive bars where discovery-TF DMI agrees with trade direction.
    # At 3+ bars, direction is confirmed (research: MAE drops 40%).
    dmi_confirmed_bars: int = 0
    dmi_direction_confirmed: bool = False

    # Anchor: expected trade outcome (price + time)
    anchor_mfe_ticks: float = 0.0   # template p75_mfe, brain-adjusted
    anchor_mfe_bars: float = 0.0    # template avg_mfe_bar, brain-adjusted

    # Shape primitive tracking (two-stage primitives)
    entry_primitive_id: Optional[int] = None
    exit_primitive_id: Optional[int] = None
    exit_primitive_confidence: float = 0.0
    envelope_halflife_mult: float = 1.0   # from exit primitive or --shapes calibration

    # Template shape calibration (from --shapes flag, set at open_position)
    # Dict with: giveback_pct, delay_bars, hl_mult, peak_bar. None = not calibrated.
    template_shape_params: Optional[Dict] = None

    # Peak override: track suppressed exit requests
    # peak_overrides is a list of (bar_idx, exit_action_name, reason) tuples
    # showing every exit the cascade WANTED but peak detection overrode.
    peak_overrides: Optional[list] = None
    is_peak_trade: bool = False  # True when template_id == -100


class ExitEngine:
    """
    Unified exit engine for training and live.

    Mode differences (data availability only):
        training: receives intra-bar high/low for wick checking
        live:     receives 15s bar OHLC (use high/low, not just close)

    All logic is identical regardless of mode.
    """

    def __init__(
        self,
        mode: str = 'live',
        tick_size: float = 0.25,
        tick_value: float = 0.50,
        config=None,
        exit_primitives=None,
        min_hold_bars: int = 0,
    ):
        assert mode in ('training', 'live'), f"Invalid mode: {mode}"
        self.mode = mode
        self.tick_size = tick_size
        self.tick_value = tick_value
        self._exit_primitives = exit_primitives  # ExitPrimitiveLibrary or None
        self.min_hold_bars = min_hold_bars  # 0 = disabled; suppresses non-reversal exits

        # Lazy import to avoid circular deps
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self.config = config

        # ── Exit modules (params from config) ──
        from core.exits.stop_loss import StopLossCheck
        from core.exits.take_profit import TakeProfitCheck
        from core.exits.breakeven import TrailingStop
        from core.exits.envelope import EnvelopeDecay
        from core.exits.giveback import PeakGiveback
        from core.exits.band_exit import BandUrgentExit
        from core.exits.watchdog import WatchdogCheck
        from core.exits.belief_flip import BeliefFlipExit

        self.stop_loss = StopLossCheck()
        self.take_profit = TakeProfitCheck()
        self.breakeven = TrailingStop(
            activation_pct=config.trail_activation_pct,
            activation_floor_ticks=config.trail_activation_floor,
            activation_ceiling_ticks=config.trail_activation_ceiling,
            buffer_ticks=config.be_buffer_ticks)
        self.envelope = EnvelopeDecay(
            half_life_bars=config.envelope_halflife_bars,
            floor_pct=config.envelope_floor_pct,
            min_bars=config.envelope_min_bars,
            config=config)
        self.giveback = PeakGiveback(
            min_mfe_ticks=config.giveback_min_mfe_ticks,
            giveback_pct=config.giveback_pct,
            config=config)
        self.band_exit = BandUrgentExit(config=config)
        self.watchdog = WatchdogCheck(
            tick_threshold=config.watchdog_tick_threshold,
            bar_threshold=config.watchdog_bar_threshold,
            worker_threshold=config.watchdog_worker_threshold,
            config=config)
        self.belief_flip = BeliefFlipExit(
            di_gap_threshold=config.belief_flip_di_gap,
            min_bars=config.belief_flip_min_bars)

        from core.exits.fractal_exhaust import FractalExhaustExit
        from core.exits.regime_decay import RegimeDecayExit
        from core.exits.survival_stop import SurvivalStopExit
        from core.exits.tidal_wave import TidalWaveExit
        from core.exits.peak_state_exit import PeakStateExit
        self.fractal_exhaust = FractalExhaustExit(config=config)
        self.regime_decay = RegimeDecayExit(config=config)
        self.survival_stop = SurvivalStopExit(config=config)
        self.tidal_wave = TidalWaveExit(config=config)
        self.peak_state = PeakStateExit(config=config)

        # Self-tuning state  -- two independent counters
        self._tune_too_early = 0
        self._tune_too_late = 0
        self._tune_total = 0
        self._tune_hl_min = config.tune_hl_min
        self._tune_hl_max = config.tune_hl_max
        self._tune_gb_min = config.tune_gb_min
        self._tune_gb_max = config.tune_gb_max
        self._tune_window = config.tune_window

    def set_brain(self, brain):
        """Attach Bayesian brain for ePnL-based exits."""
        self.survival_stop.set_brain(brain)

    # ── Convenience aliases for tuned params ──

    @property
    def envelope_half_life_bars(self):
        return self.envelope.half_life_bars

    @envelope_half_life_bars.setter
    def envelope_half_life_bars(self, val):
        self.envelope.half_life_bars = val

    @property
    def giveback_pct(self):
        return self.giveback.giveback_pct

    @giveback_pct.setter
    def giveback_pct(self, val):
        self.giveback.giveback_pct = val

    @property
    def envelope_floor_pct(self):
        return self.envelope.floor_pct

    @property
    def envelope_min_bars(self):
        return self.envelope.min_bars

    @property
    def giveback_min_mfe_ticks(self):
        return self.giveback.min_mfe_ticks

    @property
    def be_activation_ticks(self):
        return self.breakeven.activation_ticks

    @be_activation_ticks.setter
    def be_activation_ticks(self, val):
        self.breakeven.activation_ticks = val

    # ==================================================================
    # SELF-TUNING
    # ==================================================================

    def record_trade_outcome(self, trade_mfe_ticks: float, actual_pnl_ticks: float,
                             capture_rate: float):
        """Feed closed-trade stats back to auto-tune exit parameters."""
        c = self.config
        self._tune_total += 1

        if 0 < capture_rate < c.tune_too_early_capture and trade_mfe_ticks < c.tune_too_early_mfe:
            self._tune_too_early += 1

        if trade_mfe_ticks >= c.giveback_min_mfe_ticks:
            gave_back = (trade_mfe_ticks - actual_pnl_ticks) / trade_mfe_ticks
            if gave_back >= c.tune_too_late_giveback:
                self._tune_too_late += 1

        if self._tune_total % self._tune_window == 0 and self._tune_total > 0:
            if self._tune_too_early >= 3:
                self.envelope_half_life_bars = min(
                    self._tune_hl_max,
                    self.envelope_half_life_bars * c.tune_growth_rate)
            if self._tune_too_late >= 3:
                self.giveback_pct = max(
                    self._tune_gb_min,
                    self.giveback_pct - c.tune_shrink_step)
            if self._tune_too_early < 3 and self._tune_too_late < 3:
                self.envelope_half_life_bars += (
                    c.envelope_halflife_bars - self.envelope_half_life_bars
                ) * c.tune_revert_rate
                self.giveback_pct += (
                    c.giveback_pct - self.giveback_pct
                ) * c.tune_revert_rate
            self._tune_too_early = 0
            self._tune_too_late = 0

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def open_position(
        self,
        side: str,
        entry_price: float,
        entry_bar_index: int,
        template_id: int,
        sl_ticks: float,
        tp_ticks: float,
        trail_ticks: float = 0.0,
        trail_activation_ticks: float = 0.0,
        max_hold_bars: int = 120,
        lib_entry: dict = None,
        entry_primitive_id: int = None,
    ) -> PositionState:
        """Initialize a new position with pre-computed exit parameters."""
        if lib_entry:
            _p75_bar = lib_entry.get('p75_mfe_bar', 0.0)
            if _p75_bar > 0:
                # Convert discovery TF bars to 15s execution bars
                _disc_tf_sec = lib_entry.get('discovery_tf_seconds', 15.0)
                _p75_bar_exec = _p75_bar * (_disc_tf_sec / 15.0)
                max_hold_bars = int(_p75_bar_exec * self.config.timescale_urgent_mult)
            else:
                max_hold_bars = lib_entry.get('max_hold_bars', max_hold_bars)

        _anchor_mfe = 0.0
        _anchor_bars = 0.0
        if lib_entry:
            _disc_tf_sec = lib_entry.get('discovery_tf_seconds', 15.0)
            _tf_ratio = _disc_tf_sec / 15.0
            _tf_scale = _tf_ratio ** 0.5 if _tf_ratio > 1.0 else 1.0
            # Scale MFE from discovery TF to execution TF (sqrt diffusion)
            _anchor_mfe = lib_entry.get('p75_mfe_ticks', 0.0) / _tf_scale
            _anchor_bars = lib_entry.get('avg_mfe_bar', 0.0)
            # Convert bar-based stats from discovery TF to 15s execution TF
            if _tf_ratio > 1.0 and _anchor_bars > 0:
                _anchor_bars = _anchor_bars * _tf_ratio

        # Template shape calibration (from --shapes)
        _tsp = None
        _hl_mult = 1.0
        if lib_entry and lib_entry.get('shape_giveback_pct') is not None:
            _disc_tf_sec = lib_entry.get('discovery_tf_seconds', 15.0)
            _tf_ratio = _disc_tf_sec / 15.0
            _tsp = {
                'giveback_pct': lib_entry['shape_giveback_pct'],
                'delay_bars': lib_entry.get('shape_delay_bars', 3) * _tf_ratio,
            }
            _hl_mult = lib_entry.get('shape_envelope_hl_mult', 1.0)

        pos = PositionState(
            side=side,
            entry_price=entry_price,
            entry_bar_index=entry_bar_index,
            template_id=template_id,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            sl_ticks=float(sl_ticks),
            tp_ticks=float(tp_ticks),
            trailing_stop_ticks=float(trail_ticks) if trail_ticks else float(sl_ticks),
            trail_activation_ticks=float(trail_activation_ticks or 0),
            max_hold_bars=max_hold_bars,
            anchor_mfe_ticks=float(_anchor_mfe),
            anchor_mfe_bars=float(_anchor_bars),
            discovery_tf_seconds=float(lib_entry.get('discovery_tf_seconds', 15.0)) if lib_entry else 15.0,
            template_shape_params=_tsp,
            envelope_halflife_mult=_hl_mult,
        )

        # Min-hold experiment: SL is last resort if all reversal exits fail  -- wide floor
        if self.min_hold_bars > 0:
            sl_ticks = max(sl_ticks, 40.0)
        _sd = sl_ticks * self.tick_size
        _td = tp_ticks * self.tick_size
        if side == 'long':
            pos.stop_loss = entry_price - _sd
            pos.profit_target = entry_price + _td if tp_ticks > 0 else 0.0
        else:
            pos.stop_loss = entry_price + _sd
            pos.profit_target = entry_price - _td if tp_ticks > 0 else 0.0

        pos.peak_favorable = entry_price
        pos.is_peak_trade = (template_id == -100)
        pos.peak_overrides = []
        pos.envelope_active = True
        pos.envelope_level = tp_ticks * self.tick_size
        pos.envelope_T0 = float(sl_ticks)

        import time as _time
        pos.entry_time = _time.time()

        if entry_primitive_id is not None:
            pos.entry_primitive_id = entry_primitive_id

        return pos

    def evaluate(
        self,
        pos: PositionState,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        current_bar_index: int,
        band_context: dict = None,
        net_force: float = 0.0,
        exit_signal: dict = None,
        sub_bar_highs: list = None,
        sub_bar_lows: list = None,
        noise_ticks: float = 0.0,
        belief_network=None,
    ) -> ExitResult:
        """
        Evaluate all exit conditions for current bar.

        Cascade order (first trigger wins):
        === Structural exits (thesis invalidation  -- before SL/TP) ===
        1. Death Hook        -- liquidity absorption at macro wall
        2. Regime Decay      -- Hurst/ADX collapse + DI trend reversal
        3. Survival Stop     -- Bayesian ePnL / time-survival probability
        4. Tidal Wave        -- adverse volatility expansion
        === Standard exits ===
        5. Stop Loss        6. Take Profit      7. Watchdog
        8. Band Urgent      9. Breakeven lock   10. Envelope decay
        11. Peak Giveback   12. Belief Flip     13. HOLD
        """
        pos.bars_held = current_bar_index - pos.entry_bar_index
        pos.bars_in_trade = pos.bars_held  # sync alias

        # ── Resolve worst/best price this bar ──
        if sub_bar_highs is not None and sub_bar_lows is not None and len(sub_bar_highs) > 0:
            worst_price = min(sub_bar_lows) if pos.side == 'long' else max(sub_bar_highs)
            best_price = max(sub_bar_highs) if pos.side == 'long' else min(sub_bar_lows)
        else:
            worst_price = bar_low if pos.side == 'long' else bar_high
            best_price = bar_high if pos.side == 'long' else bar_low

        # ── Update MFE tracking ──
        _old_peak = pos.peak_favorable
        if pos.side == 'long':
            pos.peak_favorable = max(pos.peak_favorable, best_price)
        else:
            pos.peak_favorable = min(pos.peak_favorable, best_price)
        if pos.peak_favorable != _old_peak:
            pos.bars_since_peak = 0
        else:
            pos.bars_since_peak += 1

        # ── Discovery TF DMI direction confirmation (3 bars = confirmed) ──
        # Research: at 3 confirmed bars, MAE drops 40%, MFE increases 10%.
        # Uses discovery TF DMI (from exit_signal, routed by TBN).
        if exit_signal is not None:
            _di_p = exit_signal.get('di_plus', 0.0)
            _di_m = exit_signal.get('di_minus', 0.0)
            _dmi_agrees = (
                (pos.side == 'long' and _di_p > _di_m) or
                (pos.side == 'short' and _di_m > _di_p)
            )
            if _dmi_agrees:
                pos.dmi_confirmed_bars += 1
            else:
                pos.dmi_confirmed_bars = 0  # reset  -- DMI flipped against
            if pos.dmi_confirmed_bars >= 3 and not pos.dmi_direction_confirmed:
                pos.dmi_direction_confirmed = True

        # ── Cascade ──
        ts = self.tick_size

        # ── Min-hold suppression: only SL + strong DMI reversal during hold ──
        _in_hold_period = (self.min_hold_bars > 0 and pos.bars_held < self.min_hold_bars)

        # Strong DMI reversal check: DI crossed against position with large gap
        _strong_dmi_reversal = False
        if _in_hold_period and exit_signal is not None and pos.bars_held >= 4:
            _di_plus = exit_signal.get('di_plus', 0.0)
            _di_minus = exit_signal.get('di_minus', 0.0)
            _di_plus_prev = exit_signal.get('di_plus_prev', _di_plus)
            _di_minus_prev = exit_signal.get('di_minus_prev', _di_minus)
            _di_gap = abs(_di_plus - _di_minus)
            if pos.side == 'long':
                _crossed = (_di_plus_prev > _di_minus_prev and _di_minus >= _di_plus)
            else:
                _crossed = (_di_minus_prev > _di_plus_prev and _di_plus >= _di_minus)
            # Strong = crossed AND DI gap >= 5 (5m DMI is 87% accurate at gap≥5)
            _strong_dmi_reversal = _crossed and _di_gap >= 5.0

        # === ADAPTIVE THRESHOLDS (based on whether trade has peaked) ===
        # Never-profitable trades get tighter exits. Peaked trades get room.
        # This replaces the fixed wrong-direction exit with data-driven adaptation.
        if pos.side == 'long':
            _peak_ticks = (pos.peak_favorable - pos.entry_price) / ts
        else:
            _peak_ticks = (pos.entry_price - pos.peak_favorable) / ts
        _never_profitable = _peak_ticks < 2.0 and pos.bars_held >= 4

        # === PEAK OVERRIDE: DISABLED (reverted) ===
        # Peak override held losers too long, driving PF below 1.0.
        # Exits fire normally. ADX chop filter on entry handles noise instead.
        # pos.peak_overrides still tracked for observational logging.

        # === PROFIT PROTECTION (giveback first  -- if trade peaked, protect it) ===

        # 1. Peak Giveback
        if not _in_hold_period:
            _shape_params = None
            if (self._exit_primitives is not None
                    and pos.exit_primitive_id is not None
                    and pos.exit_primitive_confidence >= 0.3):
                _shape_params = self._exit_primitives.get_exit_params(pos.exit_primitive_id)
            elif pos.template_shape_params is not None:
                _shape_params = pos.template_shape_params
            r = self.giveback.evaluate(pos, bar_close, ts, exit_signal, noise_ticks,
                                       shape_params=_shape_params)
            if r:
                # Adaptive learning: record volume state at exit
                _peak_vol = getattr(pos, 'peak_volume', 0.0)
                _curr_vol = exit_signal.get('current_volume', 0.0) if exit_signal else 0.0
                _vol_drop = _curr_vol / _peak_vol if _peak_vol > 0 else 1.0
                _peak_t = r.pnl_ticks + (r.pnl_ticks if r.pnl_ticks > 0 else 0)  # approximate
                if pos.side == 'long':
                    _peak_t = (pos.peak_favorable - pos.entry_price) / pos.tick_size
                else:
                    _peak_t = (pos.entry_price - pos.peak_favorable) / pos.tick_size
                self.giveback.record_exit(_vol_drop, r.pnl_ticks, _peak_t)
                return r

        # === INVERTED ENTRY EXIT (would the system enter against me?) ===
        if not _in_hold_period:
            r = self.peak_state.evaluate(
                pos, bar_close, ts, current_bar_index, exit_signal)
            if r: return r

        # === STRUCTURAL EXITS (thesis invalidation) ===

        # 2. Death Hook (Liquidity Absorption)
        if not _in_hold_period:
            r = self.fractal_exhaust.evaluate(pos, bar_close, ts, belief_network)
            if r: return r

        # 3. Regime Decay
        if not _in_hold_period or _strong_dmi_reversal:
            r = self.regime_decay.evaluate(pos, bar_close, ts, belief_network)
            if r: return r

        # 4. Survival Stop
        if not _in_hold_period:
            r = self.survival_stop.evaluate(pos, bar_close, ts, belief_network)
            if r: return r

        # 5. Tidal Wave
        if not _in_hold_period:
            r = self.tidal_wave.evaluate(pos, bar_close, ts, belief_network, exit_signal)
            if r: return r

        # === STANDARD EXITS ===

        # 5. Stop Loss  -- ALWAYS allowed (capital protection)
        r = self.stop_loss.evaluate(pos, worst_price, ts)
        if r: return r

        # 6. Take Profit
        if not _in_hold_period:
            r = self.take_profit.evaluate(pos, best_price, ts)
            if r: return r

        # 7. Watchdog
        if not _in_hold_period:
            r = self.watchdog.evaluate(pos, bar_close, ts, exit_signal)
            if r: return r

        # 8. Band Urgent
        if not _in_hold_period:
            r = self.band_exit.evaluate(pos, bar_close, ts, band_context)
            if r: return r

        # 9. Trailing stop (adjusts SL in-place, ratchets behind peak)
        self.breakeven.apply(pos, ts, exit_signal=exit_signal)

        # 9b. V-reversal exit
        if pos.bars_since_peak >= 4 and pos.breakeven_locked:
            if pos.side == 'long':
                _pnl_ticks = (bar_close - pos.entry_price) / ts
            else:
                _pnl_ticks = (pos.entry_price - bar_close) / ts
            if _pnl_ticks > 0:
                _mfe = ((pos.peak_favorable - pos.entry_price) / ts
                        if pos.side == 'long'
                        else (pos.entry_price - pos.peak_favorable) / ts)
                return ExitResult(
                    action=ExitAction.V_REVERSAL,
                    exit_price=bar_close,
                    reason=f"V-reversal: {pos.bars_since_peak} bars off peak "
                           f"(MFE={_mfe:.0f}t, exit={_pnl_ticks:.1f}t)",
                    pnl_ticks=_pnl_ticks,
                    bars_held=pos.bars_held,
                    trail_level=pos.stop_loss,
                )

        # 10. Envelope Decay
        if not _in_hold_period:
            r = self.envelope.evaluate(pos, bar_close, ts, net_force, band_context,
                                       noise_ticks)
            if r: return r

        # 11. (Giveback moved to position #1)

        # 12. Belief Flip
        if not _in_hold_period or _strong_dmi_reversal:
            r = self.belief_flip.evaluate(pos, bar_close, ts, exit_signal)
            if r: return r

        # 13. HOLD
        return ExitResult(
            action=ExitAction.HOLD,
            exit_price=0.0,
            reason='hold',
            bars_held=pos.bars_held,
            trail_level=pos.stop_loss,
            envelope_level=pos.envelope_level,
            band_zone=band_context.get('band_summary', '') if band_context else '',
        )

    # ==================================================================
    # CME Maintenance Flatten
    # ==================================================================

    MAINT_FLATTEN_MINUTE = 16 * 60 + 45   # 16:45 ET
    MAINT_REOPEN_MINUTE = 18 * 60         # 18:00 ET

    @staticmethod
    def is_maintenance_window(bar_timestamp: float) -> bool:
        """Check if bar falls in CME maintenance flatten window (16:45-18:00 ET)."""
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        _et = datetime.fromtimestamp(bar_timestamp, tz=timezone.utc).astimezone(
            ZoneInfo('US/Eastern'))
        _minute_of_day = _et.hour * 60 + _et.minute
        return (ExitEngine.MAINT_FLATTEN_MINUTE
                <= _minute_of_day
                < ExitEngine.MAINT_REOPEN_MINUTE)

    # ==================================================================
    # PRIVATE -- Utilities (kept for backward compat)
    # ==================================================================

    def _calc_pnl_ticks(self, pos: PositionState, exit_price: float) -> float:
        if pos.side == 'long':
            return (exit_price - pos.entry_price) / self.tick_size
        else:
            return (pos.entry_price - exit_price) / self.tick_size
