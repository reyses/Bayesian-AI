"""Peak Giveback  -- exit when trade retraces too far from MFE peak.

Uses volume-relative thresholds + adaptive learning from recent trades.
The probability of winning decays with time AND volume exhaustion.
"""
from typing import Optional
from collections import deque

from core.exit_engine import ExitAction, ExitResult, PositionState


class PeakGiveback:

    def __init__(self, min_mfe_ticks: float = 16, giveback_pct: float = 0.10,
                 config=None):
        self.min_mfe_ticks = min_mfe_ticks
        self.giveback_pct = giveback_pct
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._aggressive_mult = config.giveback_aggressive_mult
        self._aggressive_pct = config.giveback_aggressive_pct
        self._slow_flip_reduction = config.giveback_slow_flip_reduction
        self._slow_flip_floor = config.giveback_slow_flip_floor
        self._anchor_patience_pct = config.giveback_anchor_patience_pct
        self._shape_blend = config.giveback_shape_blend

        # Adaptive volume exit: learn optimal vol_drop threshold from recent trades
        self._vol_drop_threshold = 0.50  # start at 50% drop from peak volume
        self._recent_exits = deque(maxlen=30)  # last 30 giveback exits
        self._adapt_interval = 10  # recalibrate every 10 exits
        self._adapt_count = 0

    def get_threshold(self, peak_ticks: float, noise_ticks: float = 0.0) -> float:
        """Tiered giveback threshold.

        Peak MFE (ticks)  ->  Giveback trigger
        2x noise+         ->  aggressive_pct (40%)
        1x noise - 2x     ->  self.giveback_pct (~55-70%)
        < noise            ->  disabled (move within noise floor)
        """
        min_mfe = (max(self.min_mfe_ticks, noise_ticks)
                   if noise_ticks > 0 else self.min_mfe_ticks)
        if peak_ticks >= min_mfe * self._aggressive_mult:
            return self._aggressive_pct
        elif peak_ticks >= min_mfe:
            return self.giveback_pct
        else:
            return 1.01  # >100% = never triggers

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 exit_signal: dict = None,
                 noise_ticks: float = 0.0,
                 shape_params: dict = None) -> Optional[ExitResult]:
        # 30m slow-flip detection (sticky once set)
        if exit_signal and exit_signal.get('slow_flip_tighten'):
            pos.slow_flip_active = True

        if pos.side == 'long':
            peak_ticks = (pos.peak_favorable - pos.entry_price) / tick_size
            current_ticks = (bar_close - pos.entry_price) / tick_size
        else:
            peak_ticks = (pos.entry_price - pos.peak_favorable) / tick_size
            current_ticks = (pos.entry_price - bar_close) / tick_size

        if peak_ticks <= 0:
            return None

        # Research-backed giveback: peak detection + volume + range analysis.
        #
        # 1. Peak detected via bars_since_peak (P_center + F_momentum change at +1 bar)
        #    83% detection rate, 85% precision, 0% false alarm (IS validated)
        # 2. After peak: check post-peak bar characteristics
        #    - Range expanding (>1.2x pre-peak): danger, tighten (IS 75% -> OOS 67% WR)
        #    - Range shrinking (<0.8x): safe, hold (IS 87% -> OOS 84% WR)
        #    - Volume collapse (post < 50% pre): move done (IS 88% -> OOS confirmed)
        #    - Volume spike (post > 150% pre): counter-force, tighten (IS 55% WR)
        #
        _min_peak = self.min_mfe_ticks
        if pos.anchor_mfe_ticks > 0:
            _min_peak = max(_min_peak, pos.anchor_mfe_ticks * 0.20)
        if peak_ticks < _min_peak:
            return None  # trade never reached projected peak  -- not giveback's job

        gave_back = peak_ticks - current_ticks
        if gave_back <= 0:
            return None  # still at or above peak

        # Peak must be at least 2 bars old to confirm (1 bar lag for detection)
        if pos.bars_since_peak < 2:
            return None

        # Volume-relative exit: when volume drops below threshold of peak trade volume,
        # the move is exhausting. Tighten giveback proportionally.
        # Volume tells you the move is dying BEFORE price confirms it.
        _vol_exhausted = False
        if exit_signal is not None:
            _current_vol = abs(exit_signal.get('current_volume', 0.0))
            _peak_vol = getattr(pos, 'peak_volume', 0.0) or 0.0
            # Track peak volume during trade
            if _current_vol > _peak_vol:
                pos.peak_volume = _current_vol
                _peak_vol = _current_vol
            # Volume dropped below threshold of peak
            if _peak_vol > 0 and _current_vol < _peak_vol * self._vol_drop_threshold:
                _vol_exhausted = True

        # Fake peak override: if volume + momentum are STILL flowing with the trade,
        # the pullback is noise — hold. Giveback says "exit," fake peak says "not yet."
        # Research: high vol+fm at peak = continuation, not reversal.
        # The fake peak flag is set by advance_engine when log_vol > 2.5 AND log_fm > 3.0.
        _fake_peak_hold = False
        if exit_signal is not None:
            _current_vol = abs(exit_signal.get('current_volume', 0.0))
            _peak_vol = getattr(pos, 'peak_volume', 0.0) or 0.0
            # Volume still > 60% of peak = move isn't done, this is a fake peak (noise dip)
            if _peak_vol > 0 and _current_vol > _peak_vol * 0.60:
                _fake_peak_hold = True
                # Track the override for accountability
                if not hasattr(pos, '_giveback_overrides'):
                    pos._giveback_overrides = 0
                pos._giveback_overrides = getattr(pos, '_giveback_overrides', 0) + 1
                # Safety: if overridden 5+ times, volume is lying — exit anyway
                if pos._giveback_overrides >= 5:
                    _fake_peak_hold = False

        # Base threshold: 65% giveback = exit (widened for honest entries)
        _threshold_pct = 0.65

        # Fake peak override: widen threshold (harder to trigger = hold longer)
        if _fake_peak_hold:
            _threshold_pct = 0.80  # volume says "not done" -- give it room

        # Volume exhaustion tightens threshold: move is dying, lock profit
        if _vol_exhausted:
            _threshold_pct = 0.40  # volume says "done" -- exit but not panic

        # Modulate by sensor fusion: 1s velocity (fast) + 1m volume (slow/accurate)
        if exit_signal is not None:
            _vel_flipped = exit_signal.get('vel_flipped', False)
            _vol_collapsing = exit_signal.get('vol_collapsing', False)
            _adx_slope = exit_signal.get('adx_slope', 0.0)
            _exec_flip = exit_signal.get('exec_tf_flip', False)

            # Sensor fusion: both sensors agree = high confidence exit
            if _vel_flipped and _vol_collapsing:
                _threshold_pct = 0.40  # both agree -> exit (widened from 0.20)

            # Fast sensor only: not enough for action alone
            elif _vel_flipped:
                pass  # removed — 1s velocity alone is noise without lookahead

            # Slow sensor only: 1m volume collapsing
            elif _vol_collapsing:
                _threshold_pct = 0.50  # institutional flow dying (widened from 0.30)

            # ADX collapsing — disabled, was too aggressive
            # if _adx_slope < -2.0:
            #     _threshold_pct = min(_threshold_pct, 0.30)

            # 15s execution TF flipped -> tighten aggressively
            if _exec_flip:
                _threshold_pct = min(_threshold_pct, 0.25)

        # Never-profitable override: if trade never peaked meaningfully, exit at 30%
        if peak_ticks < _min_peak * 1.5:
            _threshold_pct = min(_threshold_pct, 0.30)

        if peak_ticks > 0 and gave_back / peak_ticks >= _threshold_pct:
            return ExitResult(
                action=ExitAction.PEAK_GIVEBACK,
                exit_price=bar_close,
                reason=f"Peak giveback: peak={peak_ticks:.1f}t "
                       f"now={current_ticks:.1f}t gave_back={gave_back/peak_ticks:.0%} "
                       f"(threshold={_threshold_pct:.0%})",
                pnl_ticks=current_ticks,
                bars_held=pos.bars_held,
            )

        if not pos.dmi_direction_confirmed:
            return None  # below threshold AND no DMI confirmation  -- hold

        # Anchor patience: trade still developing  -- suppress if still in profit
        # and hasn't reached expected peak within expected time.
        if (pos.anchor_mfe_ticks > 0 and pos.anchor_mfe_bars > 0
                and pos.bars_held < pos.anchor_mfe_bars
                and peak_ticks < pos.anchor_mfe_ticks * self._anchor_patience_pct
                and current_ticks > 0):  # still in profit  -- patience OK
            return None

        threshold = self.get_threshold(peak_ticks, noise_ticks)

        # Shape-aware override: blend data-derived threshold with tier-based
        if shape_params is not None:
            shape_threshold = shape_params['giveback_pct']
            blend = self._shape_blend  # shape vs tier blend (default 70/30)
            threshold = blend * shape_threshold + (1 - blend) * threshold
            # Delay: suppress giveback before shape's expected peak bar
            if pos.bars_held < shape_params.get('delay_bars', 0):
                threshold = 1.01  # effectively disabled until delay expires

        # ADX slope tightening: rapid trend deceleration -> tighten giveback by 10pp
        if exit_signal is not None:
            _adx_slope = exit_signal.get('adx_slope', 0.0)
            if _adx_slope < -2.0 and threshold < 1.0:
                threshold = max(0.25, threshold - 0.10)

        # 15s execution-TF flip: micro structure reversed -> tighten aggressively
        # OOS validated: 66% of losses had 15s flip vs 44% of wins (+22% edge)
        if exit_signal is not None and exit_signal.get('exec_tf_flip') and threshold < 1.0:
            threshold = max(0.15, threshold * 0.5)  # halve threshold = exit faster

        # 30m flip tightens threshold
        if pos.slow_flip_active and threshold < 1.0:
            threshold = max(self._slow_flip_floor,
                            threshold - self._slow_flip_reduction)

        gave_back = peak_ticks - current_ticks
        if gave_back / peak_ticks >= threshold:
            return ExitResult(
                action=ExitAction.PEAK_GIVEBACK,
                exit_price=bar_close,
                reason=f"Peak giveback: peak={peak_ticks:.1f}t now={current_ticks:.1f}t "
                       f"gave_back={gave_back/peak_ticks:.0%} "
                       f"({'shape' if shape_params else 'tier'}={threshold:.0%})",
                pnl_ticks=(bar_close - pos.entry_price) / tick_size
                          if pos.side == 'long'
                          else (pos.entry_price - bar_close) / tick_size,
                bars_held=pos.bars_held,
            )

        return None

    def record_exit(self, vol_drop_pct: float, pnl_ticks: float, peak_ticks: float):
        """Record a giveback exit for adaptive threshold learning.

        Called after every giveback exit with:
          vol_drop_pct: current_volume / peak_volume during trade
          pnl_ticks: actual PnL in ticks at exit
          peak_ticks: peak MFE in ticks
        """
        capture_pct = pnl_ticks / peak_ticks if peak_ticks > 0 else 0
        self._recent_exits.append({
            'vol_drop': vol_drop_pct,
            'capture': capture_pct,
            'pnl': pnl_ticks,
        })
        self._adapt_count += 1

        if self._adapt_count >= self._adapt_interval and len(self._recent_exits) >= 10:
            self._adapt_count = 0
            self._recalibrate()

    def _recalibrate(self):
        """Adapt vol_drop_threshold from recent exits.

        Finds the volume drop level that maximizes average capture rate.
        Moves 20% toward optimal each recalibration (smooth adaptation).
        """
        if len(self._recent_exits) < 10:
            return

        exits = list(self._recent_exits)
        best_threshold = self._vol_drop_threshold
        best_score = -999

        for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            captures = []
            for e in exits:
                if e['vol_drop'] <= t:
                    captures.append(e['capture'])
                else:
                    captures.append(e['capture'] * 0.8)
            if captures:
                score = sum(captures) / len(captures)
                if score > best_score:
                    best_score = score
                    best_threshold = t

        self._vol_drop_threshold = (0.8 * self._vol_drop_threshold +
                                     0.2 * best_threshold)
