"""
DMI Smoothed Cross Flipper — simplest profitable engine.

Architecture:
  - 1m DMI diff (dmi_plus - dmi_minus), 3-bar moving average
  - Flip direction when smoothed DMI crosses zero
  - Repeating TP: bank 10 ticks, stay in trade, repeat
  - SL: 40 ticks from entry ends the trade
  - Re-enter on next DMI cross

No seeds, no K-NN, no normalization. Just DMI from SFE states.
"""
from dataclasses import dataclass, field
from collections import deque


# Tunable defaults (overridden by live_tuning.json)
DEFAULT_TP_TICKS = 10
DEFAULT_SL_TICKS = 40
DEFAULT_SMOOTH_WINDOW = 3


@dataclass
class FlipperResult:
    action: str = 'HOLD'       # ENTER, EXIT, FLIP, HOLD, TP_BANK
    direction: str = ''        # LONG, SHORT
    pnl_ticks: float = 0.0
    reason: str = ''
    tp_count: int = 0          # how many TPs banked this trade
    consensus: float = 0.0     # compatibility with PhysicsEngine interface
    matched_indices: list = field(default_factory=list)


class DmiFlipper:
    """DMI smoothed cross flipper with repeating TP."""

    def __init__(
        self,
        tp_ticks: float = DEFAULT_TP_TICKS,
        sl_ticks: float = DEFAULT_SL_TICKS,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    ):
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.smooth_window = smooth_window

        # State
        self._dmi_diffs = deque(maxlen=smooth_window + 1)
        self._in_trade = False
        self._trade_dir = ''
        self._entry_price = 0.0
        self._last_tp_price = 0.0
        self._tp_count = 0
        self._bar_count = 0

        # Stats
        self.stats = {
            'bars': 0, 'entries': 0, 'exits': 0,
            'flips': 0, 'tp_banks': 0, 'sl_exits': 0,
        }

    def on_bar(self, price: float, high: float, low: float,
               timestamp: float, state, volume: float = 0.0,
               nt8_dmi_plus: float = 0.0, nt8_dmi_minus: float = 0.0) -> FlipperResult:
        """Process one 1m bar. Returns FlipperResult."""
        self._bar_count += 1
        self.stats['bars'] += 1

        # Extract DMI — prefer NT8 native DI+/DI- (0-100) when available
        if nt8_dmi_plus > 0 or nt8_dmi_minus > 0:
            dmi_p = nt8_dmi_plus
            dmi_m = nt8_dmi_minus
        else:
            dmi_p = getattr(state, 'dmi_plus', 0.0)
            dmi_m = getattr(state, 'dmi_minus', 0.0)
        z = getattr(state, 'z_score', 0.0)
        dmi_diff = dmi_p - dmi_m
        self._dmi_diffs.append(dmi_diff)

        # Track DMI+, DMI-, z for divergence widening detection
        if not hasattr(self, '_dmi_p_hist'):
            self._dmi_p_hist = deque(maxlen=5)
            self._dmi_m_hist = deque(maxlen=5)
            self._z_hist = deque(maxlen=5)
            self._price_hist = deque(maxlen=30)
        self._dmi_p_hist.append(dmi_p)
        self._dmi_m_hist.append(dmi_m)
        self._price_hist.append(price)

        # Z using standard error instead of std: SE = std / sqrt(n)
        # Answers "is this deviation significant given sample size?"
        import numpy as _np
        if len(self._price_hist) >= 5:
            _prices = list(self._price_hist)
            _mean = sum(_prices) / len(_prices)
            _std = float(_np.std(_prices))
            _se = _std / (len(_prices) ** 0.5) if len(_prices) > 1 else _std
            z_se = (price - _mean) / _se if _se > 1e-8 else 0.0
        else:
            z_se = z  # fallback to SFE z during warmup
        self._z_hist.append(z_se)

        # Track volume
        if not hasattr(self, '_volumes'):
            self._volumes = deque(maxlen=30)
        self._volumes.append(volume)
        self._last_volume = volume
        self._avg_volume = sum(self._volumes) / len(self._volumes) if self._volumes else 1.0

        # Peak signature from research (98.8% recall):
        # DMI+ rising AND DMI- falling AND z rising
        self._divergence_signal = False
        if len(self._dmi_p_hist) >= 2 and len(self._z_hist) >= 2:
            _dp_rising = self._dmi_p_hist[-1] > self._dmi_p_hist[-2]
            _dm_falling = self._dmi_m_hist[-1] < self._dmi_m_hist[-2]
            _z_rising = self._z_hist[-1] > self._z_hist[-2]
            # Either direction: LONG signal or SHORT signal
            _dp_falling = self._dmi_p_hist[-1] < self._dmi_p_hist[-2]
            _dm_rising = self._dmi_m_hist[-1] > self._dmi_m_hist[-2]
            _z_falling = self._z_hist[-1] < self._z_hist[-2]
            self._divergence_signal = (
                (_dp_rising and _dm_falling and _z_rising) or   # LONG peak
                (_dp_falling and _dm_rising and _z_falling)     # SHORT peak
            )

        # Smoothed DMI diff (3-bar MA, or raw if not enough bars)
        n_dmi = len(self._dmi_diffs)
        if n_dmi < 2:
            return FlipperResult(reason=f'warmup {n_dmi}/2')

        recent = list(self._dmi_diffs)
        w = min(self.smooth_window, n_dmi - 1)  # use fewer bars if warming up
        smooth_now = sum(recent[-w:]) / w if w > 0 else recent[-1]
        smooth_prev = sum(recent[-(w + 1):-1]) / w if w > 0 else recent[-2]

        # --- IN TRADE: check SL and TP ---
        if self._in_trade:
            ref = self._last_tp_price if self._tp_count > 0 else self._entry_price

            if self._trade_dir == 'LONG':
                # SL check (from entry)
                sl_pnl = (low - self._entry_price) / 0.25
                if sl_pnl <= -self.sl_ticks:
                    total_pnl = -self.sl_ticks + (self._tp_count * self.tp_ticks)
                    self._in_trade = False
                    self.stats['exits'] += 1
                    self.stats['sl_exits'] += 1
                    return FlipperResult(
                        action='EXIT', direction=self._trade_dir,
                        pnl_ticks=total_pnl, tp_count=self._tp_count,
                        reason=f'SL after {self._tp_count} TPs (banked {self._tp_count * self.tp_ticks}t)',
                    )
                # TP check (from last TP level)
                tp_pnl = (high - ref) / 0.25
                if tp_pnl >= self.tp_ticks:
                    self._tp_count += 1
                    self._last_tp_price = ref + self.tp_ticks * 0.25
                    self.stats['tp_banks'] += 1
                    return FlipperResult(
                        action='TP_BANK', direction=self._trade_dir,
                        pnl_ticks=self._tp_count * self.tp_ticks,
                        tp_count=self._tp_count,
                        reason=f'TP #{self._tp_count} banked (+{self._tp_count * self.tp_ticks}t total)',
                    )
            else:  # SHORT
                sl_pnl = (self._entry_price - high) / 0.25
                if sl_pnl <= -self.sl_ticks:
                    total_pnl = -self.sl_ticks + (self._tp_count * self.tp_ticks)
                    self._in_trade = False
                    self.stats['exits'] += 1
                    self.stats['sl_exits'] += 1
                    return FlipperResult(
                        action='EXIT', direction=self._trade_dir,
                        pnl_ticks=total_pnl, tp_count=self._tp_count,
                        reason=f'SL after {self._tp_count} TPs (banked {self._tp_count * self.tp_ticks}t)',
                    )
                tp_pnl = (ref - low) / 0.25
                if tp_pnl >= self.tp_ticks:
                    self._tp_count += 1
                    self._last_tp_price = ref - self.tp_ticks * 0.25
                    self.stats['tp_banks'] += 1
                    return FlipperResult(
                        action='TP_BANK', direction=self._trade_dir,
                        pnl_ticks=self._tp_count * self.tp_ticks,
                        tp_count=self._tp_count,
                        reason=f'TP #{self._tp_count} banked (+{self._tp_count * self.tp_ticks}t total)',
                    )

        # --- DIVERGENCE-BASED ENTRY (not cross-based) ---
        # Enter/flip when dominant DMI starts FADING, not when it crosses zero.
        # Cross is too late — the move already happened.
        # Divergence = the approach to the cross, 10-20 bars early.

        _vol_ratio = self._last_volume / self._avg_volume if self._avg_volume > 0 else 1.0
        _vol_ok = self._last_volume >= self._avg_volume if self._avg_volume > 0 else True

        # Detect divergence: the gap between DMI+ and DMI- is NARROWING
        # DMI- dominant (negative dmi_diff) but gap shrinking = LONG setup
        # DMI+ dominant (positive dmi_diff) but gap shrinking = SHORT setup
        _diverge_long = False
        _diverge_short = False
        if len(self._dmi_diffs) >= 3:
            d0 = self._dmi_diffs[-1]   # current
            d1 = self._dmi_diffs[-2]   # previous
            d2 = self._dmi_diffs[-3]   # 2 bars ago
            # Gap narrowing toward zero from negative = sellers losing grip
            if d0 < 0 and d0 > d1 and d1 > d2:
                _diverge_long = True
            # Gap narrowing toward zero from positive = buyers losing grip
            if d0 > 0 and d0 < d1 and d1 < d2:
                _diverge_short = True

        # Combined signal: divergence + volume confirmation
        _signal_long = _diverge_long and (_vol_ok or self._divergence_signal)
        _signal_short = _diverge_short and (_vol_ok or self._divergence_signal)

        # No trade yet: enter on first signal or on initial DMI direction
        if not self._in_trade and self.stats['entries'] == 0:
            if _signal_long:
                new_dir = 'LONG'
            elif _signal_short:
                new_dir = 'SHORT'
            else:
                return FlipperResult(
                    reason=f'FLAT dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x divL={_diverge_long} divS={_diverge_short}',
                )
            self._in_trade = True
            self._trade_dir = new_dir
            self._entry_price = price
            self._last_tp_price = 0.0
            self._tp_count = 0
            self.stats['entries'] += 1
            return FlipperResult(
                action='ENTER', direction=new_dir,
                reason=f'ENTER {new_dir} divergence dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x',
            )

        # Detect flip signal: divergence in opposite direction while in trade
        _flip_signal = False
        new_dir = ''
        if self._in_trade:
            if self._trade_dir == 'LONG' and _signal_short:
                _flip_signal = True
                new_dir = 'SHORT'
            elif self._trade_dir == 'SHORT' and _signal_long:
                _flip_signal = True
                new_dir = 'LONG'

        if not _flip_signal:
            if self._in_trade:
                return FlipperResult(
                    reason=f'HOLD {self._trade_dir} TPs={self._tp_count} dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x divL={_diverge_long} divS={_diverge_short}',
                )
            # Not in trade, no signal
            if _signal_long or _signal_short:
                new_dir = 'LONG' if _signal_long else 'SHORT'
                self._in_trade = True
                self._trade_dir = new_dir
                self._entry_price = price
                self._last_tp_price = 0.0
                self._tp_count = 0
                self.stats['entries'] += 1
                return FlipperResult(
                    action='ENTER', direction=new_dir,
                    reason=f'ENTER {new_dir} divergence dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x',
                )
            return FlipperResult(reason=f'FLAT dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x')

        # FLIP: close existing + open new direction
        if self._trade_dir == 'LONG':
            pnl = (price - self._entry_price) / 0.25
        else:
            pnl = (self._entry_price - price) / 0.25

        self.stats['flips'] += 1
        self.stats['exits'] += 1
        self.stats['entries'] += 1

        old_dir = self._trade_dir
        old_tps = self._tp_count
        self._trade_dir = new_dir
        self._entry_price = price
        self._last_tp_price = 0.0
        self._tp_count = 0

        return FlipperResult(
            action='FLIP', direction=new_dir,
            pnl_ticks=pnl, tp_count=old_tps,
            reason=f'FLIP {old_dir}->{new_dir} pnl={pnl:+.0f}t (had {old_tps} TPs) divergence',
        )

    def check_sl_1s(self, price: float) -> FlipperResult:
        """Check SL at 1s resolution (called from live_engine _process_1s)."""
        if not self._in_trade:
            return FlipperResult()

        if self._trade_dir == 'LONG':
            sl_pnl = (price - self._entry_price) / 0.25
        else:
            sl_pnl = (self._entry_price - price) / 0.25

        if sl_pnl <= -self.sl_ticks:
            total_pnl = -self.sl_ticks + (self._tp_count * self.tp_ticks)
            self._in_trade = False
            self.stats['exits'] += 1
            self.stats['sl_exits'] += 1
            return FlipperResult(
                action='EXIT', direction=self._trade_dir,
                pnl_ticks=total_pnl, tp_count=self._tp_count,
                reason=f'SL_1s after {self._tp_count} TPs',
            )

        # Check TP at 1s too
        ref = self._last_tp_price if self._tp_count > 0 else self._entry_price
        if self._trade_dir == 'LONG':
            tp_pnl = (price - ref) / 0.25
        else:
            tp_pnl = (ref - price) / 0.25

        if tp_pnl >= self.tp_ticks:
            self._tp_count += 1
            if self._trade_dir == 'LONG':
                self._last_tp_price = ref + self.tp_ticks * 0.25
            else:
                self._last_tp_price = ref - self.tp_ticks * 0.25
            self.stats['tp_banks'] += 1
            return FlipperResult(
                action='TP_BANK', direction=self._trade_dir,
                pnl_ticks=self._tp_count * self.tp_ticks,
                tp_count=self._tp_count,
                reason=f'TP_1s #{self._tp_count}',
            )

        return FlipperResult()
