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

        # --- DMI CROSS DETECTION ---
        cross_long = smooth_prev < 0 and smooth_now > 0
        cross_short = smooth_prev > 0 and smooth_now < 0

        # Entry quality from peak research:
        # 1. Volume >= average (peaks have 1.3x avg volume)
        # 2. DMI divergence widening (98.8% of peaks)
        _vol_ok = self._last_volume >= self._avg_volume if self._avg_volume > 0 else True
        _vol_ratio = self._last_volume / self._avg_volume if self._avg_volume > 0 else 1.0
        _quality_ok = _vol_ok or self._divergence_signal  # either confirms the move

        # No trade yet and no cross: enter based on current DMI direction (once)
        if not self._in_trade and not cross_long and not cross_short and self.stats['entries'] == 0:
            if not _quality_ok:
                return FlipperResult(reason=f'FLAT dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x div={self._divergence_signal} (low quality)')
            new_dir = 'LONG' if smooth_now > 0 else 'SHORT'
            self._in_trade = True
            self._trade_dir = new_dir
            self._entry_price = price
            self._last_tp_price = 0.0
            self._tp_count = 0
            self.stats['entries'] += 1
            return FlipperResult(
                action='ENTER', direction=new_dir,
                reason=f'ENTER {new_dir} initial dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x div={self._divergence_signal}',
            )

        if not cross_long and not cross_short:
            if self._in_trade:
                return FlipperResult(
                    reason=f'HOLD {self._trade_dir} TPs={self._tp_count} dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x',
                )
            return FlipperResult(reason=f'FLAT dmi={smooth_now:.1f} vol={_vol_ratio:.1f}x')

        new_dir = 'LONG' if cross_long else 'SHORT'

        # Quality gate on flips: skip low-volume + no-divergence crosses (fakeouts)
        if not _quality_ok:
            if self._in_trade:
                return FlipperResult(
                    reason=f'HOLD {self._trade_dir} cross_blocked vol={_vol_ratio:.1f}x div={self._divergence_signal}',
                )
            return FlipperResult(reason=f'SKIP cross vol={_vol_ratio:.1f}x div={self._divergence_signal}')

        # FLIP: close existing + open new
        if self._in_trade and new_dir != self._trade_dir:
            if self._trade_dir == 'LONG':
                pnl = (price - self._entry_price) / 0.25
            else:
                pnl = (self._entry_price - price) / 0.25

            self.stats['flips'] += 1
            self.stats['exits'] += 1
            self.stats['entries'] += 1

            # Reset for new trade
            old_dir = self._trade_dir
            old_tps = self._tp_count
            self._trade_dir = new_dir
            self._entry_price = price
            self._last_tp_price = 0.0
            self._tp_count = 0

            return FlipperResult(
                action='FLIP', direction=new_dir,
                pnl_ticks=pnl, tp_count=old_tps,
                reason=f'FLIP {old_dir}->{new_dir} pnl={pnl:+.0f}t (had {old_tps} TPs)',
            )

        # ENTER: new trade
        if not self._in_trade:
            self._in_trade = True
            self._trade_dir = new_dir
            self._entry_price = price
            self._last_tp_price = 0.0
            self._tp_count = 0
            self.stats['entries'] += 1

            return FlipperResult(
                action='ENTER', direction=new_dir,
                reason=f'ENTER {new_dir} dmi_cross={smooth_now:.1f}',
            )

        return FlipperResult(reason=f'HOLD {self._trade_dir} dmi={smooth_now:.1f}')

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
