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
               timestamp: float, state) -> FlipperResult:
        """Process one 1m bar. Returns FlipperResult."""
        self._bar_count += 1
        self.stats['bars'] += 1

        # Extract DMI diff
        dmi_p = getattr(state, 'dmi_plus', 0.0)
        dmi_m = getattr(state, 'dmi_minus', 0.0)
        dmi_diff = dmi_p - dmi_m
        self._dmi_diffs.append(dmi_diff)

        # Need enough bars for smoothing
        if len(self._dmi_diffs) < self.smooth_window + 1:
            return FlipperResult(reason=f'warmup {len(self._dmi_diffs)}/{self.smooth_window + 1}')

        # Smoothed DMI diff (3-bar MA)
        recent = list(self._dmi_diffs)
        smooth_now = sum(recent[-self.smooth_window:]) / self.smooth_window
        smooth_prev = sum(recent[-(self.smooth_window + 1):-1]) / self.smooth_window

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

        if not cross_long and not cross_short:
            if self._in_trade:
                return FlipperResult(
                    reason=f'HOLD {self._trade_dir} TPs={self._tp_count} dmi={smooth_now:.1f}',
                )
            return FlipperResult(reason=f'FLAT dmi={smooth_now:.1f}')

        new_dir = 'LONG' if cross_long else 'SHORT'

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
