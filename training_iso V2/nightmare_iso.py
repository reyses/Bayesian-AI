"""
Isolated NightmareEngine — only non-NMP entries.

REGIME_FLIP, EXHAUSTION_BAR, ABSORPTION with their own exits.
No NMP, no CASCADE, no KILL_SHOT. Separate CNN training.
"""
import numpy as np
from typing import Dict
from datetime import datetime

TICK = 0.25
TV = 0.50

# Entry conditions
REGIME_VR_MAX = 0.35
REGIME_HURST_MAX = 0.45
EXHAUST_BAR_RANGE_MIN = 80.0
EXHAUST_ACCEL_MIN = 2.0
ABSORB_VOL_MIN = 1.5
ABSORB_RANGE_MAX = 20.0
ABSORB_WICK_MIN = 0.50

# All require vr < 1.0 and |z| < 2.0 (non-NMP territory)
VR_ENTRY = 1.0
Z_MAX = 2.0

# Feature indices
_1M_OFFSET = 10
_Z = 0
_VR = 2
_VELOCITY = 3
_ACCEL = 4
_VOL_REL = 5
_BAR_RANGE = 6
_HURST = 7
_WICK = 65  # 1m_wick_ratio (helper)
_5M_VELOCITY = 23
_5M_ACCEL = 24

TIER_MAP = {
    'REGIME_FLIP': 0,
    'EXHAUSTION_BAR': 1,
    'ABSORPTION': 2,
}


class IsoEngine:
    """Trades only non-NMP entries with isolated CNN."""

    def __init__(self):
        self.in_pos = False
        self.direction = None
        self.entry_price = 0.0
        self.entry_79d = None
        self.entry_tier = None
        self.bars_held = 0
        self.peak_pnl = 0.0
        self._trade_path = []
        self._approach_buffer = []
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0

    def on_state(self, state: Dict):
        self._bar_count += 1
        feat = state['features']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price

        is_1m = (int(ts) % 60) < 5
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')

        z = feat[_1M_OFFSET + _Z]
        vr = feat[_1M_OFFSET + _VR]

        # Approach buffer
        if not self.in_pos:
            self._approach_buffer.append({
                'timestamp': ts, 'price': price,
                'features': feat.copy(),
            })
            if len(self._approach_buffer) > 10:
                self._approach_buffer = self._approach_buffer[-10:]

        # === EXIT ===
        if self.in_pos:
            self.bars_held += 1
            if self.direction == 'long':
                pnl = (price - self.entry_price) / TICK * TV
            else:
                pnl = (self.entry_price - price) / TICK * TV
            self.peak_pnl = max(self.peak_pnl, pnl)

            self._trade_path.append({
                'bar': self.bars_held, 'timestamp': ts, 'price': price,
                'pnl': pnl, 'peak_pnl': self.peak_pnl,
                'features': feat.copy(),
            })

            if is_1m:
                exit_reason = self._check_exit(feat, z, vr, pnl)
                if exit_reason:
                    self._close_trade(price, ts, time_str, exit_reason, feat)

        # === ENTRY (1m boundaries, non-NMP zone) ===
        if not self.in_pos and is_1m:
            if abs(z) <= Z_MAX and vr < VR_ENTRY:
                hurst = feat[_1M_OFFSET + _HURST]
                v5_accel = feat[_5M_ACCEL]
                v5 = abs(feat[_5M_VELOCITY])
                v1 = abs(feat[_1M_OFFSET + _VELOCITY])
                bar_range = feat[_1M_OFFSET + _BAR_RANGE]
                accel = feat[_1M_OFFSET + _ACCEL]
                vol_rel = feat[_1M_OFFSET + _VOL_REL]
                wick = feat[_WICK]

                # REGIME_FLIP
                if vr < REGIME_VR_MAX and hurst < REGIME_HURST_MAX:
                    direction = 'short' if z > 0 else 'long'
                    self._open_trade(direction, price, ts, time_str, feat, 'REGIME_FLIP')

                # EXHAUSTION_BAR
                elif (bar_range > EXHAUST_BAR_RANGE_MIN and
                      abs(accel) > EXHAUST_ACCEL_MIN and
                      accel * feat[_1M_OFFSET + _VELOCITY] < 0):
                    direction = 'short' if feat[_1M_OFFSET + _VELOCITY] > 0 else 'long'
                    self._open_trade(direction, price, ts, time_str, feat, 'EXHAUSTION_BAR')

                # ABSORPTION
                elif (vol_rel > ABSORB_VOL_MIN and
                      bar_range < ABSORB_RANGE_MAX and
                      wick > ABSORB_WICK_MIN):
                    direction = 'short' if z > 0 else 'long'
                    self._open_trade(direction, price, ts, time_str, feat, 'ABSORPTION')

    def _check_exit(self, feat, z, vr, pnl):
        if self.entry_tier == 'REGIME_FLIP':
            hurst = feat[_1M_OFFSET + _HURST]
            if vr > 0.7:
                return 'regime_vr_rising'
            if hurst > 0.55:
                return 'regime_hurst_rising'
            if abs(z) < 0.3:
                return 'regime_mean_reached'
            return None

        if self.entry_tier == 'EXHAUSTION_BAR':
            bar_range = feat[_1M_OFFSET + _BAR_RANGE]
            vel = abs(feat[_1M_OFFSET + _VELOCITY])
            if bar_range < 30:
                return 'exhaust_range_compressed'
            if vel < 0.3:
                return 'exhaust_velocity_dead'
            return None

        if self.entry_tier == 'ABSORPTION':
            vol = feat[_1M_OFFSET + _VOL_REL]
            bar_range = feat[_1M_OFFSET + _BAR_RANGE]
            wick = feat[_WICK]
            if vol < 0.5:
                return 'absorb_volume_died'
            if bar_range > 50:
                return 'absorb_range_expanded'
            if wick < 0.25:
                return 'absorb_wicks_gone'
            return None

        return None

    def _open_trade(self, direction, price, ts, time_str, feat, tier):
        self.in_pos = True
        self.direction = direction
        self.entry_price = price
        self.entry_79d = feat.copy()
        self.entry_tier = tier
        self.bars_held = 0
        self.peak_pnl = 0.0
        self._trade_path = []
        self._entry_approach = list(self._approach_buffer)
        self._trade_path.append({
            'bar': 0, 'timestamp': ts, 'price': price,
            'pnl': 0.0, 'peak_pnl': 0.0,
            'features': feat.copy(),
        })

    def _close_trade(self, price, ts, time_str, exit_reason, feat):
        if not self.in_pos:
            return
        if self.direction == 'long':
            pnl = (price - self.entry_price) / TICK * TV
        else:
            pnl = (self.entry_price - price) / TICK * TV
        self.daily_pnl += pnl

        self.trades.append({
            'trade_id': len(self.trades),
            'time': time_str,
            'timestamp': self._trade_path[0]['timestamp'] if self._trade_path else ts,
            'dir': self.direction,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'held': self.bars_held,
            'peak': self.peak_pnl,
            'entry_tier': self.entry_tier,
            'exit_reason': exit_reason,
            'entry_79d': self.entry_79d.tolist() if hasattr(self.entry_79d, 'tolist') else list(self.entry_79d),
            'exit_79d': feat.tolist() if hasattr(feat, 'tolist') else list(feat),
            'approach': self._entry_approach,
            'path': self._trade_path,
        })
        self.in_pos = False
        self.direction = None

    def force_close(self, reason='end_of_day'):
        if self.in_pos:
            ts = self._trade_path[-1]['timestamp'] if self._trade_path else 0
            feat = self._trade_path[-1]['features'] if self._trade_path else self.entry_79d
            if feat is None:
                feat = np.zeros(79)
            time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M') if ts > 0 else '??:??'
            self._close_trade(self._last_price, ts, time_str, reason, feat)

    def reset(self):
        self.in_pos = False
        self.direction = None
        self.entry_price = 0.0
        self.bars_held = 0
        self.peak_pnl = 0.0
        self.daily_pnl = 0.0
        self._trade_path = []
        self._approach_buffer = []

    def get_full_trades(self):
        return self.trades

    def summary(self):
        n = len(self.trades)
        if n == 0:
            return 'No trades'
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        return f'{n} trades | WR={wins/n*100:.0f}% | ${self.daily_pnl:.0f}'
