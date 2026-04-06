"""
NMP Blended Engine — one engine, tiered exits based on setup quality.

Entry: NMP base (z + vr) at 1m boundaries. At entry, classifies the tier:
  Tier 1 (CASCADE):   wick rejection + 1h z aligned → p_center exit
  Tier 2 (KILL_SHOT): wick rejection, no 1h         → p_center exit
  Tier 3 (BASE_NMP):  no wick → z exit, BUT at center check energy:
                       high energy → hold for overshoot (opposite extreme)
                       low energy  → exit at center

One position at a time. Exit rules follow the tier assigned at entry.
Each trade tagged with entry_tier and exit_reason for analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

TICK = 0.25
TV = 0.50

# NMP entry
ROCHE = 2.0
VR_ENTRY = 1.0

# Wick rejection thresholds
WICK_5M_MIN = 0.83
WICK_15M_MIN = 0.77

# Cascade: 1h z alignment
H1_Z_MIN = 1.0

# Tier 1-2 exit (kill shot / cascade)
P_CENTER_EXIT = 0.60

# Tier 3 exits (base NMP)
Z_EXIT = 0.5

# Regime shift early detection: DMI leading + VR confirming
DMI_AGAINST_THRESHOLD = 5.0   # |dmi_diff| opposing trade direction
VR_CONFIRMING = 0.8           # vr approaching trending (not yet 1.0)

# Tier 3 overshoot: energy check at center
ENERGY_BAR_RANGE_MIN = 100.0   # 5m_bar_range threshold for "high energy"
VELOCITY_EXHAUSTED = 0.3
Z_OPPOSITE_EXTREME = 1.0

# 79D absolute indices
_1M_OFFSET = 10
_Z = 0
_VR = 2
_5M_WICK_IDX = 68
_15M_WICK_IDX = 71
_1H_Z_IDX = 40
_1M_P_CENTER_IDX = 19
_1M_VELOCITY_IDX = 13
_5M_BAR_RANGE_IDX = 26  # 5m block starts at 20, bar_range is index 6
_1M_DMI_IDX = 11        # 1m_dmi_diff

APPROACH_BUFFER_SIZE = 10


class BlendedEngine:
    """One NMP engine with tiered exit physics."""

    def __init__(self):
        self.in_pos = False
        self.direction = None
        self.entry_price = 0.0
        self.entry_79d = None
        self.entry_1m = None
        self.entry_tier = None
        self.bars_held = 0
        self.peak_pnl = 0.0
        self.passed_center = False  # for tier 3 overshoot decision

        self._approach_buffer = []
        self._trade_path = []

        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0

    def on_state(self, state: Dict):
        self._bar_count += 1
        feat = state['features_79d']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price

        is_1m = (int(ts) % 60) < 5
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')

        # Read 1m state
        z = feat[_1M_OFFSET + _Z]
        vr = feat[_1M_OFFSET + _VR]

        # Approach buffer when flat
        if not self.in_pos:
            self._approach_buffer.append({
                'timestamp': ts, 'price': price,
                'features_79d': feat.copy(),
            })
            if len(self._approach_buffer) > APPROACH_BUFFER_SIZE:
                self._approach_buffer = self._approach_buffer[-APPROACH_BUFFER_SIZE:]

        # === EXIT CHECK ===
        if self.in_pos:
            self.bars_held += 1

            if self.direction == 'long':
                pnl = (price - self.entry_price) / TICK * TV
            else:
                pnl = (self.entry_price - price) / TICK * TV

            self.peak_pnl = max(self.peak_pnl, pnl)

            # Record path
            self._trade_path.append({
                'bar': self.bars_held, 'timestamp': ts, 'price': price,
                'pnl': pnl, 'peak_pnl': self.peak_pnl,
                'features_79d': feat.copy(),
            })

            # Exit logic — only at 1m boundaries
            if is_1m:
                exit_reason = self._check_exit(feat, z, vr, pnl)
                if exit_reason:
                    old_direction = self.direction
                    self._close_trade(price, ts, time_str, exit_reason, feat)

                    # FLIP on regime shift: close + immediately enter opposite
                    if exit_reason == 'regime_shift_early':
                        flip_dir = 'short' if old_direction == 'long' else 'long'
                        self._open_trade(flip_dir, price, ts, time_str, feat, 'REGIME_FLIP')

        # === ENTRY CHECK — 1m boundaries only ===
        if not self.in_pos and is_1m:
            if abs(z) > ROCHE and vr < VR_ENTRY:
                direction = 'short' if z > 0 else 'long'
                tier = self._classify_tier(feat, direction)
                self._open_trade(direction, price, ts, time_str, feat, tier)

    def _classify_tier(self, feat, direction):
        """Classify entry setup into tier based on conditions present."""
        wick_5m = feat[_5M_WICK_IDX]
        wick_15m = feat[_15M_WICK_IDX]
        h1_z = feat[_1H_Z_IDX]

        has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN

        h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                      (direction == 'short' and h1_z > H1_Z_MIN))

        if has_wick and h1_aligned:
            return 'CASCADE'
        elif has_wick:
            return 'KILL_SHOT'
        else:
            return 'BASE_NMP'

    def _check_exit(self, feat, z, vr, pnl):
        """Check exit based on entry tier."""
        # REGIME_FLIP trade: exit when regime normalizes
        if self.entry_tier == 'REGIME_FLIP':
            dmi = feat[_1M_DMI_IDX]
            # Exit when DMI supports our flip direction (trend confirmed and exhausting)
            dmi_supporting = ((self.direction == 'long' and dmi > DMI_AGAINST_THRESHOLD) or
                              (self.direction == 'short' and dmi < -DMI_AGAINST_THRESHOLD))
            vr_normalizing = vr < VR_CONFIRMING
            # Exit when regime shift is over (VR drops) or momentum exhausts
            if vr_normalizing:
                return 'regime_flip_vr_normal'
            # Also exit at p_center (reversion to new mean)
            p_center = feat[_1M_P_CENTER_IDX]
            if p_center > P_CENTER_EXIT:
                return 'regime_flip_center'
            return None

        if self.entry_tier in ('CASCADE', 'KILL_SHOT'):
            # Tier 1-2: exit at p_center
            p_center = feat[_1M_P_CENTER_IDX]
            if p_center > P_CENTER_EXIT:
                return f'{self.entry_tier.lower()}_center'
            return None

        # Tier 3: BASE_NMP — multi-phase exit
        p_center = feat[_1M_P_CENTER_IDX]
        velocity = feat[_1M_VELOCITY_IDX]
        bar_range_5m = feat[_5M_BAR_RANGE_IDX]

        # Phase 1: approaching center
        if not self.passed_center and p_center > P_CENTER_EXIT:
            self.passed_center = True

            # Energy check at center: high energy → hold for overshoot
            if bar_range_5m > ENERGY_BAR_RANGE_MIN:
                return None  # hold — overshoot likely

            # Low energy → exit at center (reversion done)
            return 'nmp_center_low_energy'

        # Phase 2: passed center, holding for overshoot
        if self.passed_center:
            entry_z_sign = 1.0 if self.entry_1m['z_se'] > 0 else -1.0
            current_z_sign = 1.0 if z > 0 else -1.0

            # Exit: reached opposite extreme
            if current_z_sign != entry_z_sign and abs(z) > Z_OPPOSITE_EXTREME:
                return 'nmp_opposite_extreme'

            # Exit: momentum exhausted after passing center
            if abs(velocity) < VELOCITY_EXHAUSTED:
                return 'nmp_momentum_exhausted'

            return None

        # Phase 0: not yet at center
        # Standard NMP mean exit
        if abs(z) < Z_EXIT:
            return 'nmp_mean_reached'

        # Early regime shift: DMI opposes + VR confirming (approaching trending)
        dmi = feat[_1M_DMI_IDX]
        dmi_against = ((self.direction == 'long' and dmi < -DMI_AGAINST_THRESHOLD) or
                       (self.direction == 'short' and dmi > DMI_AGAINST_THRESHOLD))
        vr_confirming = vr > VR_CONFIRMING

        if dmi_against and vr_confirming:
            return 'regime_shift_early'

        return None

    def _open_trade(self, direction, price, ts, time_str, feat, tier):
        self.in_pos = True
        self.direction = direction
        self.entry_price = price
        self.entry_79d = feat.copy()
        self.entry_1m = {
            'z_se': feat[_1M_OFFSET + _Z],
            'vr': feat[_1M_OFFSET + _VR],
        }
        self.entry_tier = tier
        self.bars_held = 0
        self.peak_pnl = 0.0
        self.passed_center = False
        self._entry_approach = list(self._approach_buffer)
        self._trade_path = []

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
            'timestamp': ts,
            'dir': self.direction,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'held': self.bars_held,
            'peak': self.peak_pnl,
            'entry_tier': self.entry_tier,
            'exit_reason': exit_reason,
            'entry_79d': self.entry_79d.tolist() if self.entry_79d is not None else [],
            'exit_79d': feat.tolist(),
            'approach': self._entry_approach,
            'path': self._trade_path.copy(),
        })

        self.in_pos = False
        self.direction = None
        self.entry_tier = None
        self._trade_path = []

    def force_close(self, reason='end_of_day'):
        if self.in_pos:
            ts = self._trade_path[-1]['timestamp'] if self._trade_path else 0
            feat = self._trade_path[-1]['features_79d'] if self._trade_path else self.entry_79d
            if feat is None:
                feat = np.zeros(79)
            time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M') if ts > 0 else '??:??'
            self._close_trade(self._last_price, ts, time_str, reason, feat)

    def reset(self):
        self.in_pos = False
        self.direction = None
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._trade_path = []
        self._approach_buffer = []
        self.entry_tier = None
        self.passed_center = False

    def summary(self) -> str:
        n = len(self.trades)
        if n == 0:
            return 'Blended: 0 trades'
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = sum(t['pnl'] for t in self.trades)
        tiers = {}
        for t in self.trades:
            tier = t['entry_tier']
            if tier not in tiers:
                tiers[tier] = 0
            tiers[tier] += 1
        tier_str = ' '.join(f'{k}={v}' for k, v in sorted(tiers.items()))
        return f'Blended: {n} trades | WR={wins/n*100:.0f}% | ${total:.0f} | {tier_str}'

    def get_full_trades(self) -> List[Dict]:
        return self.trades
