"""
Nightmare Engine — trades the Nightmare Protocol on 79D states.

Receives 79D states from the SFE (live or test mode). Does NOT compute features.
The 79D is the contract — this engine doesn't care where it came from.

Entry: z_se extreme (|z| > ROCHE) + vr < 1.0 -> fade the z
Exit:  inverse NMP: |z_se| < 0.5 (mean reached) OR vr > 1.0 (regime flip)

Records the full trade path (79D at every bar during the trade) for tree+NN learning.

Usage (test mode — from pre-computed features):
    from training.sfe_ticker import FeatureTicker
    from training.nightmare import NightmareEngine

    nmp = NightmareEngine()
    for state in FeatureTicker('DATA/FEATURES_79D/2026_01_06.parquet',
                                price_file='DATA/ATLAS/1m/2026_01_06.parquet'):
        nmp.on_state(state)

    nmp.force_close()
    print(nmp.summary())

Usage (live mode — from aggregator):
    agg = Aggregator()
    nmp = NightmareEngine()

    def on_1m(tf, bar):
        if tf == '1m':
            state = sfe.compute_features(agg)  # SFE computes from aggregator
            nmp.on_state(state)

    agg.on_bar_close = on_1m
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

TICK = 0.25
TV = 0.50

# NMP constants
ROCHE = 2.0          # z_se threshold for entry
VR_ENTRY = 1.0       # variance_ratio must be below this for entry
Z_EXIT = 0.5         # z_se threshold for exit (mean reached)
VR_EXIT = 1.0        # variance_ratio above this = regime flip -> exit
MAX_DRAWDOWN = 50.0  # emergency exit: max $ loss per trade before forced close
APPROACH_BUFFER_SIZE = 10  # CNN 1 loads approach from feature files directly, not buffer

# 79D layout: 1m is index 1 in TF_ORDER (15s=0, 1m=1, 5m=2, ...)
# 10 core features per TF
_1M_OFFSET = 1 * 10
_5M_OFFSET = 2 * 10
_15M_OFFSET = 3 * 10
_1H_OFFSET = 4 * 10
_1D_OFFSET = 5 * 10

# Feature indices within each TF block
_Z = 0
_DMI = 1
_VR = 2
_VEL = 3
_ACCEL = 4
_VOL = 5
_RANGE = 6
_HURST = 7
_REV_PROB = 8
_P_CENTER = 9


def _read_tf(feat: np.ndarray, tf_offset: int) -> Dict:
    """Extract key features from a TF block in the 79D vector."""
    return {
        'z_se': float(feat[tf_offset + _Z]),
        'dmi_diff': float(feat[tf_offset + _DMI]),
        'vr': float(feat[tf_offset + _VR]),
        'velocity': float(feat[tf_offset + _VEL]),
        'acceleration': float(feat[tf_offset + _ACCEL]),
        'vol_rel': float(feat[tf_offset + _VOL]),
        'bar_range': float(feat[tf_offset + _RANGE]),
        'hurst': float(feat[tf_offset + _HURST]),
        'reversion_prob': float(feat[tf_offset + _REV_PROB]),
        'p_at_center': float(feat[tf_offset + _P_CENTER]),
    }


class NightmareEngine:
    """Trades the Nightmare Protocol. Receives 79D states. No feature computation."""

    def __init__(self):
        # Position state
        self.in_pos = False
        self.direction = None
        self.entry_price = 0.0
        self.entry_79d = None
        self.entry_1m = None
        self.bars_held = 0
        self.peak_pnl = 0.0

        # Approach buffer: rolling window of last N states before entry
        self._approach_buffer = []

        # Trade path: 79D at every bar during the trade (for tree+NN)
        self._trade_path = []

        # Trade log
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0

    def on_state(self, state: Dict):
        """Process one 79D state. Entry and exit decisions happen here.

        Decisions only at 1m boundaries (timestamp aligned to 60s).
        Between 1m closes, only records approach buffer and trade path.
        This prevents micro-noise entries/exits at sub-minute resolution.

        Args:
            state: dict with at minimum:
                'features': np.ndarray(79,)
                'price': float (current close price)
                'timestamp': float
                Optionally: 'bar_data', 'bar_idx', 'day'
        """
        self._bar_count += 1
        feat = state['features']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price

        # Check if this is a 1m boundary (decisions only on 1m closes)
        is_1m_boundary = (int(ts) % 60) < 5  # within 5s of a minute mark

        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')

        # Read 1m state from 79D
        s1m = _read_tf(feat, _1M_OFFSET)
        z = s1m['z_se']
        vr = s1m['vr']

        # === APPROACH BUFFER: always record when not in position ===
        if not self.in_pos:
            self._approach_buffer.append({
                'timestamp': ts,
                'price': price,
                'z_1m': z,
                'vr_1m': vr,
                'features': feat.copy(),
            })
            # Keep only last N bars
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

            # Record path (79D at every bar during trade — always, not just 1m)
            self._trade_path.append({
                'bar': self.bars_held,
                'timestamp': ts,
                'price': price,
                'pnl': pnl,
                'peak_pnl': self.peak_pnl,
                'z_1m': z,
                'vr_1m': vr,
                'features': feat.copy(),
            })

            # Inverse NMP exit — only on 1m boundaries
            if is_1m_boundary:
                exit_reason = None
                if abs(z) < Z_EXIT:
                    exit_reason = 'mean_reached'
                elif vr > VR_EXIT:
                    exit_reason = 'regime_flip'

                if exit_reason:
                    self._close_trade(price, ts, time_str, exit_reason, feat, s1m)

        # === ENTRY CHECK — only on 1m boundaries ===
        if not self.in_pos and is_1m_boundary:
            if abs(z) > ROCHE and vr < VR_ENTRY:
                direction = 'short' if z > 0 else 'long'
                self._open_trade(direction, price, ts, time_str, feat, s1m)

    def _open_trade(self, direction: str, price: float, ts: float,
                    time_str: str, feat: np.ndarray, s1m: Dict):
        """Enter a trade."""
        self.in_pos = True
        self.direction = direction
        self.entry_price = price
        self.entry_79d = feat.copy()
        self.entry_1m = s1m.copy()
        self.bars_held = 0
        self.peak_pnl = 0.0
        self._trade_path = []

        # Snapshot approach path (79D in bars leading up to entry)
        self._entry_approach = list(self._approach_buffer)

        # Record entry as first path point
        self._trade_path.append({
            'bar': 0,
            'timestamp': ts,
            'price': price,
            'pnl': 0.0,
            'peak_pnl': 0.0,
            'z_1m': s1m['z_se'],
            'vr_1m': s1m['vr'],
            'features': feat.copy(),
        })

    def _close_trade(self, price: float, ts: float, time_str: str,
                     exit_reason: str, feat: np.ndarray, s1m: Dict):
        """Exit a trade and record full context."""
        if self.direction == 'long':
            pnl = (price - self.entry_price) / TICK * TV
        else:
            pnl = (self.entry_price - price) / TICK * TV

        self.daily_pnl += pnl

        # Read all TFs at exit for full context
        s5m = _read_tf(feat, _5M_OFFSET)
        s15m = _read_tf(feat, _15M_OFFSET)
        s1h = _read_tf(feat, _1H_OFFSET)

        self.trades.append({
            'trade_id': len(self.trades),
            'time': time_str,
            'timestamp': ts,
            'dir': self.direction,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'exit': exit_reason,
            'held': self.bars_held,
            'peak': self.peak_pnl,
            # Entry state (1m)
            'entry_z': self.entry_1m['z_se'],
            'entry_vr': self.entry_1m['vr'],
            'entry_dmi': self.entry_1m['dmi_diff'],
            'entry_vol': self.entry_1m['vol_rel'],
            'entry_hurst': self.entry_1m['hurst'],
            # Exit state (1m)
            'exit_z': s1m['z_se'],
            'exit_vr': s1m['vr'],
            'exit_dmi': s1m['dmi_diff'],
            # Higher TF context at exit
            'exit_z_5m': s5m['z_se'],
            'exit_z_15m': s15m['z_se'],
            'exit_z_1h': s1h['z_se'],
            'exit_dmi_1h': s1h['dmi_diff'],
            # Full 79D at entry and exit (for tree+NN)
            'entry_79d': self.entry_79d.tolist(),
            'exit_79d': feat.tolist(),
            # Approach path (79D in bars before entry — how we got here)
            'approach': self._entry_approach.copy() if hasattr(self, '_entry_approach') else [],
            'approach_length': len(self._entry_approach) if hasattr(self, '_entry_approach') else 0,
            # Trade path (79D at every bar — the score sheet)
            'path': self._trade_path.copy(),
            'path_length': len(self._trade_path),
        })

        self.in_pos = False
        self.direction = None
        self._trade_path = []

    def force_close(self, reason: str = 'end_of_day'):
        """Force close at end of day."""
        if not self.in_pos:
            return

        price = self._last_price
        ts = self._trade_path[-1]['timestamp'] if self._trade_path else 0
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M') if ts > 0 else '??:??'
        # Use last known features
        feat = self._trade_path[-1]['features'] if self._trade_path else self.entry_79d
        if feat is None:
            feat = np.zeros(79)
        s1m = _read_tf(feat, _1M_OFFSET)
        self._close_trade(price, ts, time_str, reason, feat, s1m)

    def summary(self) -> str:
        """End-of-day summary."""
        n = len(self.trades)
        if n == 0:
            return f'NMP: 0 trades | $0.00'

        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        wr = wins / n * 100
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_held = np.mean([t['held'] for t in self.trades])

        lines = [
            f'NMP: {n} trades | WR={wr:.0f}% | PnL=${total_pnl:.2f} | avg_held={avg_held:.1f}',
            '',
        ]

        # Exit breakdown
        from collections import Counter
        exits = Counter(t['exit'] for t in self.trades)
        for ex, count in exits.most_common():
            sub = [t for t in self.trades if t['exit'] == ex]
            sub_wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            sub_pnl = sum(t['pnl'] for t in sub)
            avg = sub_pnl / len(sub)
            lines.append(f'  {ex:<20} {count:>3}  WR={sub_wr:>4.0f}%  ${sub_pnl:>8.2f}  ${avg:>6.2f}/tr')

        # Direction
        for d in ['long', 'short']:
            sub = [t for t in self.trades if t['dir'] == d]
            if sub:
                sub_wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
                sub_pnl = sum(t['pnl'] for t in sub)
                lines.append(f'  {d:<20} {len(sub):>3}  WR={sub_wr:>4.0f}%  ${sub_pnl:>8.2f}')

        return '\n'.join(lines)

    def get_trade_log(self) -> pd.DataFrame:
        """Get trades as DataFrame (without path arrays — for summary analysis)."""
        if not self.trades:
            return pd.DataFrame()
        flat = []
        for t in self.trades:
            row = {k: v for k, v in t.items()
                   if k not in ('entry_79d', 'exit_79d', 'path')}
            flat.append(row)
        return pd.DataFrame(flat)

    def get_full_trades(self) -> List[Dict]:
        """Get trades with full 79D paths (for tree+NN training)."""
        return self.trades

    def reset(self):
        """Reset for next day."""
        self.in_pos = False
        self.direction = None
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._trade_path = []
        self._approach_buffer = []
        self._entry_approach = []
