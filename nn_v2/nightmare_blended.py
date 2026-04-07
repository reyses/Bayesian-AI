"""
NMP Blended Engine — one engine, tiered exits + CNN direction flip.

Entry: NMP base (z + vr) at 1m boundaries. At entry, classifies the tier:
  Tier 1 (CASCADE):   wick rejection + 1h z aligned → p_center exit
  Tier 2 (KILL_SHOT): wick rejection, no 1h         → p_center exit
  Tier 3 (BASE_NMP):  no wick → CNN predicts direction (flip if counter)
                       exits: z/center/overshoot based on energy

One position at a time. Exit rules follow the tier assigned at entry.
Each trade tagged with entry_tier, exit_reason, and cnn_flipped for analysis.
"""
import os
import numpy as np
import pandas as pd
import torch
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

# Tier encoding for CNN
TIER_MAP = {'CASCADE': 2, 'KILL_SHOT': 1, 'BASE_NMP': 0}

# CNN model paths
CNN_FLIP_PATH = 'nn_v2/output/tree/cnn_flip.pt'
CNN_HOLD_PATH = 'nn_v2/output/tree/cnn_hold.pt'
CNN_RISK_PATH = 'nn_v2/output/tree/cnn_risk.pt'

# Grid layout for CNN
_N_CORE = 10
_N_HELPER = 3
_N_TFS = 6
_HELPER_START = _N_CORE * _N_TFS  # 60


def _feat_to_grid(feat_79d):
    """Reshape 79D to 6×13 grid for CNN."""
    grid = np.zeros((_N_TFS, _N_CORE + _N_HELPER), dtype=np.float32)
    for tf_idx in range(_N_TFS):
        grid[tf_idx, :_N_CORE] = feat_79d[tf_idx * _N_CORE:(tf_idx + 1) * _N_CORE]
        h_start = _HELPER_START + tf_idx * _N_HELPER
        grid[tf_idx, _N_CORE:_N_CORE + _N_HELPER] = feat_79d[h_start:h_start + _N_HELPER]
    return grid


class BlendedEngine:
    """One NMP engine with tiered exit physics + CNN direction flip."""

    def __init__(self, use_cnn=True):
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

        # CNN flip predictor for BASE_NMP trades
        self.use_cnn = use_cnn
        self._cnn_device = None

        # CNN flip (direction)
        self.cnn_flip = None
        self.cnn_flip_mean = None
        self.cnn_flip_std = None

        # CNN hold (exit timing)
        self.cnn_hold = None
        self.cnn_hold_mean = None
        self.cnn_hold_std = None

        # CNN risk (cut losers)
        self.cnn_risk = None
        self.cnn_risk_mean = None
        self.cnn_risk_std = None

        if use_cnn:
            self._cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_cnn_flip()
            self._load_cnn_hold()
            self._load_cnn_risk()

    def _load_cnn_flip(self):
        """Load CNN flip predictor."""
        if not os.path.exists(CNN_FLIP_PATH):
            return
        from nn_v2.cnn_flip import FlipCNN
        checkpoint = torch.load(CNN_FLIP_PATH, map_location='cpu', weights_only=False)
        self.cnn_flip = FlipCNN(use_path=False).to(self._cnn_device)
        try:
            self.cnn_flip.load_state_dict(checkpoint['model_state'], strict=False)
        except Exception:
            self.cnn_flip = FlipCNN(use_path=False).to(self._cnn_device)
        self.cnn_flip.eval()
        self.cnn_flip_mean = checkpoint.get('entry_mean', np.zeros((1, 6, 13)))
        self.cnn_flip_std = checkpoint.get('entry_std', np.ones((1, 6, 13)))
        print(f'  CNN flip loaded ({self._cnn_device})')

    def _load_cnn_hold(self):
        """Load CNN hold predictor."""
        if not os.path.exists(CNN_HOLD_PATH):
            return
        from nn_v2.cnn_hold import HoldCNN
        checkpoint = torch.load(CNN_HOLD_PATH, map_location='cpu', weights_only=False)
        self.cnn_hold = HoldCNN().to(self._cnn_device)
        self.cnn_hold.load_state_dict(checkpoint['model_state'])
        self.cnn_hold.eval()
        self.cnn_hold_mean = checkpoint.get('grid_mean', np.zeros((1, 6, 13)))
        self.cnn_hold_std = checkpoint.get('grid_std', np.ones((1, 6, 13)))
        print(f'  CNN hold loaded ({self._cnn_device})')

    def _load_cnn_risk(self):
        """Load CNN risk predictor."""
        if not os.path.exists(CNN_RISK_PATH):
            return
        from nn_v2.cnn_risk import RiskCNN
        checkpoint = torch.load(CNN_RISK_PATH, map_location='cpu', weights_only=False)
        self.cnn_risk = RiskCNN().to(self._cnn_device)
        self.cnn_risk.load_state_dict(checkpoint['model_state'])
        self.cnn_risk.eval()
        self.cnn_risk_mean = checkpoint.get('grid_mean', np.zeros((1, 6, 13)))
        self.cnn_risk_std = checkpoint.get('grid_std', np.ones((1, 6, 13)))
        print(f'  CNN risk loaded ({self._cnn_device})')

    def _cnn_predict_risk(self, feat_79d, bars_held, pnl, peak_pnl, direction, tier):
        """Predict RECOVER(1) or DEAD(0) when trade is negative."""
        if self.cnn_risk is None:
            return None

        grid = _feat_to_grid(feat_79d)
        grid = (grid - self.cnn_risk_mean[0]) / self.cnn_risk_std[0].clip(min=1e-8)

        grid_t = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self._cnn_device)

        bars_norm = bars_held / max(300, 1)  # approximate proportion
        pnl_norm = pnl / 50.0
        peak_norm = peak_pnl / 50.0
        depth = (peak_pnl - pnl) / 50.0
        dir_sign = 1.0 if direction == 'long' else -1.0
        tier_num = TIER_MAP.get(tier, 0)

        ctx_t = torch.FloatTensor([[bars_norm, pnl_norm, peak_norm, depth, dir_sign, float(tier_num)]]).to(self._cnn_device)

        with torch.no_grad():
            out = self.cnn_risk(grid_t, ctx_t)
            pred = out.argmax(dim=1).item()

        return pred  # 0=DEAD, 1=RECOVER

    def _cnn_predict_flip(self, feat_79d, tier_num):
        """Predict SAME(0) or COUNTER(1) from 79D at entry."""
        if self.cnn_flip is None:
            return 0  # default: SAME (no flip)

        grid = _feat_to_grid(feat_79d)
        grid = (grid - self.cnn_flip_mean[0]) / self.cnn_flip_std[0].clip(min=1e-8)

        entry_t = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self._cnn_device)
        tier_t = torch.FloatTensor([[tier_num]]).to(self._cnn_device)

        with torch.no_grad():
            out = self.cnn_flip(entry_t, tier=tier_t)
            pred = out.argmax(dim=1).item()

        return pred  # 0=SAME, 1=COUNTER

    def _cnn_predict_hold(self, feat_79d, bars_held, pnl, peak_pnl, direction, tier):
        """Predict HOLD(1) or EXIT(0) from current 79D + context."""
        if self.cnn_hold is None:
            return None  # no prediction, fall through to physics exit

        grid = _feat_to_grid(feat_79d)
        grid = (grid - self.cnn_hold_mean[0]) / self.cnn_hold_std[0].clip(min=1e-8)

        grid_t = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self._cnn_device)

        # Context: [bars_norm, pnl_norm, peak_norm, dir_sign, tier]
        # Note: training used proportional (bar/path_len), inference uses /500.
        # The v1 model is robust to this mismatch ($605/$563 results used /500).
        bars_norm = min(bars_held / 500.0, 1.0)
        pnl_norm = pnl / 50.0
        peak_norm = peak_pnl / 50.0
        dir_sign = 1.0 if direction == 'long' else -1.0
        tier_num = TIER_MAP.get(tier, 0)

        ctx_t = torch.FloatTensor([[bars_norm, pnl_norm, peak_norm, dir_sign, float(tier_num)]]).to(self._cnn_device)

        with torch.no_grad():
            out = self.cnn_hold(grid_t, ctx_t)
            pred = out.argmax(dim=1).item()

        return pred  # 0=EXIT, 1=HOLD

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
                # CASCADE and KILL_SHOT: proven physics exit (p_center)
                # BASE_NMP: CNN hold exit (trained on BASE_NMP regret)
                if self.entry_tier in ('CASCADE', 'KILL_SHOT'):
                    exit_reason = self._check_exit(feat, z, vr, pnl)
                    if exit_reason:
                        self._close_trade(price, ts, time_str, exit_reason, feat)
                elif self.entry_tier == 'BASE_NMP':
                    # BASE_NMP: two CNNs based on PnL state
                    if pnl < 0 and self.cnn_risk is not None:
                        # Negative: CNN risk decides recover or cut
                        risk_pred = self._cnn_predict_risk(
                            feat, self.bars_held, pnl, self.peak_pnl,
                            self.direction, self.entry_tier)
                        if risk_pred == 0:  # DEAD → cut the trade
                            self._close_trade(price, ts, time_str, 'cnn_risk_cut', feat)
                    elif self.cnn_hold is not None:
                        # Positive or no risk CNN: CNN hold decides
                        hold_pred = self._cnn_predict_hold(
                            feat, self.bars_held, pnl, self.peak_pnl,
                            self.direction, self.entry_tier)
                        if hold_pred == 0:  # CNN says EXIT
                            self._close_trade(price, ts, time_str, 'cnn_exit', feat)
                else:
                    # No CNN → physics exits for all tiers
                    exit_reason = self._check_exit(feat, z, vr, pnl)
                    if exit_reason:
                        self._close_trade(price, ts, time_str, exit_reason, feat)

        # === ENTRY CHECK — 1m boundaries only ===
        if not self.in_pos and is_1m:
            if abs(z) > ROCHE and vr < VR_ENTRY:
                direction = 'short' if z > 0 else 'long'
                tier = self._classify_tier(feat, direction)
                cnn_flipped = False

                # CNN flip for BASE_NMP trades
                if tier == 'BASE_NMP' and self.use_cnn:
                    tier_num = TIER_MAP.get(tier, 0)
                    pred = self._cnn_predict_flip(feat, tier_num)
                    if pred == 1:  # CNN says COUNTER → flip direction
                        direction = 'long' if direction == 'short' else 'short'
                        cnn_flipped = True

                self._open_trade(direction, price, ts, time_str, feat, tier,
                                 cnn_flipped=cnn_flipped)

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

        return None

    def _open_trade(self, direction, price, ts, time_str, feat, tier, cnn_flipped=False):
        self.in_pos = True
        self.direction = direction
        self.entry_price = price
        self.entry_79d = feat.copy()
        self.entry_1m = {
            'z_se': feat[_1M_OFFSET + _Z],
            'vr': feat[_1M_OFFSET + _VR],
        }
        self.entry_tier = tier
        self.cnn_flipped = cnn_flipped
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
            'cnn_flipped': getattr(self, 'cnn_flipped', False),
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
