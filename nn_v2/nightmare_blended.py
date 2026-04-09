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

# Velocity threshold: separates calm (mean-reversion) from momentum (freight train)
VELOCITY_THRESHOLD = 50.0    # |1m_velocity| above this = momentum entry
FREIGHT_TRAIN_THRESHOLD = 100.0  # |1m_velocity| above this = always ride, never fade

# Bar range gate — disabled (low bar_range still profitable via volume)
BAR_RANGE_MIN = 0.0  # set to ~30 to activate (filters tight chop)

# Cascade: 1h z alignment
H1_Z_MIN = 1.0

# Tier 1-2 exit (kill shot / cascade)
P_CENTER_EXIT = 0.60
P_CENTER_EXIT_BARS_CASCADE = 3   # cascade: 3-bar confirmation (rare, high conviction)
P_CENTER_EXIT_BARS_KILLSHOT = 2  # kill shot: 2-bar confirmation (faster exit)

# Tier 3 exits (base NMP)
Z_EXIT = 0.5

# Hard stop — circuit breaker, overrides all CNNs
# Disabled for training (let regret see full paths). Live engine has its own hard stop.
HARD_STOP = -99999.0  # disabled — set to -150.0 for live

# Giveback stop — protect profits from round-tripping
GIVEBACK_MIN_PEAK = 99999.0   # disabled — set to ~$10 to activate
GIVEBACK_KEEP = 0.0           # disabled — set to ~0.40 to activate

# Regime shift early detection: DMI leading + VR confirming
DMI_AGAINST_THRESHOLD = 5.0   # |dmi_diff| opposing trade direction
VR_CONFIRMING = 0.8           # vr approaching trending (not yet 1.0)

# Tier 3 overshoot: energy check at center
ENERGY_BAR_RANGE_MIN = 100.0   # 5m_bar_range threshold for "high energy"
VELOCITY_EXHAUSTED = 0.3
Z_OPPOSITE_EXTREME = 1.0

# ── BASE_NMP Exit Physics (two modes) ──────────────────────────────
# FADE exit (same direction as NMP): entered against z, fading to mean
FADE_Z_EXIT = 0.5              # exit when |z| approaches zero
FADE_Z_EXIT_BARS = 3           # must hold below threshold for N consecutive bars
FADE_P_CENTER_CI = 0.60        # confidence that price is at the mean
FADE_P_CENTER_BARS = 3         # must hold above CI for N consecutive bars
FADE_OSCILLATION_DECAY = 0.40  # if oscillating: exit when amplitude < 40% of peak amplitude

# RIDE exit (flipped by CNN): entered with z, riding momentum
RIDE_VELOCITY_EXHAUSTED = 0.3  # exit when 1m velocity near zero (momentum dead)
RIDE_VR_TRENDING = 1.0         # exit when vr > 1.0 (regime shift, trend forming against us)
RIDE_REVERSION_HIGH = 0.95     # exit when reversion_prob very high (market wants to snap back)
RIDE_WICK_HIGH = 0.60          # exit when wick_ratio high (indecision = momentum lost)
RIDE_EXIT_BARS = 3             # consecutive bars for ride exit confirmation

# 79D absolute indices
_1M_OFFSET = 10
_Z = 0
_VR = 2
_5M_WICK_IDX = 68
_15M_WICK_IDX = 71
_1H_Z_IDX = 40
_1H_VELOCITY_IDX = 43  # 1h block starts at 40, velocity is core[3]
_1M_P_CENTER_IDX = 19
_1M_VELOCITY_IDX = 13
_5M_BAR_RANGE_IDX = 26  # 5m block starts at 20, bar_range is index 6
_1M_DMI_IDX = 11        # 1m_dmi_diff
_1M_WICK_IDX = 65       # 1m_wick_ratio (helper: 60+1*3+2)
_1M_REVERSION_IDX = 18  # 1m_reversion_prob (core: 10+8)

APPROACH_BUFFER_SIZE = 10  # CNN 1 loads approach from feature files directly, not buffer

# Tier encoding for CNN
TIER_MAP = {
    'CASCADE': 7, 'KILL_SHOT': 6,
    'FADE_CALM': 5, 'FADE_MOMENTUM': 4,
    'FADE_AGAINST': 3,   # fading z but 1h extreme against you
    'RIDE_CALM': 2, 'RIDE_MOMENTUM': 1,
    'RIDE_AGAINST': 0,   # CNN flipped but 1h opposes
    'FREIGHT_TRAIN': -1,
    'BASE_NMP': 0, 'MANUAL': 0,  # legacy compat
}

# 1h opposition threshold for FADE_AGAINST / RIDE_AGAINST
H1_AGAINST_Z_MIN = 1.5  # |1h_z| must be this extreme to count as "against"

# CNN model paths (default — training output dir)
CNN_FLIP_PATH = 'nn_v2/output/tree/cnn_flip.pt'
CNN_HOLD_PATH = 'nn_v2/output/tree/cnn_hold.pt'
CNN_RISK_PATH = 'nn_v2/output/tree/cnn_risk.pt'

# Live release dir (packaged by nn_v2/release.py)
LIVE_RELEASE_DIR = 'checkpoints/live_release'

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

    def __init__(self, use_cnn=True, release_dir=None):
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

        # CNN model directory — release_dir overrides defaults
        self._model_dir = release_dir
        if self._model_dir and not os.path.isdir(self._model_dir):
            print(f'  WARNING: release_dir {self._model_dir} not found, falling back to defaults')
            self._model_dir = None

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

    def _resolve_cnn_path(self, default_path: str) -> str:
        """Resolve CNN path: release_dir if set, else default."""
        if self._model_dir:
            return os.path.join(self._model_dir, os.path.basename(default_path))
        return default_path

    def _load_cnn_flip(self):
        """Load CNN flip predictor."""
        path = self._resolve_cnn_path(CNN_FLIP_PATH)
        if not os.path.exists(path):
            return
        from nn_v2.cnn_flip import FlipCNN
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
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
        path = self._resolve_cnn_path(CNN_HOLD_PATH)
        if not os.path.exists(path):
            return
        from nn_v2.cnn_hold import HoldCNN
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.cnn_hold = HoldCNN().to(self._cnn_device)
        self.cnn_hold.load_state_dict(checkpoint['model_state'])
        self.cnn_hold.eval()
        self.cnn_hold_mean = checkpoint.get('grid_mean', np.zeros((1, 6, 13)))
        self.cnn_hold_std = checkpoint.get('grid_std', np.ones((1, 6, 13)))
        print(f'  CNN hold loaded ({self._cnn_device})')

    def _load_cnn_risk(self):
        """Load CNN risk predictor."""
        path = self._resolve_cnn_path(CNN_RISK_PATH)
        if not os.path.exists(path):
            return
        from nn_v2.cnn_risk import RiskCNN
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
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

            # Update oscillation tracker
            curr_z_sign = 1.0 if z > 0 else -1.0
            if curr_z_sign != self._z_sign:
                self._zero_crossings += 1
                self._z_sign = curr_z_sign
                # New half-cycle: amplitude = distance from last extreme to zero
                self._current_amplitude = max(self._z_peak, self._z_trough)
                self._peak_amplitude = max(self._peak_amplitude, self._current_amplitude)
                self._z_peak = abs(z)
                self._z_trough = abs(z)
            else:
                self._z_peak = max(self._z_peak, abs(z))
                self._z_trough = min(self._z_trough, abs(z))

            # Record path
            self._trade_path.append({
                'bar': self.bars_held, 'timestamp': ts, 'price': price,
                'pnl': pnl, 'peak_pnl': self.peak_pnl,
                'features_79d': feat.copy(),
            })

            # Hard stop — fires every bar, overrides everything
            if pnl <= HARD_STOP:
                self._close_trade(price, ts, time_str, 'hard_stop', feat)

            # Exit logic — only at 1m boundaries
            elif is_1m:
                # CASCADE and KILL_SHOT: proven physics exit (p_center)
                # BASE_NMP: CNN hold exit (trained on BASE_NMP regret)
                if self.entry_tier in ('CASCADE', 'KILL_SHOT'):
                    exit_reason = self._check_exit(feat, z, vr, pnl)
                    if exit_reason:
                        self._close_trade(price, ts, time_str, exit_reason, feat)
                elif self.entry_tier in ('FADE_CALM', 'FADE_MOMENTUM', 'FADE_AGAINST',
                                        'RIDE_CALM', 'RIDE_MOMENTUM',
                                        'RIDE_AGAINST', 'FREIGHT_TRAIN',
                                        'BASE_NMP', 'MANUAL'):
                    # Giveback stop: if peak was meaningful and we gave back too much, exit
                    if self.peak_pnl >= GIVEBACK_MIN_PEAK and pnl < self.peak_pnl * GIVEBACK_KEEP:
                        self._close_trade(price, ts, time_str, 'giveback_stop', feat)
                    # CNN exit layer
                    elif pnl < 0 and self.cnn_risk is not None:
                        # Negative: CNN risk decides recover or cut
                        risk_pred = self._cnn_predict_risk(
                            feat, self.bars_held, pnl, self.peak_pnl,
                            self.direction, self.entry_tier)
                        if risk_pred == 0:  # DEAD -> cut the trade
                            self._close_trade(price, ts, time_str, 'cnn_risk_cut', feat)
                    elif self.cnn_hold is not None:
                        # Positive: CNN hold decides
                        hold_pred = self._cnn_predict_hold(
                            feat, self.bars_held, pnl, self.peak_pnl,
                            self.direction, self.entry_tier)
                        if hold_pred == 0:  # CNN says EXIT
                            self._close_trade(price, ts, time_str, 'cnn_exit', feat)
                    else:
                        # No CNN loaded — fall back to physics exits
                        exit_reason = self._check_exit(feat, z, vr, pnl)
                        if exit_reason:
                            self._close_trade(price, ts, time_str, exit_reason, feat)
                else:
                    # No CNN → physics exits for all tiers
                    exit_reason = self._check_exit(feat, z, vr, pnl)
                    if exit_reason:
                        self._close_trade(price, ts, time_str, exit_reason, feat)

        # === ENTRY CHECK — 1m boundaries only ===
        if not self.in_pos and is_1m:
            if abs(z) > ROCHE and vr < VR_ENTRY:
                # Bar range gate (disabled by default)
                if feat[_5M_BAR_RANGE_IDX] < BAR_RANGE_MIN:
                    return  # skip — too tight

                # Classify the full ExNMP tier + direction
                direction, tier, cnn_flipped = self._classify_full_tier(feat, z)

                self._open_trade(direction, price, ts, time_str, feat, tier,
                                 cnn_flipped=cnn_flipped)

    def _classify_full_tier(self, feat, z):
        """Classify entry into one of 8 ExNMP tiers.

        Returns (direction, tier, cnn_flipped).

        Tier priority (waterfall):
          1. CASCADE:        wick rejection + 1h z aligned
          2. KILL_SHOT:      wick rejection, no 1h alignment
          3. FREIGHT_TRAIN:  |velocity| > 100, ride the momentum
          4. FADE_MOMENTUM:  |velocity| > 50, fade z (no flip)
          5. FADE_CALM:      |velocity| < 50, fade z (no flip)
          -- CNN flip applied below this line --
          6. RIDE_AGAINST:   CNN flipped + 1h velocity opposes
          7. RIDE_MOMENTUM:  CNN flipped + |velocity| > 50
          8. RIDE_CALM:      CNN flipped + |velocity| < 50
        """
        # NMP default direction: fade the z
        direction = 'short' if z > 0 else 'long'
        cnn_flipped = False

        # Read conditions
        wick_5m = feat[_5M_WICK_IDX]
        wick_15m = feat[_15M_WICK_IDX]
        h1_z = feat[_1H_Z_IDX]
        velocity = feat[_1M_VELOCITY_IDX]
        h1_vel = feat[_1H_VELOCITY_IDX]
        abs_vel = abs(velocity)

        has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN
        h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                      (direction == 'short' and h1_z > H1_Z_MIN))

        # Tier 1: CASCADE — wick + 1h aligned
        if has_wick and h1_aligned:
            return direction, 'CASCADE', False

        # Tier 2: KILL_SHOT — wick, no 1h
        if has_wick:
            return direction, 'KILL_SHOT', False

        # Tier 3: FREIGHT_TRAIN — extreme velocity, always ride
        if abs_vel >= FREIGHT_TRAIN_THRESHOLD:
            direction = 'long' if velocity > 0 else 'short'
            return direction, 'FREIGHT_TRAIN', True

        # Check if 1h z is extreme against the fade direction
        h1_against_fade = ((direction == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                           (direction == 'short' and h1_z < -H1_AGAINST_Z_MIN))

        # FADE_AGAINST: 1h_z extreme against fade — keep fading (don't follow 1h)
        if h1_against_fade:
            return direction, 'FADE_AGAINST', False

        # RIDE_AGAINST: 1h_vel opposes fade direction (milder than z, but velocity confirms)
        h1_vel_against = ((direction == 'long' and h1_vel < -H1_AGAINST_Z_MIN) or
                          (direction == 'short' and h1_vel > H1_AGAINST_Z_MIN))
        if h1_vel_against:
            direction = 'long' if h1_vel > 0 else 'short'
            return direction, 'RIDE_AGAINST', False

        # CNN predicts SAME/COUNTER (1h is not opposing, physics tiers exhausted)
        if self.use_cnn and self.cnn_flip is not None:
            tier_num = TIER_MAP.get('FADE_CALM', 0)
            pred = self._cnn_predict_flip(feat, tier_num)

            if pred == 1:  # COUNTER — flip direction, ride momentum
                direction = 'long' if direction == 'short' else 'short'

                # RIDE_MOMENTUM
                if abs_vel >= VELOCITY_THRESHOLD:
                    return direction, 'RIDE_MOMENTUM', True

                # RIDE_CALM
                return direction, 'RIDE_CALM', True

        # FADE_MOMENTUM
        if abs_vel >= VELOCITY_THRESHOLD:
            return direction, 'FADE_MOMENTUM', False

        # FADE_CALM (default)
        return direction, 'FADE_CALM', False

    def _check_exit(self, feat, z, vr, pnl):
        """Check exit based on entry tier."""
        if self.entry_tier in ('CASCADE', 'KILL_SHOT'):
            # Tier 1-2: exit at p_center (per-tier bar confirmation)
            p_center = feat[_1M_P_CENTER_IDX]
            if p_center > P_CENTER_EXIT:
                self._tier_p_center_bars += 1
            else:
                self._tier_p_center_bars = 0
            required_bars = (P_CENTER_EXIT_BARS_CASCADE if self.entry_tier == 'CASCADE'
                             else P_CENTER_EXIT_BARS_KILLSHOT)
            if self._tier_p_center_bars >= required_bars:
                return f'{self.entry_tier.lower()}_center'
            return None

        # Tier 3: BASE_NMP — two exit modes based on trade type
        p_center = feat[_1M_P_CENTER_IDX]
        velocity = feat[_1M_VELOCITY_IDX]
        wick = feat[_1M_WICK_IDX]
        reversion = feat[_1M_REVERSION_IDX]

        if not getattr(self, 'cnn_flipped', False):
            # ── FADE MODE: entered against z, fading toward mean ──
            # 54% never cross zero, 26% oscillate

            # Track consecutive bars for exit confirmation
            if p_center > FADE_P_CENTER_CI:
                self._p_center_bars += 1
            else:
                self._p_center_bars = 0

            if abs(z) < FADE_Z_EXIT:
                self._z_near_zero_bars += 1
            else:
                self._z_near_zero_bars = 0

            # Phase 0: approaching mean
            if self._zero_crossings == 0:
                if self._z_near_zero_bars >= FADE_Z_EXIT_BARS:
                    return 'fade_mean_reached'
                if self._p_center_bars >= FADE_P_CENTER_BARS:
                    return 'fade_p_center'
                return None

            # Phase 1: crossed zero at least once — oscillation mode
            # Exit when amplitude decays (oscillation dying)
            if (self._peak_amplitude > 0 and
                    self._current_amplitude < self._peak_amplitude * FADE_OSCILLATION_DECAY):
                # Oscillation energy decayed — exit on favorable z
                entry_z_sign = 1.0 if self.entry_1m['z_se'] > 0 else -1.0
                # Favorable = z opposite to entry (we profited from the fade)
                z_favorable = (z > 0) != (entry_z_sign > 0)
                if z_favorable or self._zero_crossings >= 3:
                    return 'fade_oscillation_decay'

            # Safety: if oscillating and sustained at center, take the profit
            if self._zero_crossings >= 2 and self._p_center_bars >= FADE_P_CENTER_BARS:
                return 'fade_oscillation_center'

            return None

        else:
            # ── RIDE MODE: flipped by CNN, riding with z ──
            # 69% never cross zero, momentum trades

            # Track consecutive bars for each condition
            if abs(velocity) < RIDE_VELOCITY_EXHAUSTED:
                self._ride_vel_bars += 1
            else:
                self._ride_vel_bars = 0

            if vr > RIDE_VR_TRENDING:
                self._ride_vr_bars += 1
            else:
                self._ride_vr_bars = 0

            if reversion > RIDE_REVERSION_HIGH and wick > RIDE_WICK_HIGH:
                self._ride_rev_wick_bars += 1
            else:
                self._ride_rev_wick_bars = 0

            # Exit when momentum exhausted (sustained)
            if self._ride_vel_bars >= RIDE_EXIT_BARS:
                return 'ride_velocity_exhausted'

            # Exit when regime shifts (sustained)
            if self._ride_vr_bars >= RIDE_EXIT_BARS:
                return 'ride_regime_shift'

            # Exit when market wants to snap back (sustained)
            if self._ride_rev_wick_bars >= RIDE_EXIT_BARS:
                return 'ride_reversion_wick'

            return None

    def inject_manual_trade(self, direction: str, price: float, ts: float, feat):
        """Open a trade from external trigger (dashboard button). Gets full CNN management."""
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')
        self._open_trade(direction, price, ts, time_str, feat, 'MANUAL')

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

        # Consecutive bar counters for exit confirmation
        self._p_center_bars = 0
        self._z_near_zero_bars = 0
        self._tier_p_center_bars = 0   # CASCADE/KILL_SHOT p_center
        self._ride_vel_bars = 0        # RIDE velocity exhausted
        self._ride_vr_bars = 0         # RIDE regime shift
        self._ride_rev_wick_bars = 0   # RIDE reversion + wick

        # Oscillation tracking (for FADE exit mode)
        self._z_sign = 1.0 if feat[_1M_OFFSET + _Z] > 0 else -1.0
        self._zero_crossings = 0
        self._z_peak = abs(feat[_1M_OFFSET + _Z])  # entry |z| = initial amplitude
        self._z_trough = abs(feat[_1M_OFFSET + _Z])
        self._peak_amplitude = abs(feat[_1M_OFFSET + _Z])  # tracks max oscillation swing
        self._current_amplitude = abs(feat[_1M_OFFSET + _Z])

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
