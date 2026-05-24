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

def _wick_ratio(body, bar_range):
    return 1.0 - abs(body) / max(bar_range, 1e-8)

TICK = 0.25
TV = 0.50

# NMP entry
ROCHE = 1.87
VR_ENTRY = 1.0

# PEAK entry — DISABLED (DMI extreme is late signal, not the entry)
# Regret shows optimal entry is 60 bars earlier at vr drop + 5m deceleration
# Replaced by REGIME_FLIP and MTF_EXHAUSTION below
PEAK_DMI_MIN = 99999.0   # disabled

# REGIME_FLIP entry — variance_ratio dropping = regime changing
# Fires when |z| < ROCHE (NMP wouldn't trigger)
REGIME_VR_DROP = 0.15     # vr dropped by at least this from recent high
REGIME_VR_MAX = 0.35       # low vr = mean-reverting regime
REGIME_HURST_MAX = 0.45    # low hurst = anti-persistent

# MTF_EXHAUSTION entry — DISABLED (triggers 7862 trades at 21% WR = -$142K)
# Too loose: 5m deceleration happens constantly
MTF_5M_VEL_MIN = 30.0      # 5m |velocity| must be above this
MTF_5M_DECEL = 0.50
MTF_1M_VEL_ALIVE = 10.0    # 1m |velocity| must show life

# EXHAUSTION_BAR entry — bar_range climax + velocity decelerating
EXHAUST_BAR_RANGE_MIN = 80.0     # bar_range climax threshold
EXHAUST_ACCEL_MIN = 20.0         # |acceleration| threshold
# Direction: decel means velocity and acceleration have opposite signs

# ABSORPTION entry — high volume + low range = big player absorbing
ABSORB_VOL_MIN = 1.5            # high relative volume
ABSORB_RANGE_MAX = 40.0         # low bar range (absorption = tight range)
ABSORB_WICK_MIN = 0.3           # wicks present (rejection)

# Wick rejection thresholds
WICK_5M_MIN = 0.83
WICK_15M_MIN = 0.77

# Velocity threshold: separates calm (mean-reversion) from momentum (freight train)
VELOCITY_THRESHOLD = 50.0    # |1m_velocity| above this = momentum entry
FREIGHT_TRAIN_THRESHOLD = 100.0  # |1m_velocity| above this = always ride, never fade

# FREIGHT_TRAIN entry filter (from EDA: winners accelerate, losers decelerate)
FREIGHT_TRAIN_VR_MAX = 0.85        # reject late entries (regime already committed)
# Exit: velocity collapsed to < 50% of entry AND decelerating
FREIGHT_TRAIN_VEL_DECAY = 0.50     # exit when |vel| drops below this fraction of entry |vel|

# REGIME_FLIP early conviction (from EDA: winners show z shrinking by bar 12)
REGIME_FLIP_CONVICTION_BARS = 12   # 1 minute (at 5s cadence) to prove thesis
REGIME_FLIP_VR_BAIL = 0.30         # vr above this = regime shifting back to trending

# ABSORPTION early conviction (from EDA: winners show z shrinking + volume fading)
ABSORB_CONVICTION_BARS = 12        # 1 minute to prove absorption thesis
ABSORB_Z_SHRINK_MIN = 0.10        # |z| must shrink by at least 10% from entry
ABSORB_VOL_PERSIST_MAX = 1.5      # if vol_rel still above this at bar 24, bail
ABSORB_VR_BAIL = 0.65             # vr above this = trending against absorption

# EXHAUSTION_BAR (from EDA: winners enter deeper z + higher vr, revert fast)
EXHAUST_Z_MIN = 1.31               # entry: must be deep in z extreme
EXHAUST_VR_MIN = 0.70             # entry: must be trending (real exhaustion, not chop)
EXHAUST_CONVICTION_BARS = 12      # exit: 1 minute to prove reversal
EXHAUST_Z_SHRINK_MIN = 0.20      # exit: |z| must shrink 20%+ from entry

# MTF_EXHAUSTION (from EDA: 17% WR, winners are deep z + high vr + high vol)
MTF_Z_MIN = 1.31                   # entry: must be deep in z extreme
MTF_VR_MIN = 0.58                 # entry: must be somewhat trending
MTF_VOL_MIN = 2.0                 # entry: must have volume conviction
MTF_CONVICTION_BARS = 12          # exit: 1 minute to prove thesis
MTF_Z_SHRINK_MIN = 0.10           # exit: |z| must shrink 10%+ from entry

# Bar range gate — disabled (low bar_range still profitable via volume)
BAR_RANGE_MIN = 0.0  # set to ~30 to activate (filters tight chop)

# Cascade: 1h z alignment
H1_Z_MIN = 0.88

# Tier 1-2 exit (kill shot / cascade)
P_CENTER_EXIT = 0.60
P_CENTER_EXIT_BARS_CASCADE = 3   # cascade: 3-bar confirmation (rare, high conviction)
P_CENTER_EXIT_BARS_KILLSHOT = 2  # kill shot: 2-bar confirmation ($740 config)

# Tier 3 exits (base NMP)
Z_EXIT = 0.5

# Hard stop — circuit breaker, overrides all CNNs
# Training: disabled (let regret see full paths)
# Live: set via BlendedEngine(live_mode=True) which overrides these
HARD_STOP = -99999.0  # disabled in training
# Per-contract hard stop ceiling (backtest + backup). LIVE uses the
# ACCOUNT-LEVEL stop in live/live_engine.py at -$40 unrealized total.
# Account-level stop naturally halves per-contract room as chains stack:
#   1 contract: $40 room  |  2: $20 each  |  3: $13 each  |  4: $10 each
# — so no explicit per-chain halving needed in the physics engine.
HARD_STOP_LIVE = -40.0  # per-contract ceiling (account stop usually fires first)

# Risk-aware entry filter + sizing (2026-04-20 research).
# Cohen d on blended OOS showed 1h_z_range (1h_z_high - 1h_z_low) is the
# strongest z-based BIG_LOSS predictor: BL avg range 1.80 vs winner 1.43
# (d=+0.32). Wide 1h oscillation = choppy hourly regime = high risk.
#
# Policy:
#   range > Z_RANGE_REJECT     : REJECT entry entirely (too choppy)
#   Z_RANGE_SIZE_1 to REJECT   : 1 contract only (moderate chop)
#   Z_RANGE_SIZE_2 to SIZE_1   : 2 contracts max
#   range < Z_RANGE_SIZE_2     : up to 3 contracts (safe regime)
Z_RANGE_REJECT = 2.2   # hard entry filter
Z_RANGE_SIZE_1 = 1.76   # above this -> max 1 contract
Z_RANGE_SIZE_2 = 1.32   # above this -> max 2 contracts

# Giveback stop — protect profits from round-tripping
# Training: disabled. Live: activated via live_mode=True
GIVEBACK_MIN_PEAK = 99999.0   # disabled in training
GIVEBACK_KEEP = 0.0           # disabled in training
GIVEBACK_MIN_PEAK_LIVE = 15.0  # once peak > $15, protect it
GIVEBACK_KEEP_LIVE = 0.40      # keep 40% of peak

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

# Tiered RIDE exits — stronger 1h = more patient
RIDE_EXIT_BARS_TIERS = {
    'strong': 5,    # |1h_z| > 2.0 at entry — hold longer, high conviction
    'medium': 3,    # |1h_z| 1.5-2.0 — standard
    'weak': 2,      # |1h_z| < 1.5 — exit fast, weak signal
}

# V2 dynamic indices via core_v2.features.FEATURE_NAMES
from core_v2.features import FEATURE_NAMES
_1M_Z_IDX = FEATURE_NAMES.index('L3_1m_z_se_15')
_1M_VELOCITY_IDX = FEATURE_NAMES.index('L2_1m_price_velocity_15')
_1M_ACCEL_IDX = FEATURE_NAMES.index('L2_1m_price_accel_15')
_1M_HURST_IDX = FEATURE_NAMES.index('L3_1m_hurst_15')
_1M_REVERSION_IDX = FEATURE_NAMES.index('L3_1m_reversion_prob_15')
_5M_VELOCITY_IDX = FEATURE_NAMES.index('L2_5m_price_velocity_9')
_5M_ACCEL_IDX = FEATURE_NAMES.index('L2_5m_price_accel_9')
_5M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_5m_bar_range')
_1H_Z_IDX = FEATURE_NAMES.index('L3_1h_z_se_12')
_1H_Z_HIGH_IDX = FEATURE_NAMES.index('L3_1h_z_high_12')
_1H_Z_LOW_IDX = FEATURE_NAMES.index('L3_1h_z_low_12')
_1H_VELOCITY_IDX = FEATURE_NAMES.index('L2_1h_price_velocity_12')

_1M_BODY_IDX = FEATURE_NAMES.index('L1_1m_body')
_1M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_1m_bar_range')
_5M_BODY_IDX = FEATURE_NAMES.index('L1_5m_body')
_5M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_5m_bar_range')
_15M_BODY_IDX = FEATURE_NAMES.index('L1_15m_body')
_15M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_15m_bar_range')

_5M_Z_IDX = FEATURE_NAMES.index('L3_5m_z_se_9')
_15M_Z_IDX = FEATURE_NAMES.index('L3_15m_z_se_12')

# Core dimensions (for grid logic)
_N_CORE = 23 # In V2 there are 23 features per TF
_N_HELPER = 0 # No helpers in V2
_N_TFS = 8 # V2 has 8 TFs typically, or 6


APPROACH_BUFFER_SIZE = 10  # CNN 1 loads approach from feature files directly, not buffer

# Tier encoding for CNN
TIER_MAP = {
    'CASCADE': 7, 'KILL_SHOT': 6,
    'FADE_CALM': 5, 'FADE_MOMENTUM': 4,
    'FADE_AGAINST': 3,   # fading z but 1h extreme against you
    'RIDE_CALM': 2, 'RIDE_MOMENTUM': 1,
    'RIDE_AGAINST': 0,   # CNN flipped but 1h opposes
    'PEAK': -2,  # disabled
    'REGIME_FLIP': -3,
    'MTF_EXHAUSTION': -4,
    'EXHAUSTION_BAR': -5,
    'ABSORPTION': -6,
    'FREIGHT_TRAIN': -1,
    'MTF_BREAKOUT': -7,
    'BASE_NMP': 0, 'MANUAL': 0,  # legacy compat
}

# 1h opposition threshold for FADE_AGAINST / RIDE_AGAINST
H1_AGAINST_Z_MIN = 1.32  # |1h_z| must be this extreme to count as "against"

# Tier conviction strength (higher = stronger signal, used for negative exits)
# An opposing tier exits the current trade only if it's stronger
TIER_STRENGTH = {
    'FREIGHT_TRAIN': 8,     # extreme velocity — strongest
    'KILL_SHOT': 7,         # wick rejection — very high conviction
    'CASCADE': 6,           # wick + 1h aligned
    'FADE_AGAINST': 5,      # fade with 1h opposition
    'RIDE_AGAINST': 4,      # 1h velocity ride
    'MTF_BREAKOUT': 3,      # multi-TF alignment
    'MTF_EXHAUSTION': 2,    # 5m deceleration
    'FADE_CALM': 1,         # default fade — weakest, never overrides
}

# Per-tier CNN model dir (5 jobs per tier)
CNN_PER_TIER_DIR = 'training/output/nn'

# Live release dir (packaged by training/release.py)
LIVE_RELEASE_DIR = 'checkpoints/live_release'

# Confidence thresholds (per-tier CNNs only act when confident)
ENTRY_GATE_MIN = 0.60
DIRECTION_CONFIDENCE_MIN = 0.75
DURATION_CONFIDENCE_MIN = 0.60
EXIT_CONFIDENCE_MIN = 0.70
LOSER_CONFIDENCE_MIN = 0.80
EXIT_CONFIRMATION_BARS = 3
LOSER_CONFIRMATION_BARS = 3

# Grid layout for CNN
_N_CORE = 12
_N_HELPER = 3
_N_TFS = 6
_HELPER_START = _N_CORE * _N_TFS  # 72


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

    def __init__(self, use_cnn=True, release_dir=None, skip_thin_market=False,
                 live_mode=False):
        self.skip_thin_market = skip_thin_market  # skip Sunday + holiday entries
        self.live_mode = live_mode  # True = PnL from NT8 fills, False = instafill (training)

        # Live circuit breakers (training keeps them disabled for regret)
        self._hard_stop = HARD_STOP_LIVE if live_mode else HARD_STOP
        self._giveback_min_peak = GIVEBACK_MIN_PEAK_LIVE if live_mode else GIVEBACK_MIN_PEAK
        self._giveback_keep = GIVEBACK_KEEP_LIVE if live_mode else GIVEBACK_KEEP
        self.in_pos = False
        self.direction = None
        self.entry_price = 0.0
        self.entry_79d = None
        self.entry_1m = None
        self.entry_tier = None
        self.bars_held = 0
        self._entry_ts = 0.0       # timestamp at entry — bars_held derived from this
        self._last_1m_ts = 0.0     # last 1m boundary seen — for cadence-independent counting
        self._cnn_exit_confirm = 0     # consecutive bars CNN says EXIT
        self._cnn_loser_confirm = 0    # consecutive bars CNN says DEAD
        self._cnn_duration_class = 1   # default MEDIUM
        self.peak_pnl = 0.0
        self.passed_center = False  # for tier 3 overshoot decision

        self._approach_buffer = []
        self._entry_approach = []
        self._trade_path = []
        self._chain_contracts = []  # parallel contracts from chained lightning
        self._reverse_warning = False  # tighten exits when opposite signal fires

        # Oscillation tracking (set properly in _open_trade, defaults here for safety)
        self._z_sign = 1.0
        self._zero_crossings = 0
        self._z_peak = 0.0
        self._z_trough = 0.0
        self._peak_amplitude = 0.0
        self._current_amplitude = 0.0

        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0
        self._close_history_1m = []
        self._volume_history_1m = []

        # CNN model directory — release_dir overrides defaults
        self._model_dir = release_dir
        if self._model_dir and not os.path.isdir(self._model_dir):
            print(f'  WARNING: release_dir {self._model_dir} not found, falling back to defaults')
            self._model_dir = None

        # CNN flip predictor for BASE_NMP trades
        self.use_cnn = use_cnn
        self._cnn_device = None

        # CNN flip (direction)
        # Per-tier CNN state (loaded by _load_per_tier_cnns)
        self._tier_entry_dir_models = {}
        self._tier_trade_mgr_models = {}
        self._tier_cnn_enabled = {}

        # CNN hold (exit timing)
        # (legacy CNN fields removed — per-tier system replaces them)

        # CNN risk (cut losers)

        if use_cnn:
            self._cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_per_tier_cnns()

    def _resolve_cnn_path(self, default_path: str) -> str:
        """Resolve CNN path: release_dir if set, else default."""
        if self._model_dir:
            return os.path.join(self._model_dir, os.path.basename(default_path))
        return default_path

    # ═══════════════════════════════════════════════════════════════════════
    # PER-TIER CNN SYSTEM (v3 — 5 jobs per tier, confidence-gated)
    # ═══════════════════════════════════════════════════════════════════════

    def _load_per_tier_cnns(self):
        """Load per-tier entry/direction/duration + exit/loser models."""
        self._tier_entry_dir_models = {}   # {tier: EntryDirectionNet}
        self._tier_trade_mgr_models = {}   # {tier: TradeManagerNet}
        self._tier_cnn_enabled = {}        # {tier: bool}

        cnn_dir = self._model_dir or CNN_PER_TIER_DIR
        if not os.path.exists(cnn_dir):
            return

        for tier_name in os.listdir(cnn_dir):
            tier_path = os.path.join(cnn_dir, tier_name)
            if not os.path.isdir(tier_path):
                continue

            # Entry/Direction/Duration
            ed_path = os.path.join(tier_path, 'entry_direction.pt')
            if os.path.exists(ed_path):
                try:
                    from training.cnn_entry_direction import EntryDirectionNet
                    cp = torch.load(ed_path, map_location='cpu', weights_only=False)
                    model = EntryDirectionNet().to(self._cnn_device)
                    model.load_state_dict(cp['state_dict'])
                    model.eval()
                    self._tier_entry_dir_models[tier_name] = model
                except Exception as e:
                    print(f'  WARNING: Failed to load {ed_path}: {e}')

            # Exit/Loser
            tm_path = os.path.join(tier_path, 'trade_manager.pt')
            if os.path.exists(tm_path):
                try:
                    from training.cnn_trade_manager import TradeManagerNet
                    cp = torch.load(tm_path, map_location='cpu', weights_only=False)
                    model = TradeManagerNet().to(self._cnn_device)
                    model.load_state_dict(cp['state_dict'])
                    model.eval()
                    self._tier_trade_mgr_models[tier_name] = model
                except Exception as e:
                    print(f'  WARNING: Failed to load {tm_path}: {e}')

            # Tier is CNN-enabled if at least entry_direction model loaded
            if tier_name in self._tier_entry_dir_models:
                self._tier_cnn_enabled[tier_name] = True

        if self._tier_cnn_enabled:
            tiers = ', '.join(sorted(self._tier_cnn_enabled.keys()))
            print(f'  Per-tier CNNs loaded: {tiers}')

    def _predict_entry_direction(self, feat, tier):
        """Per-tier entry gate + direction + duration prediction.

        Returns:
            (entry_conf, direction, direction_conf, duration_class, duration_conf)
            or None if no per-tier model for this tier.
        """
        model = self._tier_entry_dir_models.get(tier)
        if model is None:
            return None

        grid = _feat_to_grid(feat)
        grid_t = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self._cnn_device)

        with torch.no_grad():
            entry_p, dir_p, dur_p = model.predict_proba(grid_t)

        entry_conf = float(entry_p[0, 1])  # P(good_entry)
        dir_long_conf = float(dir_p[0, 1])  # P(long)
        direction = 'long' if dir_long_conf > 0.5 else 'short'
        direction_conf = max(dir_long_conf, 1 - dir_long_conf)

        dur_probs = dur_p[0].cpu().numpy()
        duration_class = int(dur_probs.argmax())
        duration_conf = float(dur_probs.max())

        return entry_conf, direction, direction_conf, duration_class, duration_conf

    def _predict_exit_loser(self, feat, tier, bars_held, pnl, peak_pnl, entry_z):
        """Per-tier exit + loser ID prediction.

        Returns:
            (exit_conf, loser_conf) — P(EXIT), P(DEAD)
            or None if no per-tier model for this tier.
        """
        model = self._tier_trade_mgr_models.get(tier)
        if model is None:
            return None

        feat_flat = np.array(feat, dtype=np.float32).flatten()
        context = np.array([
            bars_held / 60.0,      # normalize to hours
            pnl / 100.0,           # normalize to $100 units
            peak_pnl / 100.0,
            entry_z,
        ], dtype=np.float32)

        x = np.concatenate([feat_flat, context])
        x_t = torch.FloatTensor(x).unsqueeze(0).to(self._cnn_device)

        with torch.no_grad():
            exit_p, loser_p = model.predict_proba(x_t)

        exit_conf = float(exit_p[0, 0])    # P(EXIT) — class 0
        loser_conf = float(loser_p[0, 0])  # P(DEAD) — class 0

        return exit_conf, loser_conf

    def on_state(self, state: Dict):
        self._bar_count += 1
        feat = state['features']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price

        is_1m = (int(ts) % 60) < 5
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')

        # Read 1m state
        z = feat[_1M_Z_IDX]
        
        # Maintain history for V1 concepts
        if is_1m and state.get('bar_data'):
            bar = state['bar_data']
            self._close_history_1m.append(bar['close'])
            self._volume_history_1m.append(bar['volume'])
            if len(self._close_history_1m) > 60:
                self._close_history_1m.pop(0)
            if len(self._volume_history_1m) > 30:
                self._volume_history_1m.pop(0)
                
        # V1 concepts removed; use dummy defaults or direct mappings
        vr = 1.0
        vol_rel = 1.0
            
        p_center = 0.0
        dmi = np.sign(feat[_1M_VELOCITY_IDX]) * 5.0
        wick_5m = _wick_ratio(feat[_5M_BODY_IDX], feat[_5M_BAR_RANGE_IDX])
        wick_15m = _wick_ratio(feat[_15M_BODY_IDX], feat[_15M_BAR_RANGE_IDX])
        wick_1m = _wick_ratio(feat[_1M_BODY_IDX], feat[_1M_BAR_RANGE_IDX])

        # Approach buffer when flat
        if not self.in_pos:
            self._approach_buffer.append({
                'timestamp': ts, 'price': price,
                'features': feat.copy(),
            })
            if len(self._approach_buffer) > APPROACH_BUFFER_SIZE:
                self._approach_buffer = self._approach_buffer[-APPROACH_BUFFER_SIZE:]

        # === EXIT CHECK ===
        if self.in_pos:
            # bars_held = elapsed 1m bars since entry (cadence-independent)
            # Works correctly whether on_state is called at 5s, 15s, or 1m
            self.bars_held = int((ts - self._entry_ts) // 60)

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
                'features': feat.copy(),
            })

            # Hard stop — fires every bar, overrides everything.
            # Per-contract check kept for backtest. LIVE uses account-level
            # stop driven by NT8 bridge's unrealized PnL — see live_engine.py.
            if pnl <= self._hard_stop:
                self._close_trade(price, ts, time_str, 'hard_stop', feat)

            # Exit logic — every bar (5s cadence matches training)
            else:
                # 1. Giveback stop (all tiers)
                if self.peak_pnl >= self._giveback_min_peak and pnl < self.peak_pnl * self._giveback_keep:
                    self._close_trade(price, ts, time_str, 'giveback_stop', feat)
                    return

                # 2. Per-tier CNN exit + loser (confidence-gated, confirmation required)
                # Short-circuit when CNN disabled to avoid per-bar dict lookup
                if self.use_cnn:
                    entry_z = self.entry_1m.get('z_se', 0) if self.entry_1m else 0
                    tier_pred = self._predict_exit_loser(
                        feat, self.entry_tier, self.bars_held, pnl, self.peak_pnl, entry_z)
                else:
                    tier_pred = None

                if tier_pred is not None:
                    exit_conf, loser_conf = tier_pred

                    # Loser ID: only when underwater
                    if pnl < 0 and loser_conf > LOSER_CONFIDENCE_MIN:
                        self._cnn_loser_confirm += 1
                        if self._cnn_loser_confirm >= LOSER_CONFIRMATION_BARS:
                            self._close_trade(price, ts, time_str, 'cnn_loser_cut', feat)
                            return
                    else:
                        self._cnn_loser_confirm = 0

                    # Exit: when confident trade is done
                    if exit_conf > EXIT_CONFIDENCE_MIN:
                        self._cnn_exit_confirm += 1
                        if self._cnn_exit_confirm >= EXIT_CONFIRMATION_BARS:
                            self._close_trade(price, ts, time_str, 'cnn_exit', feat)
                            return
                    else:
                        self._cnn_exit_confirm = 0

                # 3. Physics exits (fallback — fires at 1m boundaries only)
                elif is_1m:
                    exit_reason = self._check_exit(feat, z, vr, pnl, p_center, vol_rel, wick_1m)
                    if exit_reason:
                        self._close_trade(price, ts, time_str, exit_reason, feat)

        # === CHAIN CONTRACT EXITS — each exits independently ===
        if self._chain_contracts and is_1m:
            closed_chains = []
            for i, cc in enumerate(self._chain_contracts):
                cc['bars_held'] = int((ts - cc['entry_ts']) // 60)

                if cc['direction'] == 'long':
                    cc_pnl = (price - cc['entry_price']) / TICK * TV
                else:
                    cc_pnl = (cc['entry_price'] - price) / TICK * TV
                cc['peak_pnl'] = max(cc['peak_pnl'], cc_pnl)

                # Run this contract's tier exit physics
                # Save/restore main trade state to reuse _check_exit
                saved = (self.entry_tier, self.bars_held, self.peak_pnl,
                         self._entry_abs_z, self._entry_velocity,
                         self._tier_p_center_bars, self._p_center_bars)

                self.entry_tier = cc['entry_tier']
                self.bars_held = cc['bars_held']
                self.peak_pnl = cc['peak_pnl']
                self._entry_abs_z = cc['entry_abs_z']
                self._entry_velocity = cc['entry_velocity']
                self._tier_p_center_bars = cc.get('_tier_p_center_bars', 0)
                self._p_center_bars = cc.get('_p_center_bars', 0)

                exit_reason = self._check_exit(feat, z, vr, cc_pnl, p_center, vol_rel, wick_1m)

                # Save back counter state
                cc['_tier_p_center_bars'] = self._tier_p_center_bars
                cc['_p_center_bars'] = self._p_center_bars

                # Restore main trade state
                (self.entry_tier, self.bars_held, self.peak_pnl,
                 self._entry_abs_z, self._entry_velocity,
                 self._tier_p_center_bars, self._p_center_bars) = saved

                # Hard stop — per-contract fallback. Account-level stop
                # in live_engine.py ($-40 unrealized total) naturally
                # halves per-contract room as chains stack (-$40/N), so
                # we don't need explicit per-chain halving here.
                if cc_pnl <= self._hard_stop:
                    exit_reason = 'hard_stop'

                # Giveback
                if cc['peak_pnl'] >= self._giveback_min_peak and cc_pnl < cc['peak_pnl'] * self._giveback_keep:
                    exit_reason = 'giveback_stop'

                if exit_reason:
                    self.trades.append({
                        'dir': cc['direction'],
                        'entry_price': cc['entry_price'],
                        'pnl': cc_pnl,
                        'peak': cc['peak_pnl'],
                        'held': cc['bars_held'],
                        'entry_tier': cc['entry_tier'],
                        'exit_reason': f'chain_{exit_reason}',
                        'cnn_flipped': cc.get('cnn_flipped', False),
                        'entry_79d': cc['entry_79d'].tolist() if hasattr(cc['entry_79d'], 'tolist') else cc['entry_79d'],
                        'exit_79d': feat.tolist() if hasattr(feat, 'tolist') else list(feat),
                        'approach': [],
                        'path': [],
                    })
                    self.daily_pnl += cc_pnl
                    closed_chains.append(i)

            # Remove closed chains (reverse order to preserve indices)
            for i in reversed(closed_chains):
                self._chain_contracts.pop(i)

        # === CHAINED LIGHTNING — parallel contracts ===
        # Same direction: open additional contract (max 3 total)
        # === NEGATIVE EXIT + CHAINED LIGHTNING ===
        MAX_CHAIN_CONTRACTS = 3
        if self.in_pos and is_1m and price > 100:
            direction_new, tier_new, flipped_new = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)

            # Negative exit: opposing setup fires = current trade's thesis is dead
            # Only exit if the opposing tier has HIGHER conviction than current tier
            if (tier_new is not None and direction_new != self.direction):
                opposing_strength = TIER_STRENGTH.get(tier_new, 0)
                current_strength = TIER_STRENGTH.get(self.entry_tier, 0)
                if opposing_strength > current_strength:
                    self._close_trade(price, ts, time_str,
                                      f'negative_exit_{tier_new}', feat)
                    return

            if (tier_new is not None and
                    direction_new == self.direction and
                    tier_new != self.entry_tier and
                    len(self._chain_contracts) < MAX_CHAIN_CONTRACTS):
                # Open parallel contract
                self._chain_contracts.append({
                    'entry_price': price,
                    'entry_tier': tier_new,
                    'direction': direction_new,
                    'entry_ts': ts,
                    'bars_held': 0,
                    'peak_pnl': 0.0,
                    'entry_79d': feat.copy(),
                    'entry_1m': {'z_se': z, 'vr': vr},
                    'entry_abs_z': abs(z),
                    'entry_velocity': abs(feat[_1M_VELOCITY_IDX]),
                    '_p_center_bars': 0,
                    '_tier_p_center_bars': 0,
                    'cnn_flipped': flipped_new,
                })

        # === ENTRY CHECK — 1m boundaries only ===
        # Skip thin-market sessions (Sundays + holiday-adjacent)
        if self.skip_thin_market:
            dt = datetime.utcfromtimestamp(ts)
            if dt.weekday() == 6:  # Sunday
                return
        if not self.in_pos and is_1m and price > 100:
            # Unified tier classification (no NMP/non-NMP split)
            direction, tier, cnn_flipped = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)

            if tier is None:
                return  # no tier qualifies

            # Per-tier CNN (when available)
            if self.use_cnn and tier in self._tier_cnn_enabled:
                pred = self._predict_entry_direction(feat, tier)
                if pred is not None:
                    entry_conf, cnn_dir, dir_conf, dur_class, dur_conf = pred
                    if entry_conf < ENTRY_GATE_MIN:
                        return
                    if dir_conf > DIRECTION_CONFIDENCE_MIN and cnn_dir != direction:
                        direction = cnn_dir
                        cnn_flipped = True
                    if dur_conf > DURATION_CONFIDENCE_MIN:
                        self._cnn_duration_class = dur_class

            self._open_trade(direction, price, ts, time_str, feat, tier,
                             cnn_flipped=cnn_flipped)

    def _classify_full_tier(self, feat, z, vr, wick_5m, wick_15m, dmi):
        """Classify entry into tiered strategy.

        Returns (direction, tier, cnn_flipped).

        Ordered by RARITY (least common first). Rare setups have specific
        physics = high conviction. They never steal many bars from common tiers.

          1. FREIGHT_TRAIN:   0.1/day, 86% WR, $103/tr — ride velocity
          2. MTF_EXHAUSTION:  3.3/day, 76% WR, $52/tr  — ride 5m (flipped)
          3. FADE_AGAINST:    2.9/day, 77% WR, $13/tr  — fade z, 1h opposing
          4. CASCADE:         5.8/day, 71% WR, $4/tr   — wick + 1h
          5. KILL_SHOT:      23.7/day, 72% WR, $7/tr   — wick, no 1h
          6. MTF_BREAKOUT:   81/day,   56% WR, $13/tr  — ride multi-TF breakout
          7. RIDE_AGAINST:   36/day,   52% WR, $8/tr   — ride 1h velocity
          8. FADE_CALM:      40/day,   66% WR, $7/tr   — default fade

        Lookback filters from EDA applied per tier.
        """
        # Risk-aware entry filter RESEARCH-ONLY for now (not wired in live).
        # Validate lift via tools/z_range_filter_backtest.py first.

        # NMP default direction: fade the z
        direction = 'short' if z > 0 else 'long'
        cnn_flipped = False

        # Read all conditions
        wick_5m = wick_5m
        wick_15m = wick_15m
        h1_z = feat[_1H_Z_IDX]
        velocity = feat[_1M_VELOCITY_IDX]
        h1_vel = feat[_1H_VELOCITY_IDX]
        abs_vel = abs(velocity)
        acceleration = feat[_1M_ACCEL_IDX]
        vr = vr
        v5_vel = feat[_5M_VELOCITY_IDX]
        dmi = dmi

        has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN
        h1_against_fade = ((direction == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                           (direction == 'short' and h1_z < -H1_AGAINST_Z_MIN))
        h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                      (direction == 'short' and h1_z > H1_Z_MIN))

        # Additional features for all tiers
        v5_accel = feat[_5M_ACCEL_IDX]
        v1 = abs(velocity)
        hurst = feat[_1M_HURST_IDX]

        # Ordered by SEQUENCE OF APPEARANCE (which signal fires first in chains):
        # KILL_SHOT triggers first 60% of the time (wick = earliest physics)
        # Then general conditions confirm (FADE_CALM, MTF_BREAKOUT)
        #
        # Chain data: KILL_SHOT→FADE_CALM (4150), RIDE_AGAINST→MTF_BREAKOUT (1467),
        #             CASCADE→FADE_CALM (867), KILL_SHOT→FADE_AGAINST (622)

        # 1. FREIGHT_TRAIN — extreme velocity, accelerating (rare, highest $/tr)
        if (abs_vel >= FREIGHT_TRAIN_THRESHOLD and
                velocity * acceleration > 0 and
                vr < FREIGHT_TRAIN_VR_MAX):
            ft_dir = 'long' if velocity > 0 else 'short'
            return ft_dir, 'FREIGHT_TRAIN', True

        # 2. KILL_SHOT — wick rejection (triggers 60% of chains)
        if has_wick and not h1_aligned:
            return direction, 'KILL_SHOT', False

        # 3. CASCADE — wick + 1h aligned
        if has_wick and h1_aligned:
            return direction, 'CASCADE', False

        # 4. RIDE_AGAINST — 1h velocity opposes (fires first 22% of chains)
        h1_vel_against = ((direction == 'long' and h1_vel < -3.0) or
                          (direction == 'short' and h1_vel > 3.0))
        if h1_vel_against and not h1_against_fade:
            ride_dir = 'long' if h1_vel > 0 else 'short'
            return ride_dir, 'RIDE_AGAINST', False

        # 5. FADE_AGAINST — 1h z extreme against fade
        if h1_against_fade and abs(v5_vel) < 10.0:
            return direction, 'FADE_AGAINST', False

        # 6. MTF_EXHAUSTION — ride 5m (flipped)
        if (v5_accel < 0 and abs(v5_vel) > MTF_5M_VEL_MIN and
                v1 > MTF_1M_VEL_ALIVE and
                abs(z) > MTF_Z_MIN):
            mtf_dir = 'long' if v5_vel > 0 else 'short'
            return mtf_dir, 'MTF_EXHAUSTION', True

        # 7. MTF_BREAKOUT — multi-TF aligned (confirms RIDE_AGAINST)
        z_5m = abs(feat[_5M_Z_IDX])
        z_15m = abs(feat[_15M_Z_IDX])
        if z_5m > 1.3 and z_15m > 1.3:
            breakout_dir = 'long' if z > 0 else 'short'
            dmi_aligned = ((breakout_dir == 'long' and dmi > -5) or
                           (breakout_dir == 'short' and dmi < 5))
            if dmi_aligned:
                return breakout_dir, 'MTF_BREAKOUT', True

        # 8. FADE_CALM — default (confirms all specific triggers)
        higher_tf_opposing = False
        if direction == 'long' and v5_vel < -3 and h1_vel < -3:
            higher_tf_opposing = True
        if direction == 'short' and v5_vel > 3 and h1_vel > 3:
            higher_tf_opposing = True
        if not higher_tf_opposing:
            return direction, 'FADE_CALM', False

        return None, None, False

    def _check_exit(self, feat, z, vr, pnl, p_center, vol_rel, wick_1m):
        """Check exit based on entry tier."""
        if self.entry_tier in ('CASCADE', 'KILL_SHOT'):
            # Tier 1-2: exit at p_center (per-tier bar confirmation)
            p_center = p_center
            if abs(z) < 0.6:
                self._tier_p_center_bars += 1
            else:
                self._tier_p_center_bars = 0
            required_bars = (P_CENTER_EXIT_BARS_CASCADE if self.entry_tier == 'CASCADE'
                             else P_CENTER_EXIT_BARS_KILLSHOT)
            if self._tier_p_center_bars >= required_bars:
                return f'{self.entry_tier.lower()}_center'

            # Z conviction fallback: if z hasn't moved by bar 24, wick failed
            # EDA: winners |z| 2.33->1.00 by bar 24, losers 2.35->1.58 (stuck)
            abs_z = abs(z)
            if self.bars_held >= 24 and self.bars_held < 27:
                z_shrink = (self._entry_abs_z - abs_z) / max(self._entry_abs_z, 0.01)
                if z_shrink < 0.20:  # less than 20% z movement
                    return f'{self.entry_tier.lower()}_no_conviction'

            return None

        # FREIGHT_TRAIN exit: velocity collapsed + decelerating
        # The train entered on extreme velocity. Exit when:
        # 1. |velocity| dropped below 50% of entry |velocity| (train slowing)
        # 2. velocity * acceleration < 0 (decelerating)
        if self.entry_tier == 'FREIGHT_TRAIN':
            velocity = feat[_1M_VELOCITY_IDX]
            acceleration = feat[_1M_ACCEL_IDX]
            abs_vel = abs(velocity)
            vel_ratio = abs_vel / max(self._entry_velocity, 1.0)

            # Primary: velocity collapsed AND decelerating
            if vel_ratio < FREIGHT_TRAIN_VEL_DECAY and velocity * acceleration < 0:
                return 'freight_train_decel'
            # Fallback: velocity dropped below threshold (no longer a freight train)
            if abs_vel < VELOCITY_THRESHOLD:
                return 'freight_train_vel_dead'
            return None

        # REGIME_FLIP exit: early conviction + regime physics
        # EDA: by bar 12, winners have |z| shrinking, losers have |z| growing
        # By bar 24, losers have vr > 0.28 (regime shifting back)
        if self.entry_tier == 'REGIME_FLIP':
            abs_z = abs(z)

            # Early conviction: if z moved AWAY from zero by bar 12, thesis failed
            if self.bars_held >= REGIME_FLIP_CONVICTION_BARS and self.bars_held < REGIME_FLIP_CONVICTION_BARS + 3:
                if abs_z > self._entry_abs_z:
                    return 'regime_no_conviction'

            # VR rising = regime shifting back to trending (against our mean-reversion)
            if False: # vr exit removed
                return 'regime_vr_rising'

            # Mean reached = reversion complete, take profit
            if abs_z < 0.3:
                return 'regime_mean_reached'

            return None

        # MTF_EXHAUSTION exit: SAME exit physics as original (fade exit)
        # Direction is flipped at entry, but exit logic unchanged.
        # Exit when the original fade thesis would have exited.
        if self.entry_tier == 'MTF_EXHAUSTION':
            v5_accel = feat[_5M_ACCEL_IDX]
            v1 = abs(feat[_1M_VELOCITY_IDX])

            # 5m reaccelerated = original move resuming
            if v5_accel > 0 and abs(feat[_5M_VELOCITY_IDX]) > 30:
                return 'mtf_5m_reaccelerated'

            # 1m velocity exhausted
            if v1 < 0.3:
                return 'mtf_1m_exhausted'

            # Mean reached
            if abs(z) < 0.3:
                return 'mtf_mean_reached'

            return None

        # MTF_BREAKOUT exit: ride until multi-TF alignment breaks
        if self.entry_tier == 'MTF_BREAKOUT':
            z_5m = abs(feat[_5M_Z_IDX])
            z_15m = abs(feat[_15M_Z_IDX])

            # Alignment broken: either 5m or 15m z dropped below 0.8
            if z_5m < 0.8 or z_15m < 0.8:
                return 'breakout_alignment_lost'

            # 1m z crossed zero (overshot to other side)
            if self.direction == 'long' and z < -0.3:
                return 'breakout_overshot'
            if self.direction == 'short' and z > 0.3:
                return 'breakout_overshot'

            # 1m velocity exhausted — momentum driving breakout is dead
            if abs(feat[_1M_VELOCITY_IDX]) < RIDE_VELOCITY_EXHAUSTED and self.bars_held >= 5:
                return 'breakout_vel_exhausted'

            return None

        # EXHAUSTION_BAR exit: early z conviction + mean reached
        # EDA: winners z 1.59->1.08 by bar 12, losers z 1.26->1.24 (stuck)
        if self.entry_tier == 'EXHAUSTION_BAR':
            abs_z = abs(z)

            # Early conviction: z must shrink 20%+ by bar 12
            if self.bars_held >= EXHAUST_CONVICTION_BARS and self.bars_held < EXHAUST_CONVICTION_BARS + 3:
                z_shrink = (self._entry_abs_z - abs_z) / max(self._entry_abs_z, 0.01)
                if z_shrink < EXHAUST_Z_SHRINK_MIN:
                    return 'exhaust_no_conviction'

            # Mean reached
            if abs_z < 0.3:
                return 'exhaust_mean_reached'

            return None

        # ABSORPTION exit: early conviction + volume fade + vr
        # EDA: winners have z shrinking + volume fading by bar 12
        # Losers: z flat, volume persistent, vr rising
        if self.entry_tier == 'ABSORPTION':
            abs_z = abs(z)
            vol_rel = vol_rel

            # Early conviction: z must shrink by 10%+ from entry by bar 12
            if self.bars_held >= ABSORB_CONVICTION_BARS and self.bars_held < ABSORB_CONVICTION_BARS + 3:
                z_shrink = (self._entry_abs_z - abs_z) / max(self._entry_abs_z, 0.01)
                if z_shrink < ABSORB_Z_SHRINK_MIN:
                    return 'absorb_no_conviction'

            # Volume still elevated at bar 24 = absorption failing
            if self.bars_held >= 24 and vol_rel > ABSORB_VOL_PERSIST_MAX:
                return 'absorb_vol_persistent'

            # VR rising = trending against us
            if False: # vr exit removed
                return 'absorb_vr_rising'

            # Mean reached = absorption complete
            if abs_z < 0.3:
                return 'absorb_mean_reached'

            return None

        # RIDE_AGAINST exit: riding 1h velocity, exit when momentum dies
        if self.entry_tier == 'RIDE_AGAINST':
            velocity = feat[_1M_VELOCITY_IDX]
            h1_vel = feat[_1H_VELOCITY_IDX]

            # 1h velocity reversed (the 1h trend we're riding is dying)
            entry_h1_sign = 1.0 if self.direction == 'long' else -1.0
            h1_against = (h1_vel * entry_h1_sign) < 0
            if h1_against and self.bars_held >= 3:
                return 'ride_h1_reversed'

            # 1m velocity exhausted (momentum dead)
            if abs(velocity) < RIDE_VELOCITY_EXHAUSTED and self.bars_held >= 3:
                self._ride_vel_bars = getattr(self, '_ride_vel_bars', 0) + 1
                if self._ride_vel_bars >= RIDE_EXIT_BARS:
                    return 'ride_velocity_exhausted'
            else:
                self._ride_vel_bars = 0

            # Mean reached (z faded back to center — ride done)
            if abs(z) < FADE_Z_EXIT and self.bars_held >= 3:
                return 'fade_mean_reached'

            return None

        # FADE_AGAINST exit: fading z with 1h opposing, same as FADE_CALM
        # Falls through to the generic fade/ride exit below

        # Other tiers: two exit modes based on trade type
        p_center = p_center
        velocity = feat[_1M_VELOCITY_IDX]
        wick = wick_1m
        reversion = feat[_1M_REVERSION_IDX]

        if not getattr(self, 'cnn_flipped', False):
            # ── FADE MODE: entered against z, fading toward mean ──
            # 54% never cross zero, 26% oscillate

            # Track consecutive bars for exit confirmation
            if abs(z) < 0.6:
                self._p_center_bars += 1
            else:
                self._p_center_bars = 0

            if abs(z) < FADE_Z_EXIT:
                self._z_near_zero_bars += 1
            else:
                self._z_near_zero_bars = 0

            # Phase 0: approaching mean (uniform 3-bar, $740 config)
            if self._zero_crossings == 0:
                if self._z_near_zero_bars >= FADE_Z_EXIT_BARS:
                    return 'fade_mean_reached'
                if self._p_center_bars >= FADE_P_CENTER_BARS:
                    return 'fade_p_center'
                # Fade stalled: z moved AWAY from zero (momentum against us)
                if self.bars_held >= 12:
                    abs_z = abs(z)
                    if abs_z > self._entry_abs_z * 1.2:
                        return 'fade_z_expanding'
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

            # High crossing count = exhausted oscillation, take what we have
            if self._zero_crossings >= 5:
                return 'fade_oscillation_exhausted'

            return None

        else:
            # ── RIDE MODE: flipped by CNN, riding with z ──
            # 69% never cross zero, momentum trades

            # Track consecutive bars for each condition
            if abs(velocity) < RIDE_VELOCITY_EXHAUSTED:
                self._ride_vel_bars += 1
            else:
                self._ride_vel_bars = 0

            if False: # vr exit removed
                self._ride_vr_bars += 1
            else:
                self._ride_vr_bars = 0

            if reversion > RIDE_REVERSION_HIGH and wick > RIDE_WICK_HIGH:
                self._ride_rev_wick_bars += 1
            else:
                self._ride_rev_wick_bars = 0

            # Tiered exit patience based on 1h_z at entry
            required_bars = getattr(self, '_ride_exit_bars', RIDE_EXIT_BARS)

            # Exit when momentum exhausted (sustained)
            if self._ride_vel_bars >= required_bars:
                return 'ride_velocity_exhausted'

            # Exit when regime shifts (sustained)
            if self._ride_vr_bars >= required_bars:
                return 'ride_regime_shift'

            # Exit when market wants to snap back (sustained)
            if self._ride_rev_wick_bars >= required_bars:
                return 'ride_reversion_wick'

            return None

    def inject_manual_trade(self, direction: str, price: float, ts: float, feat):
        """Open a trade from external trigger (dashboard button).

        Classifies current market state into the closest physics tier so the
        trade gets proper tier-specific exit management (not a generic exit).
        """
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')
        z = feat[_1M_Z_IDX]
        vr = vr

        # Classify tier from current market state
        tier = 'FADE_CALM'  # safe default — gets full CNN exit management
        cnn_flipped = False

        if abs(z) > ROCHE:
            # NMP conditions met — full tier classification
            _, tier, cnn_flipped = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)
            # If classified direction differs from manual, mark as flipped
            classified_dir = 'short' if z > 0 else 'long'
            if classified_dir != direction:
                cnn_flipped = True
        else:
            # Non-NMP conditions — pick closest non-NMP tier by physics
            hurst = feat[_1M_HURST_IDX]
            if hurst < REGIME_HURST_MAX:
                tier = 'REGIME_FLIP'
            elif abs(feat[_1M_VELOCITY_IDX]) >= VELOCITY_THRESHOLD:
                tier = 'FADE_MOMENTUM'
            else:
                tier = 'FADE_CALM'

        self._open_trade(direction, price, ts, time_str, feat, tier,
                         cnn_flipped=cnn_flipped)

    def _open_trade(self, direction, price, ts, time_str, feat, tier, cnn_flipped=False):
        self.in_pos = True
        self.direction = direction
        self.entry_price = price
        self._reverse_warning = False
        self._reverse_logged = False
        self.entry_79d = feat.copy()
        self.entry_1m = {
            'z_se': feat[_1M_Z_IDX],
        }
        self.entry_tier = tier
        self.cnn_flipped = cnn_flipped
        self.bars_held = 0
        self._entry_ts = ts
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

        # Entry context for tiered exits
        self._entry_h1_z = abs(feat[_1H_Z_IDX])
        self._entry_velocity = abs(feat[_1M_VELOCITY_IDX])  # for FREIGHT_TRAIN decay exit
        self._entry_abs_z = abs(feat[_1M_Z_IDX])     # for REGIME_FLIP/ABSORPTION conviction
        # 5m velocity alignment with trade direction (exit patience signal)
        v5 = feat[_5M_VELOCITY_IDX]
        self._v5_aligned = ((direction == 'long' and v5 > 0) or
                            (direction == 'short' and v5 < 0))
        if self._entry_h1_z > 2.0:
            self._ride_exit_bars = RIDE_EXIT_BARS_TIERS['strong']
        elif self._entry_h1_z > 1.5:
            self._ride_exit_bars = RIDE_EXIT_BARS_TIERS['medium']
        else:
            self._ride_exit_bars = RIDE_EXIT_BARS_TIERS['weak']

        # Oscillation tracking (for FADE exit mode)
        self._z_sign = 1.0 if feat[_1M_Z_IDX] > 0 else -1.0
        self._zero_crossings = 0
        self._z_peak = abs(feat[_1M_Z_IDX])  # entry |z| = initial amplitude
        self._z_trough = abs(feat[_1M_Z_IDX])
        self._peak_amplitude = abs(feat[_1M_Z_IDX])  # tracks max oscillation swing
        self._current_amplitude = abs(feat[_1M_Z_IDX])

    def _flatten_all_chains(self, price, ts, feat, reason):
        """Close all chain contracts immediately."""
        for cc in self._chain_contracts:
            if cc['direction'] == 'long':
                cc_pnl = (price - cc['entry_price']) / TICK * TV
            else:
                cc_pnl = (cc['entry_price'] - price) / TICK * TV
            self.daily_pnl += cc_pnl
            self.trades.append({
                'dir': cc['direction'],
                'entry_price': cc['entry_price'],
                'pnl': cc_pnl,
                'peak': cc.get('peak_pnl', 0),
                'held': int((ts - cc['entry_ts']) // 60) if cc.get('entry_ts') else 0,
                'entry_tier': cc['entry_tier'],
                'exit_reason': f'chain_{reason}',
                'cnn_flipped': cc.get('cnn_flipped', False),
                'entry_79d': cc.get('entry_79d', []),
                'exit_79d': feat.tolist() if hasattr(feat, 'tolist') else list(feat),
                'approach': [], 'path': [],
            })
        self._chain_contracts = []

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
            'v5_aligned': getattr(self, '_v5_aligned', True),
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

        # Chain contracts stay alive — they exit independently
        # Only reverse_signal flattens all chains

    def get_recommended_max_contracts(self, feat) -> int:
        """Risk-aware contract cap based on 1h z-range oscillation at entry.

        Called by live engine to size entry orders. Returns 1/2/3 based on
        choppy-regime risk. Trade already rejected in _classify if range
        >= Z_RANGE_REJECT, so this only sees the acceptable range.
        """
        z1h_range = feat[_1H_Z_HIGH_IDX] - feat[_1H_Z_LOW_IDX]
        if z1h_range >= Z_RANGE_SIZE_1:
            return 1
        if z1h_range >= Z_RANGE_SIZE_2:
            return 2
        return 3

    def force_close(self, reason='end_of_day'):
        # Flatten all chain contracts first
        if self._chain_contracts:
            ts = self._trade_path[-1]['timestamp'] if self._trade_path else 0
            feat = self._trade_path[-1]['features'] if self._trade_path else self.entry_79d
            if feat is None:
                feat = np.zeros(91)
            self._flatten_all_chains(self._last_price, ts, feat, reason)

        if self.in_pos:
            ts = self._trade_path[-1]['timestamp'] if self._trade_path else 0
            feat = self._trade_path[-1]['features'] if self._trade_path else self.entry_79d
            if feat is None:
                feat = np.zeros(91)
            time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M') if ts > 0 else '??:??'
            self._close_trade(self._last_price, ts, time_str, reason, feat)

    # get_trade_state / restore_trade_state — DELETED in Phase 5.
    # Checkpoint save/restore now uses core.ledger.Ledger directly.
    # See live/engine_v2.py _periodic_save() and _step5b_recover_trade().

    def reset(self):
        self.in_pos = False
        self.direction = None
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._trade_path = []
        self._approach_buffer = []
        self.entry_tier = None
        self._entry_ts = 0.0
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

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: stateless signal path (evaluate + helpers)
    # ══════════════════════════════════════════════════════════════════════
    #
    # The methods below replace on_state() with a pure function:
    #
    #     evaluate(state) -> DecisionBatch
    #
    # They read positions via state['positions'] (a PositionsView from the
    # ledger) and return a DecisionBatch describing what the engine would
    # recommend. They do NOT mutate self.*. Both sim and live executors
    # apply the batch by updating the ledger.
    #
    # on_state() remains the default path until Phase 4 flips the switch.
    # Both paths coexist so Phase 3 can run them side-by-side in testing.
    #
    # Spec: docs/JULES_ENGINE_DECOUPLE_ORDERS.md
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(self, state: Dict) -> 'DecisionBatch':
        """Stateless signal pass over the current ledger snapshot.

        Reads:
            state['features_79d']  — 91-D feature vector
            state['price']         — current bar close
            state['timestamp']     — current bar timestamp (seconds)
            state['positions']     — PositionsView from the ledger

        Returns:
            DecisionBatch — per-position counter updates + exit reasons,
            plus optional entry / chain_entry / negative_exit signals.

        Does not mutate self. The old on_state path is unaffected.
        """
        # Local imports keep the engine module independent of core/ at module
        # load time — training/nightmare_blended.py is imported by a lot of
        # analysis scripts, some of which may not have core/ on sys.path.
        from core_v2.engine_signals import (
            DecisionBatch, EntrySignal, ExitSignal, PositionDecision,
            PositionsView,
        )

        feat = state['features_79d']
        price = state['price']
        ts = state['timestamp']
        positions: PositionsView = state.get('positions') or PositionsView()

        is_1m = (int(ts) % 60) < 5
        z = feat[_1M_Z_IDX]
        
        vr = state.get('variance_ratio', 1.0)
        wick_5m = _wick_ratio(feat[_5M_BODY_IDX], feat[_5M_BAR_RANGE_IDX])
        wick_15m = _wick_ratio(feat[_15M_BODY_IDX], feat[_15M_BAR_RANGE_IDX])
        dmi = np.sign(feat[_1M_VELOCITY_IDX]) * 5.0

        batch = DecisionBatch()

        # ── Per-position exit evaluation ───────────────────────────────
        # Every open position (primary + chains) gets a PositionDecision.
        # Counter updates happen every bar; physics exits gate on is_1m.
        for pos in positions.all_positions:
            # Compute stateless variables
            p_center = 0.0
            vol_rel = state.get('vol_rel', 1.0)
            wick_1m = _wick_ratio(feat[_1M_BODY_IDX], feat[_1M_BAR_RANGE_IDX])
            new_counters, exit_reason = self._evaluate_position_exit(
                pos, feat, z, vr, price, is_1m, p_center, vol_rel, wick_1m
            )
            batch.position_decisions.append(PositionDecision(
                contract_id=pos.contract_id,
                ride_vel_bars=new_counters['ride_vel_bars'],
                ride_vr_bars=new_counters['ride_vr_bars'],
                ride_rev_wick_bars=new_counters['ride_rev_wick_bars'],
                tier_p_center_bars=new_counters['tier_p_center_bars'],
                p_center_bars=new_counters['p_center_bars'],
                z_near_zero_bars=new_counters['z_near_zero_bars'],
                slow_flip_active=new_counters['slow_flip_active'],
                exit_reason=exit_reason,
            ))

        # ── Chain entry / negative exit (on primary, 1m boundaries) ─────
        if positions.primary is not None and is_1m and price > 100:
            direction_new, tier_new, flipped_new = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)

            # Negative exit: opposing tier with higher conviction
            if tier_new is not None and direction_new != positions.primary.direction:
                opposing_strength = TIER_STRENGTH.get(tier_new, 0)
                current_strength = TIER_STRENGTH.get(positions.primary.entry_tier, 0)
                if opposing_strength > current_strength:
                    batch.negative_exit = ExitSignal(
                        contract_id=positions.primary.contract_id,
                        reason=f'negative_exit_{tier_new}',
                    )

            # Chain entry: same direction, different tier, under cap
            MAX_CHAIN = 3   # matches MAX_CHAIN_CONTRACTS in on_state()
            if (tier_new is not None
                    and direction_new == positions.primary.direction
                    and tier_new != positions.primary.entry_tier
                    and len(positions.chains) < MAX_CHAIN):
                batch.chain_entry = EntrySignal(
                    tier=tier_new,
                    direction=direction_new,
                    cnn_flipped=flipped_new,
                )

        # ── Fresh entry (flat ledger, 1m boundaries) ────────────────────
        if positions.is_flat and is_1m and price > 100:
            # Skip thin-market sessions (Sundays) — matches on_state() gate
            if getattr(self, 'skip_thin_market', False):
                dt = datetime.utcfromtimestamp(ts)
                if dt.weekday() == 6:   # Sunday
                    return batch

            direction, tier, cnn_flipped = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)
            if tier is not None:
                batch.entry = EntrySignal(
                    tier=tier,
                    direction=direction,
                    cnn_flipped=cnn_flipped,
                )

        return batch

    def _evaluate_position_exit(self, pos, feat, z, vr, price, is_1m, p_center, vol_rel, wick_1m):
        """Evaluate exit conditions for ONE position in isolation.

        This is the stateless parallel to _check_exit(). Instead of reading
        and mutating self.*, it reads from `pos` (a PositionView) and
        returns the new counter values alongside an optional exit reason.

        Returns:
            (new_counters: dict, exit_reason: Optional[str])

        `new_counters` contains the updated values for every counter field
        on PositionDecision. If the tier doesn't update a particular
        counter, the value is passed through unchanged from `pos`.

        Note: does NOT handle the CNN exit path (use_cnn=False at present).
        CNN exits will re-enter in a later phase once CNN is re-enabled.
        """
        # Base: all counters pass through unchanged unless we explicitly update them.
        new_counters = {
            'ride_vel_bars': pos.ride_vel_bars,
            'ride_vr_bars': pos.ride_vr_bars,
            'ride_rev_wick_bars': pos.ride_rev_wick_bars,
            'tier_p_center_bars': pos.tier_p_center_bars,
            'p_center_bars': pos.p_center_bars,
            'z_near_zero_bars': pos.z_near_zero_bars,
            'slow_flip_active': pos.slow_flip_active,
        }

        # Compute pnl for this specific position
        if pos.direction == 'long':
            pnl = (price - pos.entry_price) / TICK * TV
        else:
            pnl = (pos.entry_price - price) / TICK * TV

        # Hard stop — every bar, overrides everything
        if pnl <= self._hard_stop:
            return new_counters, 'hard_stop'

        # Giveback stop — every bar, once peak is meaningful
        if pos.peak_pnl >= self._giveback_min_peak and pnl < pos.peak_pnl * self._giveback_keep:
            return new_counters, 'giveback_stop'

        # Physics exits — 1m boundaries only
        if not is_1m:
            return new_counters, None

        tier = pos.entry_tier

        # ── CASCADE / KILL_SHOT: p_center exit ──────────────────────
        if tier in ('CASCADE', 'KILL_SHOT'):
            p_center = p_center
            if abs(z) < 0.6:
                new_counters['tier_p_center_bars'] = pos.tier_p_center_bars + 1
            else:
                new_counters['tier_p_center_bars'] = 0
            required = (P_CENTER_EXIT_BARS_CASCADE if tier == 'CASCADE'
                        else P_CENTER_EXIT_BARS_KILLSHOT)
            if new_counters['tier_p_center_bars'] >= required:
                return new_counters, f'{tier.lower()}_center'

            # Z conviction fallback at bar 24-27
            abs_z = abs(z)
            if 24 <= pos.bars_held < 27:
                z_shrink = (pos.entry_abs_z - abs_z) / max(pos.entry_abs_z, 0.01)
                if z_shrink < 0.20:
                    return new_counters, f'{tier.lower()}_no_conviction'
            return new_counters, None

        # ── FREIGHT_TRAIN: velocity decay ───────────────────────────
        if tier == 'FREIGHT_TRAIN':
            velocity = feat[_1M_VELOCITY_IDX]
            acceleration = feat[_1M_ACCEL_IDX]
            abs_vel = abs(velocity)
            vel_ratio = abs_vel / max(pos.entry_velocity, 1.0)
            if vel_ratio < FREIGHT_TRAIN_VEL_DECAY and velocity * acceleration < 0:
                return new_counters, 'freight_train_decel'
            if abs_vel < VELOCITY_THRESHOLD:
                return new_counters, 'freight_train_vel_dead'
            return new_counters, None

        # ── REGIME_FLIP ─────────────────────────────────────────────
        if tier == 'REGIME_FLIP':
            abs_z = abs(z)
            if (REGIME_FLIP_CONVICTION_BARS <= pos.bars_held
                    < REGIME_FLIP_CONVICTION_BARS + 3):
                if abs_z > pos.entry_abs_z:
                    return new_counters, 'regime_no_conviction'
            if False: # vr exit removed
                return new_counters, 'regime_vr_rising'
            if abs_z < 0.3:
                return new_counters, 'regime_mean_reached'
            return new_counters, None

        # ── MTF_EXHAUSTION ──────────────────────────────────────────
        if tier == 'MTF_EXHAUSTION':
            v5_accel = feat[_5M_ACCEL_IDX]
            v1 = abs(feat[_1M_VELOCITY_IDX])
            if v5_accel > 0 and abs(feat[_5M_VELOCITY_IDX]) > 30:
                return new_counters, 'mtf_5m_reaccelerated'
            if v1 < 0.3:
                return new_counters, 'mtf_1m_exhausted'
            if abs(z) < 0.3:
                return new_counters, 'mtf_mean_reached'
            return new_counters, None

        # ── MTF_BREAKOUT ────────────────────────────────────────────
        if tier == 'MTF_BREAKOUT':
            z_5m = abs(feat[2 * _N_CORE + _Z])
            z_15m = abs(feat[3 * _N_CORE + _Z])
            if z_5m < 0.8 or z_15m < 0.8:
                return new_counters, 'breakout_alignment_lost'
            if pos.direction == 'long' and z < -0.3:
                return new_counters, 'breakout_overshot'
            if pos.direction == 'short' and z > 0.3:
                return new_counters, 'breakout_overshot'
            if abs(feat[_1M_VELOCITY_IDX]) < RIDE_VELOCITY_EXHAUSTED and pos.bars_held >= 5:
                return new_counters, 'breakout_vel_exhausted'
            return new_counters, None

        # ── EXHAUSTION_BAR ──────────────────────────────────────────
        if tier == 'EXHAUSTION_BAR':
            abs_z = abs(z)
            if (EXHAUST_CONVICTION_BARS <= pos.bars_held
                    < EXHAUST_CONVICTION_BARS + 3):
                z_shrink = (pos.entry_abs_z - abs_z) / max(pos.entry_abs_z, 0.01)
                if z_shrink < EXHAUST_Z_SHRINK_MIN:
                    return new_counters, 'exhaust_no_conviction'
            if abs_z < 0.3:
                return new_counters, 'exhaust_mean_reached'
            return new_counters, None

        # ── ABSORPTION ──────────────────────────────────────────────
        if tier == 'ABSORPTION':
            abs_z = abs(z)
            vol_rel = vol_rel
            if (ABSORB_CONVICTION_BARS <= pos.bars_held
                    < ABSORB_CONVICTION_BARS + 3):
                z_shrink = (pos.entry_abs_z - abs_z) / max(pos.entry_abs_z, 0.01)
                if z_shrink < ABSORB_Z_SHRINK_MIN:
                    return new_counters, 'absorb_no_conviction'
            if pos.bars_held >= 24 and vol_rel > ABSORB_VOL_PERSIST_MAX:
                return new_counters, 'absorb_vol_persistent'
            if False: # vr exit removed
                return new_counters, 'absorb_vr_rising'
            if abs_z < 0.3:
                return new_counters, 'absorb_mean_reached'
            return new_counters, None

        # ── RIDE_AGAINST ────────────────────────────────────────────
        if tier == 'RIDE_AGAINST':
            velocity = feat[_1M_VELOCITY_IDX]
            h1_vel = feat[_1H_VELOCITY_IDX]
            entry_h1_sign = 1.0 if pos.direction == 'long' else -1.0
            h1_against = (h1_vel * entry_h1_sign) < 0
            if h1_against and pos.bars_held >= 3:
                return new_counters, 'ride_h1_reversed'

            if abs(velocity) < RIDE_VELOCITY_EXHAUSTED and pos.bars_held >= 3:
                new_counters['ride_vel_bars'] = pos.ride_vel_bars + 1
                if new_counters['ride_vel_bars'] >= RIDE_EXIT_BARS:
                    return new_counters, 'ride_velocity_exhausted'
            else:
                new_counters['ride_vel_bars'] = 0

            if abs(z) < FADE_Z_EXIT and pos.bars_held >= 3:
                return new_counters, 'fade_mean_reached'
            return new_counters, None

        # ── Default: FADE_CALM / FADE_AGAINST / RIDE_CALM / RIDE_MOMENTUM / FADE_MOMENTUM ──
        # Two exit modes based on cnn_flipped: FADE (entered against z) vs RIDE (with z).
        p_center = p_center
        velocity = feat[_1M_VELOCITY_IDX]
        wick = wick_1m
        reversion = feat[_1M_REVERSION_IDX]

        if not pos.cnn_flipped:
            # ── FADE MODE ──
            if abs(z) < 0.6:
                new_counters['p_center_bars'] = pos.p_center_bars + 1
            else:
                new_counters['p_center_bars'] = 0

            if abs(z) < FADE_Z_EXIT:
                new_counters['z_near_zero_bars'] = pos.z_near_zero_bars + 1
            else:
                new_counters['z_near_zero_bars'] = 0

            # Phase 0: approaching mean (never crossed zero)
            if pos.zero_crossings == 0:
                if new_counters['z_near_zero_bars'] >= FADE_Z_EXIT_BARS:
                    return new_counters, 'fade_mean_reached'
                if new_counters['p_center_bars'] >= FADE_P_CENTER_BARS:
                    return new_counters, 'fade_p_center'
                if pos.bars_held >= 12:
                    if abs(z) > pos.entry_abs_z * 1.2:
                        return new_counters, 'fade_z_expanding'
                return new_counters, None

            # Phase 1: oscillation mode (crossed zero at least once)
            if (pos.peak_amplitude > 0
                    and pos.current_amplitude < pos.peak_amplitude * FADE_OSCILLATION_DECAY):
                # Entry z_se is at feature index _1M_Z_IDX
                entry_z = float(pos.entry_features[_1M_Z_IDX])
                entry_z_sign = 1.0 if entry_z > 0 else -1.0
                z_favorable = (z > 0) != (entry_z_sign > 0)
                if z_favorable or pos.zero_crossings >= 3:
                    return new_counters, 'fade_oscillation_decay'

            if pos.zero_crossings >= 2 and new_counters['p_center_bars'] >= FADE_P_CENTER_BARS:
                return new_counters, 'fade_oscillation_center'

            if pos.zero_crossings >= 5:
                return new_counters, 'fade_oscillation_exhausted'

            return new_counters, None

        # ── RIDE MODE (cnn_flipped == True) ──
        if abs(velocity) < RIDE_VELOCITY_EXHAUSTED:
            new_counters['ride_vel_bars'] = pos.ride_vel_bars + 1
        else:
            new_counters['ride_vel_bars'] = 0

        if False: # vr exit removed
            new_counters['ride_vr_bars'] = pos.ride_vr_bars + 1
        else:
            new_counters['ride_vr_bars'] = 0

        if reversion > RIDE_REVERSION_HIGH and wick > RIDE_WICK_HIGH:
            new_counters['ride_rev_wick_bars'] = pos.ride_rev_wick_bars + 1
        else:
            new_counters['ride_rev_wick_bars'] = 0

        required = pos.ride_exit_bars if pos.ride_exit_bars > 0 else RIDE_EXIT_BARS

        if new_counters['ride_vel_bars'] >= required:
            return new_counters, 'ride_velocity_exhausted'
        if new_counters['ride_vr_bars'] >= required:
            return new_counters, 'ride_regime_shift'
        if new_counters['ride_rev_wick_bars'] >= required:
            return new_counters, 'ride_reversion_wick'

        return new_counters, None
