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
EXHAUST_Z_MIN = 1.4               # entry: must be deep in z extreme
EXHAUST_VR_MIN = 0.70             # entry: must be trending (real exhaustion, not chop)
EXHAUST_CONVICTION_BARS = 12      # exit: 1 minute to prove reversal
EXHAUST_Z_SHRINK_MIN = 0.20      # exit: |z| must shrink 20%+ from entry

# MTF_EXHAUSTION (from EDA: 17% WR, winners are deep z + high vr + high vol)
MTF_Z_MIN = 1.4                   # entry: must be deep in z extreme
MTF_VR_MIN = 0.58                 # entry: must be somewhat trending
MTF_VOL_MIN = 2.0                 # entry: must have volume conviction
MTF_CONVICTION_BARS = 12          # exit: 1 minute to prove thesis
MTF_Z_SHRINK_MIN = 0.10           # exit: |z| must shrink 10%+ from entry

# Bar range gate — disabled (low bar_range still profitable via volume)
BAR_RANGE_MIN = 0.0  # set to ~30 to activate (filters tight chop)

# Cascade: 1h z alignment
H1_Z_MIN = 1.0

# Tier 1-2 exit (kill shot / cascade)
P_CENTER_EXIT = 0.60
P_CENTER_EXIT_BARS_CASCADE = 3   # cascade: 3-bar confirmation (rare, high conviction)
P_CENTER_EXIT_BARS_KILLSHOT = 2  # kill shot: 2-bar confirmation ($740 config)

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

# Tiered RIDE exits — stronger 1h = more patient
RIDE_EXIT_BARS_TIERS = {
    'strong': 5,    # |1h_z| > 2.0 at entry — hold longer, high conviction
    'medium': 3,    # |1h_z| 1.5-2.0 — standard
    'weak': 2,      # |1h_z| < 1.5 — exit fast, weak signal
}

# 79D absolute indices
# 91D feature indices (12 core per TF, helpers at 72+)
# TF order: 15s=0, 1m=1, 5m=2, 15m=3, 1h=4, 1D=5
# Core offset per TF: tf_idx * 12
_1M_OFFSET = 12       # TF1 * 12 (was 10)
_Z = 0
_VR = 2
_5M_WICK_IDX = 80     # helper_start(72) + TF2*3 + 2 (was 68)
_15M_WICK_IDX = 83    # helper_start(72) + TF3*3 + 2 (was 71)
_1H_Z_IDX = 48        # TF4 * 12 (was 40)
_1H_VELOCITY_IDX = 51 # TF4*12 + 3 (was 43)
_1M_P_CENTER_IDX = 21 # TF1*12 + 9 (was 19)
_1M_VELOCITY_IDX = 15 # TF1*12 + 3 (was 13)
_5M_BAR_RANGE_IDX = 30 # TF2*12 + 6 (was 26)
_5M_VELOCITY_IDX = 27  # TF2*12 + 3 (was 23)
_5M_ACCEL_IDX = 28     # TF2*12 + 4 (was 24)
_1M_HURST_IDX = 19     # TF1*12 + 7 (was 17)
_1M_VOL_REL_IDX = 17   # TF1*12 + 5 (was 15)
_1M_DMI_IDX = 13       # TF1*12 + 1 (was 11)
_1M_WICK_IDX = 77      # helper_start(72) + TF1*3 + 2 (was 65)
_1M_REVERSION_IDX = 20 # TF1*12 + 8 (was 18)

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
H1_AGAINST_Z_MIN = 1.5  # |1h_z| must be this extreme to count as "against"

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

    def __init__(self, use_cnn=True, release_dir=None):
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
                'features_79d': feat.copy(),
            })

            # Hard stop — fires every bar, overrides everything
            if pnl <= HARD_STOP:
                self._close_trade(price, ts, time_str, 'hard_stop', feat)

            # Exit logic — every bar (5s cadence matches training)
            else:
                # 1. Giveback stop (all tiers)
                if self.peak_pnl >= GIVEBACK_MIN_PEAK and pnl < self.peak_pnl * GIVEBACK_KEEP:
                    self._close_trade(price, ts, time_str, 'giveback_stop', feat)
                    return

                # 2. Per-tier CNN exit + loser (confidence-gated, confirmation required)
                entry_z = self.entry_1m.get('z_se', 0) if self.entry_1m else 0
                tier_pred = self._predict_exit_loser(
                    feat, self.entry_tier, self.bars_held, pnl, self.peak_pnl, entry_z)

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
                    exit_reason = self._check_exit(feat, z, vr, pnl)
                    if exit_reason:
                        self._close_trade(price, ts, time_str, exit_reason, feat)

        # === ENTRY CHECK — 1m boundaries only ===
        if not self.in_pos and is_1m and price > 100:  # sanity: reject price=0
            # Path 1: NMP entry (z extreme + vr < 1)
            if abs(z) > ROCHE and vr < VR_ENTRY:
                # Bar range gate (disabled by default)
                if feat[_5M_BAR_RANGE_IDX] < BAR_RANGE_MIN:
                    return  # skip — too tight

                # Classify the full ExNMP tier + direction
                direction, tier, cnn_flipped = self._classify_full_tier(feat, z)

                if tier is None:
                    return  # breakout filter rejected this bar

                # Per-tier CNN: entry gate + direction override + duration
                if self.use_cnn and tier in self._tier_cnn_enabled:
                    pred = self._predict_entry_direction(feat, tier)
                    if pred is not None:
                        entry_conf, cnn_dir, dir_conf, dur_class, dur_conf = pred

                        # Entry gate: skip if CNN says bad entry
                        if entry_conf < ENTRY_GATE_MIN:
                            return  # CNN vetoed this entry

                        # Direction override: if CNN is confident, use its direction
                        if dir_conf > DIRECTION_CONFIDENCE_MIN and cnn_dir != direction:
                            direction = cnn_dir
                            cnn_flipped = True

                        # Duration: set exit patience
                        if dur_conf > DURATION_CONFIDENCE_MIN:
                            self._cnn_duration_class = dur_class

                self._open_trade(direction, price, ts, time_str, feat, tier,
                                 cnn_flipped=cnn_flipped)

            # Path 2: Non-NMP entries (z NOT extreme, other physics trigger)
            elif abs(z) <= ROCHE and vr < VR_ENTRY:
                hurst = feat[_1M_HURST_IDX]
                v5 = abs(feat[_5M_VELOCITY_IDX])
                v5_accel = feat[_5M_ACCEL_IDX]
                v1 = abs(feat[_1M_VELOCITY_IDX])
                dmi = feat[_1M_DMI_IDX]

                # REGIME_FLIP: vr low + hurst low = regime shift
                # EDA: 27% WR fading → 73% WR riding. Ride z, don't fade.
                if vr < REGIME_VR_MAX and hurst < REGIME_HURST_MAX:
                    direction = 'long' if z > 0 else 'short'
                    self._open_trade(direction, price, ts, time_str, feat, 'REGIME_FLIP',
                                     cnn_flipped=False)

                # MTF_EXHAUSTION: 5m decelerating + 1m alive + deep z + vr + volume
                # EDA: 13% WR fading → 76% WR riding. Exhaustion = continuation, not reversal.
                # RIDE the 5m direction (same sign), don't fade it.
                elif (v5_accel < 0 and v5 > MTF_5M_VEL_MIN and v1 > MTF_1M_VEL_ALIVE and
                      abs(z) > MTF_Z_MIN and vr > MTF_VR_MIN and
                      feat[_1M_VOL_REL_IDX] > MTF_VOL_MIN):
                    direction = 'long' if feat[_5M_VELOCITY_IDX] > 0 else 'short'
                    self._open_trade(direction, price, ts, time_str, feat, 'MTF_EXHAUSTION',
                                     cnn_flipped=False)

                # EXHAUSTION_BAR: bar_range climax + decelerating + deep z + trending + DMI
                # EDA: |dmi|>15 lifts WR from 38% to 44%, removes weak signals
                elif (feat[_1M_OFFSET + 6] > EXHAUST_BAR_RANGE_MIN and       # bar_range climax
                      abs(feat[_1M_OFFSET + 4]) > EXHAUST_ACCEL_MIN and      # |acceleration|
                      feat[_1M_OFFSET + 4] * feat[_1M_VELOCITY_IDX] < 0 and  # decelerating
                      abs(z) > EXHAUST_Z_MIN and                              # deep in z extreme
                      vr > EXHAUST_VR_MIN and                                 # trending
                      abs(feat[_1M_DMI_IDX]) > 15.0):                         # DMI committed
                    direction = 'short' if feat[_1M_VELOCITY_IDX] > 0 else 'long'
                    self._open_trade(direction, price, ts, time_str, feat, 'EXHAUSTION_BAR',
                                     cnn_flipped=False)

                # ABSORPTION: high volume + low range + wicks
                # EDA: 24% WR fading → 76% WR riding. Absorption = continuation.
                # RIDE z direction (same sign as z), don't fade it.
                elif (feat[_1M_VOL_REL_IDX] > ABSORB_VOL_MIN and
                      feat[_1M_OFFSET + 6] < ABSORB_RANGE_MAX and  # bar_range
                      feat[_1M_WICK_IDX] > ABSORB_WICK_MIN):
                    direction = 'long' if z > 0 else 'short'
                    self._open_trade(direction, price, ts, time_str, feat, 'ABSORPTION',
                                     cnn_flipped=False)

    def _classify_full_tier(self, feat, z):
        """Classify entry into tiered strategy.

        Returns (direction, tier, cnn_flipped).

        Priority: KILL_SHOT first (massive edge), then by WR descending:
          1. KILL_SHOT:     $13.3/tr, 72% WR — wick rejection, proven edge
          2. FADE_AGAINST:  $19.2/tr, 69% WR — 1h extreme against, keep fading
          3. CASCADE:       $5.4/tr,  67% WR — wick + 1h aligned
          4. FADE_MOMENTUM: $10.1/tr, 58% WR — high velocity, fade z
          5. RIDE_AGAINST:  $8.7/tr,  51% WR — 1h velocity opposes, ride
          6. FADE_CALM:     $1.0/tr,  58% WR — default fade (CNN opportunity)

        KILLED (negative $/trade in max-fill):
          - FREIGHT_TRAIN:    -$116/tr, 45% WR
          - MTF_EXHAUSTION:   -$52/tr,  17% WR
          - EXHAUSTION_BAR:   -$6.5/tr, 50% WR
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

        # Check 1h conditions
        h1_against_fade = ((direction == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                           (direction == 'short' and h1_z < -H1_AGAINST_Z_MIN))
        h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                      (direction == 'short' and h1_z > H1_Z_MIN))
        h1_vel_against = ((direction == 'long' and h1_vel < -H1_AGAINST_Z_MIN) or
                          (direction == 'short' and h1_vel > H1_AGAINST_Z_MIN))

        # 1. KILL_SHOT — $13.3/tr, 72% WR (massive edge, gets priority)
        if has_wick and not h1_aligned:
            return direction, 'KILL_SHOT', False

        # 2. FADE_AGAINST — $19.2/tr, 69% WR
        if h1_against_fade:
            return direction, 'FADE_AGAINST', False

        # 3. CASCADE — $5.4/tr, 67% WR
        if has_wick and h1_aligned:
            return direction, 'CASCADE', False

        # 4. FREIGHT_TRAIN — extreme velocity, accelerating, regime not committed
        #    Entry: |vel| > 100 AND vel*accel > 0 (accelerating) AND vr < 0.85
        #    Direction: ride the velocity (not fade z)
        acceleration = feat[_1M_OFFSET + 4]  # 1m acceleration
        vr = feat[_1M_OFFSET + _VR]
        if (abs_vel >= FREIGHT_TRAIN_THRESHOLD and
                velocity * acceleration > 0 and
                vr < FREIGHT_TRAIN_VR_MAX):
            ft_direction = 'long' if velocity > 0 else 'short'
            return ft_direction, 'FREIGHT_TRAIN', True

        # 5. FADE_MOMENTUM — $10.1/tr, 58% WR
        if abs_vel >= VELOCITY_THRESHOLD:
            return direction, 'FADE_MOMENTUM', False

        # 6. RIDE_AGAINST — $8.7/tr, 51% WR
        if h1_vel_against and not h1_against_fade:
            direction = 'long' if h1_vel > 0 else 'short'
            return direction, 'RIDE_AGAINST', False

        # 7. MTF_BREAKOUT — all TFs aligned, ride the breakout (not fade)
        # EDA: these are the 2,597 bars that killed FADE_CALM (322 hard stops at -$176)
        # When 5m AND 15m z both > 1.3, every TF confirms the move. Ride it.
        z_5m = abs(feat[2 * _N_CORE + _Z])   # 5m z (TF index 2)
        z_15m = abs(feat[3 * _N_CORE + _Z])  # 15m z (TF index 3)
        if z_5m > 1.3 and z_15m > 1.3:
            # Ride z direction (opposite of fade)
            breakout_dir = 'long' if z > 0 else 'short'
            return breakout_dir, 'MTF_BREAKOUT', True

        # 8. FADE_CALM — default fade (CNN opportunity)
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
            acceleration = feat[_1M_OFFSET + 4]
            abs_vel = abs(velocity)
            vel_ratio = abs_vel / max(self._entry_velocity, 1.0)

            if vel_ratio < FREIGHT_TRAIN_VEL_DECAY and velocity * acceleration < 0:
                return 'freight_train_decel'
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
            if vr > REGIME_FLIP_VR_BAIL:
                return 'regime_vr_rising'

            # Mean reached = reversion complete, take profit
            if abs_z < 0.3:
                return 'regime_mean_reached'

            return None

        # MTF_EXHAUSTION exit: RIDING the 5m direction (flipped from fade)
        # Entry: ride 5m momentum. Exit: 5m momentum dying.
        if self.entry_tier == 'MTF_EXHAUSTION':
            v5_vel = feat[_5M_VELOCITY_IDX]
            v5_accel = feat[_5M_ACCEL_IDX]

            # 5m velocity flipped against our direction = ride is over
            if self.direction == 'long' and v5_vel < -10:
                return 'mtf_5m_reversed'
            if self.direction == 'short' and v5_vel > 10:
                return 'mtf_5m_reversed'

            # 5m decelerating hard = momentum exhausting
            if abs(v5_vel) > 20 and v5_vel * v5_accel < 0:
                return 'mtf_5m_decel'

            # VR dropped = regime shifting to mean-reverting (ride done)
            if vr < 0.30:
                return 'mtf_vr_dropped'

            return None

        # MTF_BREAKOUT exit: ride until multi-TF alignment breaks
        if self.entry_tier == 'MTF_BREAKOUT':
            z_5m = abs(feat[2 * _N_CORE + _Z])
            z_15m = abs(feat[3 * _N_CORE + _Z])

            # Alignment broken: either 5m or 15m z dropped below 0.8
            if z_5m < 0.8 or z_15m < 0.8:
                return 'breakout_alignment_lost'

            # 1m z crossed zero (overshot to other side)
            if self.direction == 'long' and z < -0.3:
                return 'breakout_overshot'
            if self.direction == 'short' and z > 0.3:
                return 'breakout_overshot'

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
            vol_rel = feat[_1M_VOL_REL_IDX]

            # Early conviction: z must shrink by 10%+ from entry by bar 12
            if self.bars_held >= ABSORB_CONVICTION_BARS and self.bars_held < ABSORB_CONVICTION_BARS + 3:
                z_shrink = (self._entry_abs_z - abs_z) / max(self._entry_abs_z, 0.01)
                if z_shrink < ABSORB_Z_SHRINK_MIN:
                    return 'absorb_no_conviction'

            # Volume still elevated at bar 24 = absorption failing
            if self.bars_held >= 24 and vol_rel > ABSORB_VOL_PERSIST_MAX:
                return 'absorb_vol_persistent'

            # VR rising = trending against us
            if vr > ABSORB_VR_BAIL:
                return 'absorb_vr_rising'

            # Mean reached = absorption complete
            if abs_z < 0.3:
                return 'absorb_mean_reached'

            return None

        # Other tiers: two exit modes based on trade type
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

            # Phase 0: approaching mean (uniform 3-bar, $740 config)
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
        z = feat[_1M_OFFSET + _Z]
        vr = feat[_1M_OFFSET + _VR]

        # Classify tier from current market state
        tier = 'FADE_CALM'  # safe default — gets full CNN exit management
        cnn_flipped = False

        if abs(z) > ROCHE and vr < VR_ENTRY:
            # NMP conditions met — full tier classification
            _, tier, cnn_flipped = self._classify_full_tier(feat, z)
            # If classified direction differs from manual, mark as flipped
            classified_dir = 'short' if z > 0 else 'long'
            if classified_dir != direction:
                cnn_flipped = True
        else:
            # Non-NMP conditions — pick closest non-NMP tier by physics
            hurst = feat[_1M_HURST_IDX]
            if vr < REGIME_VR_MAX and hurst < REGIME_HURST_MAX:
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
        self.entry_79d = feat.copy()
        self.entry_1m = {
            'z_se': feat[_1M_OFFSET + _Z],
            'vr': feat[_1M_OFFSET + _VR],
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
        self._entry_abs_z = abs(feat[_1M_OFFSET + _Z])     # for REGIME_FLIP/ABSORPTION conviction
        self._entry_vol_rel = feat[_1M_VOL_REL_IDX]       # for ABSORPTION volume fade check
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
