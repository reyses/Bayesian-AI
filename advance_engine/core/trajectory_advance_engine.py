"""
TrajectoryAdvanceEngine — AdvanceEngine powered by trajectory navigation.

Replaces the old entry/exit logic (pattern matching, 13-layer exit cascade,
TBN voting) with TrajectoryPredictor models reading P(D) decay curves.

Same interface as AdvanceEngine.process_bar() for drop-in replacement.

Entry: 1h trend direction + 1m wave starting + 1s timing confirmation
Exit: trajectory shape (inflection, chop, sight contraction, disagreement)
No fixed SL/TP/trail — physics of uncertainty drives all exits.

Hard SL at 40 ticks is the only fixed rule (circuit breaker, should never fire).
"""
import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Dict


TICK = 0.25
LOOKBACK = 10
HARD_SL_TICKS = 40  # circuit breaker only


@dataclass
class NavBarResult:
    """Result from processing one bar."""
    action: str       # 'ENTER', 'EXIT', 'HOLD', 'WAIT'
    direction: str    # 'long', 'short', 'flat'
    confidence: float
    sight: int
    reason: str
    pnl: float = 0.0  # only on EXIT


class FeatureBuffer:
    """Maintains rolling 13D feature buffer for one TF."""

    def __init__(self, lookback=LOOKBACK):
        self.lookback = lookback
        self._buffer = []  # list of (13,) feature vectors
        self._prices = []
        self._volumes = []
        self._prev_vel = 0.0

    @property
    def ready(self):
        return len(self._buffer) >= self.lookback

    def add_bar(self, price, high, low, volume, timestamp, state):
        """Add bar, extract 13D features, append to buffer."""
        feat = np.zeros(13, dtype=np.float32)

        dmi_p = getattr(state, 'dmi_plus', 0.0)
        dmi_m = getattr(state, 'dmi_minus', 0.0)
        vel = getattr(state, 'velocity', 0.0)

        self._prices.append(price)
        self._volumes.append(volume)
        if len(self._prices) > 60:
            self._prices = self._prices[-60:]
        if len(self._volumes) > 30:
            self._volumes = self._volumes[-30:]

        vol_avg = np.mean(self._volumes) if self._volumes else 1.0
        if vol_avg <= 0:
            vol_avg = 1.0

        feat[0] = dmi_p - dmi_m                                     # dmi_diff
        feat[1] = abs(dmi_p - dmi_m)                                 # dmi_gap
        feat[2] = volume / vol_avg                                    # vol_rel
        if len(self._prices) > 1:
            _dir = 1.0 if price > self._prices[-2] else -1.0
            feat[3] = _dir * volume / vol_avg                        # dir_vol
        feat[4] = vel                                                 # velocity
        if len(self._prices) >= 15:
            _mean = np.mean(self._prices)
            _std = np.std(self._prices)
            _se = _std / (len(self._prices) ** 0.5)
            feat[5] = (price - _mean) / _se if _se > 1e-8 else 0.0  # z_se
        feat[6] = vel - self._prev_vel                                # price_accel
        self._prev_vel = vel

        if len(self._prices) >= 30:
            feat[7] = np.std(self._prices[-30:])                      # std_price
            if len(self._prices) >= 60:
                feat[8] = np.std(self._prices[-10:]) / (np.std(self._prices) + 1e-8)  # variance_ratio

        bar_range = high - low
        feat[9] = bar_range / TICK                                    # bar_range
        if bar_range > 0:
            feat[10] = 1.0 - (abs(price - (high + low) / 2) / bar_range)  # wick_ratio approx

        if len(self._prices) >= 30 and len(self._volumes) >= 30:
            p30 = np.array(self._prices[-30:])
            v30 = np.array(self._volumes[-30:])
            vwap = np.sum(p30 * v30) / (np.sum(v30) + 1e-8)
            feat[11] = (price - vwap) / TICK                          # vwap_distance
        feat[12] = (timestamp % 86400) / 86400.0                      # time_of_day

        self._buffer.append(feat)
        if len(self._buffer) > self.lookback:
            self._buffer = self._buffer[-self.lookback:]

        return feat

    def get_window(self):
        """Return (lookback, 13) array for model input."""
        if not self.ready:
            return None
        return np.array(self._buffer[-self.lookback:], dtype=np.float32)


class TrajectoryAdvanceEngine:
    """Drop-in replacement for AdvanceEngine using trajectory navigation.

    Processes bars from multiple TFs. Each TF has its own TrajectoryPredictor
    and feature buffer. The TrajectoryEngine combines them for decisions.
    """

    def __init__(self, checkpoint_base='checkpoints', device=None):
        from core.direction_cnn import TrajectoryPredictor
        from core.trajectory_engine import TrajectoryEngine
        from core.calibration import TrajectoryCalibrator

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models per TF
        self.models: Dict[str, torch.nn.Module] = {}
        self.buffers: Dict[str, FeatureBuffer] = {}
        horizons_per_tf = {}

        tf_configs = {
            '1h': f'{checkpoint_base}/trajectory_1h/best_model.pt',
            '1m': f'{checkpoint_base}/trajectory_1m/best_model.pt',
            '1s': f'{checkpoint_base}/trajectory_1s/best_model.pt',
        }

        for tf, path in tf_configs.items():
            if not os.path.exists(path):
                print(f"[NavEngine] {tf}: no model at {path}, skipping")
                continue

            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            horizons = ckpt.get('horizons', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            model = TrajectoryPredictor(
                n_features=13, latent_dim=64, n_state=7, horizons=horizons
            ).to(self.device)
            model.load_state_dict(ckpt['model_state'])
            model.eval()

            self.models[tf] = model
            self.buffers[tf] = FeatureBuffer(lookback=LOOKBACK)
            horizons_per_tf[tf] = horizons
            print(f"[NavEngine] {tf}: loaded (horizons={horizons})")

        # Load calibrators if available
        calibrators = {}
        for tf in self.models:
            cal_path = f'{checkpoint_base}/trajectory_{tf}/calibration.json'
            if os.path.exists(cal_path):
                calibrators[tf] = TrajectoryCalibrator.load(cal_path)
                print(f"[NavEngine] {tf}: calibration loaded")

        # Trajectory engine
        self.engine = TrajectoryEngine(
            calibrators=calibrators,
            horizons_per_tf=horizons_per_tf,
        )

        # Trade state
        self._in_trade = False
        self._trade_dir = ''
        self._entry_price = 0.0
        self._entry_bar = 0
        self._peak_price = 0.0
        self._bar_count = 0

    def process_bar(self, tf, price, high, low, volume, timestamp, state):
        """Process one bar for a given TF.

        Call this for each TF bar as they arrive:
          engine.process_bar('1h', ...)  # every 1h bar
          engine.process_bar('1m', ...)  # every 1m bar
          engine.process_bar('1s', ...)  # every 1s bar

        Returns NavBarResult with action.
        """
        if tf not in self.models:
            return NavBarResult('WAIT', 'flat', 0, 0, f'{tf}_no_model')

        self._bar_count += 1
        buf = self.buffers[tf]
        buf.add_bar(price, high, low, volume, timestamp, state)

        if not buf.ready:
            return NavBarResult('WAIT', 'flat', 0, 0, 'warmup')

        # Get trajectory from model
        window = buf.get_window()
        x_t = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, p_longs = self.models[tf](x_t)
        raw_curve = p_longs.cpu().numpy()[0]

        # Update trajectory engine
        self.engine.update(tf, raw_curve)

        # Only make decisions on 1m bars (primary TF)
        if tf != '1m':
            return NavBarResult('HOLD', self._trade_dir or 'flat', 0, 0, f'{tf}_update_only')

        # Hard SL check (circuit breaker)
        if self._in_trade:
            if self._trade_dir == 'long':
                self._peak_price = max(self._peak_price, high)
                pnl_worst = (low - self._entry_price) / TICK
            else:
                self._peak_price = min(self._peak_price, low)
                pnl_worst = (self._entry_price - high) / TICK

            if pnl_worst <= -HARD_SL_TICKS:
                pnl = -HARD_SL_TICKS
                self._in_trade = False
                self.engine.on_exit()
                return NavBarResult('EXIT', self._trade_dir, 0, 0, 'hard_sl', pnl=pnl * TICK * 2)

        # Get trajectory decision
        action = self.engine.decide()

        if action.action == 'ENTER' and not self._in_trade:
            self._in_trade = True
            self._trade_dir = action.direction
            self._entry_price = price
            self._entry_bar = self._bar_count
            self._peak_price = price
            return NavBarResult('ENTER', action.direction, action.confidence,
                                action.sight, action.reason)

        if action.action == 'EXIT' and self._in_trade:
            if self._trade_dir == 'long':
                pnl = (price - self._entry_price) / TICK
            else:
                pnl = (self._entry_price - price) / TICK
            self._in_trade = False
            self.engine.on_exit()
            return NavBarResult('EXIT', self._trade_dir, action.confidence,
                                action.sight, action.reason, pnl=pnl)

        if action.action == 'HOLD':
            return NavBarResult('HOLD', self._trade_dir, action.confidence,
                                action.sight, action.reason)

        return NavBarResult('WAIT', 'flat', action.confidence, action.sight, action.reason)

    def on_fill(self, fill_price, direction):
        """Called when entry fill confirmed."""
        self._entry_price = fill_price
        self._trade_dir = direction
        self._peak_price = fill_price
        self._in_trade = True
        self.engine.on_fill(direction)

    def on_exit(self):
        """Called when position closed externally."""
        self._in_trade = False
        self._trade_dir = ''
        self.engine.on_exit()
