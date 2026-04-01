"""
CNN Augmentor — bridges StatePredictor CNN to the template system.

Loads a frozen StatePredictor checkpoint, maintains a rolling 10-bar
feature buffer, and produces:
  - 7D predicted features at t+5 (to augment 16D template centroids → 23D)
  - Direction signal (LONG/SHORT/None from predicted dmi_diff)

Usage:
  augmentor = CNNAugmentor('checkpoints/trade_cnn/best_model.pt')
  augmentor.update(state, price, high, low, open_price, volume, timestamp)
  cnn_7d = augmentor.get_augmented_7d()   # (7,) or zeros
  direction = augmentor.get_direction()    # 'LONG', 'SHORT', or None
"""
import numpy as np
import torch

from core.trade_cnn import StatePredictor

# CNN feature names — must match training/train_trade_cnn.py exactly
FEATURE_NAMES_7D = ['dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol',
                    'velocity', 'z_se', 'price_accel']
FEATURE_NAMES_13D = FEATURE_NAMES_7D + [
    'std_price', 'variance_ratio', 'bar_range', 'wick_ratio',
    'vwap_distance', 'time_of_day',
]
CNN_LOOKBACK = 10       # bars needed before first prediction
CNN_FEATURES_DIM = 7    # 7D predicted state appended to 16D templates
CNN_N_FEATURES = 13     # 13D input to StatePredictor
TICK = 0.25             # MNQ tick size


class CNNAugmentor:
    """Bridges the StatePredictor CNN to the template clustering system.

    Maintains rolling 13D features, produces 7D predictions + direction.
    Falls back to zeros when not ready (< CNN_LOOKBACK bars).
    """

    def __init__(self, checkpoint_path: str, horizon_index: int = 1,
                 direction_threshold: float = 2.0, device: str = 'cuda'):
        """Load frozen StatePredictor.

        Args:
            checkpoint_path: path to best_model.pt
            horizon_index: index into HORIZONS=[1,5,10], default 1 = t+5
            direction_threshold: min |predicted dmi_diff| for direction signal
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.horizon_index = horizon_index
        self.direction_threshold = direction_threshold

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model_state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
        self.model = StatePredictor(
            n_features=CNN_N_FEATURES,
            latent_dim=64,
            n_labels=21,  # 7 features × 3 horizons
        ).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

        # Rolling buffers
        self._buffer = np.zeros((CNN_LOOKBACK, CNN_N_FEATURES), dtype=np.float32)
        self._bar_count = 0
        self._prices = np.zeros(100, dtype=np.float64)   # rolling price window
        self._volumes = np.zeros(100, dtype=np.float64)   # rolling volume window
        self._prev_velocity = 0.0

        # Cached prediction
        self._last_7d = np.zeros(CNN_FEATURES_DIM, dtype=np.float32)
        self._last_direction = None

    def update(self, state, price: float, high: float, low: float,
               open_price: float, volume: float, timestamp: float) -> None:
        """Feed one bar of data. Computes 13D features and updates rolling buffer.

        Args:
            state: MarketState from SFE (has dmi_plus, dmi_minus, velocity)
            price: close price
            high, low, open_price: bar OHLC
            volume: bar volume
            timestamp: unix timestamp
        """
        idx = self._bar_count
        # Shift rolling windows
        if idx >= len(self._prices):
            self._prices = np.roll(self._prices, -1)
            self._volumes = np.roll(self._volumes, -1)
            self._prices[-1] = price
            self._volumes[-1] = volume
            w_idx = len(self._prices) - 1
        else:
            self._prices[idx] = price
            self._volumes[idx] = volume
            w_idx = idx

        # Extract 13D features for this bar (mirrors extract_features_13d)
        feat = np.zeros(CNN_N_FEATURES, dtype=np.float32)

        dmi_p = getattr(state, 'dmi_plus', 0.0)
        dmi_m = getattr(state, 'dmi_minus', 0.0)
        vel = getattr(state, 'velocity', 0.0)

        # Volume SMA (30-bar or available)
        n_avail = min(idx + 1, 30)
        vol_start = max(0, w_idx - n_avail + 1)
        vol_avg = self._volumes[vol_start:w_idx + 1].mean()
        if vol_avg <= 0:
            vol_avg = 1.0

        # --- 7D directional ---
        feat[0] = dmi_p - dmi_m                               # dmi_diff
        feat[1] = abs(dmi_p - dmi_m)                          # dmi_gap
        feat[2] = volume / vol_avg                             # vol_rel
        if idx > 0:
            prev_price = self._prices[max(0, w_idx - 1)]
            _dir = 1.0 if price > prev_price else -1.0
            feat[3] = _dir * volume / vol_avg                 # dir_vol
        feat[4] = vel                                          # velocity
        if idx >= 15:
            _win = self._prices[max(0, w_idx - 60):w_idx + 1]
            _mean = _win.mean()
            _std = _win.std()
            _se = _std / (len(_win) ** 0.5) if len(_win) > 1 else _std
            feat[5] = (price - _mean) / _se if _se > 1e-8 else 0.0  # z_se
        feat[6] = vel - self._prev_velocity                    # price_accel
        self._prev_velocity = vel

        # --- 4D regime ---
        if idx >= 30:
            feat[7] = np.std(self._prices[max(0, w_idx - 30):w_idx + 1])  # std_price
            if idx >= 60:
                _short_std = np.std(self._prices[max(0, w_idx - 10):w_idx + 1])
                _long_std = np.std(self._prices[max(0, w_idx - 60):w_idx + 1])
                feat[8] = _short_std / _long_std if _long_std > 1e-8 else 1.0

        _range = high - low
        feat[9] = _range / TICK                                # bar_range
        if _range > 0:
            feat[10] = 1.0 - (abs(price - open_price) / _range)  # wick_ratio

        # --- 2D context ---
        if idx >= 30:
            _p = self._prices[max(0, w_idx - 30):w_idx + 1]
            _v = self._volumes[max(0, w_idx - 30):w_idx + 1]
            _vwap_num = np.sum(_p * _v)
            _vwap_den = np.sum(_v)
            _vwap = _vwap_num / _vwap_den if _vwap_den > 0 else price
            feat[11] = (price - _vwap) / TICK                 # vwap_distance

        feat[12] = (timestamp % 86400) / 86400                # time_of_day

        # Push into rolling buffer
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = feat
        self._bar_count += 1

        # Run CNN if we have enough bars
        if self._bar_count >= CNN_LOOKBACK:
            self._predict()

    def _predict(self):
        """Run StatePredictor on the current 10-bar buffer."""
        with torch.no_grad():
            x = torch.FloatTensor(self._buffer).unsqueeze(0).to(self.device)
            pred = self.model(x)[0].cpu().numpy()  # (21,) = 7 × 3 horizons

        # Extract 7D at selected horizon
        start = self.horizon_index * CNN_FEATURES_DIM
        self._last_7d = pred[start:start + CNN_FEATURES_DIM].copy()

        # Direction from predicted dmi_diff (index 0 of the 7D)
        predicted_dmi = self._last_7d[0]
        if abs(predicted_dmi) >= self.direction_threshold:
            self._last_direction = 'LONG' if predicted_dmi > 0 else 'SHORT'
        else:
            self._last_direction = None  # uncertain

    def get_augmented_7d(self) -> np.ndarray:
        """Return 7D predicted features or zeros if not ready."""
        if self._bar_count < CNN_LOOKBACK:
            return np.zeros(CNN_FEATURES_DIM, dtype=np.float32)
        return self._last_7d.copy()

    def get_direction(self):
        """Return 'LONG', 'SHORT', or None."""
        if self._bar_count < CNN_LOOKBACK:
            return None
        return self._last_direction

    def is_ready(self) -> bool:
        """True when CNN has enough bars for prediction."""
        return self._bar_count >= CNN_LOOKBACK

    def reset(self):
        """Clear all state (for new session/day)."""
        self._buffer[:] = 0
        self._bar_count = 0
        self._prices[:] = 0
        self._volumes[:] = 0
        self._prev_velocity = 0.0
        self._last_7d[:] = 0
        self._last_direction = None

    def preload_from_atlas(self, atlas_root: str = 'DATA/ATLAS', tf: str = '15s',
                           n_bars: int = 60):
        """Pre-warm the CNN buffer from historical ATLAS data.

        Loads the last n_bars from ATLAS, computes SFE states + 13D features,
        and feeds them into the rolling buffer. After this, is_ready() = True
        and the CNN can predict from bar 1 in live.
        """
        import glob as _glob
        import pandas as pd
        from core.statistical_field_engine import StatisticalFieldEngine
        from training.train_trade_cnn import extract_features_13d

        files = sorted(_glob.glob(f'{atlas_root}/{tf}/*.parquet'))
        if not files:
            return

        # Load last file (most recent data)
        df = pd.read_parquet(files[-1]).sort_values('timestamp').reset_index(drop=True)
        if len(df) < n_bars:
            # Concat last two files
            if len(files) >= 2:
                df2 = pd.read_parquet(files[-2])
                df = pd.concat([df2, df], ignore_index=True).sort_values('timestamp').reset_index(drop=True)

        # Take last n_bars
        df = df.tail(n_bars).reset_index(drop=True)
        if len(df) < CNN_LOOKBACK:
            return

        # Compute features
        sfe = StatisticalFieldEngine()
        states = sfe.batch_compute_states(df)
        feats = extract_features_13d(states, df)

        # Feed into buffer bar by bar
        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        timestamps = df['timestamp'].values

        for i in range(len(df)):
            # Directly push precomputed features into buffer
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = feats[i]

            idx = self._bar_count
            if idx < len(self._prices):
                self._prices[idx] = prices[i]
                self._volumes[idx] = volumes[i]
            else:
                self._prices = np.roll(self._prices, -1)
                self._volumes = np.roll(self._volumes, -1)
                self._prices[-1] = prices[i]
                self._volumes[-1] = volumes[i]

            self._bar_count += 1

        # Run final prediction
        if self._bar_count >= CNN_LOOKBACK:
            self._predict()

        del df, states, feats, sfe
        print(f"  [CNNAugmentor] Preloaded {self._bar_count} bars from ATLAS/{tf} — ready={self.is_ready()}")
