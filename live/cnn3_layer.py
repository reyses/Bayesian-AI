"""
CNN3Layer — Three-layer CNN live inference module.

Encapsulates 29D feature computation + L1/L2/L3 inference for live trading.
Called from LiveEngine on each 1m bar.

Layers:
  L1 (StatePredictor 29D): direction prediction
  L2 (DurationPredictor): hold duration commitment
  L3 (RetreatPredictor): per-bar retreat signal

Usage:
  module = CNN3Layer(checkpoint_dir, device)
  module.on_bar(price, high, low, volume, timestamp, state)  # every 1m bar
  module.on_1s_bar(price, volume, timestamp)  # every 1s bar (MTF features)
  module.on_higher_tf_bar(tf_seconds, price, high, low, volume, timestamp, state)
"""
import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Optional

TICK = 0.25
LOOKBACK = 10


@dataclass
class CNN3Signal:
    """Output from the three-layer CNN module."""
    action: str  # 'ENTER', 'EXIT', 'HOLD', 'NONE'
    direction: str  # 'long', 'short', ''
    confidence: float  # L1 confidence (abs pred_dmi)
    hold_bars: int  # L2 predicted hold duration
    retreat_prob: float  # L3 P(retreat) if in trade
    reason: str  # human-readable reason


class MTFFeatureBuffer:
    """Maintains rolling feature buffers for one timeframe.

    Computes 4 features from raw OHLCV: dmi_diff, z_se, velocity, vol_rel.
    Uses lightweight extraction (no SFE) — same as training pipeline for 1s.
    """

    def __init__(self, tf_name, warmup_bars=60):
        self.tf_name = tf_name
        self.warmup_bars = warmup_bars

        # Rolling price/volume buffers
        self._prices = []
        self._volumes = []
        self._timestamps = []

        # EWM state for DMI proxy
        self._smooth_up = 0.0
        self._smooth_dn = 0.0
        self._alpha = 1.0 / 14  # Wilder's smoothing

        # EWM state for velocity
        self._ewm_vel = 0.0
        self._prev_price = None

    @property
    def n_bars(self):
        return len(self._prices)

    @property
    def ready(self):
        return self.n_bars >= self.warmup_bars

    @property
    def last_timestamp(self):
        return self._timestamps[-1] if self._timestamps else 0

    def add_bar(self, price, volume, timestamp):
        """Add a new bar and update feature state."""
        self._prices.append(price)
        self._volumes.append(volume)
        self._timestamps.append(timestamp)

        # Update DMI proxy (EWM of up/down moves)
        if self._prev_price is not None:
            diff = price - self._prev_price
            up = max(diff, 0.0)
            dn = max(-diff, 0.0)
            self._smooth_up = self._alpha * up + (1 - self._alpha) * self._smooth_up
            self._smooth_dn = self._alpha * dn + (1 - self._alpha) * self._smooth_dn

        # Update velocity EWM
        if self._prev_price is not None:
            raw_vel = price - self._prev_price
            alpha_vel = 2.0 / (5 + 1)  # span=5
            self._ewm_vel = alpha_vel * raw_vel + (1 - alpha_vel) * self._ewm_vel

        self._prev_price = price

    def get_features(self):
        """Return current 4D feature vector: [dmi_diff, z_se, velocity, vol_rel]."""
        feats = np.zeros(4, dtype=np.float32)

        # dmi_diff proxy
        feats[0] = self._smooth_up - self._smooth_dn

        # z_se: z-score over last 60 bars
        n = len(self._prices)
        if n >= 15:
            window = self._prices[max(0, n - 60):]
            mean = np.mean(window)
            std = np.std(window)
            se = std / (len(window) ** 0.5) if len(window) > 1 else std
            feats[1] = (self._prices[-1] - mean) / se if se > 1e-8 else 0.0

        # velocity (EWM smoothed)
        feats[2] = self._ewm_vel

        # vol_rel: volume / 30-bar SMA
        if n >= 1:
            vol_window = self._volumes[max(0, n - 30):]
            vol_avg = np.mean(vol_window) if vol_window else 1.0
            feats[3] = self._volumes[-1] / vol_avg if vol_avg > 0 else 0.0

        return feats

    def add_bar_from_sfe(self, price, volume, timestamp, state):
        """Add bar using SFE state (for 5m/15m/1h where SFE runs anyway)."""
        self._prices.append(price)
        self._volumes.append(volume)
        self._timestamps.append(timestamp)
        self._prev_price = price

        # Store SFE-derived features directly
        self._last_sfe_feats = np.array([
            getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0),
            0.0,  # z_se computed below
            getattr(state, 'velocity', 0.0),
            0.0,  # vol_rel computed below
        ], dtype=np.float32)

        # z_se from price window
        n = len(self._prices)
        if n >= 15:
            window = self._prices[max(0, n - 60):]
            mean = np.mean(window)
            std = np.std(window)
            se = std / (len(window) ** 0.5) if len(window) > 1 else std
            self._last_sfe_feats[1] = (price - mean) / se if se > 1e-8 else 0.0

        # vol_rel
        if n >= 1:
            vol_window = self._volumes[max(0, n - 30):]
            vol_avg = np.mean(vol_window) if vol_window else 1.0
            self._last_sfe_feats[3] = volume / vol_avg if vol_avg > 0 else 0.0

    def get_features_sfe(self):
        """Return SFE-derived 4D features (for higher TFs)."""
        if hasattr(self, '_last_sfe_feats'):
            return self._last_sfe_feats
        return self.get_features()


class CNN3Layer:
    """Three-layer CNN module for live trading."""

    def __init__(self, checkpoint_dir, device=None, horizons=None):
        from core.trade_cnn import StatePredictor
        from core.trade_selector import DurationPredictor
        from core.trade_retreat import RetreatPredictor

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.horizons = horizons or [10]
        n_h = len(self.horizons)
        n_labels = 7 * n_h

        cache_29d = os.path.join(checkpoint_dir, '29d')

        # Load L1
        l1_path = os.path.join(checkpoint_dir, 'best_model.pt')
        ckpt = torch.load(l1_path, map_location=self.device, weights_only=False)
        self.l1_model = StatePredictor(n_features=29, latent_dim=64, n_labels=n_labels).to(self.device)
        self.l1_model.load_state_dict(ckpt['model_state'])
        self.l1_model.eval()
        print(f"[CNN3Layer] L1 loaded from {l1_path}")

        # Load L2
        l2_path = os.path.join(cache_29d, 'l2_model.pt')
        l2_ckpt = torch.load(l2_path, map_location=self.device, weights_only=False)
        self.l2_model = DurationPredictor(input_dim=l2_ckpt['input_dim']).to(self.device)
        self.l2_model.load_state_dict(l2_ckpt['model_state'])
        self.l2_model.eval()
        print(f"[CNN3Layer] L2 loaded from {l2_path}")

        # Load L3
        l3_path = os.path.join(cache_29d, 'l3_model.pt')
        l3_ckpt = torch.load(l3_path, map_location=self.device, weights_only=False)
        self.l3_model = RetreatPredictor(input_dim=l3_ckpt['input_dim']).to(self.device)
        self.l3_model.load_state_dict(l3_ckpt['model_state'])
        self.l3_model.eval()
        print(f"[CNN3Layer] L3 loaded from {l3_path}")

        # MTF feature buffers
        self._buf_1s = MTFFeatureBuffer('1s', warmup_bars=60)
        self._buf_5m = MTFFeatureBuffer('5m', warmup_bars=30)
        self._buf_15m = MTFFeatureBuffer('15m', warmup_bars=20)
        self._buf_1h = MTFFeatureBuffer('1h', warmup_bars=10)

        # 13D base feature state
        self._feat_buffer = []  # rolling 10-bar window of 29D features
        self._vol_buffer = []   # 30-bar volume SMA
        self._price_buffer = [] # 60-bar price window
        self._prev_vel = 0.0

        # Trade state
        self._in_trade = False
        self._pending_entry = None  # set on ENTER signal, cleared on fill
        self._side = ''
        self._entry_fill = 0.0
        self._entry_bar = 0
        self._peak_price = 0.0
        self._predicted_hold = 0
        self._bar_count = 0

        # Config
        self.conf_threshold = 3.0
        self.hard_sl = 40  # backstop only
        self.n_h = n_h

    def warmup_from_atlas(self, atlas_root='DATA/ATLAS_LIVE', n_bars=300):
        """Pre-fill all buffers from ATLAS data so predictions start from bar 1.

        Loads the last n_bars from each TF in ATLAS and feeds them through the buffers.
        No SFE computation — uses raw extraction (same as training pipeline for 1s).
        """
        import glob as _glob

        print(f"[CNN3Layer] Warming up from {atlas_root}...")

        # 1m bars for base 13D features
        _1m_files = sorted(_glob.glob(os.path.join(atlas_root, '60s', '*.parquet')))
        if not _1m_files:
            _1m_files = sorted(_glob.glob(os.path.join(atlas_root, '1m', '*.parquet')))
        if _1m_files:
            df_1m = pd.concat([pd.read_parquet(f) for f in _1m_files], ignore_index=True)
            df_1m = df_1m.sort_values('timestamp').tail(n_bars).reset_index(drop=True)
            for _, row in df_1m.iterrows():
                self._price_buffer.append(row['close'])
                self._vol_buffer.append(row.get('volume', 0))
            # Trim to window sizes
            self._price_buffer = self._price_buffer[-60:]
            self._vol_buffer = self._vol_buffer[-30:]
            self._prev_vel = 0.0
            print(f"  1m: {len(df_1m)} bars -> price/vol buffers ready")

        # 1s bars
        _1s_files = sorted(_glob.glob(os.path.join(atlas_root, '1s', '*.parquet')))
        if _1s_files:
            df_1s = pd.read_parquet(_1s_files[-1])  # last file only (most recent)
            df_1s = df_1s.sort_values('timestamp').tail(500).reset_index(drop=True)
            for _, row in df_1s.iterrows():
                self._buf_1s.add_bar(row['close'], row.get('volume', 0), row['timestamp'])
            print(f"  1s: {len(df_1s)} bars -> MTF buffer ready")

        # Higher TFs
        for tf_name, tf_secs, buf in [
            ('5m', 300, self._buf_5m), ('15m', 900, self._buf_15m), ('1h', 3600, self._buf_1h)
        ]:
            _files = sorted(_glob.glob(os.path.join(atlas_root, tf_name, '*.parquet')))
            if not _files:
                # Try seconds-based folder name
                _files = sorted(_glob.glob(os.path.join(atlas_root, f'{tf_secs}s', '*.parquet')))
            if _files:
                df_tf = pd.concat([pd.read_parquet(f) for f in _files], ignore_index=True)
                df_tf = df_tf.sort_values('timestamp').tail(100).reset_index(drop=True)
                for _, row in df_tf.iterrows():
                    buf.add_bar(row['close'], row.get('volume', 0), row['timestamp'])
                print(f"  {tf_name}: {len(df_tf)} bars -> MTF buffer ready")

        print(f"[CNN3Layer] Warmup complete — predictions active from bar 1")

    def on_1s_bar(self, price, volume, timestamp):
        """Feed 1s bar for MTF feature computation."""
        self._buf_1s.add_bar(price, volume, timestamp)

    def on_higher_tf_bar(self, tf_seconds, price, high, low, volume, timestamp, state=None):
        """Feed 5m/15m/1h bar. If SFE state available, use it."""
        buf = {300: self._buf_5m, 900: self._buf_15m, 3600: self._buf_1h}.get(tf_seconds)
        if buf is None:
            return
        if state is not None:
            buf.add_bar_from_sfe(price, volume, timestamp, state)
        else:
            buf.add_bar(price, volume, timestamp)

    def on_bar(self, price, high, low, volume, timestamp, state) -> CNN3Signal:
        """Process a 1m bar. Returns entry/exit/hold signal.

        Args:
            price: close price
            high: bar high
            low: bar low
            volume: bar volume
            timestamp: unix timestamp
            state: MarketState from SFE
        """
        self._bar_count += 1

        # --- Build 29D feature vector ---
        feat_29d = self._build_29d_features(price, high, low, volume, timestamp, state)
        if feat_29d is None:
            return CNN3Signal('NONE', '', 0, 0, 0, 'warmup')

        # Maintain rolling buffer
        self._feat_buffer.append(feat_29d)
        if len(self._feat_buffer) > LOOKBACK:
            self._feat_buffer = self._feat_buffer[-LOOKBACK:]

        if len(self._feat_buffer) < LOOKBACK:
            return CNN3Signal('NONE', '', 0, 0, 0, 'warmup')

        # --- L1: Direction prediction ---
        x = np.array(self._feat_buffer, dtype=np.float32)
        x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.l1_model(x_t).cpu().numpy()[0]

        pred_dmi = pred[(self.n_h - 1) * 7]
        confidence = abs(pred_dmi)
        pred_dir = 'long' if pred_dmi > 0 else 'short'

        all_agree = True
        for hi in range(1, self.n_h):
            if np.sign(pred[hi * 7]) != np.sign(pred[0]):
                all_agree = False
                break

        # --- If in trade: check exits ---
        if self._in_trade:
            return self._check_exits(price, high, low, timestamp, pred, pred_dir, confidence, feat_29d)

        # --- Entry evaluation ---
        if self._pending_entry:
            return CNN3Signal('NONE', '', confidence, 0, 0, 'pending_fill')

        if confidence <= self.conf_threshold or not all_agree:
            return CNN3Signal('NONE', '', confidence, 0, 0, 'no_signal')

        # L2: take/skip + duration
        l2_feat = self._build_l2_features(pred, feat_29d, timestamp, confidence)
        l2_input = torch.FloatTensor(l2_feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_take, hold_bars = self.l2_model(l2_input)

        if p_take.item() < 0.5:
            return CNN3Signal('NONE', '', confidence, 0, 0, 'l2_skip')

        hold = max(3, int(hold_bars.item()))

        # Signal entry — do NOT set _in_trade yet.
        # Live engine calls on_fill() after NT8 confirms the entry.
        self._pending_entry = {
            'side': pred_dir, 'hold': hold, 'bar': self._bar_count, 'price': price
        }

        return CNN3Signal('ENTER', pred_dir, confidence, hold, 0, 'l1_signal')

    def on_fill(self, fill_price, side):
        """Called when entry fill confirmed — NOW we're in a trade."""
        _signal_price = self._pending_entry['price'] if self._pending_entry else 0.0
        _slip = (fill_price - _signal_price) / TICK if _signal_price > 0 else 0.0

        self._in_trade = True
        self._entry_fill = fill_price
        self._side = side
        self._peak_price = fill_price
        if self._pending_entry:
            self._predicted_hold = self._pending_entry['hold']
            self._entry_bar = self._pending_entry['bar']
            self._pending_entry = None
        else:
            self._predicted_hold = 10
            self._entry_bar = self._bar_count

        print(f"[CNN3] FILL CONFIRMED: {side.upper()} @ {fill_price:.2f} "
              f"(signal={_signal_price:.2f} slip={_slip:+.1f}t) "
              f"hold={self._predicted_hold} bars")

    def on_exit(self):
        """Called when position closed — reset trade state."""
        self._in_trade = False
        self._pending_entry = None
        self._side = ''
        self._entry_fill = 0.0
        self._peak_price = 0.0
        self._predicted_hold = 0

    def _check_exits(self, price, high, low, timestamp, pred, pred_dir, confidence, feat_29d):
        """Check exit conditions while in trade."""
        side = self._side

        # Track peak
        if side == 'long':
            self._peak_price = max(self._peak_price, high)
            pnl_close = (price - self._entry_fill) / TICK
            pnl_worst = (low - self._entry_fill) / TICK
        else:
            self._peak_price = min(self._peak_price, low)
            pnl_close = (self._entry_fill - price) / TICK
            pnl_worst = (self._entry_fill - high) / TICK

        peak_pnl = (self._peak_price - self._entry_fill) / TICK if side == 'long' \
            else (self._entry_fill - self._peak_price) / TICK
        drawdown = peak_pnl - pnl_close
        bars_held = self._bar_count - self._entry_bar

        # 1. Hard SL backstop
        if pnl_worst <= -self.hard_sl:
            self._in_trade = False
            return CNN3Signal('EXIT', side, confidence, 0, 1.0, 'hard_sl')

        # 2. L3 retreat
        pred_dmi = pred[(self.n_h - 1) * 7]
        side_agree = 1.0 if pred_dir == side else -1.0

        l3_feat = np.array([
            pnl_close,
            peak_pnl,
            drawdown,
            bars_held / max(1, self._predicted_hold),
            pred_dmi,
            confidence,
            side_agree,
            float(feat_29d[4]),   # velocity
            float(feat_29d[5]),   # z_se
            float(feat_29d[2]),   # vol_rel
            float(feat_29d[9]),   # bar_range
            float(feat_29d[1]),   # dmi_gap
        ], dtype=np.float32)

        l3_input = torch.FloatTensor(l3_feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_retreat = self.l3_model(l3_input).item()

        if p_retreat > 0.5:
            self._in_trade = False
            return CNN3Signal('EXIT', side, confidence, 0, p_retreat, 'retreat')

        # 3. Duration exit
        if bars_held >= self._predicted_hold:
            self._in_trade = False
            return CNN3Signal('EXIT', side, confidence, 0, p_retreat, 'duration')

        # 4. Hold
        return CNN3Signal('HOLD', side, confidence, self._predicted_hold - bars_held,
                          p_retreat, 'hold')

    def _build_29d_features(self, price, high, low, volume, timestamp, state):
        """Assemble 29D feature vector from current bar + MTF buffers."""
        feat = np.zeros(29, dtype=np.float32)

        # --- 13D base (same as extract_features_13d) ---
        dmi_p = getattr(state, 'dmi_plus', 0.0)
        dmi_m = getattr(state, 'dmi_minus', 0.0)
        vel = getattr(state, 'velocity', 0.0)

        self._price_buffer.append(price)
        if len(self._price_buffer) > 60:
            self._price_buffer = self._price_buffer[-60:]

        self._vol_buffer.append(volume)
        if len(self._vol_buffer) > 30:
            self._vol_buffer = self._vol_buffer[-30:]

        vol_avg = np.mean(self._vol_buffer) if self._vol_buffer else 1.0
        if vol_avg <= 0:
            vol_avg = 1.0

        feat[0] = dmi_p - dmi_m                                    # dmi_diff
        feat[1] = abs(dmi_p - dmi_m)                               # dmi_gap
        feat[2] = volume / vol_avg                                  # vol_rel
        if len(self._price_buffer) > 1:
            _dir = 1.0 if price > self._price_buffer[-2] else -1.0
            feat[3] = _dir * volume / vol_avg                      # dir_vol
        feat[4] = vel                                               # velocity
        if len(self._price_buffer) >= 15:
            _mean = np.mean(self._price_buffer)
            _std = np.std(self._price_buffer)
            _se = _std / (len(self._price_buffer) ** 0.5)
            feat[5] = (price - _mean) / _se if _se > 1e-8 else 0.0  # z_se
        feat[6] = vel - self._prev_vel                              # price_accel
        self._prev_vel = vel

        # Regime
        if len(self._price_buffer) >= 30:
            feat[7] = np.std(self._price_buffer[-30:])              # std_price
            if len(self._price_buffer) >= 60:
                short_std = np.std(self._price_buffer[-10:])
                long_std = np.std(self._price_buffer)
                feat[8] = short_std / long_std if long_std > 1e-8 else 1.0  # variance_ratio

        bar_range = high - low
        feat[9] = bar_range / TICK                                  # bar_range
        if bar_range > 0:
            body = abs(price - (high + low) / 2 * 2 - price)       # approximate open
            feat[10] = 1.0 - (abs(price - low) / bar_range)        # wick_ratio (approx)

        # Context
        if len(self._price_buffer) >= 30 and len(self._vol_buffer) >= 30:
            prices_30 = self._price_buffer[-30:]
            vols_30 = self._vol_buffer[-30:]
            vwap = np.sum(np.array(prices_30) * np.array(vols_30)) / (np.sum(vols_30) + 1e-8)
            feat[11] = (price - vwap) / TICK                       # vwap_distance
        feat[12] = (timestamp % 86400) / 86400.0                   # time_of_day

        # --- 16D MTF features (4 per TF) ---
        # 1s
        if self._buf_1s.ready:
            feat[13:17] = self._buf_1s.get_features()
        # 5m
        if self._buf_5m.ready:
            feat[17:21] = self._buf_5m.get_features_sfe()
        # 15m
        if self._buf_15m.ready:
            feat[21:25] = self._buf_15m.get_features_sfe()
        # 1h
        if self._buf_1h.ready:
            feat[25:29] = self._buf_1h.get_features_sfe()

        # Need at least warmup period
        if len(self._price_buffer) < 15:
            return None

        return feat

    def _build_l2_features(self, pred, feat_29d, timestamp, confidence):
        """Build L2 input features (15D)."""
        n_h = self.n_h
        h_start = (n_h - 1) * 7

        l2_feat = np.zeros(15, dtype=np.float32)
        l2_feat[0:7] = pred[h_start:h_start + 7]
        l2_feat[7] = float(feat_29d[7])    # std_price
        l2_feat[8] = float(feat_29d[8])    # variance_ratio
        l2_feat[9] = float(feat_29d[9])    # bar_range
        l2_feat[10] = float(feat_29d[10])  # wick_ratio
        ts_sec = int(timestamp) % 86400
        l2_feat[11] = ts_sec / 86400.0
        l2_feat[12] = max(0, (23 * 3600 - ts_sec)) / 3600.0
        l2_feat[13] = confidence
        l2_feat[14] = 0.5  # bars_since_last normalized (no history in live)
        return l2_feat
