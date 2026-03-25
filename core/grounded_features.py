"""
Grounded Feature Extractor — 6 features × 8 TFs = 48D per bar.

Features (all grounded in Price, Time, Volume):
  1. dmi_diff    — (DMI+ - DMI-) directional battle (level 2: Price/Time)
  2. dmi_gap     — |DMI+ - DMI-| battle intensity (level 2)
  3. volume_rel  — volume / rolling_avg(30) relative participation (level 2: Volume/Time)
  4. velocity    — price change / tick_size normalized speed (level 2: Price/Time)
  5. z_se        — (price - mean) / SE significance of deviation (level 3)
  6. price_accel — velocity change, is force changing? (level 3: Price/Time²)

TFs: 1m, 3m, 5m, 15m, 30m, 1h, 4h, noise_floor(1s std)
Total: 6 × 8 = 48 dimensions

Interface:
  - extract(states_by_tf: dict) -> np.array(48,)
  - extract_batch(all_states: dict) -> np.array(N, 48)
  - FEATURE_NAMES: list of 48 names for inspection
"""
import numpy as np
from collections import deque


# TFs used for template features (ordered fast -> slow)
# 1s = noise floor, 1m-4h = trading scales, 1D/1W = macro direction (orthogonal)
TEMPLATE_TFS = ['1s', '1m', '3m', '5m', '15m', '30m', '1h', '4h', '1D', '1W']

# 7 grounded features per TF
FEATURES_PER_TF = ['dmi_diff', 'dmi_gap', 'volume_rel', 'dir_volume', 'velocity', 'z_se', 'price_accel']

# Full feature names for inspection
FEATURE_NAMES = [f'{tf}_{feat}' for tf in TEMPLATE_TFS for feat in FEATURES_PER_TF]

N_FEATURES = len(FEATURE_NAMES)  # 48


class GroundedFeatureExtractor:
    """Extracts 48D grounded feature vector from multi-TF market states."""

    def __init__(self, vol_window: int = 30, se_window: int = 15):
        self.vol_window = vol_window
        self.se_window = se_window

        # Rolling buffers per TF
        self._volumes = {tf: deque(maxlen=vol_window) for tf in TEMPLATE_TFS}
        self._prices = {tf: deque(maxlen=max(vol_window, se_window)) for tf in TEMPLATE_TFS}
        self._velocities = {tf: deque(maxlen=3) for tf in TEMPLATE_TFS}

    def extract(self, states_by_tf: dict, prices_by_tf: dict = None,
                volumes_by_tf: dict = None) -> np.ndarray:
        """
        Extract 48D feature vector from current multi-TF state.

        Args:
            states_by_tf: {tf_name: MarketState} for each TF
            prices_by_tf: {tf_name: close_price} (optional, extracted from state if not given)
            volumes_by_tf: {tf_name: volume} (optional)

        Returns:
            np.array of shape (48,)
        """
        features = np.zeros(N_FEATURES)
        tick = 0.25  # MNQ tick size

        for tf_idx, tf in enumerate(TEMPLATE_TFS):
            offset = tf_idx * len(FEATURES_PER_TF)
            state = states_by_tf.get(tf)
            if state is None:
                continue

            # Get raw values from state
            dmi_p = getattr(state, 'dmi_plus', 0.0)
            dmi_m = getattr(state, 'dmi_minus', 0.0)
            velocity = getattr(state, 'velocity', 0.0)
            z_score = getattr(state, 'z_score', 0.0)

            # Price and volume from explicit args or state
            price = 0.0
            volume = 0.0
            if prices_by_tf and tf in prices_by_tf:
                price = prices_by_tf[tf]
            if volumes_by_tf and tf in volumes_by_tf:
                volume = volumes_by_tf[tf]

            # Update rolling buffers
            if price > 0:
                self._prices[tf].append(price)
            if volume > 0:
                self._volumes[tf].append(volume)
            self._velocities[tf].append(velocity)

            # 1. DMI diff — who's winning (-100 to +100)
            features[offset + 0] = dmi_p - dmi_m

            # 2. DMI gap — battle intensity (0 to 100)
            features[offset + 1] = abs(dmi_p - dmi_m)

            # 3. Volume relative — participation vs recent average
            vols = list(self._volumes[tf])
            if len(vols) >= 2:
                avg_vol = sum(vols) / len(vols)
                features[offset + 2] = volume / avg_vol if avg_vol > 0 else 1.0
            else:
                features[offset + 2] = 1.0

            # 4. Directional volume — sign(close-open) * volume / avg_volume
            #    Positive = buying pressure, negative = selling pressure
            prices_list = list(self._prices[tf])
            if len(prices_list) >= 2 and volume > 0:
                price_dir = 1.0 if prices_list[-1] >= prices_list[-2] else -1.0
                avg_vol = sum(vols) / len(vols) if len(vols) >= 2 else volume
                features[offset + 3] = price_dir * volume / avg_vol if avg_vol > 0 else 0.0
            else:
                features[offset + 3] = 0.0

            # 5. Velocity — normalized price speed
            features[offset + 4] = velocity / tick if tick > 0 else velocity

            # 6. Z using standard error — significance of deviation
            prices_for_se = list(self._prices[tf])
            if len(prices_for_se) >= self.se_window:
                chunk = prices_for_se[-self.se_window:]
                mean = sum(chunk) / len(chunk)
                std = np.std(chunk)
                se = std / (len(chunk) ** 0.5) if len(chunk) > 1 else std
                features[offset + 5] = (price - mean) / se if se > 1e-8 else 0.0
            else:
                features[offset + 5] = z_score  # fallback

            # 7. Price acceleration — is velocity changing?
            vels = list(self._velocities[tf])
            if len(vels) >= 2:
                features[offset + 6] = vels[-1] - vels[-2]
            else:
                features[offset + 6] = 0.0

        return features

    def extract_from_single_tf(self, state, price: float = 0.0,
                                volume: float = 0.0, tf: str = '1m') -> np.ndarray:
        """
        Extract features for a single TF (used in live when only 1m is available).
        Returns 6D vector for that TF.
        """
        return self.extract(
            states_by_tf={tf: state},
            prices_by_tf={tf: price},
            volumes_by_tf={tf: volume},
        )[TEMPLATE_TFS.index(tf) * len(FEATURES_PER_TF):
          (TEMPLATE_TFS.index(tf) + 1) * len(FEATURES_PER_TF)]

    def reset(self):
        """Clear all rolling buffers."""
        for tf in TEMPLATE_TFS:
            self._volumes[tf].clear()
            self._prices[tf].clear()
            self._velocities[tf].clear()

    @staticmethod
    def feature_name(index: int) -> str:
        """Get human-readable name for feature at index."""
        return FEATURE_NAMES[index] if index < len(FEATURE_NAMES) else f'unknown_{index}'

    @staticmethod
    def feature_names_for_tf(tf: str) -> list:
        """Get feature names for a specific TF."""
        return [f'{tf}_{feat}' for feat in FEATURES_PER_TF]
