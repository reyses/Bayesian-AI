# Proxy for legacy tests to import from legacy nightmare_blended
import sys
import os
import numpy as np

# Map 'core' module to 'core_v2' to support legacy imports
import core_v2
import core_v2.engine_signals
import core_v2.features

sys.modules['core'] = core_v2
sys.modules['core.engine_signals'] = core_v2.engine_signals
sys.modules['core.features'] = core_v2.features

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "reference")))
from nightmare_blended_2026_05_20 import (
    BlendedEngine as LegacyBlendedEngine,
    P_CENTER_EXIT,
    P_CENTER_EXIT_BARS_CASCADE,
    RIDE_EXIT_BARS_TIERS,
)

# ─── V2 to V1 Feature Mapper ──────────────────────────────────────────────
FEATURE_NAMES = core_v2.features.FEATURE_NAMES
TF_ORDER_V1 = ['15s', '1m', '5m', '15m', '1h', '1D']

def _get_v2_index(pattern: str) -> int:
    for idx, name in enumerate(FEATURE_NAMES):
        if pattern in name:
            return idx
    raise ValueError(f"Pattern {pattern} not found in V2 FEATURE_NAMES")

V2_INDICES = {}
for tf in TF_ORDER_V1:
    V2_INDICES[tf] = {
        'z_se': _get_v2_index(f'L3_{tf}_z_se_'),
        'velocity': _get_v2_index(f'L2_{tf}_price_velocity_'),
        'price_accel': _get_v2_index(f'L2_{tf}_price_accel_'),
        'vol_rel': _get_v2_index(f'L1_{tf}_vol_velocity_1b'),
        'bar_range': _get_v2_index(f'L1_{tf}_bar_range'),
        'body': _get_v2_index(f'L1_{tf}_body'),
        'hurst': _get_v2_index(f'L3_{tf}_hurst_'),
        'reversion_prob': _get_v2_index(f'L3_{tf}_reversion_prob_'),
        'z_high': _get_v2_index(f'L3_{tf}_z_high_'),
        'z_low': _get_v2_index(f'L3_{tf}_z_low_'),
    }

def map_v2_to_v1(v2_feat, vr_1m=1.0) -> np.ndarray:
    v1_feat = np.zeros(91, dtype=np.float32)
    for tf_idx, tf in enumerate(TF_ORDER_V1):
        offset = tf_idx * 12
        helper_offset = 72 + tf_idx * 3
        
        indices = V2_INDICES[tf]
        
        # Core features
        v1_feat[offset + 0] = v2_feat[indices['z_se']]
        v1_feat[offset + 1] = 0.0  # dmi_diff
        
        if tf == '1m':
            v1_feat[offset + 2] = vr_1m
        else:
            v1_feat[offset + 2] = 1.0
            
        v1_feat[offset + 3] = v2_feat[indices['velocity']]
        v1_feat[offset + 4] = v2_feat[indices['price_accel']]
        v1_feat[offset + 5] = v2_feat[indices['vol_rel']]
        v1_feat[offset + 6] = v2_feat[indices['bar_range']]
        v1_feat[offset + 7] = v2_feat[indices['hurst']]
        v1_feat[offset + 8] = v2_feat[indices['reversion_prob']]
        
        # p_center = exp(-0.5 * z_se^2)
        z_val = v1_feat[offset + 0]
        v1_feat[offset + 9] = np.exp(-0.5 * (z_val ** 2))
        
        v1_feat[offset + 10] = v2_feat[indices['z_high']]
        v1_feat[offset + 11] = v2_feat[indices['z_low']]
        
        # Helpers
        v1_feat[helper_offset + 0] = 0.0
        v1_feat[helper_offset + 1] = 0.0
        
        # helper 2 is wick_ratio = 1.0 - body/bar_range
        body = v2_feat[indices['body']]
        br = v2_feat[indices['bar_range']]
        if br > 0.0:
            v1_feat[helper_offset + 2] = 1.0 - (body / br)
        else:
            v1_feat[helper_offset + 2] = 0.0
            
    return v1_feat

class BlendedEngine(LegacyBlendedEngine):
    def evaluate(self, state: dict):
        mapped_state = state.copy()
        
        v2_feat = None
        if 'features_79d' in state:
            v2_feat = state['features_79d']
        elif 'features' in state:
            v2_feat = state['features']
            
        if v2_feat is not None:
            vr = state.get('variance_ratio', 1.0)
            v1_feat = map_v2_to_v1(v2_feat, vr)
            mapped_state['features_79d'] = v1_feat
            mapped_state['features'] = v1_feat
            
        return super().evaluate(mapped_state)
        
    def on_state(self, state: dict):
        mapped_state = state.copy()
        
        v2_feat = None
        if 'features' in state:
            v2_feat = state['features']
        elif 'features_79d' in state:
            v2_feat = state['features_79d']
            
        if v2_feat is not None:
            vr = state.get('variance_ratio', 1.0)
            v1_feat = map_v2_to_v1(v2_feat, vr)
            mapped_state['features'] = v1_feat
            mapped_state['features_79d'] = v1_feat
            
        return super().on_state(mapped_state)
