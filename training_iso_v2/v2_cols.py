"""Canonical V2 column-name helpers.

V2 columns carry the actual N-value as suffix (e.g., L3_1m_z_se_15 for 1m's
N_BASE=15, L3_5m_z_se_9 for 5m's N_BASE=9). Strategies should use these
helpers instead of hardcoding suffixes — keeps strategy code stable across
N changes and avoids the "feature name typo" bug class.
"""
from __future__ import annotations

from core_v2.features import N_BASE, TF_ORDER


# ── L1 (per-TF, window-free primitives, suffix `_1b`) ─────────────────────

def price_velocity_1b(tf: str) -> str:
    return f'L1_{tf}_price_velocity_1b'

def price_accel_1b(tf: str) -> str:
    return f'L1_{tf}_price_accel_1b'

def vol_velocity_1b(tf: str) -> str:
    return f'L1_{tf}_vol_velocity_1b'

def vol_accel_1b(tf: str) -> str:
    return f'L1_{tf}_vol_accel_1b'

def bar_range(tf: str) -> str:
    return f'L1_{tf}_bar_range'

def body(tf: str) -> str:
    return f'L1_{tf}_body'


# ── L2 (per-TF, rolling-window stats, suffix `_{N}`) ──────────────────────

def price_velocity_w(tf: str) -> str:
    return f'L2_{tf}_price_velocity_{N_BASE[tf]}'

def price_accel_w(tf: str) -> str:
    return f'L2_{tf}_price_accel_{N_BASE[tf]}'

def vol_velocity_w(tf: str) -> str:
    return f'L2_{tf}_vol_velocity_{N_BASE[tf]}'

def vol_accel_w(tf: str) -> str:
    return f'L2_{tf}_vol_accel_{N_BASE[tf]}'

def price_mean_w(tf: str) -> str:
    return f'L2_{tf}_price_mean_{N_BASE[tf]}'

def price_sigma_w(tf: str) -> str:
    return f'L2_{tf}_price_sigma_{N_BASE[tf]}'

def vol_mean_w(tf: str) -> str:
    return f'L2_{tf}_vol_mean_{N_BASE[tf]}'

def vol_sigma_w(tf: str) -> str:
    return f'L2_{tf}_vol_sigma_{N_BASE[tf]}'

def vwap_w(tf: str) -> str:
    return f'L2_{tf}_vwap_{N_BASE[tf]}'


# ── L3 (per-TF, approved exceptions, suffix `_{N}` mostly) ────────────────

def z_se_w(tf: str) -> str:
    return f'L3_{tf}_z_se_{N_BASE[tf]}'

def z_high_w(tf: str) -> str:
    return f'L3_{tf}_z_high_{N_BASE[tf]}'

def z_low_w(tf: str) -> str:
    return f'L3_{tf}_z_low_{N_BASE[tf]}'

def SE_high_w(tf: str) -> str:
    return f'L3_{tf}_SE_high_{N_BASE[tf]}'

def SE_low_w(tf: str) -> str:
    return f'L3_{tf}_SE_low_{N_BASE[tf]}'

def hurst_w(tf: str) -> str:
    return f'L3_{tf}_hurst_{N_BASE[tf]}'

def reversion_prob_w(tf: str) -> str:
    return f'L3_{tf}_reversion_prob_{N_BASE[tf]}'

def swing_noise_w(tf: str) -> str:
    return f'L3_{tf}_swing_noise_{N_BASE[tf]}'


# ── L0 (global) ───────────────────────────────────────────────────────────

L0_TIME_OF_DAY = 'L0_time_of_day'


# ── Convenience: per-TF feature blocks for CNN consumption ────────────────

# Order MUST match the canonical 23-feature-per-TF block (L1 then L2 then L3)
# that matches FEATURE_NAMES in core_v2.features.py.
def per_tf_block(tf: str) -> list:
    return [
        # L1 (6)
        price_velocity_1b(tf), price_accel_1b(tf),
        vol_velocity_1b(tf),   vol_accel_1b(tf),
        bar_range(tf),         body(tf),
        # L2 (9)
        price_velocity_w(tf),  price_accel_w(tf),
        vol_velocity_w(tf),    vol_accel_w(tf),
        price_mean_w(tf),      price_sigma_w(tf),
        vol_mean_w(tf),        vol_sigma_w(tf),
        vwap_w(tf),
        # L3 (8)
        z_se_w(tf),            z_high_w(tf),
        z_low_w(tf),           SE_high_w(tf),
        SE_low_w(tf),          hurst_w(tf),
        reversion_prob_w(tf),  swing_noise_w(tf),
    ]


__all__ = [
    'TF_ORDER', 'N_BASE',
    'price_velocity_1b', 'price_accel_1b', 'vol_velocity_1b', 'vol_accel_1b',
    'bar_range', 'body',
    'price_velocity_w', 'price_accel_w', 'vol_velocity_w', 'vol_accel_w',
    'price_mean_w', 'price_sigma_w', 'vol_mean_w', 'vol_sigma_w', 'vwap_w',
    'z_se_w', 'z_high_w', 'z_low_w', 'SE_high_w', 'SE_low_w',
    'hurst_w', 'reversion_prob_w', 'swing_noise_w',
    'L0_TIME_OF_DAY', 'per_tf_block',
]
