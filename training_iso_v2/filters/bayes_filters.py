"""Bayesian-table-driven filters for tier and zigzag augmentation.

Reads the V0 Bayesian tables built 2026-05-10 and exposes simple lookup
functions usable from any strategy (tier or NT8-bridged zigzag):

    1. TOD_RISK            position-size multiplier from P(cat) lookup
    2. TIER_BLEED          per-tier skip filter from chord bleed cells
    3. COMPRESSION_BOUNCE  long-bias signal when 15m vol_sigma deep below RM
    4. CAT_HARVEST         pre-position SHORT during danger windows

All functions consume the existing CSVs:
    reports/findings/segments/bayes_table_at_bar_cat_risk/p_cat_by_tod_x_sigma.csv
    reports/findings/segments/bayes_table_cat_risk_dow/p_cat_by_dow_x_tod.csv
    reports/findings/segments/bayes_table_cat_harvest/strong_directional_cells.csv
    reports/findings/segments/bayes_table_feature_native_rm_FULL/STRONG_native_rm_events.csv
    reports/findings/segments/diagnostic_tier_bleed/BLEED_tier_x_sub_motif_x_measure.csv
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# ─── 1. TOD RISK SIZING ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_tod_dow_p_cat() -> Dict[Tuple[str, int], float]:
    path = os.path.join(REPO_ROOT,
        'reports/findings/segments/bayes_table_cat_risk_dow/p_cat_by_dow_x_tod.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {(r['dow'], int(r['tod_hour'])): float(r['P_cat_60m'])
            for _, r in df.iterrows()}


def tod_risk_size_multiplier(timestamp_utc: float) -> float:
    """Return position-size multiplier based on TOD/DOW catastrophic risk.
        UTC hour 1 (any DOW): multiplier 0.1 (effectively avoid)
        High-risk (P_cat >= 0.10):   multiplier 0.3
        Elevated (P_cat 0.05-0.10):  multiplier 0.6
        Baseline (P_cat 0.01-0.05):  multiplier 1.0
        Safest (P_cat < 0.01):       multiplier 1.0
    """
    dt = datetime.fromtimestamp(int(timestamp_utc), tz=timezone.utc)
    dow = dt.strftime('%a')
    hr = dt.hour
    table = _load_tod_dow_p_cat()
    p = table.get((dow, hr), 0.034)  # baseline if missing
    if p >= 0.20:    return 0.1   # peak danger (Tue/Wed UTC 1)
    if p >= 0.10:   return 0.3   # high
    if p >= 0.05:   return 0.6   # elevated
    return 1.0                    # normal


# ─── 2. TIER BLEED FILTER ──────────────────────────────────────────────

@lru_cache(maxsize=16)
def _load_bleed_set(tier: str) -> Set[Tuple[str, str]]:
    """Returns set of (sub_motif_shape, measure_shape) chord cells where
    `tier` historically bleeds (IS<0 AND OOS<0)."""
    path = os.path.join(REPO_ROOT,
        'reports/findings/segments/diagnostic_tier_bleed/BLEED_tier_x_sub_motif_x_measure.csv')
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    df = df[df['tier'] == tier]
    return {(r['sub_motif_shape'], r['measure_shape']) for _, r in df.iterrows()}


def is_tier_bleed_cell(tier: str, sub_motif_shape: str,
                         measure_shape: str) -> bool:
    """True if this chord cell is in `tier`'s bleed set → SKIP entry."""
    return (sub_motif_shape, measure_shape) in _load_bleed_set(tier)


# ─── 3. COMPRESSION BOUNCE FILTER ──────────────────────────────────────

class CompressionBounce:
    """Tracks 15m vol_sigma's rolling RM and detects compression events.

    Per the 2026-05-10 finding (full feature scan):
      L2_15m_vol_sigma_12 < native_RM - 3σ  → P(up_60m) = 0.55 IS / 0.64 OOS

    Maintain per-day rolling stats on the native path of L2_15m_vol_sigma_12.
    """
    NATIVE_WINDOW = 12   # 12 distinct 15m bars = 3 hours
    K_SIGMA = 3.0

    def __init__(self):
        self._day = None
        self._native_path = []   # list of distinct 15m vol_sigma values seen so far
        self._last_value = None
        self._active = False

    def reset(self, day: str):
        self._day = day
        self._native_path = []
        self._last_value = None
        self._active = False

    def update(self, vol_sigma_15m: float) -> str:
        """Call at every bar. Returns:
            'LONG_BIAS'  if compression event active (below -3σ from RM)
            'NEUTRAL'    otherwise
        """
        if not np.isfinite(vol_sigma_15m):
            return 'NEUTRAL'
        # Detect native bar boundary (value changed)
        if self._last_value is None or vol_sigma_15m != self._last_value:
            self._native_path.append(vol_sigma_15m)
            self._last_value = vol_sigma_15m
        if len(self._native_path) < self.NATIVE_WINDOW + 1:
            return 'NEUTRAL'
        # Use prior NATIVE_WINDOW values (exclude current to avoid lookahead)
        prior = np.array(self._native_path[-(self.NATIVE_WINDOW + 1):-1])
        rm = float(prior.mean())
        sd = float(prior.std(ddof=1))
        if sd < 1e-9: return 'NEUTRAL'
        z = (vol_sigma_15m - rm) / sd
        return 'LONG_BIAS' if z <= -self.K_SIGMA else 'NEUTRAL'


# ─── 4. CAT HARVEST SIGNAL ──────────────────────────────────────────────

# Tier-1 danger windows (from 2026-05-10 DOW analysis) where SHORT bias is structural
CAT_HARVEST_WINDOWS = [
    # (dow, utc_hour_start, utc_hour_end, expected_bias_direction)
    ('Tue', 1, 2, 'short'),    # 40% P_cat, 95% crash bias
    ('Tue', 2, 3, 'short'),    # 12% P_cat continues
    ('Tue', 14, 15, 'short'),  # 10% P_cat, US morning
    ('Wed', 1, 2, 'short'),    # 32% P_cat
    ('Wed', 2, 3, 'short'),
    ('Thu', 1, 2, 'short'),    # 24% P_cat
    ('Thu', 2, 3, 'short'),    # 10% P_cat, deepest crashes
    ('Thu', 14, 15, 'short'),
    ('Fri', 2, 3, 'short'),
]


def cat_harvest_signal(timestamp_utc: float) -> Optional[str]:
    """Returns 'PRE_SHORT' if currently in a known cat-harvest danger window.
    Else None.
    """
    dt = datetime.fromtimestamp(int(timestamp_utc), tz=timezone.utc)
    dow = dt.strftime('%a')
    hr = dt.hour
    for d, h_start, h_end, side in CAT_HARVEST_WINDOWS:
        if d == dow and h_start <= hr < h_end:
            return f'PRE_{side.upper()}'
    return None


# ─── 5. COMPOSITE FILTER (for a base strategy) ──────────────────────────

class AugmentedStrategy:
    """Wraps a base Strategy with all four filters.

    Composition order:
      1. Base strategy proposes a signal
      2. TIER_BLEED filter — skip if chord in bleed set for this tier
      3. COMPRESSION_BOUNCE — if active LONG_BIAS, only allow long signals
         (or weight long entries 1.5x, suppress shorts 0.5x)
      4. CAT_HARVEST — overrides base. If PRE_SHORT active, force short.
      5. TOD_RISK — final sizing multiplier on the surviving signal

    Returns (signal, size_multiplier, reason_dict) — None signal means SKIP.
    """
    def __init__(self, base, tier_name: str):
        self.base = base
        self.tier_name = tier_name
        self.compression = CompressionBounce()
        self._current_day = None

    def evaluate(self, state, sub_motif_shape: str = '',
                  measure_shape: str = '') -> Tuple[Optional[object], float, dict]:
        if state.day != self._current_day:
            self.compression.reset(state.day)
            self._current_day = state.day

        base_signal = self.base.evaluate(state)

        # Always update compression tracker so it's ready when called
        vol_sigma_15m = state.get('L2_15m_vol_sigma_12', float('nan'))
        comp_state = self.compression.update(vol_sigma_15m)

        # Cat-harvest override (independent of base signal)
        harvest = cat_harvest_signal(state.timestamp)

        reasons = {
            'base_signal': base_signal.direction if base_signal else None,
            'compression': comp_state,
            'harvest': harvest,
            'tod_size_mult': tod_risk_size_multiplier(state.timestamp),
        }

        # 1. Cat harvest forces SHORT in danger windows (overrides base)
        if harvest == 'PRE_SHORT':
            from training_iso_v2.strategies.base import EntrySignal
            forced = EntrySignal(direction='short',
                                  tier=f'{self.tier_name}+CAT_HARVEST',
                                  extras={'reason': 'cat_harvest_window'})
            return forced, reasons['tod_size_mult'], reasons

        # 2. No base signal → nothing to filter
        if base_signal is None:
            return None, 0.0, reasons

        # 3. Tier-bleed filter — skip if in known bleed cell
        if is_tier_bleed_cell(self.tier_name, sub_motif_shape, measure_shape):
            reasons['skipped'] = 'tier_bleed_cell'
            return None, 0.0, reasons

        # 4. Compression-bounce — block shorts when long-bias active
        if comp_state == 'LONG_BIAS' and base_signal.direction == 'short':
            reasons['skipped'] = 'compression_long_bias'
            return None, 0.0, reasons

        # 5. TOD risk sizing applied to surviving signal
        return base_signal, reasons['tod_size_mult'], reasons
