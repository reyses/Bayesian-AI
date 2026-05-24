"""Conditional Bayesian exit oracle — corrected framework (2026-05-10).

Replaces the failed 60-min forward-return oracle. Keys exit decisions on
the CURRENT TRADE STATE:
    - t_since_peak_bucket    seconds since the trade's max-favorable-PnL bar
    - capture_bucket         current_pnl / peak_pnl_so_far  bin

Lookup returns P(current peak IS the FINAL peak) from per-tier IS table.
Exit when P_final >= threshold.

Tier-specific thresholds learned from data:
    FADE-family:  threshold 0.85  (peaks are fleeting; exit fast)
    RIDE-family:  threshold 0.70  (peaks keep coming; exit only late)
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

T_SINCE_PEAK_BINS = (5, 15, 30, 60, 120, 300, 900)


def bucket_time(s: float) -> int:
    for i, b in enumerate(T_SINCE_PEAK_BINS):
        if s <= b: return i
    return len(T_SINCE_PEAK_BINS)


def bucket_capture(r: float) -> int:
    if r <= 0: return 0
    if r < 0.3: return 1
    if r < 0.6: return 2
    if r < 0.9: return 3
    return 4


@lru_cache(maxsize=1)
def _load_conditional_table() -> dict:
    """Returns nested dict {tier: {(t_bucket, cap_bucket): P_final_IS}}."""
    path = os.path.join(REPO_ROOT,
        'reports/findings/segments/tier_conditional_exit/conditional_exit_table.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    table = {}
    for _, r in df.iterrows():
        tier = r['tier']
        key = (int(r['t_since_peak_bucket']), int(r['capture_bucket']))
        table.setdefault(tier, {})[key] = float(r['P_is_peak_final_IS'])
    return table


def get_threshold(tier: str) -> float:
    """Tier-specific exit threshold."""
    if tier.startswith('RIDE_') or tier == 'NMP_RIDE_RAW':
        return 0.70
    return 0.85


class BayesConditionalExit:
    """Per-bar exit oracle.

    Internal state per trade: peak_pnl_so_far, ts_at_peak.

    Usage:
        oracle = BayesConditionalExit(tier='FADE_CALM')
        oracle.reset()
        # at each bar:
        should_exit, reason = oracle.update_and_query(
            current_pnl_dollar=..., current_ts=...)
    """
    def __init__(self, tier: str, threshold: float = None):
        self.tier = tier
        self.threshold = threshold if threshold is not None else get_threshold(tier)
        self.table = _load_conditional_table().get(tier, {})
        self.peak_pnl = -float('inf')
        self.peak_ts = 0

    def reset(self, entry_ts: int = 0):
        self.peak_pnl = -float('inf')
        self.peak_ts = entry_ts

    def update_and_query(self, current_pnl: float,
                            current_ts: int) -> Tuple[bool, str, float]:
        """Update peak tracker and return (should_exit, reason, p_final)."""
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl
            self.peak_ts = current_ts
        if self.peak_pnl <= 0:
            return False, 'no_peak_yet', np.nan
        t_since_peak = max(0, current_ts - self.peak_ts)
        capture = current_pnl / self.peak_pnl if self.peak_pnl > 0 else 0
        t_b = bucket_time(t_since_peak)
        cap_b = bucket_capture(capture)
        p_final = self.table.get((t_b, cap_b), np.nan)
        if np.isnan(p_final):
            return False, f'no_table_cell_t{t_b}_cap{cap_b}', np.nan
        if p_final >= self.threshold:
            return True, f'bayes_exit_p{p_final:.2f}_t{t_b}_cap{cap_b}', p_final
        return False, f'hold_p{p_final:.2f}', p_final
