"""Phase gate — extracts the impulse detector v2 state machine into a reusable
per-bar phase array, for use as a gate in fade/reversion strategies.

The journal (docs/daily/2026-05-09.md) established that reversion works in
NORMAL / STABILIZING phases and FAILS during DIRECTIONAL / FLATTENED /
PIVOT_CANDIDATE phases (impulse onset).

This module: given a day and CRM TF, returns a phase-per-5s-bar array. The
state machine logic is byte-equivalent to `tools/crm_pivot_detector_v2.py`
(extracted, not reimplemented).

Returned phases (per 5s bar):
    NORMAL          → reversion WORKS — fade strategies safe to fire
    DIRECTIONAL     → impulse building — DON'T fade
    FLATTENED       → pre-pivot — wait
    PIVOT_CANDIDATE → in confirmation window — wait
    STABILIZING     → post-impulse — fade safe again
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


TF_CONFIG = {
    '1m':  ('DATA/ATLAS/1m',  15),
    '5m':  ('DATA/ATLAS/5m',   9),
    '15m': ('DATA/ATLAS/15m', 12),
    '1h':  ('DATA/ATLAS/1h',  12),
}
PERIOD_S = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}

SAFE_PHASES = {'NORMAL', 'STABILIZING'}


def _load_ohlcv(tf: str, day: str) -> pd.DataFrame:
    base, _ = TF_CONFIG[tf]
    path = os.path.join(base, f'{day}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def compute_phases(day: str, crm_tf: str = '5m',
                          atlas_5s_dir: str = 'DATA/ATLAS/5s',
                          slope_window: int = 60,
                          rank_window: int = 720,
                          directional_quantile: float = 0.70,
                          flatten_quantile: float = 0.30,
                          curvature_quantile: float = 0.70,
                          monitor_minutes: float = 5.0,
                          min_hold_bars: int = 12,
                          sigma_rank_confirm: float = 0.65,
                          volvel_rank_confirm: float = 0.65,
                          volvel_window: int = 60) -> tuple:
    """Return (timestamps_5s, phase_str_array) for `day` using the impulse
    detector v2 state machine. Phase labels per SAFE_PHASES + others above.

    Returns empty arrays if day data is missing.
    """
    path_5s = os.path.join(atlas_5s_dir, f'{day}.parquet')
    if not os.path.exists(path_5s):
        return np.array([]), np.array([])

    ohlcv_5s = pd.read_parquet(path_5s)
    if pd.api.types.is_datetime64_any_dtype(ohlcv_5s['timestamp']):
        ohlcv_5s = ohlcv_5s.copy()
        ohlcv_5s['timestamp'] = (ohlcv_5s['timestamp'].astype('int64') // 10**9)
    ohlcv_5s = ohlcv_5s.sort_values('timestamp').reset_index(drop=True)
    oh_ts = ohlcv_5s['timestamp'].values.astype(np.int64)

    crm_oh = _load_ohlcv(crm_tf, day)
    if crm_oh.empty:
        return oh_ts, np.array(['NORMAL'] * len(oh_ts), dtype=object)

    _, N = TF_CONFIG[crm_tf]
    period_s = PERIOD_S[crm_tf]
    crm_oh['close_mean']  = crm_oh['close'].rolling(N, min_periods=2).mean()
    crm_oh['close_sigma'] = crm_oh['close'].rolling(N, min_periods=2).std()
    crm_ts = crm_oh['timestamp'].values.astype(np.int64)
    target = oh_ts - period_s
    idx = np.searchsorted(crm_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(crm_ts) - 1)
    M = crm_oh['close_mean'].values[idx]
    S = crm_oh['close_sigma'].values[idx]

    n = len(M)
    slope = np.full(n, np.nan)
    if n > slope_window:
        slope[slope_window:] = (M[slope_window:] - M[:-slope_window]) / slope_window
    curv = np.full(n, np.nan)
    if n > slope_window:
        curv[slope_window:] = (slope[slope_window:] - slope[:-slope_window]) / slope_window

    s_rank = pd.Series(np.abs(slope)).rolling(rank_window, min_periods=10).rank(pct=True).values
    c_rank = pd.Series(np.abs(curv)).rolling(rank_window, min_periods=10).rank(pct=True).values
    sigma_rank = pd.Series(S).rolling(rank_window, min_periods=10).rank(pct=True).values

    vol = ohlcv_5s['volume'].values.astype(np.float64)
    vol_mean = pd.Series(vol).rolling(volvel_window, min_periods=2).mean().values
    vol_velocity = np.full(n, np.nan)
    if n > volvel_window:
        vol_velocity[volvel_window:] = (vol_mean[volvel_window:]
                                                  - vol_mean[:-volvel_window]) / volvel_window
    volvel_rank = pd.Series(np.abs(vol_velocity)).rolling(rank_window, min_periods=10).rank(pct=True).values

    monitor_bars = int(round(monitor_minutes * 60 / 5))

    phases = np.empty(n, dtype=object)
    phases[:] = 'NORMAL'

    state = 'NORMAL'
    state_start_slope_sign = 0
    candidate_idx = None
    candidate_new_sign = 0
    candidate_hold_run = 0
    candidate_sigma_rank_seen = False
    candidate_volvel_rank_seen = False

    for i in range(slope_window + 1, n):
        if not np.isfinite(slope[i]) or not np.isfinite(s_rank[i]):
            phases[i] = state
            continue

        if state == 'NORMAL' and s_rank[i] >= directional_quantile:
            state = 'DIRECTIONAL'
            state_start_slope_sign = float(np.sign(slope[i]))
        elif state == 'DIRECTIONAL' and s_rank[i] <= flatten_quantile:
            state = 'FLATTENED'
        elif state == 'FLATTENED':
            if (np.sign(slope[i]) != state_start_slope_sign
                  and np.sign(slope[i]) != 0
                  and np.isfinite(c_rank[i])
                  and c_rank[i] >= curvature_quantile):
                state = 'PIVOT_CANDIDATE'
                candidate_idx = i
                candidate_new_sign = float(np.sign(slope[i]))
                candidate_hold_run = 1
                candidate_sigma_rank_seen = False
                candidate_volvel_rank_seen = False
        elif state == 'PIVOT_CANDIDATE':
            elapsed = i - candidate_idx
            cur_sign = float(np.sign(slope[i]))
            if cur_sign == candidate_new_sign:
                candidate_hold_run += 1
            else:
                candidate_hold_run = 0
            if np.isfinite(sigma_rank[i]) and sigma_rank[i] >= sigma_rank_confirm:
                candidate_sigma_rank_seen = True
            if np.isfinite(volvel_rank[i]) and volvel_rank[i] >= volvel_rank_confirm:
                candidate_volvel_rank_seen = True
            confirmed = (candidate_hold_run >= min_hold_bars
                                 and candidate_sigma_rank_seen
                                 and candidate_volvel_rank_seen)
            if confirmed:
                state = 'STABILIZING'
            elif elapsed >= monitor_bars:
                state = 'NORMAL'
        elif state == 'STABILIZING':
            if np.isfinite(s_rank[i]) and s_rank[i] <= flatten_quantile:
                state = 'NORMAL'

        phases[i] = state

    return oh_ts, phases


def phases_at_1m(day: str, target_1m_ts: np.ndarray, crm_tf: str = '5m') -> np.ndarray:
    """Compute phases at 5s, then sample at the supplied 1m timestamps."""
    ts_5s, phases_5s = compute_phases(day, crm_tf=crm_tf)
    if len(ts_5s) == 0:
        return np.array(['NORMAL'] * len(target_1m_ts), dtype=object)
    idx = np.searchsorted(ts_5s, target_1m_ts.astype(np.int64), side='right') - 1
    idx = np.clip(idx, 0, len(ts_5s) - 1)
    return phases_5s[idx]


def is_reversion_safe(phase: str) -> bool:
    return phase in SAFE_PHASES


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_09_08')
    ap.add_argument('--crm-tf', default='5m')
    args = ap.parse_args()
    ts, phases = compute_phases(args.day, crm_tf=args.crm_tf)
    if len(phases) == 0:
        print(f'No data for {args.day}')
    else:
        from collections import Counter
        print(f'{args.day} phase distribution ({len(phases)} bars):')
        for p, n in Counter(phases).most_common():
            print(f'  {p:<18} {n:>6}  ({100*n/len(phases):.1f}%)')
        safe_pct = sum(1 for p in phases if p in SAFE_PHASES) / len(phases)
        print(f'\nSAFE-PHASE coverage: {100*safe_pct:.1f}% of bars')
