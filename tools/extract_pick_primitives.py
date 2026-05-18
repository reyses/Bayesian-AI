"""Extract primitive features for every user pick across all DATA/cusp_picks/
canonical files. The saved picks already carry static z-anchors at entry,
but not slopes / curvature / band-compression — the features your real
decision uses (per 2026-05-12 "bands compressed and CRM slowed down").

For each pick, recomputes at the pick's timestamp:

  Static anchors (mirror of what's in the file, recomputed for consistency):
    z_15s, z_1m, z_15m, z_1h_high, z_1h_low, z_1h_close

  Slopes (price-points / min):
    slope_15s_3m    = (M_15s[t] - M_15s[t-3])  / 3       fast direction now
    slope_15s_10m   = (M_15s[t] - M_15s[t-10]) / 10      direction over 10min
    slope_1m_10m    = (M_1m[t]  - M_1m[t-10])  / 10
    slope_15m_5m    = (M_15m[t] - M_15m[t-5])  / 5       15m near-term
    slope_15m_15m   = (M_15m[t] - M_15m[t-15]) / 15      15m sustained
    slope_15m_decel = slope_15m_5m - slope_15m_15m       slope acceleration

  Curvature (slope-of-slope, price-points / min²):
    curv_15m = (slope_15m_5m - slope_15m_15m) / 10       positive → bottoming

  Compression / volatility-rank features:
    band_width   = Mh_1h - Ml_1h
    band_rank_60 = rolling percentile of band_width over last 60 1m bars
                       (near 0 = BANDS COMPRESSED)
    sigma_15m_rank_60 = rolling percentile of S_15m over last 60 bars
                            (near 0 = VOLATILITY COMPRESSED)

  Setup classification:
    v6_fires_short = (z_15m >= +1.5 AND slope_15s sign-flip down)
    v6_fires_long  = (z_15m <= -1.5 AND slope_15s sign-flip up)
    compression_short = (z_1h_high > 0 AND band_rank_60 < 0.3)
    compression_long  = (z_1h_low  < 0 AND band_rank_60 < 0.3)

  Outcome (already in the pick file):
    direction, mfe_dollars, mae_dollars, rr (= mfe/max(mae,1)), is_winner

Output:
    reports/findings/cusp_picks_primitives.csv  — one row per pick, all features

Usage:
    python tools/extract_pick_primitives.py
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars


TICK = 0.25
DOL = 0.50
OUT_PATH = 'reports/findings/cusp_picks_primitives.csv'


def _pct_rank(values: np.ndarray, idx: int, window: int) -> float:
    """Rolling percentile rank of values[idx] within last `window` bars."""
    start = max(0, idx - window + 1)
    sub = values[start:idx + 1]
    finite = sub[np.isfinite(sub)]
    if len(finite) < 2:
        return 0.5
    v = values[idx]
    if not np.isfinite(v):
        return 0.5
    return float((finite < v).sum() / len(finite))


def extract_for_day(day_str: str, picks: list) -> list:
    """Recompute features at each pick's bar for one day's data."""
    t_start = datetime.strptime(day_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
    # Pad on both sides so slope lookbacks and percentile ranks are well-defined.
    # Some picks may extend into following days if --days 5 was used.
    t_end = t_start + 7 * 86400

    df = load_1m_bars(t_start, t_end)
    if df.empty:
        print(f'  [{day_str}] no 1m bars, skipping')
        return []
    ts = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)

    # Anchors
    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    # Slopes (vectorized)
    n = len(ts)
    def _slope(arr, lb):
        s = np.full(n, np.nan)
        if n > lb:
            s[lb:] = (arr[lb:] - arr[:-lb]) / lb
        return s
    slope_15s_3m  = _slope(M_15s, 3)
    slope_15s_10m = _slope(M_15s, 10)
    slope_1m_10m  = _slope(M_1m,  10)
    slope_15m_5m  = _slope(M_15m, 5)
    slope_15m_15m = _slope(M_15m, 15)
    slope_15m_decel = slope_15m_5m - slope_15m_15m
    curv_15m = slope_15m_decel / 10.0

    # Band width + compression ranks
    band_width = Mh - Ml

    # ── CRM CROSSINGS (user 2026-05-12 "when they cross stuff happens") ──
    # For each adjacent pair, compute the sign of (faster - slower) at each
    # bar. A sign change is a crossing event. Track "bars since" each
    # crossing in each direction.
    def cross_history(fast: np.ndarray, slow: np.ndarray):
        """Returns (bars_since_cross_up, bars_since_cross_dn, current_above).
        bars_since == -1 means no such crossing yet in this window."""
        diff = fast - slow
        sign = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
        n = len(sign)
        b_up = np.full(n, -1, dtype=int)
        b_dn = np.full(n, -1, dtype=int)
        last_up = -1
        last_dn = -1
        for i in range(1, n):
            if not (np.isfinite(diff[i]) and np.isfinite(diff[i-1])):
                continue
            # Sign flip from <=0 to >0 = upcross (fast crosses above slow)
            if sign[i-1] <= 0 and sign[i] > 0:
                last_up = i
            if sign[i-1] >= 0 and sign[i] < 0:
                last_dn = i
            if last_up >= 0:
                b_up[i] = i - last_up
            if last_dn >= 0:
                b_dn[i] = i - last_dn
        return b_up, b_dn

    # Three important pairs: 15s×1m (fastest), 1m×15m, 15s×15m
    bu_15s_1m,  bd_15s_1m  = cross_history(M_15s, M_1m)
    bu_1m_15m,  bd_1m_15m  = cross_history(M_1m,  M_15m)
    bu_15s_15m, bd_15s_15m = cross_history(M_15s, M_15m)
    # Price × M_15m (price reclaim of medium)
    bu_px_15m, bd_px_15m = cross_history(close, M_15m)
    # Fast-anchor × slow-rail crossings (user 2026-05-12 part 3)
    bu_15s_Mh,  bd_15s_Mh  = cross_history(M_15s, Mh)
    bu_15s_Ml,  bd_15s_Ml  = cross_history(M_15s, Ml)
    # MEDIUM-anchor × slow-rail crossings (15m CRM crossing 1h rails — the user's
    # actual decision feature; 15m vs 1h is a structural-shift signal, while
    # 15s vs 1h is noise-frequency)
    bu_15m_Mh,  bd_15m_Mh  = cross_history(M_15m, Mh)
    bu_15m_Ml,  bd_15m_Ml  = cross_history(M_15m, Ml)

    # ── CRM DISTANCES / FAN GEOMETRY (user 2026-05-12 part 2) ───────────
    # User's observation: "I'm seeing the distance between each CRM".
    # Distances normalized in 1m sigma units (consistent timescale).
    # Fan width = total spread; rolling-rank = compression of the fan.
    safe_S_1m = np.where(S_1m > 0, S_1m, np.nan)
    safe_S_15m = np.where(S_15m > 0, S_15m, np.nan)
    dist_15s_1m = (M_15s - M_1m) / safe_S_1m
    dist_1m_15m = (M_1m  - M_15m) / safe_S_15m
    dist_15s_15m = (M_15s - M_15m) / safe_S_15m
    fan_width = np.abs(dist_15s_1m) + np.abs(dist_1m_15m) + np.abs(dist_15s_15m)
    # Rate of change of distances (positive = gap widening; negative = converging)
    def _delta(arr, lb):
        d = np.full(n, np.nan)
        if n > lb:
            d[lb:] = (arr[lb:] - arr[:-lb]) / lb
        return d
    n_local = len(close)
    n = n_local
    delta_dist_15s_1m_10m  = _delta(dist_15s_1m, 10)
    delta_dist_1m_15m_10m  = _delta(dist_1m_15m, 10)
    delta_dist_15s_15m_10m = _delta(dist_15s_15m, 10)

    out = []
    for p in picks:
        pts = int(p['timestamp'])
        # Find this bar's index
        idx = int(np.searchsorted(ts, pts))
        if idx >= n or ts[idx] != pts:
            # try idx-1 fallback (snap to last bar at-or-before pts)
            if idx > 0 and ts[idx - 1] <= pts:
                idx = idx - 1
            else:
                continue

        c = close[idx]

        def _z(num, denom):
            if denom and not np.isnan(num) and not np.isnan(denom) and denom > 0:
                return (c - num) / denom
            return np.nan

        row = {
            'date': day_str,
            'pick_time_utc': p.get('time_utc'),
            'direction': p.get('direction'),
            'price': p.get('price'),
            'snap': p.get('snap'),
            'mfe_dollars': p.get('mfe_dollars', 0),
            'mae_dollars': p.get('mae_dollars', 0),
            'rr': p['mfe_ticks'] / max(p.get('mae_ticks', 0), 1),
            'is_winner': int(p.get('mfe_dollars', 0) >= 2 * max(p.get('mae_dollars', 0), 1)),
            # Static anchors (recomputed)
            'z_15s':       round(_z(M_15s[idx], S_15s[idx]), 3),
            'z_1m':        round(_z(M_1m[idx],  S_1m[idx]),  3),
            'z_15m':       round(_z(M_15m[idx], S_15m[idx]), 3),
            'z_1h_high':   round(_z(Mh[idx], Sh[idx]),       3) if not np.isnan(Sh[idx]) else None,
            'z_1h_low':    round(_z(Ml[idx], Sl[idx]),       3) if not np.isnan(Sl[idx]) else None,
            'z_1h_close':  round(_z(Mc[idx], Sc[idx]),       3) if not np.isnan(Sc[idx]) else None,
            # Slopes
            'slope_15s_3m':    round(slope_15s_3m[idx],    3),
            'slope_15s_10m':   round(slope_15s_10m[idx],   3),
            'slope_1m_10m':    round(slope_1m_10m[idx],    3),
            'slope_15m_5m':    round(slope_15m_5m[idx],    3),
            'slope_15m_15m':   round(slope_15m_15m[idx],   3),
            'slope_15m_decel': round(slope_15m_decel[idx], 3),
            'curv_15m':        round(curv_15m[idx],        4),
            # Compression
            'band_width':      round(band_width[idx], 2) if not np.isnan(band_width[idx]) else None,
            'band_rank_60':    round(_pct_rank(band_width, idx, 60),       3),
            'sigma_15m_rank_60': round(_pct_rank(S_15m,    idx, 60),       3),
            # Slope sign-flip flags
            'slope_15s_flipped_dn': int(slope_15s_10m[idx] > 0.10
                                                       and slope_15s_3m[idx] < -0.05) if not np.isnan(slope_15s_10m[idx]) else 0,
            'slope_15s_flipped_up': int(slope_15s_10m[idx] < -0.10
                                                       and slope_15s_3m[idx] > 0.05) if not np.isnan(slope_15s_10m[idx]) else 0,
        }
        # ── Multi-CRM SLOPE ALIGNMENT (user 2026-05-12) ─────────────────
        # User's gate: "took decisions when direction aligned with the 3 CRMs."
        # Count of CRMs whose slope sign matches each direction.
        # Threshold: slope ≥ +0.05 = "up" / slope ≤ -0.05 = "down".
        EPS = 0.05
        slopes_to_check = {
            '15s': slope_15s_10m[idx],
            '1m':  slope_1m_10m[idx],
            '15m': slope_15m_15m[idx],
        }
        up_count   = sum(1 for s in slopes_to_check.values()
                                 if not np.isnan(s) and s >= +EPS)
        down_count = sum(1 for s in slopes_to_check.values()
                                 if not np.isnan(s) and s <= -EPS)
        row['align_up_count']   = up_count       # 0..3
        row['align_down_count'] = down_count     # 0..3
        row['align_state'] = (
            'FULL_UP'    if up_count == 3 else
            'STRONG_UP'  if up_count == 2 and down_count == 0 else
            'FULL_DOWN'  if down_count == 3 else
            'STRONG_DN'  if down_count == 2 and up_count == 0 else
            'DIVERGENT'  if up_count >= 1 and down_count >= 1 else
            'FLAT'
        )
        # CRM stack order at this bar (top-to-bottom by price)
        stack = []
        if not np.isnan(M_15s[idx]): stack.append(('15s', float(M_15s[idx])))
        if not np.isnan(M_1m[idx]):  stack.append(('1m',  float(M_1m[idx])))
        if not np.isnan(M_15m[idx]): stack.append(('15m', float(M_15m[idx])))
        stack.sort(key=lambda x: -x[1])
        row['crm_stack'] = '>'.join(s[0] for s in stack)
        # Distance features in σ units
        row['dist_15s_1m']  = round(float(dist_15s_1m[idx]),  3) if not np.isnan(dist_15s_1m[idx])  else None
        row['dist_1m_15m']  = round(float(dist_1m_15m[idx]),  3) if not np.isnan(dist_1m_15m[idx])  else None
        row['dist_15s_15m'] = round(float(dist_15s_15m[idx]), 3) if not np.isnan(dist_15s_15m[idx]) else None
        row['fan_width']    = round(float(fan_width[idx]),    3) if not np.isnan(fan_width[idx])    else None
        # Fan convergence (negative = gaps narrowing → cusp building)
        row['fan_change_10m'] = (round(float(delta_dist_15s_1m_10m[idx] + delta_dist_1m_15m_10m[idx]
                                                              + delta_dist_15s_15m_10m[idx]), 3)
                                              if not (np.isnan(delta_dist_15s_1m_10m[idx])
                                                          or np.isnan(delta_dist_1m_15m_10m[idx])
                                                          or np.isnan(delta_dist_15s_15m_10m[idx]))
                                              else None)
        # CRM crossings: bars since most recent cross (each pair, each direction)
        # -1 means "no such crossing within loaded window"
        row['bars_since_15s_x_1m_up']  = int(bu_15s_1m[idx])  if bu_15s_1m[idx]  >= 0 else None
        row['bars_since_15s_x_1m_dn']  = int(bd_15s_1m[idx])  if bd_15s_1m[idx]  >= 0 else None
        row['bars_since_1m_x_15m_up']  = int(bu_1m_15m[idx])  if bu_1m_15m[idx]  >= 0 else None
        row['bars_since_1m_x_15m_dn']  = int(bd_1m_15m[idx])  if bd_1m_15m[idx]  >= 0 else None
        row['bars_since_px_x_15m_up']  = int(bu_px_15m[idx])  if bu_px_15m[idx]  >= 0 else None
        row['bars_since_px_x_15m_dn']  = int(bd_px_15m[idx])  if bd_px_15m[idx]  >= 0 else None
        # 15s CRM vs 1h structural rails (HL RM z-space)
        # dist_15s_to_Mh_sigma > 0 means 15s CRM is above the upper rail
        # dist_15s_to_Ml_sigma < 0 means 15s CRM is below the lower rail
        if not np.isnan(Mh[idx]) and Sh[idx] > 0:
            d_hi = (M_15s[idx] - Mh[idx]) / Sh[idx]
            row['dist_15s_to_Mh_sigma'] = round(float(d_hi), 3)
            row['near_1h_Mh']           = int(abs(d_hi) <= 0.5)
            row['above_1h_Mh']          = int(d_hi > 0)
        else:
            row['dist_15s_to_Mh_sigma'] = None
            row['near_1h_Mh'] = 0
            row['above_1h_Mh'] = 0
        if not np.isnan(Ml[idx]) and Sl[idx] > 0:
            d_lo = (M_15s[idx] - Ml[idx]) / Sl[idx]
            row['dist_15s_to_Ml_sigma'] = round(float(d_lo), 3)
            row['near_1h_Ml']           = int(abs(d_lo) <= 0.5)
            row['below_1h_Ml']          = int(d_lo < 0)
        else:
            row['dist_15s_to_Ml_sigma'] = None
            row['near_1h_Ml'] = 0
            row['below_1h_Ml'] = 0
        # 15s CRM × 1h rail crossings (regime-transition events, noisy)
        row['bars_since_15s_x_Mh_up']  = int(bu_15s_Mh[idx])  if bu_15s_Mh[idx]  >= 0 else None
        row['bars_since_15s_x_Mh_dn']  = int(bd_15s_Mh[idx])  if bd_15s_Mh[idx]  >= 0 else None
        row['bars_since_15s_x_Ml_up']  = int(bu_15s_Ml[idx])  if bu_15s_Ml[idx]  >= 0 else None
        row['bars_since_15s_x_Ml_dn']  = int(bd_15s_Ml[idx])  if bd_15s_Ml[idx]  >= 0 else None
        # 15m CRM vs 1h rails — the SIGNIFICANT structural relationship
        if not np.isnan(Mh[idx]) and Sh[idx] > 0:
            d_15m_hi = (M_15m[idx] - Mh[idx]) / Sh[idx]
            row['dist_15m_to_Mh_sigma'] = round(float(d_15m_hi), 3)
            row['15m_near_1h_Mh']       = int(abs(d_15m_hi) <= 0.5)
            row['15m_above_1h_Mh']      = int(d_15m_hi > 0)
        else:
            row['dist_15m_to_Mh_sigma'] = None
            row['15m_near_1h_Mh'] = 0
            row['15m_above_1h_Mh'] = 0
        if not np.isnan(Ml[idx]) and Sl[idx] > 0:
            d_15m_lo = (M_15m[idx] - Ml[idx]) / Sl[idx]
            row['dist_15m_to_Ml_sigma'] = round(float(d_15m_lo), 3)
            row['15m_near_1h_Ml']       = int(abs(d_15m_lo) <= 0.5)
            row['15m_below_1h_Ml']      = int(d_15m_lo < 0)
        else:
            row['dist_15m_to_Ml_sigma'] = None
            row['15m_near_1h_Ml'] = 0
            row['15m_below_1h_Ml'] = 0
        # 15m × 1h rail crossings (THE structural regime-shift event)
        row['bars_since_15m_x_Mh_up']  = int(bu_15m_Mh[idx])  if bu_15m_Mh[idx]  >= 0 else None
        row['bars_since_15m_x_Mh_dn']  = int(bd_15m_Mh[idx])  if bd_15m_Mh[idx]  >= 0 else None
        row['bars_since_15m_x_Ml_up']  = int(bu_15m_Ml[idx])  if bu_15m_Ml[idx]  >= 0 else None
        row['bars_since_15m_x_Ml_dn']  = int(bd_15m_Ml[idx])  if bd_15m_Ml[idx]  >= 0 else None
        # Most recent crossing (regardless of pair, direction-signed). Useful for "stuff happens" event.
        recent_up = [b for b in (bu_15s_1m[idx], bu_1m_15m[idx], bu_px_15m[idx]) if b >= 0]
        recent_dn = [b for b in (bd_15s_1m[idx], bd_1m_15m[idx], bd_px_15m[idx]) if b >= 0]
        row['bars_since_any_cross_up'] = min(recent_up) if recent_up else None
        row['bars_since_any_cross_dn'] = min(recent_dn) if recent_dn else None

        # Match user direction: does alignment AGREE with their trade?
        if p.get('direction') == 'LONG':
            row['align_matches_dir'] = int(up_count >= 2 and down_count == 0)
            row['align_against_dir'] = int(down_count >= 2)
        elif p.get('direction') == 'SHORT':
            row['align_matches_dir'] = int(down_count >= 2 and up_count == 0)
            row['align_against_dir'] = int(up_count >= 2)
        else:
            row['align_matches_dir'] = 0
            row['align_against_dir'] = 0
        # Setup classifications
        zhi = row['z_1h_high'] if row['z_1h_high'] is not None else 0
        zlo = row['z_1h_low']  if row['z_1h_low']  is not None else 0
        row['v6_fires_short']    = int(row['z_15m'] >= 1.5 and row['slope_15s_flipped_dn'])
        row['v6_fires_long']     = int(row['z_15m'] <= -1.5 and row['slope_15s_flipped_up'])
        row['compression_short'] = int(zhi > 0 and row['band_rank_60'] < 0.3)
        row['compression_long']  = int(zlo < 0 and row['band_rank_60'] < 0.3)
        row['structural_short']  = int(zhi >= 0.3)
        row['structural_long']   = int(zlo <= -0.3)
        out.append(row)
    return out


def main():
    # Collect all canonical pick files (one per date_range)
    pick_files = sorted(glob.glob('DATA/cusp_picks/picks_*_multi.json'))
    if not pick_files:
        print('No pick files found in DATA/cusp_picks/picks_*_multi.json')
        return

    all_rows = []
    for path in pick_files:
        with open(path) as f:
            data = json.load(f, parse_constant=lambda x: None)
        date_str = data.get('date_range', os.path.basename(path).split('_')[1])
        # Handle date-range syntax  YYYY-MM-DD_to_YYYY-MM-DD
        if '_to_' in date_str:
            date_str = date_str.split('_to_')[0]
        picks = data.get('picks', [])
        print(f'{path}:  {len(picks)} picks  (start date {date_str})')
        rows = extract_for_day(date_str, picks)
        all_rows.extend(rows)
        print(f'  -> {len(rows)} primitives extracted')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if not all_rows:
        print('No rows extracted.')
        return
    cols = list(all_rows[0].keys())
    with open(OUT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f'\nWrote {len(all_rows)} rows to {OUT_PATH}')

    # Quick summary
    df = pd.DataFrame(all_rows)
    print(f'\n=== Summary ===')
    print(f'Total picks: {len(df)}  '
              f'(L: {(df["direction"]=="LONG").sum()}, S: {(df["direction"]=="SHORT").sum()})')
    print(f'Winners (MFE>=2*MAE): {df["is_winner"].sum()} / {len(df)} = {100*df["is_winner"].mean():.0f}%')
    print(f'Total fwd MFE: ${df["mfe_dollars"].sum():.0f}  total fwd MAE: ${df["mae_dollars"].sum():.0f}')
    print()
    print('Setup-classification hit rate:')
    for col in ['v6_fires_short', 'v6_fires_long', 'compression_short', 'compression_long',
                    'structural_short', 'structural_long',
                    'align_matches_dir', 'align_against_dir']:
        n = df[col].sum()
        print(f'  {col:<22}  fires on {n:>3} / {len(df)} picks  ({100*n/len(df):.0f}%)')
    print()
    # Conditional MFE by classification
    print('Mean MFE per setup class:')
    for col in ['v6_fires_short', 'v6_fires_long',
                    'compression_short', 'compression_long',
                    'align_matches_dir', 'align_against_dir']:
        sub = df[df[col]==1]
        if len(sub):
            print(f'  {col:<22}  n={len(sub):>3}  mean MFE ${sub["mfe_dollars"].mean():.0f}  '
                      f'win rate {100*sub["is_winner"].mean():.0f}%')

    # Alignment-state breakdown
    print()
    print('Per-pick alignment state distribution + outcomes:')
    for state, sub in df.groupby('align_state'):
        if len(sub) == 0:
            continue
        print(f'  {state:<11}  n={len(sub):>3}  mean MFE ${sub["mfe_dollars"].mean():.0f}  '
                  f'mean MAE ${sub["mae_dollars"].mean():.0f}  '
                  f'win rate {100*sub["is_winner"].mean():.0f}%')


if __name__ == '__main__':
    main()
