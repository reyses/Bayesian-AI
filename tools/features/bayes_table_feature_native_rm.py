"""Feature events with NATIVE-TF regression mean (corrected).

User correction (2026-05-10): the V2 features are step-functions on the
5s grid — they update only at their native TF cadence. Computing a 5s-grid
rolling mean of vol_sigma_12 just averages repeated values.

The CORRECT regression mean is computed at the feature's NATIVE cadence:
    - L2_1h_*    → window of 12 distinct 1h values (12-hour lookback)
    - L2_15m_*   → window of 12 distinct 15m values (3-hour lookback)
    - L2_5m_*    → window of 9 distinct 5m values (45-min lookback)
    - L1_5m_*    → same as L2_5m
    - L3_15m_*   → same as L2_15m

Then ffilled to 5s grid for at-bar event detection.

USAGE
    python tools/bayes_table_feature_native_rm.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Native TF -> (period_s, RM window in native bars)
NATIVE_TF_CONFIG = {
    'L1_5s':  (5,       12),    # 1 min
    'L1_15s': (15,      12),    # 3 min
    'L1_1m':  (60,      15),    # 15 min
    'L1_5m':  (300,     9),     # 45 min
    'L1_15m': (900,     12),    # 3 hr
    'L1_1h':  (3600,    12),    # 12 hr
    'L1_4h':  (14400,   12),    # 48 hr
    'L1_1D':  (86400,   12),    # 12 days
    'L2_5s':  (5,       12),
    'L2_15s': (15,      12),
    'L2_1m':  (60,      15),
    'L2_5m':  (300,     9),
    'L2_15m': (900,     12),
    'L2_1h':  (3600,    12),
    'L2_4h':  (14400,   12),
    'L2_1D':  (86400,   12),
    'L3_5s':  (5,       12),
    'L3_15s': (15,      12),
    'L3_1m':  (60,      15),
    'L3_5m':  (300,     9),
    'L3_15m': (900,     12),
    'L3_1h':  (3600,    12),
    'L3_4h':  (14400,   12),
    'L3_1D':  (86400,   12),
}


DEFAULT_FEATURES = [
    ('L2_1h',  'L2_1h_vol_sigma_12'),
    ('L2_15m', 'L2_15m_vol_sigma_12'),
    ('L2_5m',  'L2_5m_vol_sigma_9'),
    ('L1_5m',  'L1_5m_vol_velocity_1b'),
    ('L2_5m',  'L2_5m_vol_velocity_9'),
    ('L2_15m', 'L2_15m_vol_velocity_12'),
    ('L2_1h',  'L2_1h_vol_velocity_12'),
    ('L3_15m', 'L3_15m_swing_noise_12'),
    ('L3_15m', 'L3_15m_hurst_12'),
    ('L1_5m',  'L1_5m_body'),
    ('L1_5m',  'L1_5m_bar_range'),
    ('L2_15m', 'L2_15m_price_velocity_12'),
    ('L2_1h',  'L2_1h_price_velocity_12'),
]


def beta_p(k, n):
    a, b = k+1, n-k+1
    return (float(a/(a+b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def native_path_rm(feature_5s: np.ndarray, ts_5s: np.ndarray, period_s: int,
                    window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the feature's regression mean + std at the feature's NATIVE
    cadence, then ffill back to the 5s grid.

    Approach: extract the unique-value transitions (where feature changes)
    to get the feature's native path, compute rolling mean+std on that
    native path, then ffill back to the 5s grid using lookahead-clean
    (target_ts - period_s) lookup."""
    # Find native bar transitions: where feature value changes
    changes = np.concatenate([[True],
                              feature_5s[1:] != feature_5s[:-1]])
    native_idx = np.where(changes)[0]
    if len(native_idx) < window + 1:
        return np.full(len(feature_5s), np.nan), np.full(len(feature_5s), np.nan)
    native_vals = feature_5s[native_idx]
    native_ts = ts_5s[native_idx]
    # Rolling RM + SE on native path
    s = pd.Series(native_vals)
    native_mean = s.rolling(window, min_periods=max(2, window // 2)).mean().values
    native_std  = s.rolling(window, min_periods=max(2, window // 2)).std().values
    # FFill back to 5s grid (lookahead-clean: only use bars completed at-or-before)
    target = ts_5s - period_s
    idx = np.searchsorted(native_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(native_ts) - 1)
    M = native_mean[idx]
    S = native_std[idx]
    return M, S


def load_feature_day(layer: str, feature: str, day: str):
    path = f'DATA/ATLAS/FEATURES_5s_v2/{layer}/{day}.parquet'
    if not os.path.exists(path): return None
    df = pd.read_parquet(path)
    if feature not in df.columns: return None
    return df[['timestamp', feature]]


def load_close_day(day: str):
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path): return None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df[['timestamp', 'close']].sort_values('timestamp').reset_index(drop=True)


def detect_rm_crossings(z: np.ndarray, ts: np.ndarray, k: float = 3.0,
                          min_dwell_s: int = 60) -> list[dict]:
    n = len(z)
    finite = np.isfinite(z)
    above = (z >=  k) & finite
    below = (z <= -k) & finite
    events = []
    for side, mask in [('above_+3sigma', above), ('below_-3sigma', below)]:
        i = 0
        while i < n:
            if mask[i] and (i == 0 or not mask[i-1]):
                j = i
                while j < n and mask[j]: j += 1
                duration_s = int(ts[j-1] - ts[i]) if j > i else 0
                if duration_s >= min_dwell_s:
                    events.append({
                        'side': side, 'start_ts': int(ts[i]),
                        'end_ts': int(ts[j-1]),
                        'duration_min': round(duration_s / 60.0, 2),
                        'max_abs_z': float(np.max(np.abs(z[i:j]))),
                        'signed_max_z': float(z[i:j][np.argmax(np.abs(z[i:j]))]),
                    })
                i = j
            else:
                i += 1
    events.sort(key=lambda e: e['start_ts'])
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=None)
    ap.add_argument('--k-sigma', type=float, default=3.0)
    ap.add_argument('--lookahead-min', type=int, default=60)
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_feature_native_rm')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.features:
        feat_pairs = []
        for f in args.features.split(','):
            f = f.strip()
            parts = f.split('_')
            layer = '_'.join(parts[:2])
            feat_pairs.append((layer, f))
    else:
        feat_pairs = DEFAULT_FEATURES

    is_days  = sorted([os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/5s/2025_*.parquet')])
    oos_days = sorted([os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/5s/2026_*.parquet')])
    all_days = [(d, 'IS') for d in is_days] + [(d, 'OOS') for d in oos_days]

    lookahead_s = args.lookahead_min * 60
    summary_rows = []

    for layer, feature in feat_pairs:
        if layer not in NATIVE_TF_CONFIG:
            print(f'  skip {feature}: unknown layer {layer}')
            continue
        period_s, window = NATIVE_TF_CONFIG[layer]
        print(f'\n=== {feature}  native period {period_s}s  RM window {window}  '
               f'(= {period_s*window/60:.0f}min lookback) ===')
        all_rows = []
        for day, split in tqdm(all_days, desc=f'  scan'):
            f_df = load_feature_day(layer, feature, day)
            close_df = load_close_day(day)
            if f_df is None or close_df is None: continue
            close_df['timestamp'] = close_df['timestamp'].astype(np.int64)
            f_df = f_df.copy()
            f_df['timestamp'] = f_df['timestamp'].astype(np.int64)
            merged = pd.merge_asof(
                close_df.sort_values('timestamp'),
                f_df.sort_values('timestamp'),
                on='timestamp', direction='backward')
            ts = merged['timestamp'].values.astype(np.int64)
            close = merged['close'].values
            v = merged[feature].values
            M, S = native_path_rm(v, ts, period_s, window)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (v - M) / S
            events = detect_rm_crossings(z, ts, k=args.k_sigma, min_dwell_s=60)
            for e in events:
                i_start = int(np.searchsorted(ts, e['start_ts']))
                i_fwd   = int(np.searchsorted(ts, e['start_ts'] + lookahead_s))
                if i_start >= len(close) or i_fwd >= len(close): continue
                fwd_ret = float(close[i_fwd] - close[i_start])
                w = close[i_start:i_fwd+1]
                fwd_mfe = float(w.max() - close[i_start]) if len(w) else 0
                fwd_mae = float(w.min() - close[i_start]) if len(w) else 0
                all_rows.append({
                    'feature': feature, 'split': split, 'day': day,
                    'side': e['side'], 'start_ts': e['start_ts'],
                    'duration_min': e['duration_min'],
                    'max_abs_z': e['max_abs_z'],
                    'signed_max_z': e['signed_max_z'],
                    'fwd_ret': fwd_ret, 'fwd_mfe': fwd_mfe, 'fwd_mae': fwd_mae,
                    'tod_hour': datetime.fromtimestamp(e['start_ts'],
                                                         tz=timezone.utc).hour,
                    'dow': datetime.fromtimestamp(e['start_ts'],
                                                    tz=timezone.utc).strftime('%a'),
                })
        ev = pd.DataFrame(all_rows)
        if ev.empty:
            print(f'  no events'); continue
        ev.to_csv(os.path.join(args.out_dir, f'events_{feature}.csv'), index=False)
        n_is = (ev['split']=='IS').sum()
        n_oos = (ev['split']=='OOS').sum()
        print(f'  events: {len(ev):,}  (IS: {n_is:,}  OOS: {n_oos:,})')

        for side, g in ev.groupby('side'):
            is_g = g[g['split']=='IS']
            oos_g = g[g['split']=='OOS']
            n_is_s = len(is_g); n_oos_s = len(oos_g)
            if n_is_s < 30: continue
            up_is = int((is_g['fwd_ret']>0).sum())
            up_oos = int((oos_g['fwd_ret']>0).sum()) if n_oos_s > 0 else 0
            p_is, lo_is, hi_is = beta_p(up_is, n_is_s)
            if n_oos_s >= 10:
                p_oos, lo_oos, hi_oos = beta_p(up_oos, n_oos_s)
                sign_match = int(np.sign(p_is-0.5) == np.sign(p_oos-0.5))
            else:
                p_oos = lo_oos = hi_oos = np.nan
                sign_match = -1
            summary_rows.append({
                'feature': feature, 'side': side,
                'n_is': n_is_s, 'n_oos': n_oos_s,
                'P_up_IS': round(p_is, 4),
                'CI_IS_lo': round(lo_is, 4), 'CI_IS_hi': round(hi_is, 4),
                'P_up_OOS': round(p_oos, 4) if pd.notna(p_oos) else np.nan,
                'sign_match': sign_match,
                'mean_fwd_IS':  round(float(is_g['fwd_ret'].mean()), 2),
                'mean_fwd_OOS': round(float(oos_g['fwd_ret'].mean()), 2) if n_oos_s > 0 else np.nan,
                'mean_abs_z':   round(float(is_g['max_abs_z'].mean()), 2),
                'med_dur_min':  round(float(is_g['duration_min'].median()), 2),
            })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print('No events found.'); return
    summary['IS_edge'] = (summary['P_up_IS'] - 0.5).abs()
    summary = summary.sort_values('IS_edge', ascending=False)
    summary.to_csv(os.path.join(args.out_dir, 'feature_native_rm_summary.csv'),
                    index=False)
    print('\n\n=== NATIVE-TF RM FEATURE EVENT SUMMARY ===')
    cols = ['feature','side','n_is','n_oos','P_up_IS','CI_IS_lo','CI_IS_hi',
             'P_up_OOS','sign_match','mean_fwd_IS','mean_fwd_OOS','mean_abs_z',
             'med_dur_min']
    print(summary[cols].to_string(index=False))

    strong = summary[(summary['sign_match']==1) & (summary['IS_edge']>=0.05)
                      & (summary['n_oos']>=30)].copy()
    print(f'\n=== STRONG SIGN-STABLE (IS edge>=5pp + OOS sign match + n_oos>=30) ===')
    print(strong[cols].to_string(index=False) if not strong.empty else '  (none)')
    if not strong.empty:
        strong.to_csv(os.path.join(args.out_dir, 'STRONG_native_rm_events.csv'),
                       index=False)

    # Chart
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(summary) + 2)))
    sub = summary.head(30).iloc[::-1]
    y = np.arange(len(sub))
    is_p = sub['P_up_IS'].values
    oos_p = sub['P_up_OOS'].fillna(0.5).values
    ax.barh(y - 0.2, is_p, height=0.4, color=['#43A047' if p>0.5 else '#E53935' for p in is_p],
              alpha=0.85, label='IS')
    ax.barh(y + 0.2, oos_p, height=0.4, color=['#A5D6A7' if p>0.5 else '#EF9A9A' for p in oos_p],
              alpha=0.85, label='OOS')
    ax.axvline(0.5, color='black', lw=0.6)
    labels = [f"{r['feature']} / {r['side']}  n={r['n_is']}/{r['n_oos']}"
               + (' [match]' if r['sign_match']==1 else ' [flip]' if r['sign_match']==0 else ' [no-OOS]')
               for _, r in sub.iterrows()]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('P(fwd_ret_60m > 0)')
    ax.set_title(f'Native-TF-RM feature events  k={args.k_sigma}sigma')
    ax.legend(); ax.set_xlim(0, 1); ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'feature_native_rm_summary.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
