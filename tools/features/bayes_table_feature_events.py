"""Generalized feature-event harvest: apply the HL RM event framework to other V2 features.

For each feature column listed, compute:
    1. Population q05 and q95 (or user-set quantiles)
    2. Detect events: bars where feature CROSSES into extreme zone
       (from below q95 -> above q95 = HIGH event; vice versa for LOW)
    3. Per event: forward return at 60m, tod_hour, dow, sigma at entry
    4. Aggregate: directional bias of fwd_return, EV per cell

Tests the user's intuition (2026-05-10): the technique applied to CRM/HL RM
should produce edges on other V2 features too — particularly volume,
swing_noise, hurst, vol_velocity.

USAGE
    python tools/bayes_table_feature_events.py
    python tools/bayes_table_feature_events.py --features L2_1h_vol_sigma_12,L1_5m_vol_velocity_1b
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


# Default feature set (per memory/project_useful_v2_signals.md)
DEFAULT_FEATURES = [
    # Volume regime/leading signals
    ('L2_1h', 'L2_1h_vol_sigma_12'),
    ('L2_15m', 'L2_15m_vol_sigma_12'),
    ('L2_5m',  'L2_5m_vol_sigma_9'),
    ('L1_5m',  'L1_5m_vol_velocity_1b'),
    ('L1_5m',  'L1_5m_vol_accel_1b'),
    ('L2_5m',  'L2_5m_vol_velocity_9'),
    # Velocity
    ('L2_15m', 'L2_15m_price_velocity_12'),
    ('L2_1h',  'L2_1h_price_velocity_12'),
    # Swing noise + hurst
    ('L3_15m', 'L3_15m_swing_noise_12'),
    ('L3_15m', 'L3_15m_hurst_12'),
    # Z position
    ('L3_15m', 'L3_15m_z_se_12'),
    # Bar shape
    ('L1_5m',  'L1_5m_body'),
    ('L1_5m',  'L1_5m_bar_range'),
]


def beta_p(k, n):
    a, b = k+1, n-k+1
    return (float(a/(a+b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def load_feature_day(layer: str, feature: str, day: str) -> pd.DataFrame | None:
    path = f'DATA/ATLAS/FEATURES_5s_v2/{layer}/{day}.parquet'
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if feature not in df.columns:
        return None
    return df[['timestamp', feature]]


def load_close_day(day: str) -> pd.DataFrame | None:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df[['timestamp', 'close']].sort_values('timestamp').reset_index(drop=True)


def detect_threshold_crossings(values: np.ndarray, ts: np.ndarray,
                                  upper_thr: float, lower_thr: float,
                                  min_dwell_s: int = 60) -> list[dict]:
    """Detect transitions INTO extreme zones (event ENTRIES)."""
    n = len(values)
    finite = np.isfinite(values)
    above = (values > upper_thr) & finite
    below = (values < lower_thr) & finite
    events = []
    # Transitions
    for side, mask in [('above_upper', above), ('below_lower', below)]:
        i = 0
        while i < n:
            if mask[i] and (i == 0 or not mask[i-1]):
                j = i
                while j < n and mask[j]:
                    j += 1
                duration_s = int(ts[j-1] - ts[i]) if j > i else 0
                if duration_s >= min_dwell_s:
                    events.append({
                        'side': side, 'start_ts': int(ts[i]),
                        'end_ts': int(ts[j-1]),
                        'duration_s': duration_s,
                        'duration_min': round(duration_s / 60.0, 2),
                        'peak_value': float(np.max(values[i:j])) if side=='above_upper'
                                      else float(np.min(values[i:j])),
                    })
                i = j
            else:
                i += 1
    events.sort(key=lambda e: e['start_ts'])
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=None,
                     help='Comma-separated; default = built-in list')
    ap.add_argument('--upper-q', type=float, default=0.95)
    ap.add_argument('--lower-q', type=float, default=0.05)
    ap.add_argument('--lookahead-min', type=int, default=60)
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_feature_events')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.features:
        # parse "layer.feature" or just feature; assume layer derivable
        feat_pairs = []
        for f in args.features.split(','):
            f = f.strip()
            # Layer = up to second underscore (L2_1h)
            parts = f.split('_')
            layer = '_'.join(parts[:2])
            feat_pairs.append((layer, f))
    else:
        feat_pairs = DEFAULT_FEATURES

    print(f'Processing {len(feat_pairs)} features...')

    # Build IS day list (2025) and OOS (2026)
    is_days  = sorted([os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/5s/2025_*.parquet')])
    oos_days = sorted([os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/5s/2026_*.parquet')])
    all_days = [(d, 'IS') for d in is_days] + [(d, 'OOS') for d in oos_days]
    print(f'Days: {len(is_days)} IS  +  {len(oos_days)} OOS  =  {len(all_days)} total')

    lookahead_s = args.lookahead_min * 60
    summary_rows = []

    for layer, feature in feat_pairs:
        print(f'\n=== {feature} (layer {layer}) ===')
        # PASS 1: gather all values to compute population quantiles (IS only)
        all_is_values = []
        for day in tqdm(is_days, desc=f'  IS gather'):
            f_df = load_feature_day(layer, feature, day)
            if f_df is None: continue
            v = f_df[feature].dropna().values
            all_is_values.append(v)
        if not all_is_values:
            print(f'  no data, skipping'); continue
        all_is = np.concatenate(all_is_values)
        upper_thr = float(np.quantile(all_is, args.upper_q))
        lower_thr = float(np.quantile(all_is, args.lower_q))
        print(f'  thresholds  upper_q={args.upper_q}: {upper_thr:.4f}  '
               f'lower_q={args.lower_q}: {lower_thr:.4f}')
        feature_baseline_pos = float((all_is > 0).mean()) if all_is.std() > 0 else 0.5

        # PASS 2: detect events and compute forward returns
        rows = []
        for day, split in tqdm(all_days, desc=f'  detect+returns'):
            f_df = load_feature_day(layer, feature, day)
            close_df = load_close_day(day)
            if f_df is None or close_df is None: continue
            # Merge on timestamp (close is at 5s grid)
            close_df['timestamp'] = close_df['timestamp'].astype(np.int64)
            f_df = f_df.copy()
            f_df['timestamp'] = f_df['timestamp'].astype(np.int64)
            merged = pd.merge_asof(
                close_df.sort_values('timestamp'),
                f_df.sort_values('timestamp'),
                on='timestamp', direction='backward')
            ts = merged['timestamp'].values
            close = merged['close'].values
            values = merged[feature].values
            events = detect_threshold_crossings(values, ts, upper_thr, lower_thr)
            for e in events:
                # Forward return at lookahead horizon
                i_start = int(np.searchsorted(ts, e['start_ts']))
                i_fwd   = int(np.searchsorted(ts, e['start_ts'] + lookahead_s))
                if i_start >= len(close) or i_fwd >= len(close): continue
                fwd_ret = float(close[i_fwd] - close[i_start])
                w = close[i_start:i_fwd+1]
                fwd_mfe = float(w.max() - close[i_start]) if len(w) > 0 else 0
                fwd_mae = float(w.min() - close[i_start]) if len(w) > 0 else 0
                rows.append({
                    'feature': feature, 'split': split, 'day': day,
                    'side': e['side'], 'start_ts': e['start_ts'],
                    'duration_min': e['duration_min'],
                    'peak_value': e['peak_value'],
                    'fwd_ret': fwd_ret, 'fwd_mfe': fwd_mfe, 'fwd_mae': fwd_mae,
                    'tod_hour': datetime.fromtimestamp(e['start_ts'],
                                                         tz=timezone.utc).hour,
                    'dow': datetime.fromtimestamp(e['start_ts'],
                                                    tz=timezone.utc).strftime('%a'),
                })
        ev_df = pd.DataFrame(rows)
        if ev_df.empty:
            print(f'  no events detected for {feature}'); continue
        ev_df.to_csv(os.path.join(args.out_dir, f'events_{feature}.csv'),
                       index=False)
        print(f'  events: {len(ev_df):,}  (IS: {(ev_df["split"]=="IS").sum():,}  '
               f'OOS: {(ev_df["split"]=="OOS").sum():,})')

        # Aggregate per side: IS direction skew + magnitude
        for side, g in ev_df.groupby('side'):
            is_g = g[g['split'] == 'IS']
            oos_g = g[g['split'] == 'OOS']
            n_is = len(is_g); n_oos = len(oos_g)
            if n_is < 30: continue
            up_is = int((is_g['fwd_ret'] > 0).sum())
            up_oos = int((oos_g['fwd_ret'] > 0).sum()) if n_oos > 0 else 0
            p_up_is, lo_is, hi_is = beta_p(up_is, n_is)
            if n_oos >= 10:
                p_up_oos, lo_oos, hi_oos = beta_p(up_oos, n_oos)
                sign_match = int(np.sign(p_up_is - 0.5) == np.sign(p_up_oos - 0.5))
            else:
                p_up_oos = lo_oos = hi_oos = np.nan
                sign_match = -1
            summary_rows.append({
                'feature': feature, 'side': side,
                'n_is': n_is, 'n_oos': n_oos,
                'P_up_IS': round(p_up_is, 4),
                'CI_IS_lo': round(lo_is, 4),
                'CI_IS_hi': round(hi_is, 4),
                'P_up_OOS': round(p_up_oos, 4) if pd.notna(p_up_oos) else np.nan,
                'CI_OOS_lo': round(lo_oos, 4) if pd.notna(lo_oos) else np.nan,
                'CI_OOS_hi': round(hi_oos, 4) if pd.notna(hi_oos) else np.nan,
                'sign_match': sign_match,
                'mean_fwd_IS':  round(float(is_g['fwd_ret'].mean()), 2),
                'mean_fwd_OOS': round(float(oos_g['fwd_ret'].mean()), 2) if n_oos > 0 else np.nan,
                'mean_dur_min': round(float(is_g['duration_min'].mean()), 2),
                'med_dur_min':  round(float(is_g['duration_min'].median()), 2),
            })

    # ===== Master summary =====
    summary = pd.DataFrame(summary_rows)
    summary['IS_edge'] = (summary['P_up_IS'] - 0.5).abs()
    summary = summary.sort_values('IS_edge', ascending=False)
    summary.to_csv(os.path.join(args.out_dir, 'feature_event_summary.csv'),
                    index=False)
    print(f'\n\n=== MASTER SUMMARY (sorted by IS edge) ===')
    cols = ['feature', 'side', 'n_is', 'n_oos', 'P_up_IS', 'CI_IS_lo', 'CI_IS_hi',
             'P_up_OOS', 'sign_match', 'mean_fwd_IS', 'mean_fwd_OOS']
    print(summary[cols].to_string(index=False))

    # Strong + sign-stable
    strong = summary[(summary['sign_match'] == 1) & (summary['IS_edge'] >= 0.05)
                      & (summary['n_oos'] >= 30)].copy()
    print(f'\n=== STRONG SIGN-STABLE (IS edge >= 5pp, OOS sign-match, n_oos>=30) ===')
    print(strong[cols].to_string(index=False))
    strong.to_csv(os.path.join(args.out_dir, 'STRONG_feature_events.csv'), index=False)

    # ===== CHART =====
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(summary) + 2)))
        sub = summary.head(30).iloc[::-1]
        y = np.arange(len(sub))
        is_p = sub['P_up_IS'].values
        oos_p = sub['P_up_OOS'].fillna(0.5).values
        colors_is = ['#43A047' if p > 0.5 else '#E53935' for p in is_p]
        colors_oos = ['#A5D6A7' if p > 0.5 else '#EF9A9A' for p in oos_p]
        ax.barh(y - 0.2, is_p, height=0.4, color=colors_is, alpha=0.85, label='IS')
        ax.barh(y + 0.2, oos_p, height=0.4, color=colors_oos, alpha=0.85, label='OOS')
        ax.axvline(0.5, color='black', lw=0.6)
        labels = [f"{r['feature'][:30]} / {r['side']}  n_is={r['n_is']}/n_oos={r['n_oos']}"
                   + (' [match]' if r['sign_match']==1 else
                       ' [flip]' if r['sign_match']==0 else
                       ' [no-OOS]')
                   for _, r in sub.iterrows()]
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('P(fwd_ret_60m > 0)')
        ax.set_title(f'Feature-event harvest: P(fwd>0) per (feature, side)  '
                       f'IS top 30 by |edge|', fontsize=11)
        ax.legend(); ax.set_xlim(0, 1); ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(args.out_dir, 'feature_event_summary.png')
        plt.savefig(out_png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
