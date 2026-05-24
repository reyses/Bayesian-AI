"""Feature events anchored on FEATURE'S OWN regression mean (RM-anchored).

User insight (2026-05-10): the HL RM event harvest worked because events
were anchored on the slow REGRESSION MEAN, not on a TF magnitude. Without
the RM anchor, 'extreme value' has no directional meaning.

This tool extends the framework correctly:
    1. For each feature, compute ITS OWN rolling regression mean + SE
       over a slow-TF lookback (e.g., 1h = 720 5s bars)
    2. Detect events: feature crosses its OWN RM ± 3 sigma
    3. Forward returns from event entry
    4. Per (feature, side, TOD, DOW) directional bias + EV

Now features carry directional structure: 'vol_sigma > its-own-RM + 3σ'
is a SPIKE (regime change), 'vol_sigma < its-own-RM - 3σ' is a CRUSH
(compression). These are not symmetric phenomena, like HL RM crashes
weren't symmetric to HL RM rallies.

USAGE
    python tools/bayes_table_feature_rm_events.py
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


# Pick features that aren't already z-scores (avoid double z)
DEFAULT_FEATURES = [
    ('L2_1h',  'L2_1h_vol_sigma_12'),
    ('L2_15m', 'L2_15m_vol_sigma_12'),
    ('L2_5m',  'L2_5m_vol_sigma_9'),
    ('L1_5m',  'L1_5m_vol_velocity_1b'),
    ('L1_5m',  'L1_5m_vol_accel_1b'),
    ('L2_5m',  'L2_5m_vol_velocity_9'),
    ('L2_15m', 'L2_15m_vol_velocity_12'),
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


def load_feature_day(layer: str, feature: str, day: str) -> pd.DataFrame | None:
    path = f'DATA/ATLAS/FEATURES_5s_v2/{layer}/{day}.parquet'
    if not os.path.exists(path): return None
    df = pd.read_parquet(path)
    if feature not in df.columns: return None
    return df[['timestamp', feature]]


def load_close_day(day: str) -> pd.DataFrame | None:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path): return None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df[['timestamp', 'close']].sort_values('timestamp').reset_index(drop=True)


def detect_rm_crossings(z: np.ndarray, ts: np.ndarray, k: float = 3.0,
                          min_dwell_s: int = 60) -> list[dict]:
    """Detect events where the feature crosses its OWN RM by +/-k sigma."""
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
    ap.add_argument('--rm-lookback-5s', type=int, default=720,
                     help='Rolling lookback for feature RM (default 1h = 720 5s bars)')
    ap.add_argument('--k-sigma', type=float, default=3.0)
    ap.add_argument('--lookahead-min', type=int, default=60)
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_feature_rm_events')
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
        print(f'\n=== {feature} (RM lookback {args.rm_lookback_5s} 5s bars, k={args.k_sigma}) ===')
        all_event_rows = []
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
            ts = merged['timestamp'].values
            close = merged['close'].values
            v = merged[feature].values
            # Rolling RM and SE over lookback
            v_series = pd.Series(v)
            rm = v_series.rolling(args.rm_lookback_5s, min_periods=60).mean().values
            sd = v_series.rolling(args.rm_lookback_5s, min_periods=60).std().values
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (v - rm) / sd
            events = detect_rm_crossings(z, ts, k=args.k_sigma, min_dwell_s=60)
            for e in events:
                i_start = int(np.searchsorted(ts, e['start_ts']))
                i_fwd   = int(np.searchsorted(ts, e['start_ts'] + lookahead_s))
                if i_start >= len(close) or i_fwd >= len(close): continue
                fwd_ret = float(close[i_fwd] - close[i_start])
                w = close[i_start:i_fwd+1]
                fwd_mfe = float(w.max() - close[i_start]) if len(w) else 0
                fwd_mae = float(w.min() - close[i_start]) if len(w) else 0
                all_event_rows.append({
                    'feature': feature, 'split': split, 'day': day,
                    'side': e['side'],
                    'start_ts': e['start_ts'],
                    'duration_min': e['duration_min'],
                    'max_abs_z': e['max_abs_z'],
                    'signed_max_z': e['signed_max_z'],
                    'fwd_ret': fwd_ret, 'fwd_mfe': fwd_mfe, 'fwd_mae': fwd_mae,
                    'tod_hour': datetime.fromtimestamp(e['start_ts'],
                                                         tz=timezone.utc).hour,
                    'dow': datetime.fromtimestamp(e['start_ts'],
                                                    tz=timezone.utc).strftime('%a'),
                })
        ev = pd.DataFrame(all_event_rows)
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
        print('No events found across any features.'); return
    summary['IS_edge'] = (summary['P_up_IS'] - 0.5).abs()
    summary = summary.sort_values('IS_edge', ascending=False)
    summary.to_csv(os.path.join(args.out_dir, 'feature_rm_event_summary.csv'),
                    index=False)
    print('\n\n=== RM-ANCHORED FEATURE EVENT SUMMARY ===')
    cols = ['feature','side','n_is','n_oos','P_up_IS','CI_IS_lo','CI_IS_hi',
             'P_up_OOS','sign_match','mean_fwd_IS','mean_fwd_OOS','mean_abs_z',
             'med_dur_min']
    print(summary[cols].to_string(index=False))

    strong = summary[(summary['sign_match']==1) & (summary['IS_edge']>=0.05)
                      & (summary['n_oos']>=30)].copy()
    print(f'\n=== STRONG SIGN-STABLE (IS edge>=5pp + OOS sign match + n_oos>=30) ===')
    print(strong[cols].to_string(index=False))
    strong.to_csv(os.path.join(args.out_dir, 'STRONG_feature_rm_events.csv'),
                   index=False)

    # Directional asymmetry per feature
    print(f'\n=== DIRECTIONAL ASYMMETRY (above vs below event count) ===')
    asym_rows = []
    for feature, g in summary.groupby('feature'):
        above = g[g['side'].str.startswith('above')]
        below = g[g['side'].str.startswith('below')]
        n_above_is = above['n_is'].sum() if not above.empty else 0
        n_below_is = below['n_is'].sum() if not below.empty else 0
        if n_above_is + n_below_is == 0: continue
        asym_rows.append({
            'feature': feature,
            'n_above_is': n_above_is, 'n_below_is': n_below_is,
            'P_above': round(n_above_is / (n_above_is + n_below_is), 3),
        })
    asym_df = pd.DataFrame(asym_rows).sort_values('P_above', ascending=False)
    print(asym_df.to_string(index=False))
    asym_df.to_csv(os.path.join(args.out_dir, 'directional_asymmetry.csv'),
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
    ax.set_title(f'RM-anchored feature events  k={args.k_sigma}sigma  '
                  f'lookback={args.rm_lookback_5s} 5s bars')
    ax.legend(); ax.set_xlim(0, 1); ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'feature_rm_event_summary.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
