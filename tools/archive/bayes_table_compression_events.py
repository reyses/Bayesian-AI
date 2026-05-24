"""Direct compression-event Bayesian table.

User insight (2026-05-10): the 3 stable cells found earlier all described
COMPRESSION states (velocity collapsed below trend, vol_sigma crushed,
vol_velocity decelerating). The structural common element is compression,
which historically precedes expansion (validated 2026-05-09).

Direct compression detector:
    sigma_short  = std of 5s close over short window (e.g. 5 min  = 60 5s bars)
    sigma_long   = std of 5s close over long window  (e.g. 1 hr  = 720 5s bars)
    ratio        = sigma_short / sigma_long

    compression = ratio below pop q10 (extreme low short-term variation
                  relative to long-term baseline)
    expansion   = ratio above pop q90

Tests:
    1. P(fwd_ret > 0 | compression entry)
    2. Per (TF combo, side, TOD, DOW) bias and EV
    3. Multi-scale compression: 5min/1h, 15min/4h, 1h/4h
    4. Directional skew (do compressions lead to UP or DOWN expansions?)

USAGE
    python tools/bayes_table_compression_events.py
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


# (short_window_5s_bars, long_window_5s_bars, label)
DEFAULT_SCALES = [
    (60,   720,  '5m_vs_1h'),     # 5min short / 1h long
    (180,  2880, '15m_vs_4h'),    # 15min / 4h
    (720,  2880, '1h_vs_4h'),     # 1h / 4h
    (12,   720,  '1m_vs_1h'),     # 1min / 1h (fast)
    (60,   2880, '5m_vs_4h'),     # 5min / 4h
]


def beta_p(k, n):
    a, b = k+1, n-k+1
    return (float(a/(a+b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def load_close_day(day: str):
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path): return None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df[['timestamp', 'close']].sort_values('timestamp').reset_index(drop=True)


def compute_compression_ratio(close: np.ndarray, short_w: int, long_w: int):
    """ratio = std(close, short_w) / std(close, long_w)
    Both rolling, computed at each 5s bar."""
    s = pd.Series(close)
    sig_short = s.rolling(short_w, min_periods=max(2, short_w // 4)).std().values
    sig_long  = s.rolling(long_w,  min_periods=max(2, long_w  // 4)).std().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = sig_short / sig_long
    return sig_short, sig_long, ratio


def detect_threshold_crossings(values: np.ndarray, ts: np.ndarray,
                                 upper_thr: float, lower_thr: float,
                                 min_dwell_s: int = 60) -> list[dict]:
    n = len(values)
    finite = np.isfinite(values)
    high = (values > upper_thr) & finite
    low  = (values < lower_thr) & finite
    events = []
    for side, mask in [('high_q90+', high), ('low_q10-', low)]:
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
                        'peak_value': float(np.min(values[i:j])) if side=='low_q10-'
                                       else float(np.max(values[i:j])),
                    })
                i = j
            else:
                i += 1
    events.sort(key=lambda e: e['start_ts'])
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default=None,
                     help='Comma-separated scale labels (default: all)')
    ap.add_argument('--lookahead-min', type=int, default=60)
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_compression_events')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.scales:
        wanted = args.scales.split(',')
        scales = [s for s in DEFAULT_SCALES if s[2] in wanted]
    else:
        scales = DEFAULT_SCALES

    is_days  = sorted([os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/5s/2025_*.parquet')])
    oos_days = sorted([os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/5s/2026_*.parquet')])
    all_days = [(d, 'IS') for d in is_days] + [(d, 'OOS') for d in oos_days]

    lookahead_s = args.lookahead_min * 60
    summary_rows = []
    all_events_by_scale = {}

    for short_w, long_w, label in scales:
        print(f'\n=== Compression scale {label}  '
               f'(sigma_short={short_w*5}s / sigma_long={long_w*5}s) ===')
        # PASS 1: gather all IS ratios for quantile thresholds
        is_ratios = []
        for day in tqdm(is_days, desc='  gather IS ratios'):
            close_df = load_close_day(day)
            if close_df is None: continue
            close = close_df['close'].values
            if len(close) < long_w + 1: continue
            _, _, r = compute_compression_ratio(close, short_w, long_w)
            is_ratios.append(r[np.isfinite(r)])
        if not is_ratios:
            print('  no data'); continue
        all_is = np.concatenate(is_ratios)
        q10 = float(np.quantile(all_is, 0.10))
        q90 = float(np.quantile(all_is, 0.90))
        print(f'  IS thresholds:  q10={q10:.3f}  q90={q90:.3f}')
        print(f'  population min={all_is.min():.3f}  median={np.median(all_is):.3f}  max={all_is.max():.3f}')

        # PASS 2: detect events + forward returns
        rows = []
        for day, split in tqdm(all_days, desc=f'  {label} scan'):
            close_df = load_close_day(day)
            if close_df is None: continue
            ts = close_df['timestamp'].values.astype(np.int64)
            close = close_df['close'].values
            if len(close) < long_w + 1: continue
            _, _, ratio = compute_compression_ratio(close, short_w, long_w)
            events = detect_threshold_crossings(ratio, ts, q90, q10, min_dwell_s=60)
            for e in events:
                i_start = int(np.searchsorted(ts, e['start_ts']))
                i_fwd   = int(np.searchsorted(ts, e['start_ts'] + lookahead_s))
                if i_start >= len(close) or i_fwd >= len(close): continue
                fwd_ret = float(close[i_fwd] - close[i_start])
                w = close[i_start:i_fwd+1]
                fwd_mfe = float(w.max() - close[i_start])
                fwd_mae = float(w.min() - close[i_start])
                rows.append({
                    'scale': label, 'split': split, 'day': day,
                    'side': e['side'], 'start_ts': e['start_ts'],
                    'duration_min': e['duration_min'],
                    'peak_ratio': e['peak_value'],
                    'fwd_ret': fwd_ret, 'fwd_mfe': fwd_mfe, 'fwd_mae': fwd_mae,
                    'tod_hour': datetime.fromtimestamp(e['start_ts'],
                                                         tz=timezone.utc).hour,
                    'dow': datetime.fromtimestamp(e['start_ts'],
                                                    tz=timezone.utc).strftime('%a'),
                })
        ev = pd.DataFrame(rows)
        if ev.empty: continue
        ev.to_csv(os.path.join(args.out_dir, f'events_{label}.csv'), index=False)
        all_events_by_scale[label] = ev
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
                'scale': label, 'side': side,
                'n_is': n_is_s, 'n_oos': n_oos_s,
                'P_up_IS': round(p_is, 4),
                'CI_IS_lo': round(lo_is, 4), 'CI_IS_hi': round(hi_is, 4),
                'P_up_OOS': round(p_oos, 4) if pd.notna(p_oos) else np.nan,
                'sign_match': sign_match,
                'mean_fwd_IS':  round(float(is_g['fwd_ret'].mean()), 2),
                'mean_fwd_OOS': round(float(oos_g['fwd_ret'].mean()), 2) if n_oos_s > 0 else np.nan,
                'med_dur_min':  round(float(is_g['duration_min'].median()), 2),
            })

    summary = pd.DataFrame(summary_rows)
    summary['IS_edge'] = (summary['P_up_IS'] - 0.5).abs()
    summary = summary.sort_values('IS_edge', ascending=False)
    summary.to_csv(os.path.join(args.out_dir, 'compression_summary.csv'), index=False)
    print('\n\n=== COMPRESSION/EXPANSION EVENT SUMMARY ===')
    cols = ['scale','side','n_is','n_oos','P_up_IS','CI_IS_lo','CI_IS_hi',
             'P_up_OOS','sign_match','mean_fwd_IS','mean_fwd_OOS','med_dur_min']
    print(summary[cols].to_string(index=False))

    strong = summary[(summary['sign_match']==1) & (summary['IS_edge']>=0.05)
                      & (summary['n_oos']>=30)].copy()
    print(f'\n=== STRONG SIGN-STABLE (IS edge>=5pp + OOS sign match) ===')
    print(strong[cols].to_string(index=False) if not strong.empty else '  (none)')

    # ===== TOD × scale × side analysis on the strongest scale =====
    if not summary.empty:
        best_scale = summary.iloc[0]['scale']
        best_ev = all_events_by_scale.get(best_scale)
        if best_ev is not None:
            print(f'\n=== TOD pattern for best scale: {best_scale} ===')
            for side, g in best_ev.groupby('side'):
                print(f'\n  side={side}:')
                for hr, h_g in g.groupby('tod_hour'):
                    is_h = h_g[h_g['split']=='IS']
                    oos_h = h_g[h_g['split']=='OOS']
                    if len(is_h) < 20: continue
                    p_is, _, _ = beta_p((is_h['fwd_ret']>0).sum(), len(is_h))
                    if len(oos_h) >= 10:
                        p_oos, _, _ = beta_p((oos_h['fwd_ret']>0).sum(), len(oos_h))
                    else:
                        p_oos = np.nan
                    print(f'    UTC {hr:>2d}  IS:n={len(is_h):>4d} P={p_is:.3f}   '
                           f'OOS:n={len(oos_h):>3d} P={p_oos:.3f}' if pd.notna(p_oos)
                           else f'    UTC {hr:>2d}  IS:n={len(is_h):>4d} P={p_is:.3f}')

    # Chart
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(summary) + 2)))
        sub = summary.iloc[::-1]
        y = np.arange(len(sub))
        is_p = sub['P_up_IS'].values
        oos_p = sub['P_up_OOS'].fillna(0.5).values
        ax.barh(y - 0.2, is_p, height=0.4,
                  color=['#43A047' if p>0.5 else '#E53935' for p in is_p],
                  alpha=0.85, label='IS')
        ax.barh(y + 0.2, oos_p, height=0.4,
                  color=['#A5D6A7' if p>0.5 else '#EF9A9A' for p in oos_p],
                  alpha=0.85, label='OOS')
        ax.axvline(0.5, color='black', lw=0.6)
        labels = [f"{r['scale']} / {r['side']}  n={r['n_is']}/{r['n_oos']}"
                   + (' [match]' if r['sign_match']==1 else ' [flip]' if r['sign_match']==0 else ' [no-OOS]')
                   for _, r in sub.iterrows()]
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('P(fwd_ret_60m > 0)')
        ax.set_title('Compression/expansion ratio events at multiple scales')
        ax.legend(); ax.set_xlim(0, 1); ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(args.out_dir, 'compression_summary.png')
        plt.savefig(out_png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
