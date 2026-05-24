"""Event-segmented bucket explorer characterized by 15m CRM context.

Replaces the regime-label-based selection (flawed, day-aggregate, biased)
with EVENT-BASED selection (3,431 IS macro entries, each independent).

Each event is characterized by a vector of 15m CRM features measured at the
event ENTRY BAR — no lookahead, no day-aggregate label contamination:

    slope_15m       = (M_close[t] - M_close[t - 60min]) / 60min   in $/min
    curv_15m        = derivative of slope_15m over the same window
    z_close_15m     = (5s_close - M_close_15m) / SE_close_15m     at entry
    sigma_rank_15m  = rolling percentile of SE_close_15m (60min)  at entry

Then events are bucketed independently by each axis (quantiles within the
event population) and the LONGEST event per bucket is rendered with a
+/-2hr window context chart so we see what the bucket actually looks like.

USAGE
    python tools/event_bucket_15m_crm.py
    python tools/event_bucket_15m_crm.py --axis slope
    python tools/event_bucket_15m_crm.py --window-hours 1.5
    python tools/event_bucket_15m_crm.py --selection median
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CRM_TF = '15m'
CRM_WINDOW = 12
CRM_PERIOD_S = 900  # 15m
SLOPE_LOOKBACK_5S_BARS = 720    # 60min over 5s grid
RANK_WINDOW_5S_BARS = 720       # 60min rolling window for sigma rank
R2_WINDOW_5S_BARS = 60          # 5min linear regression for R^2 adjusted

AXIS_BIN_DEFS = {
    # Statistical-only labels (per memory/feedback_no_human_regime_terms.md).
    # No metaphors, no trader vocabulary. Each label names the MEASURE.
    'slope': {'n_bins': 5, 'labels': [
        'negative_high_trend',  # q1
        'negative_low_trend',   # q2
        'no_trend',             # q3
        'positive_low_trend',   # q4
        'positive_high_trend',  # q5
    ]},
    'curvature': {'n_bins': 3, 'labels': [
        'negative_curvature', 'no_curvature', 'positive_curvature']},
    'z_close': {'n_bins': 5, 'labels': [
        'negative_far_z',  'negative_near_z',
        'zero_z',
        'positive_near_z', 'positive_far_z',
    ]},
    'sigma_rank': {'n_bins': 5, 'labels': [
        'low_sigma',     'low_mid_sigma',
        'mid_sigma',
        'high_mid_sigma', 'high_sigma',
    ]},
    'r2adj_5m': {'n_bins': 5, 'labels': [
        # R^2_adj is INVERSE to variation: higher R^2 = lower variation.
        # Bin q1 = lowest R^2 = highest variation; bin q5 = highest R^2 = lowest.
        'high_variation',
        'high_mid_variation',
        'mid_variation',
        'low_mid_variation',
        'low_variation',
    ]},
}

OUTPUT_CHART_DIR = 'chart/buckets'
OUTPUT_REPORT_DIR = 'reports/findings/buckets'


def _load_5s(day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _load_15m_ohlcv(day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/{CRM_TF}/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _compute_day_crm_features(day: str) -> dict | None:
    """Compute 15m CRM characterization vectors at every 5s bar of the day.

    Returns {'ts': int64[], 'M': float[], 'S': float[], 'slope': float[],
             'curv': float[], 'z_close': float[], 'sigma_rank': float[]}
    or None if data missing.
    """
    df_5s = _load_5s(day)
    if df_5s.empty:
        return None
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values.astype(np.float64)

    crm_oh = _load_15m_ohlcv(day)
    if crm_oh.empty:
        return None
    crm_oh['M_close'] = crm_oh['close'].rolling(CRM_WINDOW, min_periods=2).mean()
    crm_oh['S_close'] = crm_oh['close'].rolling(CRM_WINDOW, min_periods=2).std()
    tf_ts = crm_oh['timestamp'].values.astype(np.int64)
    target = ts_5s - CRM_PERIOD_S
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)
    M = crm_oh['M_close'].values[idx]
    S = crm_oh['S_close'].values[idx]

    n = len(M)
    slope = np.full(n, np.nan)
    if n > SLOPE_LOOKBACK_5S_BARS:
        slope[SLOPE_LOOKBACK_5S_BARS:] = (
            (M[SLOPE_LOOKBACK_5S_BARS:] - M[:-SLOPE_LOOKBACK_5S_BARS])
            / SLOPE_LOOKBACK_5S_BARS)
    curv = np.full(n, np.nan)
    if n > SLOPE_LOOKBACK_5S_BARS:
        curv[SLOPE_LOOKBACK_5S_BARS:] = (
            (slope[SLOPE_LOOKBACK_5S_BARS:] - slope[:-SLOPE_LOOKBACK_5S_BARS])
            / SLOPE_LOOKBACK_5S_BARS)

    with np.errstate(divide='ignore', invalid='ignore'):
        z_close = (close_5s - M) / S
    sigma_rank = (pd.Series(S)
                    .rolling(RANK_WINDOW_5S_BARS, min_periods=20)
                    .rank(pct=True)
                    .values)
    r2adj = _rolling_r2_adjusted(close_5s, R2_WINDOW_5S_BARS)

    return {
        'ts':         ts_5s,
        'M':          M,
        'S':          S,
        'slope':      slope,
        'curv':       curv,
        'z_close':    z_close,
        'sigma_rank': sigma_rank,
        'r2adj_5m':   r2adj,
    }


def _rolling_r2_adjusted(y: np.ndarray, n: int) -> np.ndarray:
    """Rolling R^2 adjusted for simple linear regression of y vs index x=0..n-1.

    For each bar i (i >= n-1) fit y[i-n+1:i+1] vs x = 0..n-1, return
        R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - 2)
    where R^2 = 1 - SS_res / SS_tot.

    Vectorized using rolling means; cost O(N) per day. No lookahead - the
    R^2_adj at bar i uses only y[i-n+1 .. i].
    """
    N = len(y)
    out = np.full(N, np.nan)
    if N < n:
        return out
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    Sxx = float(((x - x_mean) ** 2).sum())
    if Sxx == 0:
        return out
    # Rolling sums via cumulative
    cs1  = np.concatenate([[0.0], np.cumsum(y)])
    cs2  = np.concatenate([[0.0], np.cumsum(y * y)])
    # We also need Sxy = sum((x - x_mean) * y) per window. Since x is fixed
    # (0..n-1), expand: Sxy = sum(x*y) - n*x_mean*y_mean
    #                       = sum(x*y) - x_mean * sum(y).
    # sum(x*y) needs a custom rolling. Do it once via convolution.
    sum_xy_window = np.convolve(y, x[::-1], mode='valid')  # length N - n + 1
    # Align: sum_xy_window[k] corresponds to bar i = k + n - 1
    for k, i in enumerate(range(n - 1, N)):
        sum_y  = cs1[i + 1] - cs1[i + 1 - n]
        sum_y2 = cs2[i + 1] - cs2[i + 1 - n]
        y_mean = sum_y / n
        SS_tot = sum_y2 - n * y_mean * y_mean
        if SS_tot <= 0:
            out[i] = np.nan
            continue
        Sxy = sum_xy_window[k] - x_mean * sum_y
        b = Sxy / Sxx
        a = y_mean - b * x_mean
        # SS_res = sum((y - (a + b*x))^2)
        # = SS_tot - b * Sxy
        SS_res = SS_tot - b * Sxy
        if SS_res < 0:
            SS_res = 0.0
        r2 = 1.0 - SS_res / SS_tot
        # Adjusted R^2 for simple regression (1 predictor)
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
        out[i] = r2_adj
    return out


def _feature_at_event(day_feats: dict, event_start_ts: int) -> dict | None:
    """Index into the day's feature arrays at the event start timestamp."""
    if day_feats is None:
        return None
    ts = day_feats['ts']
    if event_start_ts < ts[0] or event_start_ts > ts[-1]:
        return None
    i = int(np.searchsorted(ts, event_start_ts, side='right') - 1)
    i = max(0, min(i, len(ts) - 1))
    out = {
        'slope':      float(day_feats['slope'][i])      if np.isfinite(day_feats['slope'][i])      else np.nan,
        'curv':       float(day_feats['curv'][i])       if np.isfinite(day_feats['curv'][i])       else np.nan,
        'z_close':    float(day_feats['z_close'][i])    if np.isfinite(day_feats['z_close'][i])    else np.nan,
        'sigma_rank': float(day_feats['sigma_rank'][i]) if np.isfinite(day_feats['sigma_rank'][i]) else np.nan,
        'r2adj_5m':   float(day_feats['r2adj_5m'][i])   if np.isfinite(day_feats['r2adj_5m'][i])   else np.nan,
    }
    return out


def _bucket_quantile(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return integer bucket index 0..n_bins-1 for each value (NaN -> -1)."""
    out = np.full(len(values), -1, dtype=int)
    finite = np.isfinite(values)
    if finite.sum() < n_bins * 5:
        return out
    edges = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1))
    edges[0]  = -np.inf
    edges[-1] = np.inf
    out[finite] = np.clip(np.searchsorted(edges, values[finite], side='right') - 1,
                          0, n_bins - 1)
    return out


def _build_event_table(macro_path: str, split_filter: str = 'IS') -> pd.DataFrame:
    """Load macro events and attach 15m CRM features at each event entry."""
    print(f'Loading macro events from {macro_path}...')
    me = pd.read_csv(macro_path)
    if split_filter:
        me = me[me['split'] == split_filter].reset_index(drop=True)
    print(f'  {len(me)} {split_filter} events')

    days_in_events = sorted(me['day'].unique())
    print(f'  {len(days_in_events)} unique days')

    feats_per_day = {}
    for d in tqdm(days_in_events, desc='compute 15m CRM per day'):
        f = _compute_day_crm_features(d)
        if f is not None:
            feats_per_day[d] = f

    # Attach features
    rows = []
    for _, e in me.iterrows():
        f = _feature_at_event(feats_per_day.get(e['day']), int(e['start_ts']))
        if f is None:
            continue
        rec = e.to_dict()
        rec.update({
            'crm_slope':      f['slope'],
            'crm_curv':       f['curv'],
            'crm_z_close':    f['z_close'],
            'crm_sigma_rank': f['sigma_rank'],
            'crm_r2adj_5m':   f['r2adj_5m'],
        })
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f'  {len(df)} events with full 15m CRM features')
    return df


def _assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['bin_slope']      = _bucket_quantile(df['crm_slope'].values, AXIS_BIN_DEFS['slope']['n_bins'])
    df['bin_curvature']  = _bucket_quantile(df['crm_curv'].values, AXIS_BIN_DEFS['curvature']['n_bins'])
    df['bin_z_close']    = _bucket_quantile(df['crm_z_close'].values, AXIS_BIN_DEFS['z_close']['n_bins'])
    df['bin_sigma_rank'] = _bucket_quantile(df['crm_sigma_rank'].values,
                                            AXIS_BIN_DEFS['sigma_rank']['n_bins'])
    df['bin_r2adj_5m']   = _bucket_quantile(df['crm_r2adj_5m'].values,
                                            AXIS_BIN_DEFS['r2adj_5m']['n_bins'])
    return df


def _select_representative(df: pd.DataFrame, axis: str, bucket_idx: int,
                          mode: str) -> pd.Series | None:
    bin_col = f'bin_{axis}'
    sub = df[df[bin_col] == bucket_idx]
    if sub.empty:
        return None
    if mode == 'longest':
        return sub.sort_values('duration_min', ascending=False).iloc[0]
    if mode == 'median':
        # Closest to bucket median in the corresponding feature
        feat_col_map = {'slope': 'crm_slope', 'curvature': 'crm_curv',
                        'z_close': 'crm_z_close', 'sigma_rank': 'crm_sigma_rank',
                        'r2adj_5m': 'crm_r2adj_5m'}
        col = feat_col_map[axis]
        med = sub[col].median()
        sub = sub.copy()
        sub['_dist'] = (sub[col] - med).abs()
        return sub.sort_values('_dist').iloc[0]
    if mode == 'random':
        return sub.sample(1, random_state=42).iloc[0]
    raise ValueError(f'unknown selection mode: {mode}')


def _draw_event_panel(ax, event: pd.Series, window_hours: float):
    day = event['day']
    df_5s = _load_5s(day)
    if df_5s.empty:
        ax.text(0.5, 0.5, f'no data: {day}', transform=ax.transAxes,
                ha='center', va='center')
        return
    ts = df_5s['timestamp'].values.astype(np.int64)
    close = df_5s['close'].values

    # Window
    half = int(window_hours * 3600)
    t_start = int(event['start_ts']) - half
    t_end   = int(event['end_ts'])   + half
    m = (ts >= t_start) & (ts <= t_end)
    if not m.any():
        ax.text(0.5, 0.5, f'window empty: {day}', transform=ax.transAxes,
                ha='center', va='center')
        return
    ts_w = ts[m]
    cl_w = close[m]
    dt_w = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_w]

    ax.plot(dt_w, cl_w, color='black', lw=0.6, alpha=0.85, label='5s close')

    # 15m CRM overlay (M_close + sigma bands)
    feats = _compute_day_crm_features(day)
    if feats is not None:
        idx_w = (feats['ts'] >= t_start) & (feats['ts'] <= t_end)
        ts_f = feats['ts'][idx_w]
        dt_f = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_f]
        M = feats['M'][idx_w]
        S = feats['S'][idx_w]
        ax.plot(dt_f, M, color='royalblue', lw=1.4, label='15m M_close')
        ax.fill_between(dt_f, M - 2 * S, M + 2 * S, color='royalblue', alpha=0.10,
                        label='15m +/-2sigma')
        ax.plot(dt_f, M - 3 * S, color='royalblue', lw=0.6,
                linestyle='--', alpha=0.6)
        ax.plot(dt_f, M + 3 * S, color='royalblue', lw=0.6,
                linestyle='--', alpha=0.6)

    # Mark event window
    s_dt = datetime.fromtimestamp(int(event['start_ts']), tz=timezone.utc)
    e_dt = datetime.fromtimestamp(int(event['end_ts']),   tz=timezone.utc)
    color = '#D32F2F' if event['side'] == 'below' else '#388E3C'
    ax.axvspan(s_dt, e_dt, color=color, alpha=0.18)
    ax.axvline(s_dt, color=color, lw=1.4, alpha=0.85)
    ax.axvline(e_dt, color=color, lw=1.0, alpha=0.6, linestyle=':')

    title = (f'{day}  start={s_dt.strftime("%H:%M:%S")}  '
             f'dur={event["duration_min"]:.1f}m  '
             f'max_|z|={event["max_abs_z"]:.2f}  '
             f'side={event["side"]}/{event["anchor"]}\n'
             f'slope={event["crm_slope"]:+.4f}  '
             f'curv={event["crm_curv"]:+.6f}  '
             f'z_close={event["crm_z_close"]:+.2f}  '
             f'sigma_rank={event["crm_sigma_rank"]:.2f}  '
             f'r2adj={event["crm_r2adj_5m"]:+.3f}')
    ax.set_title(title, fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


def _render_axis_chart(df: pd.DataFrame, axis: str, mode: str,
                       window_hours: float) -> str:
    cells = AXIS_BIN_DEFS[axis]['labels']
    n = len(cells)
    cols = 3 if n <= 6 else min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4.5),
                             squeeze=False)

    for i, label in enumerate(cells):
        ax = axes[i // cols][i % cols]
        n_in_bucket = int((df[f'bin_{axis}'] == i).sum())
        rep = _select_representative(df, axis, i, mode)
        if rep is None:
            ax.text(0.5, 0.5, f'{label}\n(empty bucket)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        _draw_event_panel(ax, rep, window_hours)
        ax.set_title(f'{label}  (n={n_in_bucket})\n' + ax.get_title(), fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f'EVENT-BUCKETED by 15m CRM axis: {axis} '
                 f'({mode} selection, +/-{window_hours}hr window, '
                 f'n_total={len(df)} events)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUTPUT_CHART_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_CHART_DIR, f'event_15m_crm_{axis}.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out


def _write_manifest_md(df: pd.DataFrame, args) -> str:
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_REPORT_DIR, 'event_manifest_15m_crm.md')
    lines = [
        f'# Event-bucketed manifest (15m CRM characterization)',
        f'',
        f'_Generated {datetime.now().isoformat()}_',
        f'',
        f'Selection mode:  `{args.selection}`   '
        f'window: +/-{args.window_hours} hr   '
        f'split filter: `{args.split}`',
        f'',
        f'## Population',
        f'',
        f'- {len(df)} {args.split} macro events from '
        f'`reports/findings/band_touch_aggregation/macro_events_1h_hl.csv`',
        f'- 15m CRM features attached at event ENTRY bar (no lookahead)',
        f'',
        f'## Why event-segmented + 15m CRM (not regime_2d)',
        f'',
        f'`regime_labels_2d.csv` is a DAY-AGGREGATE label computed from the',
        f'same metrics that determine whether macro events appear at all,',
        f'so picking representative days from those labels is circular',
        f'(UP_SMOOTH days have 0 macro events because the SMOOTH label is',
        f'defined by the absence of those events). The unbiased substrate',
        f'is the EVENT itself, characterized by bar-level features measured',
        f'at the event entry.',
        f'',
        f'## Per-axis bucket populations',
        f'',
    ]
    for axis, info in AXIS_BIN_DEFS.items():
        n = info['n_bins']
        lines.append(f'### {axis} ({n} bins)')
        lines.append('')
        lines.append('| bin | label | n_events | feature_q_lo | feature_q_hi |')
        lines.append('|----:|-------|---------:|-------------:|-------------:|')
        bin_col = f'bin_{axis}'
        feat_col_map = {'slope': 'crm_slope', 'curvature': 'crm_curv',
                        'z_close': 'crm_z_close', 'sigma_rank': 'crm_sigma_rank',
                        'r2adj_5m': 'crm_r2adj_5m'}
        col = feat_col_map[axis]
        for i, lbl in enumerate(info['labels']):
            sub = df[df[bin_col] == i]
            n_b = len(sub)
            if n_b > 0:
                qlo = sub[col].min()
                qhi = sub[col].max()
            else:
                qlo = qhi = float('nan')
            lines.append(f'| {i} | {lbl} | {n_b} | {qlo:+.5f} | {qhi:+.5f} |')
        lines.append('')

    lines.extend([
        f'## Notes',
        f'',
        f'- LONGEST-event-in-bucket selection (default) emphasizes the most',
        f'  clinically-impactful sample per bucket.',
        f'- Charts at `chart/buckets/event_15m_crm_<axis>.png`.',
        f'- The 15m CRM is the chart-validated strategic gate from 2026-05-09.',
        f'  Slope sign + curvature + z_close + sigma_rank quantile fully',
        f'  characterize the event\'s 15m-scale context with no lookahead.',
        f'',
    ])
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro-csv',
                    default='reports/findings/band_touch_aggregation/macro_events_1h_hl.csv')
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS'])
    ap.add_argument('--axis', default='all',
                    choices=['all'] + list(AXIS_BIN_DEFS.keys()))
    ap.add_argument('--selection', default='longest',
                    choices=['longest', 'median', 'random'])
    ap.add_argument('--window-hours', type=float, default=2.0)
    args = ap.parse_args()

    df = _build_event_table(args.macro_csv, args.split)
    df = _assign_buckets(df)
    md = _write_manifest_md(df, args)
    print(f'\nManifest -> {md}')
    axes_to_run = list(AXIS_BIN_DEFS.keys()) if args.axis == 'all' else [args.axis]
    for axis in axes_to_run:
        out = _render_axis_chart(df, axis, args.selection, args.window_hours)
        print(f'Chart    -> {out}')


if __name__ == '__main__':
    main()
