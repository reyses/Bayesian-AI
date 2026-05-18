"""Enhanced 3-anchor chart with:
    1. Three-level anchors (5s close + 15m M_close + 1h M_high/M_low)
    2. 5m CRM overlay color-coded by 3-state action (continue/flatline/reverse)
    3. Event markers labeled with primitive cell + Bayesian table prediction

The cell prediction reads from the V0 trade-location Bayesian table
(duration_per_axis.csv, magnitude_per_axis.csv) computed earlier this morning.

USAGE
    python tools/chart_anchors_3level_full.py --day 2025_10_29
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s, _load_tf_ohlcv


def compute_anchor(tf: str, day: str, ts_5s: np.ndarray, window: int,
                    column: str = 'close'):
    period_s = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}[tf]
    oh = _load_tf_ohlcv(tf, day)
    if oh.empty:
        return None
    M = oh[column].rolling(window, min_periods=2).mean().values
    S = oh[column].rolling(window, min_periods=2).std().values
    tf_ts = oh['timestamp'].values.astype(np.int64)
    target = ts_5s - period_s
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)
    return M[idx], S[idx]


def classify_3state(crm: np.ndarray, slope_lookback: int,
                     flat_threshold_quantile: float = 0.33) -> np.ndarray:
    """Classify each bar as 'CONTINUE', 'FLATLINE', or 'REVERSE'.

    slope_lookback: number of bars over which to compute the slope
    flat_threshold_quantile: |slope| below this quantile of all |slopes|
                              -> FLATLINE
    """
    n = len(crm)
    slope = np.full(n, np.nan)
    if n > slope_lookback:
        slope[slope_lookback:] = crm[slope_lookback:] - crm[:-slope_lookback]
    abs_s = np.abs(slope)
    finite = np.isfinite(abs_s)
    if not finite.any():
        return np.array(['NA'] * n)
    flat_thr = float(np.quantile(abs_s[finite], flat_threshold_quantile))
    state = np.array(['NA'] * n, dtype=object)
    prev_sign = np.sign(slope)
    for i in range(slope_lookback + 1, n):
        if not np.isfinite(slope[i]):
            continue
        if abs_s[i] < flat_thr:
            state[i] = 'FLATLINE'
        elif (np.isfinite(prev_sign[i-1]) and prev_sign[i-1] != 0
              and np.sign(slope[i]) != 0
              and np.sign(slope[i]) != prev_sign[i-1]):
            state[i] = 'REVERSE'
        else:
            state[i] = 'CONTINUE'
    return state


def detect_event_entries(z_high: np.ndarray, z_low: np.ndarray,
                          ts: np.ndarray, k: float = 3.0,
                          min_dwell_s: int = 60) -> list[dict]:
    """Detect band-touch event ENTRIES with min duration filter."""
    finite = np.isfinite(z_high) & np.isfinite(z_low)
    above = (z_high >= k) & finite
    below = (z_low <= -k) & finite
    events = []
    for label, mask in [('above_high', above), ('below_low', below)]:
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                duration_s = ts[j-1] - ts[i] if j > i else 0
                if duration_s >= min_dwell_s:
                    events.append({
                        'side_anchor': label,
                        'start_idx': i,
                        'end_idx': j - 1,
                        'duration_s': int(duration_s),
                        'duration_min': round(duration_s / 60, 2),
                    })
                i = j
            else:
                i += 1
    events.sort(key=lambda e: e['start_idx'])
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_10_29')
    ap.add_argument('--out-dir', default='chart/bayes_framework')
    ap.add_argument('--start-hour', type=int, default=None,
                     help='UTC hour to start the chart from (e.g. 18)')
    ap.add_argument('--end-hour', type=int, default=None,
                     help='UTC hour to end the chart at (e.g. 24)')
    ap.add_argument('--substrate-parquet',
                     default='reports/findings/segments/bayes_table_v0_location/event_substrate.parquet')
    ap.add_argument('--duration-csv',
                     default='reports/findings/segments/bayes_table_v0_location/duration_per_axis.csv')
    ap.add_argument('--magnitude-csv',
                     default='reports/findings/segments/bayes_table_v0_location/magnitude_per_axis.csv')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_5s = _load_5s(args.day)
    if df_5s.empty:
        print(f'No 5s data for {args.day}'); return
    ts = df_5s['timestamp'].values.astype(np.int64)
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    close = df_5s['close'].values.astype(np.float64)

    M_15m, _    = compute_anchor('15m', args.day, ts, 12, 'close')
    Mh_1h, Sh_1h = compute_anchor('1h', args.day, ts, 12, 'high')
    Ml_1h, Sl_1h = compute_anchor('1h', args.day, ts, 12, 'low')

    # 5m CRM (FAST oscillator)
    M_5m, _ = compute_anchor('5m', args.day, ts, 9, 'close')
    # 1m CRM (FASTER oscillator)
    M_1m, _ = compute_anchor('1m', args.day, ts, 15, 'close')

    # 3-state classification of 5m CRM (slope lookback = 60 5s bars = 5min)
    state_5m = classify_3state(M_5m, slope_lookback=60, flat_threshold_quantile=0.33)
    state_counts_5m = pd.Series(state_5m).value_counts()
    print(f'5m CRM 3-state distribution on {args.day}:')
    print(state_counts_5m.to_string())

    # 3-state classification of 1m CRM (slope lookback = 12 5s bars = 1min)
    state_1m = classify_3state(M_1m, slope_lookback=12, flat_threshold_quantile=0.33)
    state_counts_1m = pd.Series(state_1m).value_counts()
    print(f'\n1m CRM 3-state distribution on {args.day}:')
    print(state_counts_1m.to_string())

    # z computations
    z_high = (close - Mh_1h) / Sh_1h
    z_low  = (close - Ml_1h) / Sl_1h

    # Detect events
    events = detect_event_entries(z_high, z_low, ts, k=3.0, min_dwell_s=60)
    print(f'\nDetected {len(events)} band-touch events on {args.day}')

    # Load substrate-attached events on this day for cell info
    sub = pd.read_parquet(args.substrate_parquet)
    day_events = sub[sub['day'] == args.day].copy()

    # Quantile edges from the IS substrate (for bucketization)
    is_pop = sub[sub['split'] == 'IS']
    quant_edges = {}
    for col, n_bins in [('slope', 5), ('curv', 3), ('z_close_at_entry', 5),
                         ('sigma_rank', 5), ('r2adj_5m', 5)]:
        v = is_pop[col].dropna().values
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
        quant_edges[col] = edges

    def bucket(value, col):
        if not np.isfinite(value):
            return -1
        edges = quant_edges[col]
        return int(np.clip(np.searchsorted(edges, value, side='right') - 1,
                            0, len(edges) - 2))

    # Load duration + magnitude tables
    dur_df = pd.read_csv(args.duration_csv)
    mag_df = pd.read_csv(args.magnitude_csv)

    # Build event-to-cell-info lookup
    print(f'\nEvent-by-event cell info + table prediction:')
    print(f'{ "time":<8s}  {"side_anchor":<11s}  {"slope":>6s} {"sigq":>4s} '
           f'{"zcq":>3s} {"r2q":>3s}  {"med_d":>5s}  {"P>=10m":>6s}  '
           f'{"q90_excess":>10s}  {"max_z":>6s}')
    rows_out = []
    for _, e in day_events.iterrows():
        slope_q  = bucket(e['slope'], 'slope')
        sigma_q  = bucket(e['sigma_rank'], 'sigma_rank')
        zclose_q = bucket(e['z_close_at_entry'], 'z_close_at_entry')
        r2_q     = bucket(e['r2adj_5m'], 'r2adj_5m')
        side, anchor = e['side'], e['anchor']
        # Look up duration table by sigma_rank cell as canonical
        dur_cell = dur_df[(dur_df['split']=='IS') &
                           (dur_df['side']==side) & (dur_df['anchor']==anchor) &
                           (dur_df['axis']=='sigma_rank_q') & (dur_df['bin']==sigma_q) &
                           (dur_df['threshold_min']==10)]
        mag_cell = mag_df[(mag_df['split']=='IS') &
                           (mag_df['side']==side) & (mag_df['anchor']==anchor) &
                           (mag_df['axis']=='sigma_rank_q') & (mag_df['bin']==sigma_q)]
        med_d   = float(dur_cell['med_duration'].iloc[0])  if len(dur_cell)>0 else np.nan
        p_cont  = float(dur_cell['p_continue'].iloc[0])     if len(dur_cell)>0 else np.nan
        q90     = float(mag_cell['excess_q90'].iloc[0])     if len(mag_cell)>0 else np.nan
        ts_str = datetime.fromtimestamp(int(e['start_ts']), tz=timezone.utc).strftime('%H:%M:%S')
        print(f'  {ts_str:<8s}  {side+"_"+anchor:<11s}  '
               f'{slope_q:>6d} {sigma_q:>4d} {zclose_q:>3d} {r2_q:>3d}  '
               f'{med_d:>5.1f}  {p_cont:>6.3f}      {q90:>5.2f}      {e["signed_max_z"]:>6.2f}')
        rows_out.append({
            'time': ts_str, 'side': side, 'anchor': anchor,
            'slope_q': slope_q, 'sigma_q': sigma_q, 'zclose_q': zclose_q, 'r2_q': r2_q,
            'med_duration_min': med_d, 'p_continue_10m': p_cont,
            'magnitude_q90_excess': q90, 'actual_max_z': e['signed_max_z'],
            'actual_duration_min': e['duration_min'],
        })
    pd.DataFrame(rows_out).to_csv(
        os.path.join(args.out_dir, f'cells_{args.day}.csv'), index=False)

    # ========== RENDER ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 13),
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)

    # ----- Top: 3-anchor + 5m CRM color-coded -----
    ax1.plot(dt, Mh_1h, color='#43A047', lw=1.4, alpha=0.85, label='1h M_high')
    ax1.fill_between(dt, Mh_1h-2*Sh_1h, Mh_1h+2*Sh_1h, color='#43A047', alpha=0.05)
    ax1.fill_between(dt, Mh_1h+2*Sh_1h, Mh_1h+3*Sh_1h, color='#43A047', alpha=0.13,
                       label='M_high +2sigma to +3sigma trigger zone')
    ax1.plot(dt, Mh_1h+3*Sh_1h, color='#1B5E20', lw=0.8, ls='--', alpha=0.7)

    ax1.plot(dt, Ml_1h, color='#E53935', lw=1.4, alpha=0.85, label='1h M_low')
    ax1.fill_between(dt, Ml_1h-2*Sl_1h, Ml_1h+2*Sl_1h, color='#E53935', alpha=0.05)
    ax1.fill_between(dt, Ml_1h-3*Sl_1h, Ml_1h-2*Sl_1h, color='#E53935', alpha=0.13,
                       label='M_low -2sigma to -3sigma trigger zone')
    ax1.plot(dt, Ml_1h-3*Sl_1h, color='#B71C1C', lw=0.8, ls='--', alpha=0.7)

    ax1.plot(dt, M_15m, color='#1E88E5', lw=1.6, alpha=0.95,
              label='15m M_close (medium context)')

    # 5m CRM color-coded by 3-state (THICK)
    state_colors_5m = {'CONTINUE': '#2E7D32', 'FLATLINE': '#9E9E9E',
                        'REVERSE': '#E65100', 'NA': '#CCCCCC'}
    # 1m CRM color-coded by 3-state (THIN, lighter shades)
    state_colors_1m = {'CONTINUE': '#80DEEA', 'FLATLINE': '#E0E0E0',
                        'REVERSE': '#FF5722', 'NA': '#EEEEEE'}

    def plot_crm_segments(ax, M, state_arr, colors, lw, alpha):
        for st in ['CONTINUE', 'FLATLINE', 'REVERSE']:
            idx = np.where(state_arr == st)[0]
            if len(idx) == 0: continue
            diffs = np.diff(idx)
            breaks = np.where(diffs > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, len(idx) - 1]
            for s_, e_ in zip(starts, ends):
                seg = idx[s_:e_+1]
                if len(seg) < 2: continue
                ax.plot([dt[i] for i in seg], M[seg],
                          color=colors[st], lw=lw, alpha=alpha)

    # Draw 1m CRM first (thin), then 5m CRM on top (thick)
    plot_crm_segments(ax1, M_1m, state_1m, state_colors_1m, lw=1.0, alpha=0.7)
    plot_crm_segments(ax1, M_5m, state_5m, state_colors_5m, lw=2.2, alpha=0.85)

    # Legend entries
    for st in ['CONTINUE', 'FLATLINE', 'REVERSE']:
        n_st = int((state_5m == st).sum())
        ax1.plot([], [], color=state_colors_5m[st], lw=2.5,
                   label=f'5m CRM: {st} ({n_st:,}, {100*n_st/len(state_5m):.0f}%)')
    for st in ['CONTINUE', 'FLATLINE', 'REVERSE']:
        n_st = int((state_1m == st).sum())
        ax1.plot([], [], color=state_colors_1m[st], lw=1.4,
                   label=f'1m CRM: {st} ({n_st:,}, {100*n_st/len(state_1m):.0f}%)')

    # 5s close (thin, on top so visible)
    ax1.plot(dt, close, color='black', lw=0.4, alpha=0.7, label='5s close')

    # Mark events
    for _, e in day_events.iterrows():
        s_dt = datetime.fromtimestamp(int(e['start_ts']), tz=timezone.utc)
        e_dt = datetime.fromtimestamp(int(e['end_ts']), tz=timezone.utc)
        col = '#43A047' if e['side']=='above' else '#E53935'
        ax1.axvspan(s_dt, e_dt, color=col, alpha=0.06, zorder=0)
        # annotation: cell + table prediction
        slope_q  = bucket(e['slope'], 'slope')
        sigma_q  = bucket(e['sigma_rank'], 'sigma_rank')
        zclose_q = bucket(e['z_close_at_entry'], 'z_close_at_entry')
        r2_q     = bucket(e['r2adj_5m'], 'r2adj_5m')
        dur_cell = dur_df[(dur_df['split']=='IS') &
                           (dur_df['side']==e['side']) & (dur_df['anchor']==e['anchor']) &
                           (dur_df['axis']=='sigma_rank_q') & (dur_df['bin']==sigma_q) &
                           (dur_df['threshold_min']==10)]
        p_cont = float(dur_cell['p_continue'].iloc[0]) if len(dur_cell)>0 else np.nan
        med_d  = float(dur_cell['med_duration'].iloc[0]) if len(dur_cell)>0 else np.nan
        ymax = ax1.get_ylim()[1]
        ax1.text(s_dt, ymax, f'  {e["side"][:3]}.{e["anchor"][:1]}\n'
                                f'  slo={slope_q} sig={sigma_q}\n'
                                f'  zc={zclose_q} r2={r2_q}\n'
                                f'  P>=10={p_cont:.2f}\n'
                                f'  med={med_d:.1f}m',
                  fontsize=6, color='#0D47A1', va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.85, edgecolor=col, linewidth=0.6))

    ax1.set_title(f'{args.day} — 3-LEVEL ANCHOR FRAMEWORK with 5m CRM 3-STATE coloring + EVENT CELLS\n'
                   f'green=CONTINUE  gray=FLATLINE  orange=REVERSE  '
                   f'(only REVERSE is adverse to a position)',
                   fontsize=12)
    ax1.set_ylabel('price'); ax1.legend(loc='best', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.20)

    # ----- Bottom: 5m + 1m CRM slopes color-coded by 3-state -----
    slope_5m = np.full(len(M_5m), np.nan)
    slope_5m[60:] = M_5m[60:] - M_5m[:-60]
    slope_1m = np.full(len(M_1m), np.nan)
    slope_1m[12:] = M_1m[12:] - M_1m[:-12]

    for st in ['CONTINUE', 'FLATLINE', 'REVERSE']:
        idx = np.where(state_1m == st)[0]
        if len(idx) == 0: continue
        ax2.scatter([dt[i] for i in idx], slope_1m[idx],
                     color=state_colors_1m[st], s=1.5, alpha=0.45,
                     label=f'1m {st}')
    for st in ['CONTINUE', 'FLATLINE', 'REVERSE']:
        idx = np.where(state_5m == st)[0]
        if len(idx) == 0: continue
        ax2.scatter([dt[i] for i in idx], slope_5m[idx],
                     color=state_colors_5m[st], s=4, alpha=0.7,
                     label=f'5m {st}')
    ax2.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax2.set_title('5m CRM slope (thick dots) + 1m CRM slope (thin dots) color-coded by 3-state',
                    fontsize=10)
    ax2.set_xlabel('time (UTC)'); ax2.set_ylabel('CRM slope (pts/lookback)')
    ax2.grid(True, alpha=0.20); ax2.legend(loc='best', fontsize=7, ncol=2)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    if args.start_hour is not None or args.end_hour is not None:
        sh = args.start_hour if args.start_hour is not None else 0
        eh = args.end_hour if args.end_hour is not None else 24
        from datetime import time
        date_part = dt[0].date()
        x_start = datetime.combine(date_part, time(sh, 0), tzinfo=timezone.utc)
        x_end = datetime.combine(date_part, time(min(eh, 23), 59), tzinfo=timezone.utc) \
                 if eh < 24 else dt[-1]
        ax1.set_xlim(x_start, x_end)
        ax2.set_xlim(x_start, x_end)
        suffix = f'_h{sh:02d}-h{eh:02d}'
    else:
        suffix = ''

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f'anchors_3level_full_{args.day}{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFull chart -> {out_path}')


if __name__ == '__main__':
    main()
