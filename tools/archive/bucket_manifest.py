"""Bucket manifest - define and visualize the conditioning axes that will
become rows/cols of the probability table.

Two-step plan:
    Step 1 (this tool, day-level): regime_2d + dow buckets, populations,
            representative-day picks, multi-panel charts per axis.
    Step 2 (next iter, bar-level):  tod / hurst / sn / volvel / z_se buckets,
            interval-extraction + context-window charts.

For each day-level axis this writes:
    reports/findings/buckets/<axis>_manifest.md   text breakdown
    chart/buckets/<axis>_representative.png       multi-panel grid

USAGE:
    python tools/bucket_manifest.py --axis regime
    python tools/bucket_manifest.py --axis dow
    python tools/bucket_manifest.py --axis all
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


REGIME_CELLS = ['UP_SMOOTH', 'UP_CHOPPY', 'DOWN_SMOOTH',
                'DOWN_CHOPPY', 'FLAT_SMOOTH', 'FLAT_CHOPPY']
DOW_CELLS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
DOW_FULL = {'Mon': 'Monday', 'Tue': 'Tuesday', 'Wed': 'Wednesday',
            'Thu': 'Thursday', 'Fri': 'Friday', 'Sat': 'Saturday', 'Sun': 'Sunday'}

TF_WINDOW = {'1m': 15, '5m': 9, '15m': 12, '1h': 12}
PERIOD_S  = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}


def _load_regime_labels() -> pd.DataFrame:
    df = pd.read_csv('DATA/ATLAS/regime_labels_2d.csv')
    df['day_us'] = df['date'].str.replace('-', '_')
    # Day-of-week from date
    df['dow'] = pd.to_datetime(df['date']).dt.strftime('%a')
    return df


def _representative_day(df: pd.DataFrame, cell: str, axis: str) -> str | None:
    """Pick the day whose centrality metric is closest to median within the bucket.

    For regime cells use efficiency_ratio (in [-1, +1]) — closest to its
    own bucket median is the most "typical" day of that regime.
    For dow use net_move (closer to bucket median = typical net day).
    """
    if axis == 'regime':
        sub = df[df['regime_2d'] == cell].copy()
        metric = 'efficiency_ratio'
    elif axis == 'dow':
        sub = df[df['dow'] == cell].copy()
        metric = 'efficiency_ratio'
    else:
        return None
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return None
    med = sub[metric].median()
    sub['dist'] = (sub[metric] - med).abs()
    sub = sub.sort_values('dist')
    return sub.iloc[0]['day_us']


def _load_5s(day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _load_tf_means(tf: str, day: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ts, M_close) at TF cadence."""
    path = f'DATA/ATLAS/{tf}/{day}.parquet'
    if not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    df = df.sort_values('timestamp').reset_index(drop=True)
    N = TF_WINDOW[tf]
    M = df['close'].rolling(N, min_periods=2).mean().values
    ts = df['timestamp'].values.astype(np.int64)
    return ts, M


def _ffill_to_5s(M: np.ndarray, src_ts: np.ndarray, target_ts: np.ndarray,
                 period_s: int) -> np.ndarray:
    target = target_ts - period_s
    idx = np.searchsorted(src_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(src_ts) - 1)
    return M[idx]


_MACRO_CACHE = None


def _load_macro_events() -> pd.DataFrame:
    global _MACRO_CACHE
    if _MACRO_CACHE is not None:
        return _MACRO_CACHE
    path = 'reports/findings/band_touch_aggregation/macro_events_1h_hl.csv'
    if not os.path.exists(path):
        _MACRO_CACHE = pd.DataFrame()
        return _MACRO_CACHE
    df = pd.read_csv(path)
    _MACRO_CACHE = df
    return df


def _draw_day_panel(ax, day: str, label: str, sub_metrics: dict):
    df_5s = _load_5s(day)
    if df_5s.empty:
        ax.text(0.5, 0.5, f'no data: {day}', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(f'{label}: {day}', fontsize=10)
        return
    ts = df_5s['timestamp'].values.astype(np.int64)
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    close = df_5s['close'].values

    # Plot 5s close
    ax.plot(dt, close, color='black', lw=0.5, alpha=0.8, label='5s close')

    # Overlay regression means at 1m, 5m, 15m, 1h
    overlays = [('1m', '#E53935'), ('5m', '#FB8C00'),
                ('15m', '#43A047'), ('1h', '#1E88E5')]
    for tf, color in overlays:
        src_ts, M = _load_tf_means(tf, day)
        if src_ts is None:
            continue
        M5s = _ffill_to_5s(M, src_ts, ts, PERIOD_S[tf])
        ax.plot(dt, M5s, color=color, lw=0.9, alpha=0.85, label=f'{tf}')

    # Overlay 1h HL +/-3sigma macro events with timestamps
    macro = _load_macro_events()
    if not macro.empty:
        day_ev = macro[macro['day'] == day].sort_values('start_ts')
        for _, r in day_ev.iterrows():
            sdt = datetime.fromtimestamp(int(r['start_ts']), tz=timezone.utc)
            edt = datetime.fromtimestamp(int(r['end_ts']), tz=timezone.utc)
            color = '#D32F2F' if r['side'] == 'below' else '#388E3C'
            ax.axvspan(sdt, edt, color=color, alpha=0.15)
            ax.axvline(sdt, color=color, lw=1.0, alpha=0.7)
            # Label at top of axis
            ymax = ax.get_ylim()[1]
            tag = f'{sdt.strftime("%H:%M")} {r["side"][0].upper()}'
            tag += f'/{int(round(r["duration_min"]))}m'
            tag += f'/z{r["max_abs_z"]:.1f}'
            ax.text(sdt, ymax, tag, fontsize=6.5,
                    color=color, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.7, edgecolor=color, linewidth=0.5))

    ax.set_title(
        f'{label}\n{day}  '
        f'eff={sub_metrics.get("eff", float("nan")):+.3f}  '
        f'net={sub_metrics.get("net", float("nan")):+.1f}  '
        f'range={sub_metrics.get("rng", float("nan")):.1f}'
        + (f'  macro={len(day_ev)}' if not macro.empty else ''),
        fontsize=10)
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))


def _populate_axis_manifest(df: pd.DataFrame, axis: str) -> tuple[list, dict]:
    """Returns (cells_list, {cell: {n_days, n_is, n_oos, rep_day, metrics}})."""
    if axis == 'regime':
        cells = REGIME_CELLS
        col = 'regime_2d'
    elif axis == 'dow':
        cells = DOW_CELLS
        col = 'dow'
    else:
        raise ValueError(f'unknown axis: {axis}')
    info = {}
    for cell in cells:
        sub = df[df[col] == cell]
        n_days = len(sub)
        n_is  = (sub['split'] == 'IS').sum()
        n_oos = (sub['split'] == 'OOS').sum()
        rep_day = _representative_day(df, cell, axis)
        rep_metrics = {}
        if rep_day:
            r = sub[sub['day_us'] == rep_day]
            if not r.empty:
                rep_metrics = {
                    'eff': float(r.iloc[0]['efficiency_ratio']),
                    'net': float(r.iloc[0]['net_move']),
                    'rng': float(r.iloc[0]['range']),
                }
        info[cell] = {
            'n_days': int(n_days),
            'n_is':   int(n_is),
            'n_oos':  int(n_oos),
            'rep_day': rep_day,
            'metrics': rep_metrics,
            'pct':    round(n_days / max(len(df), 1) * 100, 1),
        }
    return cells, info


def _write_manifest_md(axis: str, cells: list, info: dict, total_days: int) -> str:
    out_dir = 'reports/findings/buckets'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{axis}_manifest.md')
    lines = [
        f'# Bucket manifest - axis: `{axis}`',
        f'',
        f'_Generated {datetime.now().isoformat()}_',
        f'',
        f'Total days indexed: {total_days}',
        f'',
        f'## Cells',
        f'',
        f'| cell | n_days | n_IS | n_OOS | %total | rep_day | eff_ratio | net_move | range |',
        f'|------|-------:|-----:|------:|-------:|---------|----------:|---------:|------:|',
    ]
    for c in cells:
        i = info[c]
        m = i['metrics']
        lines.append(
            f'| {c} | {i["n_days"]} | {i["n_is"]} | {i["n_oos"]} | '
            f'{i["pct"]}% | {i["rep_day"] or "-"} | '
            f'{m.get("eff", float("nan")):+.3f} | '
            f'{m.get("net", float("nan")):+.1f} | '
            f'{m.get("rng", float("nan")):.1f} |')
    lines += [
        f'',
        f'## Notes',
        f'',
        f'- Representative day = day in cell whose `efficiency_ratio` is',
        f'  closest to the cell median.',
        f'- `eff_ratio`, `net_move`, `range` are the day-aggregate metrics from',
        f'  `DATA/ATLAS/regime_labels_2d.csv`.',
        f'- Cells with `n_days < ~10` are too thin for reliable conditioning;',
        f'  flag them and pool with the parent cell or the FLAT/CHOPPY default.',
        f'',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def _render_axis_chart(axis: str, cells: list, info: dict) -> str:
    chart_dir = 'chart/buckets'
    os.makedirs(chart_dir, exist_ok=True)
    n = len(cells)
    cols = 3 if n <= 6 else min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4),
                             squeeze=False)
    for i, cell in enumerate(cells):
        ax = axes[i // cols][i % cols]
        d = info[cell]
        if d['rep_day'] is None:
            ax.text(0.5, 0.5, f'{cell}\n(no days)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        label = (f'{cell}  '
                 f'(n={d["n_days"]} days, IS={d["n_is"]} OOS={d["n_oos"]})')
        _draw_day_panel(ax, d['rep_day'], label, d['metrics'])
    # Hide unused panels
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    plt.tight_layout()
    out = os.path.join(chart_dir, f'{axis}_representative.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', default='regime',
                    choices=['regime', 'dow', 'all'])
    args = ap.parse_args()

    df = _load_regime_labels()
    print(f'Loaded {len(df)} day labels '
          f'(IS={(df["split"]=="IS").sum()}, '
          f'VAL={(df["split"]=="VAL").sum()}, '
          f'OOS={(df["split"]=="OOS").sum()})')

    axes_to_run = ['regime', 'dow'] if args.axis == 'all' else [args.axis]
    for axis in axes_to_run:
        print(f'\n=== axis: {axis} ===')
        cells, info = _populate_axis_manifest(df, axis)
        for c in cells:
            i = info[c]
            print(f'  {c:<14} n={i["n_days"]:>3}  IS={i["n_is"]:>3}  '
                  f'OOS={i["n_oos"]:>3}  rep={i["rep_day"] or "-"}')
        md_path = _write_manifest_md(axis, cells, info, len(df))
        chart_path = _render_axis_chart(axis, cells, info)
        print(f'\n  manifest -> {md_path}')
        print(f'  chart    -> {chart_path}')


if __name__ == '__main__':
    main()
