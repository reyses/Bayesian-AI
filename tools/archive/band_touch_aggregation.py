"""Band-touch aggregation - population census before any probability work.

Goal
----
Before building a conditional probability table we must know HOW MANY
band-touch events exist at every (TF, anchor, side, k_sigma) combo.
Cell counts drive whether a probability estimate is statistically meaningful.

Z-DEFINITION (close-vs-anchor, the operationally relevant one)
--------------------------------------------------------------
For each TF and each anchor (M_close, M_high, M_low):
    z = (5s_close - M_anchor) / SE_anchor

The current 5s close is z standard deviations from the upper/lower/center
regression mean. This is the metric that captures macro crashes and rallies
where the 5s close pushes far past the slowly-tracking 1h M_low (or M_high).

NOTE: this is DIFFERENT from V2's L3_z_high which is bar_high-vs-M_high
(a smaller, bounded number). We compute z manually from TF OHLCV here.

Aggregates two complementary populations
----------------------------------------
1. ENTRIES into k*sigma bands at every (TF, anchor, side)
   - Every transition from "in-band" to "out-of-band" = one entry
   - Counted per (tf x anchor x side x k_sigma)
   - Aggregated total + per-day distribution

2. MACRO EVENTS at 1h HL k_sigma threshold
   - Contiguous runs where price stays past the threshold
   - The 2-3 hour crash/rally days like 2026_02_12 / 2026_03_03
   - Full metadata: start/end/duration/max_sigma/side/day/tod/dow

Output
------
reports/findings/band_touch_aggregation/
    summary_by_config.csv      (tf, anchor, side, k) -> N, days_active, ...
    per_day_counts.csv         day x cell wide table
    macro_events_1h_hl.csv     every 1h HL run >= min_duration_s
    summary.md                 human-readable headline report

USAGE
    python tools/band_touch_aggregation.py --include-oos
    python tools/band_touch_aggregation.py --macro-k 2.5 --include-oos
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TFS = ['1m', '5m', '15m', '1h']
# Same regression-mean window the existing pipeline uses
TF_WINDOW = {'1m': 15, '5m': 9, '15m': 12, '1h': 12, '4h': 18}
PERIOD_S  = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}
ANCHORS = ['close', 'high', 'low']  # M_close, M_high, M_low
SIGMA_LEVELS = [1.0, 2.0, 3.0]


def _load_5s(day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _load_tf_ohlcv(tf: str, day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/{tf}/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _compute_z_close_vs_anchors(day: str) -> dict:
    """Compute z = (5s_close - M_anchor) / SE_anchor at every TF and anchor.

    Returns {(tf, anchor): {'ts': int64[], 'z': float64[]}} or {}
    if 5s OHLCV or any TF OHLCV missing.
    """
    df_5s = _load_5s(day)
    if df_5s.empty:
        return {}
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values.astype(np.float64)

    out = {}
    for tf in TFS:
        tf_oh = _load_tf_ohlcv(tf, day)
        if tf_oh.empty:
            return {}
        N = TF_WINDOW[tf]
        period_s = PERIOD_S[tf]
        # Rolling regression (mean) and SE (std) per anchor
        m_close = tf_oh['close'].rolling(N, min_periods=2).mean().values
        s_close = tf_oh['close'].rolling(N, min_periods=2).std().values
        m_high  = tf_oh['high'].rolling(N, min_periods=2).mean().values
        s_high  = tf_oh['high'].rolling(N, min_periods=2).std().values
        m_low   = tf_oh['low'].rolling(N, min_periods=2).mean().values
        s_low   = tf_oh['low'].rolling(N, min_periods=2).std().values
        tf_ts   = tf_oh['timestamp'].values.astype(np.int64)
        # Forward-fill onto 5s grid using period offset (no lookahead)
        target = ts_5s - period_s
        idx = np.searchsorted(tf_ts, target, side='right') - 1
        idx = np.clip(idx, 0, len(tf_ts) - 1)
        Mc = m_close[idx]; Sc = s_close[idx]
        Mh = m_high[idx];  Sh = s_high[idx]
        Ml = m_low[idx];   Sl = s_low[idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            z_close = (close_5s - Mc) / Sc
            z_high  = (close_5s - Mh) / Sh
            z_low   = (close_5s - Ml) / Sl
        out[(tf, 'close')] = {'ts': ts_5s, 'z': z_close}
        out[(tf, 'high')]  = {'ts': ts_5s, 'z': z_high}
        out[(tf, 'low')]   = {'ts': ts_5s, 'z': z_low}
    return out


def _count_entries(z: np.ndarray, k: float, side: str) -> int:
    """Count transitions from outside to inside +/-k sigma band.

    side='above': enter when z transitions from <k to >=k (price above anchor)
    side='below': enter when z transitions from >-k to <=-k (price below anchor)
    """
    if len(z) < 2:
        return 0
    finite = np.isfinite(z)
    if side == 'above':
        in_band = (z >= k) & finite
    else:
        in_band = (z <= -k) & finite
    transitions = np.zeros(len(z), dtype=bool)
    transitions[1:] = in_band[1:] & ~in_band[:-1]
    return int(transitions.sum())


def _detect_macro_runs(z: np.ndarray, ts: np.ndarray, day: str,
                      side: str, k: float, min_duration_s: int) -> list:
    """Find contiguous runs where price stays past +/-k sigma."""
    in_extreme = (z >= k) if side == 'above' else (z <= -k)
    in_extreme = in_extreme & np.isfinite(z)
    if not in_extreme.any():
        return []

    # Run-length encoding
    runs = []
    i = 0
    n = len(in_extreme)
    while i < n:
        if not in_extreme[i]:
            i += 1
            continue
        # Start of run
        j = i
        while j < n and in_extreme[j]:
            j += 1
        # Run is [i, j)
        start_ts = int(ts[i])
        end_ts = int(ts[j - 1])
        duration_s = end_ts - start_ts + 5  # 5s bars
        if duration_s >= min_duration_s:
            run_z = z[i:j]
            max_z = float(np.max(run_z) if side == 'high' else np.min(run_z))
            start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
            runs.append({
                'day':         day,
                'side':        side,
                'k_sigma':     k,
                'start_ts':    start_ts,
                'end_ts':      end_ts,
                'start_iso':   start_dt.isoformat(),
                'end_iso':     end_dt.isoformat(),
                'duration_s':  duration_s,
                'duration_min': round(duration_s / 60.0, 1),
                'max_abs_z':   abs(max_z),
                'signed_max_z': max_z,
                'tod_hour':    start_dt.hour,
                'dow':         start_dt.strftime('%a'),
            })
        i = j
    return runs


def _resolve_days(include_oos: bool) -> tuple[list, list]:
    is_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2025_*.parquet'))
    oos_paths = sorted(glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2026_*.parquet'))
    is_days = [os.path.basename(p).replace('.parquet', '') for p in is_paths]
    oos_days = [os.path.basename(p).replace('.parquet', '') for p in oos_paths]
    if not include_oos:
        oos_days = []
    return is_days, oos_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-oos', action='store_true',
                    help='Aggregate OOS days too (separate columns)')
    ap.add_argument('--min-macro-duration-s', type=int, default=60,
                    help='Minimum duration (s) for a 1h-HL macro event '
                         '(default 60s = 12 5s bars)')
    ap.add_argument('--out', default='reports/findings/band_touch_aggregation')
    ap.add_argument('--macro-tf', default='1h',
                    help='TF on which to record macro events (default 1h)')
    ap.add_argument('--macro-k', type=float, default=3.0,
                    help='Sigma threshold for macro-event detection (default 3.0)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    is_days, oos_days = _resolve_days(args.include_oos)
    all_days = [(d, 'IS') for d in is_days] + [(d, 'OOS') for d in oos_days]
    if not all_days:
        print('No days to process.'); sys.exit(1)

    print(f'Days: {len(is_days)} IS + {len(oos_days)} OOS = {len(all_days)} total')
    print(f'TFs: {TFS}   sigma levels: {SIGMA_LEVELS}')
    print(f'Macro events: {args.macro_tf} HL +/-{args.macro_k}sigma, '
          f'min duration {args.min_macro_duration_s}s')
    print()

    # Aggregators
    # entry_counts[split][(tf, anchor, side, k)] = total over all days
    entry_counts = {'IS': defaultdict(int), 'OOS': defaultdict(int)}
    days_active  = {'IS': defaultdict(set), 'OOS': defaultdict(set)}
    day_counts   = {}            # (day, tf, anchor, side, k) -> int
    day_split    = {}            # day -> 'IS' | 'OOS'
    macro_events = []
    skipped_days = []

    for day, split in tqdm(all_days, desc='aggregating'):
        z_data = _compute_z_close_vs_anchors(day)
        if not z_data:
            skipped_days.append(day)
            continue
        day_split[day] = split

        for tf in TFS:
            for anchor in ANCHORS:
                z = z_data[(tf, anchor)]['z']
                for k in SIGMA_LEVELS:
                    n_above = _count_entries(z, k, 'above')
                    n_below = _count_entries(z, k, 'below')
                    if n_above > 0:
                        key = (tf, anchor, 'above', k)
                        entry_counts[split][key] += n_above
                        days_active[split][key].add(day)
                        day_counts[(day, tf, anchor, 'above', k)] = n_above
                    if n_below > 0:
                        key = (tf, anchor, 'below', k)
                        entry_counts[split][key] += n_below
                        days_active[split][key].add(day)
                        day_counts[(day, tf, anchor, 'below', k)] = n_below

        # Macro events at args.macro_tf, HL anchors, +/- args.macro_k
        for hl_anchor in ('high', 'low'):
            zd = z_data[(args.macro_tf, hl_anchor)]
            for side in ('above', 'below'):
                runs = _detect_macro_runs(
                    zd['z'], zd['ts'], day, side, args.macro_k,
                    args.min_macro_duration_s)
                # Tag each run with its anchor for clarity
                for r in runs:
                    r['anchor'] = hl_anchor
                macro_events.extend(runs)

    if skipped_days:
        print(f'\n[warn] skipped {len(skipped_days)} days due to missing features')

    # ---- Summary by config ---------------------------------------------------
    rows = []
    for split in ['IS', 'OOS']:
        n_split_days = len([d for d in day_split if day_split[d] == split])
        if n_split_days == 0:
            continue
        for tf in TFS:
            for anchor in ANCHORS:
                for side in ('above', 'below'):
                    for k in SIGMA_LEVELS:
                        key = (tf, anchor, side, k)
                        total = entry_counts[split][key]
                        n_active = len(days_active[split][key])
                        mean_per_day = total / n_split_days
                        per_day_vals = [
                            day_counts.get((d, tf, anchor, side, k), 0)
                            for d in day_split if day_split[d] == split
                        ]
                        rows.append({
                            'split':            split,
                            'tf':               tf,
                            'anchor':           anchor,
                            'side':             side,
                            'k_sigma':          k,
                            'total_entries':    total,
                            'days_active':      n_active,
                            'split_days':       n_split_days,
                            'frac_days_active': round(n_active / n_split_days, 3),
                            'mean_per_day':     round(mean_per_day, 2),
                            'median_per_day':   float(np.median(per_day_vals)) if per_day_vals else 0.0,
                            'max_per_day':      max(per_day_vals) if per_day_vals else 0,
                            'q90_per_day':      float(np.quantile(per_day_vals, 0.9)) if per_day_vals else 0.0,
                        })
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out, 'summary_by_config.csv')
    summary.to_csv(summary_path, index=False)
    print(f'\nWrote {summary_path}  ({len(summary)} rows)')

    # ---- Per-day wide table --------------------------------------------------
    rows_pd = []
    for d, split in day_split.items():
        row = {'day': d, 'split': split}
        for tf in TFS:
            for anchor in ANCHORS:
                for side in ('above', 'below'):
                    for k in SIGMA_LEVELS:
                        col = f'{tf}_{anchor}_{side}_{int(k)}s'
                        row[col] = day_counts.get((d, tf, anchor, side, k), 0)
        rows_pd.append(row)
    per_day = pd.DataFrame(rows_pd).sort_values('day').reset_index(drop=True)
    per_day_path = os.path.join(args.out, 'per_day_counts.csv')
    per_day.to_csv(per_day_path, index=False)
    print(f'Wrote {per_day_path}  ({len(per_day)} days)')

    # ---- Macro events --------------------------------------------------------
    if macro_events:
        macro_df = pd.DataFrame(macro_events).sort_values(
            ['day', 'start_ts']).reset_index(drop=True)
    else:
        macro_df = pd.DataFrame(columns=[
            'day', 'side', 'anchor', 'k_sigma', 'start_ts', 'end_ts',
            'duration_s', 'duration_min', 'max_abs_z', 'signed_max_z',
            'tod_hour', 'dow', 'split'])
    macro_df['split'] = macro_df['day'].map(day_split) if len(macro_df) else macro_df.get('split', pd.Series([], dtype=object))
    macro_path = os.path.join(args.out, f'macro_events_{args.macro_tf}_hl.csv')
    macro_df.to_csv(macro_path, index=False)
    print(f'Wrote {macro_path}  ({len(macro_df)} runs)')

    # ---- Headline report -----------------------------------------------------
    n_is_proc  = sum(1 for d in day_split if day_split[d] == 'IS')
    n_oos_proc = sum(1 for d in day_split if day_split[d] == 'OOS')
    md_lines = [
        f'# Band-touch aggregation report',
        f'',
        f'_Generated {datetime.now().isoformat()}_',
        f'',
        f'## Coverage',
        f'',
        f'- IS days processed:  {n_is_proc}',
        f'- OOS days processed: {n_oos_proc}',
        f'- Skipped (missing features): {len(skipped_days)}',
        f'',
        f'## Entry counts by (TF x side x k_sigma)',
        f'',
        f'```',
    ]
    for split in ['IS', 'OOS']:
        if not any(r['split'] == split for r in rows):
            continue
        md_lines.append(f'== {split} ==')
        md_lines.append(f'{"tf":<6} {"side":<6} {"k":<5} {"total":>10} '
                        f'{"days_act":>10} {"mean/d":>10} {"q90/d":>10} {"max/d":>10}')
        for r in rows:
            if r['split'] != split:
                continue
            md_lines.append(
                f'{r["tf"]:<6} {r["side"]:<6} {r["k_sigma"]:<5.1f} '
                f'{r["total_entries"]:>10d} {r["days_active"]:>10d} '
                f'{r["mean_per_day"]:>10.2f} {r["q90_per_day"]:>10.1f} '
                f'{r["max_per_day"]:>10d}')
        md_lines.append('')
    md_lines.append('```')
    md_lines.append('')

    md_lines.extend([
        f'## Macro events ({args.macro_tf} HL +/-{args.macro_k}sigma, min duration '
        f'{args.min_macro_duration_s}s)',
        f'',
        f'- Total runs: {len(macro_df)}',
    ])
    if len(macro_df):
        n_macro_is = int((macro_df['split'] == 'IS').sum())
        n_macro_oos = int((macro_df['split'] == 'OOS').sum())
        n_above = int((macro_df['side'] == 'above').sum())
        n_below = int((macro_df['side'] == 'below').sum())
        md_lines.extend([
            f'- IS runs:    {n_macro_is}',
            f'- OOS runs:   {n_macro_oos}',
            f'- Above-anchor runs (rallies): {n_above}',
            f'- Below-anchor runs (crashes): {n_below}',
            f'- Median duration: {macro_df["duration_min"].median():.1f} min',
            f'- q90 duration:    {macro_df["duration_min"].quantile(0.9):.1f} min',
            f'- Max duration:    {macro_df["duration_min"].max():.1f} min',
            f'',
            f'### Days with the most macro time (top 15 by total macro minutes)',
            f'',
            f'```',
        ])
        days_sum = macro_df.groupby('day').agg(
            n_runs=('day', 'size'),
            total_min=('duration_min', 'sum'),
            max_z=('max_abs_z', 'max'),
            split=('split', 'first'),
        ).sort_values('total_min', ascending=False)
        for d, r in days_sum.head(15).iterrows():
            md_lines.append(
                f'  {d}  {r["split"]:>3}  n_runs={r["n_runs"]:>2}  '
                f'total={r["total_min"]:>5.1f}min  max_z={r["max_z"]:>5.2f}')
        md_lines.append('```')
        md_lines.append('')
        md_lines.extend([
            f'### Top 20 single longest macro runs',
            f'',
            f'```',
            f'{"day":<12} {"side":<6} {"anchor":<6} {"start":<9} '
            f'{"dur_min":>7} {"max_|z|":>8} {"split":>5}',
        ])
        for _, r in macro_df.sort_values(
                'duration_min', ascending=False).head(20).iterrows():
            start_hm = pd.to_datetime(r['start_ts'], unit='s', utc=True).strftime('%H:%M:%S')
            md_lines.append(
                f'{r["day"]:<12} {r["side"]:<6} {r["anchor"]:<6} {start_hm:<9} '
                f'{r["duration_min"]:>7.1f} {r["max_abs_z"]:>8.2f} {r["split"]:>5}')
        md_lines.append('```')
    md_lines.append('')
    md_lines.append('## Notes')
    md_lines.append('')
    md_lines.append('- Z-definition: z = (5s_close - M_anchor) / SE_anchor.')
    md_lines.append('  M_anchor and SE_anchor are rolling regression mean/std of the')
    md_lines.append('  TF OHLCV {close, high, low} columns at TF_WINDOW periods.')
    md_lines.append('- Entry := transition from outside to inside the +/-k band on the')
    md_lines.append('  side indicated (above or below the anchor).')
    md_lines.append('- Forward-pass clean: each 5s bar reads only the most recent COMPLETED')
    md_lines.append('  TF bar (target = ts - period_s). No future-bar leakage.')
    md_lines.append('- Min N for a probability cell to be statistically meaningful is')
    md_lines.append('  roughly 100. Cells with `total_entries < 100` are already too thin')
    md_lines.append('  even before further conditioning by regime/state/tod/dow.')

    md_path = os.path.join(args.out, 'summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'Wrote {md_path}')

    # ---- Console headline ----------------------------------------------------
    print('\n' + '=' * 80)
    print('HEADLINE: entries-per-config (IS only)')
    print('=' * 80)
    is_summary = summary[summary['split'] == 'IS'].sort_values(
        ['tf', 'side', 'k_sigma'])
    print(is_summary[['tf', 'anchor', 'side', 'k_sigma', 'total_entries',
                      'days_active', 'mean_per_day', 'q90_per_day',
                      'max_per_day']].to_string(index=False))
    print()
    print(f'Macro events: {len(macro_df)} runs '
          f'({(macro_df["split"]=="IS").sum()} IS, {(macro_df["split"]=="OOS").sum()} OOS)')
    if len(macro_df):
        print(f'  Median duration: {macro_df["duration_min"].median():.1f} min')
        print(f'  Max duration:    {macro_df["duration_min"].max():.1f} min')
        print(f'  Days with macro events: {macro_df["day"].nunique()} '
              f'of {len(day_split)} processed')


if __name__ == '__main__':
    main()
