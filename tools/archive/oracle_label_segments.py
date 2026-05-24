"""Oracle-label motifs and melodies with lookahead-OK outcome variables.

Reads `reports/findings/segments/all_motifs.csv` and `all_melodies.csv`,
walks the underlying 5s OHLCV for each segment span, and computes outcomes
that the Bayesian table will key on. Lookahead is permitted here per
`memory/feedback_oracle_vs_chord_lookahead.md`.

Per-segment outcome labels added
---------------------------------
ride_pnl_pts       sign(slope) * net_move_pts (ride-with-slope payoff)
fade_pnl_pts       -ride_pnl_pts (ride-against-slope payoff)
max_mfe_ride_pts   peak favorable excursion if rode-with-slope from start
max_mae_ride_pts   peak adverse excursion if rode-with-slope from start
max_mfe_fade_pts   peak favorable excursion if faded-against-slope
max_mae_fade_pts   peak adverse excursion if faded-against-slope
time_to_max_z_min  minutes from segment start to bar with max |z|
time_to_revert_min minutes until first bar where price retraced >=50% of net_move
                    (NaN if no revert within segment)
resolved_as_cascade boolean: length_min >= 60 AND |peak_abs_z| >= 4
extended_60m       boolean: length_min >= 60
extended_120m      boolean: length_min >= 120
next_seg_continued boolean: did the NEXT segment (same level) have the same
                    slope sign (continuation of direction)
next_seg_idx       index of next segment in the same day (-1 if last)
prev_seg_shape     shape_class of preceding segment (or 'NONE' for first)
sequence_position  index within the day (0-based)

Output
------
reports/findings/segments/all_motifs_labeled.csv
reports/findings/segments/all_melodies_labeled.csv
reports/findings/segments/oracle_summary.md   per-shape outcome distributions

Usage:
    python tools/oracle_label_segments.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SEG_DIR = 'reports/findings/segments'


def _load_5s(day: str) -> pd.DataFrame | None:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _label_segment(close_5s: np.ndarray, ts_5s: np.ndarray,
                   start_ts: int, end_ts: int,
                   slope_sign: int, peak_abs_z: float,
                   length_min: float) -> dict:
    """Compute oracle-style outcome labels for one segment span."""
    s_idx = int(np.searchsorted(ts_5s, start_ts, side='left'))
    e_idx = int(np.searchsorted(ts_5s, end_ts, side='right') - 1)
    s_idx = max(0, min(s_idx, len(ts_5s) - 1))
    e_idx = max(s_idx, min(e_idx, len(ts_5s) - 1))

    closes = close_5s[s_idx:e_idx + 1]
    ts = ts_5s[s_idx:e_idx + 1]
    if len(closes) < 2:
        return {
            'ride_pnl_pts':       float('nan'),
            'fade_pnl_pts':       float('nan'),
            'max_mfe_ride_pts':   float('nan'),
            'max_mae_ride_pts':   float('nan'),
            'max_mfe_fade_pts':   float('nan'),
            'max_mae_fade_pts':   float('nan'),
            'time_to_max_z_min':  float('nan'),
            'time_to_revert_min': float('nan'),
            'resolved_as_cascade': False,
            'extended_60m':       length_min >= 60,
            'extended_120m':      length_min >= 120,
        }

    entry = closes[0]
    move = closes - entry  # signed move from entry

    # Ride-with-slope: long if slope_sign>0, short if <0
    if slope_sign >= 0:
        ride_path = move
    else:
        ride_path = -move
    fade_path = -ride_path

    max_mfe_ride = float(ride_path.max())
    max_mae_ride = float(ride_path.min())  # adverse = min of ride payoff
    max_mfe_fade = float(fade_path.max())
    max_mae_fade = float(fade_path.min())

    net_move_pts = float(closes[-1] - closes[0])
    ride_pnl_pts = abs(net_move_pts) * slope_sign * np.sign(net_move_pts) if slope_sign != 0 else 0.0
    # Simpler: ride_pnl follows slope_sign * net_move
    ride_pnl_pts = float(slope_sign * net_move_pts) if slope_sign != 0 else 0.0
    fade_pnl_pts = -ride_pnl_pts

    # time_to_max_z: max |close - entry| relative to segment span
    abs_move = np.abs(move)
    if len(abs_move) > 0 and abs_move.max() > 0:
        i_max = int(np.argmax(abs_move))
        time_to_max_z_min = float((ts[i_max] - ts[0]) / 60.0)
    else:
        time_to_max_z_min = float('nan')

    # time_to_revert: first bar where price retraces >=50% of net_move from peak
    half_move = net_move_pts / 2.0
    time_to_revert_min = float('nan')
    if abs(net_move_pts) > 0.5:  # ignore tiny moves
        # Find peak in net_move direction
        if net_move_pts > 0:
            peak_idx = int(np.argmax(closes))
        else:
            peak_idx = int(np.argmin(closes))
        peak = closes[peak_idx]
        target = peak - half_move
        # After peak, find first bar where close retraced past target
        for k in range(peak_idx + 1, len(closes)):
            if (net_move_pts > 0 and closes[k] <= target) or \
               (net_move_pts < 0 and closes[k] >= target):
                time_to_revert_min = float((ts[k] - ts[0]) / 60.0)
                break

    resolved_as_cascade = (length_min >= 60.0) and (peak_abs_z >= 4.0)

    return {
        'ride_pnl_pts':        round(ride_pnl_pts, 2),
        'fade_pnl_pts':        round(fade_pnl_pts, 2),
        'max_mfe_ride_pts':    round(max_mfe_ride, 2),
        'max_mae_ride_pts':    round(max_mae_ride, 2),
        'max_mfe_fade_pts':    round(max_mfe_fade, 2),
        'max_mae_fade_pts':    round(max_mae_fade, 2),
        'time_to_max_z_min':   round(time_to_max_z_min, 2) if np.isfinite(time_to_max_z_min) else float('nan'),
        'time_to_revert_min':  round(time_to_revert_min, 2) if np.isfinite(time_to_revert_min) else float('nan'),
        'resolved_as_cascade': bool(resolved_as_cascade),
        'extended_60m':        bool(length_min >= 60),
        'extended_120m':       bool(length_min >= 120),
    }


def _add_sequence_context(df: pd.DataFrame, parent_key: str = None) -> pd.DataFrame:
    """Add prev_seg_shape, next_seg_idx, next_seg_continued, sequence_position.

    parent_key: if given, sequence numbers are within-parent (per motif); else
                within-day for top-level motifs.
    """
    df = df.copy()
    df = df.sort_values(['day', 'start_ts']).reset_index(drop=True)
    if parent_key:
        df['_grp'] = df['day'].astype(str) + '|' + df[parent_key].astype(str)
    else:
        df['_grp'] = df['day'].astype(str)

    # sequence position within group
    df['sequence_position'] = df.groupby('_grp').cumcount()

    # prev shape (default NONE for first in group)
    df['prev_seg_shape'] = df.groupby('_grp')['shape_class'].shift(1).fillna('NONE')

    # next shape index + continuation
    next_slope = df.groupby('_grp')['slope_pts_per_min'].shift(-1)
    df['next_seg_idx'] = df.groupby('_grp').cumcount(ascending=False) - 1
    df.loc[df['next_seg_idx'] < 0, 'next_seg_idx'] = -1

    cur_sign = np.sign(df['slope_pts_per_min'].values)
    nxt_sign = np.sign(next_slope.values)
    nxt_finite = np.isfinite(next_slope.values)
    df['next_seg_continued'] = (cur_sign == nxt_sign) & nxt_finite & (cur_sign != 0)
    # If no next segment (last in group), encode as False
    df.loc[df['next_seg_idx'] < 0, 'next_seg_continued'] = False

    return df.drop(columns=['_grp'])


def label_segments(seg_df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Walk each segment, attach outcome labels."""
    seg_df = seg_df.copy()

    # Group by day to load each day's 5s once
    rows = []
    days = seg_df['day'].unique()
    print(f'  Labeling {len(seg_df)} {level}s across {len(days)} days...')
    cache = {}
    for day in tqdm(days, desc=f'load+label {level}'):
        df_5s = _load_5s(day)
        if df_5s is None:
            continue
        ts_5s = df_5s['timestamp'].values.astype(np.int64)
        close_5s = df_5s['close'].values.astype(np.float64)
        cache[day] = (ts_5s, close_5s)

    for _, seg in tqdm(seg_df.iterrows(), total=len(seg_df), desc=f'label {level}'):
        d = seg['day']
        if d not in cache:
            continue
        ts_5s, close_5s = cache[d]
        slope_val = seg['slope_pts_per_min']
        if pd.isna(slope_val) or slope_val == 0:
            slope_sign = 0
        else:
            slope_sign = int(np.sign(slope_val))
        peak_abs_z = float(seg['peak_abs_z']) if pd.notna(seg['peak_abs_z']) else 0.0
        labels = _label_segment(
            close_5s, ts_5s,
            int(seg['start_ts']), int(seg['end_ts']),
            slope_sign, peak_abs_z, float(seg['length_min']))
        row = seg.to_dict()
        row.update(labels)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(motif_lab: pd.DataFrame, melody_lab: pd.DataFrame, out_path: str):
    """Per-shape outcome distribution summary."""
    md = ['# Oracle-labeled segment summary', '',
          f'_Generated {datetime.now().isoformat()}_', '',
          f'- Motifs labeled:   {len(motif_lab)}',
          f'- Melodies labeled: {len(melody_lab)}', '']

    def _shape_block(df, label):
        lines = []
        lines.append(f'## {label} shape -> outcome (IS only)')
        lines.append('')
        lines.append('```')
        lines.append(f'{"shape_class":<22s} {"n":>5s} '
                     f'{"ride_$":>8s} {"fade_$":>8s} '
                     f'{"%cascade":>9s} {"%cont_seg":>10s} '
                     f'{"med_mfe_ride":>12s} {"med_mae_ride":>12s}')
        lines.append('-' * 100)
        sub = df[df['split'] == 'IS']
        shape_cnts = sub['shape_class'].value_counts()
        for sh, n in shape_cnts.items():
            s = sub[sub['shape_class'] == sh]
            ride_mean = s['ride_pnl_pts'].mean()
            fade_mean = s['fade_pnl_pts'].mean()
            pct_cascade = 100 * s['resolved_as_cascade'].mean()
            pct_cont = 100 * s['next_seg_continued'].mean()
            med_mfe = s['max_mfe_ride_pts'].median()
            med_mae = s['max_mae_ride_pts'].median()
            lines.append(
                f'{sh:<22s} {n:>5d} '
                f'{ride_mean:>+8.1f} {fade_mean:>+8.1f} '
                f'{pct_cascade:>8.1f}% {pct_cont:>9.1f}% '
                f'{med_mfe:>+12.1f} {med_mae:>+12.1f}')
        lines.append('```')
        lines.append('')
        return lines

    md += _shape_block(motif_lab, 'MOTIF')
    md += _shape_block(melody_lab, 'MELODY')

    md += [
        '## Notes', '',
        '- ride_pnl_pts: signed PnL of trading WITH the segment slope direction',
        '  (entry at start, exit at end). Positive = trading with the slope made money.',
        '- fade_pnl_pts: -ride_pnl_pts. The segment that "gives" ride_pnl is exactly',
        '  the segment that "takes" fade_pnl.',
        '- %cascade: fraction with length>=60min AND peak_abs_z>=4 (the macro-event criterion).',
        '- %cont_seg: fraction whose NEXT segment continues the same slope direction.',
        '- max_mfe/mae_ride: peak favorable / adverse excursion DURING the segment',
        '  if traded with the slope from segment start.',
        '- These outcomes are the lookup values the Bayesian table will key on.',
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-motifs', default=os.path.join(SEG_DIR, 'all_motifs.csv'))
    ap.add_argument('--in-melodies', default=os.path.join(SEG_DIR, 'all_melodies.csv'))
    ap.add_argument('--out-motifs', default=os.path.join(SEG_DIR, 'all_motifs_labeled.csv'))
    ap.add_argument('--out-melodies', default=os.path.join(SEG_DIR, 'all_melodies_labeled.csv'))
    args = ap.parse_args()

    motif_df = pd.read_csv(args.in_motifs)
    melody_df = pd.read_csv(args.in_melodies)
    print(f'Loaded {len(motif_df)} motifs, {len(melody_df)} melodies')

    # Add sequence context (within-day for motifs, within-motif for melodies)
    motif_df = _add_sequence_context(motif_df, parent_key=None)
    melody_df = _add_sequence_context(melody_df, parent_key='parent_motif_idx')

    motif_lab = label_segments(motif_df, 'motif')
    melody_lab = label_segments(melody_df, 'melody')

    motif_lab.to_csv(args.out_motifs, index=False)
    melody_lab.to_csv(args.out_melodies, index=False)
    print(f'\nWrote {args.out_motifs}  ({len(motif_lab)} motifs)')
    print(f'Wrote {args.out_melodies} ({len(melody_lab)} melodies)')

    summary_path = os.path.join(SEG_DIR, 'oracle_summary.md')
    write_summary(motif_lab, melody_lab, summary_path)
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
