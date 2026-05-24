"""Segment chord fingerprint — EDA of 5s notes/chords WITHIN each segment.

Step 2 of the music-hierarchy pipeline (per 2026-05-10 user spec):
  Step 1 (done):  segment hierarchically (phrase 15m -> motif 5m)
  Step 2 (THIS):  for each segment, compute the EDA of 5s at-bar
                   primitives (notes/chords) playing INSIDE it.
                   The result is the segment's chord fingerprint —
                   distribution stats that distinguish two segments
                   with the same macro shape_class.

Per-5s-bar primitives (the notes in the chord at bar t)
-------------------------------------------------------
  slope_15m_at_bar      slope of M_close_15m measured over a 60min lookback at t
  z_close_15m_at_bar    (5s_close - M_close_15m) / SE_close_15m at t
  sigma_rank_15m_at_bar rolling 60min percentile of SE_close_15m at t
  slope_5m_at_bar       slope of M_close_5m measured over a 30min lookback at t
  z_close_5m_at_bar     (5s_close - M_close_5m) / SE_close_5m at t
  sigma_rank_5m_at_bar  rolling 60min percentile of SE_close_5m at t
  r2adj_5m_at_bar       R^2_adj of 5min linear fit to 5s closes ending at t

Per-segment chord fingerprint (EDA of bar-level notes within segment)
---------------------------------------------------------------------
For each note above, compute over the segment's 5s bars:
  mean      central tendency
  std       spread
  q10, q90  tail quantiles
  pct_pos   fraction of bars with value > 0   (only for signed notes)

These collectively describe the chord that PLAYED during the segment —
not just the segment's macro slope, but the distribution of micro slopes
of every 5s bar that lived inside it.

USAGE
    python tools/segment_chord_fingerprint.py --level motif
    python tools/segment_chord_fingerprint.py --level phrase
    python tools/segment_chord_fingerprint.py --level both
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s, TF_WINDOW, PERIOD_S
)


SEG_DIR = 'reports/findings/segments'

SLOPE_LOOKBACK_5S_BARS_15M = 720   # 60min for slope at 15m anchor
SLOPE_LOOKBACK_5S_BARS_5M  = 360   # 30min for slope at 5m anchor
RANK_WINDOW_5S_BARS         = 720   # 60min for sigma rank
R2_WINDOW_5S_BARS           = 60    # 5min linear fit window


def _compute_at_bar_notes(day: str) -> dict | None:
    """Return per-5s-bar arrays of at-bar primitives ('notes')."""
    df_5s = _load_5s(day)
    if df_5s.empty:
        return None
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values.astype(np.float64)

    out = {'ts': ts_5s, 'close': close_5s}

    for tf, lookback_5s in [('15m', SLOPE_LOOKBACK_5S_BARS_15M),
                            ('5m',  SLOPE_LOOKBACK_5S_BARS_5M)]:
        oh = _load_tf_ohlcv(tf, day)
        if oh.empty:
            return None
        N = TF_WINDOW[tf]
        oh['M'] = oh['close'].rolling(N, min_periods=2).mean()
        oh['S'] = oh['close'].rolling(N, min_periods=2).std()
        tf_ts = oh['timestamp'].values.astype(np.int64)
        M = _ffill_to_5s(oh['M'].values, tf_ts, ts_5s, PERIOD_S[tf])
        S = _ffill_to_5s(oh['S'].values, tf_ts, ts_5s, PERIOD_S[tf])

        n = len(M)
        slope = np.full(n, np.nan)
        if n > lookback_5s:
            slope[lookback_5s:] = (M[lookback_5s:] - M[:-lookback_5s]) / lookback_5s
        with np.errstate(divide='ignore', invalid='ignore'):
            z_close = (close_5s - M) / S
        sigma_rank = (pd.Series(S)
                        .rolling(RANK_WINDOW_5S_BARS, min_periods=20)
                        .rank(pct=True).values)

        out[f'slope_{tf}'] = slope
        out[f'z_close_{tf}'] = z_close
        out[f'sigma_rank_{tf}'] = sigma_rank

    # r2adj_5m at-bar (5min linear fit to 5s closes ending at each bar)
    out['r2adj_5m'] = _rolling_r2adj_at_bar(close_5s, R2_WINDOW_5S_BARS)
    return out


def _rolling_r2adj_at_bar(y: np.ndarray, n: int) -> np.ndarray:
    """At each bar i, compute R^2_adj of linear fit to y[i-n+1 .. i]."""
    N = len(y)
    out = np.full(N, np.nan)
    if N < n:
        return out
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    Sxx = float(((x - x_mean) ** 2).sum())
    if Sxx == 0:
        return out
    cs1 = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])
    sum_xy_window = np.convolve(y, x[::-1], mode='valid')
    for k, i in enumerate(range(n - 1, N)):
        sum_y = cs1[i + 1] - cs1[i + 1 - n]
        sum_y2 = cs2[i + 1] - cs2[i + 1 - n]
        y_mean = sum_y / n
        SS_tot = sum_y2 - n * y_mean * y_mean
        if SS_tot <= 0:
            continue
        Sxy = sum_xy_window[k] - x_mean * sum_y
        b = Sxy / Sxx
        SS_res = SS_tot - b * Sxy
        if SS_res < 0:
            SS_res = 0.0
        r2 = 1.0 - SS_res / SS_tot
        if n - 2 <= 0:
            continue
        out[i] = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
    return out


def _aggregate_chord(notes: dict, start_ts: int, end_ts: int) -> dict:
    """Aggregate 5s at-bar notes within [start_ts, end_ts] into chord fingerprint."""
    ts = notes['ts']
    s_idx = int(np.searchsorted(ts, start_ts, side='left'))
    e_idx = int(np.searchsorted(ts, end_ts, side='right'))
    if e_idx <= s_idx:
        return {}

    out = {'n_bars_in_seg': int(e_idx - s_idx)}
    for note_name in ['slope_15m', 'z_close_15m', 'sigma_rank_15m',
                      'slope_5m', 'z_close_5m', 'sigma_rank_5m',
                      'r2adj_5m']:
        arr = notes[note_name][s_idx:e_idx]
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        out[f'{note_name}__mean']    = float(np.mean(finite))
        out[f'{note_name}__std']     = float(np.std(finite))
        out[f'{note_name}__q10']     = float(np.quantile(finite, 0.10))
        out[f'{note_name}__q50']     = float(np.quantile(finite, 0.50))
        out[f'{note_name}__q90']     = float(np.quantile(finite, 0.90))
        if note_name in ('slope_15m', 'z_close_15m', 'slope_5m', 'z_close_5m'):
            out[f'{note_name}__pct_pos'] = float((finite > 0).mean())
    return out


def attach_chord_fingerprint(seg_df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    """For each row in seg_df, compute and attach segment_chord stats."""
    seg_df = seg_df.copy().reset_index(drop=True)
    days = seg_df['day'].unique()
    print(f'[{level_name}] computing 5s notes for {len(days)} days...')

    notes_cache = {}
    for d in tqdm(days, desc=f'5s notes ({level_name})'):
        notes = _compute_at_bar_notes(d)
        if notes is not None:
            notes_cache[d] = notes

    print(f'[{level_name}] aggregating chord fingerprints for {len(seg_df)} segments...')
    rows = []
    for _, seg in tqdm(seg_df.iterrows(), total=len(seg_df), desc=f'chord ({level_name})'):
        d = seg['day']
        if d not in notes_cache:
            rows.append(seg.to_dict())
            continue
        chord = _aggregate_chord(notes_cache[d],
                                 int(seg['start_ts']), int(seg['end_ts']))
        out = seg.to_dict()
        out.update(chord)
        rows.append(out)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='both', choices=['phrase', 'motif', 'both'])
    ap.add_argument('--motif-csv',
                    default=os.path.join(SEG_DIR, 'all_motifs_labeled.csv'))
    ap.add_argument('--melody-csv',
                    default=os.path.join(SEG_DIR, 'all_melodies_labeled.csv'))
    args = ap.parse_args()

    # Note: in code, motifs are 15m-anchored (= phrases in the new vocab),
    # and melodies are 5m-anchored (= motifs in the new vocab). Pending rename.
    if args.level in ('phrase', 'both'):
        df = pd.read_csv(args.motif_csv)
        out = attach_chord_fingerprint(df, 'phrase (15m)')
        out_path = os.path.join(SEG_DIR, 'all_motifs_labeled_with_chord.csv')
        out.to_csv(out_path, index=False)
        print(f'\nWrote {out_path}  ({len(out)} phrases / 15m-motifs)')

    if args.level in ('motif', 'both'):
        df = pd.read_csv(args.melody_csv)
        out = attach_chord_fingerprint(df, 'motif (5m)')
        out_path = os.path.join(SEG_DIR, 'all_melodies_labeled_with_chord.csv')
        out.to_csv(out_path, index=False)
        print(f'\nWrote {out_path}  ({len(out)} motifs / 5m-melodies)')


if __name__ == '__main__':
    main()
