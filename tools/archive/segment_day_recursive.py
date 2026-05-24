"""Recursive multi-level day segmenter — phrase / motif / sub-motif / measure.

Generalizes segment_day_motif_melody.py to arbitrary depth. The hierarchy
consumes a list of (TF, min_duration_min) levels and produces a NESTED
segment tree. Each segment at any level carries its own segment_chord
(slope, sigma_rank, r2adj, shape_class with magnitude gate, length, peak_z, ...)
computed from its OWN TF anchor over its OWN span.

Level vocabulary (per memory/feedback_no_human_regime_terms.md):
    theme    = day                           (root)
    phrase   = 15m segment                   (the most stable line)
    motif    = 5m segment NESTED inside phrase
    sub-motif = 1m segment NESTED inside motif
    measure  = 15s segment NESTED inside sub-motif
    note/chord = 5s at-bar (terminal — handled by segment_chord_fingerprint)

Forward-pass clean: TF M_close lookback uses ts - period_s offset to avoid
peeking at the bar that hasn't closed yet.

USAGE
    python tools/segment_day_recursive.py --day 2026_02_12
    python tools/segment_day_recursive.py --day 2026_02_12 --levels 15m:30,5m:5,1m:1,15s:0.25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.research.seeds import SeedPrimitiveLibrary
from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s,
    _detect_inflections_5s, _merge_short_segments,
    _rolling_r2_adjusted, _classify_shape,
    TF_WINDOW, PERIOD_S,
)


# Add 15s to the TF table if missing
if '15s' not in TF_WINDOW:
    TF_WINDOW['15s'] = 12   # 3-min lookback (12 × 15s)
    PERIOD_S['15s']  = 15

LEVEL_NAMES = {
    '15m': 'phrase',
    '5m':  'motif',
    '1m':  'sub_motif',
    '15s': 'measure',
}

RANK_WINDOW_5S_BARS = 720


def _build_segment_record(level_name: str, parent_idx: int, seg_idx: int,
                          start_i: int, end_i: int,
                          ts_5s: np.ndarray, close_5s: np.ndarray,
                          M_5s: np.ndarray, S_5s: np.ndarray,
                          sigma_rank_5s: np.ndarray,
                          shape_lib: SeedPrimitiveLibrary) -> dict:
    """Build a single segment record with full chord EDA."""
    s_ts = int(ts_5s[start_i])
    e_ts = int(ts_5s[end_i])
    length_min = (e_ts - s_ts) / 60.0
    slope_pts_per_min = ((M_5s[end_i] - M_5s[start_i]) / length_min) if length_min > 0 else 0.0
    seg_S = S_5s[start_i:end_i + 1]
    mean_sigma = float(np.nanmean(seg_S)) if seg_S.size else float('nan')
    mid = (start_i + end_i) // 2
    sigma_rank_mid = float(sigma_rank_5s[mid]) if np.isfinite(sigma_rank_5s[mid]) else float('nan')
    closes = close_5s[start_i:end_i + 1]
    r2adj = _rolling_r2_adjusted(closes)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (closes - M_5s[start_i:end_i + 1]) / S_5s[start_i:end_i + 1]
    peak_abs_z = float(np.nanmax(np.abs(z))) if np.any(np.isfinite(z)) else float('nan')
    net_move_pts = float(close_5s[end_i] - close_5s[start_i])
    shape_input = M_5s[start_i:end_i + 1]
    shape_class, shape_r = _classify_shape(shape_input, shape_lib,
                                           mean_sigma=mean_sigma,
                                           flat_threshold_sigma=1.0)
    s_dt = datetime.fromtimestamp(s_ts, tz=timezone.utc)
    e_dt = datetime.fromtimestamp(e_ts, tz=timezone.utc)
    return {
        'level': level_name,
        'parent_seg_idx': parent_idx,
        'seg_idx': seg_idx,
        'start_ts': s_ts,
        'end_ts': e_ts,
        'start_iso': s_dt.isoformat(),
        'end_iso': e_dt.isoformat(),
        'length_min': round(length_min, 2),
        'slope_pts_per_min': round(slope_pts_per_min, 4),
        'mean_sigma': round(mean_sigma, 3),
        'sigma_rank_mid': round(sigma_rank_mid, 3),
        'r2adj': round(r2adj, 3) if np.isfinite(r2adj) else float('nan'),
        'shape_class': shape_class,
        'shape_pearson_r': round(shape_r, 3),
        'peak_abs_z': round(peak_abs_z, 3) if np.isfinite(peak_abs_z) else float('nan'),
        'tod_start_hour_utc': s_dt.hour,
        'net_move_pts': round(net_move_pts, 2),
    }


def _segment_within_span(level_name: str, parent_idx: int,
                         tf: str, min_duration_min: float,
                         start_i: int, end_i: int,
                         ts_5s: np.ndarray, close_5s: np.ndarray,
                         tf_anchor_data: dict,
                         shape_lib: SeedPrimitiveLibrary) -> list[dict]:
    """Segment the [start_i, end_i] span at the given TF anchor."""
    M = tf_anchor_data['M']
    S = tf_anchor_data['S']
    sigma_rank = tf_anchor_data['sigma_rank']

    # Inflection detection within the span
    M_span = M[start_i:end_i + 1]
    ts_span = ts_5s[start_i:end_i + 1]
    inflections = _detect_inflections_5s(M_span, ts_span, PERIOD_S[tf])
    inflections = _merge_short_segments(inflections, ts_span, int(min_duration_min * 60))
    inflections = [start_i + j for j in inflections]

    # If span is shorter than min_duration_min, just emit the whole span as
    # one segment at this level.
    if len(inflections) < 2:
        if (ts_5s[end_i] - ts_5s[start_i]) >= int(min_duration_min * 60):
            inflections = [start_i, end_i]
        else:
            return []

    out = []
    for j in range(len(inflections) - 1):
        s = inflections[j]
        e = inflections[j + 1]
        if (ts_5s[e] - ts_5s[s]) < int(min_duration_min * 60):
            continue
        rec = _build_segment_record(level_name, parent_idx, j,
                                    s, e, ts_5s, close_5s,
                                    M, S, sigma_rank, shape_lib)
        out.append(rec)
    return out


def _prepare_tf_anchor(tf: str, day: str, ts_5s: np.ndarray) -> dict | None:
    """Pre-compute M_close, SE_close, sigma_rank arrays at 5s cadence for one TF."""
    oh = _load_tf_ohlcv(tf, day)
    if oh.empty:
        return None
    N = TF_WINDOW[tf]
    oh['M'] = oh['close'].rolling(N, min_periods=2).mean()
    oh['S'] = oh['close'].rolling(N, min_periods=2).std()
    tf_ts = oh['timestamp'].values.astype(np.int64)
    M = _ffill_to_5s(oh['M'].values, tf_ts, ts_5s, PERIOD_S[tf])
    S = _ffill_to_5s(oh['S'].values, tf_ts, ts_5s, PERIOD_S[tf])
    sigma_rank = (pd.Series(S).rolling(RANK_WINDOW_5S_BARS, min_periods=20)
                  .rank(pct=True).values)
    return {'M': M, 'S': S, 'sigma_rank': sigma_rank}


def segment_day_recursive(day: str,
                           levels: list[tuple[str, float]]) -> dict:
    """levels: list of (tf, min_duration_min) ordered from coarsest to finest."""
    df_5s = _load_5s(day)
    if df_5s.empty:
        return {}
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values.astype(np.float64)

    # Prepare TF anchors for all levels
    anchors = {}
    for tf, _ in levels:
        a = _prepare_tf_anchor(tf, day, ts_5s)
        if a is None:
            print(f'  [warn] no {tf} OHLCV for {day}'); return {}
        anchors[tf] = a

    shape_lib = SeedPrimitiveLibrary(N=16)

    # ── Recursive segmentation ────────────────────────────────────────
    # tree[level_name] = list of segment dicts, each with `_children` key
    # holding child segments at next level
    def _recurse(level_idx: int, parent_idx: int,
                 start_i: int, end_i: int) -> list[dict]:
        if level_idx >= len(levels):
            return []
        tf, min_min = levels[level_idx]
        level_name = LEVEL_NAMES.get(tf, f'level_{level_idx}_{tf}')
        segs = _segment_within_span(level_name, parent_idx, tf, min_min,
                                    start_i, end_i,
                                    ts_5s, close_5s, anchors[tf], shape_lib)
        for seg in segs:
            # Map start_ts/end_ts back to indices for child recursion
            s_idx = int(np.searchsorted(ts_5s, seg['start_ts'], side='left'))
            e_idx = int(np.searchsorted(ts_5s, seg['end_ts'], side='right') - 1)
            seg['_children'] = _recurse(level_idx + 1, seg['seg_idx'], s_idx, e_idx)
        return segs

    n = len(ts_5s)
    root_segs = _recurse(0, -1, 0, n - 1)

    day_dt = datetime.fromtimestamp(int(ts_5s[0]), tz=timezone.utc)
    theme = {
        'day': day,
        'date_iso': day_dt.strftime('%Y-%m-%d'),
        'dow': day_dt.strftime('%a'),
        'n_5s_bars': int(n),
        'session_range_pts': float(close_5s.max() - close_5s.min()),
        'session_net_pts': float(close_5s[-1] - close_5s[0]),
        'levels': [{'tf': tf, 'min_duration_min': mm,
                    'level_name': LEVEL_NAMES.get(tf, tf)} for tf, mm in levels],
    }

    # Count segments per level
    counts = {LEVEL_NAMES.get(tf, tf): 0 for tf, _ in levels}

    def _count(segs, depth):
        if depth >= len(levels):
            return
        ln = LEVEL_NAMES.get(levels[depth][0], levels[depth][0])
        counts[ln] += len(segs)
        for s in segs:
            _count(s.get('_children', []), depth + 1)
    _count(root_segs, 0)
    theme['counts'] = counts

    return {'theme': theme, 'segments': root_segs}


def _flatten_to_rows(hierarchy: dict, day: str) -> dict[str, list[dict]]:
    """Flatten nested tree into per-level row lists, keeping parent links."""
    out = {ln: [] for ln in [
        l['level_name'] for l in hierarchy['theme']['levels']]}

    def _walk(segs, depth, parent_path):
        if depth >= len(hierarchy['theme']['levels']):
            return
        ln = hierarchy['theme']['levels'][depth]['level_name']
        for s in segs:
            row = {k: v for k, v in s.items() if k != '_children'}
            row['day'] = day
            row['parent_path'] = parent_path
            out[ln].append(row)
            new_parent = f'{parent_path}/{s["seg_idx"]}'
            _walk(s.get('_children', []), depth + 1, new_parent)
    _walk(hierarchy['segments'], 0, '')
    return out


def render_chart(day: str, hierarchy: dict, out_path: str):
    df_5s = _load_5s(day)
    if df_5s.empty or not hierarchy:
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    dt_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]
    close_5s = df_5s['close'].values

    # Levels with TFs
    levels_info = hierarchy['theme']['levels']
    n_levels = len(levels_info)

    # Recompute TF M_close lines for chart overlay
    tf_lines = {}
    for li in levels_info:
        tf = li['tf']
        oh = _load_tf_ohlcv(tf, day)
        if oh.empty: continue
        N = TF_WINDOW[tf]
        M = oh['close'].rolling(N, min_periods=2).mean().values
        tf_ts = oh['timestamp'].values.astype(np.int64)
        tf_lines[tf] = _ffill_to_5s(M, tf_ts, ts_5s, PERIOD_S[tf])

    fig, ax = plt.subplots(1, 1, figsize=(22, 10))
    ax.plot(dt_5s, close_5s, color='black', lw=0.5, alpha=0.85, label='5s close')

    LINE_COLORS = {'15m': '#1E88E5', '5m': '#FB8C00',
                   '1m':  '#43A047', '15s': '#8E24AA'}
    for li in levels_info:
        tf = li['tf']
        if tf in tf_lines:
            ax.plot(dt_5s, tf_lines[tf], color=LINE_COLORS.get(tf, '#999'),
                    lw=1.0, alpha=0.85, label=f'{tf} M_close ({li["level_name"]})')

    # Render segment boundaries with diminishing opacity per level
    BAND_COLORS = {'15m': '#E3F2FD', '5m': '#FFF3E0',
                   '1m':  '#E8F5E9', '15s': '#F3E5F5'}
    LEVEL_ALPHA = {'phrase': 0.30, 'motif': 0.25,
                   'sub_motif': 0.18, 'measure': 0.10}

    def _render_segs(segs, depth, alpha_factor=1.0):
        if depth >= n_levels:
            return
        tf = levels_info[depth]['tf']
        ln = levels_info[depth]['level_name']
        col = LINE_COLORS.get(tf, '#999999')
        lw = max(0.3, 1.6 - depth * 0.4)
        for s in segs:
            s_dt = datetime.fromtimestamp(s['start_ts'], tz=timezone.utc)
            e_dt = datetime.fromtimestamp(s['end_ts'], tz=timezone.utc)
            if depth == 0:
                # Phrase: full background tint
                ax.axvspan(s_dt, e_dt, color=BAND_COLORS.get(tf, '#EEE'),
                           alpha=0.35, zorder=0)
            ax.axvline(s_dt, color=col, lw=lw, alpha=0.6 * alpha_factor,
                       linestyle='-' if depth == 0 else (':' if depth >= 2 else '--'))
            # Recurse into children
            _render_segs(s.get('_children', []), depth + 1, alpha_factor * 0.7)

    _render_segs(hierarchy['segments'], 0)

    th = hierarchy['theme']
    counts_str = ', '.join(f'{ln}={n}' for ln, n in th['counts'].items())
    ax.set_title(
        f'{day}  ({th["dow"]})  range={th["session_range_pts"]:.1f}pts  '
        f'net={th["session_net_pts"]:+.1f}\n'
        f'Recursive segmentation: {counts_str}\n'
        f'Solid = phrase boundary, dashed = motif, dotted = sub-motif/measure',
        fontsize=11)
    ax.set_ylabel('price')
    ax.set_xlabel('time (UTC)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--levels', default='15m:30,5m:5,1m:1,15s:0.25',
                    help='comma-sep list of tf:min_duration_min, ordered coarse->fine')
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--out-chart', default=None)
    args = ap.parse_args()

    levels = []
    for spec in args.levels.split(','):
        tf, mm = spec.split(':')
        levels.append((tf.strip(), float(mm)))

    print(f'Recursive segmentation of {args.day}')
    for tf, mm in levels:
        print(f'  {LEVEL_NAMES.get(tf, tf):<12s} ({tf:<4s}, min={mm}min)')

    h = segment_day_recursive(args.day, levels)
    if not h:
        print(f'No data for {args.day}'); sys.exit(1)

    print('\nCounts per level:')
    for ln, n in h['theme']['counts'].items():
        print(f'  {ln:<12s}: {n}')

    out_json = args.out_json or f'reports/findings/segments/recursive/{args.day}.json'
    out_chart = args.out_chart or f'chart/segments/recursive/{args.day}.png'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(h, f, indent=2)
    print(f'\nJSON  -> {out_json}')

    render_chart(args.day, h, out_chart)
    print(f'Chart -> {out_chart}')


if __name__ == '__main__':
    main()
