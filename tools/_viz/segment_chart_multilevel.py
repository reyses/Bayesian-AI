"""Render multi-level hierarchical segmentation charts for sample days.

Produces a 5-panel stacked chart per day showing the 5s close with
boundary lines + shape labels at each level (phrase -> note).

USAGE
    python tools/segment_chart_multilevel.py --day 2026_02_12
    python tools/segment_chart_multilevel.py --days 2026_02_12,2025_05_20,2025_01_08
    python tools/segment_chart_multilevel.py --random 5  # 5 random days
    python tools/segment_chart_multilevel.py --extremes  # most/least phrases
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_simple_shapes import segment_day_simple, LEVELS
from tools.segment_day_motif_melody import _load_5s


SHAPE_COLORS = {
    'STEEP_LINEAR_UP':    '#43A047',
    'GENTLE_LINEAR_UP':   '#7CB342',
    'STEEP_LINEAR_DOWN':  '#E53935',
    'GENTLE_LINEAR_DOWN': '#EF5350',
    'STEEP_CONVEX_UP':    '#0D47A1',
    'GENTLE_CONVEX_UP':   '#1E88E5',
    'STEEP_CONCAVE_UP':   '#26A69A',
    'GENTLE_CONCAVE_UP':  '#80DEEA',
    'STEEP_CONCAVE_DOWN': '#8E24AA',
    'GENTLE_CONCAVE_DOWN':'#CE93D8',
    'STEEP_CONVEX_DOWN':  '#6D4C41',
    'GENTLE_CONVEX_DOWN': '#A1887F',
    'FLATLINE':           '#FFB300',
    'NOISE':              '#9E9E9E',
}


def collect_segments_at_level(phrases: list, level_idx: int) -> list:
    """Walk the hierarchy and collect all segments at a given depth.
    level_idx 0 = phrases, 1 = motifs, ..., 4 = notes."""
    if level_idx == 0:
        return phrases
    _, child_attr, _, _, _ = LEVELS[level_idx]
    out = []
    parents = collect_segments_at_level(phrases, level_idx - 1)
    for p in parents:
        out.extend(p.get(child_attr, []))
    return out


def render_multilevel(day: str, h: dict, out_path: str):
    df_5s = _load_5s(day)
    if df_5s.empty or not h:
        return
    ts = df_5s['timestamp'].values.astype(np.int64)
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    close = df_5s['close'].values

    fig, axes = plt.subplots(5, 1, figsize=(22, 18), sharex=True)
    levels_meta = [
        ('phrase',    '15m PHRASES'),
        ('motif',     '5m MOTIFS'),
        ('sub_motif', '1m SUB_MOTIFS'),
        ('measure',   '15s MEASURES'),
        ('note',      '5s NOTES'),
    ]

    for li, (lvl, lvl_title) in enumerate(levels_meta):
        ax = axes[li]
        ax.plot(dt, close, color='black', lw=0.4, alpha=0.7)
        segs = collect_segments_at_level(h['phrases'], li)
        for s_idx, seg in enumerate(segs):
            s_dt = datetime.fromtimestamp(int(seg['start_ts']), tz=timezone.utc)
            e_dt = datetime.fromtimestamp(int(seg['end_ts']), tz=timezone.utc)
            col = SHAPE_COLORS.get(seg['shape'], '#444')
            ax.axvspan(s_dt, e_dt, color=col, alpha=0.15, zorder=0)
            ax.axvline(s_dt, color=col, lw=0.7, alpha=0.55)
            # only label at coarser levels (phrase + motif + sub_motif)
            if li <= 2 and seg.get('length_min', 0) >= 1.0:
                shape_short = seg['shape'].replace('GENTLE_', 'g').replace(
                    'STEEP_', 's').replace('LINEAR_', 'L').replace(
                    'CONVEX_', 'X').replace('CONCAVE_', 'C')
                ymax = ax.get_ylim()[1]
                ax.text(s_dt, ymax, f' {shape_short}', fontsize=6,
                          color='#222', va='top', ha='left', rotation=0,
                          bbox=dict(boxstyle='round,pad=0.1',
                                     facecolor='white', alpha=0.6,
                                     edgecolor='none'))
        if h['phrases']:
            last_dt = datetime.fromtimestamp(
                int(h['phrases'][-1]['end_ts']), tz=timezone.utc)
            ax.axvline(last_dt, color='black', lw=0.5, alpha=0.5)
        ax.set_title(f'{lvl_title}  n={len(segs)}', fontsize=11)
        ax.grid(True, alpha=0.20)
        ax.set_ylabel('price')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    fig.suptitle(f'{day} — 5-LEVEL hierarchical segmentation '
                  f'(threshold r >= {h["threshold"]})\n'
                  f'phrases={h.get("n_phrases",0)}  motifs={h.get("n_motifs",0)}  '
                  f'sub_motifs={h.get("n_sub_motifs",0)}  '
                  f'measures={h.get("n_measures",0)}  notes={h.get("n_notes",0)}',
                  fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Chart -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None)
    ap.add_argument('--days', default=None,
                     help='comma-separated day list')
    ap.add_argument('--random', type=int, default=0,
                     help='render N random days')
    ap.add_argument('--extremes', action='store_true',
                     help='render top 3 most-phrase + top 3 fewest-phrase days')
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--out-dir', default='chart/segments/simple/multilevel')
    args = ap.parse_args()

    days_to_render = []
    if args.day:
        days_to_render.append(args.day)
    if args.days:
        days_to_render.extend([d.strip() for d in args.days.split(',')])
    if args.random > 0:
        all_days = ([os.path.basename(p).replace('.parquet', '')
                      for p in glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2025_*.parquet')]
                     + [os.path.basename(p).replace('.parquet', '')
                        for p in glob.glob('DATA/ATLAS/FEATURES_5s_v2/L0/2026_*.parquet')])
        random.seed(0)
        days_to_render.extend(random.sample(all_days, min(args.random, len(all_days))))
    if args.extremes:
        themes_csv = 'reports/findings/segments/simple_bulk_v2/themes.csv'
        if os.path.exists(themes_csv):
            t = pd.read_csv(themes_csv)
            most = t.nlargest(3, 'n_phrases')['day'].tolist()
            least = t.nsmallest(3, 'n_phrases')['day'].tolist()
            days_to_render.extend(most + least)

    days_to_render = list(dict.fromkeys(days_to_render))  # dedup keep order
    print(f'Rendering {len(days_to_render)} days')
    os.makedirs(args.out_dir, exist_ok=True)
    for day in days_to_render:
        # Try to read cached JSON first
        json_path = f'reports/findings/segments/simple_bulk_v2/per_day/{day}.json'
        if os.path.exists(json_path):
            with open(json_path) as f:
                h = json.load(f)
        else:
            h = segment_day_simple(day, threshold=args.threshold)
        if not h:
            print(f'  SKIP {day} (no data)'); continue
        out = os.path.join(args.out_dir, f'multilevel_{day}.png')
        render_multilevel(day, h, out)


if __name__ == '__main__':
    main()
