"""Phrase -> motif drill-down: see what 5m motifs played inside each phrase.

For a chosen phrase shape (e.g., LINEAR_DOWN), pick representative phrases
spanning the chord-fingerprint variation axis and render a multi-panel chart.
Each panel zooms into ONE phrase and shows:

    - 5s close (the actual price tape)
    - 15m M_close (the phrase anchor — the 'most stable line')
    - 5m M_close (the motif anchor — what we drill down into)
    - 5m motif boundaries with shape_class labels
    - Per-motif annotations: shape_class, length_min, slope_pts_per_min, r2adj

This is Step 2 of the music hierarchy made visible: the variations of a
phrase shape AS sequences of motifs.

USAGE
    python tools/phrase_drill_down.py --shape LINEAR_DOWN
    python tools/phrase_drill_down.py --shape LOGARITHMIC_UP --variation-axis r2adj_5m__mean
    python tools/phrase_drill_down.py --shape LINEAR_DOWN --n 9
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

from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s, TF_WINDOW, PERIOD_S
)


# Color palette for motif shape classes
SHAPE_COLOR = {
    'LINEAR_UP':           '#43A047',
    'LINEAR_DOWN':         '#E53935',
    'EXPONENTIAL_UP':      '#7CB342',
    'EXPONENTIAL_DOWN':    '#EF5350',
    'LOGARITHMIC_UP':      '#26A69A',
    'LOGARITHMIC_DOWN':    '#D81B60',
    'STEP_UP':             '#3949AB',
    'STEP_DOWN':           '#5E35B1',
    'ROUNDED_U_UP':        '#00897B',
    'ROUNDED_U_DOWN':      '#C62828',
    'SYMMETRIC_V_UP':      '#1E88E5',
    'SYMMETRIC_V_DOWN':    '#FB8C00',
    'BACK_SKEWED_UP':      '#8D6E63',
    'BACK_SKEWED_DOWN':    '#6D4C41',
    'FRONT_SKEWED_UP':     '#7986CB',
    'FRONT_SKEWED_DOWN':   '#A1887F',
    'EXPAND_OSCILLATOR':   '#9E9E9E',
    'DAMPED_OSCILLATOR':   '#757575',
    'SINE_WAVE':           '#BDBDBD',
    'FLATLINE_UP':         '#CFD8DC',
    'FLATLINE_DOWN':       '#B0BEC5',
    'NOISE':               '#FFB300',
}


def render_phrase_panel(ax, phrase: pd.Series, motifs_in_phrase: pd.DataFrame,
                        window_pad_min: float = 5.0):
    day = phrase['day']
    df_5s = _load_5s(day)
    if df_5s.empty:
        ax.text(0.5, 0.5, f'no data: {day}', transform=ax.transAxes,
                ha='center', va='center')
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values

    pad = int(window_pad_min * 60)
    t_start = int(phrase['start_ts']) - pad
    t_end   = int(phrase['end_ts'])   + pad
    m = (ts_5s >= t_start) & (ts_5s <= t_end)
    if not m.any():
        ax.text(0.5, 0.5, 'window empty', transform=ax.transAxes,
                ha='center', va='center')
        return
    ts_w = ts_5s[m]
    cl_w = close_5s[m]
    dt_w = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_w]

    ax.plot(dt_w, cl_w, color='black', lw=0.5, alpha=0.85, label='5s close')

    # 15m M_close (phrase anchor)
    oh_15m = _load_tf_ohlcv('15m', day)
    if not oh_15m.empty:
        N = TF_WINDOW['15m']
        M = oh_15m['close'].rolling(N, min_periods=2).mean().values
        tf_ts = oh_15m['timestamp'].values.astype(np.int64)
        M5s = _ffill_to_5s(M, tf_ts, ts_5s, PERIOD_S['15m'])[m]
        ax.plot(dt_w, M5s, color='#1E88E5', lw=1.6, alpha=0.85,
                label='15m M (phrase anchor)')

    # 5m M_close (motif anchor)
    oh_5m = _load_tf_ohlcv('5m', day)
    if not oh_5m.empty:
        N = TF_WINDOW['5m']
        M = oh_5m['close'].rolling(N, min_periods=2).mean().values
        tf_ts = oh_5m['timestamp'].values.astype(np.int64)
        M5s = _ffill_to_5s(M, tf_ts, ts_5s, PERIOD_S['5m'])[m]
        ax.plot(dt_w, M5s, color='#FB8C00', lw=1.0, alpha=0.85,
                label='5m M (motif anchor)')

    # Phrase span (full phrase, not just window)
    p_s_dt = datetime.fromtimestamp(int(phrase['start_ts']), tz=timezone.utc)
    p_e_dt = datetime.fromtimestamp(int(phrase['end_ts']),   tz=timezone.utc)
    ax.axvspan(p_s_dt, p_e_dt, color='#1E88E5', alpha=0.06, zorder=0)
    ax.axvline(p_s_dt, color='#1E88E5', lw=1.4, alpha=0.7, linestyle='-')
    ax.axvline(p_e_dt, color='#1E88E5', lw=1.4, alpha=0.5, linestyle='--')

    # Motifs inside the phrase: colored bands by motif shape
    for _, mot in motifs_in_phrase.iterrows():
        s_dt = datetime.fromtimestamp(int(mot['start_ts']), tz=timezone.utc)
        e_dt = datetime.fromtimestamp(int(mot['end_ts']),   tz=timezone.utc)
        col = SHAPE_COLOR.get(mot['shape_class'], '#999999')
        ax.axvspan(s_dt, e_dt, color=col, alpha=0.18, zorder=1)
        ax.axvline(s_dt, color=col, lw=0.6, alpha=0.65, linestyle=':')
        # Tag in middle of motif
        mid_ts = (mot['start_ts'] + mot['end_ts']) / 2
        mid_dt = datetime.fromtimestamp(int(mid_ts), tz=timezone.utc)
        y_top = ax.get_ylim()[1]
        tag = (f'{mot["shape_class"]}\n'
               f'{mot["length_min"]:.0f}m s={mot["slope_pts_per_min"]:+.2f}\n'
               f'r2={mot["r2adj"]:.2f}')
        ax.text(mid_dt, y_top, tag, fontsize=6, color=col,
                va='top', ha='center', alpha=0.95,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.85, edgecolor=col, linewidth=0.6))

    # Phrase title with chord fingerprint stats
    title = (
        f'PHRASE  {day}  P{int(phrase["seg_idx"])}  '
        f'{p_s_dt.strftime("%H:%M")}-{p_e_dt.strftime("%H:%M")}  '
        f'{phrase["length_min"]:.0f}min  '
        f'shape={phrase["shape_class"]} (r={phrase["shape_pearson_r"]:+.2f})\n'
        f'segment slope={phrase["slope_pts_per_min"]:+.3f}/min  '
        f'pk_z={phrase["peak_abs_z"]:.1f}  '
        f'ride={phrase["ride_pnl_pts"]:+.1f}\n'
        f'CHORD: slope_15m_at_bar mean={phrase.get("slope_15m__mean", float("nan")):+.4f}/'
        f'std={phrase.get("slope_15m__std", float("nan")):.4f}  '
        f'r2adj_5m_at_bar mean={phrase.get("r2adj_5m__mean", float("nan")):.2f}  '
        f'sigma_rank mean={phrase.get("sigma_rank_15m__mean", float("nan")):.2f}\n'
        f'MOTIFS INSIDE: {len(motifs_in_phrase)}'
    )
    ax.set_title(title, fontsize=7.5)
    ax.legend(loc='upper left', fontsize=6)
    ax.grid(True, alpha=0.20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
                    default='reports/findings/segments/all_motifs_labeled_with_chord.csv')
    ap.add_argument('--motif-csv',
                    default='reports/findings/segments/all_melodies_labeled_with_chord.csv')
    ap.add_argument('--shape', default='LINEAR_DOWN',
                    help='Phrase shape_class to drill into')
    ap.add_argument('--variation-axis', default='slope_15m__std',
                    help='Chord-fingerprint stat to sample along (quartile spread)')
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    phrases = pd.read_csv(args.phrase_csv)
    motifs = pd.read_csv(args.motif_csv)
    if args.split != 'BOTH':
        phrases = phrases[phrases['split'] == args.split]

    target = phrases[phrases['shape_class'] == args.shape].copy()
    target = target.dropna(subset=[args.variation_axis])
    if target.empty:
        print(f'No phrases with shape_class={args.shape}')
        sys.exit(1)
    print(f'{len(target)} {args.shape} phrases ({args.split}) with valid '
          f'{args.variation_axis}')

    # Sample N phrases spread across variation axis
    # Pick by quantile slots so we cover the full distribution
    quantiles = np.linspace(0.05, 0.95, args.n)
    targets_sorted = target.sort_values(args.variation_axis).reset_index(drop=True)
    pick_idx = (quantiles * (len(targets_sorted) - 1)).round().astype(int)
    picks = targets_sorted.iloc[pick_idx].drop_duplicates(subset=['day','seg_idx'])
    picks = picks.head(args.n)
    print(f'Picked {len(picks)} phrases spanning '
          f'{args.variation_axis} q5 ({picks[args.variation_axis].iloc[0]:.4f}) '
          f'-> q95 ({picks[args.variation_axis].iloc[-1]:.4f})')

    n = len(picks)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 11, rows * 4.5),
                              squeeze=False)

    for i, (_, phrase) in enumerate(picks.iterrows()):
        ax = axes[i // cols][i % cols]
        # Find motifs that belong to this phrase: same day + parent_motif_idx == phrase.seg_idx
        in_phrase = motifs[(motifs['day'] == phrase['day']) &
                           (motifs['parent_motif_idx'] == phrase['seg_idx'])]
        in_phrase = in_phrase.sort_values('start_ts').reset_index(drop=True)
        render_phrase_panel(ax, phrase, in_phrase)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f'Phrase drill-down: {args.shape} variations sampled by {args.variation_axis} '
        f'({args.split}, n_total={len(target)})\n'
        f'Each panel = one phrase showing the 5m motifs that played inside it.  '
        f'Blue band = phrase span. Colored bands = motif spans (color by motif shape).',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.out or f'chart/segments/phrase_drill_{args.shape}_{args.variation_axis}.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_path}')


if __name__ == '__main__':
    main()
