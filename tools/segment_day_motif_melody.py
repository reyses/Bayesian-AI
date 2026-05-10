"""Hierarchical day segmentation: theme -> motif -> melody -> chord.

Vocabulary (per memory/feedback_no_human_regime_terms.md):
    theme    = day-level aggregate (single record per day)
    motif    = macro segment defined by 15m-CRM inflections; ~30min-3hr
    melody   = micro sub-segment NESTED inside a motif, defined by 5m-CRM
                inflections; ~5-30min
    chord    = at-bar feature vector (NOT computed here; this tool emits
                segment_chords which are EDA evaluated over a segment)

Pipeline
--------
For each day:
  1. Load 5s OHLCV + 15m OHLCV + 5m OHLCV
  2. Compute 15m M_close, SE_close; forward-fill onto 5s grid
  3. Detect motif boundaries via inflections of 15m M_close
     - Merge motifs shorter than `min_motif_min` into the longer neighbor
  4. Within each motif, repeat with 5m M_close to detect melodies
     - Merge melodies shorter than `min_melody_min`
  5. Compute segment_chord for each motif AND each melody:
        slope_pts_per_min   linear slope across segment using anchor TF mean
        mean_sigma           mean SE_close during segment
        sigma_rank_mid       rolling-60min percentile of SE_close at segment midpoint
        r2adj                R^2_adjusted of linear fit to 5s closes over segment
        shape_class          best match from SeedPrimitiveLibrary on segment 5s closes
        shape_pearson_r      correlation with that shape
        length_min           minutes
        peak_abs_z           max |(5s_close - M_anchor) / SE_anchor| during segment
        tod_start_hour_utc   hour-of-day at segment start
        net_move_pts         5s_close[end] - 5s_close[start]
  6. Emit JSON with hierarchy: {theme: {...}, motifs: [{..., melodies: [...]}, ...]}
  7. Render visualization: 5s close + 15m CRM + motif boundaries +
                             melody boundaries within motifs

USAGE
    python tools/segment_day_motif_melody.py --day 2026_02_12
    python tools/segment_day_motif_melody.py --day 2026_03_03 --min-motif-min 20
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

# Reuse existing shape-classification machinery
from tools.research.seeds import SeedPrimitiveLibrary


TF_WINDOW = {'1m': 15, '5m': 9, '15m': 12, '1h': 12}
PERIOD_S  = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
RANK_WINDOW_5S_BARS = 720  # 60min for sigma rank percentile


@dataclass
class Segment:
    level: str        # 'motif' or 'melody'
    parent_motif_idx: int  # -1 for motifs, parent index for melodies
    seg_idx: int      # index within parent (or within day for motifs)
    start_ts: int
    end_ts: int
    start_iso: str
    end_iso: str
    length_min: float
    # segment chord (EDA)
    slope_pts_per_min: float
    mean_sigma: float
    sigma_rank_mid: float
    r2adj: float
    shape_class: str
    shape_pearson_r: float
    peak_abs_z: float
    tod_start_hour_utc: int
    net_move_pts: float


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


def _ffill_to_5s(values: np.ndarray, src_ts: np.ndarray, target_ts: np.ndarray,
                 period_s: int) -> np.ndarray:
    """Lookahead-clean ffill: target uses only TF bars completed at-or-before t."""
    target = target_ts - period_s
    idx = np.searchsorted(src_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(src_ts) - 1)
    return values[idx]


def _detect_inflections_5s(M_5s: np.ndarray, ts_5s: np.ndarray,
                          period_s: int) -> list[int]:
    """Return 5s-bar indices of inflection points in a TF mean (forward-filled).

    Since M_5s is step-function (only changes at TF bar closes), an inflection
    is where the SIGN of (M[next_TF_bar] - M[current_TF_bar]) flips. We walk the
    distinct step values and detect sign changes between consecutive STEPS.
    """
    inflections = [0]  # always include start
    # Find indices where M_5s changes value (TF-bar boundaries)
    change_mask = np.concatenate([[False], M_5s[1:] != M_5s[:-1]])
    change_idx = np.where(change_mask)[0]
    if len(change_idx) < 2:
        inflections.append(len(M_5s) - 1)
        return inflections
    step_values = M_5s[change_idx]
    diffs = np.diff(step_values)  # step-to-step diffs
    for k in range(1, len(diffs)):
        # Sign flip between consecutive steps = inflection at this step boundary
        if diffs[k] * diffs[k - 1] < 0:
            inflections.append(int(change_idx[k]))
    inflections.append(len(M_5s) - 1)
    return inflections


def _merge_short_segments(boundaries: list[int], ts_5s: np.ndarray,
                          min_duration_s: int) -> list[int]:
    """Drop boundary points that produce sub-min segments by merging into neighbor."""
    if len(boundaries) <= 2:
        return boundaries
    keep = [boundaries[0]]
    for b in boundaries[1:-1]:
        prev_keep = keep[-1]
        if (ts_5s[b] - ts_5s[prev_keep]) >= min_duration_s:
            keep.append(b)
    keep.append(boundaries[-1])
    # Also enforce that the last segment is long enough; if not, drop the
    # last interior boundary
    if len(keep) >= 3:
        if (ts_5s[keep[-1]] - ts_5s[keep[-2]]) < min_duration_s:
            keep.pop(-2)
    return keep


def _rolling_r2_adjusted(y: np.ndarray) -> float:
    """R^2_adjusted of a single linear regression fit. Returns NaN if too small."""
    n = len(y)
    if n < 4:
        return float('nan')
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    Sxx = float(((x - x_mean) ** 2).sum())
    if Sxx == 0:
        return float('nan')
    y_mean = float(y.mean())
    SS_tot = float(((y - y_mean) ** 2).sum())
    if SS_tot <= 0:
        return float('nan')
    Sxy = float(((x - x_mean) * (y - y_mean)).sum())
    b = Sxy / Sxx
    SS_res = SS_tot - b * Sxy
    if SS_res < 0:
        SS_res = 0.0
    r2 = 1.0 - SS_res / SS_tot
    if n - 2 <= 0:
        return float('nan')
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - 2))


def _classify_shape(values: np.ndarray, lib: SeedPrimitiveLibrary,
                    mean_sigma: float = None,
                    flat_threshold_sigma: float = 1.0) -> tuple[str, float]:
    """Classify segment shape with a magnitude gate.

    Pearson correlation is shape-only / scale-invariant: a line drifting 3pts
    on a 22000 base correlates near +1 with LINEAR_DOWN even though it's flat.
    To guard against false-directional classification of flat segments, we
    compare the segment's range to its rolling SE_close. If
        (max - min) < flat_threshold_sigma * mean_sigma
    the segment is below band-width amplitude — classify as FLATLINE
    regardless of Pearson match.
    """
    n = len(values)
    if n < 4:
        return 'NOISE', 0.0

    # Magnitude gate
    if mean_sigma is not None and np.isfinite(mean_sigma) and mean_sigma > 0:
        finite = values[np.isfinite(values)]
        if len(finite) >= 2:
            seg_range = float(finite.max() - finite.min())
            if seg_range < flat_threshold_sigma * mean_sigma:
                return 'FLATLINE', 0.0

    # Resample to lib.N evenly-spaced points and run Pearson classification
    src_x = np.linspace(0, 1, n)
    tgt_x = np.linspace(0, 1, lib.N)
    resampled = np.interp(tgt_x, src_x, values)
    return lib.classify_trajectory(resampled)


def _build_segment(level: str, parent_motif_idx: int, seg_idx: int,
                   start_i: int, end_i: int,
                   ts_5s: np.ndarray, close_5s: np.ndarray,
                   M_5s: np.ndarray, S_5s: np.ndarray,
                   sigma_rank_5s: np.ndarray,
                   shape_lib: SeedPrimitiveLibrary) -> Segment:
    """Build a Segment record with full chord EDA for [start_i, end_i)."""
    s_ts = int(ts_5s[start_i])
    e_ts = int(ts_5s[end_i])
    length_min = (e_ts - s_ts) / 60.0

    # slope across segment using TF mean values
    slope_pts_per_min = ((M_5s[end_i] - M_5s[start_i]) / length_min) if length_min > 0 else 0.0

    # mean sigma across segment (TF-anchor SE)
    seg_S = S_5s[start_i:end_i + 1]
    mean_sigma = float(np.nanmean(seg_S)) if seg_S.size else float('nan')

    # sigma rank at midpoint
    mid = (start_i + end_i) // 2
    sigma_rank_mid = float(sigma_rank_5s[mid]) if np.isfinite(sigma_rank_5s[mid]) else float('nan')

    # R^2_adj of linear fit to 5s closes over segment
    closes = close_5s[start_i:end_i + 1]
    r2adj = _rolling_r2_adjusted(closes)

    # peak |z|
    with np.errstate(divide='ignore', invalid='ignore'):
        z_close = (closes - M_5s[start_i:end_i + 1]) / S_5s[start_i:end_i + 1]
    peak_abs_z = float(np.nanmax(np.abs(z_close))) if np.any(np.isfinite(z_close)) else float('nan')

    # net move
    net_move_pts = float(close_5s[end_i] - close_5s[start_i])

    # Shape classification — match against the TF M_close line (the smoothed
    # macro structure), NOT the raw 5s tape. Magnitude gate: if segment range
    # is below 1 mean SE_close, return FLATLINE regardless of Pearson match
    # (Pearson is scale-invariant; a tiny linear drift still gets r=+0.96
    # against LINEAR_DOWN/UP — that's a calibration bug, not a real shape).
    shape_input = M_5s[start_i:end_i + 1]
    shape_class, shape_r = _classify_shape(shape_input, shape_lib,
                                           mean_sigma=mean_sigma,
                                           flat_threshold_sigma=1.0)

    s_dt = datetime.fromtimestamp(s_ts, tz=timezone.utc)
    e_dt = datetime.fromtimestamp(e_ts, tz=timezone.utc)

    return Segment(
        level=level,
        parent_motif_idx=parent_motif_idx,
        seg_idx=seg_idx,
        start_ts=s_ts,
        end_ts=e_ts,
        start_iso=s_dt.isoformat(),
        end_iso=e_dt.isoformat(),
        length_min=round(length_min, 2),
        slope_pts_per_min=round(slope_pts_per_min, 4),
        mean_sigma=round(mean_sigma, 3),
        sigma_rank_mid=round(sigma_rank_mid, 3),
        r2adj=round(r2adj, 3) if np.isfinite(r2adj) else float('nan'),
        shape_class=shape_class,
        shape_pearson_r=round(shape_r, 3),
        peak_abs_z=round(peak_abs_z, 3) if np.isfinite(peak_abs_z) else float('nan'),
        tod_start_hour_utc=s_dt.hour,
        net_move_pts=round(net_move_pts, 2),
    )


def segment_day(day: str, motif_tf: str = '15m', melody_tf: str = '5m',
                min_motif_min: float = 30.0, min_melody_min: float = 5.0) -> dict:
    df_5s = _load_5s(day)
    if df_5s.empty:
        return {}
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values.astype(np.float64)

    # Load anchor TF OHLCVs and compute mean / sigma
    motif_oh = _load_tf_ohlcv(motif_tf, day)
    melody_oh = _load_tf_ohlcv(melody_tf, day)
    if motif_oh.empty or melody_oh.empty:
        return {}

    Nm = TF_WINDOW[motif_tf]
    Nl = TF_WINDOW[melody_tf]
    motif_oh['M'] = motif_oh['close'].rolling(Nm, min_periods=2).mean()
    motif_oh['S'] = motif_oh['close'].rolling(Nm, min_periods=2).std()
    melody_oh['M'] = melody_oh['close'].rolling(Nl, min_periods=2).mean()
    melody_oh['S'] = melody_oh['close'].rolling(Nl, min_periods=2).std()

    # Forward-fill onto 5s grid (with TF lookback for no lookahead)
    motif_ts = motif_oh['timestamp'].values.astype(np.int64)
    melody_ts = melody_oh['timestamp'].values.astype(np.int64)
    Mm = _ffill_to_5s(motif_oh['M'].values, motif_ts, ts_5s, PERIOD_S[motif_tf])
    Sm = _ffill_to_5s(motif_oh['S'].values, motif_ts, ts_5s, PERIOD_S[motif_tf])
    Ml = _ffill_to_5s(melody_oh['M'].values, melody_ts, ts_5s, PERIOD_S[melody_tf])
    Sl = _ffill_to_5s(melody_oh['S'].values, melody_ts, ts_5s, PERIOD_S[melody_tf])

    # Sigma ranks (rolling 60min percentile)
    sigma_rank_motif = (pd.Series(Sm).rolling(RANK_WINDOW_5S_BARS, min_periods=20)
                          .rank(pct=True).values)
    sigma_rank_melody = (pd.Series(Sl).rolling(RANK_WINDOW_5S_BARS, min_periods=20)
                           .rank(pct=True).values)

    shape_lib = SeedPrimitiveLibrary(N=16)

    # ── Motif boundaries from 15m-CRM inflections ──
    motif_inflect = _detect_inflections_5s(Mm, ts_5s, PERIOD_S[motif_tf])
    motif_inflect = _merge_short_segments(motif_inflect, ts_5s, int(min_motif_min * 60))

    motifs = []
    for i in range(len(motif_inflect) - 1):
        start_i = motif_inflect[i]
        end_i = motif_inflect[i + 1]
        seg = _build_segment('motif', -1, i,
                             start_i, end_i,
                             ts_5s, close_5s, Mm, Sm, sigma_rank_motif,
                             shape_lib)
        # ── Melody boundaries WITHIN this motif from 5m-CRM ──
        melody_inflect = _detect_inflections_5s(
            Ml[start_i:end_i + 1], ts_5s[start_i:end_i + 1], PERIOD_S[melody_tf])
        # offset back to global indices
        melody_inflect = [start_i + j for j in melody_inflect]
        melody_inflect = _merge_short_segments(melody_inflect, ts_5s,
                                              int(min_melody_min * 60))
        melodies = []
        for j in range(len(melody_inflect) - 1):
            ms_i = melody_inflect[j]
            me_i = melody_inflect[j + 1]
            mseg = _build_segment('melody', i, j,
                                  ms_i, me_i,
                                  ts_5s, close_5s, Ml, Sl, sigma_rank_melody,
                                  shape_lib)
            melodies.append(asdict(mseg))
        motif_dict = asdict(seg)
        motif_dict['melodies'] = melodies
        motifs.append(motif_dict)

    # Theme = day-level aggregate
    day_dt = datetime.fromtimestamp(int(ts_5s[0]), tz=timezone.utc)
    theme = {
        'day': day,
        'date_iso': day_dt.strftime('%Y-%m-%d'),
        'dow': day_dt.strftime('%a'),
        'n_5s_bars': int(len(ts_5s)),
        'session_range_pts': float(close_5s.max() - close_5s.min()),
        'session_net_pts': float(close_5s[-1] - close_5s[0]),
        'session_efficiency': float((close_5s[-1] - close_5s[0]) /
                                     (close_5s.max() - close_5s.min())
                                     if close_5s.max() != close_5s.min() else 0.0),
        'n_motifs': len(motifs),
        'n_melodies': sum(len(m['melodies']) for m in motifs),
        'min_motif_min': min_motif_min,
        'min_melody_min': min_melody_min,
        'motif_tf': motif_tf,
        'melody_tf': melody_tf,
    }

    return {'theme': theme, 'motifs': motifs}


def render_chart(day: str, hierarchy: dict, out_path: str):
    df_5s = _load_5s(day)
    if df_5s.empty or not hierarchy:
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    dt_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]
    close_5s = df_5s['close'].values

    # Reconstruct M lines for overlay
    motif_tf = hierarchy['theme']['motif_tf']
    melody_tf = hierarchy['theme']['melody_tf']
    motif_oh = _load_tf_ohlcv(motif_tf, day)
    melody_oh = _load_tf_ohlcv(melody_tf, day)
    Nm = TF_WINDOW[motif_tf]
    Nl = TF_WINDOW[melody_tf]
    motif_oh['M'] = motif_oh['close'].rolling(Nm, min_periods=2).mean()
    melody_oh['M'] = melody_oh['close'].rolling(Nl, min_periods=2).mean()
    motif_ts = motif_oh['timestamp'].values.astype(np.int64)
    melody_ts = melody_oh['timestamp'].values.astype(np.int64)
    Mm = _ffill_to_5s(motif_oh['M'].values, motif_ts, ts_5s, PERIOD_S[motif_tf])
    Ml = _ffill_to_5s(melody_oh['M'].values, melody_ts, ts_5s, PERIOD_S[melody_tf])

    fig, ax = plt.subplots(1, 1, figsize=(22, 9))
    ax.plot(dt_5s, close_5s, color='black', lw=0.5, alpha=0.85, label='5s close')
    ax.plot(dt_5s, Mm, color='#1E88E5', lw=1.6, alpha=0.85,
            label=f'{motif_tf} M_close (motif anchor)')
    ax.plot(dt_5s, Ml, color='#FB8C00', lw=1.0, alpha=0.85,
            label=f'{melody_tf} M_close (melody anchor)')

    motif_colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#FCE4EC',
                    '#F3E5F5', '#FFEBEE', '#E0F2F1', '#FFFDE7']
    for m_idx, motif in enumerate(hierarchy['motifs']):
        s_dt = datetime.fromtimestamp(motif['start_ts'], tz=timezone.utc)
        e_dt = datetime.fromtimestamp(motif['end_ts'], tz=timezone.utc)
        bg = motif_colors[m_idx % len(motif_colors)]
        ax.axvspan(s_dt, e_dt, color=bg, alpha=0.5, zorder=0)
        # Motif boundary
        ax.axvline(s_dt, color='#1E88E5', lw=1.4, alpha=0.7, linestyle='-')
        # Motif label
        ymax = ax.get_ylim()[1]
        tag = (f'M{m_idx}: {motif["shape_class"]}'
               f'  {motif["length_min"]:.0f}m  '
               f'slope={motif["slope_pts_per_min"]:+.2f}  '
               f'r2={motif["r2adj"]:.2f}'
               f'  pkz={motif["peak_abs_z"]:.1f}')
        ax.text(s_dt, ymax, tag, fontsize=8.5, color='#0D47A1', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor='#1E88E5', linewidth=0.8))
        # Melody boundaries within motif
        for mel_idx, mel in enumerate(motif['melodies']):
            ms_dt = datetime.fromtimestamp(mel['start_ts'], tz=timezone.utc)
            ax.axvline(ms_dt, color='#FB8C00', lw=0.6, alpha=0.55, linestyle=':')

    # Last motif end boundary
    if hierarchy['motifs']:
        last_end = datetime.fromtimestamp(hierarchy['motifs'][-1]['end_ts'],
                                          tz=timezone.utc)
        ax.axvline(last_end, color='#1E88E5', lw=1.4, alpha=0.7, linestyle='-')

    th = hierarchy['theme']
    ax.set_title(
        f'{day}  ({th["dow"]})  THEME: range={th["session_range_pts"]:.1f}pts  '
        f'net={th["session_net_pts"]:+.1f}  '
        f'eff={th["session_efficiency"]:+.3f}  '
        f'motifs={th["n_motifs"]}  melodies={th["n_melodies"]}\n'
        f'BLUE bars=motif boundaries (15m CRM inflections)  '
        f'ORANGE dotted=melody boundaries within motif (5m CRM)',
        fontsize=11)
    ax.set_ylabel('price')
    ax.set_xlabel('time (UTC)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True, help='YYYY_MM_DD')
    ap.add_argument('--motif-tf', default='15m', choices=list(TF_WINDOW.keys()))
    ap.add_argument('--melody-tf', default='5m', choices=list(TF_WINDOW.keys()))
    ap.add_argument('--min-motif-min', type=float, default=30.0)
    ap.add_argument('--min-melody-min', type=float, default=5.0)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--out-chart', default=None)
    args = ap.parse_args()

    print(f'Segmenting {args.day}  motif={args.motif_tf} (>={args.min_motif_min}min)  '
          f'melody={args.melody_tf} (>={args.min_melody_min}min)')

    hierarchy = segment_day(args.day,
                            motif_tf=args.motif_tf,
                            melody_tf=args.melody_tf,
                            min_motif_min=args.min_motif_min,
                            min_melody_min=args.min_melody_min)
    if not hierarchy:
        print(f'No data for {args.day}')
        sys.exit(1)

    th = hierarchy['theme']
    print(f'\nTHEME: range={th["session_range_pts"]:.1f}pts  '
          f'net={th["session_net_pts"]:+.1f}  '
          f'eff={th["session_efficiency"]:+.3f}')
    print(f'  motifs={th["n_motifs"]}  total melodies={th["n_melodies"]}')
    print()
    for m in hierarchy['motifs']:
        s_hm = datetime.fromtimestamp(m['start_ts'], tz=timezone.utc).strftime('%H:%M')
        e_hm = datetime.fromtimestamp(m['end_ts'], tz=timezone.utc).strftime('%H:%M')
        print(f'  M{m["seg_idx"]}  {s_hm}-{e_hm}  {m["length_min"]:.0f}min  '
              f'shape={m["shape_class"]:<20s} r={m["shape_pearson_r"]:+.2f}  '
              f'slope={m["slope_pts_per_min"]:+.3f}/min  '
              f'r2adj={m["r2adj"]:.2f}  pk_z={m["peak_abs_z"]:.1f}  '
              f'net={m["net_move_pts"]:+.1f}')
        for mel in m['melodies']:
            ms_hm = datetime.fromtimestamp(mel['start_ts'], tz=timezone.utc).strftime('%H:%M')
            me_hm = datetime.fromtimestamp(mel['end_ts'], tz=timezone.utc).strftime('%H:%M')
            print(f'      m{mel["seg_idx"]}  {ms_hm}-{me_hm}  '
                  f'{mel["length_min"]:.0f}m  {mel["shape_class"]:<18s} '
                  f'r={mel["shape_pearson_r"]:+.2f}  slope={mel["slope_pts_per_min"]:+.3f}  '
                  f'r2={mel["r2adj"]:.2f}')

    out_json = args.out_json or f'reports/findings/segments/{args.day}.json'
    out_chart = args.out_chart or f'chart/segments/{args.day}.png'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(hierarchy, f, indent=2)
    print(f'\nJSON  -> {out_json}')
    render_chart(args.day, hierarchy, out_chart)
    print(f'Chart -> {out_chart}')


if __name__ == '__main__':
    main()
