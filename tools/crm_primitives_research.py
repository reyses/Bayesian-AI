"""CRM Primitives Research — apply SeedPrimitiveLibrary to CRMs (not raw price).

Tests user's hypothesis (2026-05-12): smoothed CRMs produce cleaner shape
classifications than raw price, reducing NOISE residual significantly.

REUSES existing infrastructure (no modifications to standalone_research.py):
    - tools.research.seeds.SeedPrimitiveLibrary  (20-shape classifier)
    - tools.research.seeds._detect_inflections   (slope-sign-change segments)
    - tools.cusp_marker.compute_anchor           (CRM time series builder)

INDEPENDENT per CRM — each timeframe has its own segmentation and chord slot.

TWO MODES per CRM:
  - fixed-window:  slide 16-bar window (matches Analysis I) → noise % comparison
  - inflection:    variable-length segments at slope sign changes
                    (natural CRM phases, what user actually observes)

Series fed to the classifier:
    M_15s    (window=20, fastest CRM)
    M_1m     (window=15)
    M_5m     (window=9)
    M_15m    (window=12)
    close    (raw price — control / baseline NOISE rate)

Output:
    reports/findings/crm_primitives/
        diagnostics_<run>.txt                       NOISE %, shape % per series
        segments_<series>_<mode>_<day>.csv          per-segment classifications
        chord_per_bar_<day>.csv                     chord at every 1m bar

Usage:
    python tools/crm_primitives_research.py --date 2025-09-08
    python tools/crm_primitives_research.py --start 2025-04-01 --end 2025-04-30
    python tools/crm_primitives_research.py --diagnostic   # noise-% summary only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.seeds import SeedPrimitiveLibrary, _detect_inflections
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars


OUT_DIR = Path('reports/findings/crm_primitives')

DEFAULT_SEED_LEN = 16
DEFAULT_MIN_SEG_LEN = 5
DEFAULT_NOISE_THRESHOLD = 0.75   # matches SeedPrimitiveLibrary.CORR_THRESHOLD


# ── Classification routines (reuse library; vary input series) ──────────────

def classify_fixed_window(series: np.ndarray, library: SeedPrimitiveLibrary,
                                  seed_len: int = DEFAULT_SEED_LEN) -> list:
    """Slide a fixed-length window across `series`, classify each.
    Returns list of {start_idx, end_idx, shape, corr}."""
    out = []
    n = len(series)
    for i in range(n - seed_len + 1):
        seg = series[i : i + seed_len]
        if np.any(np.isnan(seg)):
            continue
        shape, corr = library.classify_trajectory(seg)
        out.append({
            'start_idx': i, 'end_idx': i + seed_len - 1,
            'shape': shape, 'corr': round(corr, 3),
            'length': seed_len,
            'v0': float(seg[0]), 'v1': float(seg[-1]),
            'delta': float(seg[-1] - seg[0]),
        })
    return out


def classify_inflection(series: np.ndarray, library: SeedPrimitiveLibrary,
                                seed_len: int = DEFAULT_SEED_LEN,
                                min_seg_len: int = DEFAULT_MIN_SEG_LEN) -> list:
    """Segment at slope sign changes, classify each segment.
    Variable-length segments; resamples to seed_len for classifier."""
    valid = ~np.isnan(series)
    if not valid.any():
        return []
    inflections, segments = _detect_inflections(series[valid])

    # Map valid-array indices back to original series indices
    orig_idx = np.where(valid)[0]
    out = []
    for seg_desc in segments:
        s0_local, s1_local = seg_desc['start'], seg_desc['end']
        s0 = int(orig_idx[s0_local])
        s1 = int(orig_idx[s1_local])
        length = s1 - s0 + 1
        if length < min_seg_len:
            continue
        seg_raw = series[s0 : s1 + 1]
        # Resample (linear interpolation) to fixed length for classifier
        if len(seg_raw) == seed_len:
            seg_for_class = seg_raw
        else:
            x_old = np.linspace(0, 1, len(seg_raw))
            x_new = np.linspace(0, 1, seed_len)
            seg_for_class = np.interp(x_new, x_old, seg_raw)
        shape, corr = library.classify_trajectory(seg_for_class)
        out.append({
            'start_idx': s0, 'end_idx': s1,
            'shape': shape, 'corr': round(corr, 3),
            'length': length,
            'v0': float(seg_raw[0]), 'v1': float(seg_raw[-1]),
            'delta': float(seg_raw[-1] - seg_raw[0]),
            'inflection_label': seg_desc['label'],   # RISE/DROP/HOLD
        })
    return out


# ── Build chord-per-bar (which segment is active at each bar) ──────────────

def build_active_shape_per_bar(segments: list, n_bars: int) -> np.ndarray:
    """For each bar index, return the shape of the segment that CONTAINS it.
    NoSegment → 'UNDEFINED'."""
    out = np.array(['UNDEFINED'] * n_bars, dtype=object)
    for s in segments:
        s0, s1 = s['start_idx'], s['end_idx']
        if s0 >= n_bars or s1 < 0:
            continue
        out[max(0, s0) : min(n_bars, s1 + 1)] = s['shape']
    return out


# ── Diagnostics: NOISE % comparison ────────────────────────────────────────

def summarize_classifications(classifications: list, label: str) -> dict:
    """Tally NOISE rate + shape distribution + correlation stats."""
    if not classifications:
        return {'label': label, 'n': 0}
    n = len(classifications)
    counts = Counter(c['shape'] for c in classifications)
    n_noise = counts.get('NOISE', 0)
    corrs = np.array([c['corr'] for c in classifications])
    return {
        'label': label,
        'n_segments': n,
        'n_noise': n_noise,
        'pct_noise': round(100 * n_noise / n, 1),
        'pct_matched': round(100 * (n - n_noise) / n, 1),
        'mean_corr': round(float(corrs.mean()), 3),
        'top_shapes': counts.most_common(6),
    }


def write_diagnostic_report(diagnostics: list, run_name: str):
    out = OUT_DIR / f'diagnostics_{run_name}.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'CRM Primitives Diagnostic — {run_name}\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'{"series_mode":<28} {"n":>8} {"NOISE%":>8} '
                  f'{"matched%":>9} {"mean_corr":>10}  top shapes\n')
        f.write('-' * 100 + '\n')
        for d in diagnostics:
            if d.get('n_segments', 0) == 0:
                f.write(f'{d["label"]:<28} (no segments)\n')
                continue
            top = ', '.join(f'{s}:{c}' for s, c in d['top_shapes'][:4])
            f.write(f'{d["label"]:<28} '
                      f'{d["n_segments"]:>8} '
                      f'{d["pct_noise"]:>7.1f}% '
                      f'{d["pct_matched"]:>8.1f}% '
                      f'{d["mean_corr"]:>10.3f}  {top}\n')
        # Hypothesis test summary
        f.write('\n' + '=' * 70 + '\n')
        f.write('HYPOTHESIS: CRM-based classification produces less NOISE than raw\n')
        f.write('=' * 70 + '\n')
        raw_d = next((d for d in diagnostics if d['label'].startswith('close')), None)
        if raw_d:
            base_noise = raw_d['pct_noise']
            f.write(f'\nBaseline (raw close):  {base_noise:.1f}% NOISE\n\n')
            for d in diagnostics:
                if d.get('n_segments', 0) == 0:
                    continue
                if d['label'].startswith('close'):
                    continue
                delta = base_noise - d['pct_noise']
                verdict = (' ✓ improvement' if delta > 5
                                else ' = same' if abs(delta) <= 5
                                else ' ✗ worse')
                f.write(f'  {d["label"]:<28} {d["pct_noise"]:>5.1f}% NOISE  '
                          f'(Δ {delta:+5.1f}pp){verdict}\n')
    print(f'\nDiagnostic written: {out}')


def write_segments(segments: list, series_name: str, mode: str, run_name: str):
    if not segments:
        return
    out = OUT_DIR / f'segments_{series_name}_{mode}_{run_name}.csv'
    cols = list(segments[0].keys())
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in segments:
            w.writerow(s)


def write_chord_per_bar(chord_arrays: dict, timestamps: np.ndarray, run_name: str):
    """Save the chord at every 1m bar. Columns: ts, active_M_15s_shape,
    active_M_1m_shape, active_M_5m_shape, active_M_15m_shape (and same for close)."""
    out = OUT_DIR / f'chord_per_bar_{run_name}.csv'
    n_bars = len(timestamps)
    series_names = list(chord_arrays.keys())
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['timestamp', 'utc'] + [f'active_{s}_shape' for s in series_names]
        w.writerow(header)
        for i in range(n_bars):
            ts = int(timestamps[i])
            utc = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            row = [ts, utc] + [chord_arrays[s][i] for s in series_names]
            w.writerow(row)
    print(f'Wrote chord-per-bar: {out}')


# ── Main pipeline ──────────────────────────────────────────────────────────

def run(t_start: float, t_end: float, run_name: str, seed_len: int = DEFAULT_SEED_LEN,
          min_seg_len: int = DEFAULT_MIN_SEG_LEN, diagnostic_only: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\n=== CRM Primitives Research — {run_name} ===')
    print(f'Range: {datetime.fromtimestamp(t_start, tz=timezone.utc)} → '
              f'{datetime.fromtimestamp(t_end, tz=timezone.utc)}')
    print(f'Seed length: {seed_len}    Min inflection segment: {min_seg_len}')

    df = load_1m_bars(t_start, t_end)
    if df.empty:
        print('No data loaded'); return
    ts = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    print(f'Loaded {len(df)} 1m bars')

    # Compute CRMs at 1m grid (via cusp_marker.compute_anchor)
    print('Computing CRMs...')
    M_15s, _ = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  _ = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_5m,  _ = compute_anchor('5m',  ts, t_start, t_end, window=9,  column='close')
    M_15m, _ = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')

    series_to_classify = {
        'close':  close,      # raw price baseline
        'M_15s':  M_15s,
        'M_1m':   M_1m,
        'M_5m':   M_5m,
        'M_15m':  M_15m,
    }

    library = SeedPrimitiveLibrary(N=seed_len)
    print(f'Library: {len(library.shapes)} shapes, threshold corr ≥ {library.CORR_THRESHOLD}')

    # Classify each series in both modes
    diagnostics = []
    chord_active = {}     # series_name → active shape per bar (inflection mode)

    for name, series in series_to_classify.items():
        # Mode 1: fixed-window
        print(f'\n  [{name}] fixed-window classification...')
        fixed = classify_fixed_window(series, library, seed_len)
        diagnostics.append(summarize_classifications(fixed, f'{name}_fixed'))
        write_segments(fixed, name, 'fixed', run_name)

        # Mode 2: inflection-based
        print(f'  [{name}] inflection-based classification...')
        infl = classify_inflection(series, library, seed_len, min_seg_len)
        diagnostics.append(summarize_classifications(infl, f'{name}_inflection'))
        write_segments(infl, name, 'inflection', run_name)

        # Build active-shape array for chord-per-bar (using inflection segments)
        chord_active[name] = build_active_shape_per_bar(infl, len(ts))

    # Write diagnostic report (the noise-% comparison)
    write_diagnostic_report(diagnostics, run_name)

    if not diagnostic_only:
        write_chord_per_bar(chord_active, ts, run_name)

    # Console summary
    print(f'\n=== DIAGNOSTIC SUMMARY ({run_name}) ===')
    print(f'{"series_mode":<28} {"n":>8} {"NOISE%":>8} {"matched%":>9} {"top shapes":<40}')
    print('-' * 95)
    for d in diagnostics:
        if d.get('n_segments', 0) == 0:
            print(f'{d["label"]:<28} (no segments)')
            continue
        top = ', '.join(f'{s}:{c}' for s, c in d['top_shapes'][:3])
        print(f'{d["label"]:<28} {d["n_segments"]:>8} {d["pct_noise"]:>7.1f}% '
                  f'{d["pct_matched"]:>8.1f}%  {top}')

    print(f'\nAll outputs in: {OUT_DIR}')


# ── CLI ────────────────────────────────────────────────────────────────────

def _ts(d: str) -> float:
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Single date YYYY-MM-DD')
    ap.add_argument('--start', help='Start date YYYY-MM-DD')
    ap.add_argument('--end', help='End date YYYY-MM-DD')
    ap.add_argument('--days', type=int, default=1,
                       help='If --date given, load N days from that date')
    ap.add_argument('--name', help='Run name (default auto)')
    ap.add_argument('--seed-len', type=int, default=DEFAULT_SEED_LEN)
    ap.add_argument('--min-seg-len', type=int, default=DEFAULT_MIN_SEG_LEN)
    ap.add_argument('--diagnostic', action='store_true',
                       help='Skip chord-per-bar output; just NOISE-% comparison')
    args = ap.parse_args()

    if args.date:
        t_start = _ts(args.date)
        t_end = t_start + args.days * 86400
        run_name = args.name or args.date
    elif args.start and args.end:
        t_start = _ts(args.start)
        t_end = _ts(args.end) + 86400
        run_name = args.name or f'{args.start}_{args.end}'
    else:
        ap.error('Provide --date OR --start+--end')

    run(t_start, t_end, run_name,
         seed_len=args.seed_len,
         min_seg_len=args.min_seg_len,
         diagnostic_only=args.diagnostic)


if __name__ == '__main__':
    main()
