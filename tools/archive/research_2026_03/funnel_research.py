#!/usr/bin/env python
"""
Decision Funnel Research — simulate temporal candidate narrowing on IS data.

Replays 15s bars, tracks template×direction persistence across bars,
and correlates funnel survival with actual trade outcomes.

Uses I-MR auto seeds as ground truth for regime alignment validation.

Usage:
    python tools/funnel_research.py --data DATA/ATLAS_1WEEK
    python tools/funnel_research.py --data DATA/ATLAS --month 2025_03
    python tools/funnel_research.py --seeds DATA/regime_seeds/imr_auto/imr_seeds_all_*.json

Output: reports/findings/funnel_research_YYYYMMDD_HHMMSS.txt
"""

import argparse
import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICK_SIZE = 0.25
TICK_VALUE = 0.50


@dataclass
class FunnelCandidate:
    """A template x direction tracked across bars."""
    template_id: int
    direction: str
    first_seen_bar: int
    bars_survived: int = 0
    conviction_path: list = field(default_factory=list)
    z_score_path: list = field(default_factory=list)
    dist_path: list = field(default_factory=list)
    eliminated_bar: int = -1
    elimination_reason: str = ''


def load_checkpoints():
    """Load pattern library + scaler for template matching."""
    ckpt_dir = 'checkpoints'
    lib_path = os.path.join(ckpt_dir, 'pattern_library.pkl')
    scaler_path = os.path.join(ckpt_dir, 'clustering_scaler.pkl')

    if not os.path.exists(lib_path) or not os.path.exists(scaler_path):
        print("ERROR: checkpoints/pattern_library.pkl or clustering_scaler.pkl not found")
        sys.exit(1)

    with open(lib_path, 'rb') as f:
        lib = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Build centroid matrix + valid TIDs
    valid_tids = sorted(lib.keys())
    centroids = np.array([lib[tid]['centroid'] for tid in valid_tids])
    centroids_scaled = scaler.transform(centroids)

    return lib, scaler, valid_tids, centroids_scaled


def load_traded_signals() -> pd.DataFrame:
    """Load signal_log shards to get actual trade outcomes."""
    shard_dir = 'reports/is/shards'
    files = sorted(Path(shard_dir).glob('signal_log_*.csv'))
    if not files:
        print("  WARNING: No signal_log shards found")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    traded = df[df['trade_result'].notna() & (df['trade_result'] != '')].copy()
    print(f"  Loaded {len(traded):,} traded signals from {len(files)} shards")
    return traded


def load_imr_seeds(seed_path: str) -> dict:
    """Load I-MR auto seeds for regime alignment."""
    if not seed_path or not os.path.exists(seed_path):
        return {}

    with open(seed_path) as f:
        data = json.load(f)

    seeds_by_ts = {}
    if 'days' in data:
        for day_data in data['days'].values():
            for seed in day_data.get('seeds', []):
                ts = seed.get('ts_start', 0)
                seeds_by_ts[ts] = seed

    print(f"  Loaded {len(seeds_by_ts):,} I-MR seeds")
    return seeds_by_ts


def compute_template_match(bar_features, scaler, centroids_scaled, valid_tids,
                            gate1_dist=3.0):
    """Find which templates match the current bar's features.

    When features are 16D but scaler/centroids are 22D (lookback mode),
    compare only the first 16 dimensions to avoid zero-padding distortion.
    """
    feat_dim = bar_features.shape[0]
    expected_dim = centroids_scaled.shape[1]

    if feat_dim < expected_dim:
        # Scale 16D features manually using first 16 dims of scaler params
        # (avoids distortion from zero-padded lookback dims in scaler.transform)
        _mean = scaler.mean_[:feat_dim]
        _scale = scaler.scale_[:feat_dim]
        feat_scaled_16 = (bar_features - _mean) / _scale
        # Distance on shared dimensions only
        dists = np.linalg.norm(centroids_scaled[:, :feat_dim] - feat_scaled_16.reshape(1, -1), axis=1)
    else:
        feat_2d = bar_features.reshape(1, -1)[:, :expected_dim]
        feat_scaled = scaler.transform(feat_2d)
        dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)

    matches = []
    for i, (tid, dist) in enumerate(zip(valid_tids, dists)):
        if dist < gate1_dist:
            matches.append((tid, float(dist)))

    return matches


def run_funnel_simulation(data_dir: str, month: str = None,
                           max_bars: int = 0,
                           gate1_dist: float = 3.0,
                           min_survival: int = 8):
    """Simulate the decision funnel on 15s data."""

    print("\n  Loading data...")
    from tools.research.data import load_atlas_tf

    # Load 15s data
    months = [month] if month else None
    df_15s = load_atlas_tf(data_dir, '15s', months=months)
    if df_15s.empty:
        print("  ERROR: No 15s data")
        return None

    print(f"  {len(df_15s):,} 15s bars loaded")
    if max_bars > 0:
        df_15s = df_15s.head(max_bars)
        print(f"  Truncated to {max_bars:,} bars")

    # Load checkpoints
    print("\n  Loading checkpoints...")
    lib, scaler, valid_tids, centroids_scaled = load_checkpoints()
    print(f"  {len(lib)} templates, scaler: {centroids_scaled.shape[1]}D")

    # Load engine for feature extraction + state computation
    print("\n  Computing market states...")
    from core.statistical_field_engine import StatisticalFieldEngine
    engine = StatisticalFieldEngine()
    states = engine.batch_compute_states(df_15s, use_cuda=True)
    states_map = {s['bar_idx']: s['state'] for s in states}
    print(f"  {len(states):,} states computed")

    # Feature extractor
    # Load traded signals for outcome matching
    traded = load_traded_signals()
    traded_ts_set = set(traded['ts'].values) if len(traded) > 0 else set()

    # ── Funnel simulation ─────────────────────────────────────────────
    print(f"\n  Running funnel simulation (gate1_dist={gate1_dist}, min_survival={min_survival})...")

    active_funnel: Dict[Tuple[int, str], FunnelCandidate] = {}
    eliminated: List[FunnelCandidate] = []
    funnel_width_history = []

    # Track outcomes
    entry_with_funnel = []      # trades where template was in funnel
    entry_without_funnel = []   # trades where template appeared fresh
    entry_mature = []           # trades where template survived min_survival bars

    timestamps = df_15s['timestamp'].values.astype(float)
    closes = df_15s['close'].values.astype(float)

    for bar_i in tqdm(range(len(df_15s)), desc='Funnel sim', unit='bar',
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):

        state = states_map.get(bar_i)
        if state is None:
            funnel_width_history.append(len(active_funnel))
            continue

        ts = float(timestamps[bar_i])
        z_score = abs(getattr(state, 'z_score', 0.0))

        # Skip low-z bars (noise zone)
        if z_score < 0.5:
            # Eliminate active candidates that lost z-score support
            for key in list(active_funnel.keys()):
                fc = active_funnel[key]
                fc.eliminated_bar = bar_i
                fc.elimination_reason = 'z_collapse'
                eliminated.append(fc)
                del active_funnel[key]
            funnel_width_history.append(0)
            continue

        # Extract features for template matching
        try:
            from core.feature_extraction import extract_feature_vector
            _adx = getattr(state, 'adx_strength', 0.0) / 100.0
            _hurst = getattr(state, 'hurst_exponent', 0.5)
            _dmi_diff = (getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0
            _pid = getattr(state, 'term_pid', 0.0)
            _osc = getattr(state, 'oscillation_entropy_normalized', 0.0)
            features = np.array(extract_feature_vector(
                z_score=getattr(state, 'z_score', 0.0),
                velocity=getattr(state, 'velocity', 0.0),
                momentum=getattr(state, 'momentum', 0.0),
                entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
                tf_seconds=15,
                depth=0.0,
                parent_is_band_reversal=0.0,
                adx=_adx, hurst=_hurst, dmi_diff=_dmi_diff,
                parent_z=0.0, parent_dmi_diff=0.0,
                root_is_roche=0.0, tf_alignment=0.0,
                pid=_pid, osc_coherence=_osc,
            ))
        except Exception as _fe_err:
            funnel_width_history.append(len(active_funnel))
            continue

        # Find matching templates
        matches = compute_template_match(features, scaler, centroids_scaled,
                                          valid_tids, gate1_dist)

        # Debug: log first bar with z > 1
        if bar_i < 500 and z_score > 1.0 and not getattr(run_funnel_simulation, '_debug_done', False):
            feat_2d = features.reshape(1, -1)
            if feat_2d.shape[1] < centroids_scaled.shape[1]:
                pad = np.zeros((1, centroids_scaled.shape[1] - feat_2d.shape[1]))
                feat_2d = np.concatenate([feat_2d, pad], axis=1)
            feat_s = scaler.transform(feat_2d)
            dists = np.linalg.norm(centroids_scaled - feat_s, axis=1)
            print(f"  DEBUG bar={bar_i} z={z_score:.2f} feat={features.shape} "
                  f"min_dist={dists.min():.2f} matches={len(matches)}")
            run_funnel_simulation._debug_done = True

        # Determine direction from z-score sign (simplified)
        raw_z = getattr(state, 'z_score', 0.0)
        bar_direction = 'short' if raw_z > 0 else 'long'

        # Update funnel: new candidates + validate existing
        current_match_keys = set()
        for tid, dist in matches:
            for direction in ['long', 'short']:
                key = (tid, direction)
                current_match_keys.add(key)

                if key not in active_funnel:
                    active_funnel[key] = FunnelCandidate(
                        template_id=tid,
                        direction=direction,
                        first_seen_bar=bar_i,
                    )

                fc = active_funnel[key]
                fc.bars_survived += 1
                fc.z_score_path.append(z_score)
                fc.dist_path.append(dist)

        # Eliminate candidates no longer matching
        for key in list(active_funnel.keys()):
            if key not in current_match_keys:
                fc = active_funnel[key]
                fc.eliminated_bar = bar_i
                fc.elimination_reason = 'template_lost'
                eliminated.append(fc)
                del active_funnel[key]

        funnel_width_history.append(len(active_funnel))

        # Check if this bar had a real trade
        if ts in traded_ts_set:
            trade_rows = traded[traded['ts'] == ts]
            for _, trade in trade_rows.iterrows():
                tid = int(trade['template_id']) if pd.notna(trade['template_id']) else -1
                t_dir = str(trade.get('trade_direction', '')).lower()
                t_pnl = float(trade.get('trade_pnl', 0))
                t_result = str(trade.get('trade_result', ''))

                trade_info = {
                    'ts': ts,
                    'template_id': tid,
                    'direction': t_dir,
                    'pnl': t_pnl,
                    'result': t_result,
                    'funnel_width': len(active_funnel),
                }

                key = (tid, t_dir)
                if key in active_funnel:
                    fc = active_funnel[key]
                    trade_info['bars_survived'] = fc.bars_survived
                    trade_info['in_funnel'] = True
                    trade_info['mature'] = fc.bars_survived >= min_survival

                    # Conviction trend (z-score as proxy since we don't have TBN here)
                    if len(fc.z_score_path) >= 3:
                        trade_info['z_trend'] = fc.z_score_path[-1] - fc.z_score_path[0]
                    else:
                        trade_info['z_trend'] = 0

                    entry_with_funnel.append(trade_info)
                    if fc.bars_survived >= min_survival:
                        entry_mature.append(trade_info)
                else:
                    trade_info['bars_survived'] = 0
                    trade_info['in_funnel'] = False
                    trade_info['mature'] = False
                    trade_info['z_trend'] = 0
                    entry_without_funnel.append(trade_info)

    return {
        'funnel_width_history': funnel_width_history,
        'eliminated': eliminated,
        'entry_with_funnel': entry_with_funnel,
        'entry_without_funnel': entry_without_funnel,
        'entry_mature': entry_mature,
        'total_traded': len(entry_with_funnel) + len(entry_without_funnel),
    }


def print_results(results: dict, output_path: str):
    """Print and save funnel research results."""

    lines = []
    def p(s=''):
        print(s)
        lines.append(s)

    p("=" * 70)
    p("  DECISION FUNNEL RESEARCH RESULTS")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 70)

    total = results['total_traded']
    with_f = results['entry_with_funnel']
    without_f = results['entry_without_funnel']
    mature = results['entry_mature']
    widths = results['funnel_width_history']

    p(f"\n  Total traded signals matched: {total}")
    p(f"  In funnel at entry:     {len(with_f):>6,}  ({len(with_f)/max(1,total)*100:.1f}%)")
    p(f"  NOT in funnel at entry: {len(without_f):>6,}  ({len(without_f)/max(1,total)*100:.1f}%)")
    p(f"  Mature (>={8} bars):     {len(mature):>6,}  ({len(mature)/max(1,total)*100:.1f}%)")

    # Funnel width stats
    w = np.array(widths)
    p(f"\n  FUNNEL WIDTH (candidates tracked per bar):")
    p(f"    Mean={np.mean(w):.1f}  Med={np.median(w):.0f}  "
      f"P75={np.percentile(w, 75):.0f}  Max={np.max(w)}")
    p(f"    Bars with 0 candidates: {(w == 0).sum():,} ({(w == 0).mean()*100:.1f}%)")

    # Elimination reasons
    reasons = defaultdict(int)
    survival_durations = []
    for fc in results['eliminated']:
        reasons[fc.elimination_reason] += 1
        survival_durations.append(fc.bars_survived)

    if survival_durations:
        p(f"\n  CANDIDATE SURVIVAL DURATION:")
        sd = np.array(survival_durations)
        p(f"    Mean={np.mean(sd):.1f} bars  Med={np.median(sd):.0f}  "
          f"P75={np.percentile(sd, 75):.0f}  P90={np.percentile(sd, 90):.0f}  Max={np.max(sd)}")
        p(f"    Survived <3 bars:  {(sd < 3).sum():,} ({(sd < 3).mean()*100:.1f}%)")
        p(f"    Survived 3-8 bars: {((sd >= 3) & (sd < 8)).sum():,}")
        p(f"    Survived 8+ bars:  {(sd >= 8).sum():,} ({(sd >= 8).mean()*100:.1f}%)")

    p(f"\n  ELIMINATION REASONS:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        p(f"    {reason:20s}: {count:>8,}")

    # KEY ANALYSIS: funnel vs non-funnel outcomes
    p(f"\n{'='*70}")
    p(f"  FUNNEL vs NON-FUNNEL TRADE OUTCOMES")
    p(f"{'='*70}")

    for label, group in [('In funnel', with_f), ('NOT in funnel', without_f), ('Mature (8+ bars)', mature)]:
        if not group:
            p(f"\n  {label}: 0 trades")
            continue
        n = len(group)
        wins = sum(1 for t in group if t['result'] == 'WIN')
        wr = wins / n * 100
        pnls = [t['pnl'] for t in group]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / n
        p(f"\n  {label}: {n:,} trades")
        p(f"    WR={wr:.1f}%  Avg=${avg_pnl:.2f}  Total=${total_pnl:,.2f}")

        # Survival duration distribution (only for in-funnel)
        if label != 'NOT in funnel':
            surv = [t['bars_survived'] for t in group]
            p(f"    Survival: mean={np.mean(surv):.1f} bars, med={np.median(surv):.0f}")

            # Z-score trend for survivors
            z_trends = [t['z_trend'] for t in group]
            rising = sum(1 for z in z_trends if z > 0.1)
            falling = sum(1 for z in z_trends if z < -0.1)
            flat = n - rising - falling
            p(f"    Z-trend: rising={rising} ({rising/n*100:.0f}%)  "
              f"flat={flat} ({flat/n*100:.0f}%)  falling={falling} ({falling/n*100:.0f}%)")

            # WR by z-trend
            for trend_label, pred in [('Rising z', lambda t: t['z_trend'] > 0.1),
                                       ('Flat z', lambda t: abs(t['z_trend']) <= 0.1),
                                       ('Falling z', lambda t: t['z_trend'] < -0.1)]:
                sub = [t for t in group if pred(t)]
                if not sub:
                    continue
                sn = len(sub)
                swr = sum(1 for t in sub if t['result'] == 'WIN') / sn * 100
                savg = sum(t['pnl'] for t in sub) / sn
                p(f"      {trend_label:12s}: {sn:>4d} trades  WR={swr:5.1f}%  avg=${savg:.2f}")

    # Funnel width at entry
    if with_f:
        widths_at_entry = [t['funnel_width'] for t in with_f]
        p(f"\n  FUNNEL WIDTH AT ENTRY (in-funnel trades):")
        p(f"    Mean={np.mean(widths_at_entry):.0f}  Med={np.median(widths_at_entry):.0f}  "
          f"P75={np.percentile(widths_at_entry, 75):.0f}")

        # Narrow funnel = better?
        narrow = [t for t in with_f if t['funnel_width'] <= np.median(widths_at_entry)]
        wide = [t for t in with_f if t['funnel_width'] > np.median(widths_at_entry)]
        if narrow and wide:
            n_wr = sum(1 for t in narrow if t['result'] == 'WIN') / len(narrow) * 100
            w_wr = sum(1 for t in wide if t['result'] == 'WIN') / len(wide) * 100
            n_avg = sum(t['pnl'] for t in narrow) / len(narrow)
            w_avg = sum(t['pnl'] for t in wide) / len(wide)
            p(f"    Narrow funnel (<={np.median(widths_at_entry):.0f}): "
              f"{len(narrow)} trades  WR={n_wr:.1f}%  avg=${n_avg:.2f}")
            p(f"    Wide funnel (>{np.median(widths_at_entry):.0f}):   "
              f"{len(wide)} trades  WR={w_wr:.1f}%  avg=${w_avg:.2f}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    p(f"\n  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Decision Funnel Research')
    parser.add_argument('--data', default='DATA/ATLAS_1WEEK',
                        help='ATLAS data directory')
    parser.add_argument('--month', default=None,
                        help='Restrict to month (e.g. 2025_03)')
    parser.add_argument('--max-bars', type=int, default=0,
                        help='Limit bars for fast testing (0=all)')
    parser.add_argument('--gate-dist', type=float, default=3.0,
                        help='Gate 1 distance threshold')
    parser.add_argument('--min-survival', type=int, default=8,
                        help='Minimum bars for mature candidate')
    parser.add_argument('--seeds', default=None,
                        help='I-MR auto seed JSON for regime alignment')
    args = parser.parse_args()

    print("=" * 70)
    print("  DECISION FUNNEL RESEARCH")
    print(f"  Data: {args.data}")
    print("=" * 70)

    results = run_funnel_simulation(
        args.data,
        month=args.month,
        max_bars=args.max_bars,
        gate1_dist=args.gate_dist,
        min_survival=args.min_survival,
    )

    if results is None:
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'reports/findings/funnel_research_{ts}.txt'
    print_results(results, output_path)


if __name__ == '__main__':
    main()
