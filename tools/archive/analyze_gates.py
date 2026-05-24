#!/usr/bin/env python
"""
Gate Threshold Analyzer — uses oracle data to find optimal gate thresholds.

Reads oracle trade logs (signal_log / oracle_trade_log / fn_oracle_log) and
analyzes physics fields to determine where each gate threshold should be set
for maximum WR without killing golden-path capture.

Usage:
    python tools/analyze_gates.py                          # default: checkpoints/
    python tools/analyze_gates.py --dir checkpoints/oos    # specific run
    python tools/analyze_gates.py --gate 5b                # analyze one gate only
    python tools/analyze_gates.py --all                    # all gates

Output:
    Console report + optional CSV to reports/gate_analysis.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_signal_log(checkpoint_dir: str) -> pd.DataFrame:
    """Load the signal log CSV (all candidates: traded + skipped).
    Searches checkpoint_dir first, then reports/{is,oos,oos2}/ for shards.
    """
    # Legacy: check checkpoint dir
    for name in ('is_signal_log.csv', 'oos_signal_log.csv', 'signal_log.csv'):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {path}: {len(df)} records")
            return df
    # New layout: reports/{mode}/ (single file or shards/)
    import glob
    for mode in ('is', 'oos', 'oos2'):
        mode_dir = os.path.join('reports', mode)
        shards = sorted(glob.glob(os.path.join(mode_dir, 'shards', 'signal_log_*.csv')))
        if shards:
            dfs = [pd.read_csv(s) for s in shards]
            df = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(shards)} signal log shards from {mode_dir}: {len(df)} records")
            return df
        single = os.path.join(mode_dir, 'signal_log.csv')
        if os.path.exists(single):
            df = pd.read_csv(single)
            print(f"  Loaded {single}: {len(df)} records")
            return df
    # Legacy shards in checkpoint dir
    shards = sorted(glob.glob(os.path.join(checkpoint_dir, 'shards', '*signal_log*.csv')))
    if shards:
        dfs = [pd.read_csv(s) for s in shards]
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(shards)} signal log shards: {len(df)} records")
        return df
    return pd.DataFrame()


def load_oracle_log(checkpoint_dir: str) -> pd.DataFrame:
    """Load oracle trade log (traded signals with outcomes)."""
    for name in ('is_trade_log.csv', 'oos_trade_log.csv', 'oracle_trade_log.csv'):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {path}: {len(df)} trades")
            return df
    return pd.DataFrame()


def load_fn_log(checkpoint_dir: str) -> pd.DataFrame:
    """Load FN oracle log (missed profitable signals)."""
    import glob
    # Legacy checkpoint dir
    for name in ('fn_oracle_log.csv',):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {path}: {len(df)} FN records")
            return df
    # New layout: reports/{mode}/
    for mode in ('is', 'oos', 'oos2'):
        mode_dir = os.path.join('reports', mode)
        shards = sorted(glob.glob(os.path.join(mode_dir, 'shards', 'fn_oracle_log_*.csv')))
        if shards:
            dfs = [pd.read_csv(s) for s in shards]
            df = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(shards)} FN shards from {mode_dir}: {len(df)} records")
            return df
        single = os.path.join(mode_dir, 'fn_oracle_log.csv')
        if os.path.exists(single):
            df = pd.read_csv(single)
            print(f"  Loaded {single}: {len(df)} FN records")
            return df
    # Legacy sharded in checkpoint dir
    shards = sorted(glob.glob(os.path.join(checkpoint_dir, '*fn_*log*.csv')))
    if shards:
        dfs = [pd.read_csv(s) for s in shards]
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(shards)} FN shards: {len(df)} FN records")
        return df
    return pd.DataFrame()


def analyze_gate_5b(df: pd.DataFrame):
    """Analyze momentum/reversion ratio gate using oracle labels."""
    print("\n" + "=" * 70)
    print("GATE 5b ANALYSIS: Momentum / Reversion Ratio")
    print("=" * 70)

    if 'mom_rev_ratio' not in df.columns:
        print("  ERROR: 'mom_rev_ratio' column not found in data.")
        print("  Run a forward pass first to populate physics fields in oracle records.")
        return

    # Split by oracle outcome
    real = df[df['oracle_label'].isin(['MEGA', 'SCALP'])].copy()
    noise = df[df['oracle_label'] == 'NOISE'].copy()
    traded = df[df['gate'] == 'traded'].copy()

    print(f"\n  Total signals: {len(df)}")
    print(f"  Real moves (MEGA/SCALP): {len(real)}")
    print(f"  Noise: {len(noise)}")
    print(f"  Traded: {len(traded)}")

    # Distribution of ratio for real vs noise
    if len(real) > 0 and 'mom_rev_ratio' in real.columns:
        r_real = real['mom_rev_ratio'].dropna()
        r_noise = noise['mom_rev_ratio'].dropna() if len(noise) > 0 else pd.Series()

        print(f"\n  Ratio distribution (real moves):")
        for p in [10, 25, 50, 75, 90]:
            print(f"    p{p}: {np.percentile(r_real, p):.1f}x")

        if len(r_noise) > 0:
            print(f"\n  Ratio distribution (noise):")
            for p in [10, 25, 50, 75, 90]:
                print(f"    p{p}: {np.percentile(r_noise, p):.1f}x")

    # Sweep thresholds: for each threshold, compute WR and capture if we block above it
    print(f"\n  {'Threshold':>10} {'Blocked':>8} {'Blk Real':>9} {'Blk Noise':>10} "
          f"{'Capture':>8} {'Signal WR':>10} {'Rec':>4}")
    print("  " + "-" * 65)

    best_score = -999
    best_thresh = None

    for thresh in [0, 1.5, 3, 5, 8, 10, 15, 20, 30, 50, 100, 999]:
        blocked = df[df['mom_rev_ratio'] > thresh]
        passed = df[df['mom_rev_ratio'] <= thresh]
        blk_real = real[real['mom_rev_ratio'] > thresh]
        blk_noise = noise[noise['mom_rev_ratio'] > thresh]
        pass_real = real[real['mom_rev_ratio'] <= thresh]

        capture = len(pass_real) / len(real) * 100 if len(real) > 0 else 0
        # "Signal WR" = what % of passed signals are real moves
        signal_wr = len(pass_real) / len(passed) * 100 if len(passed) > 0 else 0

        # Score: balance capture and signal quality
        # Penalize heavily for losing capture, reward for filtering noise
        score = capture * 0.7 + signal_wr * 0.3

        marker = ""
        if score > best_score:
            best_score = score
            best_thresh = thresh
            marker = " <--"

        print(f"  {thresh:>10.0f} {len(blocked):>8} {len(blk_real):>9} {len(blk_noise):>10} "
              f"{capture:>7.1f}% {signal_wr:>9.1f}%{marker}")

    print(f"\n  RECOMMENDATION: threshold = {best_thresh}")
    if best_thresh == 0 or best_thresh >= 999:
        print("  -> Gate 5b has NO predictive power. DISABLE it.")
    else:
        print(f"  -> Set momentum_override_ratio = {best_thresh}")

    return best_thresh


def analyze_gate_hurst(df: pd.DataFrame):
    """Analyze Hurst exponent gate using oracle labels. Returns optimal threshold."""
    print("\n" + "=" * 70)
    print("GATE 5a ANALYSIS: Hurst Exponent")
    print("=" * 70)

    if 'hurst' not in df.columns:
        print("  ERROR: 'hurst' column not found. Run forward pass first.")
        return None

    real = df[df['oracle_label'].isin(['MEGA', 'SCALP'])]
    noise = df[df['oracle_label'] == 'NOISE']

    if len(real) > 0:
        print(f"\n  Hurst distribution (real moves, n={len(real)}):")
        r = real['hurst'].dropna()
        for p in [10, 25, 50, 75, 90]:
            print(f"    p{p}: {np.percentile(r, p):.3f}")

    if len(noise) > 0:
        print(f"\n  Hurst distribution (noise, n={len(noise)}):")
        r = noise['hurst'].dropna()
        for p in [10, 25, 50, 75, 90]:
            print(f"    p{p}: {np.percentile(r, p):.3f}")

    print(f"\n  {'Threshold':>10} {'Blocked':>8} {'Blk Real':>9} {'Capture':>8} "
          f"{'Signal WR':>10} {'Rec':>4}")
    print("  " + "-" * 60)

    best_score = -999
    best_thresh = None

    for thresh in [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        passed = df[df['hurst'] >= thresh]
        pass_real = real[real['hurst'] >= thresh]
        blk_real = real[real['hurst'] < thresh]
        capture = len(pass_real) / len(real) * 100 if len(real) > 0 else 0
        signal_wr = len(pass_real) / len(passed) * 100 if len(passed) > 0 else 0
        score = capture * 0.7 + signal_wr * 0.3
        marker = ""
        if score > best_score:
            best_score = score
            best_thresh = thresh
            marker = " <--"
        print(f"  {thresh:>10.2f} {len(blk_real):>8} {len(blk_real):>9} "
              f"{capture:>7.1f}% {signal_wr:>9.1f}%{marker}")

    print(f"\n  RECOMMENDATION: hurst_min = {best_thresh}")
    if best_thresh == 0.0:
        print("  -> Hurst gate has NO predictive power at these thresholds. DISABLE it.")
    else:
        print(f"  -> Set hurst_min = {best_thresh}")

    return best_thresh


def analyze_gate_tunnel(df: pd.DataFrame):
    """Analyze tunnel probability gate using oracle labels. Returns optimal threshold."""
    print("\n" + "=" * 70)
    print("GATE 5c ANALYSIS: Tunnel (Reversion) Probability")
    print("=" * 70)

    if 'tunnel_prob' not in df.columns:
        print("  ERROR: 'tunnel_prob' column not found. Run forward pass first.")
        return None

    real = df[df['oracle_label'].isin(['MEGA', 'SCALP'])]
    noise = df[df['oracle_label'] == 'NOISE']

    if len(real) > 0:
        print(f"\n  Tunnel prob distribution (real moves, n={len(real)}):")
        r = real['tunnel_prob'].dropna()
        for p in [10, 25, 50, 75, 90]:
            print(f"    p{p}: {np.percentile(r, p):.3f}")

    print(f"\n  {'Threshold':>10} {'Blocked':>8} {'Blk Real':>9} {'Capture':>8} "
          f"{'Signal WR':>10} {'Rec':>4}")
    print("  " + "-" * 60)

    best_score = -999
    best_thresh = None

    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        passed = df[df['tunnel_prob'] >= thresh]
        pass_real = real[real['tunnel_prob'] >= thresh]
        blk_real = real[real['tunnel_prob'] < thresh]
        capture = len(pass_real) / len(real) * 100 if len(real) > 0 else 0
        signal_wr = len(pass_real) / len(passed) * 100 if len(passed) > 0 else 0
        score = capture * 0.7 + signal_wr * 0.3
        marker = ""
        if score > best_score:
            best_score = score
            best_thresh = thresh
            marker = " <--"
        print(f"  {thresh:>10.2f} {len(blk_real):>8} {len(blk_real):>9} "
              f"{capture:>7.1f}% {signal_wr:>9.1f}%{marker}")

    print(f"\n  RECOMMENDATION: tunnel_prob_min = {best_thresh}")
    if best_thresh == 0.0:
        print("  -> Tunnel gate has NO predictive power at these thresholds. DISABLE it.")
    else:
        print(f"  -> Set tunnel_prob_min = {best_thresh}")

    return best_thresh


def analyze_traded_physics(df: pd.DataFrame):
    """For traded signals: do physics values predict WIN vs LOSS?"""
    print("\n" + "=" * 70)
    print("TRADED SIGNALS: Physics vs Outcome")
    print("=" * 70)

    traded = df[df['gate'] == 'traded'].copy()
    if len(traded) == 0:
        print("  No traded signals found.")
        return

    wins = traded[traded['trade_result'] == 'WIN']
    losses = traded[traded['trade_result'] == 'LOSS']
    print(f"\n  Traded: {len(traded)} ({len(wins)} W / {len(losses)} L)")

    for col, label in [('mom_rev_ratio', 'Mom/Rev Ratio'),
                        ('hurst', 'Hurst'),
                        ('tunnel_prob', 'Tunnel Prob'),
                        ('velocity', 'Velocity'),
                        ('micro_z', 'Micro Z')]:
        if col not in traded.columns:
            continue
        w_vals = wins[col].dropna()
        l_vals = losses[col].dropna()
        if len(w_vals) > 0 and len(l_vals) > 0:
            print(f"\n  {label}:")
            print(f"    WIN  median={w_vals.median():.3f}  mean={w_vals.mean():.3f}")
            print(f"    LOSS median={l_vals.median():.3f}  mean={l_vals.mean():.3f}")
            # Simple t-test
            from scipy import stats
            t, p = stats.ttest_ind(w_vals, l_vals, equal_var=False)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
            print(f"    t={t:.2f}  p={p:.4f}  {sig}")


def main():
    parser = argparse.ArgumentParser(description="Gate threshold analysis using oracle data")
    parser.add_argument('--dir', default='checkpoints',
                        help='Checkpoint directory with oracle CSVs')
    parser.add_argument('--gate', choices=['5a', '5b', '5c', 'traded', 'all'], default='all',
                        help='Which gate to analyze (default: all)')
    parser.add_argument('--save', action='store_true',
                        help='Save combined analysis to reports/gate_analysis.csv')
    parser.add_argument('--apply', action='store_true',
                        help='Write optimal thresholds to checkpoints/gate_thresholds.json')
    args = parser.parse_args()

    print("=" * 70)
    print("GATE THRESHOLD ANALYZER (oracle-driven)")
    print("=" * 70)
    print(f"  Source: {args.dir}")

    # Load data — prefer signal log (has traded + skipped), fall back to oracle + FN
    df = load_signal_log(args.dir)
    if len(df) == 0:
        # Combine oracle trade log + FN log
        trades = load_oracle_log(args.dir)
        fn = load_fn_log(args.dir)
        if len(trades) > 0 or len(fn) > 0:
            df = pd.concat([trades, fn], ignore_index=True)
            print(f"  Combined: {len(df)} records")

    if len(df) == 0:
        print("\n  ERROR: No oracle data found. Run a forward pass first:")
        print("    python training/trainer.py --forward-pass --data DATA/ATLAS_1DAY")
        return

    # Check for physics columns
    has_physics = 'mom_rev_ratio' in df.columns
    if not has_physics:
        print("\n  WARNING: Physics columns not found in oracle data.")
        print("  Re-run forward pass to populate F_momentum, hurst, etc.")
        print("  Only basic analysis available.\n")

    thresholds = {}

    if args.gate in ('5b', 'all') and has_physics:
        t = analyze_gate_5b(df)
        if t is not None:
            thresholds['momentum_override_ratio'] = t

    if args.gate in ('5a', 'all') and has_physics:
        t = analyze_gate_hurst(df)
        if t is not None:
            thresholds['hurst_min'] = t

    if args.gate in ('5c', 'all') and has_physics:
        t = analyze_gate_tunnel(df)
        if t is not None:
            thresholds['tunnel_prob_min'] = t

    if args.gate in ('traded', 'all'):
        analyze_traded_physics(df)

    if args.save:
        os.makedirs('reports', exist_ok=True)
        out = 'reports/gate_analysis.csv'
        df.to_csv(out, index=False)
        print(f"\n  Saved full data to {out}")

    # Write optimal thresholds JSON
    if thresholds and args.apply:
        import json
        out_path = os.path.join(args.dir, 'gate_thresholds.json')
        # Merge with existing if present
        existing = {}
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                existing = json.load(f)
        existing.update(thresholds)
        with open(out_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\n  THRESHOLDS WRITTEN -> {out_path}")
        for k, v in existing.items():
            print(f"    {k}: {v}")
    elif thresholds:
        print(f"\n  OPTIMAL THRESHOLDS (use --apply to write to {args.dir}/gate_thresholds.json):")
        for k, v in thresholds.items():
            print(f"    {k}: {v}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
