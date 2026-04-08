"""
Probabilistic Forward Pass — IS/OOS evaluation with 4-brain cascade.

Runs the ProbabilisticTradingEngine over IS and OOS data:
  - IS pass: builds IS brain from trade outcomes → freeze
  - OOS pass: continues with IS brain → builds OOS brain → freeze
  - Reports: per-day PnL, trade log, brain divergence

Usage:
  python -m training.train_probabilistic_forward --tf 15s --fresh
  python -m training.train_probabilistic_forward --tf 15s --oos-only
"""
import argparse
import csv
import gc
import glob
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.brain_cascade import BrainCascade
from core.probabilistic_engine import ProbabilisticTradingEngine, ProbConfig
from core.statistical_field_engine import StatisticalFieldEngine
from training.train_probabilistic import (
    ProbabilisticTrajectory, extract_features_22d, N_FEAT, N_HORIZONS,
    FEATURE_NAMES_22D, ATLAS_ROOT
)
from tools.level_shapes import get_levels

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

TICK = 0.25
OOS_START = '2026_02'  # IS/OOS boundary
CHECKPOINT_DIR = 'checkpoints/probabilistic'
BRAIN_DIR = 'checkpoints/brains'
REPORTS_DIR = 'reports/probabilistic'


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load frozen ProbabilisticTrajectory CNN."""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model = ProbabilisticTrajectory(n_features=N_FEAT, n_horizons=N_HORIZONS).to(dev)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, dev


def load_template_library(checkpoint_dir: str = 'checkpoints'):
    """Load templates + scaler + centroids from existing clustering checkpoint."""
    scaler_path = os.path.join(checkpoint_dir, 'clustering_scaler.pkl')
    templates_path = os.path.join(checkpoint_dir, 'templates.pkl')

    scaler, centroids_scaled, valid_tids, pattern_library = None, None, [], {}

    if os.path.exists(scaler_path) and os.path.exists(templates_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(templates_path, 'rb') as f:
            templates = pickle.load(f)

        # Build pattern library from templates
        centroids = []
        for t in templates:
            tid = t.template_id
            valid_tids.append(tid)
            centroids.append(t.centroid)
            pattern_library[tid] = {
                'p75_mfe_ticks': getattr(t, 'p75_mfe_ticks', 50),
                'p25_mae_ticks': getattr(t, 'p25_mae_ticks', 20),
                'mean_mfe_ticks': getattr(t, 'mean_mfe_ticks', 30),
                'mean_mae_ticks': getattr(t, 'mean_mae_ticks', 15),
                'expected_peak_bar': getattr(t, 'avg_mfe_bar', 0.5),
                'giveback_pct': getattr(t, 'giveback_pct', 0.35),
                'long_bias': getattr(t, 'direction_bias', 0.5),
            }

        if centroids:
            centroids_arr = np.array(centroids)
            # Pad centroids if scaler expects more dims
            expected_dim = getattr(scaler, 'n_features_in_', centroids_arr.shape[-1])
            if centroids_arr.shape[-1] < expected_dim:
                pad = np.zeros((len(centroids_arr), expected_dim - centroids_arr.shape[-1]))
                centroids_arr = np.concatenate([centroids_arr, pad], axis=-1)
            centroids_scaled = scaler.transform(centroids_arr)

        print(f"  Templates: {len(valid_tids)} loaded from {checkpoint_dir}")
    else:
        print(f"  Templates: none found (will use default exit params)")

    return scaler, centroids_scaled, valid_tids, pattern_library


def get_levels_for_month(month_str: str):
    """Get levels for a YYYY_MM month string."""
    m = month_str.replace('_', '-')
    return get_levels(m)


def run_forward_pass(model, device, tf: str, files: list, mode: str,
                     cascade: BrainCascade, config: ProbConfig,
                     scaler, centroids_scaled, valid_tids, pattern_library):
    """Run probabilistic forward pass over a set of parquet files.

    Args:
        mode: 'is' or 'oos'
    Returns:
        list of trade records, daily ledger
    """
    engine = ProbabilisticTradingEngine(
        cnn_model=model,
        brain_cascade=cascade,
        template_library=pattern_library,
        scaler=scaler,
        centroids_scaled=centroids_scaled,
        valid_tids=valid_tids,
        config=config,
        device=str(device),
    )

    all_trades = []
    daily_ledger = []
    sfe = StatisticalFieldEngine()

    pbar = tqdm(files, desc=f"  {mode.upper()} pass", unit='file')

    for fpath in pbar:
        fname = os.path.basename(fpath).replace('.parquet', '')
        month_str = fname[:7]  # YYYY_MM

        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        if len(df) < config.lookback + 20:
            continue

        # Compute features
        states = sfe.batch_compute_states(df)
        levels = get_levels_for_month(month_str)
        feats_22d = extract_features_22d(states, df, levels)
        del states; gc.collect()

        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'].values

        # Track per-day stats
        current_day = None
        day_trades = []
        day_date = None

        for i in range(len(df)):
            row_day = int(timestamps[i]) // 86400
            if row_day != current_day:
                # End of previous day
                if current_day is not None and day_trades:
                    day_pnl = sum(t['pnl'] for t in day_trades)
                    day_wins = sum(1 for t in day_trades if t['result'] == 'WIN')
                    daily_ledger.append({
                        'date': day_date,
                        'trades': len(day_trades),
                        'wins': day_wins,
                        'pnl': day_pnl,
                    })
                current_day = row_day
                day_date = datetime.utcfromtimestamp(int(timestamps[i])).strftime('%Y-%m-%d')
                day_trades = []

            result = engine.process_bar(
                bar_index=i,
                price=prices[i],
                high=highs[i],
                low=lows[i],
                timestamp=timestamps[i],
                features_22d=feats_22d[i],
            )

            if result.trade_closed:
                all_trades.append(result.trade_closed)
                day_trades.append(result.trade_closed)

        # Force close at end of file
        if engine.in_position:
            result = engine._close_position(
                prices[-1], timestamps[-1], len(df) - 1,
                'end_of_data', 0, None, 0.5)
            if result.trade_closed:
                all_trades.append(result.trade_closed)
                day_trades.append(result.trade_closed)

        # Final day
        if day_trades:
            day_pnl = sum(t['pnl'] for t in day_trades)
            day_wins = sum(1 for t in day_trades if t['result'] == 'WIN')
            daily_ledger.append({
                'date': day_date,
                'trades': len(day_trades),
                'wins': day_wins,
                'pnl': day_pnl,
            })

        # Update progress bar
        n_trades = engine.total_trades
        wr = engine.win_rate * 100
        pbar.set_postfix_str(
            f'{fname} | {n_trades} trades | ${engine.total_pnl:,.0f} | {wr:.0f}% WR',
            refresh=False)

        engine.reset()
        del df, feats_22d; gc.collect()

    pbar.close()
    return all_trades, daily_ledger


def write_report(trades, daily_ledger, mode: str, output_dir: str):
    """Write summary report and trade log."""
    os.makedirs(output_dir, exist_ok=True)

    total_pnl = sum(t['pnl'] for t in trades)
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t['result'] == 'WIN')
    wr = n_wins / n_trades * 100 if n_trades else 0
    n_days = len(daily_ledger)
    per_day = total_pnl / n_days if n_days else 0

    # Summary
    report_path = os.path.join(output_dir, f'{mode}_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"PROBABILISTIC {mode.upper()} FORWARD PASS\n")
        f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"  Trades:    {n_trades}\n")
        f.write(f"  Win Rate:  {wr:.1f}%\n")
        f.write(f"  Total PnL: ${total_pnl:,.2f}\n")
        f.write(f"  Per Day:   ${per_day:,.2f}\n")
        f.write(f"  Days:      {n_days}\n\n")

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t['exit_reason']
            if r not in exit_reasons:
                exit_reasons[r] = {'n': 0, 'pnl': 0.0, 'wins': 0}
            exit_reasons[r]['n'] += 1
            exit_reasons[r]['pnl'] += t['pnl']
            if t['result'] == 'WIN':
                exit_reasons[r]['wins'] += 1

        f.write(f"  EXIT REASON BREAKDOWN:\n")
        f.write(f"  {'Reason':<25} {'N':>6} {'WR%':>6} {'PnL':>12} {'Avg':>10}\n")
        f.write(f"  {'-' * 65}\n")
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]['n']):
            wr_r = stats['wins'] / stats['n'] * 100 if stats['n'] else 0
            avg = stats['pnl'] / stats['n'] if stats['n'] else 0
            f.write(f"  {reason:<25} {stats['n']:>6} {wr_r:>5.1f}% ${stats['pnl']:>10,.2f} ${avg:>8,.2f}\n")

        # Duration breakdown
        f.write(f"\n  HOLD DURATION BREAKDOWN:\n")
        duration_bins = {'<30s': [], '30s-1m': [], '1-2m': [], '2-5m': [], '5m+': []}
        for t in trades:
            bars = t.get('bars_held', 0)
            secs = bars * 15  # 15s bars
            if secs < 30:
                duration_bins['<30s'].append(t['pnl'])
            elif secs < 60:
                duration_bins['30s-1m'].append(t['pnl'])
            elif secs < 120:
                duration_bins['1-2m'].append(t['pnl'])
            elif secs < 300:
                duration_bins['2-5m'].append(t['pnl'])
            else:
                duration_bins['5m+'].append(t['pnl'])

        f.write(f"  {'Duration':<12} {'N':>6} {'WR%':>6} {'PnL':>12} {'Avg':>10}\n")
        f.write(f"  {'-' * 50}\n")
        for dur, pnls in duration_bins.items():
            if pnls:
                n = len(pnls)
                w = sum(1 for p in pnls if p > 0)
                f.write(f"  {dur:<12} {n:>6} {w/n*100:>5.1f}% ${sum(pnls):>10,.2f} ${sum(pnls)/n:>8,.2f}\n")

        # Daily ledger
        f.write(f"\n  DAILY LEDGER:\n")
        f.write(f"  {'Date':<12} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PnL':>12}\n")
        f.write(f"  {'-' * 50}\n")
        cumul = 0.0
        for day in daily_ledger:
            cumul += day['pnl']
            wr_d = day['wins'] / day['trades'] * 100 if day['trades'] else 0
            f.write(f"  {day['date']:<12} {day['trades']:>7} {day['wins']:>5} "
                    f"{wr_d:>5.1f}% ${day['pnl']:>10,.2f}\n")

    print(f"  Report: {report_path}")

    # Trade log CSV
    csv_path = os.path.join(output_dir, f'{mode}_trade_log.csv')
    if trades:
        keys = trades[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trades)
        print(f"  Trade log: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Probabilistic Forward Pass')
    parser.add_argument('--tf', default='15s', help='Timeframe for forward pass')
    parser.add_argument('--fresh', action='store_true', help='Clear brains and run IS + OOS')
    parser.add_argument('--oos-only', action='store_true', help='Skip IS, run OOS with existing IS brain')
    parser.add_argument('--oos-start', default=OOS_START, help='OOS boundary (YYYY_MM)')
    parser.add_argument('--model-path', default=None, help='Path to CNN checkpoint')
    parser.add_argument('--entry-threshold', type=float, default=0.65)
    parser.add_argument('--hold-threshold', type=float, default=0.45)
    parser.add_argument('--min-hold', type=int, default=2, help='Min hold bars before soft exits')
    parser.add_argument('--sl-ticks', type=float, default=40.0)
    parser.add_argument('--tp-ticks', type=float, default=80.0)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PROBABILISTIC FORWARD PASS — 4-Brain Cascade")
    print("=" * 70)

    # Load CNN model
    model_path = args.model_path or os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"  ERROR: CNN checkpoint not found at {model_path}")
        print(f"  Run: python -m training.train_probabilistic --tf {args.tf}")
        return

    model, device = load_model(model_path)
    print(f"  CNN loaded: {model_path}")

    # Load templates (optional — for exit params)
    scaler, centroids_scaled, valid_tids, pattern_library = load_template_library()

    # Config
    config = ProbConfig(
        entry_threshold=args.entry_threshold,
        hold_threshold=args.hold_threshold,
        min_hold_bars=args.min_hold,
        default_sl_ticks=args.sl_ticks,
        default_tp_ticks=args.tp_ticks,
    )
    print(f"  Config: entry>{config.entry_threshold} hold>{config.hold_threshold} "
          f"SL={config.default_sl_ticks}t TP={config.default_tp_ticks}t "
          f"min_hold={config.min_hold_bars} bars")

    # Brain cascade
    cascade = BrainCascade(checkpoint_dir=BRAIN_DIR)
    if args.fresh:
        print(f"  Fresh run — clearing brains")
    elif not args.oos_only:
        # Try loading existing brains
        cascade.load_all()

    # Split files into IS / OOS
    tf_dir = os.path.join(ATLAS_ROOT, args.tf)
    all_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    is_files = [f for f in all_files if os.path.basename(f)[:7] < args.oos_start]
    oos_files = [f for f in all_files if os.path.basename(f)[:7] >= args.oos_start]

    print(f"  Data: {len(is_files)} IS files, {len(oos_files)} OOS files")
    print(f"  OOS boundary: {args.oos_start}")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── IS PASS ──────────────────────────────────────────────────
    if not args.oos_only:
        print(f"\n{'=' * 50}")
        print(f"IS FORWARD PASS")
        print(f"{'=' * 50}")

        cascade.init_is_brain()
        is_trades, is_ledger = run_forward_pass(
            model, device, args.tf, is_files, 'is',
            cascade, config, scaler, centroids_scaled, valid_tids, pattern_library)

        cascade.freeze_is()
        write_report(is_trades, is_ledger, 'is', REPORTS_DIR)

        is_pnl = sum(t['pnl'] for t in is_trades)
        is_wr = sum(1 for t in is_trades if t['result'] == 'WIN') / max(len(is_trades), 1) * 100
        print(f"\n  IS RESULT: {len(is_trades)} trades | ${is_pnl:,.2f} PnL | {is_wr:.1f}% WR")

    # ── OOS PASS ─────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"OOS FORWARD PASS")
    print(f"{'=' * 50}")

    cascade.init_oos_brain()
    oos_trades, oos_ledger = run_forward_pass(
        model, device, args.tf, oos_files, 'oos',
        cascade, config, scaler, centroids_scaled, valid_tids, pattern_library)

    cascade.freeze_oos()
    write_report(oos_trades, oos_ledger, 'oos', REPORTS_DIR)

    oos_pnl = sum(t['pnl'] for t in oos_trades)
    oos_wr = sum(1 for t in oos_trades if t['result'] == 'WIN') / max(len(oos_trades), 1) * 100
    n_oos_days = len(oos_ledger)
    per_day = oos_pnl / n_oos_days if n_oos_days else 0

    print(f"\n  OOS RESULT: {len(oos_trades)} trades | ${oos_pnl:,.2f} PnL | "
          f"{oos_wr:.1f}% WR | ${per_day:,.2f}/day")

    # ── Brain divergence ─────────────────────────────────────────
    report = cascade.divergence_report()
    div_path = os.path.join(REPORTS_DIR, 'brain_divergence.txt')
    with open(div_path, 'w') as f:
        f.write(report)
    print(f"\n  Brain divergence: {div_path}")

    print(f"\n{'=' * 70}")
    print(f"  Baseline to beat: TradeCNN $1,609/day OOS")
    print(f"  This run:         ${per_day:,.2f}/day OOS")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
