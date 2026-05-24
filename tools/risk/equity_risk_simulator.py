#!/usr/bin/env python
"""
Equity-Based Risk Simulator — Dynamic position sizing from $10 floor.

Model: risk_per_trade = max($10, equity * risk_fraction)
- Start: $10 risk (1 MNQ contract, 20-tick SL)
- As equity builds, scale contracts proportionally
- At $1k equity with 10% fraction: $100 risk (10 contracts)

Simulates equity growth using actual L2 market segments from 1s ATLAS data.

Usage:
    python tools/equity_risk_simulator.py                    # default 10% fraction
    python tools/equity_risk_simulator.py --fraction 0.05    # 5% of equity
    python tools/equity_risk_simulator.py --fraction 0.10 --month 2025_07
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.golden_path import load_1s_index
from tools.l2_risk_budget import analyze_l2_segments

# ── Constants ──────────────────────────────────────────────────────────────────
TICK_SIZE = 0.25
TICK_VALUE = 0.50    # MNQ: $0.50 per tick
POINT_VALUE = 2.0    # MNQ: $2.00 per point
BASE_RISK = 10.0     # Floor: $10 per trade minimum
SL_TICKS = 20        # 20 ticks = 5 points = $10 on 1 contract
MAX_CONTRACTS = 10   # Safety cap (10 MNQ = 1 NQ equivalent)


def simulate_equity(segments, risk_fraction=0.10, base_risk=BASE_RISK,
                    sl_ticks=SL_TICKS, max_contracts=MAX_CONTRACTS):
    """Simulate equity curve with dynamic position sizing.

    For each L2 segment (sorted by timestamp):
      1. Compute current risk budget: max(base_risk, equity * risk_fraction)
      2. Compute contracts: floor(risk_budget / (sl_ticks * tick_value))
      3. If MAE > SL: lose risk_budget. Else: gain MFE * contracts * tick_value.

    Returns dict with equity curves (flat and dynamic) and trade details.
    """
    # Sort segments by timestamp
    segs = sorted(segments, key=lambda s: s['timestamp'])
    n = len(segs)

    # --- Flat sizing (1 contract, $10 risk) ---
    flat_equity = 0.0
    flat_curve = [0.0]
    flat_trades = []

    for s in segs:
        mae = s['mae_ticks']
        mfe = s['mfe_ticks']

        if mae > sl_ticks:
            # Stopped out
            pnl = -sl_ticks * TICK_VALUE
        else:
            # Win: capture MFE
            pnl = mfe * TICK_VALUE

        flat_equity += pnl
        flat_curve.append(flat_equity)
        flat_trades.append({
            'timestamp': s['timestamp'],
            'direction': s['direction'],
            'contracts': 1,
            'risk': base_risk,
            'pnl': pnl,
            'equity': flat_equity,
            'mfe_ticks': mfe,
            'mae_ticks': mae,
            'stopped': mae > sl_ticks,
        })

    # --- Dynamic sizing (equity-based) ---
    dyn_equity = 0.0
    dyn_curve = [0.0]
    dyn_trades = []
    max_dd_flat = 0.0
    max_dd_dyn = 0.0
    peak_flat = 0.0
    peak_dyn = 0.0

    for s in segs:
        mae = s['mae_ticks']
        mfe = s['mfe_ticks']

        # Dynamic risk budget
        risk_budget = max(base_risk, dyn_equity * risk_fraction)
        contracts = max(1, int(risk_budget / (sl_ticks * TICK_VALUE)))
        contracts = min(contracts, max_contracts)
        actual_risk = contracts * sl_ticks * TICK_VALUE

        if mae > sl_ticks:
            # Stopped out: lose actual_risk
            pnl = -actual_risk
        else:
            # Win: capture MFE * contracts
            pnl = mfe * TICK_VALUE * contracts

        dyn_equity += pnl
        dyn_curve.append(dyn_equity)
        dyn_trades.append({
            'timestamp': s['timestamp'],
            'direction': s['direction'],
            'contracts': contracts,
            'risk': actual_risk,
            'pnl': pnl,
            'equity': dyn_equity,
            'mfe_ticks': mfe,
            'mae_ticks': mae,
            'stopped': mae > sl_ticks,
        })

        # Track max drawdown
        peak_flat = max(peak_flat, flat_curve[-1])
        peak_dyn = max(peak_dyn, dyn_curve[-1])
        dd_flat = peak_flat - flat_curve[-1]
        dd_dyn = peak_dyn - dyn_curve[-1]
        max_dd_flat = max(max_dd_flat, dd_flat)
        max_dd_dyn = max(max_dd_dyn, dd_dyn)

    return {
        'flat_curve': flat_curve,
        'flat_trades': flat_trades,
        'flat_final': flat_equity,
        'flat_max_dd': max_dd_flat,
        'dyn_curve': dyn_curve,
        'dyn_trades': dyn_trades,
        'dyn_final': dyn_equity,
        'dyn_max_dd': max_dd_dyn,
        'n_trades': n,
    }


def print_report(result, risk_fraction):
    """Print equity simulation comparison report."""
    n = result['n_trades']
    flat_trades = result['flat_trades']
    dyn_trades = result['dyn_trades']

    flat_wins = sum(1 for t in flat_trades if not t['stopped'])
    dyn_wins = sum(1 for t in dyn_trades if not t['stopped'])
    wr = flat_wins / n * 100 if n > 0 else 0

    print(f"\n{'='*70}")
    print(f"  EQUITY-BASED RISK SIMULATION")
    print(f"{'='*70}")
    print(f"  Model: risk = max(${BASE_RISK:.0f}, equity x {risk_fraction*100:.0f}%)")
    print(f"  SL: {SL_TICKS} ticks (${SL_TICKS * TICK_VALUE:.0f} per contract)")
    print(f"  Trades: {n}")
    print(f"  Win rate: {wr:.1f}% ({flat_wins}/{n})")

    print(f"\n  {'':30s} {'FLAT ($10)':>15s}    {'DYNAMIC':>15s}    {'MULT':>8s}")
    print(f"  {'-'*30} {'-'*15}    {'-'*15}    {'-'*8}")
    print(f"  {'Final equity':30s} ${result['flat_final']:>13,.2f}    "
          f"${result['dyn_final']:>13,.2f}    "
          f"{result['dyn_final']/result['flat_final']:.1f}x" if result['flat_final'] > 0
          else f"  {'Final equity':30s} ${result['flat_final']:>13,.2f}    "
          f"${result['dyn_final']:>13,.2f}    N/A")
    print(f"  {'Max drawdown':30s} ${result['flat_max_dd']:>13,.2f}    "
          f"${result['dyn_max_dd']:>13,.2f}")
    print(f"  {'Avg PnL/trade':30s} "
          f"${result['flat_final']/n:>13,.2f}    "
          f"${result['dyn_final']/n:>13,.2f}")

    # Milestone tracking for dynamic
    print(f"\n  -- EQUITY MILESTONES (dynamic) --")
    milestones = [100, 500, 1000, 5000, 10000, 50000, 100000]
    for m in milestones:
        for i, t in enumerate(dyn_trades):
            if t['equity'] >= m:
                dt = datetime.fromtimestamp(t['timestamp'], tz=timezone.utc)
                print(f"    ${m:>7,} reached at trade #{i+1:>5} "
                      f"({dt:%Y-%m-%d %H:%M}) "
                      f"| {t['contracts']} contracts, ${t['risk']:.0f} risk")
                break
        else:
            print(f"    ${m:>7,} not reached")

    # Contract scaling progression
    print(f"\n  -- POSITION SIZE PROGRESSION --")
    checkpoints = [0, n//10, n//4, n//2, 3*n//4, n-1]
    for cp in checkpoints:
        if cp < len(dyn_trades):
            t = dyn_trades[cp]
            print(f"    Trade #{cp+1:>5}: {t['contracts']:>3} contracts, "
                  f"${t['risk']:>8,.0f} risk, equity=${t['equity']:>12,.2f}")

    # Worst drawdown periods
    print(f"\n  -- WORST LOSING STREAKS (dynamic) --")
    streak = 0
    max_streak = 0
    streak_loss = 0.0
    max_streak_loss = 0.0
    for t in dyn_trades:
        if t['stopped']:
            streak += 1
            streak_loss += t['pnl']
            if streak > max_streak:
                max_streak = streak
                max_streak_loss = streak_loss
        else:
            streak = 0
            streak_loss = 0.0

    print(f"    Longest streak: {max_streak} consecutive losses")
    print(f"    Worst streak $: ${max_streak_loss:,.2f}")

    # What-if: different fractions
    print(f"\n  -- FRACTION SENSITIVITY --")
    print(f"    (showing final equity for different risk fractions)")


def plot_simulation(result, risk_fraction, output_path):
    """4-panel equity simulation visualization."""
    flat_curve = result['flat_curve']
    dyn_curve = result['dyn_curve']
    dyn_trades = result['dyn_trades']
    n = result['n_trades']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.set_facecolor('white')
    for row in axes:
        for ax in row:
            ax.set_facecolor('white')

    # Panel 1: Equity curves comparison
    ax = axes[0, 0]
    ax.plot(range(len(flat_curve)), flat_curve, color='#1565C0', linewidth=1.5,
            label=f'Flat $10 risk (final ${result["flat_final"]:,.0f})', alpha=0.8)
    ax.plot(range(len(dyn_curve)), dyn_curve, color='#FF6F00', linewidth=1.5,
            label=f'Dynamic {risk_fraction*100:.0f}% (final ${result["dyn_final"]:,.0f})', alpha=0.8)
    ax.fill_between(range(len(dyn_curve)), dyn_curve, alpha=0.1, color='#FF6F00')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Equity ($)')
    ax.set_title('EQUITY CURVES: Flat vs Dynamic Sizing', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # Panel 2: Contracts over time
    ax = axes[0, 1]
    contracts = [t['contracts'] for t in dyn_trades]
    ax.plot(range(1, n + 1), contracts, color='#2E7D32', linewidth=1, alpha=0.7)
    ax.fill_between(range(1, n + 1), contracts, alpha=0.15, color='#2E7D32')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Contracts')
    ax.set_title('POSITION SIZE SCALING', fontweight='bold')
    ax.grid(True, alpha=0.15)

    # Panel 3: Per-trade PnL (dynamic)
    ax = axes[1, 0]
    pnls = [t['pnl'] for t in dyn_trades]
    colors = ['#2E7D32' if p >= 0 else '#C62828' for p in pnls]
    ax.bar(range(1, n + 1), pnls, color=colors, alpha=0.6, width=1.0)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('PnL ($)')
    ax.set_title('PER-TRADE PnL (dynamic sizing)', fontweight='bold')
    ax.grid(True, alpha=0.15)

    # Panel 4: Risk per trade
    ax = axes[1, 1]
    risks = [t['risk'] for t in dyn_trades]
    ax.plot(range(1, n + 1), risks, color='#C62828', linewidth=1, alpha=0.7)
    ax.fill_between(range(1, n + 1), risks, alpha=0.1, color='#C62828')
    ax.axhline(y=BASE_RISK, color='#1565C0', linestyle='--', alpha=0.5,
               label=f'Floor ${BASE_RISK:.0f}')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Risk per Trade ($)')
    ax.set_title('RISK BUDGET PROGRESSION', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    mult = result['dyn_final'] / result['flat_final'] if result['flat_final'] > 0 else 0
    fig.suptitle(
        f'EQUITY-BASED RISK SIMULATION\n'
        f'Model: risk = max({BASE_RISK:.0f}, equity x {risk_fraction*100:.0f}%) | '
        f'SL={SL_TICKS}t ({SL_TICKS * TICK_VALUE:.0f}/contract) | '
        f'{n} trades | {mult:.0f}x multiplier',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def run_sensitivity(segments, fractions=None):
    """Run simulation across multiple risk fractions for comparison."""
    if fractions is None:
        fractions = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\n{'='*70}")
    print(f"  RISK FRACTION SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"  {'Fraction':>10s} {'Final Equity':>15s} {'Max DD':>12s} "
          f"{'DD/Equity':>10s} {'Max Contracts':>15s} {'Multiplier':>12s}")
    print(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*10} {'-'*15} {'-'*12}")

    # Baseline: flat
    flat_result = simulate_equity(segments, risk_fraction=0.0, base_risk=BASE_RISK)
    flat_final = flat_result['flat_final']
    print(f"  {'FLAT':>10s} ${flat_final:>13,.2f} ${flat_result['flat_max_dd']:>10,.2f} "
          f"{'N/A':>10s} {'1':>15s} {'1.0x':>12s}")

    results = {}
    for frac in fractions:
        r = simulate_equity(segments, risk_fraction=frac)
        max_c = max(t['contracts'] for t in r['dyn_trades'])
        dd_pct = r['dyn_max_dd'] / r['dyn_final'] * 100 if r['dyn_final'] > 0 else 0
        mult = r['dyn_final'] / flat_final if flat_final > 0 else 0
        print(f"  {frac*100:>9.0f}% ${r['dyn_final']:>13,.2f} ${r['dyn_max_dd']:>10,.2f} "
              f"{dd_pct:>9.1f}% {max_c:>15} {mult:>11.1f}x")
        results[frac] = r

    return results


def main():
    parser = argparse.ArgumentParser(description='Equity-Based Risk Simulator')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--fraction', type=float, default=0.10,
                        help='Risk fraction of equity (default: 0.10 = 10%%)')
    parser.add_argument('--month', default=None,
                        help='Specific month (e.g., 2025_07)')
    parser.add_argument('--l2-min', type=float, default=30.0,
                        help='L2 minimum profit threshold (default: $30)')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run sensitivity analysis across fractions')
    parser.add_argument('--output', default=None,
                        help='Output plot path')
    args = parser.parse_args()

    l2_min_ticks = args.l2_min / TICK_VALUE

    print(f"{'='*70}")
    print(f"EQUITY-BASED RISK SIMULATOR")
    print(f"{'='*70}")
    print(f"  Risk fraction: {args.fraction*100:.0f}% of equity")
    print(f"  Base risk:     ${BASE_RISK:.0f}")
    print(f"  SL:            {SL_TICKS} ticks (${SL_TICKS * TICK_VALUE:.0f})")
    print(f"  L2 min:        ${args.l2_min:.0f}")

    # Load 1s data and extract L2 segments
    print(f"\nLoading 1s data...")
    index_1s = load_1s_index(args.data_dir)

    months = [args.month] if args.month else sorted(index_1s.keys())
    all_segments = []

    for month_key in months:
        print(f"  Processing {month_key}...")
        df_1s = pd.read_parquet(index_1s[month_key])
        if 'timestamp' in df_1s.columns:
            if pd.api.types.is_datetime64_any_dtype(df_1s['timestamp']):
                df_1s['timestamp'] = df_1s['timestamp'].astype('int64') // 10**9
            df_1s = df_1s.sort_values('timestamp').reset_index(drop=True)

        segs = analyze_l2_segments(df_1s, l2_min_ticks=l2_min_ticks)
        print(f"    {month_key}: {len(segs)} L2 segments")
        all_segments.extend(segs)

    print(f"\n  Total L2 segments: {len(all_segments)}")

    if not all_segments:
        print("No L2 segments found.")
        return

    # Run simulation
    result = simulate_equity(all_segments, risk_fraction=args.fraction)
    print_report(result, args.fraction)

    # Sensitivity analysis
    if args.sensitivity:
        run_sensitivity(all_segments)

    # Plot
    out_path = args.output or f'tools/plots/standalone/equity_sim/equity_sim_{args.fraction*100:.0f}pct.png'
    plot_simulation(result, args.fraction, out_path)


if __name__ == '__main__':
    main()
