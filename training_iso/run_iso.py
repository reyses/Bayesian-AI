"""
Isolated pipeline V2 — complete NMP, no tiers, no CNN.

Pure physics baseline per the original Nightmare Protocol (§7):
  * Two-mode NMP entry (NMP_FADE + NMP_RIDE) gated on |z|>ROCHE.
  * Bounded regret analysis (10 min before entry / 30 min after exit).
  * Peak-validity gate on EXTENDED options (post-horizon peak must
    exceed in-trade peak to count as a "hold longer" signal).
  * No CNN overlay. No chains. No negative exits.

Usage:
    python training_iso/run_iso.py
"""
import os
import sys
import glob
import time as _time
import subprocess
import pickle
import shutil
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Features live inside the atlas folder after the refactor
FEATURES_DIR_SEQ = 'DATA/ATLAS/FEATURES_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
ATLAS_5S = 'DATA/ATLAS/5s'   # source for slope (β) computation — available to any tier
OUTPUT_DIR = 'training_iso/output'


def _resolve_days(target, source_dir):
    all_files = sorted(glob.glob(os.path.join(source_dir, '*.parquet')))
    if target == 'is':
        return [f for f in all_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        return [f for f in all_files if '2026_' in os.path.basename(f)]
    return all_files


def _print_summary(results):
    if not results:
        print('No results.')
        return
    n_days = len(results)
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    winning_days = sum(1 for r in results if r['pnl'] > 0)
    print(f'\n{"="*60}')
    print(f'RESULTS: {n_days} days | {total_trades} trades | ${total_pnl:.2f}')
    print(f'  $/day: ${total_pnl / max(n_days, 1):.2f}')
    print(f'  Winning days: {winning_days}/{n_days}')
    print(f'{"="*60}')


def run_iso_forward(target='is', only_tiers=None):
    """Run isolated forward pass — each tier gets its OWN engine, no interference.

    By default all 9 tiers run in parallel on the same bar stream. Pass
    only_tiers=['TREND_FOLLOWER'] (etc) to restrict to a subset — useful
    for fast single-tier iteration.

    Output trades carry `entry_tier` so per-tier metrics roll up cleanly.
    """
    from training_iso.nightmare_iso import IsoEngine, TIER_PRIORITY
    from training.sfe_ticker import FeatureTicker
    from tqdm import tqdm
    from collections import Counter

    feat_files = _resolve_days(target, FEATURES_DIR_SEQ)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return [], []

    active_tiers = only_tiers if only_tiers else TIER_PRIORITY
    print(f'ISO FORWARD — {len(feat_files)} day(s), tiers: {active_tiers}')
    all_results = []
    all_trades = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None
        # 5s closes for slope (β) computation — loaded once, shared across engines
        sec_file = os.path.join(ATLAS_5S, f'{day_name}.parquet')
        sec_df = pd.read_parquet(sec_file) if os.path.exists(sec_file) else None

        engines = {t: IsoEngine(only_tier=t) for t in active_tiers}
        for eng in engines.values():
            eng.set_sec_closes(sec_df)
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            for eng in engines.values():
                eng.on_state(state)
        for eng in engines.values():
            eng.force_close()

        day_trades = []
        for tier, eng in engines.items():
            for t in eng.trades:
                t['day'] = day_name
            day_trades.extend(eng.get_full_trades())
        all_trades.extend(day_trades)

        day_pnl = sum(eng.daily_pnl for eng in engines.values())
        n_trades = len(day_trades)
        all_results.append({
            'day': day_name,
            'trades': n_trades,
            'pnl': day_pnl,
            'wr': sum(1 for t in day_trades if t['pnl'] > 0) / max(n_trades, 1) * 100,
        })

    _print_summary(all_results)

    if all_trades:
        print()
        print(f'{"Tier":<17} {"N":>6} {"WR":>5} {"Total":>10} {"$/trade":>9}')
        print('-' * 55)
        tiers = Counter(t.get('entry_tier', '?') for t in all_trades)
        for tier in active_tiers:
            count = tiers.get(tier, 0)
            if count == 0:
                print(f'{tier:<17} {0:>6}  (no trades)')
                continue
            sub = [t for t in all_trades if t.get('entry_tier') == tier]
            wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            total = sum(t['pnl'] for t in sub)
            per = total / len(sub)
            print(f'{tier:<17} {count:>6,} {wr:>4.0f}% ${total:>+9,.0f} ${per:>+8.2f}')

    return all_results, all_trades


def run_regret():
    """Run bounded regret + produce corrected trades.

    Uses training_iso/regret.py which caps the counterfactual window to
    LOOKBACK_MIN=10 / LOOKAHEAD_MIN=30, gates EXTENDED options on the
    peak-validity check, and produces corrected trades that exit at the
    SPECIFIC peak bar each gated best_action points at (not the overall
    argmax).
    """
    from training_iso.regret import compute_all_regrets, correct_trades

    trade_path = os.path.join(OUTPUT_DIR, 'trades', 'iso_is.pkl')
    with open(trade_path, 'rb') as f:
        trades = pickle.load(f)
    print(f'Regret on {len(trades)} ISO trades...')

    regret_df = compute_all_regrets(trades)
    actual = regret_df['actual_pnl'].sum()
    optimal = regret_df['best_pnl'].sum()
    print(f'  Actual:  ${actual:,.0f}')
    print(f'  Optimal: ${optimal:,.0f}')
    print(f'  Capture: {actual / max(optimal, 1) * 100:.1f}%')

    # Per-tier
    if 'entry_tier' in regret_df.columns:
        for tier in regret_df['entry_tier'].unique():
            sub = regret_df[regret_df['entry_tier'] == tier]
            print(f'    {tier}: {len(sub)} trades, '
                  f'actual=${sub["actual_pnl"].sum():,.0f}, '
                  f'optimal=${sub["best_pnl"].sum():,.0f}')

    # Best action breakdown + extended-validity rates
    n_counter = regret_df['best_action'].str.contains('counter').sum()
    print(f'  Counter: {n_counter} ({n_counter / len(regret_df) * 100:.0f}%)')
    if 'same_extended_valid' in regret_df.columns:
        se_valid = regret_df['same_extended_valid'].sum()
        ce_valid = regret_df['counter_extended_valid'].sum()
        print(f'  Peak-valid extended: same={se_valid} '
              f'({se_valid/len(regret_df)*100:.0f}%)  '
              f'counter={ce_valid} ({ce_valid/len(regret_df)*100:.0f}%)')

    os.makedirs(os.path.join(OUTPUT_DIR, 'tree'), exist_ok=True)
    regret_df.to_csv(os.path.join(OUTPUT_DIR, 'tree', 'regret_analysis.csv'), index=False)

    # ── Corrected trades (oracle ground truth: exit at the gated peak) ─
    print()
    print('Generating corrected trades (peak-aware oracle)...')
    corrected = correct_trades(trades)
    with open(os.path.join(OUTPUT_DIR, 'trades', 'corrected_is.pkl'), 'wb') as f:
        pickle.dump(corrected, f)
    flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
            for t in corrected]
    pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'trades', 'corrected_is.csv'),
                              index=False)

    # Compare actual vs corrected PnL (sanity + delta)
    actual_total = sum(t['original_pnl'] for t in corrected)
    corrected_total = sum(t['pnl'] for t in corrected)
    flips = sum(1 for t in corrected if t['dir'] != t['original_dir'])
    avg_corrected_held = np.mean([t['held'] for t in corrected]) if corrected else 0
    avg_original_held = np.mean([t['original_held'] for t in corrected]) if corrected else 0
    print(f'  Corrected trades: {len(corrected):,}')
    print(f'  Direction flips:  {flips:,} ({flips/max(len(corrected),1)*100:.0f}%)')
    print(f'  Actual    $: ${actual_total:+,.0f}  avg held={avg_original_held:.1f} bars')
    print(f'  Corrected $: ${corrected_total:+,.0f}  avg held={avg_corrected_held:.1f} bars')
    print(f'  Delta:      ${corrected_total - actual_total:+,.0f}  '
          f'({(corrected_total/max(actual_total,1)-1)*100:+.0f}%)')


def main():
    from training_iso.nightmare_iso import TIER_PRIORITY
    args = sys.argv[1:]
    no_regret = '--no-regret' in args

    # --tier TIER_NAME (one or more) restricts run to those tiers only
    only_tiers = None
    if '--tier' in args:
        idx = args.index('--tier')
        # Consume all following positional args until next flag
        tier_args = []
        for a in args[idx + 1:]:
            if a.startswith('--'):
                break
            tier_args.append(a)
        if not tier_args:
            raise SystemExit('--tier requires at least one tier name')
        for t in tier_args:
            if t not in TIER_PRIORITY:
                raise SystemExit(f'unknown tier {t!r}; valid: {TIER_PRIORITY}')
        only_tiers = tier_args

    print(f'{"="*60}')
    print(f'ISOLATED PIPELINE V2 — complete NMP, no tiers, no CNN')
    if only_tiers:
        print(f'  Restricted to tiers: {only_tiers}')
    print(f'  Regret window: 10 min before entry / 30 min after exit'
          + ('  [SKIPPED]' if no_regret else ''))
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}')

    pipeline_start = _time.perf_counter()

    # Phase 1: Collect trades
    print(f'\n{"="*40}')
    print(f'PHASE 1: NMP two-mode forward pass')
    print(f'{"="*40}')
    t0 = _time.perf_counter()
    results, trades = run_iso_forward('is', only_tiers=only_tiers)
    if trades:
        os.makedirs(os.path.join(OUTPUT_DIR, 'trades'), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'trades', 'iso_is.pkl'), 'wb') as f:
            pickle.dump(trades, f)
        flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
                for t in trades]
        pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'trades', 'iso_is.csv'), index=False)
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Phase 2: Regret (optional)
    if not no_regret:
        print(f'\n{"="*40}')
        print(f'PHASE 2: Bounded regret (10/30 min, peak-validity gated)')
        print(f'{"="*40}')
        t0 = _time.perf_counter()
        run_regret()
        print(f'  Done in {_time.perf_counter()-t0:.0f}s')
    else:
        print(f'\n(Phase 2 regret skipped — --no-regret flag)')

    # Summary
    print(f'\n{"="*40}')
    print(f'SUMMARY')
    print(f'{"="*40}')
    elapsed = _time.perf_counter() - pipeline_start
    print(f'ISO V2 PIPELINE COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# Need numpy for flat export
import numpy as np

if __name__ == '__main__':
    main()
