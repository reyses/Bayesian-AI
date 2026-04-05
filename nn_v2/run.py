"""
Runner — single entry point for the entire nn_v2 pipeline.

Commands:
  python nn_v2/run.py build                          # build 79D dataset (overnight)
  python nn_v2/run.py build --days 5                 # build 5 days only
  python nn_v2/run.py nmp 2026-01-06                 # run NMP on 1 day (live path: 1s → agg → SFE → NMP)
  python nn_v2/run.py nmp 2026-01-06 --fast           # run NMP from pre-computed 79D (test path)
  python nn_v2/run.py nmp all --fast                  # run NMP on all OOS days from disk
  python nn_v2/run.py nmp 2026-01-06 --fast --equity 500  # with equity tracking
"""
import sys
import os
import glob
import gc
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_1S = 'DATA/ATLAS/1s'
ATLAS_1M = 'DATA/ATLAS/1m'
FEATURES_DIR = 'DATA/FEATURES_79D'
FEATURES_DIR_1M = 'DATA/FEATURES_79D_1m'


def cmd_build(args):
    """Build 79D dataset."""
    from nn_v2.build_dataset import main as build_main
    sys.argv = ['build_dataset.py'] + args
    build_main()


def cmd_nmp(target, fast=False, equity=None, extra_args=None):
    """Run Nightmare Protocol."""
    from tqdm import tqdm

    if fast:
        _run_nmp_fast(target, equity)
    else:
        _run_nmp_live(target, equity)


def _resolve_days(target: str, source_dir: str) -> list:
    """Resolve target to list of file paths."""
    all_files = sorted(glob.glob(os.path.join(source_dir, '*.parquet')))

    if target == 'all':
        return all_files
    elif target == 'oos':
        return [f for f in all_files if '2026_' in os.path.basename(f)]
    elif target == 'is':
        return [f for f in all_files if '2025_' in os.path.basename(f)]
    elif ',' in target:
        dates = [d.replace('-', '_') for d in target.split(',')]
        return [f for f in all_files
                if any(d in os.path.basename(f) for d in dates)]
    else:
        date_key = target.replace('-', '_')
        return [f for f in all_files if date_key in os.path.basename(f)]


def _run_nmp_fast(target: str, equity: float = None):
    """Run NMP from pre-computed 79D features (fast test mode)."""
    from nn_v2.sfe_ticker import FeatureTicker
    from nn_v2.nightmare import NightmareEngine
    from tqdm import tqdm

    # Try 1m features first (full dataset), fall back to 5s
    feat_files = _resolve_days(target, FEATURES_DIR_1M)
    if not feat_files:
        feat_files = _resolve_days(target, FEATURES_DIR)
    if not feat_files:
        print(f'No feature files found for "{target}" in {FEATURES_DIR}/')
        return

    print(f'NMP (fast mode) — {len(feat_files)} day(s)')
    nmp = NightmareEngine()
    all_results = []
    all_trades = []  # accumulate ALL trades for tree+NN

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')

        # Price file for context
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        nmp.reset()
        ft = FeatureTicker(fpath, price_file=price_file)

        for state in ft:
            nmp.on_state(state)

        nmp.force_close()

        # Accumulate trades with day label
        for t in nmp.trades:
            t['day'] = day_name
        all_trades.extend(nmp.get_full_trades())

        day_pnl = nmp.daily_pnl
        day_trades = len(nmp.trades)
        all_results.append({
            'day': day_name,
            'trades': day_trades,
            'pnl': day_pnl,
            'wr': sum(1 for t in nmp.trades if t['pnl'] > 0) / max(day_trades, 1) * 100,
        })

        tqdm.write(f'  {day_name}: {day_trades} trades  ${day_pnl:>8.2f}')

    # Summary
    _print_summary(all_results)

    # Save trade log (with 79D at entry/exit) for tree+NN training
    if all_trades:
        import pickle
        os.makedirs('DATA/NMP_TRADES', exist_ok=True)
        # Determine label from target
        label = target if target in ('is', 'oos', 'all') else 'custom'
        trade_path = f'DATA/NMP_TRADES/nmp_{label}.pkl'
        with open(trade_path, 'wb') as f:
            pickle.dump(all_trades, f)
        # Also save flat CSV (without 79D arrays) for quick analysis
        flat = []
        for t in all_trades:
            row = {k: v for k, v in t.items() if k not in ('entry_79d', 'exit_79d', 'path')}
            flat.append(row)
        csv_path = f'DATA/NMP_TRADES/nmp_{label}.csv'
        pd.DataFrame(flat).to_csv(csv_path, index=False)
        print(f'\nTrade log saved: {trade_path} ({len(all_trades)} trades)')
        print(f'Trade CSV saved: {csv_path}')


def _run_nmp_live(target: str, equity: float = None):
    """Run NMP from 1s bars (live path: ticker → aggregator → SFE → NMP)."""
    import warnings
    warnings.filterwarnings('ignore', module='numba')

    from nn_v2.ticker import FileTicker
    from nn_v2.aggregator import Aggregator
    from nn_v2.nightmare import NightmareEngine
    from core.statistical_field_engine import StatisticalFieldEngine
    from core.features_79d import extract_79d, FEATURE_NAMES_79D, TF_ORDER, N_FEATURES
    from tqdm import tqdm

    SFE_MIN_BARS = 21

    day_files = _resolve_days(target, ATLAS_1S)
    if not day_files:
        print(f'No 1s files found for "{target}" in {ATLAS_1S}/')
        return

    print(f'NMP (live mode) — {len(day_files)} day(s)')
    print(f'  WARNING: live mode is slow (~10 min/day). Use --fast with pre-built dataset.')
    all_results = []

    for day_file in day_files:
        day_name = os.path.basename(day_file).replace('.parquet', '')
        print(f'\n  {day_name}:')

        sfe = StatisticalFieldEngine()
        agg = Aggregator(history_limit=2000)
        nmp = NightmareEngine()
        prev_velocities = {}

        ticker = FileTicker(day_file)
        pbar = tqdm(ticker, desc=f'    bars', unit='bar', total=len(ticker),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

        last_79d_ts = 0

        def on_bar_close(tf, bar):
            nonlocal prev_velocities, last_79d_ts

            if tf != '1m':
                return

            # Compute 79D from aggregator
            states_by_tf = {}
            ohlcv_by_tf = {}
            for _tf in TF_ORDER:
                df = agg.get_closed_bars_df(_tf)
                if len(df) < SFE_MIN_BARS:
                    continue
                ohlcv_by_tf[_tf] = df
                # SFE on tail for speed
                sfe_in = df.tail(300).reset_index(drop=True) if len(df) > 300 else df
                states = sfe.batch_compute_states(sfe_in)
                if states:
                    states_by_tf[_tf] = states[-1]

            if '1m' not in states_by_tf:
                return

            feat, prev_velocities = extract_79d(
                states_by_tf, ohlcv_by_tf, prev_velocities, bar['timestamp']
            )

            state = {
                'features_79d': feat,
                'price': bar['close'],
                'timestamp': bar['timestamp'],
            }
            nmp.on_state(state)

        agg.on_bar_close = on_bar_close

        for bar in pbar:
            agg.feed(bar)
            pbar.set_postfix_str(f'pnl=${nmp.daily_pnl:+.0f} tr={len(nmp.trades)}')

        nmp.force_close()

        day_pnl = nmp.daily_pnl
        day_trades = len(nmp.trades)
        all_results.append({
            'day': day_name,
            'trades': day_trades,
            'pnl': day_pnl,
            'wr': sum(1 for t in nmp.trades if t['pnl'] > 0) / max(day_trades, 1) * 100,
        })

        print(f'    {nmp.summary()}')

        del sfe, agg
        gc.collect()

    _print_summary(all_results)


def _run_regret():
    """Run regret analysis on IS trades."""
    import pickle
    from nn_v2.regret import compute_all_regrets, summarize_regret_by_branch
    from nn_v2.gate import Gate

    print('Regret Analysis on IS trades...')

    # Load trades
    with open('DATA/NMP_TRADES/nmp_is.pkl', 'rb') as f:
        trades = pickle.load(f)
    print(f'  Loaded {len(trades)} trades')

    # Classify into tree branches
    gate = Gate('DATA/NMP_TREE/tree.pkl')
    import numpy as np
    from core.features_79d import FEATURE_NAMES_79D
    for t in trades:
        feat = np.array(t['entry_79d']).reshape(1, -1)
        feat = np.nan_to_num(feat)
        t['leaf_id'] = int(gate.tree.apply(feat)[0])

    # Compute regret
    regret_df = compute_all_regrets(trades)
    print(f'  Regret computed for {len(regret_df)} trades')

    # Summary
    actual_total = regret_df['actual_pnl'].sum()
    optimal_total = regret_df['best_pnl'].sum()
    total_regret = regret_df['regret'].sum()
    print(f'\n  Actual PnL:  ${actual_total:>10.0f}')
    print(f'  Optimal PnL: ${optimal_total:>10.0f}')
    print(f'  Total regret: ${total_regret:>10.0f}')
    print(f'  Capture rate: {actual_total / max(optimal_total, 1) * 100:.1f}%')

    # Best action distribution
    print(f'\n  Best action distribution:')
    for action, count in regret_df['best_action'].value_counts().items():
        pct = count / len(regret_df) * 100
        avg_pnl = regret_df[regret_df['best_action'] == action]['best_pnl'].mean()
        print(f'    {action:<20} {count:>5} ({pct:>4.0f}%)  avg=${avg_pnl:.1f}')

    # Early entry gain
    avg_early_gain = regret_df['early_entry_gain'].mean()
    early_trades = (regret_df['early_entry_gain'] > 1.0).sum()
    print(f'\n  Entry timing: {early_trades} trades ({early_trades/len(regret_df)*100:.0f}%) '
          f'would benefit from earlier entry (avg gain=${avg_early_gain:.1f})')

    # Per-branch summary
    branch_summary = summarize_regret_by_branch(regret_df)
    print(f'\n  Per-branch regret (top 15 by regret):')
    print(f'  {"Leaf":>5} {"N":>5} {"Actual":>8} {"Optimal":>8} {"Regret":>8} {"Action":>18} {"Pct":>5}')
    print(f'  {"-"*60}')
    for _, row in branch_summary.head(15).iterrows():
        print(f'  {int(row["leaf_id"]):>5} {int(row["n_trades"]):>5} '
              f'${row["actual_total"]:>7.0f} ${row["optimal_total"]:>7.0f} '
              f'${row["total_regret"]:>7.0f} {row["dominant_action"]:>18} '
              f'{row["dominant_pct"]:>4.0f}%')

    # Save
    os.makedirs('DATA/NMP_TREE', exist_ok=True)
    regret_df.to_csv('DATA/NMP_TREE/regret_analysis.csv', index=False)
    branch_summary.to_csv('DATA/NMP_TREE/regret_by_branch.csv', index=False)
    print(f'\n  Saved: DATA/NMP_TREE/regret_analysis.csv')
    print(f'  Saved: DATA/NMP_TREE/regret_by_branch.csv')


def _run_gated(target: str):
    """Run NMP with strategy gate on target data."""
    from nn_v2.sfe_ticker import FeatureTicker
    from nn_v2.nightmare import NightmareEngine
    from nn_v2.gate import Gate
    from tqdm import tqdm
    import numpy as np

    tree_path = 'DATA/NMP_TREE/strategy_tree.pkl'
    if not os.path.exists(tree_path):
        print(f'No tree found at {tree_path}. Run tree.py first.')
        return

    gate = Gate(tree_path)

    feat_files = _resolve_days(target, FEATURES_DIR_1M)
    if not feat_files:
        feat_files = _resolve_days(target, FEATURES_DIR)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return

    print(f'GATED NMP — {len(feat_files)} day(s)')
    nmp = NightmareEngine()
    all_results = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        nmp.reset()
        ft = FeatureTicker(fpath, price_file=price_file)

        for state in ft:
            decision = gate.evaluate(state)

            if nmp.in_pos:
                nmp.on_state(state)
            elif decision['allowed']:
                branch = decision['branch']
                if branch and 'counter' in branch.get('strategy', ''):
                    flipped = state.copy()
                    feat = state['features_79d'].copy()
                    feat[10] = -feat[10]  # flip 1m z_se
                    flipped['features_79d'] = feat
                    nmp.on_state(flipped)
                else:
                    nmp.on_state(state)

        nmp.force_close()

        day_pnl = nmp.daily_pnl
        day_trades = len(nmp.trades)
        all_results.append({
            'day': day_name,
            'trades': day_trades,
            'pnl': day_pnl,
            'wr': sum(1 for t in nmp.trades if t['pnl'] > 0) / max(day_trades, 1) * 100,
        })

        tqdm.write(f'  {day_name}: {day_trades} trades  ${day_pnl:>8.2f}')

    _print_summary(all_results)


def _print_summary(results: list):
    """Print multi-day summary."""
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

    if n_days > 1:
        print(f'\n  Daily breakdown:')
        cumul = 0
        for r in results:
            cumul += r['pnl']
            flag = '<<<' if r['pnl'] > 50 else '!!!' if r['pnl'] < -50 else ''
            print(f'    {r["day"]}  {r["trades"]:>3} trades  {r["wr"]:>4.0f}%  '
                  f'${r["pnl"]:>8.2f}  cumul=${cumul:>8.2f} {flag}')

    print(f'{"="*60}')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'build':
        cmd_build(sys.argv[2:])

    elif cmd == 'nmp':
        target = sys.argv[2] if len(sys.argv) > 2 else 'oos'
        fast = '--fast' in sys.argv
        equity_arg = None
        if '--equity' in sys.argv:
            idx = sys.argv.index('--equity')
            equity_arg = float(sys.argv[idx + 1])
        cmd_nmp(target, fast=fast, equity=equity_arg)

    elif cmd == 'regret':
        _run_regret()

    elif cmd == 'gated':
        target = sys.argv[2] if len(sys.argv) > 2 else 'oos'
        _run_gated(target)

    else:
        print(f'Unknown command: {cmd}')
        print(__doc__)


if __name__ == '__main__':
    main()
