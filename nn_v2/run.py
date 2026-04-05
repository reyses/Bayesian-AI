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
FEATURES_DIR_SEQ = 'DATA/FEATURES_79D_1m_seq'  # honest sequential (sheet music)


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
    """Run NMP with strategy gate. Bayesian memory learns per day. Plots equity."""
    from nn_v2.sfe_ticker import FeatureTicker
    from nn_v2.nightmare import NightmareEngine
    from nn_v2.gate import Gate
    from nn_v2.memory import BayesianMemory
    from tqdm import tqdm
    import numpy as np

    tree_path = 'DATA/NMP_TREE/strategy_tree.pkl'
    if not os.path.exists(tree_path):
        print(f'No tree found at {tree_path}. Run tree.py first.')
        return

    gate = Gate(tree_path)
    memory = BayesianMemory()

    # Commit tree branches as priors
    import pickle
    with open(tree_path, 'rb') as f:
        tree_data = pickle.load(f)
    memory.commit_branches(tree_data['branches'])

    feat_files = _resolve_days(target, FEATURES_DIR_1M)
    if not feat_files:
        feat_files = _resolve_days(target, FEATURES_DIR)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return

    print(f'GATED NMP — {len(feat_files)} day(s) (with Bayesian learning)')
    nmp = NightmareEngine()
    all_results = []
    equity_curve = []
    cumul_pnl = 0

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
                    feat[10] = -feat[10]
                    flipped['features_79d'] = feat
                    nmp.on_state(flipped)
                else:
                    nmp.on_state(state)

        nmp.force_close()

        # Record to Bayesian memory (learn per day)
        for t in nmp.trades:
            entry_feat = np.array(t['entry_79d']).reshape(1, -1)
            entry_feat = np.nan_to_num(entry_feat)
            leaf_id = int(gate.tree.apply(entry_feat)[0])
            memory.record_trade(leaf_id, t['pnl'], t['peak'], t['held'])

        day_pnl = nmp.daily_pnl
        day_trades = len(nmp.trades)
        cumul_pnl += day_pnl
        equity_curve.append(cumul_pnl)

        all_results.append({
            'day': day_name,
            'trades': day_trades,
            'pnl': day_pnl,
            'cumul': cumul_pnl,
            'wr': sum(1 for t in nmp.trades if t['pnl'] > 0) / max(day_trades, 1) * 100,
        })

        tqdm.write(f'  {day_name}: {day_trades} trades  ${day_pnl:>8.2f}  cumul=${cumul_pnl:>8.2f}')

    _print_summary(all_results)

    # Bayesian memory summary
    print(f'\n{memory.summary()}')

    # Save report
    os.makedirs('DATA/NMP_TREE', exist_ok=True)
    report_path = f'DATA/NMP_TREE/gated_{target}_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        n_days = len(all_results)
        total_pnl = sum(r['pnl'] for r in all_results)
        total_trades = sum(r['trades'] for r in all_results)
        winning = sum(1 for r in all_results if r['pnl'] > 0)
        f.write(f'GATED NMP REPORT — {target.upper()}\n')
        f.write(f'{"="*60}\n')
        f.write(f'Days: {n_days} | Trades: {total_trades} | PnL: ${total_pnl:.2f}\n')
        f.write(f'$/day: ${total_pnl / max(n_days, 1):.2f}\n')
        f.write(f'Winning days: {winning}/{n_days} ({winning/max(n_days,1)*100:.0f}%)\n\n')
        f.write(f'Daily breakdown:\n')
        cumul = 0
        for r in all_results:
            cumul += r['pnl']
            flag = '<<<' if r['pnl'] > 50 else '!!!' if r['pnl'] < -50 else ''
            f.write(f'  {r["day"]}  {r["trades"]:>3} trades  {r["wr"]:>4.0f}%  '
                    f'${r["pnl"]:>8.2f}  cumul=${cumul:>8.2f} {flag}\n')
        f.write(f'\n{memory.summary()}\n')
    print(f'\nReport saved: {report_path}')

    csv_path = f'DATA/NMP_TREE/gated_{target}_daily.csv'
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f'CSV saved: {csv_path}')

    # Save memory
    memory.save(f'DATA/NMP_TREE/memory_{target}.pkl')

    # Build OOS playbook from memory (separate from IS book)
    eval_result = memory.evaluate_iteration()
    playbook_lines = [f'OOS PLAYBOOK — {target.upper()}', '=' * 60, '']
    playbook_lines.append(f'Tradeable branches: {eval_result["n_tradeable"]}')
    playbook_lines.append(f'Total trades: {eval_result["total_trades"]}')
    playbook_lines.append(f'Total PnL: ${eval_result["total_pnl"]:.0f}')
    playbook_lines.append(f'Avg WR: {eval_result["avg_wr"]:.0%}')
    playbook_lines.append('')

    # Per-branch comparison: IS expectation vs OOS reality
    playbook_lines.append(f'Per-branch performance:')
    playbook_lines.append(f'{"Leaf":>5} {"N":>5} {"WR":>6} {"AvgPnL":>8} {"TotPnL":>8} {"MaxDD":>7}')
    playbook_lines.append('-' * 45)
    for lid, bs in sorted(memory.branches.items(), key=lambda x: -x[1].total_pnl):
        if bs.n_trades == 0:
            continue
        playbook_lines.append(
            f'{lid:>5} {bs.n_trades:>5} {bs.wr:>5.0%} ${bs.avg_pnl:>7.1f} '
            f'${bs.total_pnl:>7.0f} ${bs.dd_max:>6.0f}'
        )

    # Demote/promote suggestions
    if eval_result['demote_candidates']:
        playbook_lines.append(f'\nDemote candidates (tradeable but losing on OOS):')
        for lid, wr, pnl in eval_result['demote_candidates']:
            playbook_lines.append(f'  Branch {lid}: WR={wr:.0%}, PnL=${pnl:.0f}')

    if eval_result['promote_candidates']:
        playbook_lines.append(f'\nPromote candidates (skipped but winning on OOS):')
        for lid, wr, pnl in eval_result['promote_candidates']:
            playbook_lines.append(f'  Branch {lid}: WR={wr:.0%}, PnL=${pnl:.0f}')

    playbook_path = f'DATA/NMP_TREE/playbook_{target}.txt'
    with open(playbook_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(playbook_lines))
    print(f'Playbook saved: {playbook_path}')

    # Plot equity curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Equity curve
        days = [r['day'].replace('_', '-') for r in all_results]
        ax1 = axes[0]
        ax1.plot(equity_curve, 'b-', linewidth=1.5)
        ax1.fill_between(range(len(equity_curve)), equity_curve, 0,
                         where=[e >= 0 for e in equity_curve], alpha=0.3, color='green')
        ax1.fill_between(range(len(equity_curve)), equity_curve, 0,
                         where=[e < 0 for e in equity_curve], alpha=0.3, color='red')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_title(f'Gated NMP Equity — {target.upper()} ({len(all_results)} days, ${cumul_pnl:,.0f})')
        ax1.set_ylabel('Cumulative PnL ($)')
        ax1.grid(True, alpha=0.3)

        # Daily PnL bars
        ax2 = axes[1]
        daily_pnls = [r['pnl'] for r in all_results]
        colors = ['green' if p >= 0 else 'red' for p in daily_pnls]
        ax2.bar(range(len(daily_pnls)), daily_pnls, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.set_ylabel('Daily PnL ($)')
        ax2.set_xlabel('Day')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f'DATA/NMP_TREE/gated_{target}_equity.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Plot saved: {plot_path}')
    except Exception as e:
        print(f'Plot failed: {e}')


def _run_ai(target: str):
    """Run AI continuous positioning system."""
    from nn_v2.sfe_ticker import FeatureTicker
    from nn_v2.ai import AIEngine
    from tqdm import tqdm

    # Prefer sequential (honest) features, fall back to bulk
    feat_files = _resolve_days(target, FEATURES_DIR_SEQ)
    if feat_files:
        print(f'  Using SEQUENTIAL features (honest)')
    else:
        feat_files = _resolve_days(target, FEATURES_DIR_1M)
    if not feat_files:
        feat_files = _resolve_days(target, FEATURES_DIR)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return

    print(f'AI CONTINUOUS POSITIONING — {len(feat_files)} day(s)')
    ai = AIEngine()
    all_results = []
    cumul = 0

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        ai.reset()
        ft = FeatureTicker(fpath, price_file=price_file)

        for state in ft:
            ai.on_state(state)

        ai.force_close()

        cumul += ai.daily_pnl
        all_results.append({
            'day': day_name,
            'trades': len(ai.trades),
            'pnl': ai.daily_pnl,
            'wr': sum(1 for t in ai.trades if t['pnl'] > 0) / max(len(ai.trades), 1) * 100,
        })

        tqdm.write(f'  {day_name}: {ai.summary()}  cumul=${cumul:.0f}')

    _print_summary(all_results)

    # Save report
    os.makedirs('DATA/NMP_TREE', exist_ok=True)
    report_path = f'DATA/NMP_TREE/ai_{target}_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        n_days = len(all_results)
        total_pnl = sum(r['pnl'] for r in all_results)
        winning = sum(1 for r in all_results if r['pnl'] > 0)
        f.write(f'AI REPORT — {target.upper()}\n{"="*60}\n')
        f.write(f'Days: {n_days} | PnL: ${total_pnl:.2f} | $/day: ${total_pnl/max(n_days,1):.2f}\n')
        f.write(f'Winning: {winning}/{n_days} ({winning/max(n_days,1)*100:.0f}%)\n\n')
        cumul = 0
        for r in all_results:
            cumul += r['pnl']
            flag = '<<<' if r['pnl'] > 50 else '!!!' if r['pnl'] < -50 else ''
            f.write(f'  {r["day"]}  {r["trades"]:>3} trades  {r["wr"]:>4.0f}%  '
                    f'${r["pnl"]:>8.2f}  cumul=${cumul:>8.2f} {flag}\n')
    print(f'\nReport saved: {report_path}')

    csv_path = f'DATA/NMP_TREE/ai_{target}_daily.csv'
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f'CSV saved: {csv_path}')


def _run_full_pipeline():
    """Full pipeline: NMP → regret → tree → book → brain → AI → report.

    All on honest sequential IS features. One command.
    """
    import time as _time

    print(f'{"="*60}')
    print(f'FULL PIPELINE — NMP → regret → tree → book → brain → AI → report')
    print(f'{"="*60}')
    pipeline_start = _time.perf_counter()

    # Step 1: NMP on IS (generates trades with approach buffer)
    print(f'\n--- STEP 1: NMP on IS ---')
    t0 = _time.perf_counter()
    cmd_nmp('is', fast=True)
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Step 2: Regret analysis (full counterfactual per trade)
    print(f'\n--- STEP 2: Regret Analysis ---')
    t0 = _time.perf_counter()
    _run_regret()
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Step 3: Train tree (frozen after this — no retraining)
    print(f'\n--- STEP 3: Train Strategy Tree ---')
    t0 = _time.perf_counter()
    from nn_v2.tree import main as tree_main
    sys.argv = ['tree.py']
    tree_main()
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Step 4: Build book (raw strategy + regret profiles per leaf)
    print(f'\n--- STEP 4: Build Strategy Book ---')
    t0 = _time.perf_counter()
    from nn_v2.book import main as book_main
    book_main()
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Step 5: Per-day brain builder on IS (brain accumulates evidence)
    print(f'\n--- STEP 5: Per-Day Brain Builder (IS) ---')
    t0 = _time.perf_counter()
    from nn_v2.per_day import main as per_day_main
    sys.argv = ['per_day.py', '--target', 'is']
    per_day_main()
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Step 6: Per-day brain builder on OOS (validation — separate brain)
    print(f'\n--- STEP 6: Per-Day Brain Builder (OOS) ---')
    t0 = _time.perf_counter()
    sys.argv = ['per_day.py', '--target', 'oos']
    per_day_main()
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Step 7: Final report (IS/OOS comparison, modes, typical month)
    print(f'\n--- STEP 7: System Report ---')
    from nn_v2.report import main as report_main
    sys.argv = ['report.py']
    report_main()

    total_time = _time.perf_counter() - pipeline_start
    print(f'\n{"="*60}')
    print(f'PIPELINE COMPLETE — {total_time:.0f}s total')
    print(f'{"="*60}')
    print(f'  IS brain:   DATA/NMP_TREE/memory_is.pkl')
    print(f'  OOS brain:  DATA/NMP_TREE/memory_oos.pkl')
    print(f'  Report:     DATA/NMP_TREE/system_report.txt')
    print(f'  Book:       DATA/NMP_TREE/strategy_book.txt')


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

    elif cmd == 'ai':
        target = sys.argv[2] if len(sys.argv) > 2 else 'oos'
        _run_ai(target)

    elif cmd == 'pipeline':
        _run_full_pipeline()

    else:
        print(f'Unknown command: {cmd}')
        print(__doc__)


if __name__ == '__main__':
    main()
