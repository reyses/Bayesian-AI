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

    else:
        print(f'Unknown command: {cmd}')
        print(__doc__)


if __name__ == '__main__':
    main()
