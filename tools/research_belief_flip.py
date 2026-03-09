"""
Research Workbench: Trade-Aware Belief Flip

Replays OOS trades bar-by-bar with TWO TBNs side by side:
  1. BASELINE: standard TBN (workers blind to trade context)
  2. TRADE-AWARE: modified TBN (workers know entry price, PnL, bars held, stop dist)

Compares worker consensus flip signals between the two to measure whether
trade awareness improves the workers' ability to distinguish bad trades.

Usage:
    python tools/research_belief_flip.py --data DATA/ATLAS_OOS
    python tools/research_belief_flip.py --data DATA/ATLAS_OOS --trade-aware-only
"""

import argparse
import csv
import json
import os
import sys
import glob
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.statistical_field_engine import StatisticalFieldEngine
from core.timeframe_belief_network import TimeframeBeliefNetwork  # baseline
from tools.research.tbn_trade_aware import TimeframeBeliefNetwork as TradeAwareTBN  # experiment
from config.symbols import MNQ


# ── Helpers ──────────────────────────────────────────────────────────────

def load_checkpoint(name):
    path = os.path.join(PROJECT_ROOT, 'checkpoints', name)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run --forward-pass first.")
        sys.exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_trade_log(preset='oos'):
    """Load trade log matching the data preset.
    OOS presets (1m, oos) use OOS trade log.
    IS presets (1d, 1w, 1y) use IS trade log.
    """
    if preset in ('1m', 'oos'):
        candidates = [
            os.path.join(PROJECT_ROOT, 'checkpoints', 'oos_trade_log.csv'),
            os.path.join(PROJECT_ROOT, 'reports', 'oos', 'oracle_trade_log.csv'),
        ]
        label = 'OOS'
    else:
        candidates = [
            os.path.join(PROJECT_ROOT, 'reports', 'is', 'oracle_trade_log.csv'),
        ]
        label = 'IS'
    for path in candidates:
        if os.path.exists(path):
            trades = []
            with open(path, 'r') as f:
                for row in csv.DictReader(f):
                    trades.append(row)
            print(f"  Loaded {len(trades)} {label} trades from {path}")
            return trades
    print(f"ERROR: No {label} trade log found. Run --forward-pass first.")
    sys.exit(1)


def worker_consensus(tbn: TimeframeBeliefNetwork, trade_side: str):
    """
    Check how many TBN workers disagree with the trade side RIGHT NOW.

    Returns dict with:
      flip_count, total_workers, weighted_flip_score,
      big_tf_flipped, per_worker details
    """
    TF_WEIGHTS = {
        14400: 5.0, 3600: 4.0, 1800: 3.5, 900: 3.0, 300: 2.5,
        180: 2.0, 60: 1.5, 30: 1.0, 15: 0.5, 5: 0.25, 1: 0.1,
    }
    TF_LABELS = {
        14400: '4h', 3600: '1h', 1800: '30m', 900: '15m', 300: '5m',
        180: '3m', 60: '1m', 30: '30s', 15: '15s', 5: '5s', 1: '1s',
    }
    BIG_TFS = {14400, 3600, 1800, 900}  # 4h, 1h, 30m, 15m

    trade_is_long = trade_side.lower() == 'long'
    flip_count = 0
    total_workers = 0
    flip_weight = 0.0
    total_weight = 0.0
    big_tf_flipped = []

    for tf, worker in tbn.workers.items():
        b = worker.current_belief
        if b is None:
            continue
        total_workers += 1
        w = TF_WEIGHTS.get(tf, 0.5)
        total_weight += w

        worker_says_long = b.dir_prob > 0.5
        if trade_is_long != worker_says_long:
            flip_count += 1
            flip_weight += w
            if tf in BIG_TFS:
                big_tf_flipped.append(TF_LABELS.get(tf, str(tf)))

    weighted_score = flip_weight / total_weight if total_weight > 0 else 0.0
    return {
        'flip_count': flip_count,
        'total_workers': total_workers,
        'weighted_flip_score': weighted_score,
        'big_tf_flipped': big_tf_flipped,
    }


# ── Main replay ──────────────────────────────────────────────────────────

def run_research(args):
    tick_size = MNQ.tick_size      # 0.25
    point_value = MNQ.point_value  # 2.0
    tick_value = tick_size * point_value  # 0.50

    # Load checkpoints
    print("Loading checkpoints...")
    pattern_library = load_checkpoint('pattern_library.pkl')
    scaler = load_checkpoint('clustering_scaler.pkl')
    engine = StatisticalFieldEngine(regression_period=21)

    # Valid template IDs
    valid_tids = sorted([tid for tid in pattern_library
                         if isinstance(tid, int) and pattern_library[tid].get('member_count', 0) > 0])
    centroids = np.array([pattern_library[tid]['centroid'] for tid in valid_tids])
    centroids_scaled = scaler.transform(centroids)

    # Load trade log matching the data preset
    preset = getattr(args, 'run', 'oos')
    print(f"Loading trade log for preset '{preset}'...")
    trades = load_trade_log(preset)

    # Build trade lookup: {entry_time -> trade_info}
    trade_index = {}
    for t in trades:
        entry_t = float(t.get('entry_time', 0))
        trade_index[entry_t] = {
            'direction': t.get('direction', ''),
            'entry_price': float(t.get('entry_price', 0)),
            'exit_price': float(t.get('exit_price', 0)),
            'exit_time': float(t.get('exit_time', 0)),
            'pnl': float(t.get('actual_pnl', 0)),
            'exit_reason': t.get('exit_reason', ''),
            'result': t.get('result', ''),
            'oracle_mfe': float(t.get('oracle_mfe', 0)),
            'oracle_mae': float(t.get('oracle_mae', 0)),
        }

    # Scan data time range and filter trades to only those within the data
    data_dir = args.data
    _sample_files = sorted(glob.glob(os.path.join(data_dir, '15s', '*.parquet')))
    if _sample_files:
        _df0 = pd.read_parquet(_sample_files[0])
        _dfN = pd.read_parquet(_sample_files[-1])
        _ts_col = 'timestamp'
        if _ts_col in _df0.columns and not np.issubdtype(_df0[_ts_col].dtype, np.number):
            _df0[_ts_col] = _df0[_ts_col].apply(lambda x: x.timestamp())
            _dfN[_ts_col] = _dfN[_ts_col].apply(lambda x: x.timestamp())
        _data_min = float(_df0[_ts_col].min())
        _data_max = float(_dfN[_ts_col].max())
        before = len(trade_index)
        trade_index = {k: v for k, v in trade_index.items()
                       if _data_min <= k <= _data_max}
        after = len(trade_index)
        if before > 0 and after == 0:
            from datetime import datetime
            print(f"\n  WARNING: 0 of {before} trades fall within data range")
            print(f"  Data range: {datetime.utcfromtimestamp(_data_min)} — {datetime.utcfromtimestamp(_data_max)}")
            print(f"  Trade log has different timestamps. Try --run oos (or run --forward-pass on this data first).\n")
            sys.exit(1)
        elif after < before:
            print(f"  Filtered trades: {before} → {after} (within data time range)")
        del _df0, _dfN

    # Init both TBNs: baseline (blind) and trade-aware
    print("Initializing belief networks (baseline + trade-aware)...")
    tbn_args = dict(
        pattern_library=pattern_library,
        scaler=scaler,
        engine=engine,
        valid_tids=valid_tids,
        centroids_scaled=centroids_scaled,
    )
    tbn = TimeframeBeliefNetwork(**tbn_args)          # baseline
    tbn_ta = TradeAwareTBN(**tbn_args)                # trade-aware

    # Load ATLAS data files
    daily_files = sorted(glob.glob(os.path.join(data_dir, '15s', '*.parquet')))
    if not daily_files:
        print(f"ERROR: No 15s parquet files in {data_dir}/15s/")
        sys.exit(1)
    print(f"  {len(daily_files)} data files to replay")

    # ── Threshold sweep config ───────────────────────────────────────────
    sweep_thresholds = [
        (a, w, s)
        for a in [0, 2, 4, 8, 12]
        for w in [5, 6, 7, 8, 9]
        for s in [0.40, 0.50, 0.60, 0.70, 0.80]
    ]

    # ── Results collection ───────────────────────────────────────────────
    all_trade_bars = []  # per-trade: list of per-bar snapshots
    _sweep_init = lambda: {'flips': 0, 'correct_flips': 0, 'wrong_flips': 0,
                            'saved_pnl': 0.0, 'lost_pnl': 0.0, 'bars_early': []}
    sweep_results = defaultdict(_sweep_init)       # baseline
    sweep_results_ta = defaultdict(_sweep_init)    # trade-aware

    # ── Replay loop ──────────────────────────────────────────────────────
    active_trade = None  # current trade being tracked
    trade_bar_log = []   # per-bar data for current trade
    trades_analyzed = 0

    for file_path in tqdm(daily_files, desc="Replaying files"):
        df_15s = pd.read_parquet(file_path)
        if 'timestamp' in df_15s.columns and not np.issubdtype(df_15s['timestamp'].dtype, np.number):
            df_15s['timestamp'] = df_15s['timestamp'].apply(lambda x: x.timestamp())

        # Load companion TFs for TBN prepare_day
        def _load_fine(tf):
            try:
                _dir = os.path.join(data_dir, tf)
                _name = os.path.basename(file_path)
                _f = os.path.join(_dir, _name)
                if os.path.exists(_f):
                    _df = pd.read_parquet(_f)
                    if 'timestamp' in _df.columns and not np.issubdtype(_df['timestamp'].dtype, np.number):
                        _df['timestamp'] = _df['timestamp'].apply(lambda x: x.timestamp())
                    return _df if not _df.empty else None
            except Exception:
                pass
            return None

        df_5s = _load_fine('5s')
        df_1s = _load_fine('1s')
        df_4h = _load_fine('4h')

        # Prepare both TBNs for this file
        try:
            states_15s = engine.batch_compute_states(df_15s, use_cuda=False)
            _prep = dict(states_micro=states_15s, df_5s=df_5s, df_1s=df_1s, df_4h=df_4h)
            tbn.prepare_day(df_15s, **_prep)
            tbn_ta.prepare_day(df_15s, **_prep)
        except Exception as e:
            _prep = dict(states_micro=[], df_5s=df_5s, df_1s=df_1s, df_4h=df_4h)
            tbn.prepare_day(df_15s, **_prep)
            tbn_ta.prepare_day(df_15s, **_prep)

        _trade_bars_counter = 0

        # Iterate 15s bars
        for bar_i, row in enumerate(df_15s.itertuples()):
            ts = row.timestamp
            price = getattr(row, 'close', 0.0)

            # Tick both TBNs
            tbn.tick_all(bar_i)
            tbn_ta.tick_all(bar_i)

            # Check if a new trade starts at this timestamp
            if active_trade is None and ts in trade_index:
                active_trade = trade_index[ts]
                trade_bar_log = []
                _trade_bars_counter = 0

            # If we're in a trade, feed context + log worker state
            if active_trade is not None:
                direction = active_trade['direction']
                entry_price = active_trade['entry_price']

                # Compute current PnL
                if direction == 'LONG':
                    pnl_ticks = (price - entry_price) / tick_size
                else:
                    pnl_ticks = (entry_price - price) / tick_size

                adverse_ticks = max(0, -pnl_ticks)

                # Feed trade context to trade-aware TBN
                tbn_ta.update_trade_context(
                    side=direction.lower(),
                    pnl_ticks=pnl_ticks,
                    bars_held=_trade_bars_counter,
                    stop_distance_ticks=20.0,  # approximate; real SL varies
                )
                _trade_bars_counter += 1

                # Get consensus from BOTH TBNs
                consensus_base = worker_consensus(tbn, direction)
                consensus_ta = worker_consensus(tbn_ta, direction)

                trade_bar_log.append({
                    'ts': ts,
                    'price': price,
                    'pnl_ticks': pnl_ticks,
                    'adverse_ticks': adverse_ticks,
                    # Baseline
                    'flip_count': consensus_base['flip_count'],
                    'total_workers': consensus_base['total_workers'],
                    'weighted_score': consensus_base['weighted_flip_score'],
                    'big_tf_flipped': consensus_base['big_tf_flipped'],
                    # Trade-aware
                    'ta_flip_count': consensus_ta['flip_count'],
                    'ta_weighted_score': consensus_ta['weighted_flip_score'],
                    'ta_big_tf_flipped': consensus_ta['big_tf_flipped'],
                })

                # Check if trade ended
                if ts >= active_trade['exit_time']:
                    # Finalize: analyze this trade's bar log
                    trades_analyzed += 1
                    trade_pnl = active_trade['pnl']
                    is_losing = trade_pnl < 0
                    n_bars = len(trade_bar_log)

                    all_trade_bars.append({
                        'direction': direction,
                        'pnl': trade_pnl,
                        'is_losing': is_losing,
                        'exit_reason': active_trade['exit_reason'],
                        'n_bars': n_bars,
                        'bars': trade_bar_log,
                    })

                    # Sweep BOTH models: baseline and trade-aware
                    def _sweep_one(bar_log, results_dict, fc_key, ws_key):
                        for (min_adv, min_w, min_ws) in sweep_thresholds:
                            key = f"adv>={min_adv:>2}_w>={min_w}_ws>={min_ws:.2f}"
                            for bi, bar in enumerate(bar_log):
                                if (bar['adverse_ticks'] >= min_adv and
                                    bar[fc_key] >= min_w and
                                    bar[ws_key] >= min_ws):
                                    flip_pnl_ticks = bar['pnl_ticks']
                                    flip_pnl = flip_pnl_ticks * tick_value
                                    bars_before_exit = n_bars - bi
                                    if is_losing:
                                        savings = abs(trade_pnl) - abs(flip_pnl)
                                        results_dict[key]['correct_flips'] += 1
                                        results_dict[key]['saved_pnl'] += savings
                                    else:
                                        cost = trade_pnl - flip_pnl
                                        results_dict[key]['wrong_flips'] += 1
                                        results_dict[key]['lost_pnl'] += cost
                                    results_dict[key]['flips'] += 1
                                    results_dict[key]['bars_early'].append(bars_before_exit)
                                    break

                    _sweep_one(trade_bar_log, sweep_results, 'flip_count', 'weighted_score')
                    _sweep_one(trade_bar_log, sweep_results_ta, 'ta_flip_count', 'ta_weighted_score')

                    # Clear trade context
                    tbn_ta.stop_trade_tracking()
                    active_trade = None
                    trade_bar_log = []

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TRADE-AWARE BELIEF FLIP RESEARCH (BASELINE vs TRADE-AWARE)")
    print("=" * 90)

    total_trades = len(all_trade_bars)
    losing_trades = sum(1 for t in all_trade_bars if t['is_losing'])
    winning_trades = total_trades - losing_trades
    print(f"\n  Trades replayed:   {trades_analyzed}")
    print(f"  Losing trades:     {losing_trades}")
    print(f"  Winning trades:    {winning_trades}")

    # ── Worker flip frequency: BASELINE vs TRADE-AWARE ──────────────────
    print(f"\n── WORKER FLIP FREQUENCY (losing vs winning) ──")

    for label, subset in [('LOSING', [t for t in all_trade_bars if t['is_losing']]),
                          ('WINNING', [t for t in all_trade_bars if not t['is_losing']])]:
        if not subset:
            continue
        all_bars = [b for t in subset for b in t['bars']]
        if not all_bars:
            continue
        # Baseline
        avg_flip_b = np.mean([b['flip_count'] for b in all_bars])
        avg_ws_b = np.mean([b['weighted_score'] for b in all_bars])
        peak_b = np.mean([max(b['flip_count'] for b in t['bars']) for t in subset if t['bars']])
        # Trade-aware
        avg_flip_ta = np.mean([b['ta_flip_count'] for b in all_bars])
        avg_ws_ta = np.mean([b['ta_weighted_score'] for b in all_bars])
        peak_ta = np.mean([max(b['ta_flip_count'] for b in t['bars']) for t in subset if t['bars']])
        avg_total = np.mean([b['total_workers'] for b in all_bars])

        print(f"\n  {label} trades ({len(subset)}):")
        print(f"    {'':25} {'BASELINE':>10} {'TRADE-AWARE':>12} {'DELTA':>8}")
        print(f"    Avg workers disagreeing: {avg_flip_b:>9.1f} {avg_flip_ta:>11.1f} {avg_flip_ta-avg_flip_b:>+7.1f}")
        print(f"    Avg weighted score:      {avg_ws_b:>9.3f} {avg_ws_ta:>11.3f} {avg_ws_ta-avg_ws_b:>+7.3f}")
        print(f"    Avg PEAK flip count:     {peak_b:>9.1f} {peak_ta:>11.1f} {peak_ta-peak_b:>+7.1f}")

    # ── Separation test: does trade-aware create MORE gap? ───────────────
    losing_bars = [b for t in all_trade_bars if t['is_losing'] for b in t['bars']]
    winning_bars = [b for t in all_trade_bars if not t['is_losing'] for b in t['bars']]
    if losing_bars and winning_bars:
        # Baseline gap
        base_gap = (np.mean([b['weighted_score'] for b in losing_bars]) -
                    np.mean([b['weighted_score'] for b in winning_bars]))
        # Trade-aware gap
        ta_gap = (np.mean([b['ta_weighted_score'] for b in losing_bars]) -
                  np.mean([b['ta_weighted_score'] for b in winning_bars]))
        print(f"\n── SEPARATION (losing - winning weighted score) ──")
        print(f"  Baseline:    {base_gap:+.4f}  ({'losers flip MORE' if base_gap > 0 else 'no separation'})")
        print(f"  Trade-aware: {ta_gap:+.4f}  ({'losers flip MORE' if ta_gap > 0 else 'no separation'})")
        print(f"  Improvement: {ta_gap - base_gap:+.4f}")

    # ── Threshold sweep: print both models ───────────────────────────────
    def _print_sweep(title, results):
        print(f"\n── {title} ──")
        print(f"  {'Threshold':<30} {'Flips':>6} {'Correct':>8} {'Wrong':>6} {'Prec':>6} "
              f"{'Net PnL':>10} {'Avg Bars Early':>14}")
        print("  " + "─" * 90)
        sorted_s = sorted(results.items(),
                          key=lambda x: (x[1]['saved_pnl'] - x[1]['lost_pnl']),
                          reverse=True)
        shown = 0
        for key, data in sorted_s:
            if data['flips'] < 5:
                continue
            precision = data['correct_flips'] / data['flips'] if data['flips'] else 0
            if precision < 0.50:
                continue
            net_pnl = data['saved_pnl'] - data['lost_pnl']
            avg_bars = np.mean(data['bars_early']) if data['bars_early'] else 0
            print(f"  {key:<30} {data['flips']:>6} {data['correct_flips']:>8} {data['wrong_flips']:>6} "
                  f"{precision:>5.0%} ${net_pnl:>+9,.2f} {avg_bars:>13.1f}")
            shown += 1
            if shown >= 15:
                break
        if shown == 0:
            print("  (no thresholds met the criteria)")
        return sorted_s

    sorted_base = _print_sweep("THRESHOLD SWEEP — BASELINE (blind workers)", sweep_results)
    sorted_ta = _print_sweep("THRESHOLD SWEEP — TRADE-AWARE (workers see PnL + velocity)", sweep_results_ta)

    # ── Best threshold comparison ─────────────────────────────────────────
    print(f"\n── BEST THRESHOLD COMPARISON ──")
    for label, sorted_s in [("BASELINE", sorted_base), ("TRADE-AWARE", sorted_ta)]:
        if sorted_s and sorted_s[0][1]['flips'] > 0:
            bk, bd = sorted_s[0]
            prec = bd['correct_flips'] / bd['flips'] if bd['flips'] else 0
            net = bd['saved_pnl'] - bd['lost_pnl']
            print(f"  {label}: {bk}  flips={bd['flips']}  prec={prec:.0%}  net=${net:+,.2f}")

    print("\n" + "=" * 90)


DATA_PRESETS = {
    '1d': 'DATA/ATLAS_1DAY',
    '1w': 'DATA/ATLAS_1WEEK',
    '1m': 'DATA/ATLAS_OOS',    # ~2 months OOS
    '1y': 'DATA/ATLAS',        # full 12-month training set
    'oos': 'DATA/ATLAS_OOS',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Research: Consensus-driven belief flip (bar-by-bar replay)')
    parser.add_argument('--data', default=None, help='ATLAS data directory (overrides --run)')
    parser.add_argument('--run', choices=list(DATA_PRESETS.keys()), default='1d',
                        help='Data preset: 1d (fast), 1w (screening), 1m/oos (OOS), 1y (full)')
    parser.add_argument('--min-adverse', type=int, default=4)
    parser.add_argument('--min-workers', type=int, default=7)
    parser.add_argument('--min-weighted', type=float, default=0.65)
    args = parser.parse_args()
    if args.data is None:
        args.data = DATA_PRESETS[args.run]
    print(f"  Data: {args.data}  (preset: {args.run})")
    run_research(args)
