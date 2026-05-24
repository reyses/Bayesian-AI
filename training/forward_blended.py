"""
Blended Engine Forward Pass — standalone runner to measure BlendedEngine
(the LIVE engine, CNN disabled) on IS and/or OOS historical data.

Mirrors the iso pipeline report format ($WR, tier KPI table, PnL mode
buckets, MAE-cut lift) so blended results are directly comparable to iso.

Default: use_cnn=False (same as live). Pass --cnn to run with CNN gating
(doesn't work unless retrained model checkpoints are present).

Usage:
    python training/forward_blended.py                         # IS
    python training/forward_blended.py --oos                   # OOS only
    python training/forward_blended.py --with-oos              # IS + OOS
    python training/forward_blended.py --log                   # tee stdout
    python training/forward_blended.py --cnn                   # with CNN

Output pickles:
    training_iso/output/trades/blended_is.pkl
    training_iso/output/trades/blended_oos.pkl
"""
import os
import sys
import glob
import pickle
import time as _time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.nightmare_blended import BlendedEngine
from training.sfe_ticker import FeatureTicker

FEATURES_DIR_5S = 'DATA/ATLAS/FEATURES_5s_v2/L0'
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'training/output'


def _resolve_days(target):
    all_files = sorted(glob.glob(os.path.join(FEATURES_DIR_5S, '*.parquet')))
    if target == 'is':
        return [f for f in all_files if '2025_' in os.path.basename(f)]
    return [f for f in all_files if '2026_' in os.path.basename(f)]


# ── PnL bucket definitions (mirror run_iso.py) ───────────────────────
_PNL_BUCKETS = [
    ('BIG_LOSS',      float('-inf'),  -50.0),
    ('MED_LOSS',      -50.0,          -25.0),
    ('REAL_LOSS',     -25.0,          -10.0),
    ('MARG_LOSS',     -10.0,           -5.0),
    ('NOISE_LOSS',     -5.0,            0.0),
    ('NOISE_WIN',       0.0,            5.0),
    ('MARG_WIN',        5.0,           10.0),
    ('REAL_WIN',       10.0,           25.0),
    ('STRONG_WIN',     25.0,           50.0),
    ('BIG_WIN',        50.0, float('inf')),
]


def _bucket_of(pnl):
    for name, lo, hi in _PNL_BUCKETS:
        if lo <= pnl < hi:
            return name
    return 'BIG_WIN' if pnl >= 50 else 'BIG_LOSS'


def _mae_cut_delta(trades, threshold):
    net = 0.0
    for t in trades:
        path = t.get('path', []) or []
        if not path:
            continue
        mae = min((p.get('pnl', 0.0) for p in path), default=0.0)
        if mae > threshold:
            continue
        net += threshold - t.get('pnl', 0.0)
    return net


def run_blended_forward(target, use_cnn=False):
    feat_files = _resolve_days(target)
    if not feat_files:
        print(f'No feature files for target={target!r}')
        return [], []

    print(f'BLENDED FORWARD [{target.upper()}] — {len(feat_files)} day(s), '
          f'use_cnn={use_cnn}')

    all_results = []
    all_trades = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None
        # One engine per day (fresh state each session)
        engine = BlendedEngine(use_cnn=use_cnn)
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()
        day_trades = list(engine.get_full_trades())
        for t in day_trades:
            t['day'] = day_name
        all_trades.extend(day_trades)

        day_pnl = sum(t['pnl'] for t in day_trades)
        all_results.append({
            'day': day_name,
            'trades': len(day_trades),
            'pnl': day_pnl,
        })

    return all_results, all_trades


def _print_results(results, all_trades, target):
    if not results:
        print('No results.')
        return
    n_days = len(results)
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    winning_days = sum(1 for r in results if r['pnl'] > 0)
    print(f'\n{"="*60}')
    print(f'RESULTS: {n_days} days | {total_trades:,} trades | ${total_pnl:,.2f}')
    print(f'  $/day: ${total_pnl / max(n_days, 1):.2f}')
    print(f'  Winning days: {winning_days}/{n_days}')
    print(f'{"="*60}')

    if not all_trades:
        return

    # Collect unique tiers observed
    tier_counts = Counter(t.get('entry_tier', '?') for t in all_trades)

    # Tier KPI table
    print()
    print(f'{"Tier":<17} {"N":>6} {"WR":>7} {"Total":>10} {"$/tr":>7} '
          f'{"Mode":>20}')
    print('-' * 75)
    for tier, count in sorted(tier_counts.items(),
                              key=lambda kv: -abs(kv[1])):
        sub = [t for t in all_trades if t.get('entry_tier') == tier]
        if not sub:
            continue
        wins_pnl = sum(t['pnl'] for t in sub if t['pnl'] > 0)
        loss_pnl_abs = abs(sum(t['pnl'] for t in sub if t['pnl'] < 0))
        if loss_pnl_abs > 0:
            wr = (wins_pnl / loss_pnl_abs - 1) * 100
            wr_str = f'{wr:>+5.0f}%'
        else:
            wr_str = '  inf%' if wins_pnl > 0 else '    0%'
        total = wins_pnl - loss_pnl_abs
        per = total / count
        b_counts = Counter(_bucket_of(t.get('pnl', 0.0)) for t in sub)
        mode_name, mode_n = b_counts.most_common(1)[0]
        mode_pct = mode_n / count * 100
        mode_str = f'{mode_name}({mode_pct:.0f}%)'
        print(f'{tier:<17} {count:>6,} {wr_str:>7} ${total:>+9,.0f} '
              f'${per:>+6.2f} {mode_str:>20}')

    # Mode-bucket detail table
    abbrevs = ['BL', 'ML', 'RL', 'mL', 'nL', 'nW', 'mW', 'RW', 'SW', 'BW']
    print()
    print(f'{"="*100}')
    print(f'PnL MODE BUCKETS per tier')
    print(f'{"="*100}')
    print(f'  {"Tier":<18} {"MODE":<14} ' +
          ' '.join(f'{a:>4}' for a in abbrevs))
    print('  ' + '-' * 85)
    for tier in sorted(tier_counts.keys()):
        sub = [t for t in all_trades if t.get('entry_tier') == tier]
        if not sub:
            continue
        counts = Counter(_bucket_of(t.get('pnl', 0.0)) for t in sub)
        mode_name, mode_n = counts.most_common(1)[0]
        mode_pct = mode_n / len(sub) * 100
        cells = []
        for name, _, _ in _PNL_BUCKETS:
            n = counts.get(name, 0)
            pct = n / len(sub) * 100 if n else 0
            cells.append(f'{pct:>3.0f}%' if n else '   -')
        mode_str = f'{mode_name}({mode_pct:.0f}%)'
        print(f'  {tier:<18} {mode_str:<14} ' + ' '.join(cells))

    # MAE-cut lift (approximate — requires path features in trades)
    print()
    mae_sweep = [-20, -25, -30, -35, -40, -50, -60, -75, -100]
    global_best = None
    for T in mae_sweep:
        net = _mae_cut_delta(all_trades, T)
        if global_best is None or net > global_best['net']:
            global_best = {'T': T, 'net': net}
    if global_best and global_best['net'] != 0:
        print(f'MAE-cut global best T=${global_best["T"]}  '
              f'save ${global_best["net"]:+,.0f}  -> '
              f'${global_best["net"]/n_days:+.2f}/day')


def main():
    args = sys.argv[1:]
    use_cnn = '--cnn' in args
    with_oos = '--with-oos' in args
    target = 'oos' if '--oos' in args and not with_oos else 'is'

    # --log flag (mirror run_iso.py behavior)
    log_path = None
    if '--log' in args:
        idx = args.index('--log')
        if idx + 1 < len(args) and not args[idx + 1].startswith('--'):
            log_path = args[idx + 1]
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            t_tag = 'is_oos' if with_oos else target
            log_path = os.path.join('reports', 'findings',
                                    f'blended_run_{t_tag}_{ts}.txt')
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams:
                    try: st.write(s); st.flush()
                    except Exception: pass
            def flush(self):
                for st in self.streams:
                    try: st.flush()
                    except Exception: pass
        _log_fh = open(log_path, 'w', encoding='utf-8', errors='replace')
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        print(f'[LOG] Writing output to {log_path}')

    print('=' * 60)
    print(f'BLENDED FORWARD PASS — CNN {"ON" if use_cnn else "OFF (live parity)"}')
    print(f'  Target: {target.upper()}' + (' + OOS' if with_oos else ''))
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    t_start = _time.perf_counter()

    def _run_one(tgt):
        print(f'\n{"="*40}')
        print(f'PHASE 1: BLENDED forward pass [{tgt.upper()}]')
        print(f'{"="*40}')
        t0 = _time.perf_counter()
        results, trades = run_blended_forward(tgt, use_cnn=use_cnn)
        out_pkl = os.path.join(OUTPUT_DIR, 'trades', f'blended_{tgt}.pkl')
        if trades:
            os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
            with open(out_pkl, 'wb') as f:
                pickle.dump(trades, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'  Wrote: {out_pkl}')
        _print_results(results, trades, tgt)
        print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    _run_one(target)
    if with_oos and target != 'oos':
        _run_one('oos')

    print(f'\nTOTAL: {_time.perf_counter() - t_start:.0f}s')


if __name__ == '__main__':
    main()
