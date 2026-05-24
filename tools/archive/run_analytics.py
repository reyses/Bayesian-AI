"""
Mini analytics runner — re-runs analytics suite on existing checkpoint data.

Usage:
    python -m training.run_analytics                     # default checkpoints/
    python -m training.run_analytics --checkpoint-dir checkpoints_abc
    python -m training.run_analytics --oos               # run on OOS logs
    python -m training.run_analytics --per-depth         # per-depth breakdown
    python -m training.run_analytics --per-depth --top 5 # only top N depths by trade count
"""
import argparse
import os
import sys
import csv
from collections import defaultdict


def _depth_label(d):
    return {
        0: 'daily', 1: '4h+', 2: '1h', 3: '15m', 4: '5m', 5: '1m',
        6: '30s', 7: '15s', 8: '15s', 9: '5s', 10: '5s', 11: '1s', 12: '1s',
    }.get(d, '?')


def _direction_correct(direction, oracle_label_name):
    """Check if trade direction matches oracle label."""
    if 'NOISE' in (oracle_label_name or ''):
        return 'noise'
    if direction == 'LONG' and oracle_label_name in ('MEGA_LONG', 'SCALP_LONG'):
        return 'correct'
    if direction == 'SHORT' and oracle_label_name in ('MEGA_SHORT', 'SCALP_SHORT'):
        return 'correct'
    return 'wrong'


def _capture_bucket(cr):
    if cr < 0:
        return 'Reversed'
    elif cr < 0.20:
        return 'Too early'
    elif cr < 0.80:
        return 'Partial'
    else:
        return 'Optimal'


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def run_per_depth(records, out):
    """Per-depth analytics breakdown."""
    by_depth = defaultdict(list)
    for r in records:
        d = int(r.get('entry_depth', 6))
        by_depth[d].append(r)

    out.append('')
    out.append('=' * 100)
    out.append('PER-DEPTH ANALYTICS')
    out.append('=' * 100)

    for d in sorted(by_depth.keys()):
        trades = by_depth[d]
        n = len(trades)
        if n == 0:
            continue

        pnls = [_safe_float(t['actual_pnl']) for t in trades]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / n
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100

        holds = [_safe_float(t.get('hold_bars', 0)) for t in trades]
        avg_hold = sum(holds) / n if n else 0
        max_holds = [_safe_float(t.get('max_hold_bars', 0)) for t in trades]
        avg_max_hold = sum(max_holds) / n if n else 0
        hold_util = avg_hold / avg_max_hold * 100 if avg_max_hold > 0 else 0

        longs = sum(1 for t in trades if t.get('direction') == 'LONG')
        long_pct = longs / n * 100

        # Oracle potential
        oracle_mfes = [_safe_float(t.get('oracle_mfe', 0)) for t in trades]
        oracle_maes = [_safe_float(t.get('oracle_mae', 0)) for t in trades]
        oracle_potentials = [_safe_float(t.get('oracle_potential_pnl', 0)) for t in trades]
        avg_mfe = sum(oracle_mfes) / n
        avg_mae = sum(oracle_maes) / n
        total_potential = sum(oracle_potentials)
        capture_eff = total_pnl / total_potential * 100 if total_potential > 0 else 0

        # Direction accuracy
        dir_results = [_direction_correct(t.get('direction', ''), t.get('oracle_label_name', '')) for t in trades]
        n_correct = dir_results.count('correct')
        n_wrong = dir_results.count('wrong')
        n_noise = dir_results.count('noise')

        # Exit reasons
        exit_counts = defaultdict(list)
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            exit_counts[reason].append(_safe_float(t['actual_pnl']))

        # Capture rate buckets
        cap_buckets = defaultdict(list)
        for t in trades:
            cr = _safe_float(t.get('capture_rate', 0))
            bucket = _capture_bucket(cr)
            cap_buckets[bucket].append(_safe_float(t['actual_pnl']))

        # Conviction
        win_convictions = [_safe_float(t.get('belief_conviction', 0)) for t in trades if _safe_float(t['actual_pnl']) > 0]
        loss_convictions = [_safe_float(t.get('belief_conviction', 0)) for t in trades if _safe_float(t['actual_pnl']) <= 0]
        avg_conv_win = sum(win_convictions) / len(win_convictions) if win_convictions else 0
        avg_conv_loss = sum(loss_convictions) / len(loss_convictions) if loss_convictions else 0

        # root_tf distribution
        tf_counts = defaultdict(int)
        for t in trades:
            tf_counts[t.get('root_tf', '?')] += 1

        # -- Print --
        label = f"depth {d} ({_depth_label(d)})"
        out.append(f'\n{"-" * 100}')
        out.append(f'  {label}  |  {n} trades  |  {wr:.1f}% WR  |  ${total_pnl:,.2f} PnL  |  ${avg_pnl:.2f}/trade')
        out.append(f'{"-" * 100}')

        out.append(f'  Hold: avg {avg_hold:.0f} bars / max {avg_max_hold:.0f} bars ({hold_util:.0f}% utilization)')
        out.append(f'  Direction: {long_pct:.0f}% LONG  |  Correct {n_correct} ({n_correct/n*100:.0f}%)  Wrong {n_wrong} ({n_wrong/n*100:.0f}%)  Noise {n_noise} ({n_noise/n*100:.0f}%)')
        out.append(f'  Oracle: MFE avg ${avg_mfe:.1f}  MAE avg ${avg_mae:.1f}  |  Potential ${total_potential:,.0f}  Captured {capture_eff:.1f}%')
        out.append(f'  Conviction: WIN avg {avg_conv_win:.3f}  LOSS avg {avg_conv_loss:.3f}  delta {avg_conv_win - avg_conv_loss:+.3f}')

        # Exit reasons
        out.append(f'  Exit reasons:')
        for reason in sorted(exit_counts.keys(), key=lambda r: -len(exit_counts[r])):
            pnl_list = exit_counts[reason]
            cnt = len(pnl_list)
            avg = sum(pnl_list) / cnt
            w = sum(1 for p in pnl_list if p > 0)
            out.append(f'    {reason:<20} {cnt:>5} trades  {w/cnt*100:>5.1f}% WR  avg ${avg:>8.2f}')

        # Capture buckets
        out.append(f'  Exit quality:')
        left_on_table = 0
        for bucket in ['Reversed', 'Too early', 'Partial', 'Optimal']:
            pnl_list = cap_buckets.get(bucket, [])
            if not pnl_list:
                continue
            cnt = len(pnl_list)
            avg = sum(pnl_list) / cnt
            pct = cnt / n * 100
            out.append(f'    {bucket:<15} {cnt:>5} ({pct:>5.1f}%)  avg ${avg:>8.2f}')
            if bucket != 'Optimal':
                # Estimate left on table: potential - actual for non-optimal
                for t in trades:
                    cr = _safe_float(t.get('capture_rate', 0))
                    if _capture_bucket(cr) == bucket:
                        left_on_table += _safe_float(t.get('oracle_potential_pnl', 0)) - max(0, _safe_float(t['actual_pnl']))
        out.append(f'    Left on table: ${left_on_table:>12,.2f}')

        # root_tf
        if tf_counts:
            tf_str = '  |  '.join(f'{tf}:{cnt}' for tf, cnt in sorted(tf_counts.items(), key=lambda x: -x[1]))
            out.append(f'  Parent TF: {tf_str}')

    # -- Comparison table --
    out.append(f'\n{"=" * 100}')
    out.append('DEPTH COMPARISON (side-by-side)')
    out.append(f'{"=" * 100}')
    header = (f'  {"Depth":<16} {"Trades":>7} {"WR%":>6} {"AvgPnL":>9} {"TotPnL":>12} '
              f'{"AvgHold":>8} {"Rev%":>6} {"Opt%":>6} {"CapEff":>7} {"MH_Hit%":>8} '
              f'{"Left$":>12}')
    out.append(header)
    out.append(f'  {"-"*16} {"-"*7} {"-"*6} {"-"*9} {"-"*12} {"-"*8} {"-"*6} {"-"*6} {"-"*7} {"-"*8} {"-"*12}')

    all_trades = 0
    all_pnl = 0.0
    all_wins = 0
    all_left = 0.0

    for d in sorted(by_depth.keys()):
        trades = by_depth[d]
        n = len(trades)
        if n == 0:
            continue
        pnls = [_safe_float(t['actual_pnl']) for t in trades]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / n
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        avg_hold = sum(_safe_float(t.get('hold_bars', 0)) for t in trades) / n

        crs = [_safe_float(t.get('capture_rate', 0)) for t in trades]
        rev_pct = sum(1 for c in crs if c < 0) / n * 100
        opt_pct = sum(1 for c in crs if c >= 0.80) / n * 100

        oracle_pot = sum(_safe_float(t.get('oracle_potential_pnl', 0)) for t in trades)
        cap_eff = total_pnl / oracle_pot * 100 if oracle_pot > 0 else 0

        mh_hit = sum(1 for t in trades if t.get('exit_reason') == 'MAX_HOLD') / n * 100

        left = oracle_pot - max(0, total_pnl)

        label = f'depth {d:<3} ({_depth_label(d):<4})'
        out.append(f'  {label:<16} {n:>7,} {wr:>5.1f}% ${avg_pnl:>8.2f} ${total_pnl:>11,.2f} '
                   f'{avg_hold:>7.0f}b {rev_pct:>5.1f}% {opt_pct:>5.1f}% {cap_eff:>6.1f}% {mh_hit:>7.1f}% '
                   f'${left:>11,.0f}')
        all_trades += n
        all_pnl += total_pnl
        all_wins += wins
        all_left += left

    out.append(f'  {"-"*16} {"-"*7} {"-"*6} {"-"*9} {"-"*12} {"-"*8} {"-"*6} {"-"*6} {"-"*7} {"-"*8} {"-"*12}')
    all_wr = all_wins / all_trades * 100 if all_trades else 0
    all_avg = all_pnl / all_trades if all_trades else 0
    out.append(f'  {"ALL":<16} {all_trades:>7,} {all_wr:>5.1f}% ${all_avg:>8.2f} ${all_pnl:>11,.2f} '
               f'{"":>8} {"":>6} {"":>6} {"":>7} {"":>8} ${all_left:>11,.0f}')


def main():
    parser = argparse.ArgumentParser(description='Run analytics on existing checkpoint data')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--oos', action='store_true', help='Analyze OOS logs instead')
    parser.add_argument('--per-depth', action='store_true', help='Per-depth breakdown')
    parser.add_argument('--suite', action='store_true', help='Run full trade analytics suite (t-tests, ANOVA, OLS, etc.)')
    parser.add_argument('--all', action='store_true', help='Run everything (--suite + --per-depth)')
    args = parser.parse_args()

    if args.all:
        args.suite = True
        args.per_depth = True

    # Default: if nothing specified, run everything
    if not args.suite and not args.per_depth:
        args.suite = True
        args.per_depth = True

    prefix = 'oos_' if args.oos else ''
    log_path = os.path.join(args.checkpoint_dir, f'{prefix}oracle_trade_log.csv')

    if not os.path.exists(log_path):
        print(f'ERROR: {log_path} not found. Run a forward pass first.')
        sys.exit(1)

    print(f'Loading: {log_path}')

    output_lines = []

    # -- Suite --
    if args.suite:
        report_path = os.path.join(args.checkpoint_dir, f'{prefix}is_report.txt')
        from training.trade_analytics import run_trade_analytics
        text = run_trade_analytics(log_path, report_path)
        print(text)
        # Save standalone
        out_path = os.path.join(args.checkpoint_dir, f'{prefix}trade_analytics.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'\nSaved: {out_path}')

    # -- Per-depth --
    if args.per_depth:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)
        print(f'Loaded {len(records):,} trade records')

        depth_lines = []
        run_per_depth(records, depth_lines)
        depth_text = '\n'.join(depth_lines)
        print(depth_text)

        out_path = os.path.join(args.checkpoint_dir, f'{prefix}depth_analytics.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(depth_text)
        print(f'\nSaved: {out_path}')

    print('\n  OK: Analytics complete.')


if __name__ == '__main__':
    main()
