"""
Peak-Capture Regret — how much of each trade's 20-bar peak do we capture?

For each trade, examine the path over the first 20 bars (or until exit,
whichever comes first). Compute:
  peak_in_window = max pnl in that 20-bar window
  captured      = exit pnl (final realized pnl)
  regret        = peak_in_window - captured  (always >= 0)

Regret = money left on the table. Zero regret = exited at peak. High regret
= exited after giving back OR held past peak.

Break down:
  - Per exit_reason: which exits leave the most regret?
  - Per peak-bucket: do we capture REAL peaks but miss STRONG ones?
  - Summary: if exit were oracle at bar-20 peak, how much extra $ could
    we have captured?

Usage:
    python tools/peak_capture_regret.py --tier KILL_SHOT_ACTIVE
    python tools/peak_capture_regret.py --tier KILL_SHOT_CALM
    python tools/peak_capture_regret.py --window 20  # default 20

Output: reports/findings/peak_capture_regret_<TIER>.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_DIR = 'reports/findings'

TV = 0.50


def resolve_pickle(tier, explicit):
    if explicit:
        return explicit
    for cand in (
        os.path.join(TRADES_DIR, f'iso_is_{tier}.pkl'),
        os.path.join(TRADES_DIR, 'iso_is_KILL_SHOT_ACTIVE+KILL_SHOT_CALM.pkl'),
        os.path.join(TRADES_DIR, 'iso_is.pkl'),
    ):
        if os.path.exists(cand):
            return cand
    return cand


def bucket_of(ticks):
    if ticks < 5: return 'NOISE'
    if ticks < 10: return 'FAKE'
    if ticks < 20: return 'MARGINAL'
    if ticks < 40: return 'REAL'
    if ticks < 80: return 'STRONG'
    return 'DOMINANT'


def peak_in_window(trade, window_bars):
    """Max pnl in first `window_bars` bars of the trade."""
    path = trade.get('path', [])
    if not path:
        return 0.0
    best = 0.0
    for p in path:
        if p.get('bar', 0) > window_bars:
            break
        best = max(best, p.get('pnl', 0.0))
    return best


def peak_bar_in_window(trade, window_bars):
    """Bar at which the 20-bar peak occurred."""
    path = trade.get('path', [])
    if not path:
        return None
    best_bar = 0
    best = -1e9
    for p in path:
        b = p.get('bar', 0)
        if b > window_bars:
            break
        if p.get('pnl', 0.0) > best:
            best = p.get('pnl', 0.0)
            best_bar = b
    return best_bar


def analyze(trades, window):
    per_exit = defaultdict(lambda: {'regrets': [], 'captured': [],
                                    'peaks': [], 'n': 0})
    per_bucket = defaultdict(lambda: {'regrets': [], 'captured': [],
                                      'peaks': [], 'n': 0})
    all_regrets = []
    all_peaks = []
    all_captured = []
    peak_bar_dist = []

    for t in trades:
        peak = peak_in_window(t, window)
        captured = t.get('pnl', 0.0)
        # Cap regret at 0 lower bound: if trade closed negative but had no
        # in-window peak, captured could be more negative than peak (peak=0).
        regret = max(0.0, peak - captured)
        pb = peak_bar_in_window(t, window)
        if pb is not None:
            peak_bar_dist.append(pb)

        exit_reason = t.get('exit_reason', '?')
        peak_ticks = peak / TV
        bucket = bucket_of(peak_ticks)

        per_exit[exit_reason]['regrets'].append(regret)
        per_exit[exit_reason]['captured'].append(captured)
        per_exit[exit_reason]['peaks'].append(peak)
        per_exit[exit_reason]['n'] += 1

        per_bucket[bucket]['regrets'].append(regret)
        per_bucket[bucket]['captured'].append(captured)
        per_bucket[bucket]['peaks'].append(peak)
        per_bucket[bucket]['n'] += 1

        all_regrets.append(regret)
        all_peaks.append(peak)
        all_captured.append(captured)

    def summarize(d):
        if d['n'] == 0:
            return None
        r = np.array(d['regrets'])
        c = np.array(d['captured'])
        p = np.array(d['peaks'])
        return {
            'n': d['n'],
            'mean_peak': float(p.mean()),
            'mean_capture': float(c.mean()),
            'mean_regret': float(r.mean()),
            'total_peak': float(p.sum()),
            'total_capture': float(c.sum()),
            'total_regret': float(r.sum()),
            'p50_regret': float(np.median(r)),
            'p75_regret': float(np.percentile(r, 75)),
            'p90_regret': float(np.percentile(r, 90)),
            'capture_pct': float(c.sum() / max(p.sum(), 1e-9) * 100),
        }

    return {
        'per_exit': {k: summarize(v) for k, v in per_exit.items()},
        'per_bucket': {k: summarize(v) for k, v in per_bucket.items()},
        'overall': {
            'n': len(trades),
            'total_peak': float(np.sum(all_peaks)),
            'total_capture': float(np.sum(all_captured)),
            'total_regret': float(np.sum(all_regrets)),
            'mean_peak': float(np.mean(all_peaks)) if all_peaks else 0,
            'mean_capture': float(np.mean(all_captured)) if all_captured else 0,
            'mean_regret': float(np.mean(all_regrets)) if all_regrets else 0,
            'capture_pct': (float(np.sum(all_captured)) /
                            max(np.sum(all_peaks), 1e-9) * 100),
            'peak_bar_p50': int(np.median(peak_bar_dist)) if peak_bar_dist else 0,
            'peak_bar_p75': int(np.percentile(peak_bar_dist, 75)) if peak_bar_dist else 0,
            'peak_bar_p90': int(np.percentile(peak_bar_dist, 90)) if peak_bar_dist else 0,
        },
    }


def write_report(tier, window, result, out_path):
    L = []
    ov = result['overall']
    L.append(f'# Peak-Capture Regret — {tier}  (window: {window} bars)')
    L.append('')
    L.append(f'**N**: {ov["n"]:,}  '
             f'Total peak ${ov["total_peak"]:+,.0f}  '
             f'Captured ${ov["total_capture"]:+,.0f}  '
             f'Regret ${ov["total_regret"]:+,.0f}')
    L.append(f'Capture efficiency: **{ov["capture_pct"]:.1f}%** of in-window peak.')
    L.append(f'Mean per trade: peak +${ov["mean_peak"]:.2f}, '
             f'captured ${ov["mean_capture"]:+.2f}, '
             f'regret ${ov["mean_regret"]:+.2f}')
    L.append(f'Peak bar distribution: p50={ov["peak_bar_p50"]}, '
             f'p75={ov["peak_bar_p75"]}, p90={ov["peak_bar_p90"]}')
    L.append('')

    L.append('## Regret by exit reason')
    L.append('')
    L.append('| exit_reason | N | mean peak | mean capture | mean regret | total regret | capture % |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    exits_sorted = sorted(
        ((k, v) for k, v in result['per_exit'].items() if v),
        key=lambda kv: kv[1]['total_regret'],
        reverse=True,
    )
    for name, s in exits_sorted:
        L.append(f'| {name} | {s["n"]} | ${s["mean_peak"]:+.2f} | '
                 f'${s["mean_capture"]:+.2f} | ${s["mean_regret"]:+.2f} | '
                 f'${s["total_regret"]:+,.0f} | {s["capture_pct"]:.1f}% |')
    L.append('')

    L.append('## Regret by peak bucket')
    L.append('')
    L.append('| Bucket | N | mean peak | mean capture | mean regret | total regret |')
    L.append('|---|---:|---:|---:|---:|---:|')
    bucket_order = ['NOISE', 'FAKE', 'MARGINAL', 'REAL', 'STRONG', 'DOMINANT']
    for b in bucket_order:
        s = result['per_bucket'].get(b)
        if s:
            L.append(f'| {b} | {s["n"]} | ${s["mean_peak"]:+.2f} | '
                     f'${s["mean_capture"]:+.2f} | ${s["mean_regret"]:+.2f} | '
                     f'${s["total_regret"]:+,.0f} |')
    L.append('')

    L.append('## Interpretation')
    L.append('')
    L.append(f'- **Oracle upper bound (exit at peak each trade)**: would '
             f'capture ${ov["total_peak"]:+,.0f} instead of '
             f'${ov["total_capture"]:+,.0f}. Gap = ${ov["total_regret"]:+,.0f} '
             f'({100 - ov["capture_pct"]:.1f}% missed).')
    L.append('')
    if exits_sorted:
        worst = exits_sorted[0]
        L.append(f'- **Highest-regret exit**: `{worst[0]}` '
                 f'(N={worst[1]["n"]}, avg regret ${worst[1]["mean_regret"]:.2f}, '
                 f'total ${worst[1]["total_regret"]:+,.0f}). '
                 f'These exits leave the most money on the table.')
    big_bucket = result['per_bucket'].get('BIG_WIN') or result['per_bucket'].get('DOMINANT')
    if big_bucket:
        L.append(f'- **DOMINANT bucket** (peak >= 80 ticks): mean regret '
                 f'${big_bucket["mean_regret"]:.2f}. High regret here = we\'re '
                 f'catching big winners but giving back significant profit.')
    L.append('')
    L.append('---')
    L.append(f'_Generated by `tools/peak_capture_regret.py --tier {tier} --window {window}`_')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--trades', default=None)
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pkl = resolve_pickle(args.tier, args.trades)
    print(f'Loading {pkl}...')
    with open(pkl, 'rb') as f:
        trades = pickle.load(f)
    sub = [t for t in trades if t.get('entry_tier') == args.tier]
    if not sub:
        print(f'No trades for {args.tier}')
        return
    print(f'  {args.tier}: {len(sub):,} trades, window={args.window} bars')

    result = analyze(sub, args.window)
    ov = result['overall']
    print()
    print(f'Overall:')
    print(f'  Total peak:    ${ov["total_peak"]:>+11,.0f}')
    print(f'  Total capture: ${ov["total_capture"]:>+11,.0f}')
    print(f'  Total regret:  ${ov["total_regret"]:>+11,.0f}  '
          f'({100 - ov["capture_pct"]:.1f}% missed)')
    print(f'  Mean regret/trade: ${ov["mean_regret"]:.2f}')
    print(f'  Peak bar p50/p75/p90: {ov["peak_bar_p50"]}/{ov["peak_bar_p75"]}/{ov["peak_bar_p90"]}')

    print()
    print('Regret by exit reason:')
    print(f'{"exit":<32} {"N":>5} {"mean_peak":>10} {"mean_cap":>10} '
          f'{"mean_reg":>10} {"total_reg":>12}')
    print('-' * 85)
    exits_sorted = sorted(
        ((k, v) for k, v in result['per_exit'].items() if v),
        key=lambda kv: kv[1]['total_regret'],
        reverse=True,
    )
    for name, s in exits_sorted:
        print(f'{name:<32} {s["n"]:>5} ${s["mean_peak"]:>+8.2f} '
              f'${s["mean_capture"]:>+8.2f} ${s["mean_regret"]:>+8.2f} '
              f'${s["total_regret"]:>+10,.0f}')

    out_path = args.out or os.path.join(OUT_DIR, f'peak_capture_regret_{args.tier}.md')
    write_report(args.tier, args.window, result, out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
