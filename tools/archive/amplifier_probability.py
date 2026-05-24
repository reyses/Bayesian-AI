"""
Amplifier Pair Probability Analysis — given 1 year of sample, what's the
statistical confidence that CASCADE-solo bleeds while CASCADE-confirmed wins?

Computes for each tier:
  - P(solo)           = fraction of firings with no follower in window
  - P(confirmed)      = fraction with at least one other tier same-dir follow-up
  - WR | solo         = win rate when solo
  - WR | confirmed    = win rate when confirmed
  - 95% Wilson CI on each WR
  - Significance test: does WR_confirmed > WR_solo with p < 0.05?
  - Economic test: is the $/tr lift > sampling noise?

Output tells us whether the amplifier signal is a real probabilistic edge
or just small-sample artifact.

Usage:
    python tools/amplifier_probability.py --window 15
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.nightmare_iso import TIER_PRIORITY


TRADES_PATH = 'training_iso/output/trades/iso_is.pkl'


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0, center - margin), min(1, center + margin))


def two_sample_z_test(w1: int, n1: int, w2: int, n2: int) -> float:
    """Two-proportion z-test. Returns z-score for H0: p1 == p2."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = w1 / n1, w2 / n2
    p_pool = (w1 + w2) / (n1 + n2)
    if p_pool in (0, 1):
        return 0.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (p2 - p1) / se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=TRADES_PATH)
    ap.add_argument('--window', type=int, default=15)
    args = ap.parse_args()

    print(f'Loading {args.trades}...')
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)

    primary = [t for t in trades
               if t.get('chain_idx', 0) == 0
               and t.get('entry_tier')
               and t.get('timestamp')]
    primary.sort(key=lambda t: t['timestamp'])

    window_sec = args.window * 60
    n = len(primary)
    print(f'  {n:,} primary entries over 1-year sample')

    # For each primary trade A, classify it as solo or confirmed.
    # confirmed = at least one other tier fires same-dir within window.
    tier_solo = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': []})
    tier_conf = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': []})

    for i, a in enumerate(primary):
        has_follower = False
        for j in range(i + 1, n):
            b = primary[j]
            if b['timestamp'] - a['timestamp'] > window_sec:
                break
            if b.get('dir') != a.get('dir'):
                continue
            if b['entry_tier'] == a['entry_tier']:
                continue
            has_follower = True
            break
        bucket = tier_conf if has_follower else tier_solo
        tier = a['entry_tier']
        pnl = a.get('pnl', 0.0)
        bucket[tier]['n'] += 1
        bucket[tier]['pnl'].append(pnl)
        if pnl > 0:
            bucket[tier]['wins'] += 1

    # ── Report ────────────────────────────────────────────────────
    print()
    print(f'Window: {args.window} bars ({window_sec}s). "Confirmed" = '
          f'another tier fires same-dir within window.')
    print()
    print(f'{"Tier":<18} {"N":>5} {"P(conf)":>8} | {"WR solo":>9} {"CI":<15} '
          f'{"WR conf":>9} {"CI":<15} {"z":>6} {"sig?":>6}')
    print('-' * 108)

    sig_count = 0
    for t in TIER_PRIORITY:
        n_solo = tier_solo[t]['n']
        n_conf = tier_conf[t]['n']
        n_total = n_solo + n_conf
        if n_total == 0:
            continue
        p_conf = n_conf / n_total * 100
        w_solo = tier_solo[t]['wins']
        w_conf = tier_conf[t]['wins']
        wr_solo = w_solo / max(n_solo, 1) * 100
        wr_conf = w_conf / max(n_conf, 1) * 100
        ci_solo = wilson_ci(w_solo, n_solo)
        ci_conf = wilson_ci(w_conf, n_conf)
        z = two_sample_z_test(w_solo, n_solo, w_conf, n_conf)
        sig = '**' if abs(z) > 1.96 else ' '
        if abs(z) > 1.96:
            sig_count += 1
        ci_solo_str = f'[{ci_solo[0]*100:.0f},{ci_solo[1]*100:.0f}]'
        ci_conf_str = f'[{ci_conf[0]*100:.0f},{ci_conf[1]*100:.0f}]'
        print(f'{t:<18} {n_total:>5} {p_conf:>6.1f}% | '
              f'{wr_solo:>7.1f}% {ci_solo_str:<15} '
              f'{wr_conf:>7.1f}% {ci_conf_str:<15} {z:>+5.2f} {sig:>5}')

    print()
    print(f'{sig_count} tiers show WR differences significant at p<0.05 (|z|>1.96)')

    # Economic breakdown
    print()
    print(f'{"Tier":<18} {"$/tr solo":>12} {"$/tr conf":>12} {"Lift":>10} '
          f'{"$/tr solo N":>12} {"$/tr conf N":>12}')
    print('-' * 82)
    for t in TIER_PRIORITY:
        n_solo = tier_solo[t]['n']
        n_conf = tier_conf[t]['n']
        if n_solo + n_conf == 0:
            continue
        solo_avg = np.mean(tier_solo[t]['pnl']) if tier_solo[t]['pnl'] else 0
        conf_avg = np.mean(tier_conf[t]['pnl']) if tier_conf[t]['pnl'] else 0
        lift = conf_avg - solo_avg
        print(f'{t:<18} ${solo_avg:>+10.2f} ${conf_avg:>+10.2f} ${lift:>+8.2f} '
              f'{n_solo:>12} {n_conf:>12}')


if __name__ == '__main__':
    main()
