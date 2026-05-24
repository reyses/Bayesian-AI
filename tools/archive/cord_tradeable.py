"""
Compute REALISTIC tradeable capture from a zigzag, accounting for the
pivot-confirmation tax.

Cord length (oracle metric) = sum of |leg| for legs above threshold R.
That assumes perfect foresight — we enter at the actual peak/trough.

Realistic capture requires a $R retracement to confirm each pivot:
  - Entry lag: we only know a pivot happened AFTER price moves R against
    the prior direction. We enter $R away from the extreme.
  - Exit lag: same thing at the next pivot — we give back $R confirming.
  - Net per leg = leg_length - 2R  (if positive; else skip leg)

We also report:
  - Oracle (perfect foresight): sum of all |leg|
  - Realistic all-legs: sum of max(0, leg - 2R)  (cut losses at 0 = assume
    we don't enter on legs we'd lose on — still optimistic)
  - Realistic every-leg-taken: sum of (leg - 2R) including negatives —
    the honest result if the strategy takes every pivot it sees
  - Commission: $0.50 per contract per side → $1 per round-trip

Usage:
    python tools/cord_tradeable.py --day 2025_06_09 --r 30
    python tools/cord_tradeable.py --day 2025_06_09         # all R
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
DOLLAR_PER_POINT = 2.0
COMMISSION_PER_RT = 1.0  # $0.50 per side x 2


def zigzag_legs(closes, r_pts):
    """Return list of (start_idx, end_idx, leg_points, direction) per leg.
    Uses the same algorithm as tools/cord_length_1m.zigzag_pivots: tracks
    extremes by overwriting pivots[-1] until retracement confirms. Legs
    are extreme-to-extreme on the final pivot list."""
    from tools.cord_length_1m import zigzag_pivots
    pivots = zigzag_pivots(closes, r_pts * 2.0)  # that fn takes dollars
    legs = []
    for i in range(len(pivots) - 1):
        s = pivots[i]
        e = pivots[i + 1]
        dp = closes[e] - closes[s]
        legs.append((s, e, dp, 'up' if dp > 0 else 'down'))
    return legs


def analyze_day(closes_pts, r_dollars):
    """closes_pts in points (MNQ 1-pt = $2). r in $."""
    legs = zigzag_legs(closes_pts, r_dollars / DOLLAR_PER_POINT)
    n_legs = len(legs)
    if n_legs == 0:
        return None

    leg_dollars = np.array([abs(leg[2]) * DOLLAR_PER_POINT for leg in legs])

    # Oracle (perfect foresight): capture every leg fully
    oracle = leg_dollars.sum()

    # Realistic: each round-trip costs 2R (entry + exit retracement tax)
    # If we take EVERY leg the zigzag shows:
    every_leg_net = (leg_dollars - 2 * r_dollars).sum()
    every_leg_commission = n_legs * COMMISSION_PER_RT

    # If we only take legs that are net-positive pre-commission:
    winning_legs = leg_dollars[leg_dollars > 2 * r_dollars]
    n_winners = len(winning_legs)
    selective_gross = (winning_legs - 2 * r_dollars).sum() if n_winners else 0
    selective_comm = n_winners * COMMISSION_PER_RT

    # (Selective is still oracle — you can't know which legs will exceed 2R
    # until they already did.  Listed as upper bound only.)

    # Legs by size
    big_legs = leg_dollars[leg_dollars > 2 * r_dollars]
    small_legs = leg_dollars[leg_dollars <= 2 * r_dollars]

    return {
        'r': r_dollars,
        'n_legs': n_legs,
        'oracle': oracle,
        'avg_leg': leg_dollars.mean(),
        'median_leg': np.median(leg_dollars),
        'max_leg': leg_dollars.max(),
        'n_big': len(big_legs),
        'n_small': len(small_legs),
        'big_avg': big_legs.mean() if len(big_legs) else 0,
        'every_leg_net': every_leg_net,
        'every_leg_after_comm': every_leg_net - every_leg_commission,
        'selective_gross': selective_gross,
        'selective_after_comm': selective_gross - selective_comm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2025_06_09')
    ap.add_argument('--r', type=float, default=None,
                    help='Single R threshold ($). Default: sweep 5..100')
    args = ap.parse_args()

    path = os.path.join(ATLAS_1M_DIR, f'{args.day}.parquet')
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].values.astype(np.float64)

    rs = [args.r] if args.r else [5, 10, 15, 20, 30, 50, 75, 100]

    print(f'Day: {args.day} — {len(closes)} 1m bars')
    print(f'Total 1m range: ${(closes.max() - closes.min()) * DOLLAR_PER_POINT:,.0f}')
    print()
    print(f'{"R":<5} {"legs":>5} {"oracle":>9} {"avg_leg":>8} {"max":>7} '
          f'{"big(>2R)":>9} {"every_leg":>10} {"after_comm":>10} '
          f'{"selective":>9} {"sel_comm":>9}')
    print('-' * 105)
    for r in rs:
        res = analyze_day(closes, r)
        if res is None:
            print(f'{r:<5} (no legs)')
            continue
        print(f'${res["r"]:<4.0f} {res["n_legs"]:>5} '
              f'${res["oracle"]:>8,.0f} '
              f'${res["avg_leg"]:>7.0f} '
              f'${res["max_leg"]:>6.0f} '
              f'{res["n_big"]:>4}/{res["n_legs"]} '
              f'${res["every_leg_net"]:>+9,.0f} '
              f'${res["every_leg_after_comm"]:>+9,.0f} '
              f'${res["selective_gross"]:>+8,.0f} '
              f'${res["selective_after_comm"]:>+8,.0f}')


if __name__ == '__main__':
    main()
