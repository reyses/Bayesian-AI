"""
blowout_simulation.py -- Monte Carlo & sequential blowout analysis.

Two analyses:

1) Sequential (deterministic): replay the actual historical trade sequence
   starting from each possible start point with EQUITY_START. Track
   when (if) the equity goes <= 0. Realistic — uses the actual cluster
   structure as it occurred.

2) Bootstrap (random): sample trades with replacement and walk equity
   forward. Repeat 10k times. Gives a distributional answer accounting for
   trade-level variance but ASSUMES INDEPENDENCE (no cluster structure).

The honest answer is between the two — neither overstates streak risk
(by assuming initial worst-case) nor understates it (by ignoring clusters).

Usage:
    python tools/blowout_simulation.py --csv reports/findings/zigzag_trail_sl10.csv --equity 153
    python tools/blowout_simulation.py --csv reports/findings/zigzag_trail_sl10.csv --equity 500
"""
import argparse
import os

import numpy as np
import pandas as pd


def sequential_blowout(pnls: np.ndarray, equity_start: float):
    """For each possible start index, walk forward and check blowout.
    Returns array of (terminal_equity, blowout_idx_or_-1) per start point."""
    n = len(pnls)
    results = np.empty(n, dtype=object)
    for start in range(n):
        eq = equity_start
        blowout_at = -1
        peak = eq
        for i in range(start, n):
            eq += pnls[i]
            if eq < peak:  pass  # could track drawdown
            else:          peak = eq
            if eq <= 0:
                blowout_at = i - start + 1   # # trades to blow
                break
        results[start] = (eq, blowout_at)
    return results


def bootstrap_blowout(pnls: np.ndarray, equity_start: float,
                      n_paths: int = 10000, n_trades_per_path: int = 1000,
                      seed: int = 42):
    """Random-sample paths. Returns blowout times or -1 if survived."""
    rng = np.random.default_rng(seed)
    n = len(pnls)
    blowouts = np.full(n_paths, -1, dtype=np.int32)
    final_equity = np.empty(n_paths, dtype=np.float64)
    for p in range(n_paths):
        eq = equity_start
        path = pnls[rng.integers(0, n, n_trades_per_path)]
        cum = np.cumsum(path) + eq
        zero_idx = np.argmax(cum <= 0)   # first index where cumulative drops to 0
        if cum[zero_idx] <= 0:
            blowouts[p] = int(zero_idx) + 1
            final_equity[p] = 0.0
        else:
            blowouts[p] = -1
            final_equity[p] = float(cum[-1])
    return blowouts, final_equity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--equity', type=float, default=153.0)
    ap.add_argument('--n-paths', type=int, default=10000)
    ap.add_argument('--horizon', type=int, default=1000,
                    help='Trades per bootstrap path')
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values(['day', 'entry_ts']).reset_index(drop=True)
    pnls = df['pnl_usd'].values

    n = len(pnls)
    p_loss = (pnls < 0).mean()
    p_win  = (pnls > 0).mean()
    mean_pnl = pnls.mean()
    print('=' * 92)
    print(f'  BLOWOUT SIMULATION')
    print(f'  Source: {os.path.basename(args.csv)}  Trades: {n:,}  Equity start: ${args.equity:.0f}')
    print(f'  Trade dist: WR_count={p_win*100:.1f}% loss%={p_loss*100:.1f}% mean=${mean_pnl:+.2f}')
    print('=' * 92)

    # ── 1) Sequential ────────────────────────────────────────────────────
    print(f'\n--- SEQUENTIAL: walking actual historical trade order from each start point ---')
    seq = sequential_blowout(pnls, args.equity)
    blowout_times = np.array([b[1] for b in seq])
    final_eq = np.array([b[0] for b in seq])
    blew = blowout_times >= 0
    print(f'  Start points tested      : {n:,}')
    print(f'  Blowouts                 : {blew.sum():,}  ({blew.mean()*100:.2f}% of starts)')
    if blew.any():
        bt = blowout_times[blew]
        print(f'  Trades-to-blowout p25/p50/p75: {np.quantile(bt, 0.25):.0f} / {np.median(bt):.0f} / {np.quantile(bt, 0.75):.0f}')
        print(f'  Min trades-to-blowout    : {bt.min()}')
        print(f'  Max trades-to-blowout    : {bt.max()}')

    # Realistic: only consider start points where there were enough trades remaining
    valid = np.array([n - i >= 100 for i in range(n)])
    if valid.any():
        blew_valid = blew & valid
        print(f'  Among {valid.sum():,} starts with >=100 trades remaining: '
              f'blowout rate = {blew_valid.sum()/valid.sum()*100:.2f}%')

    # ── 2) Bootstrap ─────────────────────────────────────────────────────
    print(f'\n--- BOOTSTRAP: {args.n_paths:,} random paths, {args.horizon:,} trades each ---')
    bo, fe = bootstrap_blowout(pnls, args.equity, args.n_paths, args.horizon)
    blew = bo >= 0
    print(f'  Paths tested             : {args.n_paths:,}')
    print(f'  Blowouts                 : {blew.sum():,}  ({blew.mean()*100:.2f}%)')
    if blew.any():
        bt = bo[blew]
        print(f'  Trades-to-blowout p25/p50/p75: {np.quantile(bt, 0.25):.0f} / {int(np.median(bt))} / {np.quantile(bt, 0.75):.0f}')
        print(f'  Min trades-to-blowout    : {bt.min()}')

    # Survival metrics
    survived = ~blew
    if survived.any():
        fe_s = fe[survived]
        print(f'  Survival rate            : {survived.mean()*100:.2f}%')
        print(f'  Final equity (survivors) : p25=${np.quantile(fe_s, 0.25):.0f}  '
              f'median=${np.quantile(fe_s, 0.50):.0f}  '
              f'p75=${np.quantile(fe_s, 0.75):.0f}')
        print(f'  Mean final equity (all)  : ${fe.mean():.0f}')

    # ── 3) Probability of blowout vs trades-elapsed (bootstrap) ─────────
    print(f'\n--- BLOWOUT PROB vs TRADES ELAPSED (bootstrap) ---')
    print(f'  {"trades":>8} {"P(blown_by)":>12}')
    for k in [10, 25, 50, 100, 200, 500, 1000]:
        if k <= args.horizon:
            p = (np.where(bo >= 0, bo, args.horizon + 1) <= k).mean()
            print(f'  {k:>8} {p*100:>10.2f}%')

    # ── 4) Equity-level table ────────────────────────────────────────────
    print(f'\n--- BLOWOUT RATE BY EQUITY LEVEL (bootstrap, {args.horizon} trades) ---')
    print(f'  {"equity":>8} {"surv%":>8} {"blow%":>8}')
    for eq in [100, 153, 250, 500, 1000, 2000, 5000]:
        bo2, fe2 = bootstrap_blowout(pnls, eq, n_paths=2000, n_trades_per_path=args.horizon, seed=42)
        s = (bo2 < 0).mean()
        print(f'  ${eq:>+6.0f} {s*100:>7.2f}% {(1-s)*100:>7.2f}%')


if __name__ == '__main__':
    main()
