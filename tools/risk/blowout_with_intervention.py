"""
blowout_with_intervention.py -- realistic blowout sim with operator intervention.

Models real-world deployment: a human watches the strategy and halts trading
when intra-day losses pile up. Strategy resumes next day. This is NOT a
"set-and-forget for 1000 trades" mechanical simulation.

Intervention rules tested:
  HALT_AFTER_N_LOSSES        : stop for the day after N consecutive losses
  HALT_AT_INTRADAY_DRAWDOWN  : stop for the day after total day P&L hits -$X
  HALT_AT_PORTFOLIO_DRAWDOWN : stop entirely if equity drops below floor
  RESUME_RULE                : new day always resumes (clean slate)

Plus a Python-vs-NT8 gap adjustment: Day 1 NT8 result implies +$680/day
pessimism in Python sim. Add a constant per-trade or per-day shift.

Usage:
    python tools/risk/blowout_with_intervention.py --csv reports/findings/zigzag_trail_sl10.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def simulate_paths(df: pd.DataFrame,
                   equity_start: float,
                   max_consecutive_losses: int,
                   max_daily_drawdown: float,
                   gap_per_trade: float,
                   horizon_days: int,
                   n_paths: int,
                   seed: int = 42):
    """Bootstrap-sample DAYS of trades (preserving intra-day clustering),
    apply intervention rules, track equity.

    Returns dict with blowout statistics."""
    rng = np.random.default_rng(seed)

    # Group trades by day
    by_day = df.groupby('day')['pnl_usd'].apply(list).to_dict()
    days = list(by_day.keys())

    blowouts = np.full(n_paths, -1, dtype=np.int32)
    final_equity = np.empty(n_paths, dtype=np.float64)
    days_traded = np.empty(n_paths, dtype=np.int32)

    for p in range(n_paths):
        eq = equity_start
        sample_days = rng.choice(days, size=horizon_days, replace=True)
        blew_at = -1
        d_traded = 0

        for d_idx, d in enumerate(sample_days):
            day_trades = list(by_day[d])
            consec_losses = 0
            day_pnl = 0.0
            traded_this_day = False

            for t in day_trades:
                t_adj = t + gap_per_trade
                # Apply intervention BEFORE the trade fires
                if max_consecutive_losses > 0 and consec_losses >= max_consecutive_losses:
                    break
                if max_daily_drawdown > 0 and day_pnl <= -max_daily_drawdown:
                    break

                eq += t_adj
                day_pnl += t_adj
                traded_this_day = True

                if t_adj < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0

                if eq <= 0:
                    blew_at = d_idx + 1
                    break

            if traded_this_day:
                d_traded += 1
            if blew_at > 0:
                break

        blowouts[p] = blew_at
        final_equity[p] = max(eq, 0.0)
        days_traded[p] = d_traded

    return dict(blowouts=blowouts, final_equity=final_equity, days_traded=days_traded)


def report(name, results, equity_start, n_paths, horizon_days):
    blew = results['blowouts'] >= 0
    fe   = results['final_equity']
    dt   = results['days_traded']
    surv = (~blew).sum() / n_paths
    print(f'\n{name}')
    print(f'  Survival         : {surv*100:>5.2f}%   ({(~blew).sum():,} of {n_paths:,})')
    print(f'  Blowouts         : {blew.sum():,} ({blew.mean()*100:>5.2f}%)')
    if blew.any():
        days_to_blow = results['blowouts'][blew]
        print(f'  Days to blowout  : p25={int(np.quantile(days_to_blow,0.25)):>3}  '
              f'median={int(np.median(days_to_blow)):>3}  '
              f'p75={int(np.quantile(days_to_blow,0.75)):>3}')
    print(f'  Final equity (survivors): p25=${np.quantile(fe[~blew], 0.25):>+5.0f} '
          f'median=${np.quantile(fe[~blew], 0.50):>+5.0f}  '
          f'p75=${np.quantile(fe[~blew], 0.75):>+5.0f}')
    print(f'  Mean final equity (all) : ${fe.mean():>+5.0f}')
    print(f'  Mean days actually traded: {dt.mean():.1f} of {horizon_days} simulated')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--equity', type=float, default=153.0)
    ap.add_argument('--horizon-days', type=int, default=60,
                    help='Trading days to simulate (default 60 = ~3 months)')
    ap.add_argument('--n-paths', type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values(['day','entry_ts']).reset_index(drop=True)
    n = len(df)
    p_loss = (df['pnl_usd'] < 0).mean() * 100
    print('=' * 96)
    print(f'  BLOWOUT-WITH-INTERVENTION SIMULATION')
    print(f'  Source: {os.path.basename(args.csv)}')
    print(f'  Equity start: ${args.equity:.0f}   Horizon: {args.horizon_days} days   Paths: {args.n_paths:,}')
    print(f'  Trade dist: WR_count={100-p_loss:.1f}% loss%={p_loss:.1f}% mean=${df["pnl_usd"].mean():+.2f}')
    print('=' * 96)

    # Day-1 NT8 gap = +$455 over 27 trades = +$16.85/trade. Treat as upper-bound
    # of how optimistic the gap might be. Python mean = -$2.19/trade.
    # Gap-adjusted = -2.19 + 16.85 = +14.66/trade if gap holds (very optimistic).
    # Realistic: gap could be smaller; test 0 / +5 / +10 / +17 per trade.
    gap_options = [0.0, 5.0, 10.0, 16.85]

    intervention_options = [
        # (label, halt-after-N-consec-losses, daily-drawdown-cap-USD)
        ('No intervention',                                      0,      0),
        ('Halt after 3 consec losses',                           3,      0),
        ('Halt after 5 consec losses',                           5,      0),
        ('Halt at -$50 daily drawdown',                          0,    50),
        ('Halt at -$100 daily drawdown',                         0,   100),
        ('Halt: 3 consec losses OR -$50 daily',                  3,    50),
        ('Halt: 5 consec losses OR -$100 daily',                 5,   100),
    ]

    for gap in gap_options:
        gap_label = f'GAP=+${gap:.2f}/trade' + (' (= Python only)' if gap == 0 else
                       ' (= 30% of Day 1 gap)' if gap == 5 else
                       ' (= 60% of Day 1 gap)' if gap == 10 else
                       ' (= 100% of Day 1 gap)')
        print(f'\n=== {gap_label} ===')
        for label, n_loss, dd in intervention_options:
            r = simulate_paths(df, args.equity, n_loss, dd, gap,
                                args.horizon_days, args.n_paths)
            blow = (r['blowouts'] >= 0).mean() * 100
            surv = 100 - blow
            mean_eq = r['final_equity'].mean()
            print(f'  {label:<48}  surv={surv:>5.1f}%  '
                  f'mean_final_eq=${mean_eq:>+5.0f}')


if __name__ == '__main__':
    main()
