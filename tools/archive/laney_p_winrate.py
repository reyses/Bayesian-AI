"""
laney_p_winrate.py -- Laney P' control chart for trade win-proportion.
=========================================================================

Standard Shewhart P-chart for win rate assumes binomial independence:
  sigma_p = sqrt(p_bar * (1 - p_bar) / n_i)

Trades aren't independent (chop days produce loss streaks) -> over-dispersion.
Standard P-chart will flag many days as "out of control" when really we have
a regime-dependent variance structure. Laney P' (P-prime, 2002) corrects:
  z_i = (p_i - p_bar) / sigma_p_i
  sigma_z  = std(z_i)            // over-dispersion factor (= 1 under binomial)
  sigma_p' = sigma_p * sigma_z
  UCL/LCL  = p_bar +/- 3 * sigma_p'

Two metrics computed per day:

  1) Count-based win rate    p_i = (count winning trades) / (count total)
  2) Dollar-share win rate   p_i = sum(profits) / (sum(profits) + sum(|losses|))
     Equivalent to PF / (PF + 1) where PF = profit-factor; > 0.5 = profitable.

Laney P' gives "true" CI on each, accounting for empirical cluster structure.

Usage:
    python tools/laney_p_winrate.py --csv reports/findings/zigzag_trail_sl10.csv
    python tools/laney_p_winrate.py --csv reports/findings/zigzag_trail_ticker_v15rc.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def is_2025(d): return d.startswith('2025_')
def is_2026(d): return d.startswith('2026_')


def laney_p_chart(per_day_p: np.ndarray, per_day_n: np.ndarray, name: str):
    """Compute centerline, std-binomial sigma, Laney sigma_z, control limits."""
    mask = per_day_n > 0
    p = per_day_p[mask]
    n = per_day_n[mask]
    if len(p) == 0:
        return None

    # Centerline = pooled proportion (sum of wins / sum of trials)
    # For dollar-based, "wins" already = sum(profits) so we use weighted mean
    p_bar = float(np.average(p, weights=n))

    # Binomial sigma per subgroup
    sigma_p = np.sqrt(p_bar * (1 - p_bar) / n)

    # Z-scores per subgroup
    z = (p - p_bar) / np.where(sigma_p > 0, sigma_p, 1)
    sigma_z = float(np.std(z, ddof=1))   # over-dispersion factor

    # Laney sigma per subgroup
    sigma_laney = sigma_p * sigma_z

    # Mean Laney sigma (for an "average-N day")
    mean_n = float(np.mean(n))
    avg_binom_sigma = float(np.sqrt(p_bar * (1 - p_bar) / mean_n))
    avg_laney_sigma = avg_binom_sigma * sigma_z

    return {
        'name': name,
        'n_days': int(mask.sum()),
        'p_bar': p_bar,
        'mean_n_per_day': mean_n,
        'binom_sigma_avg': avg_binom_sigma,
        'sigma_z': sigma_z,
        'laney_sigma_avg': avg_laney_sigma,
        'binom_UCL_3s': p_bar + 3 * avg_binom_sigma,
        'binom_LCL_3s': max(0.0, p_bar - 3 * avg_binom_sigma),
        'laney_UCL_3s': min(1.0, p_bar + 3 * avg_laney_sigma),
        'laney_LCL_3s': max(0.0, p_bar - 3 * avg_laney_sigma),
        'pct_overdisp': (sigma_z - 1) * 100,
    }


def report(stats, label):
    if stats is None:
        print(f'{label}: no data')
        return
    print(f'\n--- {label} ---')
    print(f'  Days                : {stats["n_days"]}')
    print(f'  Centerline (p_bar)  : {stats["p_bar"]:.4f}  (= {stats["p_bar"]*100:.2f}%)')
    print(f'  Avg trades/day      : {stats["mean_n_per_day"]:.1f}')
    print(f'  Binomial sigma      : {stats["binom_sigma_avg"]:.4f}')
    print(f'  Sigma_z (over-disp) : {stats["sigma_z"]:.3f}  '
          f'(= 1.0 under binomial; >1 means clustered)')
    print(f'  Laney sigma         : {stats["laney_sigma_avg"]:.4f}  '
          f'({stats["pct_overdisp"]:+.0f}% wider than binomial)')
    print(f'  Binomial 3s bounds  : [{stats["binom_LCL_3s"]:.4f}, {stats["binom_UCL_3s"]:.4f}]')
    print(f'  Laney   3s bounds   : [{stats["laney_LCL_3s"]:.4f}, {stats["laney_UCL_3s"]:.4f}]')
    width_b = stats["binom_UCL_3s"] - stats["binom_LCL_3s"]
    width_l = stats["laney_UCL_3s"] - stats["laney_LCL_3s"]
    print(f'  Width binom         : {width_b:.4f}')
    print(f'  Width Laney         : {width_l:.4f}  ({width_l/width_b:.2f}x binom)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--split', default='all', choices=['all', 'IS', 'OOS'])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.split == 'IS':
        df = df[df['day'].apply(is_2025)]
    elif args.split == 'OOS':
        df = df[df['day'].apply(is_2026)]
    if len(df) == 0:
        print(f'No trades for split={args.split}')
        return

    # Per-day metrics
    by_day = df.groupby('day').agg(
        n=('pnl_usd', 'size'),
        n_wins=('pnl_usd', lambda x: int((x > 0).sum())),
        sum_profits=('pnl_usd', lambda x: float(x[x > 0].sum())),
        sum_losses_abs=('pnl_usd', lambda x: float(abs(x[x < 0].sum()))),
    ).reset_index()
    by_day['p_count'] = by_day['n_wins'] / by_day['n']
    by_day['total_abs'] = by_day['sum_profits'] + by_day['sum_losses_abs']
    by_day['p_dollar'] = np.where(
        by_day['total_abs'] > 0,
        by_day['sum_profits'] / by_day['total_abs'],
        0.0,
    )

    print('=' * 96)
    print(f'  LANEY P-prime WIN PROPORTION CONTROL CHART')
    print(f'  Source: {os.path.basename(args.csv)}  Split: {args.split}')
    print(f'  Total trades: {len(df):,}  Days: {len(by_day)}  '
          f'Pooled count-WR: {df["pnl_usd"].gt(0).mean()*100:.2f}%')
    sum_p = float(df.loc[df["pnl_usd"]>0, "pnl_usd"].sum())
    sum_l = float(abs(df.loc[df["pnl_usd"]<0, "pnl_usd"].sum()))
    pf = sum_p / max(sum_l, 1)
    print(f'  Profit factor (sum_profit/|sum_loss|): {pf:.4f}  '
          f'(=> dollar-share WR = {pf/(pf+1):.4f} = {pf/(pf+1)*100:.2f}%)')
    print(f'  PF-based Trade WR (= PF - 1): {pf - 1:+.4f}')
    print('=' * 96)

    s_count  = laney_p_chart(by_day['p_count'].values,  by_day['n'].values,         'count')
    s_dollar = laney_p_chart(by_day['p_dollar'].values, by_day['total_abs'].values, 'dollar')

    report(s_count,  'COUNT-BASED win rate per day  (p = wins / total_trades)')
    report(s_dollar, 'DOLLAR-SHARE win rate per day (p = sum_profit / sum_abs_pnl)')

    # Save per-day data
    out = args.csv.replace('.csv', '_laney_p.csv')
    by_day.to_csv(out, index=False)
    print(f'\nWrote per-day data: {out}')

    # Honest takeaway
    print('\n--- INTERPRETATION ---')
    if s_dollar:
        cl = s_dollar['p_bar']
        lcl = s_dollar['laney_LCL_3s']
        ucl = s_dollar['laney_UCL_3s']
        if ucl < 0.5:
            print(f'  Dollar-share WR Laney 3s UCL = {ucl:.4f} (< 0.50). '
                  f'Even at upper control limit, strategy loses money. '
                  f'Strategy has NEGATIVE expectancy at this configuration.')
        elif lcl > 0.5:
            print(f'  Dollar-share WR Laney 3s LCL = {lcl:.4f} (> 0.50). '
                  f'Strategy is statistically profitable across the regime.')
        else:
            print(f'  Dollar-share WR Laney 3s bounds [{lcl:.4f}, {ucl:.4f}] '
                  f'straddle 0.50. Strategy expectancy is statistically '
                  f'indistinguishable from breakeven.')
    if s_count:
        od = s_count['sigma_z']
        if od > 1.5:
            print(f'  Strong over-dispersion (sigma_z = {od:.2f}) — losses cluster '
                  f'on chop days. Standard binomial CI is too tight; use Laney.')


if __name__ == '__main__':
    main()
