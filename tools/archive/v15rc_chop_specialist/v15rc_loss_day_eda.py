"""
v15rc_loss_day_eda.py -- EDA on losing days for v1.5-RC.

Goal: identify patterns that predict a losing day so we can either avoid them
(filter), size down on them, or design a targeted intervention.

Per-day features computed:
  pnl, n_trades, n_trail, n_pivot, n_eod, n_long, n_short
  pivot_share, trail_share, long_share
  pivot_pnl, trail_pnl    (sum of pnl by exit reason)
  sum_mfe_usd, avg_leg_min, best_trade, worst_trade
  day_of_week

Output:
  reports/findings/v15rc_per_day.csv
  Console: comparison tables (winners vs losers) + bucket analyses
"""
import os
import sys
import pandas as pd
import numpy as np


def main():
    src = 'reports/findings/zigzag_trail_ticker_v15rc.csv'
    df = pd.read_csv(src)

    is25 = lambda d: d.startswith('2025_')
    is26 = lambda d: d.startswith('2026_')

    df['hour_utc'] = (df['entry_ts'] // 3600 % 24).astype(int)

    agg = df.groupby('day').agg(
        pnl       = ('pnl_usd', 'sum'),
        n_trades  = ('pnl_usd', 'size'),
        n_winners = ('pnl_usd', lambda x: int((x > 0).sum())),
        n_losers  = ('pnl_usd', lambda x: int((x < 0).sum())),
        n_trail   = ('exit_reason', lambda x: int((x == 'trail').sum())),
        n_pivot   = ('exit_reason', lambda x: int((x == 'pivot').sum())),
        n_eod     = ('exit_reason', lambda x: int(x.isin(['eod','eod_final']).sum())),
        n_long    = ('direction', lambda x: int((x == 1).sum())),
        n_short   = ('direction', lambda x: int((x == -1).sum())),
        sum_mfe_usd = ('mfe_pts', lambda x: float(x.sum() * 2.0)),
        avg_leg_min = ('leg_min', 'mean'),
        best_trade  = ('pnl_usd', 'max'),
        worst_trade = ('pnl_usd', 'min'),
    ).reset_index()

    pivot_pnl = df[df['exit_reason'] == 'pivot'].groupby('day')['pnl_usd'].sum()
    trail_pnl = df[df['exit_reason'] == 'trail'].groupby('day')['pnl_usd'].sum()
    agg['pivot_pnl'] = pivot_pnl.reindex(agg['day']).fillna(0).values
    agg['trail_pnl'] = trail_pnl.reindex(agg['day']).fillna(0).values

    agg['pivot_share'] = agg['n_pivot'] / agg['n_trades']
    agg['trail_share'] = agg['n_trail'] / agg['n_trades']
    agg['long_share']  = agg['n_long']  / agg['n_trades']
    agg['is_loss_day'] = agg['pnl'] <= 0
    agg['split'] = agg['day'].apply(
        lambda d: 'IS' if is25(d) else ('OOS' if is26(d) else 'OTHER'))
    agg['dow'] = agg['day'].apply(
        lambda d: pd.Timestamp(d.replace('_', '-')).day_name())

    out = agg.copy()

    print('=' * 100)
    print(f'  v1.5-RC LOSING-DAY EDA  (N={len(agg)} active days, '
          f'losing={int(agg["is_loss_day"].sum())}, '
          f'winning={int((~agg["is_loss_day"]).sum())})')
    print('=' * 100)

    # ── Win vs loss comparison
    print(f'\nWINNING vs LOSING DAY MEANS')
    print(f'{"Metric":<22} {"Winning":>12} {"Losing":>12} {"Delta":>10}')
    print('-' * 60)
    for col in ['n_trades', 'n_trail', 'n_pivot', 'n_eod',
                'pivot_share', 'trail_share', 'long_share',
                'pivot_pnl', 'trail_pnl', 'sum_mfe_usd',
                'avg_leg_min', 'best_trade', 'worst_trade']:
        w = agg.loc[~agg['is_loss_day'], col].mean()
        L = agg.loc[ agg['is_loss_day'], col].mean()
        d = w - L
        if abs(w) > 1 or abs(L) > 1:
            print(f'{col:<22} {w:>+12.2f} {L:>+12.2f} {d:>+10.2f}')
        else:
            print(f'{col:<22} {w:>12.3f} {L:>12.3f} {d:>+10.3f}')

    # ── Pivot toxicity
    losers = agg[agg['is_loss_day']]
    print(f'\nPIVOT-EXIT TOXICITY ON LOSING DAYS')
    print(f'  N_loss_days     : {len(losers)}')
    print(f'  Total loss pnl  : ${losers["pnl"].sum():>+8.0f}')
    pp = losers["pivot_pnl"].sum()
    tp = losers["trail_pnl"].sum()
    if losers["pnl"].sum() != 0:
        print(f'  Pivot pnl       : ${pp:>+8.0f}  ({100*pp/losers["pnl"].sum():>5.1f}% of damage)')
        print(f'  Trail pnl       : ${tp:>+8.0f}  (offsetting on the same days)')
    print(f'  Avg pivot count : {losers["n_pivot"].mean():.1f} per loss day '
          f'vs {agg.loc[~agg["is_loss_day"], "n_pivot"].mean():.1f} per win day')

    # ── Volume buckets
    print(f'\nDAY OUTCOME BY TRADE VOLUME')
    agg['vol_bucket'] = pd.cut(
        agg['n_trades'], bins=[0, 15, 25, 40, 100],
        labels=['low(<=15)', 'med(16-25)', 'high(26-40)', 'very_high(>40)'])
    for bucket, sub in agg.groupby('vol_bucket', observed=True):
        if len(sub) == 0: continue
        print(f'  {str(bucket):<14}  N={len(sub):>3}  '
              f'mean_pnl=${sub["pnl"].mean():>+7.2f}  '
              f'dWR={(~sub["is_loss_day"]).mean()*100:>5.1f}%  '
              f'mean_pivot_share={sub["pivot_share"].mean()*100:>4.1f}%')

    # ── Pivot-share buckets
    print(f'\nDAY OUTCOME BY PIVOT-EXIT SHARE')
    agg['pivot_bucket'] = pd.cut(
        agg['pivot_share'], bins=[-0.001, 0.20, 0.30, 0.40, 0.50, 1.001],
        labels=['<20%', '20-30%', '30-40%', '40-50%', '>50%'])
    for bucket, sub in agg.groupby('pivot_bucket', observed=True):
        if len(sub) == 0: continue
        print(f'  pivot_share {str(bucket):<8}  N={len(sub):>3}  '
              f'mean_pnl=${sub["pnl"].mean():>+7.2f}  '
              f'dWR={(~sub["is_loss_day"]).mean()*100:>5.1f}%  '
              f'mean_pivot_pnl=${sub["pivot_pnl"].mean():>+7.0f}')

    # ── Long-share buckets
    print(f'\nDAY OUTCOME BY LONG SHARE')
    agg['long_bucket'] = pd.cut(
        agg['long_share'], bins=[-0.001, 0.30, 0.45, 0.55, 0.70, 1.001],
        labels=['<30%', '30-45%', '45-55%', '55-70%', '>70%'])
    for bucket, sub in agg.groupby('long_bucket', observed=True):
        if len(sub) == 0: continue
        print(f'  long_share {str(bucket):<8}  N={len(sub):>3}  '
              f'mean_pnl=${sub["pnl"].mean():>+7.2f}  '
              f'dWR={(~sub["is_loss_day"]).mean()*100:>5.1f}%')

    # ── Day of week
    print(f'\nDAY OUTCOME BY DAY OF WEEK')
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for dow in dow_order:
        sub = agg[agg['dow'] == dow]
        if len(sub) == 0: continue
        print(f'  {dow:<10}  N={len(sub):>3}  '
              f'mean_pnl=${sub["pnl"].mean():>+7.2f}  '
              f'dWR={(~sub["is_loss_day"]).mean()*100:>5.1f}%  '
              f'med_n_trades={int(sub["n_trades"].median())}')

    # ── Worst-trade-of-day analysis
    print(f'\nWORST-TRADE-OF-DAY DISTRIBUTION (losing vs winning)')
    print(f'  {"Quantile":<10} {"Winning":>10} {"Losing":>10}')
    for q in [0.50, 0.25, 0.10, 0.05]:
        w_q = agg.loc[~agg['is_loss_day'], 'worst_trade'].quantile(q)
        l_q = agg.loc[ agg['is_loss_day'], 'worst_trade'].quantile(q)
        print(f'  q={q:<6}  ${w_q:>+8.2f} ${l_q:>+8.2f}')

    # ── Worst losing days
    print(f'\nTOP 10 WORST LOSING DAYS (OOS)')
    worst = agg[agg['split'] == 'OOS'].nsmallest(10, 'pnl')[
        ['day', 'pnl', 'n_trades', 'n_trail', 'n_pivot',
         'pivot_pnl', 'trail_pnl', 'worst_trade', 'long_share', 'dow']]
    print(worst.to_string(index=False))

    # Save
    out_path = 'reports/findings/v15rc_per_day.csv'
    out.to_csv(out_path, index=False)
    print(f'\nSaved per-day aggregate: {out_path}')


if __name__ == '__main__':
    main()
