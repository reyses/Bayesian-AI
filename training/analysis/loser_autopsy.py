"""Loser autopsy on production OOS trades.

Multi-axis EDA on the negative trades from `NMP_REGIME` OOS run:

    1. Loss-magnitude distribution    — small losses vs catastrophic
    2. Tier × regime × direction      — where do losers cluster?
    3. Time-of-day                    — bad sessions
    4. Day-of-week / day-of-month      — calendar patterns
    5. Hold time + exit reason        — are losers held too long? exit fires the right way?
    6. Capture-on-failure              — did losers EVER show profit, or always negative?
    7. Path archetype                  — straight-down vs round-trip vs spike-and-fade
    8. Feature-quartile breakdown of TOP-3 V2 columns at entry on losers vs winners
    9. Catastrophic loss day analysis  — single-day biggest losers

Goal: surface actionable patterns. Each axis is a candidate for an entry
filter or an exit improvement.
"""
from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training.regret.regret import RegretLabel
from training.ledger import ClosedTrade
from training.calibration.tier_discovery import load_joined
from training.utils.state import REGIME_VOCAB


def load_full(trades_path: str, regret_path: str) -> pd.DataFrame:
    """Like tier_discovery.load_joined but also includes hold time + exit reason."""
    with open(trades_path, 'rb') as f:
        trades: List[ClosedTrade] = pickle.load(f)
    with open(regret_path, 'rb') as f:
        labels: List[RegretLabel] = pickle.load(f)

    label_by_key = {(l.entry_day, l.entry_ts): l for l in labels}
    rows = []
    for t in trades:
        l = label_by_key.get((t.entry_day, t.entry_ts))
        if l is None or t.entry_v2 is None or len(t.entry_v2) != len(FEATURE_NAMES):
            continue
        rows.append({
            'day': t.entry_day,
            'ts': t.entry_ts,
            'tier': t.entry_tier,
            'direction': t.direction,
            'regime_idx': t.entry_regime_idx,
            'pnl': t.pnl,
            'bars_held': t.bars_held,
            'exit_reason': t.exit_reason,
            'peak_pnl': l.peak_pnl,
            'mae_pnl': l.mae_pnl,
            'time_to_peak_s': l.time_to_peak_s,
            'capture_ratio': l.capture_ratio,
            'pnl_path': l.pnl_path,
            'entry_v2': np.asarray(t.entry_v2, dtype=np.float32),
        })
    return pd.DataFrame(rows)


def parse_hour(ts: float) -> int:
    return datetime.utcfromtimestamp(int(ts)).hour


def parse_weekday(day: str) -> str:
    """day = 'YYYY_MM_DD'"""
    try:
        d = datetime.strptime(day, '%Y_%m_%d')
        return d.strftime('%a')
    except Exception:
        return '?'


def main():
    p = argparse.ArgumentParser(description='Loser autopsy on production OOS trades')
    p.add_argument('--trades', default='training_iso_v2/output/nmp_regime_oos.pkl')
    p.add_argument('--regret', default='training_iso_v2/output/regret_nmp_regime_oos.pkl')
    p.add_argument('--out', default='reports/findings/v2_loser_autopsy.md')
    p.add_argument('--margin', type=float, default=0.0,
                       help='loss threshold; trades with pnl < -margin are losers')
    args = p.parse_args()

    df = load_full(args.trades, args.regret)
    print(f'Loaded {len(df)} trades')

    df['hour_utc'] = df['ts'].apply(parse_hour)
    df['weekday'] = df['day'].apply(parse_weekday)
    df['regime'] = df['regime_idx'].apply(
        lambda r: REGIME_VOCAB[int(r)] if int(r) < len(REGIME_VOCAB) else f'R{r}')

    losers = df[df['pnl'] < -args.margin].copy()
    winners = df[df['pnl'] > args.margin].copy()
    flat = df[(df['pnl'] >= -args.margin) & (df['pnl'] <= args.margin)].copy()

    print(f'\nDistribution at margin=${args.margin}:')
    print(f'  Winners: {len(winners):>5} ({len(winners)/len(df):.1%})  '
              f'total ${winners["pnl"].sum():>+9.0f}, ${winners["pnl"].mean():>+6.2f}/trade')
    print(f'  Losers : {len(losers):>5} ({len(losers)/len(df):.1%})  '
              f'total ${losers["pnl"].sum():>+9.0f}, ${losers["pnl"].mean():>+6.2f}/trade')
    print(f'  Flat   : {len(flat):>5} ({len(flat)/len(df):.1%})  '
              f'total ${flat["pnl"].sum():>+9.0f}, ${flat["pnl"].mean():>+6.2f}/trade')
    print(f'  Net    : ${df["pnl"].sum():>+9.0f}')

    # ── 1. Loss-magnitude distribution ────────────────────────────────────
    print(f'\n=== 1. Loss-magnitude distribution ===')
    bins = [(-1e9, -100), (-100, -50), (-50, -25), (-25, -10), (-10, 0)]
    print(f'  {"range":<20} {"n":>6} {"$_total":>10} {"$_mean":>8}')
    for lo, hi in bins:
        m = (losers['pnl'] >= lo) & (losers['pnl'] < hi)
        sub = losers[m]
        if len(sub) == 0:
            continue
        print(f'  ({lo:>+5.0f}, {hi:>+4.0f}]    {len(sub):>5}  '
                  f'${sub["pnl"].sum():>+9.0f}  ${sub["pnl"].mean():>+7.2f}')

    # ── 2. Tier × regime × direction ─────────────────────────────────────
    print(f'\n=== 2. Loser distribution by (tier, regime, direction) ===')
    grp = df.groupby(['tier', 'regime', 'direction']).agg(
        n=('pnl', 'size'),
        mean_pnl=('pnl', 'mean'),
        loser_n=('pnl', lambda x: int((x < -args.margin).sum())),
        loser_total=('pnl', lambda x: float(x[x < -args.margin].sum())),
    ).reset_index()
    grp['loser_rate'] = grp['loser_n'] / grp['n']
    grp = grp.sort_values('loser_total', ascending=True)
    print(f'  {"tier":<10} {"regime":<14} {"dir":>5} {"n":>5} {"$_mean":>8} '
              f'{"loser_n":>7} {"loser_$":>9} {"loser%":>7}')
    for _, r in grp.iterrows():
        print(f'  {r["tier"]:<10} {r["regime"]:<14} {r["direction"]:>5} '
                  f'{int(r["n"]):>5} ${r["mean_pnl"]:>+7.2f} {int(r["loser_n"]):>7} '
                  f'${r["loser_total"]:>+8.0f}  {r["loser_rate"]:>6.1%}')

    # ── 3. Time-of-day ────────────────────────────────────────────────────
    print(f'\n=== 3. Loser distribution by hour (UTC) ===')
    by_hour = df.groupby('hour_utc').agg(
        n=('pnl', 'size'),
        mean_pnl=('pnl', 'mean'),
        loser_n=('pnl', lambda x: int((x < -args.margin).sum())),
        loser_total=('pnl', lambda x: float(x[x < -args.margin].sum())),
        winner_total=('pnl', lambda x: float(x[x > args.margin].sum())),
    ).reset_index()
    by_hour['net'] = by_hour['mean_pnl'] * by_hour['n']
    print(f'  {"hr_UTC":>6} {"n":>5} {"$_mean":>8} {"$_total":>9} '
              f'{"loser_$":>9} {"winner_$":>9}')
    for _, r in by_hour.iterrows():
        print(f'  {int(r["hour_utc"]):>6} {int(r["n"]):>5} ${r["mean_pnl"]:>+7.2f} '
                  f'${r["net"]:>+8.0f} ${r["loser_total"]:>+8.0f} '
                  f'${r["winner_total"]:>+8.0f}')

    # ── 4. Weekday ────────────────────────────────────────────────────────
    print(f'\n=== 4. Loser distribution by weekday ===')
    wk_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    by_wk = df.groupby('weekday').agg(
        n=('pnl', 'size'),
        mean_pnl=('pnl', 'mean'),
        net=('pnl', 'sum'),
        loser_total=('pnl', lambda x: float(x[x < -args.margin].sum())),
    ).reindex(wk_order).dropna()
    print(f'  {"day":<5} {"n":>5} {"$_mean":>8} {"$_total":>9} {"loser_$":>9}')
    for wd, r in by_wk.iterrows():
        print(f'  {wd:<5} {int(r["n"]):>5} ${r["mean_pnl"]:>+7.2f} '
                  f'${r["net"]:>+8.0f} ${r["loser_total"]:>+8.0f}')

    # ── 5. Hold time + exit reason ───────────────────────────────────────
    print(f'\n=== 5. Loser hold time + exit reason ===')
    print(f'Loser bars_held distribution:')
    for q in [0.25, 0.5, 0.75, 0.95, 1.0]:
        v = losers['bars_held'].quantile(q)
        print(f'  Q{int(q*100):>3}: {v:>5.0f} bars  ({v*5/60:.1f} min)')
    print(f'\nLoser exit reasons:')
    er = losers.groupby('exit_reason').agg(
        n=('pnl', 'size'),
        total=('pnl', 'sum'),
        mean=('pnl', 'mean'),
    ).sort_values('n', ascending=False)
    for reason, r in er.iterrows():
        print(f'  {str(reason):<25} {int(r["n"]):>5}  ${r["total"]:>+9.0f}  '
                  f'${r["mean"]:>+6.2f}/trade')

    # ── 6. Capture-on-failure ────────────────────────────────────────────
    print(f'\n=== 6. Capture-on-failure: did losers EVER show profit? ===')
    losers['saw_profit'] = losers['peak_pnl'] > 5.0
    losers['always_underwater'] = losers['peak_pnl'] <= 0.0
    print(f'  Saw any profit (peak > $5)    : {int(losers["saw_profit"].sum()):>5} '
              f'({losers["saw_profit"].mean():.1%})')
    print(f'  Always underwater (peak <= 0) : {int(losers["always_underwater"].sum()):>5} '
              f'({losers["always_underwater"].mean():.1%})')
    print(f'  Mean peak of losers           : ${losers["peak_pnl"].mean():>+7.2f}')
    print(f'  Mean MAE of losers            : ${losers["mae_pnl"].mean():>+7.2f}')
    print(f'  Among "saw profit" losers     : peak ${losers[losers["saw_profit"]]["peak_pnl"].mean():>+6.2f}, '
              f'realized ${losers[losers["saw_profit"]]["pnl"].mean():>+6.2f}')

    # ── 7. Path archetype ────────────────────────────────────────────────
    # Bucket each loser's path:
    #   STRAIGHT_DOWN  : peak_pnl <= 0 AND mae_pnl matches realized
    #   ROUND_TRIP     : peak_pnl > 20 (had real profit)
    #   SPIKE_FADE     : peak_pnl 0-20 (small profit then fade)
    print(f'\n=== 7. Loser path archetype ===')
    cond_straight = losers['peak_pnl'] <= 0
    cond_round = losers['peak_pnl'] > 20
    cond_spike = (~cond_straight) & (~cond_round)
    print(f'  STRAIGHT_DOWN (peak <= 0)         : {int(cond_straight.sum()):>5} '
              f'({cond_straight.mean():.1%})  ${losers[cond_straight]["pnl"].sum():>+8.0f}')
    print(f'  SPIKE_FADE    (peak 0-20)         : {int(cond_spike.sum()):>5} '
              f'({cond_spike.mean():.1%})  ${losers[cond_spike]["pnl"].sum():>+8.0f}')
    print(f'  ROUND_TRIP    (peak > 20)         : {int(cond_round.sum()):>5} '
              f'({cond_round.mean():.1%})  ${losers[cond_round]["pnl"].sum():>+8.0f}')

    # ── 8. Catastrophic-loss days ────────────────────────────────────────
    print(f'\n=== 8. Top 10 worst OOS days for the strategy ===')
    by_day = df.groupby('day')['pnl'].agg(['count', 'sum', 'mean'])
    worst = by_day.sort_values('sum').head(10)
    print(f'  {"day":<14} {"trades":>7} {"$_total":>9} {"$_mean":>8}')
    for day, r in worst.iterrows():
        print(f'  {day:<14} {int(r["count"]):>7} ${r["sum"]:>+8.0f} ${r["mean"]:>+7.2f}')

    # ── 9. Save markdown report ──────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('# V2 Loser Autopsy on Production OOS\n\n')
        f.write(f'Trades: {len(df)}; margin = ${args.margin}\n\n')
        f.write(f'Winners {len(winners)} ({len(winners)/len(df):.1%}), '
                    f'Losers {len(losers)} ({len(losers)/len(df):.1%}), '
                    f'Flat {len(flat)} ({len(flat)/len(df):.1%}). '
                    f'Net ${df["pnl"].sum():+,.0f}\n\n')
        f.write('## Key takeaways\n\n')
        f.write(f'- Mean loser size: ${losers["pnl"].mean():.2f}\n')
        f.write(f'- Catastrophic losses (<-$50): {int((losers["pnl"]<-50).sum())}\n')
        f.write(f'- Round-trip losers (peak > $20 then lost): '
                    f'{int(cond_round.sum())} ({cond_round.mean():.1%})\n')
        f.write(f'- Always-underwater losers: {int(cond_straight.sum())} '
                    f'({cond_straight.mean():.1%})\n')
        f.write(f'\n## Per-cell loser rate\n\n')
        f.write(grp.to_string(index=False))
        f.write(f'\n\n## Per-hour PnL\n\n')
        f.write(by_hour.to_string(index=False))
        f.write(f'\n\n## Worst 10 days\n\n')
        f.write(worst.to_string())
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
