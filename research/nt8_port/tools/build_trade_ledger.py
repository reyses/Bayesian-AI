#!/usr/bin/env python3
"""TRADE LEDGER from the frozen golden-backtest decision vectors (owner
2026-07-26: "rerun it; we need t=pvt+n; output CSV; entry/exit bars aligned
with ATLAS for bar-by-bar review — this is what the GBM is for").

Source: research/nt8_port/golden_backtest/<day>.parquet — the PROVEN reference
decider's per-1m-bar output (entry/entry_dir, zz_confirm reversal, zz_pivot_age
_min, zz_pivot_price, P_topk). The R-trigger fires correctly here (the v0.2
0-fires bug was the NT8 WRAPPER's warmup, not this reference core).

Rebuilds the trade sequence (Architecture B: ensemble entry + R-trigger
ride-only exit + optional catastrophic stop) and emits one row per trade with
ATLAS-aligned bar indices + timestamps so each trade is reviewable bar-by-bar:
  day, pivot_ts, pivot_bar, n_bars(=entry-pivot=t-pvt), entry_ts, entry_bar,
  dir, entry_px, exit_ts, exit_bar, exit_px, exit_reason, pnl_pts, pnl_usd,
  combiner_P
CPU-only. Writes research/nt8_port/reports/trade_ledger_v04.csv + a summary md.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
_ap = argparse.ArgumentParser()
_ap.add_argument('--gb', default=os.path.join(PROJ, 'golden_backtest'))
_ap.add_argument('--fives', default=os.path.join(ROOT, 'DATA', 'ATLAS_NT8', '5s'))
_ap.add_argument('--tag', default='v04')
_A = _ap.parse_args()
GB = _A.gb
NT8_5S = _A.fives
OUT_CSV = os.path.join(PROJ, 'reports', f'trade_ledger_{_A.tag}.csv')
OUT_MD = os.path.join(PROJ, 'reports', f'trade_ledger_{_A.tag}.md')

TICK, PT_USD = 0.25, 2.0        # MNQ: 0.25 pt/tick, $2/pt (1 contract)
CAT_STOP_PTS = 50.0             # v0.4 native resting catastrophic stop
COMM_USD = 1.00                 # round-trip commission proxy


def minute_close(day):
    """1m close series for the day from the 5s substrate: {minute_ts: close}."""
    p = os.path.join(NT8_5S, f'{day}.parquet')
    if not os.path.exists(p):
        return None, None
    df = pd.read_parquet(p, columns=['timestamp', 'close', 'high', 'low'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['minute'] = (df['timestamp'] // 60) * 60
    m = df.groupby('minute').agg(close=('close', 'last')).reset_index()
    return m, df                # m = per-minute close; df = raw 5s for stop scan


def worst_adverse(df5, t0, t1, entry_px, d):
    """Most-adverse excursion (pts) over 5s bars in (t0, t1] for dir d."""
    seg = df5[(df5['timestamp'] > t0) & (df5['timestamp'] <= t1)]
    if seg.empty:
        return 0.0, None
    if d > 0:
        i = seg['low'].idxmin(); adv = entry_px - seg.loc[i, 'low']
    else:
        i = seg['high'].idxmax(); adv = seg.loc[i, 'high'] - entry_px
    return adv, seg.loc[i, 'timestamp']


def main():
    trades = []
    for f in sorted(glob.glob(os.path.join(GB, '*.parquet'))):
        day = os.path.basename(f).replace('.parquet', '')
        v = pd.read_parquet(f).sort_values('bar_ts').reset_index(drop=True)
        mclose, df5 = minute_close(day)
        if mclose is None:
            continue
        cl = dict(zip(mclose['minute'], mclose['close']))
        bar_ts = v['bar_ts'].to_numpy()
        ts_to_bar = {int(t): i for i, t in enumerate(bar_ts)}   # ATLAS-aligned idx

        pos = 0            # 0 flat, else entry_dir
        stopped_px = stopped_ts = None
        e_ts = e_bar = e_px = e_dir = e_P = None
        piv_ts = piv_bar = n_bars = None
        for i in range(len(v)):
            ts = int(bar_ts[i])
            px = cl.get(ts)
            if px is None:
                continue
            zc = int(v['entry_dir'].iloc[i]) if 'entry_dir' in v else 0
            # ---- exit checks first (if in position) ----
            if pos != 0:
                # RIDE exit = R-trigger reversal or session close (no stop)
                ride_reason = None
                if int(v['zz_confirm'].iloc[i]) == -pos:
                    ride_reason = 'R_TRIGGER'
                elif i == len(v) - 1:
                    ride_reason = 'SESSION_CLOSE'
                # STOP: did a >=CAT_STOP adverse happen intrabar BEFORE the ride exit?
                if not stopped_px:
                    adv, adv_ts = worst_adverse(df5, int(bar_ts[i - 1]), ts, e_px, pos)
                    if adv >= CAT_STOP_PTS:
                        stopped_px = e_px - pos * CAT_STOP_PTS
                        stopped_ts = int(adv_ts) if adv_ts else ts
                if ride_reason:
                    ride_pnl = pos * (px - e_px)
                    # with-stop pnl = stop if it fired before this ride exit, else ride
                    if stopped_px and stopped_ts <= ts:
                        stop_pnl = pos * (stopped_px - e_px); ereason = 'CAT_STOP'
                        x_ts = stopped_ts
                    else:
                        stop_pnl = ride_pnl; ereason = ride_reason; x_ts = ts
                    trades.append(dict(
                        day=day, pivot_ts=piv_ts, pivot_bar=piv_bar,
                        n_bars=n_bars, entry_ts=e_ts, entry_bar=e_bar,
                        dir=e_dir, entry_px=round(e_px, 2),
                        exit_ts=x_ts, exit_bar=ts_to_bar.get(int(x_ts), ''),
                        exit_px=round(px, 2), exit_reason=ereason,
                        ride_pnl_usd=round(ride_pnl * PT_USD - COMM_USD, 2),
                        stop_pnl_usd=round(stop_pnl * PT_USD - COMM_USD, 2),
                        combiner_P=round(e_P, 4)))
                    pos = 0; stopped_px = None; stopped_ts = None
            # ---- entry (if flat) ----
            if pos == 0 and int(v['entry'].iloc[i]) == 1:
                pos = int(v['entry_dir'].iloc[i])
                e_ts, e_bar, e_px, e_dir = ts, i, px, pos
                e_P = float(v['P_topk'].iloc[i])
                age = v['zz_pivot_age_min'].iloc[i]
                n_bars = int(age) if pd.notna(age) else None      # t = pvt + n
                piv_bar = (e_bar - n_bars) if n_bars is not None else None
                piv_ts = int(bar_ts[piv_bar]) if (piv_bar is not None and 0 <= piv_bar < len(bar_ts)) else None

    tdf = pd.DataFrame(trades)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    tdf.to_csv(OUT_CSV, index=False)

    # summary
    ride_net = tdf['ride_pnl_usd'].sum()
    stop_net = tdf['stop_pnl_usd'].sum()
    byday = tdf.groupby('day')['ride_pnl_usd'].sum()
    byday_s = tdf.groupby('day')['stop_pnl_usd'].sum()
    wins = tdf[tdf['ride_pnl_usd'] > 0]['ride_pnl_usd'].sum()
    loss = -tdf[tdf['ride_pnl_usd'] < 0]['ride_pnl_usd'].sum()
    pf_wr = wins / loss - 1 if loss else float('nan')
    rc = tdf['exit_reason'].value_counts().to_dict()
    n_series = tdf['n_bars'].dropna()
    lines = [
        '# Trade ledger v0.4 (rebuilt from frozen golden vectors, R-trigger LIVE)',
        f'{len(tdf)} trades, {tdf["day"].nunique()} days. '
        f'CSV: reports/trade_ledger_v04.csv (ATLAS-bar-aligned).',
        '',
        f'- RIDE-ONLY (R-trigger/session, NO stop): net ${ride_net:,.0f} | '
        f'day-WR {(byday>0).mean():.0%} | ${ride_net/max(1,len(byday)):,.0f}/day',
        f'- WITH 50pt stop: net ${stop_net:,.0f} | day-WR {(byday_s>0).mean():.0%}',
        f'- Trade WR (PF-based, ride-only): {pf_wr:+.2f}',
        f'- exit reasons: {rc}',
        f'- **t = pvt + n**: n(bars from pivot to entry) median {n_series.median():.0f}, '
        f'mean {n_series.mean():.1f}, p10-p90 [{n_series.quantile(.1):.0f}, {n_series.quantile(.9):.0f}]',
        '',
        'R-trigger firing (vs v0.2 backtest where it fired 0x) is the key check.',
        'Every row carries pivot_bar / entry_bar / exit_bar as ATLAS indices +',
        'timestamps for bar-by-bar review (the GBM input).',
    ]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
