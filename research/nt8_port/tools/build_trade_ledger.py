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
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
GB = os.path.join(PROJ, 'golden_backtest')
NT8_5S = os.path.join(ROOT, 'DATA', 'ATLAS_NT8', '5s')
OUT_CSV = os.path.join(PROJ, 'reports', 'trade_ledger_v04.csv')
OUT_MD = os.path.join(PROJ, 'reports', 'trade_ledger_v04.md')

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
                reason = None; x_ts = ts; x_px = px
                # catastrophic stop within this bar (5s scan since last bar)
                adv, adv_ts = worst_adverse(df5, int(bar_ts[i - 1]), ts, e_px, pos)
                if adv >= CAT_STOP_PTS:
                    reason = 'CAT_STOP'; x_px = e_px - pos * CAT_STOP_PTS
                    x_ts = int(adv_ts) if adv_ts else ts
                elif int(v['zz_confirm'].iloc[i]) == -pos:        # R-trigger reversal
                    reason = 'R_TRIGGER'
                elif i == len(v) - 1:
                    reason = 'SESSION_CLOSE'
                if reason:
                    pnl_pts = pos * (x_px - e_px)
                    trades.append(dict(
                        day=day, pivot_ts=piv_ts, pivot_bar=piv_bar,
                        n_bars=n_bars, entry_ts=e_ts, entry_bar=e_bar,
                        dir=e_dir, entry_px=round(e_px, 2),
                        exit_ts=x_ts, exit_bar=ts_to_bar.get(int(x_ts), ''),
                        exit_px=round(x_px, 2), exit_reason=reason,
                        pnl_pts=round(pnl_pts, 2),
                        pnl_usd=round(pnl_pts * PT_USD - COMM_USD, 2),
                        combiner_P=round(e_P, 4)))
                    pos = 0
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
    net = tdf['pnl_usd'].sum()
    byday = tdf.groupby('day')['pnl_usd'].sum()
    wins = tdf[tdf['pnl_usd'] > 0]['pnl_usd'].sum()
    loss = -tdf[tdf['pnl_usd'] < 0]['pnl_usd'].sum()
    pf_wr = wins / loss - 1 if loss else float('nan')
    rc = tdf['exit_reason'].value_counts().to_dict()
    n_series = tdf['n_bars'].dropna()
    lines = [
        '# Trade ledger v0.4 (rebuilt from frozen golden vectors, R-trigger LIVE)',
        f'{len(tdf)} trades, {tdf["day"].nunique()} days. '
        f'CSV: reports/trade_ledger_v04.csv (ATLAS-bar-aligned).',
        '',
        f'- Net: ${net:,.0f} | Day-WR: {(byday>0).mean():.0%} ({(byday>0).sum()}/{len(byday)})',
        f'- Trade WR (PF-based): {pf_wr:+.2f}',
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
