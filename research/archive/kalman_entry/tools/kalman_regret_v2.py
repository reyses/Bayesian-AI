"""Regret v2 — fixes v1's flaw: forward-from-(late)-entry can't tell 'wrong direction'
from 'right direction, late entry'. Adds (A) MAGNITUDE of wrongness (not just sign) and
(B) a LOOKBACK to test whether the entry OVER-WAITED (move already underway at trigger).

Per trade, over [entry-LB, entry+FW] on 1s price:
  pre_move   = signed(price[entry] - price[entry-LB]) in the CHOSEN dir  (how much already moved our way)
  fwd MFE    = best favorable excursion AFTER entry, chosen vs flipped
  If entries are RIGHT-direction but LATE: pre_move >> 0 and forward looks symmetric.

DIAGNOSTIC (hindsight). Input: kalman_full_trades.csv + DATA/ATLAS/1s/.
Output: reports/findings/kalman_regret_v2.md (+ .png)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'reports/findings/kalman_clean_trades.csv'   # CLEAN gap-guarded trades
ONE_S = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
LB = 600        # lookback (10m) to capture the triggering move
FW = 900        # forward (15m)
TIE = 3.0       # pts within which chosen vs flip MFE is "effectively tied"


def main():
    tr = pd.read_csv(CSV)
    rows = []
    for day, g in tr.groupby('day'):
        f = f'{ONE_S}/{day}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
        ts = d['timestamp'].to_numpy(np.int64); px = d['close'].to_numpy(np.float64)
        for _, t in g.iterrows():
            ep = float(t['entry_price']); ets = int(t['entry_ts'])
            ie = np.searchsorted(ts, ets); ilb = np.searchsorted(ts, ets - LB); ifw = np.searchsorted(ts, ets + FW, 'right')
            if ilb < 0 or ie - ilb < 30 or ifw - ie < 30 or ie >= len(ts):
                continue
            is_long = t['dir'] == 'LONG'
            sgn = 1 if is_long else -1
            pre = sgn * (px[ie] - px[ilb])                      # move already done in chosen dir
            fwd = px[ie:ifw]
            ch_mfe = max(0.0, sgn * (fwd.max() - ep) if is_long else 0.0)
            # chosen/flip MFE forward
            ch_mfe = max(0.0, (fwd.max() - ep)) if is_long else max(0.0, (ep - fwd.min()))
            fl_mfe = max(0.0, (ep - fwd.min())) if is_long else max(0.0, (fwd.max() - ep))
            # pre-window MFE in chosen dir (how big was the move BEFORE entry)
            pre_win = px[ilb:ie + 1]
            pre_mfe = max(0.0, (px[ie] - pre_win.min())) if is_long else max(0.0, (pre_win.max() - px[ie]))
            rows.append(dict(split=t['split'], pre=pre, pre_mfe=pre_mfe,
                             ch_mfe=ch_mfe, fl_mfe=fl_mfe, mag=ch_mfe - fl_mfe))
    r = pd.DataFrame(rows)
    r['oos'] = r['split'].isin(['OOS_H2_24', 'OOS_25_26'])
    L = ["# GA-Kalman REGRET v2 — magnitude of wrongness + over-wait (entry-lag) test\n",
         f"Lookback {LB}s, forward {FW}s. Tests whether '51% wrong' was direction or LATE entry.\n"]
    for scope, d in [('OOS', r[r['oos']]), ('IS (ref)', r[~r['oos']])]:
        n = len(d); mag = d['mag']
        tied = (mag.abs() <= TIE).mean() * 100
        ch_big = (mag > TIE).mean() * 100; fl_big = (mag < -TIE).mean() * 100
        # over-wait: was the move already going our way at entry?
        pre_pos = (d['pre'] > 0).mean() * 100
        L += [f"## {scope}  (N={n})",
              "### (A) MAGNITUDE of direction-rightness (not just sign)",
              f"- effectively TIED (|chosen−flip MFE| ≤ {TIE:.0f}pt): **{tied:.0f}%** of trades",
              f"- chosen CLEARLY bigger (>+{TIE:.0f}pt): {ch_big:.0f}%   |   flip clearly bigger (wrong): {fl_big:.0f}%",
              f"- mean (chosen−flip) MFE: **{mag.mean():+.1f}pt**  (median {mag.median():+.1f}pt)",
              "### (B) OVER-WAIT / entry-lag test",
              f"- price ALREADY moving our way at entry (pre_move>0): **{pre_pos:.0f}%** of entries",
              f"- mean pre-entry move (chosen dir, {LB//60}m lookback): **{d['pre'].mean():+.1f}pt** "
              f"(median {d['pre'].median():+.1f}pt)",
              f"- mean pre-entry MFE (size of move before entry): {d['pre_mfe'].mean():.1f}pt "
              f"vs forward chosen MFE {d['ch_mfe'].mean():.1f}pt",
              f"- → if pre-move ≫0 and chosen≈flip forward, the entry is RIGHT-direction but LATE "
              f"(over-waited), not directionless.",
              ""]
    # plot: distribution of (chosen-flip) MFE magnitude, OOS
    oos = r[r['oos']]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(oos['mag'].clip(-100, 100), bins=60, color='#1565c0')
    ax[0].axvline(0, color='k', lw=1); ax[0].set_title('OOS: (chosen − flip) forward MFE, pt')
    ax[0].set_xlabel('pt  (>0 = chosen direction better)')
    ax[1].hist(oos['pre'].clip(-100, 100), bins=60, color='#2e7d32')
    ax[1].axvline(0, color='k', lw=1); ax[1].set_title('OOS: pre-entry move in chosen dir (over-wait), pt')
    ax[1].set_xlabel('pt  (>0 = move already going our way at entry)')
    fig.tight_layout(); fig.savefig('reports/findings/kalman_regret_v2.png', dpi=120); plt.close(fig)
    rep = "\n".join(L)
    open('reports/findings/kalman_regret_v2.md', 'w', encoding='utf-8').write(rep)
    print(rep.encode('ascii', 'replace').decode())
    print("\n[plot -> reports/findings/kalman_regret_v2.png]")


if __name__ == '__main__':
    main()
