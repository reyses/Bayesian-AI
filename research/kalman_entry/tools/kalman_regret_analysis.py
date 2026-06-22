"""Regret analysis of the GA-Kalman trades — decompose ENTRY (direction) vs EXIT (giveback).

(A) ENTRY/DIRECTION regret: over a FIXED forward window from entry (decoupled from the
    flawed exit), compute MFE if LONG vs MFE if SHORT. Did the entry pick the right side?
    - picked_wrong = flipped direction had the larger MFE.
    - If ~50% picked wrong → coin flip (no directional edge). >50% → anti-predictive.

(B) EXIT regret (only for trades that DID get a favorable move, i.e. right direction at some
    point): within the realized window, find the MFE peak, the realized exit, the giveback
    (MFE − exit), and the TIME from the peak to the exit. Bucket by time-from-peak.

Causal? No — regret is inherently hindsight; this is a DIAGNOSTIC (firewall-fine), not a signal.

Input:  research/kalman_tuning_eda/reports/findings/kalman_full_trades.csv  + DATA/ATLAS/1s/
Output: reports/findings/kalman_regret_analysis.md (+ .png)
Run: python research/kalman_regret_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'research/kalman_tuning_eda/reports/findings/kalman_full_trades.csv'
ONE_S = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
FWD_WINDOW = 900          # entry-regret forward horizon (15 min @ 1s) — decoupled from exit
USD_PER_PT = 2.0


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
            ep = float(t['entry_price']); ets = int(t['entry_ts']); xts = int(t['exit_ts'])
            i0 = np.searchsorted(ts, ets)
            iw = np.searchsorted(ts, ets + FWD_WINDOW, 'right')   # fixed forward window
            ix = np.searchsorted(ts, xts, 'right')                # realized exit
            if i0 >= len(ts) or iw <= i0:
                continue
            fwd = px[i0:iw]                                       # raw price over forward window
            mfe_long = max(0.0, float(fwd.max() - ep))
            mfe_short = max(0.0, float(ep - fwd.min()))
            is_long = t['dir'] == 'LONG'
            chosen_mfe = mfe_long if is_long else mfe_short
            flip_mfe = mfe_short if is_long else mfe_long
            net_drift = (float(fwd[-1] - ep)) * (1 if is_long else -1)   # signed to chosen dir
            # realized-window path (chosen-signed) for exit regret
            real = px[i0:max(ix, i0 + 1)]
            sp = (real - ep) if is_long else (ep - real)
            pk = int(np.argmax(sp)); mfe_real = float(sp[pk]); exitv = float(sp[-1])
            rows.append(dict(split=t['split'], dir=t['dir'], net_usd=t['net_usd'],
                             chosen_mfe=chosen_mfe, flip_mfe=flip_mfe,
                             oracle_mfe=max(mfe_long, mfe_short), net_drift=net_drift,
                             mfe_real=mfe_real, giveback=mfe_real - exitv,
                             secs_from_peak=(len(sp) - 1 - pk)))
    r = pd.DataFrame(rows)
    r['oos'] = r['split'].isin(['OOS_H2_24', 'OOS_25_26'])
    L = ["# GA-Kalman REGRET analysis — entry (direction) vs exit (giveback)\n",
         f"Entry regret over a fixed {FWD_WINDOW}s forward window (decoupled from exit). "
         f"Exit regret on the realized window. DIAGNOSTIC (hindsight), not a signal.\n"]

    for scope, d in [('OOS', r[r['oos']]), ('IS (ref)', r[~r['oos']])]:
        n = len(d)
        picked_wrong = (d['flip_mfe'] > d['chosen_mfe']).mean() * 100
        drift_right = (d['net_drift'] > 0).mean() * 100
        L += [f"## {scope}  (N={n})",
              "### (A) ENTRY / DIRECTION regret",
              f"- picked the WRONG side (flip had larger MFE): **{picked_wrong:.0f}%** "
              f"(50% = coin flip; >50% = anti-predictive)",
              f"- net price drifted our way over {FWD_WINDOW//60}m: {drift_right:.0f}% of entries",
              f"- mean MFE chosen {d['chosen_mfe'].mean():.1f}pt vs flipped {d['flip_mfe'].mean():.1f}pt "
              f"(edge {d['chosen_mfe'].mean()-d['flip_mfe'].mean():+.1f}pt) | mean regret "
              f"(oracle−chosen) {(d['oracle_mfe']-d['chosen_mfe']).mean():.1f}pt"]
        # (B) exit regret on RIGHT-DIRECTION trades (had a real favorable move)
        rd = d[d['mfe_real'] >= 5]    # got at least 5pt our way = right direction at some point
        bins = [0, 30, 60, 120, 300, 1e9]; labels = ['0-30s', '30-60s', '1-2m', '2-5m', '5m+']
        rd = rd.assign(bucket=pd.cut(rd['secs_from_peak'], bins=bins, labels=labels, right=False))
        gb = rd.groupby('bucket', observed=True)['giveback'].agg(['count', 'mean', 'sum'])
        L += ["### (B) EXIT regret — of right-direction trades (MFE≥5pt), giveback by time-from-peak",
              f"- right-direction trades: {len(rd)} ({len(rd)/n*100:.0f}% of {scope}); "
              f"total giveback **{rd['giveback'].sum()*USD_PER_PT:,.0f}$** ({rd['giveback'].sum():,.0f}pt)",
              "| time peak→exit | trades | mean giveback (pt) | total giveback ($) |",
              "|---|---|---|---|"]
        for b, row in gb.iterrows():
            L.append(f"| {b} | {int(row['count'])} | {row['mean']:.1f} | {row['sum']*USD_PER_PT:,.0f} |")
        L.append("")

    # plot: OOS giveback $ by time-from-peak bucket
    oos = r[r['oos']]; rd = oos[oos['mfe_real'] >= 5]
    bins = [0, 30, 60, 120, 300, 1e9]; labels = ['0-30s', '30-60s', '1-2m', '2-5m', '5m+']
    rd = rd.assign(bucket=pd.cut(rd['secs_from_peak'], bins=bins, labels=labels, right=False))
    gb = rd.groupby('bucket', observed=True)['giveback'].sum() * USD_PER_PT
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(gb.index.astype(str), gb.values, color='#c62828')
    ax.set_title('OOS exit giveback $ by time from MFE peak to exit'); ax.set_ylabel('total giveback $')
    fig.tight_layout(); fig.savefig('reports/findings/kalman_regret_giveback.png', dpi=120); plt.close(fig)

    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/kalman_regret_analysis.md', 'w', encoding='utf-8').write(rep)
    print(rep.encode('ascii', 'replace').decode())
    print("\n[plot -> reports/findings/kalman_regret_giveback.png]")


if __name__ == '__main__':
    main()
