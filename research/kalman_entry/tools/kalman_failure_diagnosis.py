"""Where did the GA-Kalman fail OOS? Concrete failure anatomy on the 5,463 trades.
Decides the lever: is OOS break-even driven by (a) entry over-firing / chop churn,
(b) winners giving back via the 79pt trail, or (c) big stop losers? No new model.

Input:  research/kalman_tuning_eda/reports/findings/kalman_full_trades.csv
Output: reports/findings/kalman_failure_diagnosis.md
"""
import os
import numpy as np
import pandas as pd

CSV = 'research/kalman_tuning_eda/reports/findings/kalman_full_trades.csv'
USD_PER_PT = 2.0
NY_OFFSET_H = -5  # unix->NY (approx; DST ignored, bucket-level only)
RNG = np.random.RandomState(42)


def ci(x):
    if len(x) < 3:
        return (np.nan, np.nan)
    b = [x[RNG.randint(0, len(x), len(x))].mean() for _ in range(4000)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def main():
    df = pd.read_csv(CSV)
    df['oos'] = df['split'].isin(['OOS_H2_24', 'OOS_25_26'])
    df['mfe_usd'] = df['mfe_pts'] * USD_PER_PT
    df['hour'] = ((df['entry_ts'] // 3600 + NY_OFFSET_H) % 24).astype(int)
    L = ["# GA-Kalman OOS failure anatomy (where did it bleed?)\n"]

    for scope, d in [('OOS (H2-24 + 25-26)', df[df['oos']]), ('IS (H1-24, reference)', df[~df['oos']])]:
        net = d['net_usd'].to_numpy()
        nd = d['day'].nunique()
        wins = d[d['net_usd'] > 0]; loss = d[d['net_usd'] <= 0]
        # loss anatomy: big stops vs scratch churn
        stop_thr = -90.0  # ~the trail/stop band ($); below = a real stop-out
        big_stops = loss[loss['net_usd'] <= stop_thr]
        scratch = loss[loss['net_usd'] > stop_thr]
        # exit giveback on winners ($ left on table vs MFE peak)
        giveback = (wins['mfe_usd'] - wins['net_usd']).sum()
        # chop entries: trades whose MFE never developed (peak < 10pt) -> entry never had a move
        chop = d[d['mfe_pts'] < 10]
        L += [f"## {scope}  (N={len(d)} trades, {nd} days)",
              f"- net total ${net.sum():,.0f}  | $/day {net.sum()/nd:+.1f}  | $/trade {net.mean():+.2f}",
              f"- winners {len(wins)} (sum ${wins['net_usd'].sum():,.0f}) | losers {len(loss)} (sum ${loss['net_usd'].sum():,.0f})",
              f"- **big stop-outs** (≤${stop_thr:.0f}): {len(big_stops)} trades, sum **${big_stops['net_usd'].sum():,.0f}** "
              f"({len(big_stops)/len(d)*100:.0f}% of trades, {big_stops['net_usd'].sum()/net.sum()*100 if net.sum() else 0:.0f}% of net)",
              f"- **scratch/chop losers** (>{stop_thr:.0f} & ≤0): {len(scratch)} trades, sum ${scratch['net_usd'].sum():,.0f}",
              f"- **exit giveback on winners**: ${giveback:,.0f} left on table vs MFE peak "
              f"(winners kept {wins['net_usd'].sum()/ (wins['mfe_usd'].sum()) *100:.0f}% of peak)",
              f"- **chop entries** (MFE<10pt, never developed): {len(chop)} = {len(chop)/len(d)*100:.0f}% of trades, "
              f"net ${chop['net_usd'].sum():,.0f}",
              f"- direction: LONG net ${d[d['dir']=='LONG']['net_usd'].sum():,.0f} | SHORT net ${d[d['dir']=='SHORT']['net_usd'].sum():,.0f}",
              ""]
        # time-of-day PnL (OOS only, the actionable one)
        if 'OOS' in scope:
            tod = d.groupby('hour')['net_usd'].agg(['sum', 'count'])
            worst = tod.sort_values('sum').head(4)
            L.append("- worst entry-hours (NY approx): " +
                      ", ".join(f"{h:02d}h ${r['sum']:,.0f}({int(r['count'])}t)" for h, r in worst.iterrows()))
            # losing-day concentration
            pday = d.groupby('day')['net_usd'].sum().sort_values()
            L.append(f"- losing-day concentration: worst 10 days sum ${pday.head(10).sum():,.0f} "
                     f"of ${pday[pday<0].sum():,.0f} total loss ({pday.head(10).sum()/pday[pday<0].sum()*100:.0f}%)")
            L.append("")

    # verdict logic
    oos = df[df['oos']]
    chop_share = (oos['mfe_pts'] < 10).mean()
    giveback = (oos[oos.net_usd>0]['mfe_usd'] - oos[oos.net_usd>0]['net_usd']).sum()
    stop_sum = oos[oos.net_usd <= -90]['net_usd'].sum()
    L += ["## Read",
          f"- chop entries (no move) = {chop_share*100:.0f}% of OOS trades → ENTRY-side leak if high.",
          f"- exit giveback = ${giveback:,.0f} → EXIT-side opportunity (risk-reduction, not new edge).",
          f"- big stop-outs cost ${stop_sum:,.0f} → STOP-side if dominant.",
          "Pick the lever from the LARGEST concrete leak above — pre-commit + OOS + CI before building."]
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/kalman_failure_diagnosis.md', 'w', encoding='utf-8').write(rep)
    print(rep.encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
