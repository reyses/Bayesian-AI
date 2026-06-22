"""NMP as designed: PURE z-round-trip. Enter at |z|>=Z_ENTRY (1.85sigma), exit at
|z|<=Z_EXIT (0.475sigma) -- both halves z-driven (no R-trigger price-stop).
Compares to the R-trigger baseline (nmp_fade_2024_02_smoke) on the same Feb 2024.
Research-only. Run: python research/nmp_z_roundtrip.py
"""
import os, glob, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Z_ENTRY, Z_EXIT = 1.8481, 0.4752
ATLAS = 'DATA/ATLAS'
FEAT = f'{ATLAS}/FEATURES_5s_v2'
ZCOL = 'L3_1m_z_se_15'
BASE = 'reports/findings/nmp_fade_2024_02_smoke.csv'
RNG = np.random.RandomState(42)


def dayblock_ci(per_day):
    nd = len(per_day)
    boots = [per_day[RNG.randint(0, nd, nd)].mean() for _ in range(4000)]
    return float(per_day.mean()), tuple(np.percentile(boots, [2.5, 97.5]))


def main():
    base = pd.read_csv(BASE)
    upp = (base['pnl_usd'] / base['pnl_pts']).replace([np.inf, -np.inf], np.nan).dropna()
    USD_PER_PT = float(np.median(upp))

    days = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{FEAT}/L3_1m/2024_02_*.parquet'))
    trades, holds = [], []
    for day in days:
        z = pd.read_parquet(f'{FEAT}/L3_1m/{day}.parquet', columns=['timestamp', ZCOL])
        px = pd.read_parquet(f'{ATLAS}/5s/{day}.parquet', columns=['timestamp', 'close'])
        df = z.merge(px, on='timestamp').sort_values('timestamp').reset_index(drop=True)
        zz, cl, ts = df[ZCOL].to_numpy(), df['close'].to_numpy(), df['timestamp'].to_numpy(np.int64)
        pos, ei, edir = 0, None, None
        for i in range(len(df)):
            zi = zz[i]
            if np.isnan(zi):
                continue
            if pos == 0:
                if abs(zi) >= Z_ENTRY:
                    pos, ei, edir = 1, i, ('SHORT' if zi > 0 else 'LONG')
            elif abs(zi) <= Z_EXIT:
                pts = (cl[i] - cl[ei]) if edir == 'LONG' else (cl[ei] - cl[i])
                trades.append((day, int(ts[ei]), edir, float(cl[ei]), int(ts[i]), float(cl[i]),
                               float(pts), float(pts * USD_PER_PT), float(zz[ei]), 'z_exit'))
                holds.append((ts[i] - ts[ei]) / 60.0)
                pos = 0
        if pos == 1:
            pts = (cl[-1] - cl[ei]) if edir == 'LONG' else (cl[ei] - cl[-1])
            trades.append((day, int(ts[ei]), edir, float(cl[ei]), int(ts[-1]), float(cl[-1]),
                           float(pts), float(pts * USD_PER_PT), float(zz[ei]), 'eod'))
            holds.append((ts[-1] - ts[ei]) / 60.0)

    tr = pd.DataFrame(trades, columns=['day', 'entry_ts', 'leg_dir', 'entry_price', 'exit_ts',
                                       'exit_price', 'pnl_pts', 'pnl_usd', 'extra_z_se', 'exit_reason'])
    out_csv = 'reports/findings/nmp_z_roundtrip_2024_02.csv'
    tr.to_csv(out_csv, index=False)

    p = tr['pnl_usd'].to_numpy()
    win, loss = p[p > 0].sum(), abs(p[p < 0].sum())
    pf = win / loss if loss > 0 else float('inf')
    per_day = tr.groupby('day')['pnl_usd'].sum().reindex(days).fillna(0).to_numpy()
    m, ci = dayblock_ci(per_day)
    eod = (tr['exit_reason'] == 'eod').mean() * 100

    L = []
    L.append("# NMP pure z-round-trip vs R-trigger (Feb 2024)\n")
    L.append(f"USD/point = {USD_PER_PT:.2f} (from baseline)  |  exit at |z|<= {Z_EXIT}, enter |z|>= {Z_ENTRY}\n")
    L.append("| metric | **z-round-trip** | R-trigger baseline |")
    L.append("|---|---|---|")
    L.append(f"| trades | {len(tr)} | 1860 |")
    L.append(f"| trades/day | {len(tr)/len(days):.0f} | 89 |")
    L.append(f"| $/trade mean | {p.mean():+.2f} | -4.75 |")
    L.append(f"| $/trade median | {np.median(p):+.2f} | -5.00 |")
    L.append(f"| PF-based Trade WR | {pf-1:+.3f} (PF {pf:.2f}) | -0.730 (PF 0.27) |")
    L.append(f"| $/day mean | {m:+.0f} | -421 |")
    L.append(f"| $/day 95% day-block CI | [{ci[0]:+.0f}, {ci[1]:+.0f}] | [-469, -378] |")
    L.append(f"| winning days | {(per_day>0).sum()}/{len(days)} | 0/21 |")
    L.append(f"| avg hold (min) | {np.mean(holds):.1f} | ~short (R-stop) |")
    L.append(f"| EOD-timeout exits | {eod:.0f}% | n/a |")
    L.append(f"\nVerdict: {'BEATS' if m > -421 else 'WORSE THAN'} the R-trigger baseline on $/day "
             f"({m:+.0f} vs -421). Still {'PROFITABLE' if ci[0] > 0 else 'NEGATIVE' if ci[1] < 0 else 'INCONCLUSIVE'} "
             f"(CI {'excludes' if (ci[0]>0 or ci[1]<0) else 'includes'} 0).")
    L.append("CAVEAT: z reverts partly because the mean slides to price (trend) -> 'reverted in z, lost in price'.")
    rep = '\n'.join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/nmp_z_roundtrip_2024_02.md', 'w').write(rep)
    print(rep)
    print(f"\n[trades -> {out_csv}]")


if __name__ == '__main__':
    main()
