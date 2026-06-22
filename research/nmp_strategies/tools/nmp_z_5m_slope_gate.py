"""Measure the 5m regression-mean SLOPE (velocity) and dSLOPE (speed-of-change),
join to the z-round-trip entries, and stratify outcomes by the 5m regime.

Idea: trigger on the 1m z (enter |z|>=Z_ENTRY, exit |z|<=Z_EXIT), but read the 5m
mean's slope as context. A snap-back LONG (price<mean) is WITH-trend if 5m slope>0,
COUNTER-trend if 5m slope<0 (longing into a falling 5m = falling knife). Mirror for
SHORT. Hypothesis: counter-trend fades are the big losers dragging the mean below
the (positive) median; flat/with-trend are the keepers.

Research-only, Feb 2024 (IS). Run: python research/nmp_z_5m_slope_gate.py
"""
import os, glob, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import _ols_slope_kernel, N_BASE

ATLAS = 'DATA/ATLAS'
TR = 'reports/findings/nmp_z_roundtrip_2024_02.csv'
RNG = np.random.RandomState(42)


def ci(per_day):
    if len(per_day) < 3:
        return np.nan, (np.nan, np.nan)
    b = [per_day[RNG.randint(0, len(per_day), len(per_day))].mean() for _ in range(4000)]
    return float(per_day.mean()), tuple(np.percentile(b, [2.5, 97.5]))


def report(name, sub, days):
    p = sub['pnl_usd'].to_numpy()
    if len(p) == 0:
        return f"| {name} | 0 | - | - | - | - |"
    win, loss = p[p > 0].sum(), abs(p[p < 0].sum())
    pf = win / loss if loss > 0 else np.inf
    pday = sub.groupby('day')['pnl_usd'].sum().reindex(days).fillna(0).to_numpy()
    m, c = ci(pday)
    return (f"| {name} | {len(p)} | {p.mean():+.2f} | {np.median(p):+.2f} | "
            f"{pf:.2f} | {m:+.0f} [{c[0]:+.0f},{c[1]:+.0f}] |")


def main():
    tr = pd.read_csv(TR)
    days = sorted(tr['day'].unique())

    # --- measure 5m mean slope + dSlope per day, map by 5m bar open ---
    slope_map = {}
    for day in days:
        m = pd.read_parquet(f'{ATLAS}/5m/{day}.parquet', columns=['timestamp', 'close'])
        sl, _se, _t = _ols_slope_kernel(m['close'].to_numpy(np.float64), N_BASE['5m'])
        acc = np.full_like(sl, np.nan)
        acc[1:] = sl[1:] - sl[:-1]                      # speed-of-change of the slope
        for o, s, a in zip(m['timestamp'].to_numpy(np.int64), sl, acc):
            slope_map[int(o)] = (s, a)

    def lookup(ts):  # last CLOSED 5m bar at entry
        return slope_map.get((int(ts) // 300) * 300 - 300, (np.nan, np.nan))

    sl = tr['entry_ts'].apply(lambda t: lookup(t)[0])
    ac = tr['entry_ts'].apply(lambda t: lookup(t)[1])
    tr['slope5m'], tr['accel5m'] = sl, ac
    tr = tr.dropna(subset=['slope5m']).copy()
    tr['is_long'] = tr['leg_dir'].str.upper().eq('LONG')
    # with-trend: long & up-slope, or short & down-slope
    tr['with_trend'] = (tr['is_long'] & (tr['slope5m'] > 0)) | (~tr['is_long'] & (tr['slope5m'] < 0))
    # |slope| terciles -> flat / mid / steep
    q = tr['slope5m'].abs().quantile([1/3, 2/3]).to_numpy()
    tr['flat'] = tr['slope5m'].abs() <= q[0]

    rows = ["# 5m-slope regime stratification of the z-round-trip (Feb 2024 IS)\n",
            f"|slope| terciles (pts/5m): flat<= {q[0]:.3f}, steep> {q[1]:.3f}\n",
            "| bucket | n | $/trade mean | median | PF | $/day [95% day-block CI] |",
            "|---|---|---|---|---|---|",
            report("ALL z-round-trip", tr, days),
            report("FLAT 5m (range)", tr[tr['flat']], days),
            report("sloping + WITH-trend", tr[~tr['flat'] & tr['with_trend']], days),
            report("sloping + COUNTER-trend", tr[~tr['flat'] & ~tr['with_trend']], days),
            report(">> KEEP: flat OR with-trend", tr[tr['flat'] | tr['with_trend']], days),
            report(">> DROP: sloping counter-trend", tr[~tr['flat'] & ~tr['with_trend']], days),
            ]
    # accel (speed-of-change) split, secondary
    amed = tr['accel5m'].abs().median()
    rows += ["", "Speed-of-change (|dSlope|) split:",
             "| bucket | n | $/trade mean | median | PF | $/day [95% CI] |",
             "|---|---|---|---|---|---|",
             report("calm (|dSlope|<=med)", tr[tr['accel5m'].abs() <= amed], days),
             report("turning (|dSlope|>med)", tr[tr['accel5m'].abs() > amed], days)]
    rep = "\n".join(rows)
    rep += ("\n\nCAVEATS: IS (Feb only); tercile/bucket selection invites overfit (graveyard: "
            "~25% cell survival IS->OOS) -> treat structural splits (with/counter sign) over "
            "magnitude cells, and OOS-validate before trusting; pre-cost.")
    open('reports/findings/nmp_z_5m_slope_gate_2024_02.md', 'w').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
