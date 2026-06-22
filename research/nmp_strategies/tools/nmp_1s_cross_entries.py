"""Prototype: NMP entry fired when the 1s price CROSSES the 1m z-threshold band,
instead of at 1m close. Research-only (no production strategy change).

Causal: the band (rm +/- Z_ENTRY*se) comes from the LAST CLOSED 1m bar; the 1s
prices of the *next* minute are tested against it. Fire on the first 1s outside
the band; re-arm when price returns inside. SHORT above the upper band, LONG below
the lower band (snap-back bet). Entry-only (no exit/PnL yet) -> emits a viz CSV.

Run: python research/nmp_1s_cross_entries.py 2024_02_20
"""
import sys
import numpy as np
import pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import _ols_fit_kernel, N_BASE

Z = 1.8481
ATLAS = 'DATA/ATLAS'
day = sys.argv[1] if len(sys.argv) > 1 else '2024_02_20'


def main():
    m = pd.read_parquet(f'{ATLAS}/1m/{day}.parquet')
    rm, se = _ols_fit_kernel(m['close'].to_numpy(np.float64), N_BASE['1m'])
    mopen = m['timestamp'].to_numpy(np.int64)
    band = {int(o): (rm[i] + Z * se[i], rm[i] - Z * se[i], rm[i], se[i])
            for i, o in enumerate(mopen) if se[i] > 1e-9 and not np.isnan(se[i])}

    s = pd.read_parquet(f'{ATLAS}/1s/{day}.parquet').sort_values('timestamp')
    ts = s['timestamp'].to_numpy(np.int64)
    px = s['close'].to_numpy(np.float64)

    entries, armed = [], True
    for t, p in zip(ts, px):
        b = band.get(int((t // 60) * 60 - 60))   # band from the PRIOR (closed) 1m bar
        if b is None:
            armed = True
            continue
        up, lo, rmv, sev = b
        if armed:
            if p > up:
                entries.append((int(t), 'SHORT', float(p), float((p - rmv) / sev))); armed = False
            elif p < lo:
                entries.append((int(t), 'LONG', float(p), float((p - rmv) / sev))); armed = False
        elif lo <= p <= up:
            armed = True

    base = pd.read_csv('reports/findings/nmp_fade_2024_02_smoke.csv')
    n_base = int((base['day'] == day).sum())
    n_long = sum(1 for e in entries if e[1] == 'LONG')
    print(f"{day}: 1s-cross entries = {len(entries)}  (L {n_long} / S {len(entries)-n_long})")
    print(f"        1m-close entries (baseline) = {n_base}")
    print(f"        ratio = {len(entries)/max(n_base,1):.2f}x")

    df = pd.DataFrame(entries, columns=['entry_ts', 'leg_dir', 'entry_price', 'extra_z_se'])
    df['day'] = day
    out = f'reports/findings/nmp_1s_cross_{day}.csv'
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
