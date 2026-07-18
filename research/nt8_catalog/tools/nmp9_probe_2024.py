"""
NMP9 quantile-match PROBE (doc 102, Opus drone). Streams the dossier pipeline and records
EVERY valid RTH 1m boundary on 2024 label days (NO gate applied), with the raw CURRENT
estimators the NMP9 waterfall reads: z21, vr, wick5m, wick15m, h1_z, h1_vel, |vel|, lambda.
This is a READ-ONLY instrument: it reuses the pipeline's own _tf_state / _nmp_lambda so
there is zero drift from the production NMP9 block. Output feeds nmp9_quantile_match.py,
which solves the RETUNE thresholds by marginal pass-rate on 2024 ONLY (no label/PnL).

Run:  python3.11 research/nt8_catalog/tools/nmp9_probe_2024.py
Out:  reports/nmp9_probe_2024.parquet
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dossier_signal_pipeline as P


def gen_probe(ctx):
    """Collect raw per-boundary features on 2024 label days only (no NMP entry gate)."""
    day = ctx.day
    if not day.startswith('2024'):
        return []
    m1 = P._tf_state(ctx, 60); m5 = P._tf_state(ctx, 300)
    m15 = P._tf_state(ctx, 900); h1 = P._tf_state(ctx, 3600)
    if getattr(ctx, 'zse', None) is not None:
        lam = P._nmp_lambda(ctx)
    else:
        lam = np.full(len(ctx.c), np.nan)
    out = []
    rows = np.flatnonzero((ctx.ts % 60 == 0) & ctx.rth &
                          (np.arange(len(ctx.c)) >= ctx.start))

    def at(st, i):
        k = st['row_closed'][i]
        return None if not np.isfinite(k) else int(k)

    def fin(x):
        return float(x) if np.isfinite(x) else np.nan

    for i in rows:
        k1, k5, k15, kh = at(m1, i), at(m5, i), at(m15, i), at(h1, i)
        if None in (k1, k5, k15, kh):
            continue
        z = m1['z'][k1]
        if not np.isfinite(z):
            continue
        out.append(dict(
            day=day, ts=int(ctx.ts[i]), i=int(i),
            z=float(z), vr=fin(m1['vr'][k1]),
            wick5=fin(m5['wick'][k5]), wick15=fin(m15['wick'][k15]),
            h1z=fin(h1['z'][kh]), h1vel=fin(h1['vel'][kh]),
            absvel=fin(abs(m1['vel'][k1])), lam=fin(lam[i])))
    return out


def main():
    P.GENS['NMP9-PROBE'] = gen_probe
    P.NMP_STREAMS.add('NMP9-PROBE')   # so run_all loads ctx.zse for the lambda head
    streams, lblf = P.run_all(['NMP9-PROBE'])
    F = streams['NMP9-PROBE']
    dest = os.path.join(P.REP, 'nmp9_probe_2024.parquet')
    F.to_parquet(dest)
    nd = F['day'].nunique() if len(F) else 0
    print(f'wrote {dest}: {len(F)} boundaries across {nd} 2024 days '
          f'({len(F)/max(nd,1):.1f} boundaries/day)')


if __name__ == '__main__':
    main()
