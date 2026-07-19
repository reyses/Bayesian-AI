"""
GOLDEN VECTORS for the OOS BACKTEST WINDOW (P3 first pass).

Reuses the frozen reference decider in golden_vector_gen.py VERBATIM (combiner fit,
generators, TMPL0 tie-rule, R-trigger zigzag, per-1m-bar aggregation) but retargets:
  * 5s substrate      -> DATA/ATLAS_NT8/5s   (the OOS tape live trades)  [vs ATLAS]
  * z_se head source  -> DATA/ATLAS_NT8/FEATURES_5s_v2/L3_1m (auto-detect z_se col)
  * day selection     -> an explicit window (default 2026-06-22..2026-07-17)
  * output            -> research/nt8_port/golden_backtest/  (NOT golden/, frozen)

CAVEAT (documented): the standard SFE build wrote L3_1m_z_se_30 (N_BASE['1m']=30 in
current code) whereas the frozen reference / C# port consume z_se_15. NMP/NMP9 head
streams (6 of 22 top-K) therefore fire off an N=30 state here -> flagged in the report.

Usage:
  python3.11 research/nt8_port/tools/golden_backtest_gen.py \
      --start 2026_06_22 --end 2026_07_17
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, HERE)

import golden_vector_gen as gvg
import dossier_signal_pipeline as dsp

NT8_5S = os.path.join(ROOT, 'DATA', 'ATLAS_NT8', '5s')
NT8_ZDIR = os.path.join(ROOT, 'DATA', 'ATLAS_NT8', 'FEATURES_5s_v2', 'L3_1m')
OUT = os.path.join(PROJ, 'golden_backtest')
os.makedirs(OUT, exist_ok=True)


def _zse_col(day):
    """Return (path, colname) for the day's z_se store, or (None, None)."""
    import pyarrow.parquet as pq
    p = os.path.join(NT8_ZDIR, f'{day}.parquet')
    if not os.path.exists(p):
        return None, None
    cols = pq.ParquetFile(p).schema.names
    zc = [c for c in cols if c.startswith('L3_1m_z_se_')]
    return (p, zc[0]) if zc else (None, None)


def build_ctx_nt8(files, j):
    """gvg.build_ctx clone with the z_se source retargeted to the NT8 store."""
    prior_daily = []
    tail = None
    for pf in files[max(0, j - 20):j]:
        df = pd.read_parquet(pf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        m = ((dt.dt.time >= dsp.RTH0) & (dt.dt.time <= dsp.RTH1)).values
        if m.any():
            entry = dict(high=float(df['high'].values[m].max()),
                         low=float(df['low'].values[m].min()),
                         close=float(df['close'].values[m][-1]))
            entry.update(dsp._day_profile(df['close'].values[m], df['volume'].values[m]))
            prior_daily.append(entry)
            prior_daily = prior_daily[-20:]
        tail = df.tail(dsp.TAIL)

    p = files[j]
    day = os.path.basename(p)[:10]
    df = pd.read_parquet(p, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
    start = len(tail) if tail is not None else 0
    ctx = dsp.DayCtx(full, start, day, prior_daily)

    zp, zc = _zse_col(day)
    ctx.zse = None
    ctx.zse_col = zc
    if zp is not None:
        zf = pd.read_parquet(zp, columns=['timestamp', zc])
        ctx.zse = pd.Series(full['timestamp']).map(
            dict(zip(zf['timestamp'].values, zf[zc].values))).values
    return ctx, day


def process_day_nt8(files, j, model):
    ctx, day = build_ctx_nt8(files, j)
    F = gvg.all_fires(ctx)
    F = gvg.score_fires(F, model)
    zz = gvg.zigzag_rtrigger(ctx)
    G = gvg.aggregate_day(ctx, day, F, zz, model)
    return day, F, zz, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026_06_22')
    ap.add_argument('--end', default='2026_07_17')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(NT8_5S, '*.parquet')))
    names = [os.path.basename(f)[:10] for f in files]
    win_idx = [i for i, nm in enumerate(names) if args.start <= nm <= args.end]
    print(f'{len(files)} NT8 5s files; window {args.start}..{args.end} -> {len(win_idx)} days present')

    model = gvg.fit_combiner()
    print(f"combiner K={model['K']} thr P>={model['thr']:.4f}  topk={model['topk']}")

    rows = []
    for j in win_idx:
        day, F, zz, G = process_day_nt8(files, j, model)
        G.to_parquet(os.path.join(OUT, f'{day}.parquet'))
        n_fires = int(len(F))
        n_topk = int(F['det'].isin(model['topk']).sum()) if len(F) else 0
        n_entries = int(G['entry'].sum())
        n_conf = int((G['zz_confirm'] != 0).sum())
        rows.append(dict(day=day, bars=len(G), fires=n_fires, topk=n_topk,
                         entries=n_entries, zz_confirms=n_conf,
                         R=int(zz['min_rev_ticks'])))
        print(f"{day}: bars={len(G)} fires={n_fires} topk={n_topk} "
              f"entries={n_entries} zz_confirms={n_conf} R={zz['min_rev_ticks']}t")

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, '_window_summary.csv'), index=False)
    print(f"\nwrote {len(rows)} golden parquets + _window_summary.csv to {OUT}")


if __name__ == '__main__':
    main()
