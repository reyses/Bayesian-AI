#!/usr/bin/env python3
"""ANTICIPATE-THE-COMBINER packets (owner 2026-07-27, TG): qwen is given the FULL
combiner (22 streams + P + gov_dir + n_fires) + the curve-regression band + price
context, in the RUN-UP to each combiner fire, and must ANTICIPATE the leg
DIRECTION (long/short) before/as the combiner commits. Judged on direction only.

One episode per combiner fire. Frames = the bars from the pivot up to the fire
(the buildup), so qwen calls direction increasingly early. Label = the actual
leg direction (dominant move from the pivot to the next R-trigger reversal).

Inputs per frame (all three families the owner insisted on):
  combiner   : P_topk P_any gov_dir n_fires + the nonzero ±1 streams by name
  curve-reg  : z_se z_high z_low SE_high SE_low hurst reversion_prob  (L3_1m_*_15)
  price      : px vs pivot, bars since pivot
Out: research/dojo_forge/reports/anticipate/packets/<eid>.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import cubic_regression as cub

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
A1 = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
A5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
ZF = os.path.join(REPO, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m')
OUTD = os.path.join(HERE, '..', 'reports', 'anticipate', 'packets')
REG = ['z_se', 'z_high', 'z_low', 'SE_high', 'SE_low', 'hurst', 'reversion_prob']
CUBIC_WIN, CUBIC_BAR_S = 90, 5    # 7.5-min cubic on 5s bars (NT8 orange line)
MAXBACK = 10                      # frames of run-up before the fire


def frame_text(k, P, Pa, gd, nf, streams, reg, cubic, px, dpx, since):
    s = ' '.join(f'{nm}{"+" if sv > 0 else "-"}' for nm, sv in streams) or '(none)'
    rg = ' '.join(f'{a}={reg[a]:+.2f}' for a in REG if reg.get(a) is not None)
    gs = {1: '+1(long)', -1: '-1(short)', 0: '0(none)'}.get(int(gd), str(gd))
    cval, cslp, ccur = cubic
    cub_line = ('  curve-reg(cubic 7.5m): dev={:+.1f}pts slope={:+.2f}pts/min '
                'curv={:+.3f}'.format(px - cval, cslp, ccur)
                if np.isfinite(cval) else '  curve-reg(cubic 7.5m): (warming)')
    return (f'[fire in {k}m]  combiner forming — ANTICIPATE the leg direction\n'
            f'  combiner: P_topk={P:.3f} P_any={Pa:.3f} gov_dir={gs} n_fires={int(nf)}\n'
            f'  streams(dir): {s}\n'
            f'{cub_line}\n'
            f'  reg-band(z15): {rg}\n'
            f'  price: px vs pivot {dpx:+.1f}pts | bars since pivot {since}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last', type=int, default=None)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    args = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    files = sorted(glob.glob(os.path.join(VEC, '*.parquet')))
    names = [os.path.basename(f)[:10] for f in files]
    lo, hi = 0, len(files)
    if args.start:
        lo = next((i for i, n in enumerate(names) if n >= args.start), lo)
    if args.end:
        hi = next((i for i, n in enumerate(names) if n > args.end), hi)
    if args.last:
        lo = max(lo, hi - args.last)
    fcols = [c for c in pd.read_parquet(files[0]).columns if c.startswith('f_')]
    nmap = [(c, c[2:]) for c in fcols]
    made = 0
    for f in files[lo:hi]:
        day = os.path.basename(f)[:10]
        v = pd.read_parquet(f).sort_values('bar_ts').reset_index(drop=True)
        p1 = os.path.join(A1, f'{day}.parquet'); zp = os.path.join(ZF, f'{day}.parquet')
        if not (os.path.exists(p1) and os.path.exists(zp)):
            continue
        m = pd.read_parquet(p1, columns=['timestamp', 'close'])
        cl = dict(zip(m['timestamp'].astype('int64'), m['close'].astype(float)))
        # cubic 7.5m endpoint on 5s closes -> map to each 1m bar (nearest 5s <= ts)
        cmap = {}
        p5 = os.path.join(A5, f'{day}.parquet')
        if os.path.exists(p5):
            d5 = pd.read_parquet(p5, columns=['timestamp', 'close']).sort_values('timestamp')
            t5 = d5['timestamp'].astype('int64').to_numpy()
            cval, cslp, ccur = cub.rolling(d5['close'].astype(float).to_numpy(),
                                           CUBIC_WIN, CUBIC_BAR_S)
        else:
            t5 = np.array([], dtype='int64')
        z = pd.read_parquet(zp)
        zt = z['timestamp'].astype('int64').to_numpy()
        zmap = {}
        for a in REG:
            col = f'L3_1m_{a}_15'
            if col in z:
                zmap[a] = dict(zip(zt, z[col].astype(float)))
        bts = v['bar_ts'].astype('int64').to_numpy()
        ent = v['entry'].to_numpy(); gdir = v['gov_dir'].to_numpy()
        Pk = v['P_topk'].to_numpy(); Pa = v['P_any'].to_numpy()
        nf = v['n_fires_topk'].to_numpy(); zc = v['zz_confirm'].to_numpy()
        age = v['zz_pivot_age_min'].to_numpy()
        fmat = v[fcols].to_numpy()
        for e in range(len(bts)):
            if ent[e] != 1 or np.isnan(age[e]):
                continue
            d = int(gdir[e]); n = int(age[e]); piv = e - n
            if piv < 0 or int(bts[piv]) not in cl:
                continue
            p0 = cl[int(bts[piv])]
            # leg direction label = dominant move pivot -> next reversal
            end = len(bts) - 1
            for j in range(e + 1, len(bts)):
                if int(zc[j]) == -d:
                    end = j; break
            seg = [cl.get(int(bts[t])) for t in range(piv, end + 1)]
            seg = [x for x in seg if x is not None]
            if len(seg) < 3:
                continue
            up = max(seg) - p0; dn = p0 - min(seg)
            leg_dir = 1 if up >= dn else -1
            # frames: run-up from max(pivot, e-MAXBACK) to the fire bar e
            frames = []
            for t in range(max(piv, e - MAXBACK), e + 1):
                ts = int(bts[t]); px = cl.get(ts)
                if px is None:
                    continue
                streams = [(nm, fmat[t, i]) for i, (_, nm) in enumerate(nmap) if fmat[t, i] != 0]
                reg = {a: (zmap[a].get(ts) if a in zmap else None) for a in REG}
                # cubic at the last 5s bar <= this 1m ts
                cubic = (np.nan, np.nan, np.nan)
                if len(t5):
                    j = int(np.searchsorted(t5, ts, side='right')) - 1
                    if j >= 0:
                        cubic = (cval[j], cslp[j], ccur[j])
                frames.append(dict(frame=len(frames), k=e - t,
                                   text=frame_text(e - t, Pk[t], Pa[t], gdir[t], nf[t],
                                                   streams, reg, cubic, px, px - p0, t - piv)))
            if len(frames) < 2:
                continue
            eid = f'{day}_{ts}_{"L" if d > 0 else "S"}'
            json.dump(dict(episode_id=eid,
                           meta=dict(fire_gov_dir=d, fire_P=round(float(Pk[e]), 4),
                                     pivot_age_min=n, leg_dir_true=leg_dir,
                                     label='anticipate leg direction (long=+1/short=-1)',
                                     n_frames=len(frames)),
                           frames=frames),
                      open(os.path.join(OUTD, f'{eid}.json'), 'w'), indent=0)
            made += 1
    print(f'built {made} anticipation packets -> {OUTD}')


if __name__ == '__main__':
    main()
