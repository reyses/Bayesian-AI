#!/usr/bin/env python3
"""PATH vs SNAPSHOT for leg-direction anticipation (owner 2026-07-28 via TG:
"use the mamba to improve it — there's got to be something"; revive the cubic
orange-line). My earlier null used a SNAPSHOT (one early frame); turns live in
PATHS. This tests whether a SEQUENCE model (GRU = mamba-proxy) over the run-up
path — including the cubic slope+curvature TRAJECTORY (orange-line: curvature
flip leads the turn ~20s) — beats the 62% snapshot GBM and the gov_dir baseline.

Per frame in the run-up [pivot..fire]: cubic(slope,curv,dev) + z_se band(7) +
combiner(P,P_any,gov_dir,n_fires,signed-stream-sum). Target = leg_dir_true.
GRU over the sequence, temporal split, OOS acc/AUC. Also a hand-crafted
curvature-flip descriptive check. GPU.  reports/path_sequence_probe.md
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import cubic_regression as cub                        # noqa: E402
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
A1 = os.path.join(REPO, 'DATA', 'ATLAS', '1m'); A5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
ZF = os.path.join(REPO, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m')
REGC = [f'L3_1m_{a}_15' for a in ['z_se', 'z_high', 'z_low', 'SE_high', 'SE_low', 'hurst', 'reversion_prob']]
MAXLEN, LAST = 10, 100
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRU(nn.Module):
    def __init__(self, d, h=48):
        super().__init__()
        self.g = nn.GRU(d, h, batch_first=True)
        self.f = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, lens):
        p = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.g(p)
        return self.f(hn[-1]).squeeze(-1)


def build(end_lead=0):
    files = sorted(glob.glob(os.path.join(VEC, '*.parquet')))[-LAST:]
    fcols = [c for c in pd.read_parquet(files[0]).columns if c.startswith('f_')]
    seqs, snaps, ys, govs, days, curvflip, pxsign = [], [], [], [], [], [], []
    for f in files:
        day = os.path.basename(f)[:10]
        p1 = os.path.join(A1, f'{day}.parquet'); p5 = os.path.join(A5, f'{day}.parquet'); zp = os.path.join(ZF, f'{day}.parquet')
        if not all(os.path.exists(x) for x in (p1, p5, zp)):
            continue
        v = pd.read_parquet(f).sort_values('bar_ts').reset_index(drop=True)
        cl = dict(zip(pd.read_parquet(p1)['timestamp'].astype('int64'), pd.read_parquet(p1)['close'].astype(float)))
        d5 = pd.read_parquet(p5, columns=['timestamp', 'close']).sort_values('timestamp')
        t5 = d5['timestamp'].astype('int64').to_numpy()
        cval, cslp, ccur = cub.rolling(d5['close'].astype(float).to_numpy(), 90, 5)
        z = pd.read_parquet(zp); zt = z['timestamp'].astype('int64').to_numpy()
        zmap = {c: dict(zip(zt, z[c].astype(float))) for c in REGC if c in z}
        bts = v['bar_ts'].astype('int64').to_numpy(); ent = v['entry'].to_numpy()
        gd = v['gov_dir'].to_numpy(); age = v['zz_pivot_age_min'].to_numpy(); zc = v['zz_confirm'].to_numpy()
        fmat = v[fcols].to_numpy(); Pk = v['P_topk'].to_numpy(); Pa = v['P_any'].to_numpy(); nf = v['n_fires_topk'].to_numpy()

        def feat(t):
            ts = int(bts[t]); px = cl.get(ts)
            if px is None:
                return None
            j = int(np.searchsorted(t5, ts, side='right')) - 1
            cv, cs, cc = (cval[j], cslp[j], ccur[j]) if j >= 0 else (np.nan, 0., 0.)
            row = [cs, cc, (px - cv) if np.isfinite(cv) else 0.,
                   Pk[t], Pa[t], gd[t], nf[t], float(np.sign(fmat[t]).sum())]
            row += [zmap.get(c, {}).get(ts, 0.) for c in REGC]
            return row, (cc if np.isfinite(cc) else 0.)
        for e in range(len(bts)):
            if ent[e] != 1 or np.isnan(age[e]):
                continue
            n = int(age[e]); piv = e - n
            if piv < 0 or int(bts[piv]) not in cl:
                continue
            p0 = cl[int(bts[piv])]
            end = len(bts) - 1
            for j in range(e + 1, len(bts)):
                if int(zc[j]) == -int(gd[e]):
                    end = j; break
            seg = [cl.get(int(bts[t])) for t in range(piv, end + 1)]; seg = [x for x in seg if x is not None]
            if len(seg) < 3:
                continue
            leg = 1 if (max(seg) - p0) >= (p0 - min(seg)) else -1
            last = e - end_lead                       # sequence ENDS here (anticipation cutoff)
            t0 = max(piv, last - MAXLEN + 1)
            if last <= t0 or int(bts[last]) not in cl:
                continue
            seq = []; curvs = []
            for t in range(t0, last + 1):
                r = feat(t)
                if r:
                    seq.append(r[0]); curvs.append(r[1])
            if len(seq) < 2:
                continue
            seqs.append(np.array(seq, np.float32)); snaps.append(seq[0])
            ys.append(1 if leg > 0 else 0); govs.append(int(gd[t0])); days.append(day)
            curvflip.append(int(len(curvs) >= 2 and np.sign(curvs[0]) != np.sign(curvs[-1])))
            pxsign.append(1 if (cl[int(bts[last])] - p0) > 0 else 0)   # leak baseline
    return (seqs, np.array(snaps, np.float32), np.array(ys), np.array(govs),
            days, np.array(curvflip), np.array(pxsign))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--end-lead', type=int, default=0,
                    help='drop the last N bars before the fire (anticipation cutoff)')
    A = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    seqs, snaps, ys, govs, days, curvflip, pxsign = build(A.end_lead)
    seqs = [np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) for s in seqs]  # cubic warmup NaNs
    print(f'{len(seqs)} episodes, {len(set(days))} days, dim={seqs[0].shape[1]}, '
          f'end_lead={A.end_lead}', flush=True)
    ud = sorted(set(days)); cut = ud[int(len(ud) * 0.7)]
    tr = np.array([d < cut for d in days]); te = ~tr
    d = seqs[0].shape[1]
    mu = np.concatenate([s for i, s in enumerate(seqs) if tr[i]]).mean(0)
    sd = np.concatenate([s for i, s in enumerate(seqs) if tr[i]]).std(0) + 1e-6

    def pad(idx):
        L = [min(len(seqs[i]), MAXLEN) for i in idx]
        X = np.zeros((len(idx), MAXLEN, d), np.float32)
        for k, i in enumerate(idx):
            s = ((seqs[i] - mu) / sd)[-MAXLEN:]
            X[k, :len(s)] = s
        return torch.tensor(X), torch.tensor(L)
    tri = np.where(tr)[0]; tei = np.where(te)[0]
    Xtr, Ltr = pad(tri); ytr = torch.tensor(ys[tri], dtype=torch.float32)
    Xte, Lte = pad(tei)
    m = GRU(d).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    for ep in range(40):
        m.train(); perm = torch.randperm(len(tri))
        for b in range(0, len(tri), 256):
            bi = perm[b:b + 256]
            opt.zero_grad(); out = m(Xtr[bi], Ltr[bi]); loss = lossf(out, ytr[bi]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pte = torch.sigmoid(m(Xte.to(DEV), Lte)).cpu().numpy()
    yte = ys[tei]
    gru_acc = ((pte > 0.5).astype(int) == yte).mean(); gru_auc = roc_auc_score(yte, pte)
    gov_acc = (((govs[tei] > 0).astype(int)) == yte).mean()
    px_acc = (pxsign[tei] == yte).mean()          # LEAK baseline: already-realized move
    # snapshot logistic-ish baseline via the same GRU-less linear? use gov as ref; report cubic-curv-flip corr
    cf = curvflip[tei]
    flip_dir_acc = ((pte > 0.5).astype(int)[cf == 1] == yte[cf == 1]).mean() if (cf == 1).any() else float('nan')
    out = (f"# Path vs snapshot — leg-direction anticipation (GRU mamba-proxy)\n"
           f"{len(seqs)} episodes, {len(set(days))} days, seq feat dim {d}. Temporal 70/30 split.\n\n"
           f"- **GRU over the run-up path: acc {gru_acc:.1%}, AUC {gru_auc:.3f}**\n"
           f"- price-sign LEAK baseline sign(px-pivot)@cutoff: {px_acc:.1%}\n"
           f"- gov_dir baseline (same test): {gov_acc:.1%}\n"
           f"- snapshot GBM (earlier, all ingredients): 62.4%, AUC 0.658\n"
           f"- curvature-flip episodes in test: {int((cf==1).sum())}/{len(cf)} "
           f"({(cf==1).mean():.0%}); GRU acc on them {flip_dir_acc:.1%}\n\n"
           f"LEAK CHECK: if the GRU ≈ the price-sign baseline, the 'signal' is just the "
           f"already-realized move (not anticipation). Real anticipation = GRU >> price-sign.\n"
           f"Verdict: {'PATH beats the leak baseline -> real sequence anticipation' if gru_acc > px_acc + 0.03 else 'GRU ~= price-sign LEAK -> mostly reading the realized move, NOT anticipation'}\n")
    open(os.path.join(REPO, 'research', 'edge_probe', 'reports', 'path_sequence_probe.md'), 'w').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
