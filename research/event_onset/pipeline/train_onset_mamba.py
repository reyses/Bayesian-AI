"""ONSET MAMBA — training (spec §11 step 2). USER-RUN (project rule: the
assistant does not launch training).

  python research/event_onset/pipeline/train_onset_mamba.py --epochs 3

Trains on the NATURAL distribution (every sampleable RTH second, class
imbalance handled by pos_weight) and validates on the MATCHED design, which
is the only honest yardstick — quiet-stretch negatives inflated the probe to
AUC 0.9965 by letting a model answer "is the tape active?".

Split is TEMPORAL (spec §5): train 2024, val 2025-H1. The test window stays
sealed until eval_onset.py is run once, after an audit.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SEQ = os.path.join(REPO, 'research', 'event_onset', 'seq')
CKPT = os.path.join(REPO, 'research', 'event_onset', 'checkpoints')
HEADS = ('fakeout_poke', 'leg_descent', 'ultra_chop')
HORIZONS = (5, 10, 30)
WINDOW, NFEAT, NOUT = 300, 8, 9
PRIMARY = 1                       # index of H=10s inside HORIZONS


def day_split(day):
    y = day[:4]
    return 'train' if y == '2024' else ('val' if day <= '2025_06_30'
                                        else 'test')


class DayCache:
    """npz per day, loaded once, windows sliced as views."""

    def __init__(self, days):
        self.d = {}
        for day in days:
            z = np.load(os.path.join(SEQ, f'{day}.npz'))
            self.d[day] = (z['f'], z['y'], z['mask'], z['ts'])

    def window(self, day, i):
        f = self.d[day][0][i - WINDOW:i]
        return torch.from_numpy(f.astype(np.float32))


class OnsetDS(Dataset):
    def __init__(self, days, cache, stride=1):
        self.cache, self.idx = cache, []
        for day in days:
            m = cache.d[day][2]
            pos = np.flatnonzero(m)[::stride]
            self.idx += [(day, int(i)) for i in pos]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        day, i = self.idx[k]
        y = self.cache.d[day][1][i].astype(np.float32)
        return self.cache.window(day, i), torch.from_numpy(y)


class OnsetMamba(nn.Module):
    def __init__(self, d_model=128, n_layer=4, d_state=16):
        super().__init__()
        from mamba_ssm import Mamba
        self.inp = nn.Linear(NFEAT, d_model)
        self.blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=d_state)
                                     for _ in range(n_layer)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model)
                                    for _ in range(n_layer)])
        self.head = nn.Linear(d_model, NOUT)

    def forward(self, x):
        h = self.inp(x)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))          # pre-norm residual
        return self.head(h[:, -1])       # decision at the LAST second only


def matched_eval(model, dev, event='fakeout_poke', hz=10, split='val'):
    """AUC on the matched design — same rows the GBM baseline was scored on."""
    import pandas as pd
    p = os.path.join(REPO, 'research', 'event_onset',
                     f'matched_{event}_{hz}s.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['day', 'ts', 'y'])
    df = df[df['day'].map(day_split) == split]
    days = sorted(set(df['day']) & {os.path.basename(f)[:-4]
                                    for f in glob.glob(os.path.join(SEQ, '*.npz'))})
    if not days:
        return None
    cache = DayCache(days)
    ki = HEADS.index(event) * 3 + HORIZONS.index(hz)
    xs, ys = [], []
    for day in days:
        _, _, mask, ts = cache.d[day]
        pos = {int(t): i for i, t in enumerate(ts)}
        sub = df[df['day'] == day]
        for t, y in zip(sub['ts'], sub['y']):
            i = pos.get(int(t))
            if i is None or i < WINDOW or not mask[i]:
                continue
            xs.append(cache.window(day, i))
            ys.append(y)
    if len(ys) < 200:
        return None
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(xs), 512):
            xb = torch.stack(xs[b:b + 512]).to(dev)
            out.append(torch.sigmoid(model(xb))[:, ki].cpu().numpy())
    return roc_auc_score(np.array(ys), np.concatenate(out)), len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--stride', type=int, default=5,
                    help='sample every Nth second (adjacent windows overlap '
                         '299/300 — stride 1 is 300x redundant compute)')
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    days = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(os.path.join(SEQ, '*.npz')))
    tr = [d for d in days if day_split(d) == 'train']
    va = [d for d in days if day_split(d) == 'val']
    print(f'train {len(tr)} days | val {len(va)} days | TEST SEALED')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = DayCache(tr)
    ds = OnsetDS(tr, cache, stride=a.stride)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4,
                    drop_last=True)
    print(f'{len(ds):,} training windows (stride {a.stride})')
    # pos_weight from the training labels themselves
    ycat = np.concatenate([cache.d[d][1][cache.d[d][2]] for d in tr])
    rate = ycat.mean(0).clip(1e-4, 1 - 1e-4)
    pw = torch.tensor((1 - rate) / rate, dtype=torch.float32, device=dev)
    print('label rates:', np.round(rate, 4))
    model = OnsetMamba().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in tqdm(dl, desc=f'epoch {ep}'):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        torch.save(model.state_dict(), os.path.join(CKPT, f'onset_ep{ep}.pt'))
        line = f'epoch {ep} loss {tot/max(len(dl),1):.4f}'
        for ev in HEADS:
            r = matched_eval(model, dev, ev, 10, 'val')
            if r:
                line += f' | {ev} matched-AUC {r[0]:.4f} (n={r[1]})'
        print(line, flush=True)
    print('\nBASELINE TO BEAT (GBM, matched design): fakeout 0.769 / '
          'leg_descent 0.868 / ultra_chop 0.830')
    print('Spec §7: +0.02 on 2 of 3 -> ship | within +-0.02 -> KEEP THE GBM '
          '| -0.02 -> kill')


if __name__ == '__main__':
    main()
