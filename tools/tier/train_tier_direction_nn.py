"""
Train a CNN on 91D features at TIER entry → P(win).

Same architecture as train_pivot_direction_nn.py, but trained on the 9-tier
engine's trades (iso_is.pkl). Goal: produce a tier-specific direction
predictor that works on tier entries.

Why a separate model: post-hoc transfer of the RM-pivot NN to tier trades
showed inverted calibration (low P → HIGH actual win%) because the tier
entries occupy a different region of feature space. A model trained
directly on tier trades should fit their regime.

Walk-forward split by day:
  - Train: first 70% of tier-trade days
  - Val:   next 15%
  - Test:  last 15%

Regularization tuned for smaller dataset (~5k trades vs ~21k for pivot):
  - Higher dropout (0.4)
  - Lower LR (5e-4)
  - Early stopping on val AUC

Outputs:
  reports/findings/tier_direction_nn.md
  training_RM_physics/output/tier_direction_cnn.pt

Usage:
  python tools/train_tier_direction_nn.py
  python tools/train_tier_direction_nn.py --trades training_iso/output/trades/iso_is.pkl
  python tools/train_tier_direction_nn.py --epochs 40 --lr 5e-4
"""
import os
import sys
import pickle
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_TRADES_PKL = 'training_iso/output/trades/iso_is.pkl'
DEFAULT_OUT_MD = 'reports/findings/tier_direction_nn.md'
DEFAULT_OUT_MODEL = 'training_RM_physics/output/tier_direction_cnn.pt'

N_CORE_PER_TF = 12
N_HELPER_PER_TF = 3
N_TFS = 6
FEAT_DIM = N_CORE_PER_TF + N_HELPER_PER_TF   # 15
GRID_H = N_TFS                                # 6
GRID_W = FEAT_DIM                             # 15


def feat_to_grid(f91):
    """Reshape 91D → 6 × 15 grid. Drops the 91st element if present.
    Same layout as train_pivot_direction_nn."""
    f = np.asarray(f91[:90], dtype=np.float32)
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    core_block = f[:N_CORE_PER_TF * N_TFS]
    helper_block = f[N_CORE_PER_TF * N_TFS:]
    for tf in range(N_TFS):
        grid[tf, :N_CORE_PER_TF] = core_block[tf * N_CORE_PER_TF:(tf + 1) * N_CORE_PER_TF]
        grid[tf, N_CORE_PER_TF:] = helper_block[tf * N_HELPER_PER_TF:(tf + 1) * N_HELPER_PER_TF]
    return grid


class TierDataset(Dataset):
    def __init__(self, grids, labels, tiers=None):
        self.grids = torch.from_numpy(grids).unsqueeze(1)   # (N, 1, 6, 15)
        self.labels = torch.from_numpy(labels).float()
        self.tiers = tiers   # optional array of strings for reporting

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.grids[i], self.labels[i]


class TierCNN(nn.Module):
    """Slightly smaller + more regularized than PivotCNN to handle the
    smaller tier-trade dataset."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 3 * 7, 48)
        self.fc2 = nn.Linear(48, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trades', default=DEFAULT_TRADES_PKL,
                   help=f'Trades pickle (default {DEFAULT_TRADES_PKL})')
    p.add_argument('--out-model', default=DEFAULT_OUT_MODEL)
    p.add_argument('--out-md', default=DEFAULT_OUT_MD)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=6,
                   help='Early-stop patience on val AUC')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Trades: {args.trades}')

    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades)} trades')

    # Filter valid (has features, nonzero pnl)
    valid = [t for t in trades
             if len(t.get('entry_79d', [])) >= 90 and t['pnl'] != 0]
    print(f'Valid: {len(valid)}')

    # Tier mix, for context
    tier_counts = Counter(t.get('entry_tier', '?') for t in valid)
    print(f'Tier mix (top 10): {dict(tier_counts.most_common(10))}')

    # Walk-forward split by day
    valid.sort(key=lambda t: t['day'])
    days = sorted(set(t['day'] for t in valid))
    n_days = len(days)
    train_days = set(days[:int(n_days * 0.70)])
    val_days = set(days[int(n_days * 0.70):int(n_days * 0.85)])
    test_days = set(days[int(n_days * 0.85):])
    print(f'Days: train {len(train_days)}  val {len(val_days)}  test {len(test_days)}')

    train_data, val_data, test_data = [], [], []
    for t in valid:
        grid = feat_to_grid(t['entry_79d'])
        label = 1.0 if t['pnl'] > 0 else 0.0
        if t['day'] in train_days:
            train_data.append((grid, label, t.get('entry_tier', '?')))
        elif t['day'] in val_days:
            val_data.append((grid, label, t.get('entry_tier', '?')))
        elif t['day'] in test_days:
            test_data.append((grid, label, t.get('entry_tier', '?')))

    def pack(data):
        if not data:
            return (np.empty((0, GRID_H, GRID_W), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    [])
        gs = np.stack([d[0] for d in data])
        ls = np.array([d[1] for d in data], dtype=np.float32)
        ts = [d[2] for d in data]
        return gs, ls, ts

    gs_train, ls_train, tr_tiers = pack(train_data)
    gs_val, ls_val, vl_tiers = pack(val_data)
    gs_test, ls_test, ts_tiers = pack(test_data)
    print(f'Train: {len(ls_train)}  Val: {len(ls_val)}  Test: {len(ls_test)}')
    print(f'Win%: train {ls_train.mean()*100:.1f}  val {ls_val.mean()*100:.1f}  '
          f'test {ls_test.mean()*100:.1f}')

    # Standardize per-feature using train stats only
    mean = gs_train.mean(axis=0, keepdims=True)
    std = gs_train.std(axis=0, keepdims=True) + 1e-6
    gs_train = (gs_train - mean) / std
    gs_val = (gs_val - mean) / std
    gs_test = (gs_test - mean) / std

    train_loader = DataLoader(TierDataset(gs_train, ls_train),
                               batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TierDataset(gs_val, ls_val),
                             batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(TierDataset(gs_test, ls_test),
                              batch_size=1024, shuffle=False, num_workers=0)

    model = TierCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # Validate
        model.eval()
        preds_acc, labels_acc = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                preds_acc.append(logits)
                labels_acc.append(yb.numpy())
        preds = np.concatenate(preds_acc) if preds_acc else np.empty(0)
        labels = np.concatenate(labels_acc) if labels_acc else np.empty(0)
        if len(labels) > 0 and len(set(labels)) > 1:
            auc = roc_auc_score(labels, preds)
            probs = 1 / (1 + np.exp(-preds))
            acc = accuracy_score(labels, (probs > 0.5).astype(int))
        else:
            auc, acc = float('nan'), float('nan')
        marker = ''
        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = ' *'
        else:
            patience_counter += 1
        print(f'epoch {epoch+1:2d}  val AUC {auc:.4f}  acc {acc:.3f}{marker}')
        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch+1} (no val improvement in {args.patience})')
            break

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    preds_acc, labels_acc = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            preds_acc.append(logits)
            labels_acc.append(yb.numpy())
    test_preds = np.concatenate(preds_acc)
    test_labels = np.concatenate(labels_acc)
    test_auc = roc_auc_score(test_labels, test_preds) if len(set(test_labels)) > 1 else float('nan')
    test_probs = 1 / (1 + np.exp(-test_preds))
    test_acc = accuracy_score(test_labels, (test_probs > 0.5).astype(int))

    print()
    print(f'TEST — AUC {test_auc:.4f}  accuracy {test_acc:.3f}')

    # Calibration by P(win) bucket
    print('\nCalibration on test set:')
    print('bucket       N    actual win%')
    for lo, hi in [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]:
        m = (test_probs >= lo) & (test_probs < hi)
        if m.sum() > 0:
            print(f'{lo:.1f}-{hi:.1f}  {int(m.sum()):4d}  {test_labels[m].mean()*100:.1f}%')

    # Per-tier check (does the model do better/worse per tier?)
    print('\nPer-tier test accuracy:')
    tier_acc = {}
    test_tier_arr = np.array(ts_tiers)
    preds_bin = (test_probs > 0.5).astype(int)
    for tier in sorted(set(ts_tiers)):
        mask = test_tier_arr == tier
        if mask.sum() >= 5:
            acc_t = (preds_bin[mask] == test_labels[mask].astype(int)).mean()
            tier_acc[tier] = (int(mask.sum()), float(acc_t))
            print(f'  {tier:<22} n={int(mask.sum()):4d}  acc={acc_t*100:.1f}%')

    # Write report
    lines = []
    lines.append('# Tier direction NN')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'Trades source: `{args.trades}`')
    lines.append(f'Valid: {len(valid)}. Train {len(ls_train)} / Val {len(ls_val)} / Test {len(ls_test)}.')
    lines.append(f'Best val AUC: {best_val_auc:.4f}')
    lines.append(f'Test AUC: {test_auc:.4f}  accuracy: {test_acc:.3f}')
    lines.append('')
    lines.append('## Calibration (test)')
    lines.append('')
    lines.append('| P(win) bucket | N | actual win% |')
    lines.append('|---|---:|---:|')
    for lo, hi in [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]:
        m = (test_probs >= lo) & (test_probs < hi)
        if m.sum() > 0:
            lines.append(f'| {lo:.1f}–{hi:.1f} | {int(m.sum())} | {test_labels[m].mean()*100:.1f}% |')
    lines.append('')
    lines.append('## Per-tier test accuracy')
    lines.append('')
    lines.append('| Tier | N | Accuracy |')
    lines.append('|---|---:|---:|')
    for tier, (n, a) in sorted(tier_acc.items(), key=lambda kv: -kv[1][0]):
        lines.append(f'| {tier} | {n} | {a*100:.1f}% |')
    lines.append('')
    lines.append('## Tier mix (train + val + test)')
    lines.append('')
    lines.append('| Tier | N |')
    lines.append('|---|---:|')
    for tier, n in tier_counts.most_common():
        lines.append(f'| {tier} | {n} |')
    lines.append('')
    lines.append('## Reproduction')
    lines.append('')
    lines.append('```')
    lines.append(f'python tools/train_tier_direction_nn.py --trades {args.trades}')
    lines.append('```')

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save({
        'state_dict': best_state,
        'mean': mean,
        'std': std,
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
    }, args.out_model)
    print(f'\nWrote: {args.out_md}')
    print(f'Wrote: {args.out_model}')


if __name__ == '__main__':
    main()
