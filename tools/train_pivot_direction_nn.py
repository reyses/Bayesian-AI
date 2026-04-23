"""
Train a CNN on 91D features at RM pivot entry → P(win).

Goal: given the features at a pivot-confirmed entry bar, predict whether
the trade will end profitable. Used as a filter on the RM engine.

Walk-forward split:
  - Train: first 70% of IS days (by date)
  - Val:   next 15% of IS days
  - Test:  last 15% of IS days

Model: small CNN on 6×N_FEAT_PER_TF grid. 91D = 12 core × 6 TF + 3 helper
× 6 TF + 1 extra = 91. We reshape as 6 × 15 (12 core + 3 helper per TF)
and drop the 91st if present.

Output:
    reports/findings/pivot_direction_nn.md
    training_RM_physics/output/pivot_direction_cnn.pt
"""
import os
import sys
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_PKL = 'training_RM_physics/output/trades/rm_is.pkl'
OUT_MD = 'reports/findings/pivot_direction_nn.md'
OUT_MODEL = 'training_RM_physics/output/pivot_direction_cnn.pt'

N_CORE_PER_TF = 12
N_HELPER_PER_TF = 3
N_TFS = 6
FEAT_DIM = N_CORE_PER_TF + N_HELPER_PER_TF   # 15
GRID_H = N_TFS                                # 6
GRID_W = FEAT_DIM                             # 15
# With 91D: 12 core * 6 = 72 core, 3 helper * 6 = 18 helper. 72 + 18 = 90.
# The 91st element is usually the additional 1m close or similar. We drop.


def feat_to_grid(f91):
    """Reshape 91D → 6 × 15 grid. Drops the 91st element if present."""
    f = np.asarray(f91[:90], dtype=np.float32)  # ensure we only take 90
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    core_block = f[:N_CORE_PER_TF * N_TFS]       # first 72
    helper_block = f[N_CORE_PER_TF * N_TFS:]     # last 18
    for tf in range(N_TFS):
        grid[tf, :N_CORE_PER_TF] = core_block[tf * N_CORE_PER_TF:(tf + 1) * N_CORE_PER_TF]
        grid[tf, N_CORE_PER_TF:] = helper_block[tf * N_HELPER_PER_TF:(tf + 1) * N_HELPER_PER_TF]
    return grid


class PivotDataset(Dataset):
    def __init__(self, grids, labels):
        self.grids = torch.from_numpy(grids).unsqueeze(1)  # (N, 1, 6, 15)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.grids[i], self.labels[i]


class PivotCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 7, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    with open(TRADES_PKL, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades)} trades')

    # Filter out trades missing features
    valid = [t for t in trades if len(t.get('entry_79d', [])) >= 90 and t['pnl'] != 0]
    print(f'Valid (features present, pnl != 0): {len(valid)}')

    # Sort by day for walk-forward
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
            train_data.append((grid, label))
        elif t['day'] in val_days:
            val_data.append((grid, label))
        elif t['day'] in test_days:
            test_data.append((grid, label))

    def pack(data):
        if not data:
            return np.empty((0, GRID_H, GRID_W), dtype=np.float32), np.empty(0, dtype=np.float32)
        gs = np.stack([d[0] for d in data])
        ls = np.array([d[1] for d in data], dtype=np.float32)
        return gs, ls

    gs_train, ls_train = pack(train_data)
    gs_val, ls_val = pack(val_data)
    gs_test, ls_test = pack(test_data)
    print(f'Train: {len(ls_train)}  Val: {len(ls_val)}  Test: {len(ls_test)}')
    print(f'Train win%: {ls_train.mean()*100:.1f}%  Val: {ls_val.mean()*100:.1f}%  Test: {ls_test.mean()*100:.1f}%')

    # Standardize per-feature using train stats
    mean = gs_train.mean(axis=0, keepdims=True)
    std = gs_train.std(axis=0, keepdims=True) + 1e-6
    gs_train = (gs_train - mean) / std
    gs_val = (gs_val - mean) / std
    gs_test = (gs_test - mean) / std

    train_loader = DataLoader(PivotDataset(gs_train, ls_train),
                               batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(PivotDataset(gs_val, ls_val),
                             batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(PivotDataset(gs_test, ls_test),
                              batch_size=1024, shuffle=False, num_workers=0)

    model = PivotCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    for epoch in range(25):
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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                all_preds.append(logits)
                all_labels.append(yb.numpy())
        preds = np.concatenate(all_preds) if all_preds else np.empty(0)
        labels = np.concatenate(all_labels) if all_labels else np.empty(0)
        if len(labels) > 0 and len(set(labels)) > 1:
            auc = roc_auc_score(labels, preds)
            probs = 1 / (1 + np.exp(-preds))
            acc = accuracy_score(labels, (probs > 0.5).astype(int))
        else:
            auc, acc = float('nan'), float('nan')
        print(f'epoch {epoch+1:2d}  val AUC {auc:.4f}  acc {acc:.3f}')
        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            all_preds.append(logits)
            all_labels.append(yb.numpy())
    test_preds = np.concatenate(all_preds)
    test_labels = np.concatenate(all_labels)
    test_auc = roc_auc_score(test_labels, test_preds) if len(set(test_labels)) > 1 else float('nan')
    test_probs = 1 / (1 + np.exp(-test_preds))
    test_acc = accuracy_score(test_labels, (test_probs > 0.5).astype(int))

    # Bucket analysis: does P(win) correlate with actual win rate?
    print()
    print(f'TEST — AUC {test_auc:.4f}  acc {test_acc:.3f}')
    lines = ['# Pivot direction NN', '']
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'Trades: {len(valid)}  Train {len(ls_train)} Val {len(ls_val)} Test {len(ls_test)}')
    lines.append(f'Best val AUC: {best_val_auc:.4f}')
    lines.append(f'Test AUC: {test_auc:.4f}  accuracy: {test_acc:.3f}')
    lines.append('')
    lines.append('## Prediction-calibration on test set')
    lines.append('')
    lines.append('| prob bucket | N trades | actual win% |')
    lines.append('|---|---:|---:|')
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for lo, hi in bins:
        mask = (test_probs >= lo) & (test_probs < hi)
        if mask.sum() > 0:
            lines.append(f'| {lo:.1f} – {hi:.1f} | {int(mask.sum())} | {test_labels[mask].mean()*100:.1f}% |')
    lines.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    torch.save({'state_dict': best_state, 'mean': mean, 'std': std}, OUT_MODEL)
    print(f'Wrote: {OUT_MD}')
    print(f'Wrote: {OUT_MODEL}')


if __name__ == '__main__':
    main()
