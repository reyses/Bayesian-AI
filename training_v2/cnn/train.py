"""V2-native CNN training.

Trains V2DirectionCNN on (grid, tod, regime) → 3-class direction.

Split:
  - Train : 2025-02 to 2025-09 inclusive
  - Val   : 2025-10 to 2025-12 (used for best-epoch selection)
  - OOS   : 2026-* (NOT used during training; evaluated separately by run.py)

Class imbalance handled via inverse-frequency CrossEntropyLoss weights.

Usage:
    python -m training_v2.cnn.train
    python -m training_v2.cnn.train --epochs 30 --lr 5e-4
    python -m training_v2.cnn.train --tick-thr 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training_v2.cnn.dataset import build_dataset, _resolve_days
from training_v2.cnn.model import V2DirectionCNN, GRID_H, GRID_W


OUTPUT_DIR = 'training_v2/output/cnn'

CLASS_NAMES = ['SHORT', 'FLAT', 'LONG']


class GridDataset(Dataset):
    def __init__(self, X_grid, X_tod, X_reg, y):
        self.X_grid = torch.from_numpy(X_grid).unsqueeze(1)  # (N, 1, 8, 23)
        self.X_tod = torch.from_numpy(X_tod)                  # (N, 1)
        self.X_reg = torch.from_numpy(X_reg)                  # (N,)
        self.y = torch.from_numpy(y)                            # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_grid[i], self.X_tod[i], self.X_reg[i], self.y[i]


def parse_args():
    p = argparse.ArgumentParser(description='V2-native direction CNN trainer')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--cadence', type=int, default=300, help='sample cadence in seconds')
    p.add_argument('--horizon', type=int, default=300, help='forward-return horizon in seconds')
    p.add_argument('--tick-thr', type=float, default=4.0, help='tick threshold for SHORT/LONG label')
    p.add_argument('--train-end', type=str, default='2025-09-30')
    p.add_argument('--val-start', type=str, default='2025-10-01')
    p.add_argument('--val-end', type=str, default='2025-12-31')
    p.add_argument('--out', type=str, default=os.path.join(OUTPUT_DIR, 'direction_cnn.pt'))
    return p.parse_args()


def split_days(train_end: str, val_start: str, val_end: str):
    is_days = _resolve_days('is')
    train_days = [d for d in is_days if d.replace('_', '-') <= train_end]
    val_days = [d for d in is_days
                     if val_start <= d.replace('_', '-') <= val_end]
    return train_days, val_days


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for grid, tod, reg, y in loader:
        grid, tod, reg, y = (grid.to(device), tod.to(device),
                                  reg.to(device), y.to(device))
        out = model(grid, tod, reg)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for grid, tod, reg, y in loader:
            grid, tod, reg, y = (grid.to(device), tod.to(device),
                                      reg.to(device), y.to(device))
            out = model(grid, tod, reg)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.zeros(0, dtype=np.int64)
        all_labels = np.zeros(0, dtype=np.int64)
    return total_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def per_class_report(preds, labels):
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        n = int(mask.sum())
        acc = float((preds[mask] == i).mean()) if n else 0.0
        called = int((preds == i).sum())
        rows.append({'class': name, 'n_true': n, 'n_pred': called, 'recall': acc})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_days, val_days = split_days(args.train_end, args.val_start, args.val_end)
    print(f'Train days: {len(train_days)}  Val days: {len(val_days)}')

    print('Building train dataset...')
    Xtg, Xtt, Xtr, yt = build_dataset(days=train_days,
                                                cadence_s=args.cadence,
                                                forward_s=args.horizon,
                                                tick_threshold=args.tick_thr)
    print('Building val dataset...')
    Xvg, Xvt, Xvr, yv = build_dataset(days=val_days,
                                                cadence_s=args.cadence,
                                                forward_s=args.horizon,
                                                tick_threshold=args.tick_thr)

    print(f'Train: {len(yt)} samples; Val: {len(yv)} samples')
    if len(yt) == 0 or len(yv) == 0:
        print('Empty dataset — bailing.')
        return

    # Class weights — inverse frequency (clip extreme rarity)
    cls_count = np.bincount(yt, minlength=3).astype(np.float32)
    weights = 1.0 / np.clip(cls_count, 1.0, None)
    weights = weights / weights.sum() * 3.0
    print(f'Class counts (train): {dict(zip(CLASS_NAMES, cls_count.astype(int)))}')
    print(f'Loss weights:         {dict(zip(CLASS_NAMES, weights.round(3)))}')

    train_ds = GridDataset(Xtg, Xtt, Xtr, yt)
    val_ds = GridDataset(Xvg, Xvt, Xvr, yv)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = V2DirectionCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vp, vy = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl)

        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({'epoch': epoch, 'train_loss': tl, 'train_acc': ta,
                              'val_loss': vl, 'val_acc': va})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:>3}: train_acc={ta:.1%} val_acc={va:.1%} '
                      f'train_loss={tl:.4f} val_loss={vl:.4f}')

    print(f'\nBest val_acc: {best_val_acc:.1%}')
    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, vp, vy = eval_epoch(model, val_loader, criterion, device)
    pcr = per_class_report(vp, vy)
    print('\nPer-class val report:')
    print(pcr.to_string(index=False))

    torch.save({
        'model_state': best_state if best_state is not None else model.state_dict(),
        'val_acc': float(best_val_acc),
        'history': history,
        'args': vars(args),
        'per_class': pcr.to_dict(orient='records'),
        'class_names': CLASS_NAMES,
    }, args.out)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
