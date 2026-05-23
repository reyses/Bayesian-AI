"""
CNN Exit — learns exit physics overrides from CNN hold's mistakes.

Trains on bars where CNN hold said HOLD but regret says EXIT was optimal.
Predicts which physics exit condition should have fired:
  0 = NO_OVERRIDE (hold was right)
  1 = MEAN_REACHED (z near zero)
  2 = OSCILLATION_DECAY (amplitude decayed)
  3 = VELOCITY_EXHAUSTED (momentum dead)
  4 = P_CENTER (sustained at center)
  5 = REGIME_SHIFT (vr trending)

Input: 6x13 grid + context (bars_held, pnl, peak, direction, tier, v5_aligned)
Output: exit physics class

Usage:
    python training/cnn_exit.py
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.features import FEATURE_NAMES, N_FEATURES

# Data paths
BLENDED_TRADES = os.environ.get('CNN_TRADES_PATH', 'training/output/trades/blended_is.pkl')
REGRET_FILE = os.environ.get('CNN_REGRET_PATH', 'training/output/nn/regret_cnn_flipped.csv')
OUTPUT_DIR = os.environ.get('CNN_OUTPUT_DIR', 'training/output/nn')

# Grid layout
GRID_H = 6
GRID_W = 15  # 12 core + 3 helper (91D)
N_CORE = 12
N_HELPER = 3
N_TFS = 6
HELPER_START = N_CORE * N_TFS  # 72

# Exit physics labels
EXIT_LABELS = {
    'NO_OVERRIDE': 0,      # hold was right, keep holding
    'MEAN_REACHED': 1,     # |z| < 0.5
    'OSCILLATION_DECAY': 2, # amplitude decayed
    'VELOCITY_EXHAUSTED': 3, # |velocity| < 0.3
    'P_CENTER': 4,         # p_center > 0.6 sustained
    'REGIME_SHIFT': 5,     # vr > 1.0
}
N_CLASSES = len(EXIT_LABELS)

# Feature indices
FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
TIER_MAP = {
    'CASCADE': 7, 'KILL_SHOT': 6, 'FADE_CALM': 5, 'FADE_MOMENTUM': 4,
    'FADE_AGAINST': 3, 'RIDE_CALM': 2, 'RIDE_MOMENTUM': 1, 'RIDE_AGAINST': 0,
    'FREIGHT_TRAIN': -1, 'REGIME_FLIP': -3, 'MTF_EXHAUSTION': -4,
    'EXHAUSTION_BAR': -5, 'ABSORPTION': -6, 'PEAK': -2,
    'BASE_NMP': 0, 'MANUAL': 0,
}


def feat_to_grid(feat_79d):
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for tf_idx in range(N_TFS):
        c_start = tf_idx * N_CORE
        grid[tf_idx, :N_CORE] = feat_79d[c_start:c_start + N_CORE]
        h_start = HELPER_START + tf_idx * N_HELPER
        grid[tf_idx, N_CORE:N_CORE + N_HELPER] = feat_79d[h_start:h_start + N_HELPER]
    return grid


def classify_exit_physics(feat_79d):
    """Determine which exit physics condition is active at this bar."""
    z = feat_79d[FEAT_IDX['1m_z_se']]
    vr = feat_79d[FEAT_IDX['1m_variance_ratio']]
    vel = feat_79d[FEAT_IDX['1m_velocity']]
    pc = feat_79d[FEAT_IDX['1m_p_at_center']]

    if abs(z) < 0.5:
        return EXIT_LABELS['MEAN_REACHED']
    if abs(vel) < 0.3:
        return EXIT_LABELS['VELOCITY_EXHAUSTED']
    if pc > 0.6:
        return EXIT_LABELS['P_CENTER']
    if vr > 1.0:
        return EXIT_LABELS['REGIME_SHIFT']
    # Can't easily detect oscillation decay without history, use NO_OVERRIDE
    return EXIT_LABELS['NO_OVERRIDE']


def build_dataset(sample_bars=5):
    """Build training dataset from trade paths + regret.

    For each trade path:
      - At regret's optimal exit bar: label = classify_exit_physics
      - At bars before optimal exit: label = NO_OVERRIDE
      - Sample `sample_bars` bars per trade for balance
    """
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)

    regret = pd.read_csv(REGRET_FILE)
    n = min(len(trades), len(regret))

    grids = []
    contexts = []
    labels = []

    for i in tqdm(range(n), desc='Building exit dataset', unit='trade'):
        t = trades[i]
        r = regret.iloc[i]
        path = t.get('path', [])
        if len(path) < 5:
            continue

        direction = t.get('dir', 'long')
        tier = t.get('entry_tier', 'FADE_CALM')
        v5_aligned = t.get('v5_aligned', True)
        entry_price = t.get('entry_price', 0)

        # Regret's optimal exit bar
        best_action = r.get('best_action', 'same_extended')
        if 'same' in best_action:
            opt_bar = int(r.get('same_best_bar', len(path) - 1))
        else:
            opt_bar = int(r.get('counter_best_bar', len(path) - 1))
        opt_bar = min(opt_bar, len(path) - 1)

        # Sample bars: some at optimal exit (positive label), some before (NO_OVERRIDE)
        sampled_bars = set()

        # Optimal exit bar(s)
        for b in range(max(0, opt_bar - 1), min(opt_bar + 2, len(path))):
            sampled_bars.add(b)

        # Random bars before optimal exit (NO_OVERRIDE)
        pre_bars = list(range(0, max(1, opt_bar - 2)))
        if pre_bars and len(pre_bars) > sample_bars:
            np.random.seed(i)
            pre_bars = list(np.random.choice(pre_bars, sample_bars, replace=False))
        for b in pre_bars[:sample_bars]:
            sampled_bars.add(b)

        for bar_idx in sorted(sampled_bars):
            p = path[bar_idx]
            feat = np.array(p.get('features', []))
            if len(feat) != N_FEATURES:
                continue

            grid = feat_to_grid(feat)
            pnl = p.get('pnl', 0)
            peak = p.get('peak_pnl', 0)
            bars_held = bar_idx

            # Label: at/near optimal exit bar → physics class. Before → NO_OVERRIDE
            if abs(bar_idx - opt_bar) <= 1:
                label = classify_exit_physics(feat)
            else:
                label = EXIT_LABELS['NO_OVERRIDE']

            # Context
            bars_norm = min(bars_held / 500.0, 1.0)
            pnl_norm = pnl / 50.0
            peak_norm = peak / 50.0
            dir_sign = 1.0 if direction == 'long' else -1.0
            tier_num = float(TIER_MAP.get(tier, 0))
            v5_sign = 1.0 if v5_aligned else -1.0

            grids.append(grid)
            contexts.append([bars_norm, pnl_norm, peak_norm, dir_sign, tier_num, v5_sign])
            labels.append(label)

    grids = np.array(grids, dtype=np.float32)
    contexts = np.array(contexts, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f'Dataset: {len(grids)} samples')
    for name, idx in EXIT_LABELS.items():
        print(f'  {name}: {(labels == idx).sum()} ({(labels == idx).sum() / len(labels) * 100:.0f}%)')

    return grids, contexts, labels


class ExitDataset(Dataset):
    def __init__(self, grids, contexts, labels):
        self.grids = torch.FloatTensor(grids)
        self.contexts = torch.FloatTensor(contexts)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return self.grids[idx], self.contexts[idx], self.labels[idx]


class ExitCNN(nn.Module):
    """CNN for exit physics classification from 6x13 grid + context."""

    def __init__(self, n_context=6, n_classes=N_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)),
        )
        grid_flat = 64 * 3 * 6  # 1152

        self.classifier = nn.Sequential(
            nn.Linear(grid_flat + n_context, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, grid, context):
        g = self.conv(grid.unsqueeze(1))
        g = g.view(g.size(0), -1)
        x = torch.cat([g, context], dim=1)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for grids, ctx, labels in loader:
        grids, ctx, labels = grids.to(device), ctx.to(device), labels.to(device)
        out = model(grids, ctx)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for grids, ctx, labels in loader:
            grids, ctx, labels = grids.to(device), ctx.to(device), labels.to(device)
            out = model(grids, ctx)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    grids, contexts, labels = build_dataset()

    # Normalize grids
    grid_mean = grids.mean(axis=0, keepdims=True)
    grid_std = grids.std(axis=0, keepdims=True).clip(min=1e-8)
    grids = (grids - grid_mean) / grid_std

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_acc = 0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(grids)):
        print(f'\n--- Fold {fold + 1}/5 ---')
        train_ds = ExitDataset(grids[train_idx], contexts[train_idx], labels[train_idx])
        val_ds = ExitDataset(grids[val_idx], contexts[val_idx], labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)

        model = ExitCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        fold_best = 0
        fold_state = None

        for epoch in range(30):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, preds, true = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_acc > fold_best:
                fold_best = val_acc
                fold_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:3d}: train={train_acc:.1%} val={val_acc:.1%}')

        # Per-class accuracy
        model.load_state_dict(fold_state)
        _, _, preds, true = eval_epoch(model, val_loader, criterion, device)
        print(f'  Best: {fold_best:.1%}')
        for name, idx in EXIT_LABELS.items():
            mask = true == idx
            if mask.sum() > 0:
                acc = (preds[mask] == idx).sum() / mask.sum()
                print(f'    {name}: {acc:.1%} ({mask.sum()} samples)')

        if fold_best > best_acc:
            best_acc = fold_best
            best_state = fold_state

    # Save best model
    print(f'\nBest overall: {best_acc:.1%}')

    checkpoint = {
        'model_state': best_state,
        'grid_mean': grid_mean,
        'grid_std': grid_std,
        'exit_labels': EXIT_LABELS,
        'n_classes': N_CLASSES,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'cnn_exit.pt')
    torch.save(checkpoint, path)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
