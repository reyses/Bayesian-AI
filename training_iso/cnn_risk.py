"""
CNN Risk Predictor — will this negative trade recover or die?

Only fires when PnL is negative during a trade. Trained on:
  RECOVER (1): bars where PnL < 0 but trade ends as winner
  DEAD (0):    bars where PnL < 0 and trade ends as loser

Learns the 79D physics that distinguishes "temporary dip" from "dead trade."
Combined with CNN hold, this creates the exit system:
  CNN Hold: when to exit winners (hold vs take profit)
  CNN Risk: when to cut losers (recover vs dead)

Usage:
    python training/cnn_risk.py                    # train
    python training/cnn_risk.py --epochs 50
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D, N_CORE, N_HELPER

BLENDED_TRADES = 'training/output/trades/blended_is.pkl'
OUTPUT_DIR = 'training/output/nn'

N_TFS = 6
N_FEAT_PER_TF = N_CORE + N_HELPER  # 13
HELPER_START = N_CORE * N_TFS      # 60

TICK = 0.25
TV = 0.50


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='CNN risk predictor')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--sample-bars', type=int, default=10,
                   help='Sample N negative bars per trade')
    return p.parse_args()


def feat_to_grid(feat_79d):
    """Reshape 79D to 6x13 grid."""
    grid = np.zeros((N_TFS, N_FEAT_PER_TF), dtype=np.float32)
    for tf_idx in range(N_TFS):
        grid[tf_idx, :N_CORE] = feat_79d[tf_idx * N_CORE:(tf_idx + 1) * N_CORE]
        h_start = HELPER_START + tf_idx * N_HELPER
        grid[tf_idx, N_CORE:N_CORE + N_HELPER] = feat_79d[h_start:h_start + N_HELPER]
    return grid


def build_dataset(sample_bars=10):
    """Build training data: negative bars only, labeled by final outcome."""
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)

    print(f'Building risk dataset from {len(trades)} trades...')

    grids = []
    contexts = []
    labels = []
    trade_ids = []

    for i, t in enumerate(tqdm(trades, desc='Trades')):
        path = t.get('path', [])
        if not path or len(path) < 6:
            continue

        win = t['pnl'] > 0
        dir_sign = 1.0 if t.get('dir', '') == 'long' else -1.0
        tier_num = {'CASCADE': 2, 'KILL_SHOT': 1, 'BASE_NMP': 0}.get(
            t.get('entry_tier', 'BASE_NMP'), 0)

        # Find negative bars only
        neg_indices = [j for j, p in enumerate(path)
                       if p.get('pnl', 0) < 0 and
                       p.get('features_79d') is not None and
                       len(p.get('features_79d', [])) == len(FEATURE_NAMES_79D)]

        if not neg_indices:
            continue

        # Sample N negative bars
        if len(neg_indices) > sample_bars:
            sampled = np.linspace(0, len(neg_indices) - 1, sample_bars, dtype=int)
            neg_indices = [neg_indices[s] for s in sampled]

        for bar_idx in neg_indices:
            p = path[bar_idx]
            feat = np.array(p['features_79d'])

            grid = feat_to_grid(feat)
            grids.append(grid)

            # Context: bars_held proportion, current pnl, depth from peak, direction, tier
            bars_norm = bar_idx / max(len(path), 1)
            pnl = p.get('pnl', 0)
            pnl_norm = pnl / 50.0
            peak = p.get('peak_pnl', 0)
            peak_norm = peak / 50.0
            # Depth: how far from peak (peak - current)
            depth = (peak - pnl) / 50.0

            contexts.append([bars_norm, pnl_norm, peak_norm, depth, dir_sign, float(tier_num)])

            # Label: will this trade recover?
            label = 1 if win else 0
            labels.append(label)
            trade_ids.append(i)

    grids = np.array(grids)
    contexts = np.array(contexts, dtype=np.float32)
    labels = np.array(labels)
    trade_ids = np.array(trade_ids)

    recover_n = (labels == 1).sum()
    dead_n = (labels == 0).sum()
    print(f'Dataset: {len(grids)} negative-bar samples | RECOVER: {recover_n} ({recover_n/len(grids)*100:.0f}%) | DEAD: {dead_n} ({dead_n/len(grids)*100:.0f}%)')

    return grids, contexts, labels, trade_ids


class RiskDataset(Dataset):
    def __init__(self, grids, contexts, labels):
        self.grids = torch.FloatTensor(grids).unsqueeze(1)  # (N, 1, 6, 13)
        self.contexts = torch.FloatTensor(contexts)           # (N, 6)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.grids[idx], self.contexts[idx], self.labels[idx]


class RiskCNN(nn.Module):
    """CNN: will this negative trade recover or die?"""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)),
        )
        conv_flat = 64 * 3 * 6  # 1152

        # Context: bars_norm, pnl, peak, depth, direction, tier
        context_dim = 6

        self.classifier = nn.Sequential(
            nn.Linear(conv_flat + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, grid, context):
        g = self.conv(grid)
        g = g.view(g.size(0), -1)
        x = torch.cat([g, context], dim=1)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for grid, ctx, lab in loader:
        grid, ctx, lab = grid.to(device), ctx.to(device), lab.to(device)
        out = model(grid, ctx)
        loss = criterion(out, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * lab.size(0)
        correct += (out.argmax(1) == lab).sum().item()
        total += lab.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for grid, ctx, lab in loader:
            grid, ctx, lab = grid.to(device), ctx.to(device), lab.to(device)
            out = model(grid, ctx)
            loss = criterion(out, lab)
            total_loss += loss.item() * lab.size(0)
            pred = out.argmax(1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    grids, contexts, labels, trade_ids = build_dataset(sample_bars=args.sample_bars)

    # Normalize grids
    grid_mean = grids.mean(axis=0, keepdims=True)
    grid_std = grids.std(axis=0, keepdims=True).clip(min=1e-8)
    grids = (grids - grid_mean) / grid_std

    # Trade-level splits
    unique_trades = np.unique(trade_ids)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_trade_idx, val_trade_idx) in enumerate(kf.split(unique_trades)):
        print(f'\n--- Fold {fold+1}/5 ---')

        train_trades = set(unique_trades[train_trade_idx])
        val_trades = set(unique_trades[val_trade_idx])

        train_mask = np.array([tid in train_trades for tid in trade_ids])
        val_mask = np.array([tid in val_trades for tid in trade_ids])

        train_ds = RiskDataset(grids[train_mask], contexts[train_mask], labels[train_mask])
        val_ds = RiskDataset(grids[val_mask], contexts[val_mask], labels[val_mask])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = RiskCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_acc = 0
        best_state = None

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:>3}: train={train_acc:.1%} val={val_acc:.1%}')

        model.load_state_dict(best_state)
        _, _, val_preds, val_labels_arr = eval_epoch(model, val_loader, criterion, device)

        recover_mask = val_labels_arr == 1
        dead_mask = val_labels_arr == 0
        recover_acc = (val_preds[recover_mask] == 1).mean() if recover_mask.sum() > 0 else 0
        dead_acc = (val_preds[dead_mask] == 0).mean() if dead_mask.sum() > 0 else 0

        print(f'  Best: {best_val_acc:.1%} | RECOVER acc: {recover_acc:.1%} | DEAD acc: {dead_acc:.1%}')

        fold_results.append({
            'fold': fold, 'val_acc': best_val_acc,
            'recover_acc': recover_acc, 'dead_acc': dead_acc,
        })

    avg_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_recover = np.mean([r['recover_acc'] for r in fold_results])
    avg_dead = np.mean([r['dead_acc'] for r in fold_results])

    print(f'\n{"="*60}')
    print(f'CNN RISK PREDICTOR RESULTS')
    print(f'{"="*60}')
    print(f'  CV Accuracy: {avg_acc:.1%}')
    print(f'  RECOVER accuracy: {avg_recover:.1%} (correctly says "it will recover")')
    print(f'  DEAD accuracy: {avg_dead:.1%} (correctly says "cut this trade")')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'cnn_risk.pt')
    torch.save({
        'model_state': best_state,
        'grid_mean': grid_mean,
        'grid_std': grid_std,
        'fold_results': fold_results,
        'accuracy': avg_acc,
    }, save_path)
    print(f'Model saved: {save_path}')


if __name__ == '__main__':
    main()
