"""
CNN Hold Predictor — should this trade stay open or close?

Trained on trade paths: at each bar during a trade, regret tells us if
the optimal exit has been reached yet. The CNN learns the 79D physics
that predicts HOLD vs EXIT from current-bar features only.

This is a real-time regret approximator — learns to predict what regret
knows from hindsight, using only features available NOW.

Input:  6×13 grid (79D at current bar) + trade context (bars_held, pnl, direction)
Output: HOLD(1) / EXIT(0)

Training data:
  For each trade path, at each bar:
    - If bar < optimal_exit_bar → HOLD
    - If bar >= optimal_exit_bar → EXIT

Usage:
    python training/cnn_hold.py                    # train
    python training/cnn_hold.py --epochs 50
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
from core.features import FEATURE_NAMES, N_CORE, N_HELPER

BLENDED_TRADES = os.environ.get('CNN_TRADES_PATH', 'training/output/trades/blended_is.pkl')
REGRET_FILE = os.environ.get('CNN_REGRET_PATH', 'training/output/nn/regret_cnn_flipped.csv')
OUTPUT_DIR = os.environ.get('CNN_OUTPUT_DIR', 'training/output/nn')

N_TFS = 6
N_FEAT_PER_TF = N_CORE + N_HELPER  # 15 (12 core + 3 helper)
HELPER_START = N_CORE * N_TFS      # 72

TICK = 0.25
TV = 0.50


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='CNN hold predictor')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--sample-bars', type=int, default=10,
                   help='Sample N bars per trade for training (not all bars)')
    return p.parse_args()


def feat_to_grid(feat_79d):
    """Reshape 79D to 6×13 grid."""
    grid = np.zeros((N_TFS, N_FEAT_PER_TF), dtype=np.float32)
    for tf_idx in range(N_TFS):
        grid[tf_idx, :N_CORE] = feat_79d[tf_idx * N_CORE:(tf_idx + 1) * N_CORE]
        h_start = HELPER_START + tf_idx * N_HELPER
        grid[tf_idx, N_CORE:N_CORE + N_HELPER] = feat_79d[h_start:h_start + N_HELPER]
    return grid


def build_dataset(sample_bars=10):
    """Build training data: for each bar during each trade, HOLD or EXIT label.

    Samples `sample_bars` per trade to keep dataset manageable.
    """
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)
    regret = pd.read_csv(REGRET_FILE)

    print(f'Building hold/exit dataset from {len(trades)} trades...')

    grids = []       # 6×13 at each sampled bar
    contexts = []    # [bars_held_normalized, pnl_normalized, direction_sign, tier]
    labels = []      # 0=EXIT, 1=HOLD
    trade_ids = []   # which trade this sample came from

    for i, t in enumerate(tqdm(trades, desc='Trades')):
        path = t.get('path', [])
        if not path or len(path) < 6:
            continue
        if i >= len(regret):
            continue

        r = regret.iloc[i]
        best_action = r['best_action']

        # Optimal exit bar (from regret)
        if 'same' in best_action:
            optimal_bar = int(r['same_best_bar'])
        else:
            optimal_bar = int(r['counter_best_bar'])

        # Clamp to path length
        optimal_bar = min(optimal_bar, len(path) - 1)
        optimal_bar = max(optimal_bar, 1)

        # Direction sign: +1 for long, -1 for short
        dir_sign = 1.0 if t.get('dir', '') == 'long' else -1.0
        tier_num = {'CASCADE': 2, 'KILL_SHOT': 1, 'BASE_NMP': 0}.get(
            t.get('entry_tier', 'BASE_NMP'), 0)

        # Sample bars evenly across the trade
        if len(path) <= sample_bars:
            sample_indices = list(range(len(path)))
        else:
            sample_indices = np.linspace(0, len(path) - 1, sample_bars, dtype=int).tolist()

        for bar_idx in sample_indices:
            p = path[bar_idx]
            feat = p.get('features', None)
            if feat is None or len(feat) != len(FEATURE_NAMES):
                continue

            grid = feat_to_grid(np.array(feat))
            grids.append(grid)

            # Context: normalized bars held, normalized PnL, direction, tier
            bars_norm = bar_idx / max(len(path), 1)  # 0-1 progress through trade
            pnl = p.get('pnl', 0)
            pnl_norm = pnl / 50.0  # normalize to ~[-1, 1] range
            peak = p.get('peak_pnl', 0)
            peak_norm = peak / 50.0

            contexts.append([bars_norm, pnl_norm, peak_norm, dir_sign, float(tier_num)])

            # Label: HOLD if before optimal exit, EXIT if at/past
            label = 1 if bar_idx < optimal_bar else 0
            labels.append(label)
            trade_ids.append(i)

    grids = np.array(grids)
    contexts = np.array(contexts, dtype=np.float32)
    labels = np.array(labels)
    trade_ids = np.array(trade_ids)

    hold_n = (labels == 1).sum()
    exit_n = (labels == 0).sum()
    print(f'Dataset: {len(grids)} samples | HOLD: {hold_n} ({hold_n/len(grids)*100:.0f}%) | EXIT: {exit_n} ({exit_n/len(grids)*100:.0f}%)')

    return grids, contexts, labels, trade_ids


class HoldDataset(Dataset):
    def __init__(self, grids, contexts, labels):
        self.grids = torch.FloatTensor(grids).unsqueeze(1)  # (N, 1, 6, 13)
        self.contexts = torch.FloatTensor(contexts)          # (N, 5)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.grids[idx], self.contexts[idx], self.labels[idx]


class HoldCNN(nn.Module):
    """CNN: should this trade stay open?"""

    def __init__(self):
        super().__init__()

        # 79D grid branch
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

        # Context branch (bars_held, pnl, peak, direction, tier)
        context_dim = 5

        # Classifier
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

    # Use trade-level splits (not bar-level) to prevent leakage
    unique_trades = np.unique(trade_ids)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_trade_idx, val_trade_idx) in enumerate(kf.split(unique_trades)):
        print(f'\n--- Fold {fold+1}/5 ---')

        train_trades = set(unique_trades[train_trade_idx])
        val_trades = set(unique_trades[val_trade_idx])

        train_mask = np.array([tid in train_trades for tid in trade_ids])
        val_mask = np.array([tid in val_trades for tid in trade_ids])

        train_ds = HoldDataset(grids[train_mask], contexts[train_mask], labels[train_mask])
        val_ds = HoldDataset(grids[val_mask], contexts[val_mask], labels[val_mask])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = HoldCNN().to(device)
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

        # Eval best model
        model.load_state_dict(best_state)
        _, _, val_preds, val_labels_arr = eval_epoch(model, val_loader, criterion, device)

        # Breakdown: accuracy on HOLD vs EXIT separately
        hold_mask = val_labels_arr == 1
        exit_mask = val_labels_arr == 0
        hold_acc = (val_preds[hold_mask] == 1).mean() if hold_mask.sum() > 0 else 0
        exit_acc = (val_preds[exit_mask] == 0).mean() if exit_mask.sum() > 0 else 0

        print(f'  Best: {best_val_acc:.1%} | HOLD acc: {hold_acc:.1%} | EXIT acc: {exit_acc:.1%}')

        fold_results.append({
            'fold': fold, 'val_acc': best_val_acc,
            'hold_acc': hold_acc, 'exit_acc': exit_acc,
        })

    # Summary
    avg_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_hold = np.mean([r['hold_acc'] for r in fold_results])
    avg_exit = np.mean([r['exit_acc'] for r in fold_results])

    print(f'\n{"="*60}')
    print(f'CNN HOLD PREDICTOR RESULTS')
    print(f'{"="*60}')
    print(f'  CV Accuracy: {avg_acc:.1%}')
    print(f'  HOLD accuracy: {avg_hold:.1%} (correctly says "keep holding")')
    print(f'  EXIT accuracy: {avg_exit:.1%} (correctly says "exit now")')

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'cnn_hold.pt')
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
