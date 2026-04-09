"""
CNN Flip Predictor — classifies entry as FADE / RIDE / SKIP from 79D grid.

Reshapes 79D into 6×13 grid (6 TFs × 13 features per TF).
CNN convolves across TFs AND features simultaneously — detects
spatial patterns like "z extreme at 1m + wick at 5m + 1h aligned"
as ONE pattern, not three independent features.

Also accepts the 5-point trade path (entry, 25%, 50%, 75%, exit)
reshaped as 5×13 per TF = 5×6×13 volume. This captures the curve.

Usage:
    python nn_v2/cnn_flip.py                    # train with defaults
    python nn_v2/cnn_flip.py --epochs 50        # more epochs
    python nn_v2/cnn_flip.py --no-path          # entry only, no path
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
from core.features_79d import FEATURE_NAMES_79D, TF_ORDER, N_CORE, N_HELPER, N_TFS

BLENDED_TRADES = 'nn_v2/output/trades/blended_is.pkl'
REGRET_FILE = 'nn_v2/output/tree/regret_analysis.csv'
OUTPUT_DIR = 'nn_v2/output/tree'

# Grid layout: 6 TFs × 13 features (10 core + 3 helper)
N_FEATURES_PER_TF = N_CORE + N_HELPER  # 13
GRID_H = N_TFS                          # 6
GRID_W = N_FEATURES_PER_TF              # 13

# 5 time points for path
N_TIME_POINTS = 5

TIER_MAP = {'CASCADE': 2, 'KILL_SHOT': 1, 'BASE_NMP': 0}

# Sample features for path (indices into 79D for key features per TF)
# We use the full 13 features per TF at each time point
CORE_START = 0
HELPER_START = N_CORE * N_TFS  # 60


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='CNN flip predictor')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--no-path', action='store_true', help='Entry only, no path data')
    return p.parse_args()


def feat_to_grid(feat_79d):
    """Reshape 79D vector into 6×13 grid.

    Core features: indices 0-59 → 6 TFs × 10 features
    Helper features: indices 60-77 → 6 TFs × 3 features
    Combine into 6 × 13 grid.
    """
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for tf_idx in range(N_TFS):
        # Core: 10 features
        core_start = tf_idx * N_CORE
        grid[tf_idx, :N_CORE] = feat_79d[core_start:core_start + N_CORE]
        # Helper: 3 features
        helper_start = HELPER_START + tf_idx * N_HELPER
        grid[tf_idx, N_CORE:N_CORE + N_HELPER] = feat_79d[helper_start:helper_start + N_HELPER]
    return grid


def build_dataset(use_path=True):
    """Build training dataset from blended trades + regret."""
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)
    regret = pd.read_csv(REGRET_FILE)

    entries = []      # 6×13 grids at entry
    paths = []        # 5×6×13 volumes (entry + 4 path points)
    labels = []       # 0=FADE, 1=RIDE, 2=SKIP
    tiers = []        # tier as numeric
    pnls = []

    SKIP_THRESHOLD = 2.0  # best_pnl below this = untradeable

    for i, t in enumerate(trades):
        e = np.array(t.get('entry_79d', []))
        if len(e) != len(FEATURE_NAMES_79D) or i >= len(regret):
            continue

        r = regret.iloc[i]
        best_pnl = r.get('best_pnl', 0)

        # 3-class label: FADE / RIDE / SKIP
        if best_pnl <= SKIP_THRESHOLD:
            label = 2  # SKIP
        elif 'counter' in r['best_action']:
            label = 1  # RIDE (counter = go with z instead of fading)
        else:
            label = 0  # FADE (same = fade z as NMP intended)

        # Entry grid
        entry_grid = feat_to_grid(e)
        entries.append(entry_grid)
        labels.append(label)
        tiers.append(TIER_MAP.get(t.get('entry_tier', 'FADE_CALM'), 0))
        pnls.append(t['pnl'])

        # Path: 5 points (entry, 25%, 50%, 75%, exit)
        if use_path:
            path = t.get('path', [])
            if path and len(path) >= 6:
                indices = [0, len(path)//4, len(path)//2, 3*len(path)//4, len(path)-1]
                path_vol = np.zeros((N_TIME_POINTS, GRID_H, GRID_W), dtype=np.float32)
                path_vol[0] = entry_grid  # entry is point 0
                for pi, pidx in enumerate(indices[1:], 1):
                    pfeat = path[pidx].get('features_79d', None)
                    if pfeat is not None and len(pfeat) == len(FEATURE_NAMES_79D):
                        path_vol[pi] = feat_to_grid(np.array(pfeat))
                    else:
                        path_vol[pi] = entry_grid  # fallback
                paths.append(path_vol)
            else:
                # No path — repeat entry
                path_vol = np.stack([entry_grid] * N_TIME_POINTS)
                paths.append(path_vol)

    entries = np.array(entries)
    labels = np.array(labels)
    tiers = np.array(tiers, dtype=np.float32)
    pnls = np.array(pnls, dtype=np.float32)

    if use_path:
        paths = np.array(paths)
    else:
        paths = None

    print(f'Dataset: {len(entries)} trades | FADE: {(labels==0).sum()} | RIDE: {(labels==1).sum()} | SKIP: {(labels==2).sum()}')
    return entries, paths, labels, tiers, pnls


class FlipDataset(Dataset):
    def __init__(self, entries, paths, labels, tiers):
        self.entries = torch.FloatTensor(entries).unsqueeze(1)  # (N, 1, 6, 13)
        self.labels = torch.LongTensor(labels)
        self.tiers = torch.FloatTensor(tiers).unsqueeze(1)  # (N, 1)
        if paths is not None:
            self.paths = torch.FloatTensor(paths)  # (N, 5, 6, 13)
            self.use_path = True
        else:
            self.paths = None
            self.use_path = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_path:
            return self.entries[idx], self.paths[idx], self.tiers[idx], self.labels[idx]
        return self.entries[idx], self.tiers[idx], self.labels[idx]


class FlipCNN(nn.Module):
    """CNN for flip prediction from 6×13 TF grid + optional path."""

    def __init__(self, use_path=True):
        super().__init__()
        self.use_path = use_path

        # Entry CNN: 1×6×13 → features
        self.entry_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),   # (32, 6, 13)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  # (64, 6, 13)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)),                       # (64, 3, 6)
        )
        entry_flat = 64 * 3 * 6  # 1152

        if use_path:
            # Path CNN: 5×6×13 → treats 5 time points as channels
            self.path_conv = nn.Sequential(
                nn.Conv2d(N_TIME_POINTS, 32, kernel_size=(3, 3), padding=1),  # (32, 6, 13)
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  # (64, 6, 13)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 6)),                       # (64, 3, 6)
            )
            path_flat = 64 * 3 * 6  # 1152
        else:
            path_flat = 0

        # Classifier: entry features + path features + tier
        total_flat = entry_flat + path_flat + 1  # +1 for tier
        self.classifier = nn.Sequential(
            nn.Linear(total_flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # FADE=0, RIDE=1, SKIP=2
        )

    def forward(self, entry, path=None, tier=None):
        # Entry branch
        e = self.entry_conv(entry)
        e = e.view(e.size(0), -1)

        if self.use_path and path is not None:
            p = self.path_conv(path)
            p = p.view(p.size(0), -1)
            x = torch.cat([e, p, tier], dim=1)
        else:
            x = torch.cat([e, tier], dim=1)

        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        if len(batch) == 4:
            entry, path, tier, labels = batch
            entry, path, tier, labels = entry.to(device), path.to(device), tier.to(device), labels.to(device)
            out = model(entry, path, tier)
        else:
            entry, tier, labels = batch
            entry, tier, labels = entry.to(device), tier.to(device), labels.to(device)
            out = model(entry, tier=tier)

        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                entry, path, tier, labels = batch
                entry, path, tier, labels = entry.to(device), path.to(device), tier.to(device), labels.to(device)
                out = model(entry, path, tier)
            else:
                entry, tier, labels = batch
                entry, tier, labels = entry.to(device), tier.to(device), labels.to(device)
                out = model(entry, tier=tier)

            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    use_path = not args.no_path
    print(f'Path data: {"YES (5-point curve)" if use_path else "NO (entry only)"}')

    entries, paths, labels, tiers, pnls = build_dataset(use_path=use_path)

    # Normalize features (per-feature z-score)
    entry_mean = entries.mean(axis=0, keepdims=True)
    entry_std = entries.std(axis=0, keepdims=True).clip(min=1e-8)
    entries = (entries - entry_mean) / entry_std

    if paths is not None:
        path_mean = paths.mean(axis=0, keepdims=True)
        path_std = paths.std(axis=0, keepdims=True).clip(min=1e-8)
        paths = (paths - path_mean) / path_std

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(entries)):
        print(f'\n--- Fold {fold+1}/5 ---')

        train_ds = FlipDataset(
            entries[train_idx], paths[train_idx] if paths is not None else None,
            labels[train_idx], tiers[train_idx])
        val_ds = FlipDataset(
            entries[val_idx], paths[val_idx] if paths is not None else None,
            labels[val_idx], tiers[val_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = FlipCNN(use_path=use_path).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:>3}: train_acc={train_acc:.1%} val_acc={val_acc:.1%} '
                      f'train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

        # Load best model, evaluate PnL impact
        model.load_state_dict(best_model_state)
        _, _, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)

        val_pnl = pnls[val_idx]
        kept_mask = val_preds == 0  # predicted SAME → keep
        kept_pnl = val_pnl[kept_mask].sum()
        skip_pnl = val_pnl[~kept_mask].sum()

        print(f'  Best val_acc: {best_val_acc:.1%}')
        print(f'  Kept (pred SAME): {kept_mask.sum()} trades, ${kept_pnl:,.0f} '
              f'(${kept_pnl/max(kept_mask.sum(),1):.1f}/trade)')
        print(f'  Skipped (pred COUNTER): {(~kept_mask).sum()} trades, ${skip_pnl:,.0f} '
              f'(${skip_pnl/max((~kept_mask).sum(),1):.1f}/trade)')

        fold_results.append({
            'fold': fold, 'val_acc': best_val_acc,
            'kept_n': kept_mask.sum(), 'kept_pnl': kept_pnl,
            'skip_n': (~kept_mask).sum(), 'skip_pnl': skip_pnl,
        })

    # Summary
    print(f'\n{"="*60}')
    print(f'CNN FLIP PREDICTOR RESULTS')
    print(f'{"="*60}')
    avg_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_kept = np.mean([r['kept_pnl'] for r in fold_results])
    avg_skip = np.mean([r['skip_pnl'] for r in fold_results])
    avg_kept_n = np.mean([r['kept_n'] for r in fold_results])
    avg_skip_n = np.mean([r['skip_n'] for r in fold_results])

    print(f'  CV Accuracy: {avg_acc:.1%} (tree was 55.5%)')
    print(f'  Kept (pred SAME): {avg_kept_n:.0f} trades, ${avg_kept:,.0f} (${avg_kept/max(avg_kept_n,1):.1f}/trade)')
    print(f'  Skipped (pred COUNTER): {avg_skip_n:.0f} trades, ${avg_skip:,.0f} (${avg_skip/max(avg_skip_n,1):.1f}/trade)')
    print(f'  Path used: {use_path}')

    # Save best model from last fold
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'cnn_flip.pt')
    torch.save({
        'model_state': best_model_state,
        'use_path': use_path,
        'entry_mean': entry_mean,
        'entry_std': entry_std,
        'path_mean': path_mean if paths is not None else None,
        'path_std': path_std if paths is not None else None,
        'fold_results': fold_results,
        'accuracy': avg_acc,
    }, save_path)
    print(f'\nModel saved: {save_path}')


if __name__ == '__main__':
    main()
