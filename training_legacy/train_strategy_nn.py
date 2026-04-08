"""
Strategy Router NN — predicts direction + hold duration + risk profile.

Input:  79D feature vector (10 core x 6 TFs + 3 helpers x 6 TFs + time_of_day)
Output: direction (LONG/SHORT/SKIP), duration (1,3,5,10,15,20,30 bars),
        expected PnL, expected drawdown, P(profit)

Architecture:
  Shared backbone: 79D → 128 → 64 (ReLU + BatchNorm + Dropout)
  Direction head:  64 → 32 → 3 (softmax: LONG/SHORT/SKIP)
  Duration head:   64 → 32 → 7 (softmax: 7 duration buckets)
  PnL head:        64 → 32 → 1 (regression: expected PnL)
  Drawdown head:   64 → 32 → 1 (regression: expected max drawdown)
  P(profit) head:  64 → 32 → 1 (sigmoid: probability of profit)

Training: walk-forward validation (train on months 1-8, val 9-10, test 11-12)
Loss: weighted sum of cross-entropy (dir + dur) + MSE (pnl + dd) + BCE (p_profit)

Usage:
  python training/train_strategy_nn.py                    # full training
  python training/train_strategy_nn.py --epochs 50        # custom epochs
  python training/train_strategy_nn.py --lr 0.001         # custom LR
  python training/train_strategy_nn.py --val-months 9,10  # custom val split

Spec: docs/Active/NN_SPEC.md
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features_79d import FEATURE_NAMES_79D, N_FEATURES

# ─── Config ──────────────────────────────────────────────────────────
LABELS_DIR = 'DATA/TRAINING_LABELS'
CHECKPOINT_DIR = 'checkpoints/strategy_nn'
RESULTS_DIR = 'reports/findings'

# Duration buckets (must match label generator)
DURATION_BUCKETS = [1, 3, 5, 10, 15, 20, 30]
N_DURATIONS = len(DURATION_BUCKETS)
DUR_TO_IDX = {d: i for i, d in enumerate(DURATION_BUCKETS)}

# Direction classes
DIR_CLASSES = ['skip', 'long', 'short']  # 0=skip, 1=long, 2=short
N_DIRS = len(DIR_CLASSES)
DIR_TO_IDX = {'none': 0, 'skip': 0, 'long': 1, 'short': 2}

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.001
DEFAULT_BATCH = 512
DEFAULT_DROPOUT = 0.2
EARLY_STOP_PATIENCE = 15

# Loss weights (balance classification vs regression)
W_DIR = 1.0       # direction classification
W_DUR = 1.0       # duration classification
W_PNL = 0.1       # PnL regression (scale down — dollar values are large)
W_DD = 0.1        # drawdown regression
W_PROB = 0.5      # P(profit) binary cross-entropy


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Train strategy router NN')
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--lr', type=float, default=DEFAULT_LR)
    p.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    p.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT)
    p.add_argument('--val-months', type=str, default='9,10',
                   help='Validation months (comma-separated, 1-indexed)')
    p.add_argument('--test-months', type=str, default='11,12',
                   help='Test months (comma-separated)')
    return p.parse_args()


# ─── Dataset ─────────────────────────────────────────────────────────

class StrategyDataset(Dataset):
    """Loads 79D features + labels from training label parquets."""

    def __init__(self, files: list):
        dfs = [pd.read_parquet(f) for f in tqdm(files, desc='Loading', unit='file')]
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        # Features: 79D columns
        feat_cols = FEATURE_NAMES_79D
        self.features = torch.tensor(df[feat_cols].values, dtype=torch.float32)

        # Replace NaN/inf with 0
        self.features = torch.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        # Direction label (0=skip, 1=long, 2=short)
        self.dir_labels = torch.tensor(
            df['best_direction'].map(DIR_TO_IDX).fillna(0).astype(int).values,
            dtype=torch.long
        )

        # Duration label (index into DURATION_BUCKETS)
        self.dur_labels = torch.tensor(
            df['best_duration'].map(DUR_TO_IDX).fillna(0).astype(int).values,
            dtype=torch.long
        )

        # Regression targets
        self.pnl = torch.tensor(df['best_pnl'].fillna(0).values, dtype=torch.float32)
        self.drawdown = torch.tensor(df['expected_drawdown'].fillna(0).values, dtype=torch.float32)

        # P(profit): 1 if best_pnl > 0, else 0
        self.p_profit = torch.tensor(
            (df['best_pnl'] > 0).astype(float).values, dtype=torch.float32
        )

        # Direction consistency (for analysis, not training target)
        self.consistency = torch.tensor(
            df['direction_consistency'].fillna(0).values, dtype=torch.float32
        )

        print(f'  {len(self)} samples, {self.features.shape[1]}D features')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'dir_label': self.dir_labels[idx],
            'dur_label': self.dur_labels[idx],
            'pnl': self.pnl[idx],
            'drawdown': self.drawdown[idx],
            'p_profit': self.p_profit[idx],
        }


# ─── Model ───────────────────────────────────────────────────────────

class StrategyRouterNN(nn.Module):
    """Multi-head network: shared backbone + 5 output heads."""

    def __init__(self, input_dim=N_FEATURES, dropout=DEFAULT_DROPOUT):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Direction head: 3 classes (skip, long, short)
        self.dir_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_DIRS),
        )

        # Duration head: 7 classes (1,3,5,10,15,20,30 bars)
        self.dur_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_DURATIONS),
        )

        # PnL head: regression (expected PnL in dollars)
        self.pnl_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Drawdown head: regression (expected max drawdown in dollars)
        self.dd_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),  # drawdown is always positive
        )

        # P(profit) head: binary classification
        self.prob_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.backbone(x)
        return {
            'dir_logits': self.dir_head(h),
            'dur_logits': self.dur_head(h),
            'pnl': self.pnl_head(h).squeeze(-1),
            'drawdown': self.dd_head(h).squeeze(-1),
            'p_profit': self.prob_head(h).squeeze(-1),
            'embedding': h,  # 64D backbone output — the NN's internal state representation
        }

    def predict(self, x):
        """Full prediction with strategy ID for Bayesian brain.

        Returns dict with:
          direction: 'long'/'short'/'skip'
          duration:  1/3/5/10/15/20/30
          strategy_id: (direction, duration) — brain lookup key
          expected_pnl: $
          expected_drawdown: $
          p_profit: [0, 1]
          confidence: how sure the NN is (softmax entropy)
          embedding: 64D internal representation (for brain state clustering)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)

            # Direction
            dir_probs = torch.softmax(out['dir_logits'], dim=-1)
            dir_idx = dir_probs.argmax(dim=-1).item()
            dir_name = DIR_CLASSES[dir_idx]
            dir_confidence = dir_probs[0, dir_idx].item()

            # Duration
            dur_probs = torch.softmax(out['dur_logits'], dim=-1)
            dur_idx = dur_probs.argmax(dim=-1).item()
            dur_bars = DURATION_BUCKETS[dur_idx]
            dur_confidence = dur_probs[0, dur_idx].item()

            # Strategy ID: the discrete key for Bayesian brain
            strategy_id = (dir_name, dur_bars)

            # Overall confidence: geometric mean of dir + dur confidence
            confidence = (dir_confidence * dur_confidence) ** 0.5

            return {
                'direction': dir_name,
                'duration': dur_bars,
                'strategy_id': strategy_id,
                'expected_pnl': out['pnl'].item(),
                'expected_drawdown': out['drawdown'].item(),
                'p_profit': out['p_profit'].item(),
                'confidence': confidence,
                'dir_confidence': dir_confidence,
                'dur_confidence': dur_confidence,
                'dir_probs': dir_probs[0].cpu().numpy(),   # [skip, long, short]
                'dur_probs': dur_probs[0].cpu().numpy(),   # [1,3,5,10,15,20,30]
                'embedding': out['embedding'][0].cpu().numpy(),  # 64D
            }


# ─── Training loop ──────────────────────────────────────────────────

def compute_loss(outputs, batch):
    """Weighted multi-task loss."""
    # Direction: cross-entropy
    loss_dir = nn.CrossEntropyLoss()(outputs['dir_logits'], batch['dir_label'])

    # Duration: cross-entropy (only for non-skip trades)
    trade_mask = batch['dir_label'] > 0  # skip = 0
    if trade_mask.sum() > 0:
        loss_dur = nn.CrossEntropyLoss()(
            outputs['dur_logits'][trade_mask],
            batch['dur_label'][trade_mask]
        )
    else:
        loss_dur = torch.tensor(0.0, device=loss_dir.device)

    # PnL: MSE (only for trades)
    if trade_mask.sum() > 0:
        loss_pnl = nn.MSELoss()(outputs['pnl'][trade_mask], batch['pnl'][trade_mask])
    else:
        loss_pnl = torch.tensor(0.0, device=loss_dir.device)

    # Drawdown: MSE (only for trades)
    if trade_mask.sum() > 0:
        loss_dd = nn.MSELoss()(outputs['drawdown'][trade_mask], batch['drawdown'][trade_mask])
    else:
        loss_dd = torch.tensor(0.0, device=loss_dir.device)

    # P(profit): BCE
    loss_prob = nn.BCELoss()(outputs['p_profit'], batch['p_profit'])

    total = (W_DIR * loss_dir + W_DUR * loss_dur +
             W_PNL * loss_pnl + W_DD * loss_dd + W_PROB * loss_prob)

    return total, {
        'dir': loss_dir.item(), 'dur': loss_dur.item(),
        'pnl': loss_pnl.item(), 'dd': loss_dd.item(),
        'prob': loss_prob.item(), 'total': total.item(),
    }


def evaluate(model, loader, device):
    """Evaluate model on a dataloader. Returns loss dict + accuracy metrics."""
    model.eval()
    total_loss = 0
    loss_parts = {'dir': 0, 'dur': 0, 'pnl': 0, 'dd': 0, 'prob': 0}
    n_batches = 0

    all_dir_pred = []
    all_dir_true = []
    all_dur_pred = []
    all_dur_true = []
    all_pnl_pred = []
    all_pnl_true = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['features'])
            loss, parts = compute_loss(outputs, batch)

            total_loss += loss.item()
            for k in loss_parts:
                loss_parts[k] += parts[k]
            n_batches += 1

            # Predictions
            dir_pred = outputs['dir_logits'].argmax(dim=1)
            dur_pred = outputs['dur_logits'].argmax(dim=1)

            all_dir_pred.extend(dir_pred.cpu().numpy())
            all_dir_true.extend(batch['dir_label'].cpu().numpy())
            all_dur_pred.extend(dur_pred.cpu().numpy())
            all_dur_true.extend(batch['dur_label'].cpu().numpy())
            all_pnl_pred.extend(outputs['pnl'].cpu().numpy())
            all_pnl_true.extend(batch['pnl'].cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    for k in loss_parts:
        loss_parts[k] /= max(n_batches, 1)

    dir_pred = np.array(all_dir_pred)
    dir_true = np.array(all_dir_true)
    dur_pred = np.array(all_dur_pred)
    dur_true = np.array(all_dur_true)

    # Direction accuracy (all samples)
    dir_acc = (dir_pred == dir_true).mean() * 100

    # Duration accuracy (trades only)
    trade_mask = dir_true > 0
    if trade_mask.sum() > 0:
        dur_acc = (dur_pred[trade_mask] == dur_true[trade_mask]).mean() * 100
    else:
        dur_acc = 0.0

    # PnL correlation
    pnl_pred = np.array(all_pnl_pred)
    pnl_true = np.array(all_pnl_true)
    if len(pnl_true) > 10:
        pnl_corr = np.corrcoef(pnl_pred, pnl_true)[0, 1]
    else:
        pnl_corr = 0.0

    return {
        'loss': avg_loss, **loss_parts,
        'dir_acc': dir_acc, 'dur_acc': dur_acc, 'pnl_corr': pnl_corr,
    }


def train_epoch(model, loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch['features'])
        loss, _ = compute_loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Strategy Router NN Training')
    print(f'  Device: {device}')
    print(f'  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch}')

    # Split files by month for walk-forward
    all_files = sorted(glob.glob(os.path.join(LABELS_DIR, '*.parquet')))
    print(f'  Label files: {len(all_files)}')

    val_months = set(int(m) for m in args.val_months.split(','))
    test_months = set(int(m) for m in args.test_months.split(','))

    train_files = []
    val_files = []
    test_files = []
    for f in all_files:
        name = os.path.basename(f).replace('.parquet', '')
        month = int(name.split('_')[1])
        if month in test_months:
            test_files.append(f)
        elif month in val_months:
            val_files.append(f)
        else:
            train_files.append(f)

    print(f'  Train: {len(train_files)} files | Val: {len(val_files)} | Test: {len(test_files)}')

    # Load datasets
    print('\nLoading training data...')
    train_ds = StrategyDataset(train_files)
    print('Loading validation data...')
    val_ds = StrategyDataset(val_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = StrategyRouterNN(input_dim=N_FEATURES, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n_params:,} parameters')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5,
                                                      factor=0.5, min_lr=1e-6)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print(f'\n{"Epoch":>5} {"TrLoss":>8} {"VaLoss":>8} {"DirAcc":>7} {"DurAcc":>7} {"PnLCorr":>8} {"LR":>10}')
    print('-' * 65)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch, 'train_loss': train_loss, **val_metrics,
        })

        print(f'{epoch+1:>5} {train_loss:>8.4f} {val_metrics["loss"]:>8.4f} '
              f'{val_metrics["dir_acc"]:>6.1f}% {val_metrics["dur_acc"]:>6.1f}% '
              f'{val_metrics["pnl_corr"]:>8.4f} {current_lr:>10.6f}')

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f'\nEarly stop at epoch {epoch+1} (patience={EARLY_STOP_PATIENCE})')
                break

    # Load best model for test evaluation
    if test_files:
        print(f'\nLoading test data ({len(test_files)} files)...')
        test_ds = StrategyDataset(test_files)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

        best_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        model.load_state_dict(best_ckpt['model_state'])

        test_metrics = evaluate(model, test_loader, device)
        print(f'\nTEST RESULTS (months {args.test_months}):')
        print(f'  Loss: {test_metrics["loss"]:.4f}')
        print(f'  Direction accuracy: {test_metrics["dir_acc"]:.1f}%')
        print(f'  Duration accuracy:  {test_metrics["dur_acc"]:.1f}%')
        print(f'  PnL correlation:    {test_metrics["pnl_corr"]:.4f}')

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(RESULTS_DIR, f'strategy_nn_training_{datetime.now():%Y%m%d_%H%M}.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f'\nHistory saved: {hist_path}')

    # Summary report
    report_path = os.path.join(RESULTS_DIR, f'strategy_nn_report_{datetime.now():%Y%m%d}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'# Strategy Router NN Training Report\n\n')
        f.write(f'**Date**: {datetime.now():%Y-%m-%d %H:%M}\n')
        f.write(f'**Epochs**: {epoch+1}/{args.epochs}\n')
        f.write(f'**Parameters**: {n_params:,}\n')
        f.write(f'**Train samples**: {len(train_ds):,}\n')
        f.write(f'**Val samples**: {len(val_ds):,}\n')
        f.write(f'**Best val loss**: {best_val_loss:.4f}\n\n')
        if test_files:
            f.write(f'## Test Results (months {args.test_months})\n')
            f.write(f'- Direction accuracy: {test_metrics["dir_acc"]:.1f}%\n')
            f.write(f'- Duration accuracy: {test_metrics["dur_acc"]:.1f}%\n')
            f.write(f'- PnL correlation: {test_metrics["pnl_corr"]:.4f}\n')
    print(f'Report saved: {report_path}')

    print(f'\nDone. Best model: {CHECKPOINT_DIR}/best_model.pt')


if __name__ == '__main__':
    main()
