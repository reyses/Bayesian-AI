"""
CNN Entry/Direction/Duration — shared backbone, 3 classification heads.

Trained PER TIER on oracle (regret) labels. At inference:
  Head A (entry gate):  P(good_entry) — skip if below threshold
  Head B (direction):   P(long), P(short) — override physics if confident
  Head C (duration):    P(SHORT), P(MEDIUM), P(LONG) — sets exit patience

Input: 91D features at entry, reshaped to 6×15 grid (6 TFs × 15 features).
Output: 3 probability distributions (one per head).

Usage:
    python training/cnn_entry_direction.py                    # train all tiers
    python training/cnn_entry_direction.py --tier CASCADE     # single tier
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import N_FEATURES, N_CORE, N_HELPER, N_TFS, FEATURE_NAMES

OUTPUT_DIR = 'training_v2/output/nn'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Grid shape: 6 TFs × (12 core + 3 helper) = 6 × 15
GRID_ROWS = N_TFS       # 6
GRID_COLS = N_CORE + N_HELPER  # 12 + 3 = 15

# Training
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # early stopping
SEED = 42


class EntryDirectionDataset(Dataset):
    """Dataset for entry-time predictions (entry gate + direction + duration)."""

    def __init__(self, features_91d: np.ndarray, entry_labels: np.ndarray,
                 direction_labels: np.ndarray, duration_labels: np.ndarray):
        self.grids = self._to_grids(features_91d)
        self.entry_labels = torch.tensor(entry_labels, dtype=torch.long)
        self.direction_labels = torch.tensor(direction_labels, dtype=torch.long)
        self.duration_labels = torch.tensor(duration_labels, dtype=torch.long)

    def _to_grids(self, features: np.ndarray) -> torch.Tensor:
        """Reshape 91D flat features to 6×15 grid."""
        n = len(features)
        grids = np.zeros((n, 1, GRID_ROWS, GRID_COLS), dtype=np.float32)
        for i in range(n):
            feat = features[i]
            for tf_idx in range(N_TFS):
                # Core features: indices tf_idx*N_CORE .. (tf_idx+1)*N_CORE
                core_start = tf_idx * N_CORE
                grids[i, 0, tf_idx, :N_CORE] = feat[core_start:core_start + N_CORE]
                # Helper features
                helper_start = N_CORE * N_TFS + tf_idx * N_HELPER
                grids[i, 0, tf_idx, N_CORE:N_CORE + N_HELPER] = feat[helper_start:helper_start + N_HELPER]
        return torch.tensor(grids)

    def __len__(self):
        return len(self.entry_labels)

    def __getitem__(self, idx):
        return (self.grids[idx],
                self.entry_labels[idx],
                self.direction_labels[idx],
                self.duration_labels[idx])


class EntryDirectionNet(nn.Module):
    """Shared backbone + 3 classification heads."""

    def __init__(self):
        super().__init__()
        # Shared backbone (conv on 6×15 grid)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Linear(32, 64)

        # Head A: Entry gate (binary: good/bad)
        self.entry_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        # Head B: Direction (binary: short/long)
        self.direction_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        # Head C: Duration (3-class: SHORT/MEDIUM/LONG)
        self.duration_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        # Shared backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = F.relu(self.shared_fc(x))

        # 3 heads
        entry_logits = self.entry_head(x)
        direction_logits = self.direction_head(x)
        duration_logits = self.duration_head(x)

        return entry_logits, direction_logits, duration_logits

    def predict_proba(self, x):
        """Get probabilities from all 3 heads."""
        entry_l, dir_l, dur_l = self.forward(x)
        return (F.softmax(entry_l, dim=1),
                F.softmax(dir_l, dim=1),
                F.softmax(dur_l, dim=1))


def train_one_tier(tier: str, entry_df: pd.DataFrame, direction_df: pd.DataFrame,
                   duration_df: pd.DataFrame, val_split: float = 0.2):
    """Train the 3-head model for one tier. Walk-forward split."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n = len(entry_df)
    if n < 50:
        print(f'  {tier}: {n} trades — too few, skipping')
        return None

    # Extract features
    features = np.array(entry_df['entry_79d'].tolist(), dtype=np.float32)
    entry_labels = entry_df['label'].values
    dir_labels = direction_df['label'].values
    dur_labels = duration_df['label'].values

    # Walk-forward split (time-based, not random)
    split_idx = int(n * (1 - val_split))
    train_ds = EntryDirectionDataset(
        features[:split_idx], entry_labels[:split_idx],
        dir_labels[:split_idx], dur_labels[:split_idx])
    val_ds = EntryDirectionDataset(
        features[split_idx:], entry_labels[split_idx:],
        dir_labels[split_idx:], dur_labels[split_idx:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = EntryDirectionNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    pbar = tqdm(range(EPOCHS), desc=f'    {tier}', unit='ep', leave=False)
    for epoch in pbar:
        # Train
        model.train()
        train_loss = 0
        for batch in train_dl:
            grid, y_entry, y_dir, y_dur = [b.to(DEVICE) for b in batch]
            entry_l, dir_l, dur_l = model(grid)

            loss = (F.cross_entropy(entry_l, y_entry)
                    + F.cross_entropy(dir_l, y_dir)
                    + F.cross_entropy(dur_l, y_dur))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        correct = {'entry': 0, 'direction': 0, 'duration': 0}
        total = 0
        with torch.no_grad():
            for batch in val_dl:
                grid, y_entry, y_dir, y_dur = [b.to(DEVICE) for b in batch]
                entry_l, dir_l, dur_l = model(grid)

                val_loss += (F.cross_entropy(entry_l, y_entry)
                             + F.cross_entropy(dir_l, y_dir)
                             + F.cross_entropy(dur_l, y_dur)).item()

                correct['entry'] += (entry_l.argmax(1) == y_entry).sum().item()
                correct['direction'] += (dir_l.argmax(1) == y_dir).sum().item()
                correct['duration'] += (dur_l.argmax(1) == y_dur).sum().item()
                total += len(y_entry)

        val_loss /= max(len(val_dl), 1)
        scheduler.step(val_loss)

        # Update progress bar
        ae = correct['entry'] / max(total, 1) * 100
        ad = correct['direction'] / max(total, 1) * 100
        adur = correct['duration'] / max(total, 1) * 100
        pbar.set_postfix_str(f'e={ae:.0f}% d={ad:.0f}% dur={adur:.0f}% vl={val_loss:.2f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Restore best
    model.load_state_dict(best_state)

    # Report
    acc_e = correct['entry'] / max(total, 1) * 100
    acc_d = correct['direction'] / max(total, 1) * 100
    acc_dur = correct['duration'] / max(total, 1) * 100
    print(f'  {tier}: entry={acc_e:.0f}% dir={acc_d:.0f}% dur={acc_dur:.0f}% '
          f'(val {split_idx}/{n}, {epoch+1} epochs)')

    # Save
    tier_dir = os.path.join(OUTPUT_DIR, tier)
    os.makedirs(tier_dir, exist_ok=True)
    model_path = os.path.join(tier_dir, 'entry_direction.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'tier': tier,
        'n_train': split_idx,
        'n_val': n - split_idx,
        'val_acc': {'entry': acc_e, 'direction': acc_d, 'duration': acc_dur},
        'epochs': epoch + 1,
        'grid_shape': (GRID_ROWS, GRID_COLS),
    }, model_path)
    print(f'    Saved: {model_path}')

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tier', type=str, default=None, help='Train single tier')
    return p.parse_args()


def main():
    """Train entry/direction/duration CNNs for all tiers."""
    import pickle
    args = parse_args()

    # Load trades + regret
    trades_path = 'training_v2/output/trades/blended_is.pkl'
    regret_path = 'training_v2/output/nn/regret_analysis.csv'

    if not os.path.exists(trades_path) or not os.path.exists(regret_path):
        print(f'Missing: {trades_path} or {regret_path}')
        print(f'Run: python training/run.py blended --from 1 --to 2')
        return

    with open(trades_path, 'rb') as f:
        trades = pickle.load(f)
    regret_df = pd.read_csv(regret_path)

    print(f'Loaded {len(trades)} trades, {len(regret_df)} regret rows')

    # Generate labels
    from training.physics_labels import generate_all_labels
    print(f'\nGenerating per-tier labels...')
    tier_labels = generate_all_labels(trades, regret_df)

    # Train per tier
    print(f'\nTraining per-tier models...')
    for tier, labels in tier_labels.items():
        if args.tier and tier != args.tier:
            continue
        if not labels['trainable']:
            continue

        train_one_tier(
            tier,
            labels['entry'],
            labels['direction'],
            labels['duration'])


if __name__ == '__main__':
    main()
