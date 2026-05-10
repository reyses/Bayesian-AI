"""
CNN Trade Manager — exit timing + loser identification.

Trained PER TIER on oracle labels. Runs every 5s bar while in position:
  Head D (exit):     P(EXIT) — is this trade done?
  Head E (loser ID): P(DEAD) — is this trade failing? (only when pnl < 0)

Input: 91D current features (flat) + trade context (bars_held, pnl, peak_pnl, entry_z).
Output: 2 probability distributions.

Usage:
    python training/cnn_trade_manager.py                    # train all tiers
    python training/cnn_trade_manager.py --tier CASCADE     # single tier
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import N_FEATURES

OUTPUT_DIR = 'training_v2/output/nn'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Context features appended to 91D
N_CONTEXT = 4  # bars_held, pnl, peak_pnl, entry_z
N_INPUT = N_FEATURES + N_CONTEXT  # 91 + 4 = 95

# Training
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
SEED = 42


class TradeManagerDataset(Dataset):
    """Dataset for during-trade predictions (exit + loser ID)."""

    def __init__(self, features: np.ndarray, context: np.ndarray,
                 exit_labels: np.ndarray, loser_labels: np.ndarray,
                 loser_mask: np.ndarray):
        """
        Args:
            features: (N, 91) 91D features per bar
            context: (N, 4) [bars_held, pnl, peak_pnl, entry_z]
            exit_labels: (N,) HOLD=1, EXIT=0
            loser_labels: (N,) RECOVER=1, DEAD=0
            loser_mask: (N,) 1 if pnl < 0 (loser head active), 0 otherwise
        """
        combined = np.concatenate([features, context], axis=1).astype(np.float32)
        self.x = torch.tensor(combined)
        self.exit_labels = torch.tensor(exit_labels, dtype=torch.long)
        self.loser_labels = torch.tensor(loser_labels, dtype=torch.long)
        self.loser_mask = torch.tensor(loser_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.exit_labels)

    def __getitem__(self, idx):
        return (self.x[idx],
                self.exit_labels[idx],
                self.loser_labels[idx],
                self.loser_mask[idx])


class TradeManagerNet(nn.Module):
    """Shared backbone + 2 heads for during-trade decisions."""

    def __init__(self, input_dim=N_INPUT):
        super().__init__()
        # Shared backbone (MLP — no spatial structure in during-trade features)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        # Head D: Exit (HOLD/EXIT)
        self.exit_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        # Head E: Loser ID (RECOVER/DEAD)
        self.loser_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        shared = self.backbone(x)
        exit_logits = self.exit_head(shared)
        loser_logits = self.loser_head(shared)
        return exit_logits, loser_logits

    def predict_proba(self, x):
        exit_l, loser_l = self.forward(x)
        return F.softmax(exit_l, dim=1), F.softmax(loser_l, dim=1)


def train_one_tier(tier: str, exit_df: pd.DataFrame, loser_df: pd.DataFrame,
                   val_split: float = 0.2):
    """Train exit + loser ID for one tier."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build combined dataset from exit bars
    # Loser bars are a subset — we use loser_mask to only backprop loser head on those
    if len(exit_df) < 100:
        print(f'  {tier}: {len(exit_df)} exit bars — too few, skipping')
        return None

    features = np.array(exit_df['features'].tolist(), dtype=np.float32)
    context = np.column_stack([
        exit_df['bars_held'].values.astype(np.float32) / 60.0,  # normalize to hours
        exit_df['pnl'].values.astype(np.float32) / 100.0,        # normalize to $100 units
        exit_df['peak_pnl'].values.astype(np.float32) / 100.0,
        exit_df['entry_z'].values.astype(np.float32),
    ])
    exit_labels = exit_df['label'].values

    # Loser labels: match by (trade_idx, bar_idx) — loser_df is a subset of exit_df
    loser_labels = np.zeros(len(exit_df), dtype=np.int64)
    loser_mask = np.zeros(len(exit_df), dtype=np.float32)

    if len(loser_df) > 0:
        loser_lookup = {}
        for _, row in loser_df.iterrows():
            key = (int(row['trade_idx']), int(row['bar_idx']))
            loser_lookup[key] = int(row['label'])

        for i, (_, row) in enumerate(exit_df.iterrows()):
            key = (int(row['trade_idx']), int(row['bar_idx']))
            if key in loser_lookup:
                loser_labels[i] = loser_lookup[key]
                loser_mask[i] = 1.0

    # Walk-forward split
    n = len(exit_df)
    split_idx = int(n * (1 - val_split))

    train_ds = TradeManagerDataset(
        features[:split_idx], context[:split_idx],
        exit_labels[:split_idx], loser_labels[:split_idx], loser_mask[:split_idx])
    val_ds = TradeManagerDataset(
        features[split_idx:], context[split_idx:],
        exit_labels[split_idx:], loser_labels[split_idx:], loser_mask[split_idx:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TradeManagerNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(EPOCHS), desc=f'    {tier}', unit='ep', leave=False)
    for epoch in pbar:
        model.train()
        train_loss_sum = 0
        for batch in train_dl:
            x, y_exit, y_loser, mask = [b.to(DEVICE) for b in batch]
            exit_l, loser_l = model(x)

            # Exit loss: all bars
            loss_exit = F.cross_entropy(exit_l, y_exit)

            # Loser loss: only bars where pnl < 0 (masked)
            if mask.sum() > 0:
                loss_loser = F.cross_entropy(loser_l, y_loser, reduction='none')
                loss_loser = (loss_loser * mask).sum() / mask.sum()
            else:
                loss_loser = torch.tensor(0.0, device=DEVICE)

            loss = loss_exit + loss_loser

            train_loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        correct_exit = 0
        correct_loser = 0
        total = 0
        total_loser = 0
        with torch.no_grad():
            for batch in val_dl:
                x, y_exit, y_loser, mask = [b.to(DEVICE) for b in batch]
                exit_l, loser_l = model(x)

                val_loss += F.cross_entropy(exit_l, y_exit).item()
                correct_exit += (exit_l.argmax(1) == y_exit).sum().item()
                total += len(y_exit)

                if mask.sum() > 0:
                    loser_preds = loser_l.argmax(1)
                    correct_loser += ((loser_preds == y_loser) * mask.bool()).sum().item()
                    total_loser += int(mask.sum().item())

        val_loss /= max(len(val_dl), 1)
        scheduler.step(val_loss)

        # Update progress bar
        acc_e = correct_exit / max(total, 1) * 100
        acc_l = correct_loser / max(total_loser, 1) * 100
        pbar.set_postfix_str(f'exit={acc_e:.0f}% loser={acc_l:.0f}% vl={val_loss:.3f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)

    acc_exit = correct_exit / max(total, 1) * 100
    acc_loser = correct_loser / max(total_loser, 1) * 100
    print(f'  {tier}: exit={acc_exit:.0f}% loser={acc_loser:.0f}% '
          f'({total} val bars, {total_loser} loser bars, {epoch+1} epochs)')

    # Save
    tier_dir = os.path.join(OUTPUT_DIR, tier)
    os.makedirs(tier_dir, exist_ok=True)
    model_path = os.path.join(tier_dir, 'trade_manager.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'tier': tier,
        'n_train': split_idx,
        'n_val': n - split_idx,
        'val_acc': {'exit': acc_exit, 'loser': acc_loser},
        'epochs': epoch + 1,
        'input_dim': N_INPUT,
    }, model_path)
    print(f'    Saved: {model_path}')

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tier', type=str, default=None)
    return p.parse_args()


def main():
    """Train exit/loser CNNs for all tiers."""
    import pickle
    args = parse_args()

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

    from training.physics_labels import generate_all_labels
    print(f'\nGenerating per-tier labels...')
    tier_labels = generate_all_labels(trades, regret_df)

    print(f'\nTraining per-tier trade managers...')
    for tier, labels in tier_labels.items():
        if args.tier and tier != args.tier:
            continue
        if not labels['trainable']:
            continue

        train_one_tier(tier, labels['exit'], labels['loser'])


if __name__ == '__main__':
    main()
