"""
CNN Flip Predictor — V2-NATIVE.

Reads the full V2 layered feature snapshot (185D = L0 + 8 TFs × 23) saved on
each trade by nightmare_blended.py, reshapes the per-TF block into an 8×23
grid, and adds the regime_2d label as a 4-dim embedding side input.

Why V2-native:
  - V1-shape (6×13) collapses 8 TFs into 6 and 23 V2 concepts into 13 — that's
    half the V2 information thrown away before the CNN even sees it.
  - V2 has dedicated rows for 5s and 4h (not in V1) and dedicated columns for
    swing_noise, vol_velocity, vwap, sigma — all of which carry signal per the
    EDA stack (D1-D9) and the MA-alignment / regime-conditional findings.
  - Regime carries direction in the chord/triplet results, so feeding regime
    directly to the head is cheaper than asking conv layers to infer it from
    raw features.

Grid: (8 TFs, 23 features) = 184 floats per bar
Side: time_of_day (1) + regime_2d embedding (4) + tier (1) = 6 floats
Path: 5×8×23 (entry, 25%, 50%, 75%, exit) — same as V1 path but on V2 grid.

Output: 2 classes (SAME=0, COUNTER=1).

Usage:
    python training_v2/cnn_flip.py
    python training_v2/cnn_flip.py --no-path
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_v2.sfe_ticker import (V2_COLUMNS, N_V2_TOTAL,
                                          V2_TFS, V2_PER_TF_FEATS,
                                          REGIME_VOCAB)

BLENDED_TRADES = os.environ.get('CNN_TRADES_PATH', 'training_v2/output/trades/blended_is.pkl')
REGRET_FILE = os.environ.get('CNN_REGRET_PATH', 'training_v2/output/nn/regret_analysis.csv')
OUTPUT_DIR = os.environ.get('CNN_OUTPUT_DIR', 'training_v2/output/nn')

GRID_H = len(V2_TFS)               # 8 TFs
GRID_W = len(V2_PER_TF_FEATS)      # 23 features per TF
N_REGIMES = len(REGIME_VOCAB)      # 7 (UNKNOWN + 6 labels)
REGIME_EMBED = 4
N_TIME_POINTS = 5

TIER_MAP = {
    'CASCADE': 0, 'KILL_SHOT': 1, 'FREIGHT_TRAIN': 2,
    'FADE_AGAINST': 3, 'RIDE_AGAINST': 4, 'RIDE_MOMENTUM': 5,
    'RIDE_CALM': 6, 'FADE_MOMENTUM': 7, 'FADE_CALM': 8,
    'MTF_EXHAUSTION': 9, 'MTF_BREAKOUT': 10,
    'BASE_NMP': 11, 'REGIME_FLIP': 12, 'EXHAUSTION_BAR': 13,
    'ABSORPTION': 14,
}

# L0 column lives at index 0 of V2_COLUMNS; remaining 184 are the 8×23 block
# laid out as: for tf in V2_TFS: for feat in (L1+L2+L3): col_name
L0_IDX = V2_COLUMNS.index('L0_time_of_day')
GRID_FLAT_IDX = [i for i in range(N_V2_TOTAL) if i != L0_IDX]
assert len(GRID_FLAT_IDX) == GRID_H * GRID_W, \
    f'V2 grid index mismatch: {len(GRID_FLAT_IDX)} vs {GRID_H * GRID_W}'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='V2-native CNN flip predictor')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--no-path', action='store_true')
    return p.parse_args()


def v2_to_grid(v2_185):
    """Reshape 185D V2 vector → (8, 23) per-TF grid + L0 scalar.

    V2_COLUMNS layout: [L0_time_of_day] then for tf in V2_TFS:
        for feat in (L1_FEATS + L2_FEATS + L3_FEATS): col_name
    Picking GRID_FLAT_IDX in canonical order yields TF-major rows.
    """
    arr = np.asarray(v2_185, dtype=np.float32)
    if arr.shape[0] != N_V2_TOTAL:
        return None, None
    tod = float(arr[L0_IDX])
    grid = arr[GRID_FLAT_IDX].reshape(GRID_H, GRID_W)
    return grid, tod


def build_dataset(use_path=True):
    """Build training dataset from blended trades + regret.

    Reads entry_v2 (185D), entry_regime_idx, and per-path-frame v2_features
    saved by the V2-native engine. Skips trades that lack entry_v2 (e.g.,
    trades opened before the migration or chain contracts that didn't snapshot V2).
    """
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)
    regret = pd.read_csv(REGRET_FILE)

    grids = []        # (N, 8, 23) entry V2 grid
    paths = []        # (N, 5, 8, 23) path V2 volumes
    tods = []         # (N,) time of day from L0
    regimes = []      # (N,) int regime code
    labels = []       # (N,) SAME=0 / COUNTER=1
    tiers = []        # (N,) int tier code
    pnls = []         # (N,) realized PnL

    for i, t in enumerate(trades):
        ev2 = t.get('entry_v2', None)
        if ev2 is None or len(ev2) != N_V2_TOTAL or i >= len(regret):
            continue
        grid, tod = v2_to_grid(ev2)
        if grid is None:
            continue

        r = regret.iloc[i]
        label = 1 if 'counter' in r['best_action'] else 0

        regime_idx = int(t.get('entry_regime_idx', 0))
        tier_idx = TIER_MAP.get(t.get('entry_tier', 'BASE_NMP'), 11)

        grids.append(grid)
        tods.append(tod)
        regimes.append(regime_idx)
        labels.append(label)
        tiers.append(tier_idx)
        pnls.append(t['pnl'])

        if use_path:
            path = t.get('path', [])
            path_vol = np.zeros((N_TIME_POINTS, GRID_H, GRID_W), dtype=np.float32)
            path_vol[0] = grid
            if path and len(path) >= 6:
                indices = [0, len(path)//4, len(path)//2,
                              3*len(path)//4, len(path)-1]
                for pi, pidx in enumerate(indices[1:], 1):
                    pv2 = path[pidx].get('v2_features', None)
                    if pv2 is not None and len(pv2) == N_V2_TOTAL:
                        pgrid, _ = v2_to_grid(pv2)
                        if pgrid is not None:
                            path_vol[pi] = pgrid
                            continue
                    path_vol[pi] = grid
            else:
                for pi in range(1, N_TIME_POINTS):
                    path_vol[pi] = grid
            paths.append(path_vol)

    grids = np.array(grids)
    tods = np.array(tods, dtype=np.float32)
    regimes = np.array(regimes, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    tiers = np.array(tiers, dtype=np.float32)
    pnls = np.array(pnls, dtype=np.float32)
    paths = np.array(paths) if use_path else None

    print(f'Dataset: {len(grids)} trades | SAME: {(labels==0).sum()} '
          f'| COUNTER: {(labels==1).sum()}')
    print(f'Regimes: {dict(zip(*np.unique(regimes, return_counts=True)))}')
    return grids, paths, tods, regimes, labels, tiers, pnls


class FlipDataset(Dataset):
    def __init__(self, grids, paths, tods, regimes, labels, tiers):
        self.grids = torch.FloatTensor(grids).unsqueeze(1)  # (N, 1, 8, 23)
        self.tods = torch.FloatTensor(tods).unsqueeze(1)    # (N, 1)
        self.regimes = torch.LongTensor(regimes)            # (N,)
        self.labels = torch.LongTensor(labels)              # (N,)
        self.tiers = torch.FloatTensor(tiers).unsqueeze(1)  # (N, 1)
        if paths is not None:
            self.paths = torch.FloatTensor(paths)           # (N, 5, 8, 23)
            self.use_path = True
        else:
            self.paths = None
            self.use_path = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_path:
            return (self.grids[idx], self.paths[idx], self.tods[idx],
                    self.regimes[idx], self.tiers[idx], self.labels[idx])
        return (self.grids[idx], self.tods[idx], self.regimes[idx],
                self.tiers[idx], self.labels[idx])


class V2FlipCNN(nn.Module):
    """V2-native flip CNN.

    Input:
      entry  : (B, 1, 8, 23) per-TF V2 grid
      path   : (B, 5, 8, 23) optional 5-point path
      tod    : (B, 1) L0 time-of-day scalar
      regime : (B,) int code (0..N_REGIMES-1), embedded to 4D
      tier   : (B, 1) int tier code as float
    """

    def __init__(self, use_path=True):
        super().__init__()
        self.use_path = use_path

        self.entry_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # (64, 4, 8) = 2048
        )
        entry_flat = 64 * 4 * 8

        if use_path:
            self.path_conv = nn.Sequential(
                nn.Conv2d(N_TIME_POINTS, 32, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 8)),
            )
            path_flat = 64 * 4 * 8
        else:
            path_flat = 0

        self.regime_embed = nn.Embedding(N_REGIMES, REGIME_EMBED)
        # entry + path + regime_embed + tod(1) + tier(1)
        total_flat = entry_flat + path_flat + REGIME_EMBED + 2

        self.classifier = nn.Sequential(
            nn.Linear(total_flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, entry, tod, regime, tier, path=None):
        e = self.entry_conv(entry)
        e = e.view(e.size(0), -1)
        r = self.regime_embed(regime)  # (B, 4)
        if self.use_path and path is not None:
            p = self.path_conv(path)
            p = p.view(p.size(0), -1)
            x = torch.cat([e, p, r, tod, tier], dim=1)
        else:
            x = torch.cat([e, r, tod, tier], dim=1)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        if len(batch) == 6:
            entry, path, tod, regime, tier, labels = batch
            entry, path = entry.to(device), path.to(device)
            tod, regime, tier, labels = (tod.to(device), regime.to(device),
                                              tier.to(device), labels.to(device))
            out = model(entry, tod, regime, tier, path=path)
        else:
            entry, tod, regime, tier, labels = batch
            entry, tod, regime, tier, labels = (entry.to(device), tod.to(device),
                                                       regime.to(device), tier.to(device),
                                                       labels.to(device))
            out = model(entry, tod, regime, tier, path=None)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(dim=1) == labels).sum().item()
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
            if len(batch) == 6:
                entry, path, tod, regime, tier, labels = batch
                entry, path = entry.to(device), path.to(device)
                tod, regime, tier, labels = (tod.to(device), regime.to(device),
                                                  tier.to(device), labels.to(device))
                out = model(entry, tod, regime, tier, path=path)
            else:
                entry, tod, regime, tier, labels = batch
                entry, tod, regime, tier, labels = (entry.to(device), tod.to(device),
                                                           regime.to(device), tier.to(device),
                                                           labels.to(device))
                out = model(entry, tod, regime, tier, path=None)
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
    print(f'Path data: {"YES (5-point V2 curve)" if use_path else "NO (entry only)"}')

    grids, paths, tods, regimes, labels, tiers, pnls = build_dataset(use_path=use_path)

    if len(grids) == 0:
        print('No V2 trades found — engine must save entry_v2 (re-run engine first).')
        return

    grid_mean = grids.mean(axis=0, keepdims=True)
    grid_std = grids.std(axis=0, keepdims=True).clip(min=1e-8)
    grids = (grids - grid_mean) / grid_std

    if paths is not None:
        path_mean = paths.mean(axis=0, keepdims=True)
        path_std = paths.std(axis=0, keepdims=True).clip(min=1e-8)
        paths = (paths - path_mean) / path_std

    tod_mean = tods.mean()
    tod_std = max(tods.std(), 1e-8)
    tods = (tods - tod_mean) / tod_std

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    fold_states = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(grids)):
        print(f'\n--- Fold {fold+1}/5 ---')

        train_ds = FlipDataset(
            grids[train_idx],
            paths[train_idx] if paths is not None else None,
            tods[train_idx], regimes[train_idx],
            labels[train_idx], tiers[train_idx])
        val_ds = FlipDataset(
            grids[val_idx],
            paths[val_idx] if paths is not None else None,
            tods[val_idx], regimes[val_idx],
            labels[val_idx], tiers[val_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = V2FlipCNN(use_path=use_path).to(device)
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
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:>3}: train_acc={train_acc:.1%} '
                      f'val_acc={val_acc:.1%} train_loss={train_loss:.4f} '
                      f'val_loss={val_loss:.4f}')

        model.load_state_dict(best_model_state)
        _, _, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)

        val_pnl = pnls[val_idx]
        kept_mask = val_preds == 0
        kept_pnl = val_pnl[kept_mask].sum()
        skip_pnl = val_pnl[~kept_mask].sum()
        print(f'  Best val_acc: {best_val_acc:.1%}')
        print(f'  Kept (pred SAME): {kept_mask.sum()} trades, ${kept_pnl:,.0f} '
              f'(${kept_pnl/max(kept_mask.sum(),1):.1f}/trade)')
        print(f'  Skipped (pred COUNTER): {(~kept_mask).sum()} trades, '
              f'${skip_pnl:,.0f} (${skip_pnl/max((~kept_mask).sum(),1):.1f}/trade)')

        fold_states.append(best_model_state)
        fold_results.append({
            'fold': fold, 'val_acc': best_val_acc,
            'kept_n': int(kept_mask.sum()), 'kept_pnl': float(kept_pnl),
            'skip_n': int((~kept_mask).sum()), 'skip_pnl': float(skip_pnl),
        })

    print(f'\n{"="*60}')
    print(f'V2-NATIVE CNN FLIP PREDICTOR RESULTS')
    print(f'{"="*60}')
    avg_acc = float(np.mean([r['val_acc'] for r in fold_results]))
    avg_kept = float(np.mean([r['kept_pnl'] for r in fold_results]))
    avg_skip = float(np.mean([r['skip_pnl'] for r in fold_results]))
    avg_kept_n = float(np.mean([r['kept_n'] for r in fold_results]))
    avg_skip_n = float(np.mean([r['skip_n'] for r in fold_results]))
    print(f'  CV Accuracy: {avg_acc:.1%}')
    print(f'  Kept (pred SAME): {avg_kept_n:.0f} trades, ${avg_kept:,.0f} '
          f'(${avg_kept/max(avg_kept_n,1):.1f}/trade)')
    print(f'  Skipped (pred COUNTER): {avg_skip_n:.0f} trades, ${avg_skip:,.0f} '
          f'(${avg_skip/max(avg_skip_n,1):.1f}/trade)')
    print(f'  Path used: {use_path}')
    print(f'  Grid: {GRID_H} TFs × {GRID_W} V2 features (vs old 6×13 V1)')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'cnn_flip.pt')
    torch.save({
        'ensemble': fold_states,
        'n_models': len(fold_states),
        'model_state': fold_states[0] if fold_states else None,
        'use_path': use_path,
        'grid_mean': grid_mean,
        'grid_std': grid_std,
        'path_mean': path_mean if paths is not None else None,
        'path_std': path_std if paths is not None else None,
        'tod_mean': tod_mean,
        'tod_std': tod_std,
        'fold_results': fold_results,
        'accuracy': avg_acc,
        'v2_native': True,
        'grid_shape': (GRID_H, GRID_W),
        'n_regimes': N_REGIMES,
    }, save_path)
    print(f'\nEnsemble saved: {save_path} ({len(fold_states)} models)')


if __name__ == '__main__':
    main()
