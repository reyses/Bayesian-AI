"""
Train StatePredictor per timeframe — grounded in feature output.

Predicts 7D future state (dmi_diff, dmi_gap, vol_rel, dir_vol, velocity, z_se, price_accel).
Direction, confidence, and reliability are DERIVED from predictions, not direct outputs:
  - Direction = sign(predicted dmi_diff)
  - Confidence = |predicted dmi_diff| (magnitude = conviction)
  - Reliability = when confidence > threshold, accuracy of direction

Gate: 95% direction accuracy when confidence is in the top 5% of predictions.
If the model can't achieve this, the TF doesn't contribute.

Usage:
  python -m training.train_direction --tf 1h
  python -m training.train_direction --tf 1m
  python -m training.train_direction --tf 1s --max-bars 500000
"""
import argparse
import gc
import glob
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

TICK = 0.25
LOOKBACK = 10
ATLAS_ROOT = 'DATA/ATLAS'

# Direction label: price change over forward window
FORWARD_BARS = {
    '1h': 1,    # 1 hour ahead (structural)
    '1m': 4,    # 4 minutes ahead (half-cycle of ~8 min oscillation)
    '1s': 5,    # 5 seconds ahead (micro half-cycle)
}

FEATURE_NAMES_7D = ['dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel']
FEATURE_NAMES_13D = FEATURE_NAMES_7D + [
    'std_price', 'variance_ratio', 'bar_range', 'wick_ratio',
    'vwap_distance', 'time_of_day',
]
N_FEAT_7D = len(FEATURE_NAMES_7D)


def extract_features_13d(states, df):
    """Extract 13D features — same as train_trade_cnn.py."""
    from training.train_trade_cnn import extract_features_13d as _extract
    return _extract(states, df)


def build_state_labels(feats_7d, forward_bars):
    """Labels = actual 7D features at t+forward_bars."""
    n = len(feats_7d)
    labels = np.zeros((n, N_FEAT_7D), dtype=np.float32)
    for i in range(n - forward_bars):
        labels[i] = feats_7d[i + forward_bars]
    return labels


class StateDataset(Dataset):
    def __init__(self, features, labels, lookback=LOOKBACK, forward=10):
        self.features = features
        self.labels = labels
        self.lookback = lookback
        self.n = len(features) - lookback - forward

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        i = idx + self.lookback
        x = self.features[i - self.lookback:i]  # (lookback, 13)
        y = self.labels[i]                        # (7,)
        return torch.FloatTensor(x), torch.FloatTensor(y)


def train_and_validate(tf, max_bars=0, epochs_per_day=10):
    """Full pipeline: load data, extract features, train DualHeadPredictor, evaluate."""
    from core.statistical_field_engine import StatisticalFieldEngine
    from core.direction_cnn import DualHeadPredictor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward = FORWARD_BARS[tf]
    ckpt_dir = f'checkpoints/direction_{tf}'
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"DUAL HEAD PREDICTOR: {tf.upper()} (7D state + P(long) at t+{forward})")
    print(f"{'=' * 60}")

    # Load data
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
    if not files:
        print(f"No data in {ATLAS_ROOT}/{tf}/")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    if max_bars > 0:
        df = df.tail(max_bars).reset_index(drop=True)
    print(f"  Bars: {len(df):,}")

    # Features + labels (cached per shard)
    feat_cache = os.path.join(ckpt_dir, 'features_13d.npy')
    label_cache = os.path.join(ckpt_dir, 'labels_7d.npy')

    if os.path.exists(feat_cache) and os.path.exists(label_cache):
        print("  Loading cached features + labels...")
        feats = np.load(feat_cache)
        labels = np.load(label_cache)
        if len(feats) != len(df):
            print(f"  Cache mismatch — rebuilding")
            feats = None
    else:
        feats = None

    if feats is None:
        print("  Computing SFE states...")
        sfe = StatisticalFieldEngine()
        states = sfe.batch_compute_states(df)

        print("  Extracting 13D features...")
        feats = extract_features_13d(states, df)
        del states; gc.collect()

        print("  Building 7D state labels...")
        labels = build_state_labels(feats[:, :N_FEAT_7D], forward)

        np.save(feat_cache, feats)
        np.save(label_cache, labels)
        print(f"  Saved: {feat_cache}, {label_cache}")

    prices = df['close'].values
    print(f"  Features: {feats.shape}, Labels: {labels.shape}")

    # Day boundaries
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    day_boundaries = []
    for date, group in df.groupby('date'):
        day_boundaries.append({'date': date, 'start': group.index[0], 'end': group.index[-1]})
    print(f"  Trading days: {len(day_boundaries)}")

    # Model: DualHeadPredictor — 7D state + P(long)
    model = DualHeadPredictor(n_features=13, latent_dim=64, n_state=N_FEAT_7D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    DIR_LOSS_WEIGHT = 0.1  # state prediction dominates, direction is a nudge

    day_results = []
    _epochs_per_day = epochs_per_day

    for di, day in enumerate(tqdm(day_boundaries, desc=f"Walk-Forward {tf.upper()}")):
        _start = day['start']
        _end = day['end']
        n_bars = _end - _start + 1

        if n_bars < LOOKBACK + forward + 5:
            continue

        day_feats = feats[_start:_end + 1]
        day_labels = labels[_start:_end + 1]

        # Score BEFORE training
        if di > 0:
            model.eval()
            ds = StateDataset(day_feats, day_labels, lookback=LOOKBACK, forward=forward)
            if len(ds) > 0:
                dl = DataLoader(ds, batch_size=512, shuffle=False)
                all_state = []
                all_plong = []
                all_true = []
                with torch.no_grad():
                    for x, y in dl:
                        x = x.to(device)
                        state_pred, p_long = model(x)
                        all_state.append(state_pred.cpu().numpy())
                        all_plong.append(p_long.cpu().numpy())
                        all_true.append(y.numpy())

                preds = np.concatenate(all_state)
                p_longs = np.concatenate(all_plong)
                trues = np.concatenate(all_true)

                # Direction from BOTH heads
                state_dir = preds[:, 0] > 0     # state head: dmi_diff sign
                class_dir = p_longs > 0.5        # direction head: P(long) > 0.5
                true_dir = trues[:, 0] > 0       # actual dmi_diff sign

                # Accuracy from each head
                state_acc = (state_dir == true_dir).mean() * 100
                class_acc = (class_dir == true_dir).mean() * 100
                # Agreement between heads
                agreement = (state_dir == class_dir).mean() * 100
                # Combined: only count when both heads agree AND correct
                both_agree = state_dir == class_dir
                if both_agree.sum() > 0:
                    agreed_acc = (state_dir[both_agree] == true_dir[both_agree]).mean() * 100
                else:
                    agreed_acc = 0
                acc = class_acc  # primary metric from classification head

                # Confidence from classification head
                confidence = np.abs(p_longs - 0.5) * 2

                # Reliability at various confidence percentiles
                results_row = {
                    'date': str(day['date']),
                    'day': di + 1,
                    'n_bars': n_bars,
                    'accuracy': acc,
                    'state_acc': state_acc,
                    'class_acc': class_acc,
                    'agreement': agreement,
                    'agreed_acc': agreed_acc,
                    'avg_confidence': confidence.mean(),
                    'n_predictions': len(preds),
                }

                # Top 5%/10%/25% most confident predictions
                for pct, label in [(95, 'top5'), (90, 'top10'), (75, 'top25')]:
                    threshold = np.percentile(confidence, pct)
                    mask = confidence >= threshold
                    if mask.sum() > 0:
                        pct_acc = (class_dir[mask] == true_dir[mask]).mean() * 100
                        pct_coverage = mask.mean() * 100
                    else:
                        pct_acc = 0
                        pct_coverage = 0
                    results_row[f'{label}_acc'] = pct_acc
                    results_row[f'{label}_coverage'] = pct_coverage
                    results_row[f'{label}_threshold'] = threshold

                # Feature correlation (how well does each feature predict?)
                from scipy import stats as sp_stats
                corrs = []
                for j in range(N_FEAT_7D):
                    if trues[:, j].std() > 1e-8 and preds[:, j].std() > 1e-8:
                        r, _ = sp_stats.spearmanr(preds[:, j], trues[:, j])
                        corrs.append(r)
                    else:
                        corrs.append(0)
                results_row['avg_corr'] = np.mean(corrs)

                day_results.append(results_row)

                if (di + 1) % 30 == 0:
                    r = results_row
                    print(f"  Day {di+1}: acc={r['accuracy']:.1f}% corr={r['avg_corr']:.3f} | "
                          f"top5%: {r['top5_acc']:.1f}% | "
                          f"top10%: {r['top10_acc']:.1f}% | "
                          f"top25%: {r['top25_acc']:.1f}%")

        # Train on this day
        model.train()
        ds = StateDataset(day_feats, day_labels, lookback=LOOKBACK, forward=forward)
        if len(ds) < 10:
            continue
        dl = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)

        _epochs = 30 if di == 0 else _epochs_per_day
        _lr = 1e-3 if di == 0 else 1e-4
        for pg in optimizer.param_groups:
            pg['lr'] = _lr

        for _ in range(_epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                state_pred, p_long = model(x)

                # State loss: MSE on 7D features
                loss_state = mse_fn(state_pred, y)

                # Direction loss: BCE on P(long) vs actual direction
                dir_label = (y[:, 0] > 0).float()  # dmi_diff > 0 = long
                loss_dir = bce_fn(p_long, dir_label)

                loss = loss_state + DIR_LOSS_WEIGHT * loss_dir
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Checkpoint every 30 days
        if (di + 1) % 30 == 0:
            torch.save({
                'model_state': model.state_dict(),
                'day': di + 1, 'date': str(day['date']), 'tf': tf,
                'forward': forward,
            }, os.path.join(ckpt_dir, f'model_day{di+1}.pt'))

    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'day': len(day_boundaries), 'tf': tf,
        'date': str(day_boundaries[-1]['date']),
        'forward': forward,
    }, os.path.join(ckpt_dir, 'best_model.pt'))
    print(f"  Saved: {ckpt_dir}/best_model.pt")

    # Report
    if day_results:
        df_r = pd.DataFrame(day_results)

        print(f"\n{'=' * 60}")
        print(f"STATE PREDICTOR REPORT: {tf.upper()} (t+{forward})")
        print(f"{'=' * 60}")
        print(f"  Days scored: {len(df_r)}")
        print(f"  State head accuracy: {df_r['state_acc'].mean():.1f}%")
        print(f"  Direction head accuracy: {df_r['class_acc'].mean():.1f}%")
        print(f"  Head agreement: {df_r['agreement'].mean():.1f}%")
        print(f"  When both agree, accuracy: {df_r['agreed_acc'].mean():.1f}%")
        print(f"  Avg feature correlation: {df_r['avg_corr'].mean():.3f}")
        print(f"  Avg confidence: {df_r['avg_confidence'].mean():.2f}")
        print(f"")
        print(f"  RELIABILITY BY CONFIDENCE:")
        print(f"    Top 25% confident: {df_r['top25_acc'].mean():.1f}% accuracy")
        print(f"    Top 10% confident: {df_r['top10_acc'].mean():.1f}% accuracy")
        print(f"    Top  5% confident: {df_r['top5_acc'].mean():.1f}% accuracy")
        print(f"")
        print(f"  GATE CHECK (95% accuracy in top 5% confident):")
        _gate = df_r['top5_acc'].mean()
        if _gate >= 95:
            print(f"    PASS ({_gate:.1f}%)")
        else:
            print(f"    FAIL ({_gate:.1f}% — need 95%)")

        # Monthly
        df_r['month'] = df_r['date'].str[:7]
        print(f"\n  MONTHLY:")
        for month, grp in df_r.groupby('month'):
            print(f"    {month}: acc={grp['accuracy'].mean():.1f}% "
                  f"corr={grp['avg_corr'].mean():.3f} "
                  f"top5%={grp['top5_acc'].mean():.1f}%")

        df_r.to_csv(os.path.join(ckpt_dir, 'daily_results.csv'), index=False)
        print(f"\n  Saved: {ckpt_dir}/daily_results.csv")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_epoch_based(tf, max_bars=0, val_start='2026-01-01', n_epochs=100):
    """Train on full dataset with epoch-based training + held-out validation.

    Better for sparse TFs (1h: ~19 bars/day). Uses all data before val_start
    for training, data after val_start for validation.
    """
    from core.statistical_field_engine import StatisticalFieldEngine
    from core.direction_cnn import DualHeadPredictor
    from scipy import stats as sp_stats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward = FORWARD_BARS[tf]
    ckpt_dir = f'checkpoints/direction_{tf}'
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"DUAL HEAD (EPOCH): {tf.upper()} (t+{forward}) | train<{val_start} | val>={val_start}")
    print(f"{'=' * 60}")

    # Load data
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    if max_bars > 0:
        df = df.tail(max_bars).reset_index(drop=True)
    print(f"  Bars: {len(df):,}")

    # Features + labels
    feat_cache = os.path.join(ckpt_dir, 'features_13d.npy')
    label_cache = os.path.join(ckpt_dir, 'labels_7d.npy')

    if os.path.exists(feat_cache) and os.path.exists(label_cache):
        feats = np.load(feat_cache)
        labels = np.load(label_cache)
        if len(feats) != len(df):
            feats = None
    else:
        feats = None

    if feats is None:
        sfe = StatisticalFieldEngine()
        states = sfe.batch_compute_states(df)
        feats = extract_features_13d(states, df)
        del states; gc.collect()
        labels = build_state_labels(feats[:, :N_FEAT_7D], forward)
        np.save(feat_cache, feats)
        np.save(label_cache, labels)

    prices = df['close'].values

    # Split train/val by date
    val_ts = pd.Timestamp(val_start).timestamp()
    train_mask = df['timestamp'].values < val_ts
    val_mask = df['timestamp'].values >= val_ts

    train_feats = feats[train_mask]
    train_labels = labels[train_mask]
    val_feats = feats[val_mask]
    val_labels = labels[val_mask]
    val_prices = prices[val_mask]

    print(f"  Train: {len(train_feats):,} bars | Val: {len(val_feats):,} bars")
    print(f"  Label balance (train): {(train_labels[:, 0] > 0).mean()*100:.1f}% positive dmi_diff")

    # Model
    model = DualHeadPredictor(n_features=13, latent_dim=64, n_state=N_FEAT_7D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    DIR_LOSS_WEIGHT = 0.1

    train_ds = StateDataset(train_feats, train_labels, lookback=LOOKBACK, forward=forward)
    val_ds = StateDataset(val_feats, val_labels, lookback=LOOKBACK, forward=forward)
    print(f"  Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_val_acc = 0
    best_epoch = 0

    for epoch in tqdm(range(1, n_epochs + 1), desc=f"Epochs {tf.upper()}"):
        # Train
        model.train()
        epoch_loss = 0
        n_batches = 0
        for x, y in tqdm(train_dl, desc=f"  E{epoch} train", leave=False):
            x, y = x.to(device), y.to(device)
            state_pred, p_long = model(x)
            loss_state = mse_fn(state_pred, y)
            dir_label = (y[:, 0] > 0).float()
            loss_dir = bce_fn(p_long, dir_label)
            loss = loss_state + DIR_LOSS_WEIGHT * loss_dir
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        all_state = []
        all_plong = []
        all_true = []
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"  E{epoch} val", leave=False):
                x = x.to(device)
                s, p = model(x)
                all_state.append(s.cpu().numpy())
                all_plong.append(p.cpu().numpy())
                all_true.append(y.numpy())

        preds = np.concatenate(all_state)
        p_longs = np.concatenate(all_plong)
        trues = np.concatenate(all_true)

        state_dir = preds[:, 0] > 0
        class_dir = p_longs > 0.5
        true_dir = trues[:, 0] > 0

        state_acc = (state_dir == true_dir).mean() * 100
        class_acc = (class_dir == true_dir).mean() * 100
        both_agree = state_dir == class_dir
        agreed_acc = (state_dir[both_agree] == true_dir[both_agree]).mean() * 100 if both_agree.sum() > 0 else 0
        agreement = both_agree.mean() * 100

        # Confidence + reliability
        confidence = np.abs(p_longs - 0.5) * 2
        top5_thresh = np.percentile(confidence, 95)
        top5_mask = confidence >= top5_thresh
        top5_acc = (class_dir[top5_mask] == true_dir[top5_mask]).mean() * 100 if top5_mask.sum() > 0 else 0

        # Feature correlation
        corrs = []
        for j in range(N_FEAT_7D):
            if trues[:, j].std() > 1e-8 and preds[:, j].std() > 1e-8:
                r, _ = sp_stats.spearmanr(preds[:, j], trues[:, j])
                corrs.append(r)
        avg_corr = np.mean(corrs) if corrs else 0

        scheduler.step(epoch_loss / n_batches)
        _lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}: loss={epoch_loss/n_batches:.4f} lr={_lr:.1e} | "
                  f"state={state_acc:.1f}% class={class_acc:.1f}% agree={agreement:.0f}%->{agreed_acc:.1f}% | "
                  f"corr={avg_corr:.3f} top5%={top5_acc:.1f}%")

        # Save best
        if agreed_acc > best_val_acc:
            best_val_acc = agreed_acc
            best_epoch = epoch
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch, 'tf': tf, 'forward': forward,
                'val_acc': agreed_acc, 'val_state_acc': state_acc,
                'val_class_acc': class_acc, 'val_corr': avg_corr,
            }, os.path.join(ckpt_dir, 'best_model.pt'))

    print(f"\n  Best epoch: {best_epoch} (agreed_acc={best_val_acc:.1f}%)")
    print(f"  Saved: {ckpt_dir}/best_model.pt")

    # Final report
    print(f"\n{'=' * 60}")
    print(f"FINAL REPORT: {tf.upper()}")
    print(f"{'=' * 60}")
    print(f"  State head: {state_acc:.1f}%")
    print(f"  Direction head: {class_acc:.1f}%")
    print(f"  Agreement: {agreement:.0f}% | When agreed: {agreed_acc:.1f}%")
    print(f"  Top 5% confident: {top5_acc:.1f}%")
    print(f"  Feature correlation: {avg_corr:.3f}")
    print(f"  GATE (95% at top 5%): {'PASS' if top5_acc >= 95 else 'FAIL'} ({top5_acc:.1f}%)")


def build_1s_shards(ckpt_dir='checkpoints/direction_1s'):
    """Build 13D features + 7D labels per month for 1s data. Saves shards to disk."""
    from core.statistical_field_engine import StatisticalFieldEngine

    shard_dir = os.path.join(ckpt_dir, 'shards')
    os.makedirs(shard_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1s', '*.parquet')))
    forward = FORWARD_BARS['1s']

    for f in tqdm(files, desc="Building 1s shards"):
        month = os.path.basename(f).replace('.parquet', '')
        feat_path = os.path.join(shard_dir, f'{month}_feat.npy')
        label_path = os.path.join(shard_dir, f'{month}_label.npy')

        if os.path.exists(feat_path) and os.path.exists(label_path):
            _n = len(np.load(feat_path, mmap_mode='r'))
            print(f"  {month}: cached ({_n:,} bars)")
            continue

        df = pd.read_parquet(f).sort_values('timestamp').reset_index(drop=True)
        print(f"  {month}: {len(df):,} bars — computing SFE...", end=' ', flush=True)

        sfe = StatisticalFieldEngine()
        states = sfe.batch_compute_states(df)
        feats = extract_features_13d(states, df)
        labels = build_state_labels(feats[:, :N_FEAT_7D], forward)

        np.save(feat_path, feats)
        np.save(label_path, labels)
        print(f"saved")

        del states, feats, labels, df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  All shards saved to {shard_dir}/")
    return shard_dir


def train_1s_from_shards(n_epochs=100, val_start='2026-01-01'):
    """Train 1s DualHeadPredictor by streaming monthly shards."""
    from core.direction_cnn import DualHeadPredictor
    from scipy import stats as sp_stats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward = FORWARD_BARS['1s']
    ckpt_dir = 'checkpoints/direction_1s'
    shard_dir = os.path.join(ckpt_dir, 'shards')

    if not os.path.exists(shard_dir):
        build_1s_shards(ckpt_dir)

    # Classify shards as train or val
    shard_files = sorted(glob.glob(os.path.join(shard_dir, '*_feat.npy')))
    train_shards = []
    val_shards = []
    val_year_month = val_start[:7].replace('-', '_')  # '2026_01'

    for sf in shard_files:
        month = os.path.basename(sf).replace('_feat.npy', '')
        label_f = sf.replace('_feat.npy', '_label.npy')
        if month >= val_year_month:
            val_shards.append((sf, label_f, month))
        else:
            train_shards.append((sf, label_f, month))

    print(f"\n{'=' * 60}")
    print(f"DUAL HEAD (SHARDED): 1S (t+{forward}) | {len(train_shards)} train months, {len(val_shards)} val months")
    print(f"{'=' * 60}")

    # Count total samples
    train_total = sum(len(np.load(s[0], mmap_mode='r')) for s in train_shards)
    val_total = sum(len(np.load(s[0], mmap_mode='r')) for s in val_shards)
    print(f"  Train: {train_total:,} bars | Val: {val_total:,} bars")

    model = DualHeadPredictor(n_features=13, latent_dim=64, n_state=N_FEAT_7D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    DIR_LOSS_WEIGHT = 0.1

    best_val_acc = 0
    best_epoch = 0

    for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs 1S"):
        # Train: stream through each monthly shard
        model.train()
        epoch_loss = 0
        n_batches = 0

        for sf, lf, month in train_shards:
            feats = np.load(sf)
            labels = np.load(lf)
            ds = StateDataset(feats, labels, lookback=LOOKBACK, forward=forward)
            if len(ds) < 10:
                continue
            dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

            for x, y in dl:
                x, y = x.to(device), y.to(device)
                state_pred, p_long = model(x)
                loss_state = mse_fn(state_pred, y)
                dir_label = (y[:, 0] > 0).float()
                loss_dir = bce_fn(p_long, dir_label)
                loss = loss_state + DIR_LOSS_WEIGHT * loss_dir
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            del feats, labels, ds, dl
            gc.collect()

        # Validate: stream val shards
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            all_state = []
            all_plong = []
            all_true = []

            with torch.no_grad():
                for sf, lf, month in val_shards:
                    feats = np.load(sf)
                    labels = np.load(lf)
                    ds = StateDataset(feats, labels, lookback=LOOKBACK, forward=forward)
                    if len(ds) < 10:
                        continue
                    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)

                    for x, y in dl:
                        x = x.to(device)
                        s, p = model(x)
                        all_state.append(s.cpu().numpy())
                        all_plong.append(p.cpu().numpy())
                        all_true.append(y.numpy())

                    del feats, labels, ds, dl
                    gc.collect()

            preds = np.concatenate(all_state)
            p_longs = np.concatenate(all_plong)
            trues = np.concatenate(all_true)

            state_dir = preds[:, 0] > 0
            class_dir = p_longs > 0.5
            true_dir = trues[:, 0] > 0

            state_acc = (state_dir == true_dir).mean() * 100
            class_acc = (class_dir == true_dir).mean() * 100
            both_agree = state_dir == class_dir
            agreed_acc = (state_dir[both_agree] == true_dir[both_agree]).mean() * 100 if both_agree.sum() > 0 else 0
            agreement = both_agree.mean() * 100

            confidence = np.abs(p_longs - 0.5) * 2
            top5_thresh = np.percentile(confidence, 95)
            top5_mask = confidence >= top5_thresh
            top5_acc = (class_dir[top5_mask] == true_dir[top5_mask]).mean() * 100 if top5_mask.sum() > 0 else 0

            corrs = []
            for j in range(N_FEAT_7D):
                if trues[:, j].std() > 1e-8 and preds[:, j].std() > 1e-8:
                    r, _ = sp_stats.spearmanr(preds[:, j], trues[:, j])
                    corrs.append(r)
            avg_corr = np.mean(corrs) if corrs else 0

            _lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss / max(1, n_batches))

            print(f"  Epoch {epoch:>3}: loss={epoch_loss/max(1,n_batches):.4f} lr={_lr:.1e} | "
                  f"state={state_acc:.1f}% class={class_acc:.1f}% agree={agreement:.0f}%->{agreed_acc:.1f}% | "
                  f"corr={avg_corr:.3f} top5%={top5_acc:.1f}%")

            if agreed_acc > best_val_acc:
                best_val_acc = agreed_acc
                best_epoch = epoch
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch': epoch, 'tf': '1s', 'forward': forward,
                    'val_acc': agreed_acc, 'top5_acc': top5_acc,
                }, os.path.join(ckpt_dir, 'best_model.pt'))

    print(f"\n  Best epoch: {best_epoch} (agreed_acc={best_val_acc:.1f}%)")
    print(f"  Final top 5%: {top5_acc:.1f}%")
    print(f"  GATE (95% at top 5%): {'PASS' if top5_acc >= 95 else 'FAIL'} ({top5_acc:.1f}%)")
    print(f"  Saved: {ckpt_dir}/best_model.pt")


def train_1s_walkforward(epochs_per_day=10, cold_epochs=30):
    """Walk-forward 1s training: process monthly shards, train daily.

    Cold start: 30 epochs on first week of data.
    Then daily carry-forward with epochs_per_day.
    Features built per-month shard (1M bars each).
    """
    from core.direction_cnn import DualHeadPredictor
    from scipy import stats as sp_stats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward = FORWARD_BARS['1s']
    ckpt_dir = 'checkpoints/direction_1s'
    shard_dir = os.path.join(ckpt_dir, 'shards')

    # Build shards if needed
    if not os.path.exists(shard_dir) or not glob.glob(os.path.join(shard_dir, '*_feat.npy')):
        build_1s_shards(ckpt_dir)

    shard_files = sorted(glob.glob(os.path.join(shard_dir, '*_feat.npy')))
    print(f"\n{'=' * 60}")
    print(f"DUAL HEAD (WALK-FORWARD): 1S (t+{forward}) | {len(shard_files)} months")
    print(f"{'=' * 60}")

    model = DualHeadPredictor(n_features=13, latent_dim=64, n_state=N_FEAT_7D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    DIR_LOSS_WEIGHT = 0.1

    day_results = []
    cold_done = False

    for sf in shard_files:
        month = os.path.basename(sf).replace('_feat.npy', '')
        lf = sf.replace('_feat.npy', '_label.npy')

        feats = np.load(sf)
        if os.path.exists(lf):
            labels = np.load(lf)
        else:
            # Rebuild labels from cached features with current forward horizon
            print(f"  {month}: rebuilding labels at t+{forward}...", end=' ', flush=True)
            labels = build_state_labels(feats[:, :N_FEAT_7D], forward)
            np.save(lf, labels)
            print("saved")

        # Need timestamps for day splitting — load parquet just for timestamps
        pq_path = os.path.join(ATLAS_ROOT, '1s', f'{month}.parquet')
        df_ts = pd.read_parquet(pq_path, columns=['timestamp'])
        timestamps = df_ts['timestamp'].values
        dates = pd.to_datetime(timestamps, unit='s').date

        # Split into days
        unique_dates = sorted(set(dates))
        print(f"\n  {month}: {len(feats):,} bars, {len(unique_dates)} days")

        for di, date in enumerate(tqdm(unique_dates, desc=f"  {month}", leave=False)):
            day_mask = dates == date
            idx = np.where(day_mask)[0]
            if len(idx) < LOOKBACK + forward + 50:
                continue

            day_feats = feats[idx]
            day_labels = labels[idx]

            # Score BEFORE training (skip first week for cold start)
            if cold_done:
                model.eval()
                ds = StateDataset(day_feats, day_labels, lookback=LOOKBACK, forward=forward)
                if len(ds) > 0:
                    dl = DataLoader(ds, batch_size=1024, shuffle=False)
                    all_state, all_plong, all_true = [], [], []
                    with torch.no_grad():
                        for x, y in dl:
                            x = x.to(device)
                            s, p = model(x)
                            all_state.append(s.cpu().numpy())
                            all_plong.append(p.cpu().numpy())
                            all_true.append(y.numpy())

                    preds = np.concatenate(all_state)
                    p_longs = np.concatenate(all_plong)
                    trues = np.concatenate(all_true)

                    class_dir = p_longs > 0.5
                    true_dir = trues[:, 0] > 0
                    state_dir = preds[:, 0] > 0
                    acc = (class_dir == true_dir).mean() * 100
                    both_agree = state_dir == class_dir
                    agreed_acc = (state_dir[both_agree] == true_dir[both_agree]).mean() * 100 if both_agree.sum() > 0 else 0

                    confidence = np.abs(p_longs - 0.5) * 2
                    top5_thresh = np.percentile(confidence, 95) if len(confidence) > 20 else 0.9
                    top5_mask = confidence >= top5_thresh
                    top5_acc = (class_dir[top5_mask] == true_dir[top5_mask]).mean() * 100 if top5_mask.sum() > 0 else 0

                    corrs = []
                    for j in range(N_FEAT_7D):
                        if trues[:, j].std() > 1e-8 and preds[:, j].std() > 1e-8:
                            r, _ = sp_stats.spearmanr(preds[:, j], trues[:, j])
                            corrs.append(r)
                    avg_corr = np.mean(corrs) if corrs else 0

                    day_results.append({
                        'date': str(date), 'accuracy': acc,
                        'agreed_acc': agreed_acc, 'top5_acc': top5_acc,
                        'avg_corr': avg_corr, 'n_bars': len(idx),
                    })

            # Train
            model.train()
            ds = StateDataset(day_feats, day_labels, lookback=LOOKBACK, forward=forward)
            if len(ds) < 50:
                continue
            dl = DataLoader(ds, batch_size=512, shuffle=True)

            if not cold_done:
                _epochs = cold_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = 1e-3
                cold_done = True
                print(f"    Cold start: {_epochs} epochs on {date}")
            else:
                _epochs = epochs_per_day
                for pg in optimizer.param_groups:
                    pg['lr'] = 1e-4

            for _ in range(_epochs):
                for x, y in dl:
                    x, y = x.to(device), y.to(device)
                    state_pred, p_long = model(x)
                    loss_state = mse_fn(state_pred, y)
                    dir_label = (y[:, 0] > 0).float()
                    loss_dir = bce_fn(p_long, dir_label)
                    loss = loss_state + DIR_LOSS_WEIGHT * loss_dir
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Save checkpoint after each month
        torch.save({
            'model_state': model.state_dict(),
            'month': month, 'tf': '1s', 'forward': forward,
        }, os.path.join(ckpt_dir, f'model_{month}.pt'))
        print(f"    Saved: model_{month}.pt")

        del feats, labels, df_ts
        gc.collect()

    # Save final
    torch.save({
        'model_state': model.state_dict(),
        'tf': '1s', 'forward': forward,
    }, os.path.join(ckpt_dir, 'best_model.pt'))

    # Report
    if day_results:
        df_r = pd.DataFrame(day_results)
        print(f"\n{'=' * 60}")
        print(f"STATE PREDICTOR REPORT: 1S (t+{forward})")
        print(f"{'=' * 60}")
        print(f"  Days scored: {len(df_r)}")
        print(f"  Direction accuracy: {df_r['accuracy'].mean():.1f}%")
        print(f"  When agreed: {df_r['agreed_acc'].mean():.1f}%")
        print(f"  Top 5% confident: {df_r['top5_acc'].mean():.1f}%")
        print(f"  Avg correlation: {df_r['avg_corr'].mean():.3f}")
        print(f"  GATE (95% at top 5%): {'PASS' if df_r['top5_acc'].mean() >= 95 else 'FAIL'}")

        df_r['month'] = df_r['date'].str[:7]
        print(f"\n  MONTHLY:")
        for month, grp in df_r.groupby('month'):
            print(f"    {month}: acc={grp['accuracy'].mean():.1f}% "
                  f"top5%={grp['top5_acc'].mean():.1f}%")

        df_r.to_csv(os.path.join(ckpt_dir, 'daily_results.csv'), index=False)
        print(f"  Saved: {ckpt_dir}/daily_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', required=True, choices=['1h', '1m', '1s'],
                        help='Timeframe to train')
    parser.add_argument('--max-bars', type=int, default=0,
                        help='Limit bars (useful for 1s)')
    parser.add_argument('--mode', default='epoch', choices=['epoch', 'walkforward'],
                        help='epoch = full dataset training, walkforward = daily carry-forward')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val-start', default='2026-01-01',
                        help='Validation start date (default: 2026-01-01)')
    parser.add_argument('--epochs-per-day', type=int, default=10,
                        help='Epochs per day in walk-forward mode (default: 10)')
    args = parser.parse_args()

    if args.tf == '1s':
        # 1s always uses sharded approach (15M bars too large for single pass)
        if args.mode == 'epoch':
            train_1s_from_shards(n_epochs=args.epochs, val_start=args.val_start)
        else:
            train_1s_walkforward(epochs_per_day=args.epochs_per_day, cold_epochs=30)
    elif args.mode == 'epoch':
        train_epoch_based(args.tf, max_bars=args.max_bars,
                          val_start=args.val_start, n_epochs=args.epochs)
    else:
        train_and_validate(args.tf, max_bars=args.max_bars, epochs_per_day=args.epochs_per_day)


if __name__ == '__main__':
    main()
