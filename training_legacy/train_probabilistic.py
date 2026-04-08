"""
Probabilistic Trajectory CNN — predicts feature evolution + P(direction) at 10 horizons.

Each horizon gets its own predicted features. The probability collapse
across horizons IS the trading signal.

Input: 10-bar lookback × 22D (13D base + 4 wave function + 2 reversion/breakout + 2 level distance + 1 zone position)
Output per horizon: 22D predicted features + P(long) = 23D × 10 = 230D total

Training: walk-forward on clean data with level context.
Labels: actual 22D features + direction at each horizon.

Usage:
  python -m training.train_probabilistic --tf 1m
  python -m training.train_probabilistic --tf 1h
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
from scipy import stats as sp_stats

ATLAS_ROOT = 'DATA/ATLAS'
TICK = 0.25
LOOKBACK = 10
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_HORIZONS = len(HORIZONS)

# 22D feature names
FEATURE_NAMES_BASE = [
    'dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel',
    'std_price', 'variance_ratio', 'bar_range', 'wick_ratio',
    'vwap_distance', 'time_of_day',
]
FEATURE_NAMES_WAVE = ['P_at_center', 'P_near_upper', 'P_near_lower', 'entropy_normalized']
FEATURE_NAMES_PROB = ['reversion_probability', 'breakout_probability']
FEATURE_NAMES_LEVEL = ['dist_to_resistance', 'dist_to_support', 'zone_position']
FEATURE_NAMES_22D = FEATURE_NAMES_BASE + FEATURE_NAMES_WAVE + FEATURE_NAMES_PROB + FEATURE_NAMES_LEVEL
N_FEAT = len(FEATURE_NAMES_22D)  # 22


def extract_features_22d(states, df, levels):
    """Extract 22D features: 13D base + 4 wave + 2 prob + 3 level context."""
    from training.train_trade_cnn import extract_features_13d

    n = len(states)
    feats = np.zeros((n, N_FEAT), dtype=np.float32)

    # 13D base
    feats_13d = extract_features_13d(states, df)
    feats[:, :13] = feats_13d

    # Wave function + reversion/breakout from MarketState
    for i in range(n):
        st = states[i]['state'] if isinstance(states[i], dict) else states[i]
        feats[i, 13] = getattr(st, 'P_at_center', 0.33)
        feats[i, 14] = getattr(st, 'P_near_upper', 0.33)
        feats[i, 15] = getattr(st, 'P_near_lower', 0.33)
        feats[i, 16] = getattr(st, 'entropy_normalized', 1.0)
        feats[i, 17] = getattr(st, 'reversion_probability', 0.5)
        feats[i, 18] = getattr(st, 'breakout_probability', 0.5)

    # Level context
    if levels:
        level_prices = sorted([l['price'] for l in levels])
        r_levels = [l['price'] for l in levels if l['type'] == 'resistance']
        s_levels = [l['price'] for l in levels if l['type'] == 'support']
        prices = df['close'].values

        for i in range(n):
            p = prices[i]
            # Distance to nearest resistance (positive = below, negative = above)
            if r_levels:
                nearest_r = min(r_levels, key=lambda r: abs(r - p))
                feats[i, 19] = (nearest_r - p) / TICK
            # Distance to nearest support
            if s_levels:
                nearest_s = min(s_levels, key=lambda s: abs(s - p))
                feats[i, 20] = (p - nearest_s) / TICK
            # Zone position: 0 = at support, 1 = at resistance
            if len(level_prices) >= 2:
                zone_low = max([lp for lp in level_prices if lp <= p], default=level_prices[0])
                zone_high = min([lp for lp in level_prices if lp >= p], default=level_prices[-1])
                zone_range = zone_high - zone_low
                if zone_range > 0:
                    feats[i, 21] = (p - zone_low) / zone_range
                else:
                    feats[i, 21] = 0.5

    return feats


def build_labels(feats, horizons=HORIZONS):
    """Build labels: actual 22D features + direction at each horizon.

    Per bar i, per horizon h:
      features_label = feats[i+h]  (22D)
      direction_label = 1 if feats[i+h, 0] > 0 else 0  (dmi_diff > 0 = long)

    Total label dim = (22 + 1) × 10 = 230
    """
    n = len(feats)
    max_h = max(horizons)
    n_out = (N_FEAT + 1) * N_HORIZONS  # 23 × 10 = 230
    labels = np.zeros((n, n_out), dtype=np.float32)

    for i in range(n - max_h):
        for hi, h in enumerate(horizons):
            start = hi * (N_FEAT + 1)
            labels[i, start:start + N_FEAT] = feats[i + h]  # 22D features
            labels[i, start + N_FEAT] = 1.0 if feats[i + h, 0] > 0 else 0.0  # direction

    return labels


class TrajectoryDataset(Dataset):
    def __init__(self, features, labels, lookback=LOOKBACK):
        self.features = features
        self.labels = labels
        self.lookback = lookback
        self.n = len(features) - lookback - max(HORIZONS)

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        i = idx + self.lookback
        x = self.features[i - self.lookback:i]
        y = self.labels[i]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class FeatureGroupEncoder(nn.Module):
    """Encodes one feature group from the lookback window into a latent."""

    def __init__(self, n_features, latent_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, n_features * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(n_features * 2)
        self.conv2 = nn.Conv1d(n_features * 2, latent_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, lookback, n_features_group)
        h = x.transpose(1, 2)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.pool(h).squeeze(-1)
        return h  # (batch, latent_dim)


class ProbabilisticTrajectory(nn.Module):
    """Per-feature-group encoders → merged latent → per-horizon heads.

    Feature groups (separate encoders, can be mixed/matched):
      Base (13D): dmi, velocity, z_se, regime, context
      Wave (4D): P_center, P_upper, P_lower, entropy
      Prob (2D): reversion_probability, breakout_probability
      Level (3D): dist_to_R, dist_to_S, zone_position

    Each group gets its own CNN encoder. Latents are concatenated
    and fed to per-horizon prediction heads.
    """

    # Feature group slices
    GROUP_SLICES = {
        'base': (0, 13),    # 13D base features
        'wave': (13, 17),   # 4D wave function
        'prob': (17, 19),   # 2D reversion/breakout
        'level': (19, 22),  # 3D level context
    }
    GROUP_LATENT = 16  # latent dim per group

    def __init__(self, n_features=N_FEAT, n_horizons=N_HORIZONS):
        super().__init__()
        self.n_features = n_features
        self.n_horizons = n_horizons

        # Per-group encoders
        self.encoders = nn.ModuleDict()
        total_latent = 0
        for name, (start, end) in self.GROUP_SLICES.items():
            n_feat = end - start
            self.encoders[name] = FeatureGroupEncoder(n_feat, self.GROUP_LATENT)
            total_latent += self.GROUP_LATENT

        # Merge layer: concatenated group latents → shared representation
        self.merge = nn.Sequential(
            nn.Linear(total_latent, 64),
            nn.ReLU(),
        )

        # Per-horizon heads: each predicts 22D features + 1 P(long) = 23D
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 48),
                nn.ReLU(),
                nn.Linear(48, n_features + 1),
            )
            for _ in range(n_horizons)
        ])

        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        _groups = ', '.join([f'{k}({e-s}D)' for k, (s, e) in self.GROUP_SLICES.items()])
        print(f"[ProbabilisticTrajectory] {_total:,} params | "
              f"groups: [{_groups}] | "
              f"output: ({n_features}+1)D x {n_horizons} = {(n_features+1)*n_horizons}D")

    def forward(self, x):
        # Encode each feature group separately
        group_latents = []
        for name, (start, end) in self.GROUP_SLICES.items():
            group_input = x[:, :, start:end]  # (batch, lookback, group_dim)
            group_latent = self.encoders[name](group_input)  # (batch, 16)
            group_latents.append(group_latent)

        # Merge
        merged = torch.cat(group_latents, dim=-1)  # (batch, 64)
        latent = self.merge(merged)  # (batch, 64)

        # Per-horizon predictions
        outputs = []
        for head in self.horizon_heads:
            out = head(latent)
            feat_pred = out[:, :-1]
            dir_pred = torch.sigmoid(out[:, -1:])
            outputs.append(torch.cat([feat_pred, dir_pred], dim=-1))

        return torch.cat(outputs, dim=-1)  # (batch, 230)

    def encode_groups(self, x):
        """Return per-group latents for analysis/diagnosis."""
        result = {}
        for name, (start, end) in self.GROUP_SLICES.items():
            group_input = x[:, :, start:end]
            result[name] = self.encoders[name](group_input)
        return result


def get_levels_for_month(month_str):
    """Find level file for this month."""
    files = sorted(glob.glob('DATA/levels/levels_*.json'))
    best = None
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d['date'][:7] <= month_str:
            best = d['levels']
    return best


def train_model(tf='1m', epochs_per_day=10):
    """Walk-forward training of ProbabilisticTrajectory."""
    from core.statistical_field_engine import StatisticalFieldEngine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = f'checkpoints/probabilistic_{tf}'
    os.makedirs(ckpt_dir, exist_ok=True)

    max_h = max(HORIZONS)

    print(f"\n{'='*60}")
    print(f"PROBABILISTIC TRAJECTORY: {tf.upper()} | {N_FEAT}D input | {N_HORIZONS} horizons")
    print(f"{'='*60}")

    # Load data
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Bars: {len(df):,}")

    # Check for cached features
    feat_cache = os.path.join(ckpt_dir, 'features_22d.npy')
    label_cache = os.path.join(ckpt_dir, 'labels_230d.npy')

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

        # Get levels for each bar's month
        print("  Extracting 22D features with level context...")
        # Use the most common month's levels as default
        month = pd.to_datetime(df['timestamp'].median(), unit='s').strftime('%Y-%m')
        levels = get_levels_for_month(month)
        feats = extract_features_22d(states, df, levels)

        print("  Building labels (22D + direction × 10 horizons)...")
        labels = build_labels(feats, HORIZONS)

        # Cache
        np.save(feat_cache, feats)
        np.save(label_cache, labels)
        print(f"  Saved: {feat_cache} ({feats.shape}), {label_cache} ({labels.shape})")

        del states; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Features: {feats.shape}, Labels: {labels.shape}")

    # Day boundaries
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    day_boundaries = []
    for date, group in df.groupby('date'):
        day_boundaries.append({'date': date, 'start': group.index[0], 'end': group.index[-1]})
    print(f"  Trading days: {len(day_boundaries)}")

    # Model
    model = ProbabilisticTrajectory(n_features=N_FEAT, n_horizons=N_HORIZONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    DIR_WEIGHT = 0.1

    day_results = []
    _epd = epochs_per_day

    for di, day in enumerate(tqdm(day_boundaries, desc=f"Walk-Forward {tf.upper()}")):
        _start = day['start']
        _end = day['end']
        n_bars = _end - _start + 1

        if n_bars < LOOKBACK + max_h + 5:
            continue

        day_feats = feats[_start:_end + 1]
        day_labels = labels[_start:_end + 1]

        # Score BEFORE training
        if di > 0:
            model.eval()
            ds = TrajectoryDataset(day_feats, day_labels, lookback=LOOKBACK)
            if len(ds) > 0:
                dl = DataLoader(ds, batch_size=512, shuffle=False)
                all_pred, all_true = [], []
                with torch.no_grad():
                    for x, y in dl:
                        x = x.to(device)
                        pred = model(x).cpu().numpy()
                        all_pred.append(pred)
                        all_true.append(y.numpy())

                preds = np.concatenate(all_pred)
                trues = np.concatenate(all_true)

                # Per-horizon direction accuracy
                h_accs = []
                for hi in range(N_HORIZONS):
                    start_idx = hi * (N_FEAT + 1)
                    dir_idx = start_idx + N_FEAT
                    pred_dir = preds[:, dir_idx] > 0.5
                    true_dir = trues[:, dir_idx] > 0.5
                    h_accs.append((pred_dir == true_dir).mean() * 100)

                # Feature correlation at n+1
                corrs = []
                for fi in range(N_FEAT):
                    if trues[:, fi].std() > 1e-8 and preds[:, fi].std() > 1e-8:
                        r, _ = sp_stats.spearmanr(preds[:, fi], trues[:, fi])
                        corrs.append(r)
                avg_corr = np.mean(corrs) if corrs else 0

                # Top 5% confidence at n+1
                p_n1 = preds[:, N_FEAT]
                conf = np.abs(p_n1 - 0.5) * 2
                top5_thresh = np.percentile(conf, 95) if len(conf) > 20 else 0.9
                top5_mask = conf >= top5_thresh
                true_dir_n1 = trues[:, N_FEAT] > 0.5
                top5_acc = ((p_n1[top5_mask] > 0.5) == true_dir_n1[top5_mask]).mean() * 100 if top5_mask.sum() > 0 else 0

                day_results.append({
                    'date': str(day['date']), 'day': di + 1,
                    'avg_corr': avg_corr, 'top5_acc': top5_acc,
                    **{f'acc_n{h}': h_accs[i] for i, h in enumerate(HORIZONS)},
                })

                if (di + 1) % 30 == 0:
                    h_str = ' '.join([f'n+{h}={h_accs[i]:.0f}%' for i, h in enumerate(HORIZONS[:5])])
                    print(f"  Day {di+1}: {h_str} ... n+10={h_accs[-1]:.0f}% | "
                          f"corr={avg_corr:.3f} top5%={top5_acc:.1f}%")

        # Train
        model.train()
        ds = TrajectoryDataset(day_feats, day_labels, lookback=LOOKBACK)
        if len(ds) < 10:
            continue
        dl = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)

        _epochs = 30 if di == 0 else _epd
        _lr = 1e-3 if di == 0 else 1e-4
        for pg in optimizer.param_groups:
            pg['lr'] = _lr

        for _ in range(_epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                # Split loss: MSE on features + BCE on direction per horizon
                loss_feat = 0
                loss_dir = 0
                for hi in range(N_HORIZONS):
                    start_idx = hi * (N_FEAT + 1)
                    feat_pred = pred[:, start_idx:start_idx + N_FEAT]
                    feat_true = y[:, start_idx:start_idx + N_FEAT]
                    dir_pred = pred[:, start_idx + N_FEAT]
                    dir_true = y[:, start_idx + N_FEAT]

                    loss_feat += mse_fn(feat_pred, feat_true)
                    loss_dir += bce_fn(dir_pred, dir_true)

                loss = loss_feat + DIR_WEIGHT * loss_dir
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Checkpoint every 30 days
        if (di + 1) % 30 == 0:
            torch.save({
                'model_state': model.state_dict(),
                'day': di + 1, 'date': str(day['date']),
                'tf': tf, 'n_features': N_FEAT, 'horizons': HORIZONS,
            }, os.path.join(ckpt_dir, f'model_day{di+1}.pt'))

    # Save final — full model + individual group encoders
    torch.save({
        'model_state': model.state_dict(),
        'day': len(day_boundaries), 'tf': tf,
        'n_features': N_FEAT, 'horizons': HORIZONS,
        'feature_names': FEATURE_NAMES_22D,
        'group_slices': ProbabilisticTrajectory.GROUP_SLICES,
    }, os.path.join(ckpt_dir, 'best_model.pt'))
    print(f"  Saved: {ckpt_dir}/best_model.pt")

    # Save each group encoder separately (mix and match later)
    for name in model.encoders:
        encoder_path = os.path.join(ckpt_dir, f'encoder_{name}.pt')
        torch.save({
            'state_dict': model.encoders[name].state_dict(),
            'group': name,
            'slice': ProbabilisticTrajectory.GROUP_SLICES[name],
            'latent_dim': ProbabilisticTrajectory.GROUP_LATENT,
        }, encoder_path)
        print(f"  Saved: {encoder_path}")

    # Save merge layer separately
    torch.save({
        'state_dict': model.merge.state_dict(),
    }, os.path.join(ckpt_dir, 'merge_layer.pt'))

    # Save each horizon head separately
    for hi, h in enumerate(HORIZONS):
        head_path = os.path.join(ckpt_dir, f'head_n{h}.pt')
        torch.save({
            'state_dict': model.horizon_heads[hi].state_dict(),
            'horizon': h,
        }, head_path)
    print(f"  Saved: {len(HORIZONS)} horizon heads + merge layer")

    # Report
    if day_results:
        df_r = pd.DataFrame(day_results)

        print(f"\n{'='*60}")
        print(f"PROBABILISTIC TRAJECTORY: {tf.upper()} | {N_FEAT}D | {N_HORIZONS} horizons")
        print(f"{'='*60}")
        print(f"  Days scored: {len(df_r)}")

        print(f"\n  TRAJECTORY DECAY:")
        for hi, h in enumerate(HORIZONS):
            print(f"    n+{h}: {df_r[f'acc_n{h}'].mean():.1f}%")

        print(f"\n  Avg correlation: {df_r['avg_corr'].mean():.3f}")
        print(f"  Top 5% at n+1: {df_r['top5_acc'].mean():.1f}%")

        # Monthly
        df_r['month'] = df_r['date'].str[:7]
        print(f"\n  MONTHLY (n+1):")
        for month, grp in df_r.groupby('month'):
            print(f"    {month}: n+1={grp['acc_n1'].mean():.1f}% top5%={grp['top5_acc'].mean():.1f}%")

        df_r.to_csv(os.path.join(ckpt_dir, 'daily_results.csv'), index=False)
        print(f"  Saved: {ckpt_dir}/daily_results.csv")


def train_epoch(tf='1D', n_epochs=100, val_start='2026-01-01'):
    """Epoch-based training for sparse TFs (1D, 1W)."""
    from core.statistical_field_engine import StatisticalFieldEngine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = f'checkpoints/probabilistic_{tf}'
    os.makedirs(ckpt_dir, exist_ok=True)
    max_h = max(HORIZONS)

    print(f"\n{'='*60}")
    print(f"PROBABILISTIC (EPOCH): {tf} | {N_FEAT}D | {N_HORIZONS}h | val>={val_start}")
    print(f"{'='*60}")

    # Load or build features
    feat_cache = os.path.join(ckpt_dir, 'features_22d.npy')
    label_cache = os.path.join(ckpt_dir, 'labels_230d.npy')

    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

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
        month = pd.to_datetime(df['timestamp'].median(), unit='s').strftime('%Y-%m')
        levels = get_levels_for_month(month)
        feats = extract_features_22d(states, df, levels)
        labels = build_labels(feats, HORIZONS)
        np.save(feat_cache, feats)
        np.save(label_cache, labels)
        del states; gc.collect()

    # Train/val split
    val_ts = pd.Timestamp(val_start).timestamp()
    train_mask = df['timestamp'].values < val_ts
    val_mask = df['timestamp'].values >= val_ts

    train_ds = TrajectoryDataset(feats[train_mask], labels[train_mask])
    val_ds = TrajectoryDataset(feats[val_mask], labels[val_mask])
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = ProbabilisticTrajectory(n_features=N_FEAT, n_horizons=N_HORIZONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()
    DIR_WEIGHT = 0.1

    best_metric = 0

    for epoch in tqdm(range(1, n_epochs + 1), desc=f"Epochs {tf}"):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_feat = 0
            loss_dir = 0
            for hi in range(N_HORIZONS):
                si = hi * (N_FEAT + 1)
                loss_feat += mse_fn(pred[:, si:si + N_FEAT], y[:, si:si + N_FEAT])
                loss_dir += bce_fn(pred[:, si + N_FEAT], y[:, si + N_FEAT])
            loss = loss_feat + DIR_WEIGHT * loss_dir
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for x, y in val_dl:
                    x = x.to(device)
                    all_pred.append(model(x).cpu().numpy())
                    all_true.append(y.numpy())

            if all_pred:
                preds = np.concatenate(all_pred)
                trues = np.concatenate(all_true)

                h_accs = []
                for hi in range(N_HORIZONS):
                    si = hi * (N_FEAT + 1)
                    di = si + N_FEAT
                    acc = ((preds[:, di] > 0.5) == (trues[:, di] > 0.5)).mean() * 100
                    h_accs.append(acc)

                _lr = optimizer.param_groups[0]['lr']
                scheduler.step(epoch_loss / max(1, n_batches))

                h_str = ' '.join([f'n+{h}={a:.0f}%' for h, a in zip(HORIZONS[:5], h_accs[:5])])
                print(f"  E{epoch:>3}: {h_str} ... n+10={h_accs[-1]:.0f}% | lr={_lr:.1e}")

                if h_accs[0] > best_metric:
                    best_metric = h_accs[0]
                    torch.save({
                        'model_state': model.state_dict(),
                        'epoch': epoch, 'tf': tf, 'n_features': N_FEAT,
                        'horizons': HORIZONS, 'val_n1_acc': h_accs[0],
                        'feature_names': FEATURE_NAMES_22D,
                        'group_slices': ProbabilisticTrajectory.GROUP_SLICES,
                    }, os.path.join(ckpt_dir, 'best_model.pt'))

    # Save group encoders
    for name in model.encoders:
        torch.save({
            'state_dict': model.encoders[name].state_dict(),
            'group': name,
            'slice': ProbabilisticTrajectory.GROUP_SLICES[name],
        }, os.path.join(ckpt_dir, f'encoder_{name}.pt'))
    for hi, h in enumerate(HORIZONS):
        torch.save({
            'state_dict': model.horizon_heads[hi].state_dict(),
            'horizon': h,
        }, os.path.join(ckpt_dir, f'head_n{h}.pt'))
    torch.save({'state_dict': model.merge.state_dict()}, os.path.join(ckpt_dir, 'merge_layer.pt'))

    print(f"\n  Best n+1 accuracy: {best_metric:.1f}%")
    print(f"  All components saved to {ckpt_dir}/")

    # Final decay curve
    if all_pred:
        print(f"\n  TRAJECTORY DECAY:")
        for hi, h in enumerate(HORIZONS):
            print(f"    n+{h}: {h_accs[hi]:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='1m', choices=['1D', '4h', '1h', '15m', '1m'])
    parser.add_argument('--epochs-per-day', type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    if args.tf in ('1D', '4h', '1h'):
        train_epoch(tf=args.tf, n_epochs=args.epochs_per_day * 10)
    else:
        train_model(tf=args.tf, epochs_per_day=args.epochs_per_day)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
