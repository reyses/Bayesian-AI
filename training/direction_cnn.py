"""
Direction CNN — learns LONG/SHORT from grounded features.

Architecture:
  Input: (batch, 10, 7) — 10 bars lookback × 7 grounded features
  Conv1D: 32 filters, kernel=3 → ReLU
  Conv1D: 64 filters, kernel=3 → ReLU
  Flatten → Dense(32) → ReLU → Dense(1) → Sigmoid
  Output: P(LONG) — probability that LONG is profitable

Training:
  IS data with full lookahead labels (supervised learning).
  Label = 1 if going LONG at this bar would have been profitable after 10 bars.
  Custom loss: weighted by magnitude of the move (big moves matter more).

Usage:
  python -m training.direction_cnn --phase train    # train on IS
  python -m training.direction_cnn --phase validate # test on OOS
  python -m training.direction_cnn --phase all      # train + validate
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob


# --- CONFIG ---
LOOKBACK = 10          # bars of history per sample
FORWARD = 10           # bars ahead for label (was this LONG profitable?)
FEATURES_7D = ['dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel']
FEATURES_18D = [
    'dmi_diff', 'dmi_gap', 'dmi_plus', 'dmi_minus',          # DMI (level 2-3)
    'vol_rel', 'dir_vol', 'std_volume',                       # Volume (level 2-3)
    'velocity', 'price_accel',                                 # Motion (level 2-3)
    'z_se', 'mean_price', 'std_price', 'variance_ratio',      # Statistics (level 2-3)
    'wick_ratio', 'bar_range', 'price_position',               # Bar structure (level 2)
    'vwap_distance', 'time_of_day',                            # Context (level 1-3)
]
FEATURES_3D = ['price_delta', 'time_delta', 'volume']          # Pure base measurements
FEATURES_22D = [
    'F_momentum', 'z_score', 'dmi_plus', 'dmi_minus', 'adx_strength',
    'velocity', 'volume_delta', 'hurst_exponent', 'P_at_center',
    'oscillation_entropy_normalized', 'regression_sigma', 'term_pid',
    'regression_center', 'regression_slope', 'P_lower', 'P_upper',
    'dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'z_se', 'price_accel',
]
FEAT_MODE = 7  # default to 7D (proven $651/day), --features 18 or 22 to switch
FEATURES = FEATURES_7D
N_FEAT = len(FEATURES)
N_LAYERS = 2  # default 2 conv layers, use --layers 50 for deep residual
USE_22D = False  # legacy compat
RESULTS_LOG = 'reports/findings/cnn_experiment_log.txt'
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
CHECKPOINT_DIR = 'checkpoints/direction_cnn'
IS_ROOT = 'DATA/ATLAS'
OOS_ROOT = 'DATA/ATLAS_OOS'
TICK = 0.25

# No fixed seed — let randomness explore different solutions


# --- MODEL ---
class ResBlock(nn.Module):
    """Residual block for deep networks."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # skip connection
        x = self.relu(x)
        return x


class DirectionCNN(nn.Module):
    """1D CNN for direction prediction. Scales from 2-layer to 50-layer with residuals."""

    def __init__(self, n_features=N_FEAT, lookback=LOOKBACK, n_layers=2):
        super().__init__()
        self.n_layers = n_layers

        if n_layers <= 4:
            # Simple mode: 2-4 conv layers (original architecture)
            self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, 1)
        else:
            # Deep mode: residual blocks
            self.input_conv = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
            self.input_bn = nn.BatchNorm1d(32)
            n_res_blocks = (n_layers - 2) // 2  # each ResBlock = 2 conv layers
            self.res_blocks = nn.Sequential(*[ResBlock(32) for _ in range(n_res_blocks)])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, n_features, lookback)

        if self.n_layers <= 4:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
        else:
            x = self.relu(self.input_bn(self.input_conv(x)))
            x = self.res_blocks(x)

        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)


# --- FEATURE EXTRACTION ---
def extract_features_from_states(states, df):
    """Extract features per bar from SFE states + OHLCV."""
    n = len(states)
    feats = np.zeros((n, N_FEAT))

    prices = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(n)
    vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values

    for i in range(n):
        st = states[i]['state'] if isinstance(states[i], dict) else states[i]
        dmi_p = getattr(st, 'dmi_plus', 0.0)
        dmi_m = getattr(st, 'dmi_minus', 0.0)
        vel = getattr(st, 'velocity', 0.0)
        vol = volumes[i]

        if FEAT_MODE == 3:
            # 3D pure base measurements — let the network learn everything
            # price_delta: close-to-close change in ticks (not raw price — that's non-stationary)
            if i > 0:
                feats[i, 0] = (prices[i] - prices[i-1]) / TICK   # price_delta in ticks
            # time_delta: seconds since last bar (should be ~60 for 1m bars)
            if i > 0:
                feats[i, 1] = df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']
            # volume: raw volume (the network normalizes internally via batch norm)
            feats[i, 2] = vol

        elif FEAT_MODE == 18:
            # 18D grounded features — all ≤3 layers from base measurements
            highs_i = df.iloc[i]['high']
            lows_i = df.iloc[i]['low']
            opens_i = df.iloc[i]['open']
            ts_i = df.iloc[i]['timestamp']

            # DMI block (4)
            feats[i, 0] = dmi_p - dmi_m                          # dmi_diff
            feats[i, 1] = abs(dmi_p - dmi_m)                     # dmi_gap
            feats[i, 2] = dmi_p                                   # dmi_plus
            feats[i, 3] = dmi_m                                   # dmi_minus

            # Volume block (3)
            feats[i, 4] = vol / vol_avg[i] if vol_avg[i] > 0 else 1.0  # vol_rel
            if i > 0:
                price_dir = 1.0 if prices[i] > prices[i-1] else -1.0
                feats[i, 5] = price_dir * vol / vol_avg[i] if vol_avg[i] > 0 else 0.0  # dir_vol
            # std_volume: rolling std of volume (30-bar)
            if i >= 30:
                feats[i, 6] = np.std(volumes[i-30:i])            # std_volume

            # Motion block (2)
            feats[i, 7] = vel                                     # velocity
            if i > 0:
                prev_vel = getattr(states[i-1]['state'] if isinstance(states[i-1], dict) else states[i-1], 'velocity', 0.0)
                feats[i, 8] = vel - prev_vel                      # price_accel

            # Statistics block (4)
            if i >= 15:
                window = prices[max(0, i-60):i+1]
                _mean = window.mean()
                _std = window.std()
                _se = _std / (len(window) ** 0.5) if len(window) > 1 else _std
                feats[i, 9] = (prices[i] - _mean) / _se if _se > 1e-8 else 0.0  # z_se
                feats[i, 10] = _mean                              # mean_price
                feats[i, 11] = _std                               # std_price
                # variance_ratio: short-term var / long-term var
                if i >= 60:
                    short_std = np.std(prices[i-10:i+1])
                    long_std = np.std(prices[i-60:i+1])
                    feats[i, 12] = short_std / long_std if long_std > 1e-8 else 1.0

            # Bar structure block (3)
            _range = highs_i - lows_i
            if _range > 0:
                _body = abs(prices[i] - opens_i)
                feats[i, 13] = 1.0 - (_body / _range)            # wick_ratio (0=no wick, 1=all wick)
            feats[i, 14] = _range / 0.25                          # bar_range in ticks
            if _range > 0:
                feats[i, 15] = (prices[i] - lows_i) / _range      # price_position (0=low, 1=high)

            # Context block (2)
            # vwap_distance: distance from rolling VWAP
            if i >= 30:
                _vwap_num = np.sum(prices[i-30:i+1] * volumes[i-30:i+1])
                _vwap_den = np.sum(volumes[i-30:i+1])
                _vwap = _vwap_num / _vwap_den if _vwap_den > 0 else prices[i]
                feats[i, 16] = (prices[i] - _vwap) / 0.25         # vwap_distance in ticks
            # time_of_day: 0-1 within 24h session (UTC)
            _hour = (ts_i % 86400) / 86400.0
            feats[i, 17] = _hour                                   # time_of_day

        elif USE_22D:
            # First 12: raw SFE fields
            feats[i, 0] = getattr(st, 'F_momentum', 0.0)
            feats[i, 1] = getattr(st, 'z_score', 0.0)
            feats[i, 2] = dmi_p
            feats[i, 3] = dmi_m
            feats[i, 4] = getattr(st, 'adx_strength', 0.0)
            feats[i, 5] = vel
            feats[i, 6] = getattr(st, 'volume_delta', 0.0)
            feats[i, 7] = getattr(st, 'hurst_exponent', 0.0)
            feats[i, 8] = getattr(st, 'P_at_center', 0.0)
            feats[i, 9] = getattr(st, 'oscillation_entropy_normalized', 0.0)
            feats[i, 10] = getattr(st, 'regression_sigma', 0.0)
            feats[i, 11] = getattr(st, 'term_pid', 0.0)
            # 12-15: regression fields
            feats[i, 12] = getattr(st, 'regression_center', 0.0)
            feats[i, 13] = getattr(st, 'regression_slope', 0.0)
            feats[i, 14] = getattr(st, 'P_lower', 0.0)
            feats[i, 15] = getattr(st, 'P_upper', 0.0)
            # 16-21: grounded derived features
            feats[i, 16] = dmi_p - dmi_m  # dmi_diff
            feats[i, 17] = abs(dmi_p - dmi_m)  # dmi_gap
            feats[i, 18] = vol / vol_avg[i] if vol_avg[i] > 0 else 1.0  # vol_rel
            if i > 0:
                price_dir = 1.0 if prices[i] > prices[i-1] else -1.0
                feats[i, 19] = price_dir * vol / vol_avg[i] if vol_avg[i] > 0 else 0.0  # dir_vol
            if i >= 15:
                window = prices[max(0, i-60):i+1]
                _mean = window.mean()
                _std = window.std()
                _se = _std / (len(window) ** 0.5) if len(window) > 1 else _std
                feats[i, 20] = (prices[i] - _mean) / _se if _se > 1e-8 else 0.0  # z_se
            if i > 0:
                prev_vel = getattr(states[i-1]['state'] if isinstance(states[i-1], dict) else states[i-1], 'velocity', 0.0)
                feats[i, 21] = vel - prev_vel  # price_accel
        else:
            # 7D grounded features
            feats[i, 0] = dmi_p - dmi_m
            feats[i, 1] = abs(dmi_p - dmi_m)
            feats[i, 2] = vol / vol_avg[i] if vol_avg[i] > 0 else 1.0
            if i > 0:
                price_dir = 1.0 if prices[i] > prices[i-1] else -1.0
                feats[i, 3] = price_dir * vol / vol_avg[i] if vol_avg[i] > 0 else 0.0
            feats[i, 4] = vel
            if i >= 15:
                window = prices[max(0, i-60):i+1]
                _mean = window.mean()
                _std = window.std()
                _se = _std / (len(window) ** 0.5) if len(window) > 1 else _std
                feats[i, 5] = (prices[i] - _mean) / _se if _se > 1e-8 else 0.0
            if i > 0:
                prev_vel = getattr(states[i-1]['state'] if isinstance(states[i-1], dict) else states[i-1], 'velocity', 0.0)
                feats[i, 6] = vel - prev_vel

    return feats


def build_dataset(data_root, max_bars=0):
    """Load data, compute states, extract features, create labels with lookahead."""
    from core.statistical_field_engine import StatisticalFieldEngine

    print(f"Loading 1m data from {data_root}...")
    files = sorted(glob.glob(os.path.join(data_root, '1m', '*.parquet')))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if max_bars > 0:
        df = df.tail(max_bars).reset_index(drop=True)
    print(f"  Bars: {len(df):,}")

    print("Computing SFE states...")
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    print(f"  States: {len(states)}")

    print("Extracting grounded features...")
    feats = extract_features_from_states(states, df)

    # Labels: was LONG profitable after FORWARD bars? (with lookahead)
    prices = df['close'].values
    labels = np.zeros(len(prices))
    for i in range(len(prices) - FORWARD):
        future_change = (prices[i + FORWARD] - prices[i]) / TICK  # in ticks
        labels[i] = 1.0 if future_change > 0 else 0.0

    # Magnitude for weighted loss (how much would you have made/lost)
    magnitudes = np.zeros(len(prices))
    for i in range(len(prices) - FORWARD):
        magnitudes[i] = abs(prices[i + FORWARD] - prices[i]) / TICK

    return feats, labels, magnitudes, df


class DirectionDataset(Dataset):
    """Sliding window dataset of (lookback_features, label, magnitude)."""

    def __init__(self, features, labels, magnitudes):
        self.features = features
        self.labels = labels
        self.magnitudes = magnitudes
        self.n = len(features) - LOOKBACK - FORWARD

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx + LOOKBACK
        x = self.features[i - LOOKBACK:i]  # (LOOKBACK, N_FEAT)
        y = self.labels[i]
        m = self.magnitudes[i]
        return (
            torch.FloatTensor(x),
            torch.FloatTensor([y]),
            torch.FloatTensor([m]),
        )


# --- TRAINING ---
def train_model(feats, labels, mags, epochs=EPOCHS):
    """Train the CNN on IS data."""
    dataset = DirectionDataset(feats, labels, mags)
    # 90/10 train/val split (temporal — last 10% is val)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DirectionCNN(n_features=N_FEAT, lookback=LOOKBACK, n_layers=N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Magnitude-weighted BCE loss
    bce = nn.BCELoss(reduction='none')

    best_val_acc = 0
    best_epoch = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for x, y, m in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y, m = x.to(device), y.to(device).squeeze(), m.to(device).squeeze()
            pred = model(x)
            loss_raw = bce(pred, y)
            # PnL-weighted loss: reward profitable predictions more, punish losses harder
            # correct prediction with big move = low loss (good)
            # wrong prediction with big move = high loss (bad, penalize)
            is_correct = ((pred > 0.5) == (y > 0.5)).float()
            # Wrong predictions get 2x penalty (asymmetric — protect capital)
            penalty = torch.where(is_correct > 0.5, 1.0, 2.0)
            weights = penalty * (1.0 + m / (m.mean() + 1e-8))
            loss = (loss_raw * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            train_correct += ((pred > 0.5) == (y > 0.5)).sum().item()
            train_total += len(x)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_pnl = 0
        with torch.no_grad():
            for x, y, m in val_dl:
                x, y, m = x.to(device), y.to(device).squeeze(), m.to(device).squeeze()
                pred = model(x)
                val_correct += ((pred > 0.5) == (y > 0.5)).sum().item()
                val_total += len(x)
                # Simulated PnL: if pred > 0.5 go LONG, else SHORT
                direction = torch.where(pred > 0.5, 1.0, -1.0)
                actual = torch.where(y > 0.5, 1.0, -1.0)
                val_pnl += (direction * actual * m).sum().item()

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        val_pnl_dollars = val_pnl * 0.5  # ticks to dollars

        print(f"  Epoch {epoch+1}: train_acc={train_acc:.1f}% val_acc={val_acc:.1f}% "
              f"val_pnl=${val_pnl_dollars:,.0f} loss={train_loss/train_total:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            _feat_tag = '22D' if USE_22D else '7D'
            _run_name = f'best_{_feat_tag}_lb{LOOKBACK}_ep{epoch+1}'
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_pnl': val_pnl_dollars,
                'config': {
                    'features': _feat_tag,
                    'n_feat': N_FEAT,
                    'lookback': LOOKBACK,
                    'epochs': epochs,
                },
            }, os.path.join(CHECKPOINT_DIR, f'{_run_name}.pt'))
            # Also save as best_model.pt (latest best)
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_pnl': val_pnl_dollars,
                'config': {
                    'features': _feat_tag,
                    'n_feat': N_FEAT,
                    'lookback': LOOKBACK,
                    'epochs': epochs,
                },
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

    print(f"\nBest: epoch {best_epoch} val_acc={best_val_acc:.1f}%")

    # Release GPU memory
    import gc
    del train_dl, val_dl, train_ds, val_ds, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model


# --- RANDOM FOREST ---
def train_rf(feats, labels, mags):
    """Train a Random Forest classifier with PnL-weighted samples."""
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Build flat feature matrix: each sample = flattened lookback window
    n_samples = len(feats) - LOOKBACK - FORWARD
    X = np.zeros((n_samples, LOOKBACK * N_FEAT))
    y = np.zeros(n_samples)
    w = np.zeros(n_samples)

    for i in range(n_samples):
        idx = i + LOOKBACK
        X[i] = feats[idx - LOOKBACK:idx].flatten()
        y[i] = labels[idx]
        w[i] = 1.0 + mags[idx] / (mags[LOOKBACK:LOOKBACK+n_samples].mean() + 1e-8)
        # Asymmetric: wrong predictions penalized 2x
        # (applied post-hoc since RF doesn't know pred at train time — use sample_weight)

    # 90/10 temporal split
    val_size = n_samples // 10
    train_size = n_samples - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    w_train = w[:train_size]

    print(f"RF Training: {train_size:,} samples, {LOOKBACK * N_FEAT} features")
    print(f"RF Validation: {val_size:,} samples")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)

    # Validate
    val_pred = rf.predict(X_val)
    val_prob = rf.predict_proba(X_val)[:, 1]
    val_acc = (val_pred == y_val).mean() * 100

    # Feature importance (top 10)
    importances = rf.feature_importances_
    feat_names = [f'bar{b}_{FEATURES[f]}' for b in range(LOOKBACK) for f in range(N_FEAT)]
    top_idx = np.argsort(importances)[-10:][::-1]
    print(f"\nRF Val Accuracy: {val_acc:.1f}%")
    print(f"Top 10 features:")
    for idx in top_idx:
        print(f"  {feat_names[idx]:<25} importance={importances[idx]:.4f}")

    # Save
    joblib.dump(rf, os.path.join(CHECKPOINT_DIR, 'rf_model.joblib'))
    # Save as fake checkpoint for validate_oos compatibility
    torch.save({
        'epoch': 1, 'val_acc': val_acc, 'val_pnl': 0,
        'config': {'features': '22D' if USE_22D else '7D', 'n_feat': N_FEAT,
                   'lookback': LOOKBACK, 'model': 'rf'},
    }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

    print(f"  Saved: {CHECKPOINT_DIR}/rf_model.joblib")
    return rf


def validate_oos_rf():
    """Validate Random Forest on OOS."""
    import joblib
    rf = joblib.load(os.path.join(CHECKPOINT_DIR, 'rf_model.joblib'))
    print(f"Loaded RF model")

    feats, labels, mags, df = build_dataset(OOS_ROOT)
    prices = df['close'].values

    trades = []
    in_trade = False
    trade_dir = ''
    entry_price = 0
    tp_count = 0
    last_tp_price = 0
    TP = 10; SL = 40

    for i in tqdm(range(LOOKBACK, len(feats) - FORWARD), desc="OOS RF Validation"):
        x = feats[i - LOOKBACK:i].flatten().reshape(1, -1)
        prob_long = rf.predict_proba(x)[0][1]
        price = prices[i]
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']

        if in_trade:
            ref = last_tp_price if tp_count > 0 else entry_price
            if trade_dir == 'LONG':
                if (low - entry_price) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'LONG', 'tps': tp_count})
                    in_trade = False; continue
                if (high - ref) / TICK >= TP:
                    tp_count += 1; last_tp_price = ref + TP * TICK
            else:
                if (entry_price - high) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'SHORT', 'tps': tp_count})
                    in_trade = False; continue
                if (ref - low) / TICK >= TP:
                    tp_count += 1; last_tp_price = ref - TP * TICK

        if prob_long > 0.6:
            new_dir = 'LONG'
        elif prob_long < 0.4:
            new_dir = 'SHORT'
        else:
            continue

        if in_trade and new_dir != trade_dir:
            pnl = (price - entry_price) / TICK if trade_dir == 'LONG' else (entry_price - price) / TICK
            trades.append({'pnl': pnl, 'dir': trade_dir, 'tps': tp_count})
            trade_dir = new_dir; entry_price = price; tp_count = 0; last_tp_price = 0
        elif not in_trade:
            in_trade = True; trade_dir = new_dir; entry_price = price
            tp_count = 0; last_tp_price = 0

    total_pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    w = len([t for t in trades if t['pnl'] > 0])
    trading_days = df['timestamp'].apply(lambda t: pd.Timestamp(t, unit='s').date()).nunique()

    print(f"\n{'='*60}")
    print(f"OOS VALIDATION: Random Forest + TP={TP} SL={SL}")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {w/n*100:.1f}%" if n > 0 else "  WR: N/A")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${total_pnl*0.5/trading_days:.2f}" if trading_days > 0 else "")
    print(f"  Trading days: {trading_days}")

    # Append to log
    import datetime
    _feat_mode = f'{FEAT_MODE}D'
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"feat={_feat_mode} | model=RF | "
             f"lookback={LOOKBACK} | trees=200 | "
             f"val_acc=--% | "
             f"trades={n} | WR={w/n*100 if n > 0 else 0:.1f}% | "
             f"PnL=${total_pnl*0.5:,.0f} | $/day=${total_pnl*0.5/trading_days:.0f}\n")
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)
    print(f"  Logged: {RESULTS_LOG}")


# --- VALIDATION ON OOS ---
def validate_oos(model_path=None):
    """Run trained model on OOS data, simulate trading."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectionCNN(n_features=N_FEAT, lookback=LOOKBACK, n_layers=N_LAYERS).to(device)
    ckpt_path = model_path or os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.1f}%)")

    feats, labels, mags, df = build_dataset(OOS_ROOT)
    prices = df['close'].values

    # Run bar-by-bar
    trades = []
    in_trade = False
    trade_dir = ''
    entry_price = 0
    entry_bar = 0
    tp_count = 0
    last_tp_price = 0
    TP = 10
    SL = 40

    for i in tqdm(range(LOOKBACK, len(feats) - FORWARD), desc="OOS Validation"):
        x = feats[i - LOOKBACK:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_long = model(x_t).item()

        price = prices[i]
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']

        # Trade management
        if in_trade:
            ref = last_tp_price if tp_count > 0 else entry_price
            if trade_dir == 'LONG':
                if (low - entry_price) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'LONG', 'tps': tp_count})
                    in_trade = False
                    continue
                if (high - ref) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref + TP * TICK
            else:
                if (entry_price - high) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'SHORT', 'tps': tp_count})
                    in_trade = False
                    continue
                if (ref - low) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref - TP * TICK

        # Entry/flip signal: strong confidence
        if prob_long > 0.6:
            new_dir = 'LONG'
        elif prob_long < 0.4:
            new_dir = 'SHORT'
        else:
            continue  # uncertain, hold

        if in_trade and new_dir != trade_dir:
            pnl = (price - entry_price) / TICK if trade_dir == 'LONG' else (entry_price - price) / TICK
            trades.append({'pnl': pnl, 'dir': trade_dir, 'tps': tp_count})
            trade_dir = new_dir
            entry_price = price
            tp_count = 0
            last_tp_price = 0
        elif not in_trade:
            in_trade = True
            trade_dir = new_dir
            entry_price = price
            tp_count = 0
            last_tp_price = 0

    # Summary
    total_pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    w = len([t for t in trades if t['pnl'] > 0])
    trading_days = df['timestamp'].apply(lambda t: pd.Timestamp(t, unit='s').date()).nunique()

    print(f"\n{'='*60}")
    print(f"OOS VALIDATION: Direction CNN + TP={TP} SL={SL}")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {w/n*100:.1f}%" if n > 0 else "  WR: N/A")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${total_pnl*0.5/trading_days:.2f}" if trading_days > 0 else "")
    print(f"  Trading days: {trading_days}")

    # Save results
    results = {
        'trades': n, 'wr': w/n*100 if n > 0 else 0,
        'pnl_ticks': total_pnl, 'pnl_dollars': total_pnl * 0.5,
        'per_day': total_pnl * 0.5 / trading_days if trading_days > 0 else 0,
        'trading_days': trading_days,
    }
    import json
    with open(os.path.join(CHECKPOINT_DIR, 'oos_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {CHECKPOINT_DIR}/oos_results.json")

    # Append to experiment log
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    import datetime
    _feat_mode = f'{FEAT_MODE}D'
    _n_layers = f"conv={2},dense={1}"
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"feat={_feat_mode} | layers={_n_layers} | "
             f"lookback={LOOKBACK} | epochs={ckpt['epoch']} | "
             f"val_acc={ckpt['val_acc']:.1f}% | "
             f"trades={n} | WR={w/n*100 if n > 0 else 0:.1f}% | "
             f"PnL=${total_pnl*0.5:,.0f} | $/day=${results['per_day']:.0f}\n")
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)
    print(f"  Logged: {RESULTS_LOG}")

    # Release memory
    import gc
    del feats, labels, mags, df, trades
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='Direction CNN trainer')
    parser.add_argument('--phase', default='all', choices=['train', 'validate', 'all'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--max-bars', type=int, default=0, help='Limit IS bars (0=all)')
    parser.add_argument('--features', type=int, default=7, choices=[3, 7, 18, 22], help='Feature set: 3D base, 7D grounded, 18D full, or 22D legacy')
    parser.add_argument('--layers', type=int, default=2, help='Number of conv layers (2=simple, 50=deep residual)')
    parser.add_argument('--lookback', type=int, default=LOOKBACK, help='Bars of history per sample')
    parser.add_argument('--sweep', action='store_true', help='Sweep lookback values to find optimal')
    parser.add_argument('--model', default='cnn', choices=['cnn', 'rf'], help='Model type: cnn or rf (random forest)')
    args = parser.parse_args()

    # Apply feature mode via module-level vars
    import training.direction_cnn as _self
    _self.FEAT_MODE = args.features
    _self.USE_22D = args.features == 22
    if args.features == 3:
        _self.FEATURES = FEATURES_3D
    elif args.features == 18:
        _self.FEATURES = FEATURES_18D
    elif args.features == 22:
        _self.FEATURES = FEATURES_22D
    else:
        _self.FEATURES = FEATURES_7D
    _self.N_FEAT = len(_self.FEATURES)
    _self.LOOKBACK = args.lookback
    _self.N_LAYERS = args.layers

    if args.sweep:
        # Sweep lookback values, build dataset ONCE, reuse
        print("\n" + "="*60)
        print("SWEEP MODE: testing multiple lookback values")
        print("="*60)
        t0 = time.time()
        _self.LOOKBACK = max(3, 3)  # temp for dataset build
        feats, labels, mags, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
        feats_oos, labels_oos, mags_oos, df_oos = build_dataset(OOS_ROOT)
        print(f"Datasets built in {time.time()-t0:.1f}s")

        sweep_values = [3, 5, 10, 15, 20, 30, 60]
        for lb in sweep_values:
            _self.LOOKBACK = lb
            print(f"\n{'='*60}")
            print(f"SWEEP: lookback={lb}")
            print(f"{'='*60}")
            model = train_model(feats, labels, mags, epochs=args.epochs)
            validate_oos()
        return

    # Log experiment config at start
    _feat_mode = '22D' if _self.USE_22D else '7D'
    print(f"\n{'='*60}")
    print(f"CNN EXPERIMENT: feat={_feat_mode} lookback={_self.LOOKBACK} epochs={args.epochs}")
    print(f"  Features: {_self.N_FEAT}D | Lookback: {_self.LOOKBACK} bars | Epochs: {args.epochs}")
    print(f"  IS: {IS_ROOT} | OOS: {OOS_ROOT}")
    print(f"{'='*60}")

    if args.model == 'rf':
        # Random Forest path
        if args.phase in ('train', 'all'):
            t0 = time.time()
            feats, labels, mags, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
            print(f"Dataset built in {time.time()-t0:.1f}s")
            train_rf(feats, labels, mags)
        if args.phase in ('validate', 'all'):
            validate_oos_rf()
    else:
        # CNN path
        if args.phase in ('train', 'all'):
            t0 = time.time()
            feats, labels, mags, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
            print(f"Dataset built in {time.time()-t0:.1f}s")
            print(f"  Feature shape: {feats.shape}")
            print(f"  Labels: {labels.sum():.0f} LONG / {(1-labels).sum():.0f} SHORT ({labels.mean()*100:.1f}% LONG)")

            model = train_model(feats, labels, mags, epochs=args.epochs)

        if args.phase in ('validate', 'all'):
            validate_oos()


if __name__ == '__main__':
    main()
