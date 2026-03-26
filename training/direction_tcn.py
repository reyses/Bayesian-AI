"""
Direction TCN — Temporal Convolutional Network for trade direction.

WaveNet-style dilated causal convolutions with residual connections.
Exponential receptive field: dilation [1,2,4,8,16,32,64] = 127 bars.
The model learns WHICH lookback distance matters for each prediction.

Architecture:
  Input: (batch, lookback, n_features)
  → Input projection (n_features → channels)
  → N dilated causal conv blocks with residual connections
  → Global pooling
  → 3-head output:
    - Direction: P(LONG) sigmoid
    - Magnitude: expected ticks (regression)
    - Confidence: 0-1 how sure (sigmoid)

Usage:
  python -m training.direction_tcn --phase all
  python -m training.direction_tcn --phase all --features 18 --lookback 60
  python -m training.direction_tcn --phase all --features 7 --dilation-depth 8
"""
import argparse
import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

# Re-use feature extraction from direction_cnn
from training.direction_cnn import (
    FEATURES_7D, FEATURES_18D, FEATURES_22D, FEATURES_3D,
    extract_features_from_states, build_dataset, DirectionDataset,
    CHECKPOINT_DIR as CNN_CHECKPOINT_DIR,
    IS_ROOT, OOS_ROOT, TICK,
)

# --- CONFIG ---
LOOKBACK = 30
FORWARD = 10
N_FEAT = 7
FEAT_MODE = 7
FEATURES = FEATURES_7D
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
CHANNELS = 32            # width of each conv layer
DILATION_DEPTH = 7       # number of dilation levels: [1,2,4,8,16,32,64]
CHECKPOINT_DIR = 'checkpoints/direction_tcn'
RESULTS_LOG = 'reports/findings/cnn_experiment_log.txt'


# --- TCN BLOCKS ---
class CausalConv1d(nn.Module):
    """Causal convolution — can't see future bars."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        # Remove the future-looking padding (causal: only left padding)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """One TCN residual block: causal conv → batch norm → ReLU → dropout → skip."""
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = out + residual  # skip connection
        return out


class DirectionTCN(nn.Module):
    """
    Temporal Convolutional Network with 3-head output.

    Receptive field = 2^dilation_depth bars (e.g., depth=7 → 128 bars).
    The model learns which lookback distance matters.
    """
    def __init__(self, n_features=7, channels=CHANNELS, dilation_depth=DILATION_DEPTH,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.channels = channels

        # Input projection: n_features → channels
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(channels)

        # Dilated causal conv blocks: dilation = [1, 2, 4, 8, 16, 32, 64, ...]
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(dilation_depth)
        ])

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 3-head output
        self.head_direction = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid(),
        )
        self.head_magnitude = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )
        self.head_confidence = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid(),
        )

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        receptive = 2 ** dilation_depth
        print(f"[TCN] {trainable:,} trainable params | "
              f"receptive field: {receptive} bars | "
              f"{dilation_depth} dilation levels | {channels} channels")

    def forward(self, x):
        # x: (batch, lookback, n_features) → (batch, n_features, lookback)
        x = x.transpose(1, 2)

        # Input projection
        x = torch.relu(self.input_bn(self.input_proj(x)))

        # Dilated causal conv blocks
        for block in self.tcn_blocks:
            x = block(x)

        # Pool to single vector
        x = self.pool(x).squeeze(-1)  # (batch, channels)

        # 3 heads
        direction = self.head_direction(x).squeeze(-1)       # P(LONG)
        magnitude = self.head_magnitude(x).squeeze(-1)       # expected ticks
        confidence = self.head_confidence(x).squeeze(-1)     # 0-1

        return direction, magnitude, confidence


# --- DATASET ---
class TCNDataset(Dataset):
    """Sliding window with direction + magnitude labels."""
    def __init__(self, features, labels, magnitudes):
        self.features = features
        self.labels = labels
        self.magnitudes = magnitudes
        self.n = len(features) - LOOKBACK - FORWARD

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx + LOOKBACK
        x = self.features[i - LOOKBACK:i]
        y_dir = self.labels[i]
        y_mag = self.magnitudes[i]
        # Confidence label: 1.0 if magnitude > median, 0.0 if small move
        return (
            torch.FloatTensor(x),
            torch.FloatTensor([y_dir]),
            torch.FloatTensor([y_mag]),
        )


# --- TRAINING ---
def train_tcn(feats, labels, mags, epochs=EPOCHS):
    """Train the TCN on IS data."""
    dataset = TCNDataset(feats, labels, mags)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DirectionTCN(n_features=N_FEAT, channels=CHANNELS,
                         dilation_depth=DILATION_DEPTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    best_val_pnl = -float('inf')
    best_epoch = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Magnitude median for confidence labels
    mag_median = np.median(mags[mags > 0]) if (mags > 0).any() else 10.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y_dir, y_mag in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = x.to(device)
            y_dir = y_dir.to(device).squeeze()
            y_mag = y_mag.to(device).squeeze()

            pred_dir, pred_mag, pred_conf = model(x)

            # Direction loss: PnL-weighted BCE
            loss_dir = bce(pred_dir, y_dir)
            is_correct = ((pred_dir > 0.5) == (y_dir > 0.5)).float()
            penalty = torch.where(is_correct > 0.5, 1.0, 2.0)
            weights = penalty * (1.0 + y_mag / (y_mag.mean() + 1e-8))
            loss_dir = (loss_dir * weights).mean()

            # Magnitude loss: MSE on expected ticks
            loss_mag = mse(pred_mag, y_mag).mean() * 0.01  # scale down

            # Confidence loss: should be high when magnitude is large
            y_conf = (y_mag > mag_median).float()
            loss_conf = bce(pred_conf, y_conf).mean() * 0.1

            loss = loss_dir + loss_mag + loss_conf

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(x)
            train_correct += ((pred_dir > 0.5) == (y_dir > 0.5)).sum().item()
            train_total += len(x)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_pnl = 0
        val_conf_trades = 0

        with torch.no_grad():
            for x, y_dir, y_mag in val_dl:
                x = x.to(device)
                y_dir = y_dir.to(device).squeeze()
                y_mag = y_mag.to(device).squeeze()

                pred_dir, pred_mag, pred_conf = model(x)

                val_correct += ((pred_dir > 0.5) == (y_dir > 0.5)).sum().item()
                val_total += len(x)

                # PnL: trade only when confident
                confident = pred_conf > 0.5
                direction = torch.where(pred_dir > 0.5, 1.0, -1.0)
                actual = torch.where(y_dir > 0.5, 1.0, -1.0)
                pnl = direction * actual * y_mag * confident.float()
                val_pnl += pnl.sum().item()
                val_conf_trades += confident.sum().item()

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        val_pnl_dollars = val_pnl * 0.5
        scheduler.step(-val_pnl)  # reduce LR if PnL plateaus

        print(f"  Epoch {epoch+1}: train_acc={train_acc:.1f}% val_acc={val_acc:.1f}% "
              f"val_pnl=${val_pnl_dollars:,.0f} conf_trades={val_conf_trades} "
              f"loss={train_loss/train_total:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_pnl_dollars > best_val_pnl:
            best_val_pnl = val_pnl_dollars
            best_epoch = epoch + 1
            _feat_tag = f'{FEAT_MODE}D'
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_pnl': val_pnl_dollars,
                'config': {
                    'features': _feat_tag, 'n_feat': N_FEAT,
                    'lookback': LOOKBACK, 'channels': CHANNELS,
                    'dilation_depth': DILATION_DEPTH, 'model': 'tcn',
                },
            }, os.path.join(CHECKPOINT_DIR, f'best_tcn_{_feat_tag}_lb{LOOKBACK}.pt'))
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1, 'val_acc': val_acc, 'val_pnl': val_pnl_dollars,
                'config': {
                    'features': _feat_tag, 'n_feat': N_FEAT,
                    'lookback': LOOKBACK, 'channels': CHANNELS,
                    'dilation_depth': DILATION_DEPTH, 'model': 'tcn',
                },
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

    print(f"\nBest: epoch {best_epoch} val_pnl=${best_val_pnl:,.0f}")
    return model


# --- OOS VALIDATION ---
def validate_oos():
    """Run trained TCN on OOS data, simulate trading with 3-head output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), map_location=device)
    cfg = ckpt.get('config', {})

    model = DirectionTCN(
        n_features=cfg.get('n_feat', N_FEAT),
        channels=cfg.get('channels', CHANNELS),
        dilation_depth=cfg.get('dilation_depth', DILATION_DEPTH),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded TCN from epoch {ckpt['epoch']} (val_pnl=${ckpt['val_pnl']:,.0f})")

    feats, labels, mags, df = build_dataset(OOS_ROOT)
    prices = df['close'].values

    trades = []
    in_trade = False
    trade_dir = ''
    entry_price = 0
    tp_count = 0
    last_tp_price = 0
    TP = 10
    SL = 40

    for i in tqdm(range(LOOKBACK, len(feats) - FORWARD), desc="OOS TCN Validation"):
        x = feats[i - LOOKBACK:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            prob_long, pred_mag, pred_conf = model(x_t)
            prob_long = prob_long.item()
            pred_conf = pred_conf.item()

        price = prices[i]
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']

        # Trade management
        if in_trade:
            ref = last_tp_price if tp_count > 0 else entry_price
            if trade_dir == 'LONG':
                if (low - entry_price) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'LONG', 'tps': tp_count, 'conf': pred_conf})
                    in_trade = False
                    continue
                if (high - ref) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref + TP * TICK
            else:
                if (entry_price - high) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'SHORT', 'tps': tp_count, 'conf': pred_conf})
                    in_trade = False
                    continue
                if (ref - low) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref - TP * TICK

        # Trade on direction (confidence logged but not gated — same as CNN)
        if prob_long > 0.6:
            new_dir = 'LONG'
        elif prob_long < 0.4:
            new_dir = 'SHORT'
        else:
            continue

        if in_trade and new_dir != trade_dir:
            pnl = (price - entry_price) / TICK if trade_dir == 'LONG' else (entry_price - price) / TICK
            trades.append({'pnl': pnl, 'dir': trade_dir, 'tps': tp_count, 'conf': pred_conf})
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
    print(f"OOS VALIDATION: TCN ({FEAT_MODE}D, {DILATION_DEPTH} dilations, {CHANNELS}ch)")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {w/n*100:.1f}%" if n > 0 else "  WR: N/A")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${total_pnl*0.5/trading_days:.2f}" if trading_days > 0 else "")
    print(f"  Trading days: {trading_days}")
    if n > 0:
        print(f"  Avg confidence: {np.mean([t['conf'] for t in trades]):.2f}")
        print(f"  Avg TPs/trade: {np.mean([t['tps'] for t in trades]):.1f}")

    # Append to shared experiment log
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"feat={FEAT_MODE}D | model=TCN | "
             f"lookback={LOOKBACK} | dilations={DILATION_DEPTH} | ch={CHANNELS} | "
             f"val_acc={ckpt.get('val_acc', 0):.1f}% | "
             f"trades={n} | WR={w/n*100 if n > 0 else 0:.1f}% | "
             f"PnL=${total_pnl*0.5:,.0f} | $/day=${total_pnl*0.5/trading_days if trading_days > 0 else 0:.0f}\n")
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)
    print(f"  Logged: {RESULTS_LOG}")

    # Save
    import json
    results = {
        'model': 'tcn', 'trades': n,
        'wr': w/n*100 if n > 0 else 0,
        'pnl_ticks': total_pnl, 'pnl_dollars': total_pnl * 0.5,
        'per_day': total_pnl * 0.5 / trading_days if trading_days > 0 else 0,
        'trading_days': trading_days,
        'config': cfg,
    }
    with open(os.path.join(CHECKPOINT_DIR, 'oos_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='Direction TCN trainer')
    parser.add_argument('--phase', default='all', choices=['train', 'validate', 'all'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--features', type=int, default=7, choices=[3, 7, 18, 22])
    parser.add_argument('--lookback', type=int, default=LOOKBACK)
    parser.add_argument('--channels', type=int, default=CHANNELS)
    parser.add_argument('--dilation-depth', type=int, default=DILATION_DEPTH)
    parser.add_argument('--max-bars', type=int, default=0)
    args = parser.parse_args()

    # Apply settings
    import training.direction_tcn as _self
    _self.FEAT_MODE = args.features
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
    _self.CHANNELS = args.channels
    _self.DILATION_DEPTH = args.dilation_depth

    _feat_mode = f'{_self.FEAT_MODE}D'
    receptive = 2 ** args.dilation_depth
    print(f"\n{'='*60}")
    print(f"TCN EXPERIMENT: feat={_feat_mode} lookback={args.lookback} "
          f"dilations={args.dilation_depth} ch={args.channels}")
    print(f"  Receptive field: {receptive} bars | Epochs: {args.epochs}")
    print(f"{'='*60}")

    if args.phase in ('train', 'all'):
        t0 = time.time()
        feats, labels, mags, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
        print(f"Dataset built in {time.time()-t0:.1f}s")
        print(f"  Feature shape: {feats.shape}")
        model = train_tcn(feats, labels, mags, epochs=args.epochs)

    if args.phase in ('validate', 'all'):
        validate_oos()


if __name__ == '__main__':
    main()
