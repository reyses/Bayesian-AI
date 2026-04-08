"""
Direction TCN — Temporal Convolutional Network for trade direction.

WaveNet-style dilated causal convolutions with residual connections.
Exponential receptive field: dilation [1,2,4,8,16,32,64] = 127 bars.

Architecture:
  Input: (batch, lookback, n_features)
  -> Input projection (n_features -> channels)
  -> N dilated causal conv blocks with residual connections
  -> Global pooling
  -> 3-head output: Direction (logits), Magnitude (regression), Confidence (sigmoid)

Usage:
  python -m training.direction_tcn --phase all
  python -m training.direction_tcn --phase all --features 18 --lookback 60
  python -m training.direction_tcn --phase all --features 7 --dilation-depth 8
"""
import argparse
import dataclasses
import gc
import json
import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from training.direction_cnn import (
    FEATURES_7D, FEATURES_18D, FEATURES_22D, FEATURES_3D,
    extract_features_from_states, build_dataset,
    IS_ROOT, OOS_ROOT, TICK,
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Shared log (CNN and TCN both append, tagged by model type)
RESULTS_LOG = 'reports/findings/experiment_log.txt'


# --- CONFIG ---
@dataclasses.dataclass
class TCNConfig:
    feat_mode: int = 7
    n_feat: int = 7
    lookback: int = 30
    forward: int = 10
    channels: int = 32
    dilation_depth: int = 7
    batch_size: int = 512
    epochs: int = 30
    lr: float = 1e-3
    dropout: float = 0.2
    kernel_size: int = 3
    checkpoint_dir: str = 'checkpoints/direction_tcn'


# --- TCN BLOCKS ---
class CausalConv1d(nn.Module):
    """Causal convolution -- can't see future bars."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Residual block: causal conv -> BN -> ReLU -> dropout -> skip."""
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
        return out + residual


class DirectionTCN(nn.Module):
    """TCN with 3-head output. Direction uses logits (BCEWithLogitsLoss)."""

    def __init__(self, cfg: TCNConfig):
        super().__init__()
        n_feat = cfg.n_feat
        ch = cfg.channels

        self.input_proj = nn.Conv1d(n_feat, ch, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(ch)

        self.tcn_blocks = nn.ModuleList([
            TCNBlock(ch, cfg.kernel_size, dilation=2**i, dropout=cfg.dropout)
            for i in range(cfg.dilation_depth)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Direction head outputs LOGITS (no sigmoid -- BCEWithLogitsLoss handles it)
        self.head_direction = nn.Sequential(
            nn.Linear(ch, ch // 2), nn.ReLU(), nn.Linear(ch // 2, 1),
        )
        self.head_magnitude = nn.Sequential(
            nn.Linear(ch, ch // 2), nn.ReLU(), nn.Linear(ch // 2, 1),
        )
        self.head_confidence = nn.Sequential(
            nn.Linear(ch, ch // 2), nn.ReLU(), nn.Linear(ch // 2, 1), nn.Sigmoid(),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        receptive = 2 ** cfg.dilation_depth
        print(f"[TCN] {trainable:,} params | receptive: {receptive} bars | "
              f"{cfg.dilation_depth} dilations | {ch} channels")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.input_bn(self.input_proj(x)))
        for block in self.tcn_blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)

        dir_logits = self.head_direction(x).squeeze(-1)   # raw logits
        magnitude = self.head_magnitude(x).squeeze(-1)
        confidence = self.head_confidence(x).squeeze(-1)
        return dir_logits, magnitude, confidence


# --- DATASET ---
class TCNDataset(Dataset):
    def __init__(self, features, labels, magnitudes, cfg: TCNConfig):
        self.features = features
        self.labels = labels
        self.magnitudes = magnitudes
        self.lookback = cfg.lookback
        self.forward = cfg.forward
        self.n = len(features) - cfg.lookback - cfg.forward

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx + self.lookback
        x = self.features[i - self.lookback:i]
        return (
            torch.FloatTensor(x),
            torch.FloatTensor([self.labels[i]]),
            torch.FloatTensor([self.magnitudes[i]]),
        )


# --- TRAINING ---
def train_tcn(feats, labels, mags, cfg: TCNConfig):
    dataset = TCNDataset(feats, labels, mags, cfg)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    _nw = min(2, os.cpu_count() or 1)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=_nw, persistent_workers=_nw > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=_nw, persistent_workers=_nw > 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DirectionTCN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ARCH 1: pos_weight for class imbalance
    n_long = (labels > 0.5).sum()
    n_short = (labels <= 0.5).sum()
    pos_weight = torch.tensor([n_short / (n_long + 1e-8)]).to(device)
    bce_logits = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    print(f"  Class balance: {n_long} LONG / {n_short} SHORT (pos_weight={pos_weight.item():.3f})")

    # ARCH 2: Huber loss for magnitude
    huber = nn.SmoothL1Loss(reduction='none', beta=20.0)
    bce = nn.BCELoss(reduction='none')

    mag_median = np.median(mags[mags > 0]) if (mags > 0).any() else 10.0

    # ARCH 3: Sharpe-based checkpoint selection
    best_val_sharpe = -float('inf')
    best_val_pnl = -float('inf')
    best_epoch = 0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        loss_dir_epoch = 0
        loss_mag_epoch = 0
        loss_conf_epoch = 0

        for x, y_dir, y_mag in tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False):
            x = x.to(device)
            y_dir = y_dir.to(device).squeeze()
            y_mag = y_mag.to(device).squeeze()

            dir_logits, pred_mag, pred_conf = model(x)

            # Direction loss: BCEWithLogitsLoss (no sigmoid needed) + magnitude weighting
            loss_dir = bce_logits(dir_logits, y_dir)
            weights = 1.0 + y_mag / (y_mag.mean() + 1e-8)
            loss_dir = (loss_dir * weights).mean()

            # ARCH 2: Huber loss for magnitude
            loss_mag = huber(pred_mag, y_mag).mean() * 0.05

            # Confidence loss
            y_conf = (y_mag > mag_median).float()
            loss_conf = bce(pred_conf, y_conf).mean() * 0.1

            loss = loss_dir + loss_mag + loss_conf

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(x)
            loss_dir_epoch += loss_dir.item() * len(x)
            loss_mag_epoch += loss_mag.item() * len(x)
            loss_conf_epoch += loss_conf.item() * len(x)
            # Apply sigmoid for accuracy check (logits -> probability)
            pred_prob = torch.sigmoid(dir_logits)
            train_correct += ((pred_prob > 0.5) == (y_dir > 0.5)).sum().item()
            train_total += len(x)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_trade_pnls = []

        with torch.no_grad():
            for x, y_dir, y_mag in val_dl:
                x = x.to(device)
                y_dir = y_dir.to(device).squeeze()
                y_mag = y_mag.to(device).squeeze()

                dir_logits, pred_mag, pred_conf = model(x)
                pred_prob = torch.sigmoid(dir_logits)

                val_correct += ((pred_prob > 0.5) == (y_dir > 0.5)).sum().item()
                val_total += len(x)

                # Per-trade PnL for Sharpe
                direction = torch.where(pred_prob > 0.5, 1.0, -1.0)
                actual = torch.where(y_dir > 0.5, 1.0, -1.0)
                trade_pnl = direction * actual * y_mag
                val_trade_pnls.extend(trade_pnl.cpu().numpy().tolist())

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        arr = np.array(val_trade_pnls)
        val_pnl_dollars = arr.sum() * 0.5
        val_sharpe = arr.mean() / (arr.std() + 1e-8) if len(arr) > 1 else 0.0
        scheduler.step(-val_sharpe)

        print(f"  Epoch {epoch+1}: train_acc={train_acc:.1f}% val_acc={val_acc:.1f}% "
              f"val_pnl=${val_pnl_dollars:,.0f} sharpe={val_sharpe:.3f} "
              f"loss_dir={loss_dir_epoch/train_total:.4f} "
              f"loss_mag={loss_mag_epoch/train_total:.4f} "
              f"loss_conf={loss_conf_epoch/train_total:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # ARCH 3: checkpoint on Sharpe, not raw PnL
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_val_pnl = val_pnl_dollars
            best_epoch = epoch + 1
            _feat_tag = f'{cfg.feat_mode}D'
            _save = {
                'model_state': model.state_dict(),
                'epoch': epoch + 1, 'val_acc': val_acc,
                'val_pnl': val_pnl_dollars, 'val_sharpe': val_sharpe,
                'config': dataclasses.asdict(cfg),
            }
            torch.save(_save, os.path.join(cfg.checkpoint_dir, f'best_tcn_{_feat_tag}_lb{cfg.lookback}.pt'))
            torch.save(_save, os.path.join(cfg.checkpoint_dir, 'best_model.pt'))

    print(f"\nBest: epoch {best_epoch} sharpe={best_val_sharpe:.3f} pnl=${best_val_pnl:,.0f}")

    del train_dl, val_dl, train_ds, val_ds, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model


# --- OOS VALIDATION ---
def validate_oos(cfg: TCNConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(os.path.join(cfg.checkpoint_dir, 'best_model.pt'),
                       map_location=device, weights_only=False)
    saved_cfg = ckpt.get('config', {})

    # Rebuild config from checkpoint
    _cfg = TCNConfig(**{k: v for k, v in saved_cfg.items() if k in TCNConfig.__dataclass_fields__})
    model = DirectionTCN(_cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded TCN from epoch {ckpt['epoch']} "
          f"(sharpe={ckpt.get('val_sharpe', 0):.3f} pnl=${ckpt.get('val_pnl', 0):,.0f})")

    feats, labels, mags, df = build_dataset(OOS_ROOT)
    prices = df['close'].values

    trades = []
    in_trade = False
    trade_dir = ''
    entry_price = 0
    entry_conf = 0.0
    tp_count = 0
    last_tp_price = 0
    TP = 10
    SL = 40

    for i in tqdm(range(_cfg.lookback, len(feats) - _cfg.forward), desc="OOS TCN Validation"):
        x = feats[i - _cfg.lookback:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            dir_logits, pred_mag, pred_conf = model(x_t)
            prob_long = torch.sigmoid(dir_logits).item()
            _conf = pred_conf.item()

        price = prices[i]
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']

        # Trade management
        if in_trade:
            ref = last_tp_price if tp_count > 0 else entry_price
            if trade_dir == 'LONG':
                if (low - entry_price) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'LONG',
                                   'tps': tp_count, 'conf': entry_conf})
                    in_trade = False
                    continue
                if (high - ref) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref + TP * TICK
            else:
                if (entry_price - high) / TICK <= -SL:
                    trades.append({'pnl': -SL + tp_count * TP, 'dir': 'SHORT',
                                   'tps': tp_count, 'conf': entry_conf})
                    in_trade = False
                    continue
                if (ref - low) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref - TP * TICK

        # Direction signal
        if prob_long > 0.6:
            new_dir = 'LONG'
        elif prob_long < 0.4:
            new_dir = 'SHORT'
        else:
            continue

        if in_trade and new_dir != trade_dir:
            pnl = (price - entry_price) / TICK if trade_dir == 'LONG' else (entry_price - price) / TICK
            trades.append({'pnl': pnl, 'dir': trade_dir, 'tps': tp_count, 'conf': entry_conf})
            trade_dir = new_dir
            entry_price = price
            entry_conf = _conf
            tp_count = 0
            last_tp_price = 0
        elif not in_trade:
            in_trade = True
            trade_dir = new_dir
            entry_price = price
            entry_conf = _conf
            tp_count = 0
            last_tp_price = 0

    # BUG 3: flush last trade
    if in_trade:
        _last_price = prices[min(len(feats) - _cfg.forward - 1, len(prices) - 1)]
        pnl = (_last_price - entry_price) / TICK if trade_dir == 'LONG' \
            else (entry_price - _last_price) / TICK
        trades.append({'pnl': pnl + tp_count * TP, 'dir': trade_dir,
                       'tps': tp_count, 'conf': entry_conf})

    # Summary
    total_pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    w = len([t for t in trades if t['pnl'] > 0])
    trading_days = df['timestamp'].apply(lambda t: pd.Timestamp(t, unit='s').date()).nunique()

    _wr = w / n * 100 if n > 0 else 0
    _per_day = total_pnl * 0.5 / trading_days if trading_days > 0 else 0

    print(f"\n{'='*60}")
    print(f"OOS VALIDATION: TCN ({_cfg.feat_mode}D, {_cfg.dilation_depth} dilations, {_cfg.channels}ch)")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {_wr:.1f}%")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${_per_day:.2f}")
    print(f"  Trading days: {trading_days}")
    if n > 0:
        print(f"  Avg confidence: {np.mean([t['conf'] for t in trades]):.2f}")
        print(f"  Avg TPs/trade: {np.mean([t['tps'] for t in trades]):.1f}")
        _pnls = np.array([t['pnl'] for t in trades])
        _sharpe = _pnls.mean() / (_pnls.std() + 1e-8)
        print(f"  OOS Sharpe: {_sharpe:.3f}")

    # Append to shared log
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"feat={_cfg.feat_mode}D | model=TCN | "
             f"lookback={_cfg.lookback} | dilations={_cfg.dilation_depth} | ch={_cfg.channels} | "
             f"val_acc={ckpt.get('val_acc', 0):.1f}% | "
             f"trades={n} | WR={_wr:.1f}% | "
             f"PnL=${total_pnl*0.5:,.0f} | $/day=${_per_day:.0f}\n")
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)
    print(f"  Logged: {RESULTS_LOG}")

    results = {
        'model': 'tcn', 'trades': n, 'wr': _wr,
        'pnl_ticks': total_pnl, 'pnl_dollars': total_pnl * 0.5,
        'per_day': _per_day, 'trading_days': trading_days,
        'config': dataclasses.asdict(_cfg),
    }
    with open(os.path.join(_cfg.checkpoint_dir, 'oos_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Release memory
    del feats, labels, mags, df, trades
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='Direction TCN trainer')
    parser.add_argument('--phase', default='all', choices=['train', 'validate', 'all'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--features', type=int, default=7, choices=[3, 7, 18, 22])
    parser.add_argument('--lookback', type=int, default=30)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--dilation-depth', type=int, default=7)
    parser.add_argument('--max-bars', type=int, default=0)
    args = parser.parse_args()

    # Build config from args
    _features_map = {3: FEATURES_3D, 7: FEATURES_7D, 18: FEATURES_18D, 22: FEATURES_22D}
    _feats = _features_map.get(args.features, FEATURES_7D)

    cfg = TCNConfig(
        feat_mode=args.features,
        n_feat=len(_feats),
        lookback=args.lookback,
        channels=args.channels,
        dilation_depth=args.dilation_depth,
        epochs=args.epochs,
    )

    receptive = 2 ** cfg.dilation_depth
    print(f"\n{'='*60}")
    print(f"TCN EXPERIMENT: feat={cfg.feat_mode}D lookback={cfg.lookback} "
          f"dilations={cfg.dilation_depth} ch={cfg.channels}")
    print(f"  Receptive field: {receptive} bars | Epochs: {cfg.epochs}")
    print(f"{'='*60}")

    if args.phase in ('train', 'all'):
        t0 = time.time()
        feats, labels, mags, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
        print(f"Dataset built in {time.time()-t0:.1f}s")
        print(f"  Feature shape: {feats.shape}")
        model = train_tcn(feats, labels, mags, cfg)

    if args.phase in ('validate', 'all'):
        validate_oos(cfg)


if __name__ == '__main__':
    main()
