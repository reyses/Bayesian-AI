"""
v2_lstm_trader.py — LSTM that actually TRADES.

Target: 3-class direction (LONG / FLAT / SHORT) based on forward N-bar
return sign with a FLAT band. NN outputs softmax probabilities — those
are kept as the posterior-prior input for a future Bayesian wrapper.

Trading rules (deliberately simple, "shits and giggles" first cut):
  - At each OOS bar, predict class probabilities.
  - If max(prob) > confidence_threshold AND argmax in {LONG, SHORT}:
      open a position in that direction at the bar's close
  - One position at a time. If new signal arrives while in position,
    hold (no flip) until exit.
  - Exit after hold_bars bars (default = forward_n = 12 bars at 5m
    base = 1 hour).
  - PnL in dollars at MNQ tick value: 1 tick = 0.25 price, $0.50/tick.

Bayesian-pair-readiness:
  - oos_predictions.csv keeps every bar's [p_short, p_flat, p_long]
    softmax. A downstream Bayesian step can multiply by a prior
    distribution and renormalize for the posterior. This is the cleanest
    NN→Bayesian handoff (Categorical likelihoods, Dirichlet conjugate).

Compare to: rolling-corr regime classifier (74.6% OOS regime accuracy
but no direct $/day projection) and existing 9-tier engine baseline
(-$164/day on honest features).

Usage:
  python tools/v2_lstm_trader.py --epochs 20 --hidden 128 --seq-len 50 \\
      --hold-bars 12 --flat-threshold-ticks 5 --confidence-threshold 0.45

Outputs:
  reports/findings/v2_lstm_trader/
    train_log.csv
    oos_predictions.csv     all bars: true_class, pred, p_short, p_flat, p_long
    trades.csv              each trade: entry_ts, exit_ts, dir, entry, exit, pnl
    equity_curve.png
    daily_pnl.csv           date, daily_pnl, n_trades
    summary.md              $/day, sharpe, max DD, hit rate, trade count
    checkpoint.pt
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    FEATURE_NAMES_V2, TF_HIERARCHY_V2, load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_tf_sweep_eda import feature_column_for


# Class encoding: 0=SHORT, 1=FLAT, 2=LONG
CLASS_NAMES = ['SHORT', 'FLAT', 'LONG']
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

TICK = 0.25
TV = 0.50


def build_feature_columns():
    cols = []
    for tf in TF_HIERARCHY_V2:
        for concept_short in FEATURE_NAMES_V2:
            if concept_short in ('bar_range', 'body'):
                concept = concept_short
            elif concept_short.endswith('_1b'):
                concept = concept_short
            else:
                concept = concept_short + '_w'
            cols.append(feature_column_for(concept, tf))
    return cols


def make_direction_labels(close: np.ndarray, dates: np.ndarray,
                            forward_n: int, flat_thresh_ticks: float) -> np.ndarray:
    """3-class direction label from forward N-bar return.

    Bars near end of day (where forward window crosses date) get -1
    (excluded from training).
    """
    n = len(close)
    fwd_ticks = np.full(n, np.nan)
    for i in range(n - forward_n):
        # Same-date constraint
        if dates[i] != dates[i + forward_n]:
            continue
        fwd_price = close[i + forward_n] - close[i]
        fwd_ticks[i] = fwd_price / TICK

    labels = np.full(n, -1, dtype=np.int64)
    valid = ~np.isnan(fwd_ticks)
    labels[valid & (fwd_ticks > flat_thresh_ticks)] = CLASS_TO_IDX['LONG']
    labels[valid & (fwd_ticks < -flat_thresh_ticks)] = CLASS_TO_IDX['SHORT']
    labels[valid & (np.abs(fwd_ticks) <= flat_thresh_ticks)] = CLASS_TO_IDX['FLAT']
    return labels


class SequenceDataset(Dataset):
    def __init__(self, features, labels, dates, seq_len):
        self.features = features
        self.labels = labels
        self.dates = dates
        self.seq_len = seq_len
        valid = []
        for i in range(seq_len - 1, len(features)):
            start = i - seq_len + 1
            if dates[start] != dates[i]:
                continue
            if labels[i] == -1:
                continue
            if np.any(np.isnan(features[start:i+1])):
                continue
            valid.append(i)
        self.indices = np.array(valid, dtype=np.int64)
        print(f"  {len(self.indices):,} sequences from {len(features):,} bars")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = int(self.indices[idx])
        start = end - self.seq_len + 1
        seq = self.features[start:end+1]
        label = self.labels[end]
        return torch.from_numpy(seq).float(), torch.tensor(label, dtype=torch.long), end


class TraderLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                              batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x):
        x = self.input_norm(x)
        h, _ = self.lstm(x)
        return self.head(h[:, -1])


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_targets, all_probs, all_endidx = [], [], [], []
    with torch.no_grad():
        for seq, label, endidx in loader:
            seq, label = seq.to(device), label.to(device)
            logits = model(seq)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_endidx.append(endidx.numpy())
    if total == 0:
        return 0.0, np.array([]), np.array([]), np.array([]), np.array([])
    return (correct / total,
            np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_probs),
            np.concatenate(all_endidx))


def simulate_trades(probs, end_indices, close, ts_int, dates,
                     hold_bars, confidence_threshold):
    """Simulate trading from class probabilities.

    Rules: open on confident LONG/SHORT, hold hold_bars, close.
    One position at a time, no flip during hold.

    Returns (trades_df, daily_pnl_df).
    """
    # Sort by end_index so we walk in time order
    order = np.argsort(end_indices)
    end_indices = end_indices[order]
    probs = probs[order]

    trades = []
    in_pos = False
    pos_entry_idx = -1
    pos_entry_price = 0.0
    pos_dir = 0
    pos_exit_idx = -1

    for i, end_idx in enumerate(end_indices):
        # check if existing position should close at this bar
        if in_pos and end_idx >= pos_exit_idx:
            exit_price = close[pos_exit_idx]
            if pos_dir == 1:  # LONG
                pnl_price = exit_price - pos_entry_price
            else:  # SHORT
                pnl_price = pos_entry_price - exit_price
            pnl_dollars = (pnl_price / TICK) * TV
            trades.append({
                'entry_idx': int(pos_entry_idx),
                'exit_idx': int(pos_exit_idx),
                'entry_ts': int(ts_int[pos_entry_idx]),
                'exit_ts': int(ts_int[pos_exit_idx]),
                'date': dates[pos_entry_idx],
                'direction': 'LONG' if pos_dir == 1 else 'SHORT',
                'entry_price': float(pos_entry_price),
                'exit_price': float(exit_price),
                'pnl_ticks': float(pnl_price / TICK),
                'pnl_dollars': float(pnl_dollars),
                'hold_bars': int(pos_exit_idx - pos_entry_idx),
            })
            in_pos = False

        # if not in position, look at this bar's probs
        if not in_pos:
            p = probs[i]
            max_p = p.max()
            arg = int(np.argmax(p))
            if max_p >= confidence_threshold and arg in (CLASS_TO_IDX['LONG'],
                                                                  CLASS_TO_IDX['SHORT']):
                in_pos = True
                pos_entry_idx = end_idx
                pos_entry_price = close[end_idx]
                pos_dir = 1 if arg == CLASS_TO_IDX['LONG'] else -1
                pos_exit_idx = end_idx + hold_bars

    # close any open position at the end (mark with NaN PnL — incomplete)
    if in_pos:
        # try to close at last available close
        if pos_exit_idx < len(close):
            exit_price = close[pos_exit_idx]
        else:
            exit_price = close[-1]
        if pos_dir == 1:
            pnl_price = exit_price - pos_entry_price
        else:
            pnl_price = pos_entry_price - exit_price
        pnl_dollars = (pnl_price / TICK) * TV
        trades.append({
            'entry_idx': int(pos_entry_idx),
            'exit_idx': int(min(pos_exit_idx, len(close)-1)),
            'entry_ts': int(ts_int[pos_entry_idx]),
            'exit_ts': int(ts_int[min(pos_exit_idx, len(ts_int)-1)]),
            'date': dates[pos_entry_idx],
            'direction': 'LONG' if pos_dir == 1 else 'SHORT',
            'entry_price': float(pos_entry_price),
            'exit_price': float(exit_price),
            'pnl_ticks': float(pnl_price / TICK),
            'pnl_dollars': float(pnl_dollars),
            'hold_bars': int(min(pos_exit_idx, len(close)-1) - pos_entry_idx),
        })

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return trades_df, pd.DataFrame()

    # daily aggregation
    daily = (trades_df.groupby('date')
                       .agg(daily_pnl=('pnl_dollars', 'sum'),
                             n_trades=('pnl_dollars', 'count'),
                             win_rate=('pnl_dollars', lambda s: float((s > 0).mean())))
                       .reset_index())
    return trades_df, daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--seq-len', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--forward-n', type=int, default=12,
                        help='Forward bars for label / hold for trade')
    parser.add_argument('--flat-threshold-ticks', type=float, default=5.0,
                        help='|fwd return| <= this ticks -> FLAT class')
    parser.add_argument('--hold-bars', type=int, default=12)
    parser.add_argument('--confidence-threshold', type=float, default=0.45,
                        help='Open trade only when max class prob exceeds')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_lstm_trader')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"  V2 LSTM TRADER")
    print(f"  Device: {device}")
    print(f"  Target: 3-class direction (forward {args.forward_n}-bar return)")
    print(f"  FLAT band: |fwd ticks| <= {args.flat_threshold_ticks}")
    print(f"  Hold bars: {args.hold_bars}  Conf threshold: {args.confidence_threshold}")
    print(f"{'='*70}")

    # Load data
    print(f"\n--- Loading ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int
    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base_df['date'] = dt_la.dt.date.astype(str)

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    # Feature matrix
    feat_cols = build_feature_columns()
    feat_cols = [c for c in feat_cols if c in full.columns]
    features = full[feat_cols].values.astype(np.float32)
    close = full['close'].values.astype(np.float64)
    splits = full['split'].values.astype(str)
    dates = full['date'].values.astype(str)
    ts_int_arr = full['ts_int'].values.astype(np.int64)

    # Direction labels
    print(f"\n--- Generating direction labels ---")
    labels = make_direction_labels(close, dates, args.forward_n,
                                       args.flat_threshold_ticks)
    valid_labels = labels[labels >= 0]
    label_dist = np.bincount(valid_labels, minlength=3)
    print(f"  Class distribution: SHORT={label_dist[0]:,} ({100*label_dist[0]/max(valid_labels.size,1):.1f}%),  "
          f"FLAT={label_dist[1]:,} ({100*label_dist[1]/max(valid_labels.size,1):.1f}%),  "
          f"LONG={label_dist[2]:,} ({100*label_dist[2]/max(valid_labels.size,1):.1f}%)")

    # Standardize on IS
    is_mask = (splits == 'IS')
    feat_mean = np.nanmean(features[is_mask], axis=0).astype(np.float32)
    feat_std = np.nanstd(features[is_mask], axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0
    features = (features - feat_mean) / feat_std
    np.savez(os.path.join(args.output_dir, 'feature_norm.npz'),
                mean=feat_mean, std=feat_std,
                feature_cols=np.array(feat_cols))

    # Per-split arrays
    is_idx = np.where(is_mask)[0]
    val_idx = np.where(splits == 'VAL')[0]
    oos_idx = np.where(splits == 'OOS')[0]
    print(f"  IS bars: {len(is_idx):,}  VAL: {len(val_idx):,}  OOS: {len(oos_idx):,}")

    def slice_(idx):
        return features[idx], labels[idx], dates[idx]

    print(f"\n--- Building datasets ---")
    print("  IS:")
    is_ds = SequenceDataset(*slice_(is_idx), args.seq_len)
    print("  VAL:")
    val_ds = SequenceDataset(*slice_(val_idx), args.seq_len)
    print("  OOS:")
    oos_ds = SequenceDataset(*slice_(oos_idx), args.seq_len)

    is_loader = DataLoader(is_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
    oos_loader = DataLoader(oos_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    # Model
    model = TraderLSTM(n_features=len(feat_cols), hidden=args.hidden,
                          n_layers=args.n_layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {n_params:,}")

    # Class weights
    cw = (label_dist.sum() / (3 * np.maximum(label_dist, 1)))
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    print(f"  Class weights: SHORT={cw[0]:.2f}  FLAT={cw[1]:.2f}  LONG={cw[2]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n--- Training ---")
    log = []
    best_val_acc = 0.0
    best_epoch = -1
    patience = 0
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for seq, label, _ in is_loader:
            seq, label = seq.to(device), label.to(device)
            logits = model(seq)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * label.size(0)
            correct += (logits.argmax(dim=-1) == label).sum().item()
            total += label.size(0)
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, _, _, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)")
        log.append({'epoch': epoch, 'train_loss': train_loss,
                      'train_acc': train_acc, 'val_acc': val_acc, 'sec': elapsed})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience = 0
            torch.save({'model_state': model.state_dict(),
                          'feat_cols': feat_cols, 'feat_mean': feat_mean,
                          'feat_std': feat_std, 'config': vars(args)}, ckpt_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"  Early stop at epoch {epoch} (best at {best_epoch})")
                break

    pd.DataFrame(log).to_csv(os.path.join(args.output_dir, 'train_log.csv'),
                                index=False)

    # OOS eval + trade simulation
    print(f"\n--- OOS evaluation + trade simulation ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    oos_acc, oos_pred, oos_target, oos_probs, oos_endidx = evaluate(
        model, oos_loader, device)
    print(f"  OOS class accuracy: {oos_acc:.4f} ({oos_acc*100:.1f}%)")

    # Per-class precision/recall
    print(f"\n  Per-class OOS:")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tp = int(((oos_pred == c_idx) & (oos_target == c_idx)).sum())
        fp = int(((oos_pred == c_idx) & (oos_target != c_idx)).sum())
        fn = int(((oos_pred != c_idx) & (oos_target == c_idx)).sum())
        sup = int((oos_target == c_idx).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"    {c_name:>6}  support={sup:>5}  precision={prec:.3f}  recall={rec:.3f}")

    # CRITICAL: oos_endidx are positions in the OOS slice, not the global
    # merged DataFrame. Build OOS-local close/ts/dates arrays so that
    # close[end_idx] etc. address the correct OOS bar.
    oos_close = close[oos_idx]
    oos_ts = ts_int_arr[oos_idx]
    oos_dates_arr = dates[oos_idx]

    # Save predictions w/ probs (Bayesian-prior format)
    pred_records = []
    for i, end in enumerate(oos_endidx):
        pred_records.append({
            'end_idx': int(end), 'ts': int(oos_ts[end]),
            'date': oos_dates_arr[end], 'close': float(oos_close[end]),
            'true_class': CLASS_NAMES[oos_target[i]] if oos_target[i] >= 0 else 'NA',
            'pred_class': CLASS_NAMES[oos_pred[i]],
            'p_short': float(oos_probs[i, 0]),
            'p_flat': float(oos_probs[i, 1]),
            'p_long': float(oos_probs[i, 2]),
        })
    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(args.output_dir, 'oos_predictions.csv'),
                      index=False)

    # ---- Trade simulation ----
    print(f"\n--- Trade simulation (hold={args.hold_bars}, conf>={args.confidence_threshold}) ---")
    trades_df, daily_df = simulate_trades(
        oos_probs, oos_endidx, oos_close, oos_ts, oos_dates_arr,
        args.hold_bars, args.confidence_threshold)
    trades_df.to_csv(os.path.join(args.output_dir, 'trades.csv'), index=False)
    daily_df.to_csv(os.path.join(args.output_dir, 'daily_pnl.csv'), index=False)

    if len(trades_df) == 0:
        print("  NO TRADES at this confidence threshold. Try lower --confidence-threshold")
        return

    n_oos_days = len(np.unique(oos_dates_arr))
    n_trades = len(trades_df)
    n_winners = int((trades_df['pnl_dollars'] > 0).sum())
    win_rate = n_winners / n_trades
    total_pnl = float(trades_df['pnl_dollars'].sum())
    pnl_per_trade = total_pnl / n_trades
    pnl_per_day_active = float(daily_df['daily_pnl'].mean())
    pnl_per_calendar_day = total_pnl / n_oos_days

    # Sharpe (daily, annualized)
    daily_pnl_full = (pd.Series(daily_df['daily_pnl'].values,
                                    index=daily_df['date'])
                          .reindex(np.unique(oos_dates_arr), fill_value=0))
    sharpe = (daily_pnl_full.mean() / daily_pnl_full.std()
                * np.sqrt(252)) if daily_pnl_full.std() > 0 else 0
    cum_pnl = daily_pnl_full.cumsum()
    running_max = cum_pnl.expanding().max()
    drawdown = cum_pnl - running_max
    max_dd = float(drawdown.min())

    # PF-based Trade WR
    winners = trades_df[trades_df['pnl_dollars'] > 0]['pnl_dollars']
    losers = trades_df[trades_df['pnl_dollars'] < 0]['pnl_dollars']
    pf_wr = ((winners.sum() / abs(losers.sum())) - 1) if losers.sum() != 0 else float('inf')

    # Print headline
    print(f"\n  ===  OOS TRADING RESULTS (held-out 71 days)  ===")
    print(f"  Trades:                {n_trades}")
    print(f"  Active days:           {len(daily_df)}")
    print(f"  Calendar days (OOS):   {n_oos_days}")
    print(f"  Total PnL:             ${total_pnl:.2f}")
    print(f"  $/trade:               ${pnl_per_trade:.2f}")
    print(f"  $/active day:          ${pnl_per_day_active:.2f}")
    print(f"  $/calendar day:        ${pnl_per_calendar_day:.2f}")
    print(f"  Count-WR:              {win_rate*100:.1f}%")
    print(f"  PF-WR (∑profit/|∑loss|−1):  {pf_wr:+.2f}")
    print(f"  Sharpe (annualized):   {sharpe:.2f}")
    print(f"  Max drawdown:          ${max_dd:.2f}")

    # Equity curve
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    cum_pnl_aligned = pd.Series(daily_df['daily_pnl'].values,
                                     index=pd.to_datetime(daily_df['date'])).sort_index().cumsum()
    axes[0].plot(cum_pnl_aligned.index, cum_pnl_aligned.values, color='blue', lw=1.5)
    axes[0].axhline(0, color='black', alpha=0.4)
    axes[0].set_ylabel('Cumulative PnL ($)')
    axes[0].set_title(f'OOS equity curve  '
                          f'(N_trades={n_trades}, $/day={pnl_per_calendar_day:+.1f}, '
                          f'Sharpe={sharpe:.2f}, MaxDD=${max_dd:.0f})')
    axes[0].grid(alpha=0.3)

    daily_for_plot = pd.Series(daily_df['daily_pnl'].values,
                                    index=pd.to_datetime(daily_df['date'])).sort_index()
    colors = ['green' if v > 0 else 'red' for v in daily_for_plot.values]
    axes[1].bar(daily_for_plot.index, daily_for_plot.values, color=colors,
                  width=1, alpha=0.7)
    axes[1].axhline(0, color='black', alpha=0.4)
    axes[1].set_ylabel('Daily PnL ($)')
    axes[1].set_xlabel('Date')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'equity_curve.png'),
                  dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Markdown summary
    md = os.path.join(args.output_dir, 'summary.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write(f"# V2 LSTM trader - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"## Config\n\n")
        f.write(f"- seq_len={args.seq_len}, hidden={args.hidden}, layers={args.n_layers}\n")
        f.write(f"- forward_n={args.forward_n}, flat_thresh={args.flat_threshold_ticks} ticks\n")
        f.write(f"- hold_bars={args.hold_bars}, conf_threshold={args.confidence_threshold}\n")
        f.write(f"- params={n_params:,}, best_epoch={best_epoch} (val_acc={best_val_acc:.4f})\n\n")
        f.write(f"## OOS classification\n\n")
        f.write(f"- 3-class accuracy: {oos_acc*100:.1f}%\n\n")
        f.write(f"## OOS trading\n\n")
        f.write(f"- Trades: {n_trades}\n")
        f.write(f"- Total PnL: ${total_pnl:+.2f}\n")
        f.write(f"- $/trade: ${pnl_per_trade:+.2f}\n")
        f.write(f"- $/calendar day: ${pnl_per_calendar_day:+.2f}\n")
        f.write(f"- Count-WR: {win_rate*100:.1f}%\n")
        f.write(f"- PF-WR: {pf_wr:+.2f}\n")
        f.write(f"- Sharpe (annualized): {sharpe:.2f}\n")
        f.write(f"- Max drawdown: ${max_dd:+.2f}\n\n")
        f.write(f"## Bayesian-pair handoff\n\n")
        f.write(f"`oos_predictions.csv` contains [p_short, p_flat, p_long] per "
                f"OOS bar — these are categorical likelihoods ready for a "
                f"Dirichlet-conjugate Bayesian update.\n")
    print(f"\n  [saved] {md}")


if __name__ == '__main__':
    main()
