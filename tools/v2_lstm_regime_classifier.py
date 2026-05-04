"""
v2_lstm_regime_classifier.py — train an LSTM to classify regime_2d
from sequences of v2 features.

Design choices driven by EDA findings:
  - Input: 184D v2 features × 50-bar sequence at 5m base
    (50 bars ~ 4h, the window length that worked for rolling-corr classifier)
  - Target: regime_2d at end of sequence (6-class softmax)
    Regime is the most stable target per Track A-D findings (95% OOS for
    feature×feature regime structure; price-target collapsed at 25.8%).
  - Per-feature LayerNorm THEN 2-layer LSTM (features have wildly different
    scales — z-scores ~ O(1), prices ~ O(20000), velocities ~ O(50)).
  - IS for fit, VAL for early stopping, OOS for final test.
  - Compare to rolling-corr baseline (71% OOS accuracy) before claiming
    LSTM is better.

Usage:
  python tools/v2_lstm_regime_classifier.py --epochs 20 --hidden 128 \\
      --seq-len 50 --batch-size 64

Outputs:
  reports/findings/v2_lstm_regime/
    train_log.csv          per epoch: train_loss, val_loss, val_acc
    oos_predictions.csv    per OOS bar: true regime, predicted regime, probs
    confusion_matrix.csv   6×6 OOS confusion
    summary.md             headline metrics + comparison vs rolling-corr
    checkpoint.pt          best VAL model
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


REGIME_TO_IDX = {r: i for i, r in enumerate(REGIME_2D_ORDER)}
IDX_TO_REGIME = {i: r for r, i in REGIME_TO_IDX.items()}


def build_feature_columns():
    """Return list of 184 v2 feature column names in canonical order."""
    cols = []
    for tf in TF_HIERARCHY_V2:
        for concept_short in FEATURE_NAMES_V2:
            # Map short name to full canonical (need _w / _1b / bare suffix)
            if concept_short in ('bar_range', 'body'):
                concept = concept_short
            elif concept_short.endswith('_1b'):
                concept = concept_short
            else:
                concept = concept_short + '_w'
            cols.append(feature_column_for(concept, tf))
    return cols


class SequenceDataset(Dataset):
    """Builds (seq_len-bar history, regime label) pairs.

    Skips bars where the seq_len-bar history straddles a date boundary,
    so each sequence stays within one trading day's intraday flow.
    """

    def __init__(self, features: np.ndarray, regimes: np.ndarray,
                  dates: np.ndarray, seq_len: int):
        self.features = features
        self.regimes = regimes
        self.dates = dates
        self.seq_len = seq_len
        # Find valid end-indices: end_idx is the LAST bar of the sequence
        # (history is end_idx-seq_len+1 .. end_idx), need same date AND no NaN
        valid = []
        for i in range(seq_len - 1, len(features)):
            start = i - seq_len + 1
            if dates[start] != dates[i]:  # cross-day boundary
                continue
            if np.any(np.isnan(features[start:i+1])):
                continue
            if regimes[i] == -1:  # no regime label
                continue
            valid.append(i)
        self.indices = np.array(valid, dtype=np.int64)
        print(f"  Built {len(self.indices):,} valid sequences from "
              f"{len(features):,} bars")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = int(self.indices[idx])
        start = end - self.seq_len + 1
        seq = self.features[start:end+1]
        label = self.regimes[end]
        return torch.from_numpy(seq).float(), torch.tensor(label, dtype=torch.long)


class RegimeLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, n_layers: int = 2,
                  n_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                              batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        h, _ = self.lstm(x)
        return self.head(h[:, -1])  # last timestep


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for seq, label in loader:
            seq, label = seq.to(device), label.to(device)
            logits = model(seq)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    if total == 0:
        return 0.0, np.array([]), np.array([]), np.array([])
    return (correct / total,
            np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--seq-len', type=int, default=50,
                        help='Sequence length in bars (50 ~ 4h at 5m base)')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_lstm_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"  V2 LSTM regime classifier")
    print(f"  Device: {device}")
    print(f"  Seq len: {args.seq_len} bars (~{args.seq_len*5/60:.1f}h)")
    print(f"  Hidden: {args.hidden}  Layers: {args.n_layers}")
    print(f"  Batch: {args.batch_size}  LR: {args.lr}  Epochs: {args.epochs}")
    print(f"{'='*70}")

    # ---- Load data ----
    print(f"\n--- Loading data ---")
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
    print(f"  Merged: {len(merged):,} bars, dates: "
          f"{merged['date'].min()} -> {merged['date'].max()}")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    # ---- Build feature matrix ----
    feat_cols = build_feature_columns()
    feat_cols = [c for c in feat_cols if c in full.columns]
    print(f"  Feature columns: {len(feat_cols)}")

    features = full[feat_cols].values.astype(np.float32)
    regimes = np.array([REGIME_TO_IDX.get(r, -1) for r in full['regime_2d']],
                          dtype=np.int64)
    splits = full['split'].values.astype(str)
    dates = full['date'].values.astype(str)

    # ---- Per-feature standardization on IS ----
    is_mask = (splits == 'IS')
    is_features = features[is_mask]
    # Use nanmean / nanstd to handle NaN
    feat_mean = np.nanmean(is_features, axis=0).astype(np.float32)
    feat_std = np.nanstd(is_features, axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0  # guard
    print(f"  IS standardization: mean range "
          f"[{feat_mean.min():.2f}, {feat_mean.max():.2f}], "
          f"std range [{feat_std.min():.4f}, {feat_std.max():.2f}]")

    features = (features - feat_mean) / feat_std

    # save feat_mean/std for live use
    np.savez(os.path.join(args.output_dir, 'feature_norm.npz'),
                mean=feat_mean, std=feat_std,
                feature_cols=np.array(feat_cols))

    # ---- Build per-split datasets ----
    print(f"\n--- Building sequence datasets ---")
    is_idx = np.where(is_mask)[0]
    val_idx = np.where(splits == 'VAL')[0]
    oos_idx = np.where(splits == 'OOS')[0]
    print(f"  IS: {len(is_idx):,} bars; VAL: {len(val_idx):,}; OOS: {len(oos_idx):,}")

    # We work on the full feature/regime/dates arrays but mask which end
    # indices each split uses. The Dataset class scans for valid sequences
    # within its slice.
    def slice_arrays(idx):
        return features[idx], regimes[idx], dates[idx]

    is_feat, is_reg, is_dates = slice_arrays(is_idx)
    val_feat, val_reg, val_dates = slice_arrays(val_idx)
    oos_feat, oos_reg, oos_dates = slice_arrays(oos_idx)

    print(f"\n  IS dataset:")
    is_ds = SequenceDataset(is_feat, is_reg, is_dates, args.seq_len)
    print(f"  VAL dataset:")
    val_ds = SequenceDataset(val_feat, val_reg, val_dates, args.seq_len)
    print(f"  OOS dataset:")
    oos_ds = SequenceDataset(oos_feat, oos_reg, oos_dates, args.seq_len)

    is_loader = DataLoader(is_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
    oos_loader = DataLoader(oos_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    # ---- Model + training ----
    model = RegimeLSTM(n_features=len(feat_cols), hidden=args.hidden,
                          n_layers=args.n_layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")

    # Class weights for imbalanced regime distribution
    is_label_counts = np.bincount(is_reg[is_reg >= 0], minlength=6).astype(float)
    class_weights = (is_label_counts.sum() / (6 * np.maximum(is_label_counts, 1)))
    class_weights = torch.tensor(class_weights, dtype=torch.float32,
                                     device=device)
    print(f"  IS class distribution: {dict(zip(REGIME_2D_ORDER, is_label_counts.astype(int)))}")
    print(f"  Class weights: {dict(zip(REGIME_2D_ORDER, class_weights.cpu().numpy().round(2)))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n--- Training ---")
    log = []
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for seq, label in is_loader:
            seq, label = seq.to(device), label.to(device)
            logits = model(seq)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * label.size(0)
            train_correct += (logits.argmax(dim=-1) == label).sum().item()
            train_total += label.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_acc, _, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")
        log.append({'epoch': epoch, 'train_loss': train_loss,
                      'train_acc': train_acc, 'val_acc': val_acc,
                      'elapsed_sec': elapsed})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({'model_state': model.state_dict(),
                          'feat_cols': feat_cols,
                          'feat_mean': feat_mean,
                          'feat_std': feat_std,
                          'config': vars(args)}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(best val_acc {best_val_acc:.4f} at epoch {best_epoch})")
                break

    pd.DataFrame(log).to_csv(os.path.join(args.output_dir, 'train_log.csv'),
                                index=False)

    # ---- Load best, evaluate on OOS ----
    print(f"\n--- Final OOS evaluation (best VAL checkpoint epoch {best_epoch}) ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    oos_acc, oos_pred, oos_target, oos_probs = evaluate(model, oos_loader, device)
    print(f"  OOS accuracy: {oos_acc:.4f} ({oos_acc*100:.1f}%)")
    print(f"  Compare to rolling-corr baseline: ~71-75% OOS")

    # Per-class precision/recall
    print(f"\n  Per-regime OOS metrics:")
    print(f"    {'regime':>14}  {'support':>7}  {'precision':>9}  {'recall':>7}  {'f1':>5}")
    for r_idx, r_name in IDX_TO_REGIME.items():
        true_pos = int(((oos_pred == r_idx) & (oos_target == r_idx)).sum())
        false_pos = int(((oos_pred == r_idx) & (oos_target != r_idx)).sum())
        false_neg = int(((oos_pred != r_idx) & (oos_target == r_idx)).sum())
        support = int((oos_target == r_idx).sum())
        precision = true_pos / max(true_pos + false_pos, 1)
        recall = true_pos / max(true_pos + false_neg, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-9)) if precision + recall > 0 else 0
        print(f"    {r_name:>14}  {support:>7}  {precision:>9.3f}  {recall:>7.3f}  {f1:>5.3f}")

    # Confusion matrix
    cm = np.zeros((6, 6), dtype=np.int64)
    for t, p in zip(oos_target, oos_pred):
        cm[t, p] += 1
    cm_df = pd.DataFrame(cm, index=[f'true_{r}' for r in REGIME_2D_ORDER],
                            columns=[f'pred_{r}' for r in REGIME_2D_ORDER])
    cm_df.to_csv(os.path.join(args.output_dir, 'confusion_matrix.csv'))
    print(f"\n  Confusion matrix:")
    print(cm_df.to_string())

    # Save predictions
    pred_df = pd.DataFrame({
        'true_idx': oos_target,
        'true_regime': [IDX_TO_REGIME[i] for i in oos_target],
        'pred_idx': oos_pred,
        'pred_regime': [IDX_TO_REGIME[i] for i in oos_pred],
        'correct': (oos_pred == oos_target).astype(int),
    })
    for i, r in IDX_TO_REGIME.items():
        pred_df[f'prob_{r}'] = oos_probs[:, i]
    pred_df.to_csv(os.path.join(args.output_dir, 'oos_predictions.csv'),
                      index=False)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(REGIME_2D_ORDER, rotation=30, ha='right')
    ax.set_yticks(range(6))
    ax.set_yticklabels(REGIME_2D_ORDER)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})', ha='center',
                      va='center',
                      color='white' if cm_norm[i,j] > 0.5 else 'black',
                      fontsize=8)
    plt.colorbar(im, ax=ax, label='Recall (row-normalized)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'OOS confusion matrix (acc={oos_acc:.3f})')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'),
                  dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Markdown summary
    md = os.path.join(args.output_dir, 'summary.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write(f"# V2 LSTM regime classifier - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Config**: seq_len={args.seq_len}  hidden={args.hidden}  "
                f"layers={args.n_layers}  dropout={args.dropout}  "
                f"batch={args.batch_size}  lr={args.lr}\n\n")
        f.write(f"**Parameters**: {n_params:,}\n\n")
        f.write(f"**Best epoch**: {best_epoch} (val_acc={best_val_acc:.4f})\n\n")
        f.write(f"**OOS accuracy**: {oos_acc:.4f} ({oos_acc*100:.1f}%)\n\n")
        f.write(f"**Comparison**:\n")
        f.write(f"- Always-FLAT baseline: 56.2%\n")
        f.write(f"- Rolling-corr 50-bar (best pair): 74.6% OOS\n")
        f.write(f"- LSTM: {oos_acc*100:.1f}%\n\n")
        f.write(f"## Confusion matrix (OOS)\n\n")
        f.write(cm_df.to_string())
        f.write(f"\n\n## Train log\n\n")
        f.write(pd.DataFrame(log).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md}")


if __name__ == '__main__':
    main()
