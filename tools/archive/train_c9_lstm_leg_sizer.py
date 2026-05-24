"""C9 — LSTM leg-amplitude regressor.

Sequence-input alternative to B7 GBM. Same target (leg_amp / R) but
trained on lead-in sequence of 60 1m bars × 184 V2 features.

Train/eval split (per user 2026-05-17 — "two months of observation,
train on the rest"):
  - TRAIN:      first ~237 days of IS (most of 2025) — ~15k legs
  - HOLDOUT:    last 40 days of IS (Nov-Dec 2025) — proxy for
                "two months of observation" before live deployment
  - OOS test:   NT8 OOS (32 days from 2026-03-20)

Architecture:
  Input:  (batch, 60, 184) float32 z-scored
  LSTM:   hidden=64, layers=1, dropout=0.3
  Head:   Linear(64, 1) — regression
  Loss:   L1 (MAE) — matches B7 objective for fair comparison

Z-score normalization: stats computed from training-window IS only,
applied to all 3 splits. NO PEEK.

Per CLAUDE.md: CUDA-only.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


SEQ_LEN = 60   # 60 1m bars of lead-in
HOLDOUT_WINDOW_DAYS = 40   # ~2 months of trading days held out as observation


class LegAmpLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last = h[-1]           # (batch, hidden)
        return self.head(self.dropout(last)).squeeze(-1)


def build_leg_sequences(truth_df: pd.DataFrame, legs_df: pd.DataFrame,
                          v2_cols: list, seq_len: int = SEQ_LEN):
    """For each leg in legs_df, build the lead-in sequence of seq_len
    1m bars ending at entry_ts.

    Returns:
        X        : (n_legs, seq_len, n_feat) float32
        y        : (n_legs,) float32 — leg_amp_R
        kept_legs: legs_df subset where lead-in was full
    """
    truth_df = truth_df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    feat_arr = truth_df[v2_cols].fillna(0.0).values.astype(np.float32)
    ts_per_day = {day: g['timestamp'].values.astype(np.int64)
                  for day, g in truth_df.groupby('day')}
    idx_per_day = {day: g.index.values
                   for day, g in truth_df.groupby('day')}

    n_feat = len(v2_cols)
    seqs = []
    targets = []
    kept_idx = []
    for li, leg in legs_df.iterrows():
        day = leg['day']
        entry_ts = int(leg['entry_ts'])
        if day not in ts_per_day:
            continue
        ts_day = ts_per_day[day]
        # Find the truth bar index at or just before entry_ts (last completed)
        bar_idx = int(np.searchsorted(ts_day, entry_ts, side='right') - 1)
        if bar_idx < seq_len - 1:
            continue   # not enough lead-in bars
        # Gather lead-in (bar_idx - seq_len + 1 : bar_idx + 1)
        idx_in_global = idx_per_day[day][bar_idx - seq_len + 1: bar_idx + 1]
        if len(idx_in_global) < seq_len:
            continue
        seqs.append(feat_arr[idx_in_global])
        targets.append(float(leg['leg_amp_R']))
        kept_idx.append(li)

    X = np.stack(seqs, axis=0) if seqs else np.empty((0, seq_len, n_feat), dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y, np.array(kept_idx, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--is-legs',
                    default='reports/findings/regret_oracle/b7_leg_sizer_IS.parquet')
    ap.add_argument('--oos-legs',
                    default='reports/findings/regret_oracle/b7_leg_sizer_OOS.parquet')
    ap.add_argument('--out-ckpt',
                    default='reports/findings/regret_oracle/c9_lstm_leg_sizer.pt')
    ap.add_argument('--out-cache-is',
                    default='reports/findings/regret_oracle/c9_lstm_predictions_IS.parquet')
    ap.add_argument('--out-cache-oos',
                    default='reports/findings/regret_oracle/c9_lstm_predictions_OOS.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/c9_lstm_leg_sizer.txt')
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--holdout-window-days', type=int, default=HOLDOUT_WINDOW_DAYS)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading datasets...')
    is_truth = pd.read_parquet(args.is_truth)
    oos_truth = pd.read_parquet(args.oos_truth)
    is_legs = pd.read_parquet(args.is_legs)
    oos_legs = pd.read_parquet(args.oos_legs)
    print(f'  IS truth: {len(is_truth):,}   OOS truth: {len(oos_truth):,}')
    print(f'  IS legs:  {len(is_legs):,}   OOS legs:  {len(oos_legs):,}')

    v2_cols = [c for c in is_truth.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    n_feat = len(v2_cols)
    print(f'V2 features: {n_feat}')

    # Hold out LAST N days of IS as observation (proxy for live observation)
    # Train on the EARLIER days (which is most of the data).
    is_days_sorted = sorted(is_truth['day'].unique())
    train_days = is_days_sorted[:-args.holdout_window_days]
    holdout_days = is_days_sorted[-args.holdout_window_days:]
    print(f'\nTRAIN window:    {train_days[0]} to {train_days[-1]} ({len(train_days)} days)')
    print(f'HOLDOUT (obs):   {holdout_days[0]} to {holdout_days[-1]} ({len(holdout_days)} days)')

    train_truth = is_truth[is_truth['day'].isin(train_days)]
    train_legs = is_legs[is_legs['day'].isin(train_days)]
    holdout_truth = is_truth[is_truth['day'].isin(holdout_days)]
    holdout_legs = is_legs[is_legs['day'].isin(holdout_days)]

    print(f'\nBuilding TRAIN sequences ({len(train_legs)} legs)...')
    X_train, y_train, _ = build_leg_sequences(train_truth, train_legs, v2_cols, args.seq_len)
    print(f'  TRAIN shape: {X_train.shape}   target: median {np.median(y_train):.2f}, mean {y_train.mean():.2f}')

    print(f'Building HOLDOUT sequences ({len(holdout_legs)} legs)...')
    X_rest, y_rest, rest_kept = build_leg_sequences(holdout_truth, holdout_legs, v2_cols, args.seq_len)
    print(f'  HOLDOUT shape: {X_rest.shape}')

    print(f'Building OOS sequences ({len(oos_legs)} legs)...')
    X_oos, y_oos, oos_kept = build_leg_sequences(oos_truth, oos_legs, v2_cols, args.seq_len)
    print(f'  OOS shape: {X_oos.shape}')

    # Z-score using TRAIN-only stats (no peek)
    mu = X_train.reshape(-1, n_feat).mean(axis=0).astype(np.float32)
    sd = X_train.reshape(-1, n_feat).std(axis=0).astype(np.float32) + 1e-6
    print(f'\nZ-score stats computed from TRAIN only')

    def zscore(X):
        return ((X - mu) / sd).astype(np.float32)

    X_train_z = zscore(X_train)
    X_rest_z  = zscore(X_rest)
    X_oos_z   = zscore(X_oos)

    # Train/val split inside training window
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_train_z))
    n_val = int(0.2 * len(perm))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    X_tr  = torch.from_numpy(X_train_z[tr_idx]).to(device)
    y_tr  = torch.from_numpy(y_train[tr_idx]).to(device)
    X_val = torch.from_numpy(X_train_z[val_idx]).to(device)
    y_val = torch.from_numpy(y_train[val_idx]).to(device)
    print(f'  Train: {len(X_tr):,}   Val: {len(X_val):,}')

    # Model
    torch.manual_seed(args.seed)
    model = LegAmpLSTM(n_feat=n_feat, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: LSTM(hidden={args.hidden}) -> Linear(1)   params: {n_params:,}')

    # Train loop with early stopping
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_val = float('inf')
    best_state = None
    no_improve = 0
    print('\nTraining...')
    for epoch in range(args.epochs):
        model.train()
        total = 0.0; n = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        train_mae = total / n
        # Val
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae = loss_fn(val_pred, y_val).item()
        improvement = ' *' if val_mae < best_val else ''
        print(f'  epoch {epoch+1:>2}  train MAE {train_mae:.3f}   val MAE {val_mae:.3f}{improvement}')
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'  early stopping at epoch {epoch+1}')
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'\nBest val MAE: {best_val:.3f}')

    # Predict on rest-IS + OOS
    model.eval()
    @torch.no_grad()
    def predict_batched(X_z, batch_size=512):
        n = len(X_z)
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X_z[i:i+batch_size]).to(device)
            out[i:i+batch_size] = model(xb).cpu().numpy()
        return out

    print('Predicting HOLDOUT (last 40 days of IS)...')
    p_rest = predict_batched(X_rest_z)
    print('Predicting OOS...')
    p_oos = predict_batched(X_oos_z)

    # Metrics
    rest_mae = float(np.mean(np.abs(p_rest - y_rest)))
    oos_mae = float(np.mean(np.abs(p_oos - y_oos)))
    # Pearson
    if len(p_rest) > 1 and p_rest.std() > 0:
        rest_pearson = float(np.corrcoef(p_rest, y_rest)[0, 1])
    else:
        rest_pearson = float('nan')
    if len(p_oos) > 1 and p_oos.std() > 0:
        oos_pearson = float(np.corrcoef(p_oos, y_oos)[0, 1])
    else:
        oos_pearson = float('nan')

    # Baseline: predict median
    median_target = float(np.median(y_train))
    rest_mae_baseline = float(np.mean(np.abs(median_target - y_rest)))
    oos_mae_baseline = float(np.mean(np.abs(median_target - y_oos)))

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('C9 LSTM (failed candidate) LEG-AMPLITUDE REGRESSOR -- sequence input (60 bars x 184 features)')
    out('=' * 78)
    out(f'TRAIN window:    {train_days[0]} to {train_days[-1]} ({len(train_days)} days, {len(X_train):,} legs)')
    out(f'Train/Val split: {len(X_tr):,} / {len(X_val):,}')
    out(f'Best val MAE:    {best_val:.3f}')
    out('')
    out(f'HOLDOUT obs.   ({len(holdout_days)} days, {len(p_rest):,} legs):')
    out(f'  MAE              {rest_mae:.3f}')
    out(f'  Baseline (median): {rest_mae_baseline:.3f}')
    out(f'  Lift over baseline: {(1 - rest_mae/rest_mae_baseline)*100:.1f}% MAE reduction')
    out(f'  Pearson(pred, truth): {rest_pearson:+.4f}')
    out('')
    out(f'OOS NT8    (32 days, {len(p_oos):,} legs):')
    out(f'  MAE              {oos_mae:.3f}')
    out(f'  Baseline (median): {oos_mae_baseline:.3f}')
    out(f'  Lift over baseline: {(1 - oos_mae/oos_mae_baseline)*100:.1f}% MAE reduction')
    out(f'  Pearson(pred, truth): {oos_pearson:+.4f}')
    out('')
    out('--- Comparison to B7 GBM baseline ---')
    out(f'  B7 GBM     OOS Pearson: 0.2234   MAE: 1.029')
    out(f'  C9 LSTM (failed candidate)    OOS Pearson: {oos_pearson:+.4f}   MAE: {oos_mae:.3f}')
    if abs(oos_pearson) > 0.224:
        out(f'  --> LSTM beats GBM on OOS Pearson by {(oos_pearson - 0.2234)*100:+.2f} percentage points')
    else:
        out(f'  --> LSTM does NOT beat GBM Pearson (delta {(oos_pearson - 0.2234)*100:+.2f}pp)')

    # Save checkpoint + predictions + z-score stats
    torch.save({
        'model_state': model.state_dict(),
        'mu': mu, 'sd': sd, 'v2_cols': v2_cols,
        'seq_len': args.seq_len, 'hidden': args.hidden,
        'train_days': train_days,
        'best_val_mae': best_val,
    }, args.out_ckpt)
    print(f'\nSaved checkpoint: {args.out_ckpt}')

    # Save per-leg predictions cache (HOLDOUT and OOS).
    # rest_kept and oos_kept contain DataFrame INDEX values (from iterrows),
    # not positional. Use .loc.
    rest_legs_kept = holdout_legs.loc[rest_kept].copy()
    rest_legs_kept['pred_amp_R_lstm'] = p_rest
    rest_legs_kept[['day', 'entry_ts', 'leg_dir', 'entry_price', 'leg_amp_pts',
                     'leg_amp_usd', 'leg_amp_R', 'pnl_at_R_usd', 'r_price',
                     'atr_pts', 'pred_amp_R_lstm']].to_parquet(args.out_cache_is, index=False)

    oos_legs_kept = oos_legs.loc[oos_kept].copy()
    oos_legs_kept['pred_amp_R_lstm'] = p_oos
    oos_legs_kept[['day', 'entry_ts', 'leg_dir', 'entry_price', 'leg_amp_pts',
                     'leg_amp_usd', 'leg_amp_R', 'pnl_at_R_usd', 'r_price',
                     'atr_pts', 'pred_amp_R_lstm']].to_parquet(args.out_cache_oos, index=False)
    print(f'Saved REST-IS predictions: {args.out_cache_is}')
    print(f'Saved OOS predictions:     {args.out_cache_oos}')

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved report: {args.out_report}')


if __name__ == '__main__':
    main()
