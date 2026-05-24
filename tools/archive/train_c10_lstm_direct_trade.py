"""C10 — Direct-trade LSTM: predict signed forward return, trade on sign + magnitude.

NO zigzag. NO R-trigger. NO B1-B8 composite. Just price + V2 features.

Architecture:
  Input:  60 bars × 184 V2 features at 1m cadence
  LSTM:   hidden=128, layers=2, dropout=0.3 (slightly deeper for harder task)
  Head:   Linear(hidden, 1) — regression to signed K-min forward return in USD

Labels: for each 1m bar t, target = (close(t+K) - close(t)) * direction_normalizer.
  - For training, normalize by R = 4×ATR to standardize across days
  - target_R = (price_change_in_K_min) / r_price

Train: first ~237 days of IS  (chronological)
Holdout: last 40 days of IS (2-month observation proxy)
Test: NT8 OOS (32 days)

Trade simulation:
  At each 1m close where |pred_return_R| > threshold:
    direction = sign(pred)
    enter at close
    hold for K minutes
    exit at close(t+K)
  P&L = realized_return × $2/pt × size - $6 friction
  Size options:
    - 'flat': 1.0× always
    - 'magnitude': size proportional to |predicted return|

Multiple K values trained simultaneously: 5, 15, 30 minutes.

Per CLAUDE.md: CUDA-only, MAE loss, no lookahead.
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


SEQ_LEN = 60        # 60 1m bars of lead-in
HOLDOUT_DAYS = 40   # last N days of IS held out
DOLLAR_PER_POINT = 2.0
FRICTION_PER_TRADE = 6.00   # commission + slippage


class DirectTradeLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3, n_horizons: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Multi-horizon head: predict K-min returns for several K
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.head(self.dropout(h[-1]))   # (batch, n_horizons)


def build_sequences_with_forward_returns(truth_df, v2_cols,
                                            horizons_bars: list,
                                            seq_len: int = SEQ_LEN):
    """For each 1m close, build lead-in sequence + forward returns at K-bar horizons.

    Returns:
        X:     (n_samples, seq_len, n_feat) z-scored later
        Y_R:   (n_samples, n_horizons) forward returns in R-units (price_change / r_price)
        meta:  DataFrame with timestamp, day, r_price, close — for trade sim later
    """
    truth_df = truth_df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    n_feat = len(v2_cols)

    # We need 1m close prices; the truth parquet has timestamps at 1m bars,
    # but doesn't carry the close price directly. Load it from the 1m parquet
    # per day if not present.
    if 'close' not in truth_df.columns:
        # Pull from 1m parquets (project DATA/ATLAS or DATA/ATLAS_NT8 depending on day)
        from pathlib import Path
        close_lookup = {}
        for day in truth_df['day'].unique():
            for root in ['DATA/ATLAS', 'DATA/ATLAS_NT8']:
                p = Path(root) / '1m' / f'{day}.parquet'
                if p.exists():
                    df = pd.read_parquet(p)[['timestamp', 'close']]
                    close_lookup.update(dict(zip(df['timestamp'].astype('int64'),
                                                   df['close'].astype('float64'))))
                    break
        truth_df['close'] = truth_df['timestamp'].astype('int64').map(close_lookup)

    max_h = max(horizons_bars)
    seqs, targets_R, meta_rows = [], [], []

    for day, g in tqdm(truth_df.groupby('day'), desc='build seqs'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        feat = g[v2_cols].fillna(0.0).values.astype(np.float32)
        close = g['close'].values.astype(np.float64)
        r_price = float(g['min_rev_ticks'].iloc[0] * 0.25) if 'min_rev_ticks' in g.columns else 1.0
        ts = g['timestamp'].values.astype(np.int64)
        n = len(g)
        # Iterate over bars t where:
        #   - we have seq_len-1 prior bars (t >= seq_len-1)
        #   - we have max_h forward bars (t+max_h < n)
        for t in range(seq_len - 1, n - max_h):
            seqs.append(feat[t - seq_len + 1: t + 1])
            close_now = close[t]
            row_targets = []
            for h in horizons_bars:
                fwd_change_R = (close[t + h] - close_now) / r_price
                row_targets.append(float(fwd_change_R))
            targets_R.append(row_targets)
            meta_rows.append({
                'timestamp': int(ts[t]),
                'day': day,
                'close': float(close_now),
                'r_price': r_price,
            })
    X = np.stack(seqs, axis=0) if seqs else np.empty((0, seq_len, n_feat), dtype=np.float32)
    Y = np.array(targets_R, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, Y, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-ckpt',
                    default='reports/findings/regret_oracle/c10_lstm_direct_trade.pt')
    ap.add_argument('--out-cache-is',
                    default='reports/findings/regret_oracle/c10_lstm_predictions_HOLDOUT.parquet')
    ap.add_argument('--out-cache-oos',
                    default='reports/findings/regret_oracle/c10_lstm_predictions_OOS.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/c10_lstm_direct_trade.txt')
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--holdout-days', type=int, default=HOLDOUT_DAYS)
    ap.add_argument('--horizons', nargs='+', type=int, default=[5, 15, 30],
                    help='Forward horizons in 1m bars (5=5min, 15=15min, 30=30min)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading datasets...')
    is_truth = pd.read_parquet(args.is_truth)
    oos_truth = pd.read_parquet(args.oos_truth)
    print(f'  IS truth: {len(is_truth):,}   OOS truth: {len(oos_truth):,}')

    v2_cols = [c for c in is_truth.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    n_feat = len(v2_cols)
    print(f'V2 features: {n_feat}   horizons: {args.horizons} 1m bars')

    # Train = first N-holdout_days days, Holdout = last holdout_days days
    is_days_sorted = sorted(is_truth['day'].unique())
    train_days = is_days_sorted[:-args.holdout_days]
    holdout_days = is_days_sorted[-args.holdout_days:]
    print(f'\nTRAIN:    {train_days[0]} to {train_days[-1]} ({len(train_days)} days)')
    print(f'HOLDOUT:  {holdout_days[0]} to {holdout_days[-1]} ({len(holdout_days)} days)')

    train_truth = is_truth[is_truth['day'].isin(train_days)]
    holdout_truth = is_truth[is_truth['day'].isin(holdout_days)]

    print(f'\nBuilding TRAIN sequences...')
    X_train, Y_train, meta_train = build_sequences_with_forward_returns(
        train_truth, v2_cols, args.horizons, args.seq_len)
    print(f'  shape: {X_train.shape}   targets: {Y_train.shape}')

    print(f'Building HOLDOUT sequences...')
    X_hold, Y_hold, meta_hold = build_sequences_with_forward_returns(
        holdout_truth, v2_cols, args.horizons, args.seq_len)
    print(f'  shape: {X_hold.shape}')

    print(f'Building OOS sequences...')
    X_oos, Y_oos, meta_oos = build_sequences_with_forward_returns(
        oos_truth, v2_cols, args.horizons, args.seq_len)
    print(f'  shape: {X_oos.shape}')

    # Z-score features using TRAIN-only stats
    mu = X_train.reshape(-1, n_feat).mean(axis=0).astype(np.float32)
    sd = X_train.reshape(-1, n_feat).std(axis=0).astype(np.float32) + 1e-6

    def zscore(X):
        return ((X - mu) / sd).astype(np.float32)

    X_train_z = zscore(X_train)
    X_hold_z = zscore(X_hold)
    X_oos_z = zscore(X_oos)

    # Train/val split inside TRAIN
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_train_z))
    n_val = int(0.15 * len(perm))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    # Keep training tensors on CPU and move per-batch (saves GPU memory).
    X_tr = torch.from_numpy(X_train_z[tr_idx])
    Y_tr = torch.from_numpy(Y_train[tr_idx])
    X_val_np = X_train_z[val_idx]
    Y_val_np = Y_train[val_idx]
    print(f'  Train: {len(X_tr):,}   Val: {len(X_val_np):,}')

    # Model
    torch.manual_seed(args.seed)
    model = DirectTradeLSTM(
        n_feat=n_feat, hidden=args.hidden, layers=args.layers,
        n_horizons=len(args.horizons),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: LSTM(hidden={args.hidden}, layers={args.layers}) -> Linear({len(args.horizons)})   params: {n_params:,}')

    # Train
    train_ds = TensorDataset(X_tr, Y_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               pin_memory=True, num_workers=0)
    best_val = float('inf')
    best_state = None
    no_improve = 0

    @torch.no_grad()
    def eval_batched(X_np, Y_np, batch_size=512):
        model.eval()
        total = 0.0; n = 0
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(device, non_blocking=True)
            yb = torch.from_numpy(Y_np[i:i+batch_size]).to(device, non_blocking=True)
            pred = model(xb)
            total += loss_fn(pred, yb).item() * len(yb)
            n += len(yb)
        return total / max(n, 1)

    print('\nTraining...')
    for epoch in range(args.epochs):
        model.train()
        total = 0.0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        train_mae = total / n
        val_mae = eval_batched(X_val_np, Y_val_np, batch_size=512)
        flag = ' *' if val_mae < best_val else ''
        print(f'  epoch {epoch+1:>2}  train MAE {train_mae:.4f}   val MAE {val_mae:.4f}{flag}')
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'  early stop at epoch {epoch+1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'\nBest val MAE: {best_val:.4f}')

    # Predict
    model.eval()
    @torch.no_grad()
    def predict(X_z, batch_size=512):
        out = np.empty((len(X_z), len(args.horizons)), dtype=np.float32)
        for i in range(0, len(X_z), batch_size):
            xb = torch.from_numpy(X_z[i:i+batch_size]).to(device)
            out[i:i+batch_size] = model(xb).cpu().numpy()
        return out

    print('Predicting HOLDOUT...')
    p_hold = predict(X_hold_z)
    print('Predicting OOS...')
    p_oos = predict(X_oos_z)

    # Save per-bar predictions
    for k_i, h in enumerate(args.horizons):
        meta_hold[f'pred_return_R_{h}m'] = p_hold[:, k_i]
        meta_hold[f'actual_return_R_{h}m'] = Y_hold[:, k_i]
        meta_oos[f'pred_return_R_{h}m'] = p_oos[:, k_i]
        meta_oos[f'actual_return_R_{h}m'] = Y_oos[:, k_i]
    meta_hold.to_parquet(args.out_cache_is, index=False)
    meta_oos.to_parquet(args.out_cache_oos, index=False)

    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'mu': mu, 'sd': sd, 'v2_cols': v2_cols,
        'seq_len': args.seq_len, 'hidden': args.hidden, 'layers': args.layers,
        'horizons': args.horizons,
        'best_val_mae': best_val,
    }, args.out_ckpt)

    # Report metrics per horizon
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('C10 LSTM (failed candidate) DIRECT-TRADE -- signed forward-return prediction (no zigzag)')
    out('=' * 78)
    out(f'Architecture: LSTM(hidden={args.hidden}, layers={args.layers}) -> Linear({len(args.horizons)})')
    out(f'Params: {n_params:,}')
    out(f'Train: {len(X_tr):,} samples   Val: {len(X_val):,}   Holdout: {len(X_hold):,}   OOS: {len(X_oos):,}')
    out(f'Best val MAE: {best_val:.4f}')
    out('')
    out(f'{"Horizon":<10}  {"holdout MAE":>11}  {"OOS MAE":>9}  {"holdout R":>10}  {"OOS R":>8}  {"baseline MAE":>13}')
    for k_i, h in enumerate(args.horizons):
        hold_mae = float(np.mean(np.abs(p_hold[:, k_i] - Y_hold[:, k_i])))
        oos_mae = float(np.mean(np.abs(p_oos[:, k_i] - Y_oos[:, k_i])))
        hold_r = float(np.corrcoef(p_hold[:, k_i], Y_hold[:, k_i])[0, 1]) if p_hold[:, k_i].std() > 0 else float('nan')
        oos_r = float(np.corrcoef(p_oos[:, k_i], Y_oos[:, k_i])[0, 1]) if p_oos[:, k_i].std() > 0 else float('nan')
        # Baseline = predict 0 (no movement)
        hold_baseline = float(np.mean(np.abs(Y_hold[:, k_i])))
        out(f'  {h:>3}min    {hold_mae:>10.4f}   {oos_mae:>8.4f}   {hold_r:>+9.4f}   {oos_r:>+7.4f}   {hold_baseline:>12.4f}')

    out('')
    out('Headline:')
    out('  - holdout R > 0 means LSTM extracts signal within IS distribution')
    out('  - OOS R measures how well that signal generalizes to NT8 2026 data')
    out('  - If OOS R near 0 -> model overfits to IS, no live edge')
    out('  - If OOS R > 0.1 -> meaningful signal worth simulating trades from')
    out('')
    out(f'Next: run forward_pass_c10_lstm.py to simulate trading on these predictions.')

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_ckpt}')
    print(f'Wrote: {args.out_cache_is}')
    print(f'Wrote: {args.out_cache_oos}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
