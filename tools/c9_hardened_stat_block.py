"""Honest C9 LSTM (failed candidate) stat block — apply at R-trigger entry bar (hardened).

Distinction from prior C9 analysis (failed candidate):
  - Prior: C9 predicted at PIVOT bar → applied to PIVOT-entry legs → oracle.
  - Now:   C9 predicted at R-TRIGGER fire bar → applied to hardened R-trigger legs.

This is the realistic deployment result. Day WR should NOT be 100%
because R-trigger entries give back 2R/leg via the entry+exit lag.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SEQ_LEN = 60


class LegAmpLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden,
                              num_layers=layers,
                              dropout=dropout if layers > 1 else 0.0,
                              batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.head(self.dropout(h[-1])).squeeze(-1)


def gbm_ev(p):
    return float(np.clip(max(p - 1.0, 0.0), 0.0, 3.0))


def main():
    print('Loading inputs...')
    truth = pd.read_parquet('reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    hardened = pd.read_csv('reports/findings/regret_oracle/composite_forward_pass_hardened.csv')

    ckpt = torch.load('reports/findings/regret_oracle/c9_lstm_leg_sizer.pt', weights_only=False)
    mu = ckpt['mu']; sd = ckpt['sd']; v2_cols = ckpt['v2_cols']
    hidden = ckpt['hidden']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LegAmpLSTM(n_feat=len(v2_cols), hidden=hidden, layers=1, dropout=0.3).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Build sequence per hardened leg
    truth_sorted = truth.sort_values(['day', 'timestamp']).reset_index(drop=True)
    feat_arr = truth_sorted[v2_cols].fillna(0.0).values.astype(np.float32)
    day_to_idx = {day: g.index.values for day, g in truth_sorted.groupby('day')}
    day_to_ts  = {day: g['timestamp'].values.astype(np.int64) for day, g in truth_sorted.groupby('day')}

    seqs = []
    keep_idx = []
    for li, leg in hardened.iterrows():
        day = leg['day']
        entry_ts = int(leg['entry_ts'])
        if day not in day_to_ts:
            continue
        ts_day = day_to_ts[day]
        idx_in_global = day_to_idx[day]
        bar_idx = int(np.searchsorted(ts_day, entry_ts, side='right') - 1)
        if bar_idx < SEQ_LEN - 1:
            continue
        chosen = idx_in_global[bar_idx - SEQ_LEN + 1: bar_idx + 1]
        seqs.append(feat_arr[chosen])
        keep_idx.append(li)
    X = np.stack(seqs, axis=0)
    print(f'  Built {len(X):,} R-trigger-bar sequences')

    # Z-score with training stats
    X_z = ((X - mu) / sd).astype(np.float32)

    # Predict in batches
    preds = np.empty(len(X_z), dtype=np.float32)
    with torch.no_grad():
        BS = 512
        for i in range(0, len(X_z), BS):
            xb = torch.from_numpy(X_z[i:i+BS]).to(device)
            preds[i:i+BS] = model(xb).cpu().numpy()

    # Build the hardened DataFrame with C9 predictions
    df = hardened.loc[keep_idx].copy().reset_index(drop=True)
    df['pred_amp_R_c9_hardened'] = preds

    # Apply gbm_ev sizing
    sizes_c9 = np.array([gbm_ev(p) for p in df['pred_amp_R_c9_hardened'].values])
    sizes_b7 = np.array([gbm_ev(p) for p in df['pred_amp_R_hardened'].values])
    sizes_flat = np.ones(len(df))
    pnl = df['pnl_usd'].values

    def stat_block(name, sizes):
        wpnl = pnl * sizes
        df_t = df.copy()
        df_t['wpnl'] = wpnl
        per_day = df_t.groupby('day')['wpnl'].sum().values
        n_days = len(per_day)
        n_win = sum(1 for v in per_day if v > 0)
        n_over_200 = sum(1 for v in per_day if v > 200)
        day_wr = n_win / n_days
        legs = wpnl[sizes > 0]
        profit_w = legs[legs > 0].sum()
        loss_l = abs(legs[legs < 0].sum())
        pf = profit_w / max(loss_l, 1e-9)
        pf_trade_wr = pf - 1
        leg_win_pct = (legs > 0).mean() if len(legs) else 0
        print(f'\n--- {name} ---')
        print(f'  Day WR:         {day_wr*100:.1f}%  ({n_win}/{n_days})')
        print(f'  Days >$200:     {n_over_200}/{n_days} ({n_over_200/n_days*100:.1f}%)')
        print(f'  Legs taken:     {len(legs):,}')
        print(f'  Winners:        {int((legs>0).sum()):,} ({leg_win_pct*100:.1f}%)')
        print(f'  PF:             {pf:.3f}')
        print(f'  PF Trade WR:    {pf_trade_wr:+.3f}')
        print(f'  Mean per leg:   ${legs.mean() if len(legs) else 0:+.2f}')
        print(f'  Median per leg: ${np.median(legs) if len(legs) else 0:+.2f}')
        print(f'  Mean per day:   ${per_day.mean():+.2f}')
        print(f'  Median per day: ${np.median(per_day):+.2f}')
        print(f'  Min / Max day:  ${per_day.min():+.2f} / ${per_day.max():+.2f}')
        print(f'  Total OOS:      ${legs.sum():+,.2f}')

    print('=' * 78)
    print('HARDENED OOS — C9 LSTM (failed candidate) applied at R-trigger fire bar (apples-to-apples vs B7)')
    print('=' * 78)
    print(f'Days: {df["day"].nunique()}   Hardened legs with sequence: {len(df):,}')

    stat_block('C9 LSTM (failed candidate) hardened (gbm_ev sizing)', sizes_c9)
    stat_block('B7 GBM hardened (gbm_ev sizing) — same legs for comparison', sizes_b7)
    stat_block('FLAT (no sizing) — same legs', sizes_flat)


if __name__ == '__main__':
    main()
