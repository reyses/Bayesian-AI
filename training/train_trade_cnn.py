"""
TradeCNN Training Pipeline — Walk-Forward State Prediction.

Predicts future feature states (not price direction directly).
The trading logic interprets predicted states into entry/exit decisions.

Phases:
  --phase labels    : Build + validate feature/label pipeline
  --phase train     : Walk-forward training (Model A)
  --phase all       : labels + train

Usage:
  python -m training.train_trade_cnn --phase labels
  python -m training.train_trade_cnn --phase all --model A
"""
import argparse
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
import glob

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- PATHS ---
IS_ROOT = 'DATA/ATLAS'
OOS_ROOT = 'DATA/ATLAS_OOS'
CHECKPOINT_DIR = 'checkpoints/trade_cnn'
RESULTS_LOG = 'reports/findings/experiment_log.txt'
TICK = 0.25

# --- HORIZONS ---
HORIZONS_FAST = [1, 5, 10]    # v1: scalping (1-10 bar moves)
HORIZONS_HOLD = [5, 10, 20]   # v2: sustained moves (5-20 bars)
HORIZONS_10 = [10]            # v3: single horizon at sweet spot (10 min)
HORIZONS = HORIZONS_FAST      # default, overridden by --horizons
MAX_FORWARD = max(HORIZONS)

# --- 13D FEATURES ---
FEATURE_NAMES_7D = ['dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel']
FEATURE_NAMES_REGIME = ['std_price', 'variance_ratio', 'bar_range', 'wick_ratio']
FEATURE_NAMES_CONTEXT = ['vwap_distance', 'time_of_day']
FEATURE_NAMES_13D = FEATURE_NAMES_7D + FEATURE_NAMES_REGIME + FEATURE_NAMES_CONTEXT
N_FEAT = len(FEATURE_NAMES_13D)  # 13

# Label: 7D features at each horizon = 7 * 3 = 21 outputs
N_LABELS = len(FEATURE_NAMES_7D) * len(HORIZONS)  # 21
LOOKBACK = 10


def extract_features_13d(states, df):
    """Extract 13D grounded features per bar from SFE states + OHLCV.

    7D directional (same as direction_cnn):
      dmi_diff, dmi_gap, vol_rel, dir_vol, velocity, z_se, price_accel
    4D regime:
      std_price, variance_ratio, bar_range, wick_ratio
    2D context:
      vwap_distance, time_of_day
    """
    n = len(states)
    feats = np.zeros((n, N_FEAT), dtype=np.float32)

    prices = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(n)
    timestamps = df['timestamp'].values.astype(np.float64)

    # Rolling 30-bar volume SMA (matches training/live parity)
    vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values

    prev_vel = 0.0

    for i in range(n):
        st = states[i]['state'] if isinstance(states[i], dict) else states[i]
        dmi_p = getattr(st, 'dmi_plus', 0.0)
        dmi_m = getattr(st, 'dmi_minus', 0.0)
        vel = getattr(st, 'velocity', 0.0)
        vol = volumes[i]
        _va = vol_avg[i] if vol_avg[i] > 0 else 1.0

        # --- 7D directional ---
        feats[i, 0] = dmi_p - dmi_m                            # dmi_diff
        feats[i, 1] = abs(dmi_p - dmi_m)                       # dmi_gap
        feats[i, 2] = vol / _va                                 # vol_rel
        if i > 0:
            _dir = 1.0 if prices[i] > prices[i-1] else -1.0
            feats[i, 3] = _dir * vol / _va                      # dir_vol
        feats[i, 4] = vel                                        # velocity
        if i >= 15:
            _window = prices[max(0, i-60):i+1]
            _mean = _window.mean()
            _std = _window.std()
            _se = _std / (len(_window) ** 0.5) if len(_window) > 1 else _std
            feats[i, 5] = (prices[i] - _mean) / _se if _se > 1e-8 else 0.0  # z_se
        feats[i, 6] = vel - prev_vel                             # price_accel
        prev_vel = vel

        # --- 4D regime ---
        if i >= 30:
            feats[i, 7] = np.std(prices[i-30:i+1])              # std_price
            if i >= 60:
                _short_std = np.std(prices[i-10:i+1])
                _long_std = np.std(prices[i-60:i+1])
                feats[i, 8] = _short_std / _long_std if _long_std > 1e-8 else 1.0  # variance_ratio

        _range = highs[i] - lows[i]
        feats[i, 9] = _range / TICK                              # bar_range (ticks)
        if _range > 0:
            _body = abs(prices[i] - opens[i])
            feats[i, 10] = 1.0 - (_body / _range)                # wick_ratio

        # --- 2D context ---
        if i >= 30:
            _vwap_num = np.sum(prices[i-30:i+1] * volumes[i-30:i+1])
            _vwap_den = np.sum(volumes[i-30:i+1])
            _vwap = _vwap_num / _vwap_den if _vwap_den > 0 else prices[i]
            feats[i, 11] = (prices[i] - _vwap) / TICK            # vwap_distance (ticks)
        feats[i, 12] = (timestamps[i] % 86400) / 86400.0         # time_of_day (0-1)

    return feats


def build_state_labels(feats_7d, horizons=HORIZONS):
    """Build state prediction labels: actual 7D features at t+h for each horizon.

    For each bar i, label = [feat_7d[i+1], feat_7d[i+5], feat_7d[i+10]].
    Returns (n_bars, 21) array. Bars without full forward window get zeros.
    """
    n = len(feats_7d)
    n_feat = feats_7d.shape[1]  # 7
    n_horizons = len(horizons)
    labels = np.zeros((n, n_feat * n_horizons), dtype=np.float32)

    for i in range(n):
        for hi, h in enumerate(horizons):
            if i + h < n:
                labels[i, hi * n_feat:(hi + 1) * n_feat] = feats_7d[i + h]

    return labels


def build_dataset(data_root, max_bars=0):
    """Load data, compute states, extract 13D features, build 21D labels."""
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

    print("Extracting 13D features...")
    feats = extract_features_13d(states, df)

    print("Building state labels (7D x 3 horizons = 21D)...")
    labels = build_state_labels(feats[:, :7], horizons=HORIZONS)  # labels from 7D directional only

    return feats, labels, states, df


class SlidingWindowDataset(Dataset):
    """Sliding window: input=(lookback x 13D), label=(21D state predictions)."""

    def __init__(self, features, labels, lookback=LOOKBACK):
        self.features = features
        self.labels = labels
        self.lookback = lookback
        self.n = len(features) - lookback - MAX_FORWARD

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        i = idx + self.lookback
        x = self.features[i - self.lookback:i]  # (lookback, 13)
        y = self.labels[i]                        # (21,)
        return torch.FloatTensor(x), torch.FloatTensor(y)


def validate_pipeline(feats, labels, df):
    """Print feature/label distributions and sanity checks."""
    print(f"\n{'='*60}")
    print(f"PIPELINE VALIDATION")
    print(f"{'='*60}")
    print(f"  Samples: {len(feats):,}")
    print(f"  Features: {feats.shape[1]}D")
    print(f"  Labels: {labels.shape[1]}D")

    # Feature distributions
    print(f"\n  FEATURE DISTRIBUTIONS:")
    print(f"  {'Name':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'NaN':>5} {'Zero%':>6}")
    for j, name in enumerate(FEATURE_NAMES_13D):
        col = feats[:, j]
        _nan = np.isnan(col).sum()
        _zero = (col == 0).mean() * 100
        print(f"  {name:<20} {col.mean():>10.3f} {col.std():>10.3f} "
              f"{col.min():>10.3f} {col.max():>10.3f} {_nan:>5} {_zero:>5.1f}%")

    # Label distributions
    print(f"\n  LABEL DISTRIBUTIONS (7D x {len(HORIZONS)} horizons):")
    print(f"  {'Name':<25} {'Mean':>10} {'Std':>10} {'Var>0':>6}")
    for hi, h in enumerate(HORIZONS):
        for fi, fname in enumerate(FEATURE_NAMES_7D):
            idx = hi * 7 + fi
            col = labels[:, idx]
            _var_ok = 'YES' if col.std() > 1e-6 else 'NO'
            print(f"  {fname}_t{h:<18} {col.mean():>10.3f} {col.std():>10.3f} {_var_ok:>6}")

    # NaN/Inf check
    _nan_feats = np.isnan(feats).sum()
    _inf_feats = np.isinf(feats).sum()
    _nan_labels = np.isnan(labels).sum()
    print(f"\n  NaN in features: {_nan_feats}")
    print(f"  Inf in features: {_inf_feats}")
    print(f"  NaN in labels: {_nan_labels}")

    if _nan_feats > 0 or _inf_feats > 0:
        print("  WARNING: NaN/Inf detected — clean before training!")

    # Feature correlation (check redundancy)
    print(f"\n  FEATURE CORRELATION (top pairs with |r| > 0.8):")
    from scipy import stats as sp_stats
    for i in range(N_FEAT):
        for j in range(i+1, N_FEAT):
            r, _ = sp_stats.pearsonr(feats[:, i], feats[:, j])
            if abs(r) > 0.8:
                print(f"    {FEATURE_NAMES_13D[i]} <-> {FEATURE_NAMES_13D[j]}: r={r:.3f}")

    # Trading days
    trading_days = pd.to_datetime(df['timestamp'], unit='s').dt.date.nunique()
    print(f"\n  Trading days: {trading_days}")
    print(f"  Bars/day: {len(feats) / trading_days:.0f}")

    # Save validation report
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    report = {
        'samples': len(feats), 'features': feats.shape[1], 'labels': labels.shape[1],
        'trading_days': trading_days, 'nan_feats': int(_nan_feats), 'inf_feats': int(_inf_feats),
    }
    with open(os.path.join(CHECKPOINT_DIR, 'pipeline_validation.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {CHECKPOINT_DIR}/pipeline_validation.json")


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='TradeCNN training pipeline')
    parser.add_argument('--phase', default='labels', choices=['labels', 'train', 'all', 'oos'])
    parser.add_argument('--model', default='A', choices=['A'])
    parser.add_argument('--max-bars', type=int, default=0)
    parser.add_argument('--horizons', default='fast', choices=['fast', 'hold', '10'],
                        help='fast=[1,5,10] scalp, hold=[5,10,20] sustained, 10=[10] sweet spot')
    args = parser.parse_args()

    # Set horizons and checkpoint dir based on mode
    global HORIZONS, MAX_FORWARD, N_LABELS, CHECKPOINT_DIR
    if args.horizons == 'hold':
        HORIZONS = HORIZONS_HOLD
        CHECKPOINT_DIR = 'checkpoints/trade_cnn_hold'
    elif args.horizons == '10':
        HORIZONS = HORIZONS_10
        CHECKPOINT_DIR = 'checkpoints/trade_cnn_10'
    else:
        HORIZONS = HORIZONS_FAST
        CHECKPOINT_DIR = 'checkpoints/trade_cnn'
    MAX_FORWARD = max(HORIZONS)
    N_LABELS = len(FEATURE_NAMES_7D) * len(HORIZONS)
    print(f"Horizons: {HORIZONS} -> {N_LABELS}D labels -> {CHECKPOINT_DIR}")

    if args.phase in ('labels', 'all'):
        t0 = time.time()
        feats, labels, states, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
        print(f"Dataset built in {time.time()-t0:.1f}s")
        validate_pipeline(feats, labels, df)

        # Save features + labels for reuse
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        np.save(os.path.join(CHECKPOINT_DIR, 'is_features_13d.npy'), feats)
        np.save(os.path.join(CHECKPOINT_DIR, 'is_labels_21d.npy'), labels)
        print(f"  Saved: {CHECKPOINT_DIR}/is_features_13d.npy ({feats.shape})")
        print(f"  Saved: {CHECKPOINT_DIR}/is_labels_21d.npy ({labels.shape})")

        # Release memory
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if args.phase in ('train', 'all'):
        from core.trade_cnn import StatePredictor

        # Load or compute features
        _feat_path = os.path.join(CHECKPOINT_DIR, 'is_features_13d.npy')
        _label_path = os.path.join(CHECKPOINT_DIR, 'is_labels_21d.npy')
        if os.path.exists(_feat_path) and os.path.exists(_label_path):
            print("Loading cached features + labels...")
            feats = np.load(_feat_path)
            labels = np.load(_label_path)
            # Need df for day boundaries
            files = sorted(glob.glob(os.path.join(IS_ROOT, '1m', '*.parquet')))
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"  Features: {feats.shape}, Labels: {labels.shape}")
        else:
            print("Features not cached — run --phase labels first")
            return

        walk_forward_train(feats, labels, df, args)

    if args.phase == 'oos':
        oos_single_pass()


def walk_forward_train(feats, labels, df, args):
    """Walk-forward training: carry-forward model, score before train each day."""
    from core.trade_cnn import StatePredictor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Split data into days
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    day_boundaries = []
    for date, group in df.groupby('date'):
        _start = group.index[0]
        _end = group.index[-1]
        day_boundaries.append({'date': date, 'start': _start, 'end': _end, 'n_bars': len(group)})
    print(f"  Days: {len(day_boundaries)}")

    # Model
    model = StatePredictor(n_features=N_FEAT, latent_dim=64, n_labels=N_LABELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    day_results = []

    for di, day in enumerate(tqdm(day_boundaries, desc="Walk-Forward")):
        _start = day['start']
        _end = day['end']
        _date = day['date']

        # Skip days with too few bars
        if _end - _start < LOOKBACK + MAX_FORWARD + 10:
            continue

        day_feats = feats[_start:_end+1]
        day_labels = labels[_start:_end+1]

        # --- SCORE (predict BEFORE training on this day) ---
        if di > 0:  # Day 1 has no model to score with
            model.eval()
            day_ds = SlidingWindowDataset(day_feats, day_labels, lookback=LOOKBACK)
            if len(day_ds) > 0:
                score = _validate_day(model, day_ds, day_feats, day_labels, device)
                score['date'] = str(_date)
                score['day'] = di + 1

                # Trading simulation
                trade_result = _simulate_day_trading(model, day_feats, device)
                score.update(trade_result)

                day_results.append(score)

                if (di + 1) % 30 == 0:
                    _cum_pnl = sum(r.get('sim_pnl', 0) for r in day_results)
                    print(f"  Day {di+1}: corr={score.get('avg_corr', 0):.3f} "
                          f"dir={score.get('dir_acc', 0):.1f}% "
                          f"pnl={score.get('sim_pnl', 0):+.0f}t "
                          f"cum=${_cum_pnl*0.5:,.0f}")

        # --- TRAIN on this day ---
        model.train()
        day_ds = SlidingWindowDataset(day_feats, day_labels, lookback=LOOKBACK)
        if len(day_ds) < 10:
            continue

        dl = DataLoader(day_ds, batch_size=min(256, len(day_ds)), shuffle=True)

        # Cold start (Day 1): 30 epochs. Carry-forward: 5 epochs with lower LR
        if di == 0:
            _epochs = 30
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3
        else:
            _epochs = 5
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4

        for _ep in range(_epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Checkpoint every 10 days
        if (di + 1) % 10 == 0:
            torch.save({
                'model_state': model.state_dict(),
                'day': di + 1, 'date': str(_date),
            }, os.path.join(CHECKPOINT_DIR, f'model_day{di+1}.pt'))

    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'day': len(day_boundaries), 'date': str(day_boundaries[-1]['date']),
    }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

    # Walk-forward report
    walk_forward_report(day_results)

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _validate_day(model, dataset, feats, labels, device):
    """Per-day validation: feature correlations + direction accuracy."""
    from scipy import stats as sp_stats

    model.eval()
    all_pred = []
    all_true = []

    dl = DataLoader(dataset, batch_size=512, shuffle=False)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            pred = model(x)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    # Correlation per feature per horizon
    corrs = []
    for j in range(N_LABELS):
        if true[:, j].std() > 1e-8 and pred[:, j].std() > 1e-8:
            r, _ = sp_stats.spearmanr(pred[:, j], true[:, j])
            corrs.append(r)
        else:
            corrs.append(0.0)

    # Direction accuracy: sign of predicted dmi_diff at LAST horizon vs actual
    # For [1,5,10]: index 14. For [10]: index 0. For [5,10,20]: index 14.
    _last_h_idx = (len(HORIZONS) - 1) * 7  # dmi_diff at furthest horizon
    _dir_idx = min(_last_h_idx, pred.shape[1] - 1)
    _pred_dir = pred[:, _dir_idx] > 0
    _true_dir = true[:, _dir_idx] > 0
    dir_acc = (_pred_dir == _true_dir).mean() * 100

    return {
        'avg_corr': np.mean(corrs),
        'dir_acc': dir_acc,
        'n_samples': len(pred),
    }


def _simulate_day_trading(model, feats, device):
    """Simulate trading from predicted states."""
    model.eval()

    SL = 40  # hard SL in ticks
    MIN_HOLD = 3  # minimum bars before exit
    trades = []
    in_trade = False
    trade_dir = ''
    entry_price_idx = 0
    bars_held = 0

    for i in range(LOOKBACK, len(feats) - MAX_FORWARD):
        x = feats[i - LOOKBACK:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_t).cpu().numpy()[0]

        # Extract predicted states (adaptive to horizon count)
        n_h = len(HORIZONS)
        _pred_dmi = pred[0]               # dmi_diff at first/only horizon
        _pred_gap = pred[1]               # dmi_gap at first/only horizon
        _pred_vel = pred[4]               # velocity at first/only horizon
        # For multi-horizon, use the last horizon's dmi_diff
        if n_h >= 2:
            _pred_dmi = pred[(n_h - 1) * 7]
            _pred_gap = pred[(n_h - 1) * 7 + 1]
            _pred_vel = pred[(n_h - 1) * 7 + 4]

        if in_trade:
            bars_held += 1

            # Exit: trend reversed OR momentum fading (after min hold)
            if bars_held >= MIN_HOLD:
                _trend_reversed = (trade_dir == 'LONG' and _pred_dmi < -2) or \
                                  (trade_dir == 'SHORT' and _pred_dmi > 2)
                _momentum_fading = _pred_gap < feats[i, 1] * 0.5  # gap shrinking

                if _trend_reversed or _momentum_fading:
                    # Compute actual PnL from prices (if available via label reconstruction)
                    _actual_dmi_t5 = feats[min(i+5, len(feats)-1), 0]  # actual dmi_diff 5 bars later
                    _pnl = abs(_actual_dmi_t5) * (1 if (trade_dir == 'LONG') == (_actual_dmi_t5 > 0) else -1)
                    trades.append({'pnl': _pnl, 'bars': bars_held, 'dir': trade_dir})
                    in_trade = False

        if not in_trade:
            # Entry: predicted gap growing + velocity confirming + confidence
            _gap_building = _pred_gap > feats[i, 1] * 1.2
            _vel_confirming = abs(_pred_vel) > abs(feats[i, 4])
            _confident = abs(_pred_dmi) > 2.0

            if _confident and (_gap_building or _vel_confirming):
                in_trade = True
                trade_dir = 'LONG' if _pred_dmi > 0 else 'SHORT'
                entry_price_idx = i
                bars_held = 0

    total_pnl = sum(t['pnl'] for t in trades)
    n_trades = len(trades)
    n_wins = len([t for t in trades if t['pnl'] > 0])

    return {
        'sim_pnl': total_pnl,
        'sim_trades': n_trades,
        'sim_wr': n_wins / n_trades * 100 if n_trades > 0 else 0,
    }


def walk_forward_report(day_results):
    """Summary report across all scored days."""
    if not day_results:
        print("\nNo scored days — walk-forward report empty")
        return

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD SUMMARY: StatePredictor")
    print(f"{'='*60}")

    n_days = len(day_results)
    cum_pnl = sum(r.get('sim_pnl', 0) for r in day_results)
    avg_corr = np.mean([r.get('avg_corr', 0) for r in day_results])
    avg_dir = np.mean([r.get('dir_acc', 0) for r in day_results])
    avg_trades = np.mean([r.get('sim_trades', 0) for r in day_results])
    profitable_days = len([r for r in day_results if r.get('sim_pnl', 0) > 0])

    print(f"  Scored days: {n_days}")
    print(f"  Cumulative PnL: {cum_pnl:.0f}t (${cum_pnl*0.5:,.0f})")
    print(f"  $/day avg: ${cum_pnl*0.5/n_days:.2f}")
    print(f"  Profitable days: {profitable_days}/{n_days} ({profitable_days/n_days*100:.0f}%)")
    print(f"  Avg feature correlation: {avg_corr:.4f}")
    print(f"  Avg direction accuracy: {avg_dir:.1f}%")
    print(f"  Avg trades/day: {avg_trades:.1f}")
    print(f"  vs Baseline: $736/day (direction CNN)")

    # Monthly breakdown
    _monthly = {}
    for r in day_results:
        _month = r['date'][:7]
        if _month not in _monthly:
            _monthly[_month] = []
        _monthly[_month].append(r.get('sim_pnl', 0))

    print(f"\n  MONTHLY BREAKDOWN:")
    for m in sorted(_monthly):
        _pnls = _monthly[m]
        _total = sum(_pnls)
        _days = len(_pnls)
        print(f"    {m}: ${_total*0.5:>8,.0f} ({_days} days, ${_total*0.5/_days:.0f}/day)")

    # Save results
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    _results = {
        'n_days': n_days, 'cum_pnl_ticks': cum_pnl, 'cum_pnl_dollars': cum_pnl * 0.5,
        'per_day': cum_pnl * 0.5 / n_days, 'avg_corr': avg_corr,
        'avg_dir_acc': avg_dir, 'profitable_days_pct': profitable_days / n_days * 100,
        'day_results': day_results,
    }
    with open(os.path.join(CHECKPOINT_DIR, 'walk_forward_results.json'), 'w') as f:
        json.dump(_results, f, indent=2, default=str)
    print(f"  Saved: {CHECKPOINT_DIR}/walk_forward_results.json")

    # Append to experiment log
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"model=TradeCNN_A | days={n_days} | "
             f"corr={avg_corr:.4f} | dir={avg_dir:.1f}% | "
             f"PnL=${cum_pnl*0.5:,.0f} | $/day=${cum_pnl*0.5/n_days:.0f}\n")
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)
    print(f"  Logged: {RESULTS_LOG}")


def oos_single_pass():
    """Single forward pass on OOS with the trained model. Simplified trading + full logging."""
    from core.trade_cnn import StatePredictor
    import csv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(_ckpt_path):
        print(f"No model found at {_ckpt_path} — run --phase train first")
        return

    ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    model = StatePredictor(n_features=N_FEAT, latent_dim=64, n_labels=N_LABELS).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded model from day {ckpt.get('day', '?')}")

    # Build OOS features
    feats, labels, states, df = build_dataset(OOS_ROOT)
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values

    # Load 1s data for fill price lookup (actual price 2s after signal)
    print("Loading 1s data for fill delay...")
    _1s_files = sorted(glob.glob(os.path.join(OOS_ROOT, '1s', '*.parquet')))
    if _1s_files:
        df_1s = pd.concat([pd.read_parquet(f) for f in _1s_files], ignore_index=True)
        df_1s = df_1s.sort_values('timestamp').reset_index(drop=True)
        _1s_ts = df_1s['timestamp'].values
        _1s_close = df_1s['close'].values
        print(f"  1s bars: {len(df_1s):,}")
    else:
        df_1s = None
        _1s_ts = None
        _1s_close = None
        print("  No 1s data — using close price (no delay)")

    def _get_fill_price(signal_ts):
        """Get price FILL_DELAY_S seconds after signal timestamp."""
        if _1s_ts is None:
            return None
        _target_ts = signal_ts + FILL_DELAY_S
        _idx = np.searchsorted(_1s_ts, _target_ts)
        if _idx < len(_1s_close):
            return float(_1s_close[_idx])
        return None

    # Simple trading: follow predicted direction, trail after $5, SL=40
    SL = 40
    BE_ACT = 5       # move SL to breakeven after 5 ticks profit (direction confirmed)
    TRAIL_ACT = 10   # activate trail after 10 ticks profit
    TRAIL_DIST = 10  # trail distance from peak
    CONF_THRESHOLD = 3.0  # minimum confidence to enter (was 2.0)
    FILL_DELAY_S = 2  # seconds from signal to fill

    trades = []
    trade_log = []  # full state log per trade
    in_trade = False
    trade_dir = ''
    entry_price = 0.0
    entry_bar = 0
    peak_price = 0.0
    trail_active = False

    for i in tqdm(range(LOOKBACK, len(feats) - MAX_FORWARD), desc="OOS Single Pass"):
        x = feats[i - LOOKBACK:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_t).cpu().numpy()[0]

        price = prices[i]
        high = highs[i]
        low = lows[i]

        # Predicted direction from dmi_diff — adapt to number of horizons
        n_h = len(HORIZONS)
        _h0_dmi = pred[0]                # dmi_diff at first horizon
        if n_h >= 2:
            _h1_dmi = pred[7]
        else:
            _h1_dmi = _h0_dmi
        if n_h >= 3:
            _h2_dmi = pred[14]
        else:
            _h2_dmi = _h1_dmi

        # Primary prediction: last available horizon
        _pred_dmi = pred[(n_h - 1) * 7]  # dmi_diff at the furthest horizon
        _pred_dir = 'LONG' if _pred_dmi > 0 else 'SHORT'
        _confidence = abs(_pred_dmi)
        # All horizons agree
        _all_agree = True
        for _hi in range(1, n_h):
            if np.sign(pred[_hi * 7]) != np.sign(pred[0]):
                _all_agree = False
                break

        if in_trade:
            # Update peak
            if trade_dir == 'LONG':
                peak_price = max(peak_price, high)
                _pnl = (price - entry_price) / TICK
                _pnl_from_low = (low - entry_price) / TICK
            else:
                peak_price = min(peak_price, low) if peak_price > 0 else low
                _pnl = (entry_price - price) / TICK
                _pnl_from_low = (entry_price - high) / TICK

            _peak_pnl = (peak_price - entry_price) / TICK if trade_dir == 'LONG' \
                else (entry_price - peak_price) / TICK

            # Breakeven: once direction confirmed (+5t), move SL to entry
            _be_active = _peak_pnl >= BE_ACT
            _effective_sl = 0 if _be_active else SL  # 0 = breakeven, SL = full

            # SL check (breakeven-aware)
            if _pnl_from_low <= -_effective_sl:
                _exit_pnl = -_effective_sl if _effective_sl > 0 else _pnl_from_low
                _exit_type = 'BE' if _be_active else 'SL'
                trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                               'held': i - entry_bar, 'exit': _exit_type, 'peak': _peak_pnl})
                trade_log.append({
                    'bar': i, 'price': price, 'entry': entry_price,
                    'pnl': _exit_pnl, 'dir': trade_dir, 'exit': _exit_type,
                    'held': i - entry_bar, 'peak': _peak_pnl,
                    'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                    'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                })
                in_trade = False
                continue

            # Trail activation
            if not trail_active and _peak_pnl >= TRAIL_ACT:
                trail_active = True

            # Trail check
            if trail_active:
                if trade_dir == 'LONG':
                    _trail_level = peak_price - TRAIL_DIST * TICK
                    if low <= _trail_level:
                        _exit_pnl = max(0, (_trail_level - entry_price) / TICK)
                        trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                                       'held': i - entry_bar, 'exit': 'TRAIL', 'peak': _peak_pnl})
                        trade_log.append({
                            'bar': i, 'price': price, 'entry': entry_price,
                            'pnl': _exit_pnl, 'dir': trade_dir, 'exit': 'TRAIL',
                            'held': i - entry_bar, 'peak': _peak_pnl,
                            'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                            'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                        })
                        in_trade = False
                        continue
                else:
                    _trail_level = peak_price + TRAIL_DIST * TICK
                    if high >= _trail_level:
                        _exit_pnl = max(0, (entry_price - _trail_level) / TICK)
                        trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                                       'held': i - entry_bar, 'exit': 'TRAIL', 'peak': _peak_pnl})
                        trade_log.append({
                            'bar': i, 'price': price, 'entry': entry_price,
                            'pnl': _exit_pnl, 'dir': trade_dir, 'exit': 'TRAIL',
                            'held': i - entry_bar, 'peak': _peak_pnl,
                            'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                            'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                        })
                        in_trade = False
                        continue

            # Flip: predicted direction changed AND all horizons agree
            if _pred_dir != trade_dir and _confidence > CONF_THRESHOLD and _all_agree:
                _exit_pnl = _pnl
                trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                               'held': i - entry_bar, 'exit': 'FLIP', 'peak': _peak_pnl})
                trade_log.append({
                    'bar': i, 'price': price, 'entry': entry_price,
                    'pnl': _exit_pnl, 'dir': trade_dir, 'exit': 'FLIP',
                    'held': i - entry_bar, 'peak': _peak_pnl,
                    'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                    'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                })
                # Enter opposite direction (fill at 2s after signal)
                _fill = _get_fill_price(timestamps[i])
                in_trade = True
                trade_dir = _pred_dir
                entry_price = _fill if _fill is not None else price
                entry_bar = i
                peak_price = entry_price
                trail_active = False
                continue

        # Entry: all horizons agree + confidence (momentum building optional)
        if not in_trade and _confidence > CONF_THRESHOLD and _all_agree:
            in_trade = True
            trade_dir = _pred_dir
            # Fill at actual price 2s after signal (from 1s data)
            _fill = _get_fill_price(timestamps[i])
            entry_price = _fill if _fill is not None else price
            entry_bar = i
            peak_price = entry_price
            trail_active = False

    # Flush last trade
    if in_trade:
        _pnl = (prices[-1] - entry_price) / TICK if trade_dir == 'LONG' \
            else (entry_price - prices[-1]) / TICK
        trades.append({'bar': len(prices)-1, 'pnl': _pnl, 'dir': trade_dir,
                       'held': len(prices) - 1 - entry_bar, 'exit': 'EOD', 'peak': 0})

    # Results
    total_pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    w = len([t for t in trades if t['pnl'] > 0])
    trading_days = pd.to_datetime(df['timestamp'], unit='s').dt.date.nunique()

    _wr = w / n * 100 if n > 0 else 0
    _per_day = total_pnl * 0.5 / trading_days if trading_days > 0 else 0

    print(f"\n{'='*60}")
    print(f"OOS SINGLE PASS: StatePredictor + Trail SL={SL} Trail={TRAIL_ACT}/{TRAIL_DIST}")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {_wr:.1f}%")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${_per_day:.2f}")
    print(f"  Trading days: {trading_days}")

    # Exit breakdown
    _exits = {}
    for t in trades:
        ex = t['exit']
        if ex not in _exits:
            _exits[ex] = {'n': 0, 'pnl': 0}
        _exits[ex]['n'] += 1
        _exits[ex]['pnl'] += t['pnl']
    print(f"\n  EXIT BREAKDOWN:")
    for ex, v in sorted(_exits.items(), key=lambda x: x[1]['pnl']):
        print(f"    {ex:<10} {v['n']:>5} trades  ${v['pnl']*0.5:>10,.2f}")

    # Save trade log
    _log_path = os.path.join(CHECKPOINT_DIR, 'oos_trade_log.csv')
    if trade_log:
        _keys = trade_log[0].keys()
        with open(_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_keys)
            writer.writeheader()
            writer.writerows(trade_log)
        print(f"\n  Trade log: {_log_path} ({len(trade_log)} trades)")

    # Append to experiment log
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"model=TradeCNN_A_OOS | days={trading_days} | "
             f"trades={n} | WR={_wr:.1f}% | "
             f"PnL=${total_pnl*0.5:,.0f} | $/day=${_per_day:.0f}\n")
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)

    # Cleanup
    del feats, labels, states, df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
