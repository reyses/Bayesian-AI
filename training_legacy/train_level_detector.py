"""
Train a CNN to filter auto-generated level candidates.

The auto-detector (stddev + harmonics + wick clustering) generates ~10-17
candidates per week. The human draws 4-15 levels per week. The CNN learns
which candidates to keep based on the price action around each candidate.

Input: 10-bar lookback window of 1h OHLCV around each candidate price
Label: 1 if a human level exists within 50 points, 0 otherwise
Output: P(keep) per candidate

Usage:
  python -m training.train_level_detector
"""
import gc
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

ATLAS_ROOT = 'DATA/ATLAS'
TICK = 0.25
MATCH_THRESHOLD = 50  # points — human level within this = match


def compute_candidates(df_week):
    """Generate level candidates using stddev + harmonics + wick clustering.
    Returns list of candidate prices with features.
    """
    highs = df_week['high'].values
    lows = df_week['low'].values
    closes = df_week['close'].values
    opens = df_week['open'].values
    n = len(closes)

    week_high = highs.max()
    week_low = lows.min()
    full_range = week_high - week_low
    if full_range < 10 or n < 5:
        return []

    mean = closes.mean()
    std = closes.std()
    mid = (week_high + week_low) / 2

    candidates = {}

    def add(price, source):
        snapped = round(price / TICK) * TICK
        if snapped not in candidates:
            candidates[snapped] = {'sources': [], 'touches': 0}
        candidates[snapped]['sources'].append(source)

    # StdDev bands
    for mult, label in [(2, '2sd'), (1, '1sd'), (0, 'mean'), (-1, '-1sd'), (-2, '-2sd')]:
        add(mean + mult * std, label)

    # Harmonics
    for h in range(2, 7):
        step = full_range / h
        for i in range(1, h):
            add(week_low + step * i, f'H{h}')

    # Bounds
    add(week_high, 'high')
    add(week_low, 'low')

    # Count touches for each candidate
    for price in candidates:
        touches = sum(1 for i in range(n) if lows[i] <= price <= highs[i])
        candidates[price]['touches'] = touches

    # Build feature vector per candidate
    result = []
    for price, info in candidates.items():
        if price < week_low - 100 or price > week_high + 100:
            continue

        # Features for this candidate:
        feat = np.zeros(10, dtype=np.float32)
        feat[0] = (price - mean) / (std + 1e-8)            # z-score of level
        feat[1] = (price - week_low) / (full_range + 1e-8)  # position in range (0-1)
        feat[2] = info['touches'] / max(1, n)                # touch rate
        feat[3] = len(info['sources'])                        # number of source layers
        feat[4] = 1.0 if 'high' in info['sources'] or 'low' in info['sources'] else 0.0  # is boundary
        feat[5] = 1.0 if any('H' in s for s in info['sources']) else 0.0  # has harmonic
        feat[6] = 1.0 if any('sd' in s for s in info['sources']) else 0.0  # has stddev
        feat[7] = std / (full_range + 1e-8)                  # volatility ratio

        # Price action around this level: how many bars reversed here?
        reversals = 0
        for i in range(1, n - 1):
            # Bar reversed at this level = wick touched but close moved away
            if lows[i] <= price <= highs[i]:
                if (closes[i] > price and closes[i - 1] < price) or \
                   (closes[i] < price and closes[i - 1] > price):
                    reversals += 1
        feat[8] = reversals / max(1, n)

        # Small bar density near this level
        body_sizes = np.abs(closes - opens)
        median_body = np.median(body_sizes)
        near_mask = np.abs(closes - price) < 50
        small_near = np.sum(near_mask & (body_sizes < median_body * 0.5))
        feat[9] = small_near / max(1, n)

        result.append({
            'price': price,
            'features': feat,
            'score': len(info['sources']),
            'touches': info['touches'],
        })

    return result


def build_dataset():
    """Build training dataset from all weeks with human levels."""
    human_files = sorted(glob.glob('DATA/levels/levels_*.json'))
    files_1h = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1h', '*.parquet')))
    df_1h = pd.concat([pd.read_parquet(f) for f in files_1h], ignore_index=True)
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

    all_features = []
    all_labels = []

    for hf in tqdm(human_files, desc="Building dataset"):
        with open(hf) as f:
            human = json.load(f)
        human_prices = [l['price'] for l in human.get('levels', [])]
        if len(human_prices) < 2:
            continue

        date_str = human['date']
        dt = pd.Timestamp(date_str)
        start_ts = dt.timestamp()
        end_ts = (dt + pd.Timedelta(days=7)).timestamp()
        df_week = df_1h[(df_1h['timestamp'] >= start_ts) & (df_1h['timestamp'] < end_ts)]

        if len(df_week) < 5:
            continue

        candidates = compute_candidates(df_week)

        for cand in candidates:
            # Label: is there a human level within MATCH_THRESHOLD?
            is_match = any(abs(cand['price'] - hp) < MATCH_THRESHOLD for hp in human_prices)
            all_features.append(cand['features'])
            all_labels.append(1.0 if is_match else 0.0)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    print(f"Dataset: {len(X)} candidates from {len(human_files)} weeks")
    print(f"  Positive (human match): {y.sum():.0f} ({y.mean()*100:.1f}%)")
    print(f"  Negative (no match):    {(1-y).sum():.0f} ({(1-y).mean()*100:.1f}%)")

    return X, y


class LevelFilterNet(nn.Module):
    """Small MLP that predicts P(keep) for a level candidate."""

    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LevelFilterNet] {_total} params | input: {input_dim}D")

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(X, y, n_epochs=200, val_split=0.2):
    """Train the level filter with class-balanced BCE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split train/val
    n = len(X)
    n_val = int(n * val_split)
    perm = np.random.RandomState(42).permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, y_train = torch.FloatTensor(X[train_idx]).to(device), torch.FloatTensor(y[train_idx]).to(device)
    X_val, y_val = torch.FloatTensor(X[val_idx]).to(device), torch.FloatTensor(y[val_idx]).to(device)

    # Class balance
    pos_weight = (1 - y[train_idx]).sum() / max(y[train_idx].sum(), 1)

    model = LevelFilterNet(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss(reduction='none')

    best_val_acc = 0
    best_model_state = None

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
        model.train()
        pred = model(X_train)
        weights = torch.where(y_train > 0.5, pos_weight, 1.0)
        loss = (weights * bce(pred, y_train)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_class = (val_pred > 0.5).float()
                val_acc = (val_class == y_val).float().mean().item() * 100

                # Precision/recall for positive class
                tp = ((val_class == 1) & (y_val == 1)).sum().item()
                fp = ((val_class == 1) & (y_val == 0)).sum().item()
                fn = ((val_class == 0) & (y_val == 1)).sum().item()
                prec = tp / max(tp + fp, 1) * 100
                recall = tp / max(tp + fn, 1) * 100

            tqdm.write(f"  E{epoch}: loss={loss.item():.4f} val_acc={val_acc:.1f}% "
                       f"prec={prec:.1f}% recall={recall:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final validation report
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_class = (val_pred > 0.5).cpu().numpy()
        y_val_np = y_val.cpu().numpy()

    print(f"\nFinal Validation:")
    print(classification_report(y_val_np, val_class, target_names=['Drop', 'Keep']))

    return model


def main():
    print("Building level filter training data...")
    X, y = build_dataset()

    print("\nTraining LevelFilterNet...")
    model = train(X, y)

    # Save
    ckpt_dir = 'checkpoints/level_filter'
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': X.shape[1],
        'n_samples': len(X),
        'pos_rate': float(y.mean()),
    }, os.path.join(ckpt_dir, 'best_model.pt'))
    print(f"\nSaved: {ckpt_dir}/best_model.pt")

    # Test: run on a sample week
    print("\n--- Sample prediction ---")
    with open('DATA/levels/levels_2025-01-06.json') as f:
        human = json.load(f)
    human_prices = [l['price'] for l in human['levels']]

    files_1h = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1h', '*.parquet')))
    df_1h = pd.concat([pd.read_parquet(f) for f in files_1h[:2]], ignore_index=True)
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
    dt = pd.Timestamp('2025-01-06')
    df_week = df_1h[(df_1h['timestamp'] >= dt.timestamp()) &
                     (df_1h['timestamp'] < (dt + pd.Timedelta(days=7)).timestamp())]

    candidates = compute_candidates(df_week)
    device = next(model.parameters()).device
    model.eval()

    print(f"  Candidates: {len(candidates)}")
    print(f"  Human levels: {human_prices}")
    print(f"  {'Price':>10} {'P(keep)':>8} {'Score':>6} {'Human?':>7} {'Pred':>5}")
    for c in sorted(candidates, key=lambda x: x['price'], reverse=True):
        feat_t = torch.FloatTensor(c['features']).unsqueeze(0).to(device)
        with torch.no_grad():
            p_keep = model(feat_t).item()
        is_human = any(abs(c['price'] - hp) < 50 for hp in human_prices)
        pred = 'KEEP' if p_keep > 0.5 else 'drop'
        print(f"  {c['price']:>10.2f} {p_keep:>7.2f} {c['score']:>6} "
              f"{'YES' if is_human else '':>7} {pred:>5}")


if __name__ == '__main__':
    main()
