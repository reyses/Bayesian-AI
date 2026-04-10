"""
CNN Entry — Pattern Discovery on Profitable Trades.

For each ExNMP tier, trains an encoder on profitable
trade entry 79D. Clusters the embeddings to discover distinct physics patterns.

NOT a classifier. All inputs are profitable (regret-confirmed). The question:
"What KIND of good trade is this?"

Input:  entry 79D (6x13 grid) + 10-bar approach trajectory (10x6x13)
Output: 16D embedding -> K-means clusters -> pattern labels

Approach data loaded directly from FEATURES_79D_5s parquets (no buffer limit).

Usage:
    python nn_v2/cnn_entry.py                    # full run, all tiers
    python nn_v2/cnn_entry.py --tier BASE_NMP    # single tier
    python nn_v2/cnn_entry.py --max-k 8          # max clusters
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features_79d import FEATURE_NAMES_79D, TF_ORDER, N_FEATURES

# Data paths
BLENDED_TRADES = 'nn_v2/output/trades/blended_is.pkl'
REGRET_FILE = 'nn_v2/output/tree/regret_analysis.csv'
FEATURES_DIR = 'DATA/FEATURES_79D_5s'
OUTPUT_DIR = 'nn_v2/output/entry'

# Grid layout (same as cnn_flip)
GRID_H = 6   # TFs
GRID_W = 13  # 10 core + 3 helper
N_CORE = 10
N_HELPER = 3
N_TFS = 6
HELPER_START = N_CORE * N_TFS  # 60

# Approach
APPROACH_BARS = 10  # bars of 79D before the (early) entry

# Encoder
EMBEDDING_DIM = 16

# Tier encoding
TIER_MAP = {
    'CASCADE': 6, 'KILL_SHOT': 5,
    'FADE_CALM': 4, 'FADE_MOMENTUM': 3,
    'RIDE_CALM': 2, 'RIDE_MOMENTUM': 1,
    'RIDE_AGAINST': 0, 'FREIGHT_TRAIN': -1,
    'BASE_NMP': 0, 'MANUAL': 0,
}

# All tradeable tiers for pattern discovery
ALL_TIERS = ['CASCADE', 'KILL_SHOT', 'FADE_CALM', 'FADE_MOMENTUM', 'FADE_AGAINST',
             'RIDE_CALM', 'RIDE_MOMENTUM', 'RIDE_AGAINST', 'FREIGHT_TRAIN',
             'REGIME_FLIP', 'MTF_EXHAUSTION', 'EXHAUSTION_BAR', 'ABSORPTION']

# Profitable threshold (regret best_pnl must exceed this)
MIN_PROFITABLE_PNL = 2.0  # $2 minimum to count as "good trade"


def feat_to_grid(feat_79d):
    """Reshape 79D vector to 6x13 grid (same as cnn_flip)."""
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for tf_idx in range(N_TFS):
        c_start = tf_idx * N_CORE
        grid[tf_idx, :N_CORE] = feat_79d[c_start:c_start + N_CORE]
        h_start = HELPER_START + tf_idx * N_HELPER
        grid[tf_idx, N_CORE:N_CORE + N_HELPER] = feat_79d[h_start:h_start + N_HELPER]
    return grid


def load_feature_file(day: str) -> pd.DataFrame:
    """Load 79D features for a day from disk."""
    path = os.path.join(FEATURES_DIR, f'{day}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_approach_from_features(feat_df: pd.DataFrame, entry_ts: float,
                                early_bars: int, n_bars: int = APPROACH_BARS):
    """Load N bars of 79D before the early entry from feature file.

    Args:
        feat_df: day's feature DataFrame (timestamp + 79 features)
        entry_ts: NMP trigger timestamp
        early_bars: how many bars before NMP trigger the early entry is
        n_bars: how many approach bars to grab before early entry

    Returns:
        list of 79D arrays (oldest first), or empty list if insufficient data
    """
    if feat_df.empty:
        return []

    timestamps = feat_df['timestamp'].values
    feat_cols = [c for c in feat_df.columns if c != 'timestamp']

    # Find NMP trigger bar
    trigger_idx = int(np.searchsorted(timestamps, entry_ts, side='right')) - 1
    if trigger_idx < 0:
        return []

    # Early entry is early_bars before trigger
    early_entry_idx = trigger_idx - early_bars

    # Approach: n_bars before early entry
    approach_start = max(0, early_entry_idx - n_bars)
    approach_end = max(0, early_entry_idx)

    if approach_end <= approach_start:
        return []

    approach = []
    for idx in range(approach_start, approach_end):
        row = feat_df.iloc[idx]
        feat = np.array([row[c] for c in feat_cols], dtype=np.float32)
        approach.append(feat)

    return approach


def build_dataset(trades, regret_df, tier_filter=None, min_pnl=MIN_PROFITABLE_PNL):
    """Build training dataset: entry grids + approach trajectories for profitable trades.

    Args:
        trades: list of trade dicts from blended_is.pkl
        regret_df: DataFrame from regret_analysis.csv
        tier_filter: if set, only include this tier
        min_pnl: minimum regret best_pnl to count as profitable

    Returns:
        entry_grids: (N, 6, 13) entry 79D grids
        approach_grids: (N, APPROACH_BARS, 6, 13) approach trajectory grids
        meta: list of dicts with trade metadata (day, tier, pnl, direction, etc.)
    """
    # Filter profitable trades
    profitable_mask = regret_df['best_pnl'] > min_pnl
    if tier_filter:
        # Get tier from trades (not in regret CSV)
        tier_list = [t.get('entry_tier', 'NMP') for t in trades]
        tier_mask = np.array([t == tier_filter for t in tier_list])
        mask = profitable_mask.values & tier_mask[:len(profitable_mask)]
    else:
        mask = profitable_mask.values

    indices = np.where(mask)[0]
    print(f'  Profitable trades: {len(indices)} / {len(trades)} '
          f'(tier={tier_filter or "ALL"}, min_pnl=${min_pnl})')

    # Load feature files per day (cache)
    feat_cache = {}
    entry_grids = []
    approach_grids = []
    meta = []

    for idx in tqdm(indices, desc='Building dataset', unit='trade'):
        if idx >= len(trades) or idx >= len(regret_df):
            continue

        t = trades[idx]
        r = regret_df.iloc[idx]

        # Entry 79D
        entry_79d = np.array(t.get('entry_79d', []))
        if len(entry_79d) != N_FEATURES:
            continue

        entry_grid = feat_to_grid(entry_79d)

        # Load approach from feature files
        day = t.get('day', '')
        entry_ts = t.get('timestamp', 0)
        early_bars = int(r.get('best_early_bars_before', 0))

        if day not in feat_cache:
            feat_cache[day] = load_feature_file(day)

        approach = get_approach_from_features(
            feat_cache[day], entry_ts, early_bars, APPROACH_BARS)

        # Convert approach to grids
        approach_grid = np.zeros((APPROACH_BARS, GRID_H, GRID_W), dtype=np.float32)
        for i, feat in enumerate(approach):
            if len(feat) >= N_FEATURES:
                approach_grid[i] = feat_to_grid(feat[:N_FEATURES])

        # If we got fewer than APPROACH_BARS, pad with entry (closest known state)
        n_filled = len(approach)
        if n_filled < APPROACH_BARS:
            for i in range(n_filled, APPROACH_BARS):
                approach_grid[i] = approach_grid[max(0, n_filled - 1)]

        entry_grids.append(entry_grid)
        approach_grids.append(approach_grid)
        meta.append({
            'trade_idx': int(idx),
            'day': day,
            'tier': t.get('entry_tier', 'NMP'),
            'direction': t.get('dir', ''),
            'actual_pnl': float(t.get('pnl', 0)),
            'best_pnl': float(r.get('best_pnl', 0)),
            'best_action': r.get('best_action', ''),
            'early_bars': early_bars,
            'timestamp': entry_ts,
        })

    # Free cache
    del feat_cache

    entry_grids = np.array(entry_grids, dtype=np.float32)
    approach_grids = np.array(approach_grids, dtype=np.float32)

    print(f'  Dataset: {len(entry_grids)} samples, '
          f'entry={entry_grids.shape}, approach={approach_grids.shape}')

    return entry_grids, approach_grids, meta


class EntryDataset(Dataset):
    """Dataset for entry pattern encoder."""

    def __init__(self, entry_grids, approach_grids):
        self.entries = torch.FloatTensor(entry_grids)
        self.approaches = torch.FloatTensor(approach_grids)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx], self.approaches[idx]


class EntryEncoder(nn.Module):
    """Autoencoder on entry 79D grid + approach trajectory.

    Encodes entry (1x6x13) + approach (10x6x13) into a 16D embedding.
    Decoder reconstructs the entry grid from the embedding.
    The embedding captures the essence of the setup.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, approach_bars=APPROACH_BARS):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Entry encoder: 1x6x13 -> features
        self.entry_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)),  # -> 64x3x6 = 1152
        )

        # Approach encoder: 10x6x13 -> features (treat time steps as channels)
        self.approach_conv = nn.Sequential(
            nn.Conv2d(approach_bars, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 6)),  # -> 64x3x6 = 1152
        )

        # Combined -> embedding
        self.encoder_fc = nn.Sequential(
            nn.Linear(1152 + 1152, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

        # Decoder: embedding -> entry grid reconstruction
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, GRID_H * GRID_W),
        )

    def encode(self, entry, approach):
        """Encode to embedding."""
        # Entry: (B, 6, 13) -> (B, 1, 6, 13)
        e = entry.unsqueeze(1)
        e_feat = self.entry_conv(e).flatten(1)  # (B, 1152)

        # Approach: (B, 10, 6, 13) -> already (B, C, H, W)
        a_feat = self.approach_conv(approach).flatten(1)  # (B, 1152)

        combined = torch.cat([e_feat, a_feat], dim=1)  # (B, 2304)
        embedding = self.encoder_fc(combined)  # (B, 16)
        return embedding

    def decode(self, embedding):
        """Decode embedding to entry grid."""
        out = self.decoder_fc(embedding)  # (B, 78)
        return out.view(-1, GRID_H, GRID_W)

    def forward(self, entry, approach):
        embedding = self.encode(entry, approach)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction


def train_encoder(entry_grids, approach_grids, epochs=50, lr=1e-3, batch_size=128):
    """Train autoencoder on entry + approach data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # Normalize
    entry_mean = entry_grids.mean(axis=0, keepdims=True)
    entry_std = entry_grids.std(axis=0, keepdims=True).clip(min=1e-8)
    entry_norm = (entry_grids - entry_mean) / entry_std

    # Normalize approach per-feature (across all bars and samples)
    ap_shape = approach_grids.shape  # (N, 10, 6, 13)
    ap_flat = approach_grids.reshape(-1, GRID_H, GRID_W)  # (N*10, 6, 13)
    ap_mean = ap_flat.mean(axis=0, keepdims=True)
    ap_std = ap_flat.std(axis=0, keepdims=True).clip(min=1e-8)
    approach_norm = ((approach_grids.reshape(-1, GRID_H, GRID_W) - ap_mean) / ap_std
                     ).reshape(ap_shape)

    dataset = EntryDataset(entry_norm, approach_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = EntryEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for entry_batch, approach_batch in loader:
            entry_batch = entry_batch.to(device)
            approach_batch = approach_batch.to(device)

            embedding, reconstruction = model(entry_batch, approach_batch)

            # Reconstruction loss: how well can we reconstruct entry from embedding?
            loss = criterion(reconstruction, entry_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}: loss={avg_loss:.6f} (best={best_loss:.6f})')

    model.load_state_dict(best_state)
    model.eval()

    # Compute all embeddings
    all_embeddings = []
    with torch.no_grad():
        for entry_batch, approach_batch in DataLoader(
                dataset, batch_size=256, shuffle=False):
            entry_batch = entry_batch.to(device)
            approach_batch = approach_batch.to(device)
            emb = model.encode(entry_batch, approach_batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f'  Embeddings: {embeddings.shape}')

    return model, embeddings, {
        'entry_mean': entry_mean,
        'entry_std': entry_std,
        'approach_mean': ap_mean,
        'approach_std': ap_std,
    }


def discover_patterns(embeddings, meta, max_k=10, min_k=2):
    """Cluster embeddings to discover distinct setup patterns.

    Uses silhouette score to pick optimal k.
    """
    print(f'\n  Clustering {len(embeddings)} embeddings (k={min_k}..{max_k})')

    # Normalize embeddings for clustering
    scaler = StandardScaler()
    emb_norm = scaler.fit_transform(embeddings)

    best_k = min_k
    best_score = -1
    scores = []

    for k in range(min_k, min(max_k + 1, len(embeddings))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb_norm)
        score = silhouette_score(emb_norm, labels)
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k
        print(f'    k={k}: silhouette={score:.3f}')

    # Final clustering with best k
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km_final.fit_predict(emb_norm)

    print(f'  Best k={best_k} (silhouette={best_score:.3f})')

    return labels, km_final, scaler, scores


def report_patterns(labels, meta, tier_name):
    """Print per-pattern statistics."""
    print(f'\n{"="*60}')
    print(f'PATTERN REPORT: {tier_name}')
    print(f'{"="*60}')

    n_patterns = len(set(labels))
    print(f'  Discovered {n_patterns} patterns from {len(labels)} trades\n')

    print(f'  {"Pattern":<12} {"N":>5} {"WR%":>5} {"AvgPnL":>8} {"BestPnL":>8} '
          f'{"Dir%L":>6} {"EarlyBars":>10}')
    print(f'  {"-"*60}')

    for p in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == p]
        sub = [meta[i] for i in idx]
        n = len(sub)
        wins = sum(1 for m in sub if m['actual_pnl'] > 0)
        wr = wins / n * 100
        avg_pnl = np.mean([m['actual_pnl'] for m in sub])
        avg_best = np.mean([m['best_pnl'] for m in sub])
        pct_long = sum(1 for m in sub if m['direction'] == 'long') / n * 100
        avg_early = np.mean([m['early_bars'] for m in sub])

        label = f'{tier_name}.{p}'
        print(f'  {label:<12} {n:>5} {wr:>5.0f} ${avg_pnl:>7.1f} ${avg_best:>7.1f} '
              f'{pct_long:>5.0f}% {avg_early:>9.1f}')

    # Time of day profile per pattern
    print(f'\n  Time-of-day profile (hour distribution):')
    from datetime import datetime
    for p in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == p]
        hours = []
        for i in idx:
            ts = meta[i].get('timestamp', 0)
            if ts > 0:
                hours.append(datetime.utcfromtimestamp(ts).hour)
        if hours:
            from collections import Counter
            hour_counts = Counter(hours)
            top_hours = hour_counts.most_common(3)
            hour_str = ', '.join(f'{h}h({c})' for h, c in top_hours)
            print(f'    {tier_name}.{p}: {hour_str}')


def save_results(tier_name, model, embeddings, labels, meta, norms,
                 km_model, scaler, scores):
    """Save encoder, embeddings, cluster assignments, and stats."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tier_lower = tier_name.lower()

    # Save model checkpoint
    checkpoint = {
        'model_state': model.state_dict(),
        'embedding_dim': EMBEDDING_DIM,
        'approach_bars': APPROACH_BARS,
        'entry_mean': norms['entry_mean'],
        'entry_std': norms['entry_std'],
        'approach_mean': norms['approach_mean'],
        'approach_std': norms['approach_std'],
        'tier': tier_name,
    }
    torch.save(checkpoint, os.path.join(OUTPUT_DIR, f'encoder_{tier_lower}.pt'))

    # Save cluster model
    cluster_data = {
        'kmeans': km_model,
        'scaler': scaler,
        'n_clusters': km_model.n_clusters,
        'silhouette_scores': scores,
        'tier': tier_name,
    }
    with open(os.path.join(OUTPUT_DIR, f'clusters_{tier_lower}.pkl'), 'wb') as f:
        pickle.dump(cluster_data, f)

    # Save per-trade results
    results = []
    for i, m in enumerate(meta):
        m['pattern'] = int(labels[i])
        m['pattern_label'] = f'{tier_name}.{labels[i]}'
        m['embedding'] = embeddings[i].tolist()
        results.append(m)

    with open(os.path.join(OUTPUT_DIR, f'patterns_{tier_lower}.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # CSV summary (no arrays)
    flat = [{k: v for k, v in m.items() if k != 'embedding'} for m in results]
    pd.DataFrame(flat).to_csv(
        os.path.join(OUTPUT_DIR, f'patterns_{tier_lower}.csv'), index=False)

    print(f'  Saved: {OUTPUT_DIR}/encoder_{tier_lower}.pt')
    print(f'  Saved: {OUTPUT_DIR}/clusters_{tier_lower}.pkl')
    print(f'  Saved: {OUTPUT_DIR}/patterns_{tier_lower}.csv')


def run_tier(tier_name, trades, regret_df, min_k=2, max_k=10, epochs=50):
    """Full pipeline for one tier: build data -> train encoder -> cluster -> report."""
    print(f'\n{"="*60}')
    print(f'TIER: {tier_name}')
    print(f'{"="*60}')

    # Build dataset
    entry_grids, approach_grids, meta = build_dataset(
        trades, regret_df, tier_filter=tier_name)

    if len(entry_grids) < 20:
        print(f'  Too few samples ({len(entry_grids)}) — skipping tier')
        return

    # Train encoder
    print(f'\n  Training encoder ({epochs} epochs)...')
    model, embeddings, norms = train_encoder(
        entry_grids, approach_grids, epochs=epochs)

    # Discover patterns
    labels, km_model, scaler, scores = discover_patterns(
        embeddings, meta, min_k=min_k, max_k=max_k)

    # Report
    report_patterns(labels, meta, tier_name)

    # Save
    save_results(tier_name, model, embeddings, labels, meta, norms,
                 km_model, scaler, scores)


def parse_args():
    p = argparse.ArgumentParser(description='CNN Entry Pattern Discovery')
    p.add_argument('--tier', type=str, default=None,
                   choices=ALL_TIERS,
                   help='Single tier to process (default: all)')
    p.add_argument('--min-k', type=int, default=2,
                   help='Min clusters per tier (default: 2)')
    p.add_argument('--max-k', type=int, default=10,
                   help='Max clusters per tier (default: 10)')
    p.add_argument('--epochs', type=int, default=50,
                   help='Training epochs (default: 50)')
    p.add_argument('--min-pnl', type=float, default=MIN_PROFITABLE_PNL,
                   help=f'Min regret best_pnl for profitable (default: ${MIN_PROFITABLE_PNL})')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'CNN ENTRY — Pattern Discovery')
    print(f'  Trades: {BLENDED_TRADES}')
    print(f'  Regret: {REGRET_FILE}')
    print(f'  Features: {FEATURES_DIR}')
    print(f'  Min profitable PnL: ${args.min_pnl}')

    # Load data
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)
    regret_df = pd.read_csv(REGRET_FILE)
    print(f'  Loaded {len(trades)} trades, {len(regret_df)} regret rows')

    # Align lengths
    n = min(len(trades), len(regret_df))
    trades = trades[:n]
    regret_df = regret_df.iloc[:n]

    tiers = [args.tier] if args.tier else ALL_TIERS

    for tier in tiers:
        run_tier(tier, trades, regret_df, min_k=args.min_k, max_k=args.max_k,
                 epochs=args.epochs)

    # Combined summary
    print(f'\n{"="*60}')
    print(f'ALL PATTERNS DISCOVERED')
    print(f'{"="*60}')
    for tier in tiers:
        tier_lower = tier.lower()
        csv_path = os.path.join(OUTPUT_DIR, f'patterns_{tier_lower}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            n_patterns = df['pattern'].nunique()
            print(f'  {tier}: {len(df)} trades -> {n_patterns} patterns')
            for p in sorted(df['pattern'].unique()):
                sub = df[df['pattern'] == p]
                wr = (sub['actual_pnl'] > 0).mean() * 100
                avg = sub['actual_pnl'].mean()
                print(f'    {tier}.{p}: {len(sub)} trades, WR={wr:.0f}%, avg=${avg:.1f}')


if __name__ == '__main__':
    main()
