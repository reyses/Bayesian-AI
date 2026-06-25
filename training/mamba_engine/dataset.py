import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_all_available_dates(features_dir: str) -> List[str]:
    """
    Scans the L0 directory within features_dir to find all available trading days.
    Returns a chronologically sorted list of date strings (e.g., '2024_01_02').
    """
    l0_dir = os.path.join(features_dir, "L0")
    if not os.path.exists(l0_dir):
        files = glob.glob(os.path.join(features_dir, "**", "*.parquet"), recursive=True)
    else:
        files = glob.glob(os.path.join(l0_dir, "*.parquet"))
        
    dates = set()
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace(".parquet", "")
        dates.add(date_str)
        
    sorted_dates = sorted(list(dates))
    logger.info(f"Discovered {len(sorted_dates)} total trading days in dataset.")
    return sorted_dates

class BayesianAtlasDataset(Dataset):
    """
    Loads and joins the L0, L1, L2, L3 feature parquets from the Bayesian-AI pipeline
    for a specific day, returning overlapping sequences for Mamba.
    Also computes dynamic Causal RL Truth Labels using the Auto Seeds ZigZag logic.
    """
    def __init__(self, features_dir: str, date_str: str, seq_len: int = 100, atr_mult: float = 4.0, min_bars_5s: int = 36, tick_size: float = 0.25):
        self.features_dir = features_dir
        self.date_str = date_str
        self.seq_len = seq_len
        
        # Fine-tuned ZigZag parameters
        self.atr_mult = atr_mult
        self.min_bars_5s = min_bars_5s
        self.tick_size = tick_size
        
        self.data_tensor, self.labels_tensor = self._load_and_join_day()
        self.num_features = self.data_tensor.shape[1]

    def _compute_atr(self, bars1m: pd.DataFrame, period: int = 14) -> float:
        h = bars1m['high'].values
        l = bars1m['low'].values
        c = bars1m['close'].values
        if len(h) < period + 1:
            return float((h - l).mean()) if len(h) > 0 else 1.0
        prev_c = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
        if len(tr) >= period:
            return float(np.median(tr[-period * 3:]))
        return float(tr.mean())
        
    def _compute_zigzag_labels(self, df: pd.DataFrame, atr_pts: float) -> torch.Tensor:
        """
        Applies the Fine-Tuned ATR ZigZag logic to the day's close prices
        to produce a dense causal RL truth label array.
        """
        closes = df['close'].values
        
        # We need integer timestamps (e.g. Unix seconds) for the neutral zone calculation.
        # Check if the index is a DatetimeIndex or a standard numeric index.
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.astype(np.int64) // 10**9
        else:
            timestamps = df.index.values.astype(np.int64)
            
        labels = np.zeros(len(closes), dtype=np.int64) # 0 = HOLD / NEUTRAL
        
        if len(closes) < 2:
            return torch.tensor(labels)
            
        pivots = [0]
        direction = None
        last_high = closes[0]
        last_low = closes[0]
        last_high_i = 0
        last_low_i = 0
        
        tick = self.tick_size
        min_rev_ticks = max(4, int(round(atr_pts / tick * self.atr_mult)))
        min_bars = self.min_bars_5s

        for i in range(1, len(closes)):
            if closes[i] > last_high:
                last_high = closes[i]
                last_high_i = i
            if closes[i] < last_low:
                last_low = closes[i]
                last_low_i = i

            if direction is None:
                if (last_high - last_low) / tick >= min_rev_ticks:
                    if last_high_i > last_low_i:
                        direction = 'UP'
                        pivots.append(last_low_i)
                    else:
                        direction = 'DOWN'
                        pivots.append(last_high_i)
            elif direction == 'UP':
                drop = (last_high - closes[i]) / tick
                if drop >= min_rev_ticks and (i - last_high_i) >= min_bars:
                    pivots.append(last_high_i)
                    direction = 'DOWN'
                    last_low = closes[last_high_i]
                    last_low_i = last_high_i
            elif direction == 'DOWN':
                rise = (closes[i] - last_low) / tick
                if rise >= min_rev_ticks and (i - last_low_i) >= min_bars:
                    pivots.append(last_low_i)
                    direction = 'UP'
                    last_high = closes[last_low_i]
                    last_high_i = last_low_i
                    
        if pivots[-1] < len(closes) - 1:
            pivots.append(len(closes) - 1)

        # 1. Label the segments between pivots (leg direction)
        for j in range(len(pivots) - 1):
            si = pivots[j]
            ei = pivots[j + 1]
            if ei <= si:
                continue
            
            # 1 = LONG, 2 = SHORT
            segment_direction = 1 if closes[ei] > closes[si] else 2
            labels[si:ei] = segment_direction
            
        # 2. Override with NEUTRAL (0) around ±2 minutes (120s) of any pivot
        NEUTRAL_ZONE_S = 120
        for p_idx in pivots:
            # We don't necessarily want to neutralize the boundaries of the day if they aren't real pivots
            if p_idx == 0 or p_idx == len(closes) - 1:
                continue
                
            p_ts = timestamps[p_idx]
            
            # Search locally to avoid looping through the whole array
            search_start = max(0, p_idx - int(NEUTRAL_ZONE_S/5) - 5)
            search_end = min(len(closes), p_idx + int(NEUTRAL_ZONE_S/5) + 6)
            
            for i in range(search_start, search_end):
                if abs(timestamps[i] - p_ts) <= NEUTRAL_ZONE_S:
                    labels[i] = 0

        return torch.tensor(labels, dtype=torch.long)

    def _load_and_join_day(self) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Loading features for {self.date_str}...")
        
        search_pattern = os.path.join(self.features_dir, "*", f"{self.date_str}.parquet")
        parquet_files = glob.glob(search_pattern)
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquets found for {self.date_str} in {self.features_dir}")
            
        logger.info(f"Found {len(parquet_files)} feature layers. Joining...")
        
        joined_df = None
        for file in parquet_files:
            df = pd.read_parquet(file)
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            if joined_df is None:
                joined_df = df
            else:
                joined_df = joined_df.join(df, how='outer')
        
        # Also load the RAW price data so we can calculate ZigZag labels
        raw_price_path = os.path.join(os.path.dirname(self.features_dir), "5s", f"{self.date_str}.parquet")
        if not os.path.exists(raw_price_path):
            raise FileNotFoundError(f"Missing raw prices needed for labels: {raw_price_path}")
            
        price_df = pd.read_parquet(raw_price_path)
        if 'timestamp' in price_df.columns:
            price_df.set_index('timestamp', inplace=True)
            
        # We need to keep 'close' for labeling, but remove it from the model's feature set
        joined_df = joined_df.join(price_df[['close']], how='left')
        
        joined_df.sort_index(inplace=True)
        joined_df.ffill(inplace=True)
        joined_df.fillna(0, inplace=True)
        
        # Also load the RAW 1m data so we can calculate dynamic ATR
        atlas_root = os.path.dirname(os.path.dirname(self.features_dir)) # DATA/ATLAS
        raw_1m_path = os.path.join(atlas_root, "1m", f"{self.date_str}.parquet")
        
        atr_pts = 30.0 * 0.25 # Default fallback
        if os.path.exists(raw_1m_path):
            try:
                bars1m = pd.read_parquet(raw_1m_path).sort_values('timestamp').reset_index(drop=True)
                atr_pts = self._compute_atr(bars1m)
                logger.info(f"Computed dynamic ATR for {self.date_str}: {atr_pts:.2f} pts")
            except Exception as e:
                logger.warning(f"Failed to compute ATR from {raw_1m_path}: {e}")
                
        # 1. Compute Labels before dropping 'close'
        logger.info("Computing Causal RL ZigZag Labels from Fine-Tuned ATR logic...")
        labels_tensor = self._compute_zigzag_labels(joined_df, atr_pts)
        
        # 2. Drop 'close' to prevent data leakage (we only want the model to see normalized features)
        joined_df.drop(columns=['close'], inplace=True)
        
        # Normalize features
        means = joined_df.mean()
        stds = joined_df.std()
        stds[stds == 0] = 1.0 
        normalized_df = (joined_df - means) / stds
        
        data_tensor = torch.tensor(normalized_df.values, dtype=torch.float32)
        return data_tensor, labels_tensor

    def __len__(self):
        return max(0, len(self.data_tensor) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.data_tensor[idx : idx + self.seq_len]
        
        # Label is the target for the LAST bar in the sequence
        target_label = self.labels_tensor[idx + self.seq_len - 1].item()
        return x_seq, target_label

if __name__ == "__main__":
    base_dir = r"DATA\ATLAS\FEATURES_5s_v2"
    dataset = BayesianAtlasDataset(features_dir=base_dir, date_str="2024_01_02")
    print(f"Dataset ready. Number of sequences: {len(dataset)}")
    print(f"Input Feature Dimension (input_dim): {dataset.num_features}")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    X, y = next(iter(loader))
    print(f"Batch X shape: {X.shape}")
    print(f"Batch y shape: {y.shape}")
    print(f"Sample labels: {y[:10].tolist()}")
