import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class BayesianAtlasDataset(Dataset):
    """
    Loads and joins the L0, L1, L2, L3 feature parquets from the Bayesian-AI pipeline
    for a specific day, returning overlapping sequences for Mamba.
    """
    def __init__(self, features_dir: str, date_str: str, seq_len: int = 100):
        self.features_dir = features_dir
        self.date_str = date_str
        self.seq_len = seq_len
        
        self.data_tensor = self._load_and_join_day()
        self.num_features = self.data_tensor.shape[1]
        
    def _load_and_join_day(self) -> torch.Tensor:
        """
        Finds all parquets for `date_str` across L0, L1, L2, L3 layers, 
        joins them on timestamp, and forward-fills NaNs.
        """
        logger.info(f"Loading features for {self.date_str}...")
        
        # Search all subdirectories (L0, L1_1m, L2_5m, etc.) for the date parquet
        search_pattern = os.path.join(self.features_dir, "*", f"{self.date_str}.parquet")
        parquet_files = glob.glob(search_pattern)
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquets found for {self.date_str} in {self.features_dir}")
            
        logger.info(f"Found {len(parquet_files)} feature layers. Joining...")
        
        joined_df = None
        
        for file in parquet_files:
            df = pd.read_parquet(file)
            
            # Ensure timestamp is the index for joining
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            if joined_df is None:
                joined_df = df
            else:
                joined_df = joined_df.join(df, how='outer')
        
        # Sort by timestamp to ensure chronological order
        joined_df.sort_index(inplace=True)
        
        # Forward fill any NaNs (due to different TF granularities or indicator warmup)
        joined_df.ffill(inplace=True)
        # Fill any remaining leading NaNs with 0
        joined_df.fillna(0, inplace=True)
        
        logger.info(f"Joined dataframe shape: {joined_df.shape}")
        
        # Normalize features (Z-score scaling for neural networks)
        # We do a simple intra-day normalization here for the smoke test
        means = joined_df.mean()
        stds = joined_df.std()
        # Prevent division by zero
        stds[stds == 0] = 1.0 
        normalized_df = (joined_df - means) / stds
        
        return torch.tensor(normalized_df.values, dtype=torch.float32)

    def __len__(self):
        # We can extract (total_ticks - seq_len) overlapping sequences
        return max(0, len(self.data_tensor) - self.seq_len)

    def __getitem__(self, idx):
        # Return sequence (Seq_Len, Features)
        x_seq = self.data_tensor[idx : idx + self.seq_len]
        
        # For the smoke test, we simulate an action label (0-3) for the end of the sequence
        # In production, this comes from the causal RL truth labels.
        dummy_action_label = torch.randint(0, 4, (1,)).item()
        
        return x_seq, dummy_action_label

if __name__ == "__main__":
    # Smoke Test Loader
    base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS\FEATURES_5s_v2"
    dataset = BayesianAtlasDataset(features_dir=base_dir, date_str="2024_01_02")
    print(f"Dataset ready. Number of sequences: {len(dataset)}")
    print(f"Input Feature Dimension (input_dim): {dataset.num_features}")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    X, y = next(iter(loader))
    print(f"Batch X shape: {X.shape}")
    print(f"Batch y shape: {y.shape}")
