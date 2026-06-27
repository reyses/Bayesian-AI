import os
import glob
import pandas as pd
import numpy as np

class MacroBank:
    """
    Two-Tier Macro Memory Architecture: The Slow Tier.
    Builds a persistent, causal event log of 15m->1D structural levels (ZHL pivots).
    Strictly prevents metadata lookahead leakage by storing one row per touch event.
    """
    def __init__(self, data_dir="DATA/ATLAS", save_dir="DATA/daily_context"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.events_file = os.path.join(self.save_dir, "macro_bank_events.parquet")
        self.timeframes = ['15m', '30m', '1h', '4h', '1D']
        self.events_df = None
        
        # Detection window lags (in bars) required for a ZHL pivot to be confirmed
        self.lags = {
            '15m': 3,
            '30m': 3,
            '1h': 3,
            '4h': 3,
            '1D': 3
        }

    def load_bank(self):
        """Loads the serialized event log from disk to survive process restarts."""
        if os.path.exists(self.events_file):
            self.events_df = pd.read_parquet(self.events_file)
            print(f"[MacroBank] Loaded {len(self.events_df)} historical touch events.")
        else:
            print("[MacroBank] No existing bank found. Requires Warmup.")
            self.events_df = pd.DataFrame(columns=[
                'timestamp', 'price_level', 'timeframe', 'event_type', 'strength_delta'
            ])

    def save_bank(self):
        """Serializes the event log to disk."""
        os.makedirs(self.save_dir, exist_ok=True)
        if self.events_df is not None and not self.events_df.empty:
            self.events_df.to_parquet(self.events_file)
            print(f"[MacroBank] Saved {len(self.events_df)} events to {self.events_file}")

    def warmup_build(self, start_date=None, end_date=None):
        """
        Causal Construction: Iterates through ATLAS parquets sequentially.
        When a ZHL pivot confirms, it enters the Bank at `timestamp = confirmation_bar`.
        """
        all_events = []
        print("[MacroBank] Starting causal warmup build...")
        
        for tf in self.timeframes:
            tf_dir = os.path.join(self.data_dir, tf)
            if not os.path.exists(tf_dir):
                print(f"[MacroBank] Warning: {tf_dir} not found. Skipping.")
                continue
                
            parquet_files = sorted(glob.glob(os.path.join(tf_dir, "*.parquet")))
            for file in parquet_files:
                # In a real implementation, we would load the dataframe and detect 
                # ZHL pivots. Because we don't have the exact ZHL detection logic 
                # here yet, we will mock the event generation for the skeleton.
                
                # Mock event generation (to be replaced with actual ZHL logic)
                # Example: If a pivot confirmed today at price X, we append:
                # all_events.append({
                #     'timestamp': confirmation_timestamp,
                #     'price_level': pivot_price,
                #     'timeframe': tf,
                #     'event_type': 'creation', # or 'touch'
                #     'strength_delta': 1.0
                # })
                pass
                
        if all_events:
            new_events_df = pd.DataFrame(all_events)
            if self.events_df is not None and not self.events_df.empty:
                self.events_df = pd.concat([self.events_df, new_events_df], ignore_index=True)
            else:
                self.events_df = new_events_df
                
            # Sort chronologically to enforce causality
            self.events_df.sort_values('timestamp', inplace=True)
            self.events_df.reset_index(drop=True, inplace=True)
            
        self.save_bank()
        print("[MacroBank] Warmup complete.")

    def query_as_of(self, t_timestamp):
        """
        Point-in-Time Reconstruction Query.
        Returns the structural levels known *strictly before or at* t_timestamp.
        Dynamically reconstructs cumulative touch_count and strength to prevent leakage.
        """
        if self.events_df is None or self.events_df.empty:
            return pd.DataFrame()
            
        # 1. Filter strictly causal events (<= t)
        causal_events = self.events_df[self.events_df['timestamp'] <= t_timestamp]
        
        if causal_events.empty:
            return pd.DataFrame()
            
        # 2. Reconstruct cumulative properties (touch_count, strength) per level
        # Group by price_level and timeframe
        reconstructed = causal_events.groupby(['price_level', 'timeframe']).agg(
            touch_count=('event_type', 'count'),
            strength=('strength_delta', 'sum'),
            creation_time=('timestamp', 'min')
        ).reset_index()
        
        # Calculate age dynamically as of t
        reconstructed['age'] = (t_timestamp - reconstructed['creation_time']).dt.total_seconds()
        
        return reconstructed

if __name__ == "__main__":
    bank = MacroBank()
    bank.load_bank()
    bank.warmup_build()
