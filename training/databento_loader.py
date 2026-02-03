"""
Bayesian-AI - Databento Loader
Loads and normalizes Databento DBN files
"""
import pandas as pd
import os
import databento as db
from config.settings import OPERATIONAL_MODE, RAW_DATA_PATH

class DatabentoLoader:
    @staticmethod
    def load_data(filepath: str, filter_trades: bool = True) -> pd.DataFrame:
        """
        Loads data from a databento DBN file and converts it to a pandas DataFrame
        compatible with the Bayesian AI training pipeline.

        Args:
            filepath (str): Path to the .dbn file.
            filter_trades (bool): If True, filters only for trade events (action='T').

        Returns:
            pd.DataFrame: DataFrame with columns ['timestamp', 'price', 'volume', 'type']
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Load data using databento
            data = db.DBNStore.from_file(filepath)
            df = data.to_df()
        except Exception as e:
            raise ValueError(f"Failed to load databento file: {e}") from e

        # Reset index if it's the timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # Cautious column renaming to avoid duplicates
        rename_map = {}
        used_targets = set()
        
        # Define mappings in order of preference
        potential_maps = [
            ('ts_event', 'timestamp'),
            ('ts_recv', 'timestamp'),
            ('price', 'price'),
            ('close', 'price'),
            ('size', 'volume'),
            ('volume', 'volume'),
            ('action', 'type'),
            ('side', 'type')
        ]

        for source, target in potential_maps:
            if source in df.columns and target not in used_targets:
                rename_map[source] = target
                used_targets.add(target)
        
        df = df.rename(columns=rename_map)

        # Standardize 'type' column
        if 'type' not in df.columns:
            df['type'] = 'trade'
        df['type'] = df['type'].astype(str)

        # Filter for trades if requested
        # We check for 'action' in the original rename_map to know if 'type' came from 'action'
        if filter_trades and 'action' in rename_map:
            df = df[df['type'] == 'T']
        
        # Convert timestamp to float (seconds)
        if 'timestamp' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'].astype('int64') / 1e9
            elif pd.api.types.is_integer_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'] / 1e9


        # Ensure required columns exist
        required_cols = ['timestamp', 'price', 'volume', 'type']
        for col in required_cols:
            if col not in df.columns:
                if col == 'type':
                    df['type'] = 'trade'
                else:
                    raise ValueError(f"Missing required column: {col} in databento file")

        # Preserve OHLC columns if present
        ohlc_cols = []
        for col in ['open', 'high', 'low']:
            if col in df.columns:
                ohlc_cols.append(col)

        return df[required_cols + ohlc_cols]
