import pandas as pd
import os
import databento as db

class DatabentoLoader:
    @staticmethod
    def load_data(filepath: str, filter_trades: bool = True) -> pd.DataFrame:
        """
        Loads data from a databento DBN file and converts it to a pandas DataFrame
        compatible with the ProjectX training pipeline.

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

        # Rename columns to match what TrainingOrchestrator expects
        # Mappings based on standard Databento schemas
        column_map = {}

        # Timestamp mapping
        if 'ts_event' in df.columns:
            column_map['ts_event'] = 'timestamp'
        elif 'ts_recv' in df.columns:
            column_map['ts_recv'] = 'timestamp'

        # Price mapping
        if 'price' in df.columns:
            column_map['price'] = 'price' # Redundant but clear

        # Volume mapping
        if 'size' in df.columns:
            column_map['size'] = 'volume'
        elif 'volume' in df.columns:
            column_map['volume'] = 'volume'

        # Type/Action mapping
        if 'action' in df.columns:
            column_map['action'] = 'type'
        elif 'side' in df.columns:
            # Side is not exactly type, but close enough if action missing
            column_map['side'] = 'type'

        df = df.rename(columns=column_map)

        # Standardize 'type' column
        if 'type' not in df.columns:
            df['type'] = 'trade'

        # Filter for trades if requested and 'type' column came from 'action'
        # Databento action: 'T' for trade.
        # Check if 'action' was mapped to 'type'
        action_mapped = 'action' in column_map and column_map['action'] == 'type'

        if filter_trades and action_mapped:
             # Assuming 'type' column now holds the action values
             # We need to check what 'action' values look like. usually chars 'T', 'A', 'C' or enums.
             # In the dataframe from to_df(), it usually converts enums to chars or keeps them.
             # We'll treat as string.
             df = df[df['type'].astype(str) == 'T']

        # Convert timestamp to float (seconds)
        if 'timestamp' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'].astype('int64') / 1e9
            elif pd.api.types.is_integer_dtype(df['timestamp']):
                # Assuming nanoseconds if integer and large enough, but databento usually returns datetime64[ns]
                # If it's unix timestamp in ns
                df['timestamp'] = df['timestamp'] / 1e9

        # Ensure required columns exist
        required_cols = ['timestamp', 'price', 'volume', 'type']
        for col in required_cols:
            if col not in df.columns:
                if col == 'type':
                    df['type'] = 'trade'
                else:
                    raise ValueError(f"Missing required column: {col} in databento file")

        return df[required_cols]
