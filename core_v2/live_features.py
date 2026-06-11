from __future__ import annotations

import os
import glob
import logging
import time

import numpy as np
import pandas as pd

from core_v2.statistical_field_engine import StatisticalFieldEngine
from core_v2.features import TF_ORDER, TF_SECONDS, FEATURE_NAMES

logger = logging.getLogger(__name__)

class LiveFeatureEngine:
    """
    Clean, zero-logic-duplication buffer for live feature generation.
    It maintains a rolling DataFrame of 5s bars and passes them directly
    into the StatisticalFieldEngine to guarantee 100% offline parity.
    """
    
    def __init__(self, atlas_root: str, v2_only: bool = True, max_days: int = 25):
        self._atlas_root = atlas_root
        self._v2_only = v2_only
        self._max_days = max_days
        self._sfe = StatisticalFieldEngine()
        
        # Stores the raw 5s bars as a DataFrame
        self._df_5s = pd.DataFrame()
        
        # Exposed for engine_v2.py diagnostics
        self._bars = {'5s': self._df_5s}
        
        # Internal staging for incoming bars before concatenating
        self._new_bars = []
        
    def load_history(self, exclude_day: str = None) -> dict:
        """Loads the trailing N days of 5s bars from ATLAS_NT8 to prime the SFE windows."""
        path_5s = os.path.join(self._atlas_root, '5s')
        if not os.path.exists(path_5s):
            logger.warning(f"No 5s history found at {path_5s}")
            return {'5s': 0}
            
        files = sorted(glob.glob(os.path.join(path_5s, '*.parquet')))
        
        # Filter exclude_day
        if exclude_day:
            files = [f for f in files if exclude_day not in f]
            
        # Take the last max_days
        files = files[-self._max_days:]
        
        if not files:
            return {'5s': 0}
            
        dfs = [pd.read_parquet(f) for f in files]
        self._df_5s = pd.concat(dfs, ignore_index=True)
        self._df_5s = self._df_5s.drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)
        
        # Set datetime index for fast resampling
        self._df_5s['dt'] = pd.to_datetime(self._df_5s['timestamp'], unit='s')
        self._df_5s.set_index('dt', inplace=True)
        
        self._bars['5s'] = self._df_5s
        
        # Pre-build 1m history for ATR priming
        if len(self._df_5s) > 0:
            df_1m = self._df_5s.resample('60s').agg({
                'timestamp': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index(drop=True)
            self._bars['1m'] = df_1m
            
        return {'5s': len(self._df_5s)}
        
    def on_bar(self, bar: dict):
        """Buffer incoming live 5s bars."""
        self._new_bars.append({
            'timestamp': bar['timestamp'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume'],
            'dt': pd.to_datetime(bar['timestamp'], unit='s')
        })
        
    def _flush_new_bars(self):
        """Concatenate buffered bars into the main 5s dataframe."""
        if not self._new_bars:
            return
            
        new_df = pd.DataFrame(self._new_bars)
        new_df.set_index('dt', inplace=True)
        
        # Append and drop duplicates
        self._df_5s = pd.concat([self._df_5s, new_df])
        self._df_5s = self._df_5s[~self._df_5s.index.duplicated(keep='last')]
        
        self._bars['5s'] = self._df_5s
        self._new_bars = []
        
        # Trim if it gets way too large (e.g. > 30 days) to prevent memory leak
        # 30 days * 24 * 60 * 12 = 518,400 bars
        max_bars = 30 * 24 * 60 * 12
        if len(self._df_5s) > max_bars:
            self._df_5s = self._df_5s.iloc[-max_bars:]
            self._bars['5s'] = self._df_5s

    def get_v2_vector(self, ts: int) -> np.ndarray:
        """
        Dynamically resamples the 5s dataframe into all timeframes, passes them
        to the StatisticalFieldEngine, and extracts the final row into a 1D vector.
        """
        self._flush_new_bars()
        
        if len(self._df_5s) == 0:
            return None
            
        vector_dict = {}
        
        # 1. L0 (Global)
        l0 = self._sfe.compute_L0(self._df_5s)
        vector_dict['L0_time_of_day'] = l0['L0_time_of_day'].iloc[-1]
        
        # 2. Per-TF Features
        for tf in TF_ORDER:
            tf_secs = TF_SECONDS[tf]
            
            # Resample 5s up to target TF
            if tf == '5s':
                tf_df = self._df_5s.reset_index(drop=True)
            else:
                tf_df = self._df_5s.resample(f'{tf_secs}s').agg({
                    'timestamp': 'first',
                    'open': 'first', 
                    'high': 'max', 
                    'low': 'min', 
                    'close': 'last', 
                    'volume': 'sum'
                }).dropna().reset_index(drop=True)
                
            if len(tf_df) == 0:
                continue
                
            # Compute layers
            l1 = self._sfe.compute_L1(tf_df, tf)
            l2 = self._sfe.compute_L2(tf_df, tf)
            l3 = self._sfe.compute_L3(tf_df, tf)
            
            # Extract last row
            for col in l1.columns: vector_dict[col] = l1[col].iloc[-1]
            for col in l2.columns: vector_dict[col] = l2[col].iloc[-1]
            for col in l3.columns: vector_dict[col] = l3[col].iloc[-1]
            
        # Build strict ordered 1D array
        try:
            return np.array([vector_dict.get(f, np.nan) for f in FEATURE_NAMES], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to build vector: {e}")
            return None
            
    # Alias for legacy compatibility in engine_v2.py
    def _compute_features(self, ts: int) -> np.ndarray:
        return self.get_v2_vector(ts)
            
    # Stubs for deprecated incremental load functions called by engine_v2.py
    def load_velocities(self, cp: dict):
        pass
        
    def load_accumulators(self, cp: dict):
        pass
        
    @property
    def prev_velocities(self): return {}
    
    @property
    def _accumulators(self): return {}
    
    @property
    def bar_counts(self): return {'5s': len(self._df_5s)}
    
    @property
    def _last_loaded_ts(self): return 0
    
    @property
    def _day_ends(self): return {}
