"""VRAM-aware variant for RL training.

Inherits from ForwardPassSystem to guarantee identical parsing and state alignment,
then adds the tensor precomputation needed for vectorized training.
"""
from __future__ import annotations

import os
import glob
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from core_v2.features import assemble_v2_grid
from .state import BarState
from .forward_pass_system import ForwardPassSystem


class VRAMForwardPassSystem(ForwardPassSystem):
    """VRAM-aware ticker that precomputes L0 and Grid tensors for the entire day.
    
    Yields:
        (state: BarState, l0_slice: Tensor, grid_slice: Tensor)
    """

    def __init__(self, day: str, atlas_root: str,
                 features_root: str,
                 labels_csv: str):
        super().__init__(day, atlas_root, features_root, labels_csv)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ts_arr = self._feats['timestamp'].values.astype(np.int64)
        sec_in_day = ts_arr % 86400
        tod_norms = sec_in_day / 86400.0
        dt_series = pd.to_datetime(ts_arr, unit='s', utc=True)
        day_norms = dt_series.weekday.values / 4.0
        
        l0_all = np.column_stack([self._v2_matrix[:, 0], tod_norms, day_norms])
        
        # Use the correct grid assembler
        grid_all = assemble_v2_grid(self._v2_matrix)
        
        self._l0_tensor = torch.nan_to_num(torch.tensor(l0_all, dtype=torch.float32, device=device), nan=0.0)
        self._grid_tensor = torch.nan_to_num(torch.tensor(grid_all, dtype=torch.float32, device=device), nan=0.0)

    def __iter__(self) -> Iterator[Tuple[BarState, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        for state in super().__iter__():
            i = state.bar_idx
            if i >= 59:
                grid_slice = self._grid_tensor[i-59:i+1].permute(1, 0, 2).unsqueeze(0)
                l0_slice = self._l0_tensor[i-59:i+1].unsqueeze(0)
            else:
                grid_slice = None
                l0_slice = None
                
            yield state, l0_slice, grid_slice


class MultiDayVRAMForwardPassSystem:
    """Replays multiple days through VRAMForwardPassSystem."""

    def __init__(self, atlas_root: str,
                 features_root: str,
                 labels_csv: str,
                 days: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        if days is None:
            l0_dir = os.path.join(features_root, 'L0')
            files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
            day_list = [os.path.basename(f).replace('.parquet', '') for f in files]
            if start_date:
                day_list = [d for d in day_list if d.replace('_', '-') >= start_date]
            if end_date:
                day_list = [d for d in day_list if d.replace('_', '-') <= end_date]
            days = day_list
        self._days = list(days)
        self._atlas_root = atlas_root
        self._features_root = features_root
        self._labels_csv = labels_csv
        self._current_day: Optional[str] = None

    @property
    def current_day(self) -> Optional[str]:
        return self._current_day

    def day_count(self) -> int:
        return len(self._days)

    def __iter__(self) -> Iterator[Tuple[BarState, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        for day in self._days:
            self._current_day = day
            try:
                ticker = VRAMForwardPassSystem(day=day, atlas_root=self._atlas_root,
                                               features_root=self._features_root,
                                               labels_csv=self._labels_csv)
            except FileNotFoundError as e:
                import logging
                logging.warning(f"Skipping {day}: {e}")
                continue
            for state, l0_slice, grid_slice in ticker:
                yield state, l0_slice, grid_slice
