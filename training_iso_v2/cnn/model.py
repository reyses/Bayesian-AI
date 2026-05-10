"""V2-native direction CNN.

Input shape:
    grid    : (B, 1, 8, 23)  — 8 V2 TFs × 23 layered features per TF
    tod     : (B, 1)         — L0 time_of_day scalar
    regime  : (B,)           — int regime code (0..N_REGIMES-1), embedded

Output:
    logits  : (B, 3) — softmax over {SHORT=0, FLAT=1, LONG=2}

Why this shape:
- 8×23 mirrors the canonical V2 feature block per `core_v2.features`.
- L0 fed at the FC head, NOT into the per-TF grid (Principle 6 of v2 spec —
  no scale-mixing in the conv input).
- Regime as a 4-dim embedding side input. The chord/triplet EDA established
  that regime carries directional sign, so feeding it directly to the head
  is cheaper than asking conv to infer it from raw features.

EDA-driven decisions:
- 3-class output (not 2): the FLAT class is necessary because most bars have
  no edge and forcing LONG/SHORT picks creates a coin-flip overlay that hurts.
- BatchNorm2d inside the conv stack handles per-feature scale differences,
  consistent with `train_pivot_cnn_v2.py` (Principle 6 — no z-score
  pre-normalization).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core_v2.features import FEATURE_NAMES
from training_iso_v2.state import REGIME_VOCAB
from training_iso_v2.v2_cols import TF_ORDER, per_tf_block, L0_TIME_OF_DAY


# Grid layout
GRID_H = len(TF_ORDER)              # 8
GRID_W = 23                            # L1(6) + L2(9) + L3(8)
N_REGIMES = len(REGIME_VOCAB)       # 7
REGIME_EMBED = 4
N_CLASSES = 3                          # SHORT, FLAT, LONG

# Index map: full FEATURE_NAMES list → per-TF grid positions
# Built once at import time. Used to slice a 185D vector → (8, 23) grid + tod.
def _build_index_map():
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    grid_idx = []
    for tf in TF_ORDER:
        block = per_tf_block(tf)
        if len(block) != GRID_W:
            raise RuntimeError(f'per_tf_block({tf}) returned {len(block)}, expected {GRID_W}')
        for col in block:
            if col not in name_to_idx:
                raise RuntimeError(f'V2 col not found: {col}')
            grid_idx.append(name_to_idx[col])
    if len(grid_idx) != GRID_H * GRID_W:
        raise RuntimeError(f'Expected {GRID_H*GRID_W} grid indices, got {len(grid_idx)}')
    return grid_idx, name_to_idx[L0_TIME_OF_DAY]


GRID_FLAT_IDX, L0_IDX = _build_index_map()


def v2_to_grid(v2_vector):
    """Reshape 185D V2 vector → ((8, 23) grid, tod scalar)."""
    import numpy as np
    arr = np.asarray(v2_vector, dtype=np.float32)
    grid = arr[GRID_FLAT_IDX].reshape(GRID_H, GRID_W)
    tod = float(arr[L0_IDX])
    return grid, tod


class V2DirectionCNN(nn.Module):
    """V2-native direction predictor."""

    def __init__(self, n_classes: int = N_CLASSES,
                 n_regimes: int = N_REGIMES,
                 regime_embed: int = REGIME_EMBED):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # → (64, 4, 8)
        )
        conv_flat = 64 * 4 * 8

        self.regime_embed = nn.Embedding(n_regimes, regime_embed)
        head_in = conv_flat + regime_embed + 1  # +1 for tod
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, grid: torch.Tensor, tod: torch.Tensor,
                regime: torch.Tensor) -> torch.Tensor:
        # grid : (B, 1, 8, 23)   tod : (B, 1)   regime : (B,)
        c = self.conv(grid).view(grid.size(0), -1)
        r = self.regime_embed(regime)            # (B, 4)
        x = torch.cat([c, r, tod], dim=1)
        return self.head(x)
