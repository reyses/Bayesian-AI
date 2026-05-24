"""Direction-confidence classifier strategy.

Per 2026-05-16 forward-pass validation: V2 entry features -> LR with AUC 0.864
IS (88% acc at 40% coverage). The selector signal at threshold T:
    if max(P(LONG), 1-P(LONG)) > T, fire in predicted direction.

Fires on 1m closes (not every 5s bar — too many overlapping signals). The
engine's flat-only entry rule and exit-suite handle position management.

Loads a pre-fit pickle from `training_iso_v2/output/direction_clf.pkl`
(produced by `tools/fit_direction_classifier.py`).

The strategy is OOS-honest: model is trained on full IS once at startup,
then applied per-bar without any refit during the run.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy


DEFAULT_PKL = 'training_iso_v2/output/direction_clf.pkl'
DEFAULT_THRESHOLD = 0.65   # confidence to fire (P or 1-P > threshold)
DEFAULT_FIRE_CADENCE = '1m'  # '1m' | '5m' | '5s' | '15s'


class DirectionClassifierStrategy(Strategy):
    """Fires direction signals when V2-feature LR confidence exceeds threshold."""

    name = 'DIRECTION_CLF'

    def __init__(self, pkl_path: str = DEFAULT_PKL,
                 threshold: float = DEFAULT_THRESHOLD,
                 fire_cadence: str = DEFAULT_FIRE_CADENCE):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f'Direction classifier pickle not found: {pkl_path}. '
                f'Run `python tools/fit_direction_classifier.py` first.'
            )
        with open(pkl_path, 'rb') as f:
            payload = pickle.load(f)
        self.scaler = payload['scaler']
        self.clf = payload['clf']
        self.v2_cols = list(payload['v2_cols'])
        self.threshold = float(threshold)
        self.fire_cadence = fire_cadence
        # Pre-extract scaler params for fast per-bar scoring
        self._mean = self.scaler.mean_.astype(np.float32)
        self._scale = self.scaler.scale_.astype(np.float32)
        # Pre-extract LR weights for fast predict_proba on a single vec
        self._coef = self.clf.coef_[0].astype(np.float32)
        self._intercept = float(self.clf.intercept_[0])

    def _ready(self, state: BarState) -> bool:
        return {
            '5s': True,
            '15s': state.bar_idx % 3 == 0,
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_cadence, False)

    def _p_long(self, state: BarState) -> float:
        """Single-bar P(LONG) using pre-cached scaler + LR weights (fast)."""
        v2 = state.v2
        x = np.empty(len(self.v2_cols), dtype=np.float32)
        for i, c in enumerate(self.v2_cols):
            v = v2.get(c, 0.0)
            x[i] = v if v == v else 0.0   # NaN guard
        xs = (x - self._mean) / np.where(self._scale > 1e-8, self._scale, 1.0)
        z = float(np.dot(xs, self._coef) + self._intercept)
        # Sigmoid
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        ez = np.exp(z)
        return ez / (1.0 + ez)

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        p = self._p_long(state)
        if p >= self.threshold:
            return EntrySignal(direction='long', tier=self.name,
                               extras={'p_long': p})
        if (1.0 - p) >= self.threshold:
            return EntrySignal(direction='short', tier=self.name,
                               extras={'p_long': p})
        return None
