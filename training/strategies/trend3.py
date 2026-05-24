"""Trend-3 strategy — fire trades based on the 3-class direction classifier
that predicts "what direction are we IN right now" (LONG / SHORT / NEUTRAL).

Per user 2026-05-17: the classifier paints color at every bar showing the
ML's deduced trend direction. This strategy uses those predictions to enter:
    P(LONG)  > T_long  AND  P(LONG)  > P(SHORT)   → fire LONG
    P(SHORT) > T_short AND  P(SHORT) > P(LONG)    → fire SHORT
Both conditions also require directional_strength > T_strength where
    directional_strength = max(P_long, P_short) - P_neutral

Exit logic is delegated to the engine's exit suite (TP / SL / TimeStop /
optional direction-flip — caller chooses).

Predictions are pre-cached for speed: pass `trend3_cache=<parquet path>` at
init. Falls back to live model inference if no cache provided.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy


DEFAULT_CACHE = None
DEFAULT_MODEL_PKL = 'training_iso_v2/output/trend3_clf_gbm.pkl'
DEFAULT_T = 0.50
DEFAULT_T_STRENGTH = 0.0       # require any directional advantage over NEUTRAL
DEFAULT_FIRE_CADENCE = '1m'


class Trend3Strategy(Strategy):
    """3-class trend classifier strategy. Fires whenever the directional
    prediction crosses a confidence threshold."""

    name = 'TREND3'

    def __init__(self,
                 trend3_cache: str = DEFAULT_CACHE,
                 model_pkl: str = DEFAULT_MODEL_PKL,
                 t_long: float = DEFAULT_T,
                 t_short: float = DEFAULT_T,
                 t_strength: float = DEFAULT_T_STRENGTH,
                 fire_cadence: str = DEFAULT_FIRE_CADENCE):
        self.t_long = float(t_long)
        self.t_short = float(t_short)
        self.t_strength = float(t_strength)
        self.fire_cadence = fire_cadence

        self._cache_lookup = None
        self._clf = None
        self._v2_cols = None
        self._classes = None
        self._scaler = None
        self._long_idx = self._short_idx = self._neut_idx = None

        if trend3_cache and os.path.exists(trend3_cache):
            import pandas as pd
            df = pd.read_parquet(trend3_cache)
            self._cache_lookup = {
                int(ts): (float(pl), float(ps), float(pn))
                for ts, pl, ps, pn in zip(
                    df['timestamp'], df['p_long'], df['p_short'], df['p_neutral']
                )
            }
        else:
            # Live inference fallback
            with open(model_pkl, 'rb') as f:
                pl = pickle.load(f)
            self._clf = pl['clf']
            self._v2_cols = list(pl['v2_cols'])
            self._classes = list(pl['classes'])
            self._scaler = pl.get('scaler')
            self._long_idx  = self._classes.index('LONG')
            self._short_idx = self._classes.index('SHORT')
            self._neut_idx  = self._classes.index('NEUTRAL')

    def _ready(self, state: BarState) -> bool:
        return {
            '5s': True,
            '15s': state.bar_idx % 3 == 0,
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_cadence, False)

    def _predict(self, state: BarState):
        """Return (p_long, p_short, p_neutral) for this bar."""
        if self._cache_lookup is not None:
            return self._cache_lookup.get(int(state.timestamp), (0.0, 0.0, 1.0))
        # Live inference path
        v2 = state.v2
        x = np.empty(len(self._v2_cols), dtype=np.float32)
        for i, c in enumerate(self._v2_cols):
            val = v2.get(c, 0.0)
            x[i] = val if val == val else 0.0
        if self._scaler is not None:
            x_in = self._scaler.transform(x.reshape(1, -1))
        else:
            x_in = x.reshape(1, -1)
        p = self._clf.predict_proba(x_in)[0]
        return float(p[self._long_idx]), float(p[self._short_idx]), float(p[self._neut_idx])

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        p_long, p_short, p_neut = self._predict(state)
        # Directional advantage over NEUTRAL
        dir_conf = max(p_long, p_short)
        if dir_conf - p_neut < self.t_strength:
            return None
        if p_long >= p_short:
            if p_long >= self.t_long:
                return EntrySignal(direction='long', tier=self.name,
                                    extras={'p_long': p_long,
                                            'p_short': p_short,
                                            'p_neutral': p_neut})
        else:
            if p_short >= self.t_short:
                return EntrySignal(direction='short', tier=self.name,
                                    extras={'p_long': p_long,
                                            'p_short': p_short,
                                            'p_neutral': p_neut})
        return None
