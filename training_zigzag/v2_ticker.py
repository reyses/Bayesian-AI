"""V2Ticker — bar-by-bar state iterator for the L5 / zigzag engine.

Analogue of `training/sfe_ticker.FeatureTicker` (the established "training
ticker" pattern used by `training/forward_blended.py`). FeatureTicker yields
per-5s-bar V1 91D states; this class yields the equivalent state dict that
`L5Decider.evaluate()` consumes — OHLCV from the 5s bar parquet plus the V2
feature getter (which reads the project's precomputed `FEATURES_5s_v2` L0 /
L1_* layers under the hood — same code path the live engine takes).

L5 ignores V1 features (it reasons off V2 via `v2_getter`), so `features_79d`
is a placeholder zero vector for ledger/interface compatibility only.

The consumer (run_day in forward_zigzag.py) injects the per-bar `positions`
snapshot before calling `engine.evaluate(state)`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


_PLACEHOLDER_V1 = np.zeros(91, dtype=np.float32)


class V2Ticker:
    """Yields one state dict per 5s bar for a single day."""

    def __init__(self, day_bars: pd.DataFrame, lfe):
        """
        day_bars : DataFrame with columns timestamp, open, high, low, close, volume
                   (sorted ascending by timestamp).
        lfe      : pre-warmed LiveFeatureEngineV2 — supplies get_v2_vector for the
                   V2 feature lookups (reads precomputed L0 / L1_* layers, same
                   path the live engine uses).
        """
        self._bars = day_bars.reset_index(drop=True)
        self._lfe = lfe

    def __iter__(self):
        for b in self._bars.itertuples(index=False):
            close = float(b.close)
            yield {
                'features_79d': _PLACEHOLDER_V1,            # L5 ignores V1
                'price':        close,
                'high':         float(getattr(b, 'high', close)),
                'low':          float(getattr(b, 'low', close)),
                'volume':       float(getattr(b, 'volume', 0.0)),
                'timestamp':    float(b.timestamp),
                'v2_getter':    self._lfe.get_v2_vector,     # precomputed-feature lookup
                # 'positions' is injected per-bar by the run_day consumer
            }

    @property
    def last_bar(self):
        return self._bars.iloc[-1]

    def __len__(self):
        return len(self._bars)
