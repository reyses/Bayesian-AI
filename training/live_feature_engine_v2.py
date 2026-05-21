"""LiveFeatureEngineV2 -- streaming V2 feature engine (185D vector).

Subclass of LiveFeatureEngine. Inherits all bar ingestion, per-TF
aggregation, and dedupe machinery. Adds two paths:

  1. `get_v2_vector(ts) -> np.ndarray[185]`  -- on-demand single-row V2
     vector for the latest closed bar at-or-before `ts`. Used by
     L5Decider at R-trigger fire (B7) and at K=5 (B9).

  2. `get_v2_row(ts) -> pd.Series`  -- same data, named columns. Used
     by the parity test and by the chunk dumper.

Both paths follow the same lookahead-safe alignment as
`training/build_dataset_v2.py:_last_closed_idx`:

    idx = searchsorted(tf_ts, anchor_ts - period, side='right') - 1

A bar labeled B at TF `tf` covers [B, B + period) and closes at B + period.
At anchor timestamp `ts`, we can only use bars where B + period <= ts.

This pattern is the lookahead invariant from
`docs/JULES_ENGINE_DECOUPLE_ORDERS.md` and the V2 spec.

Storage: chunks land under `DATA/ATLAS_LIVE/FEATURES_5s_v2/{family}/`
mirroring the batch builder's layout, so the engine can be warmstarted
from yesterday's chunks just like V1.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

import glob
from typing import Optional, Dict

from training.live_feature_engine import (
    LiveFeatureEngine,
    SFE_WINDOW,
    SFE_MIN_BARS,
    TF_SECONDS as V1_TF_SECONDS,
)
from core_v2.statistical_field_engine import StatisticalFieldEngine as SFE_V2
from core_v2.features import (
    FEATURE_NAMES as V2_FEATURE_NAMES,
    N_FEATURES as N_V2,
    TF_ORDER as V2_TF_ORDER,
    TF_SECONDS as V2_TF_SECONDS,
    N_BASE as V2_N_BASE,
    LAYER_FAMILIES,
    _l0_names, _l1_names, _l2_names, _l3_names,
)
from core.features import N_FEATURES as N_V1   # V1 91D length (placeholder only)

logger = logging.getLogger(__name__)


# Hurst R/S in v2 uses N_BASE[tf] * N_HURST_MULT (=8) bars of warmup
# Largest is 1D: 5 * 8 = 40 bars. We need at least that many for L3 to
# produce non-NaN. To be safe, require 2 * N_HURST_MULT * max_N_BASE
# = 2 * 8 * 18 (4h) = 288 bars for the highest TFs. Live engine loads
# WARMUP_DAYS=5 of history -- on 1D that's 5 bars, insufficient. The
# fix: load 60+ days of warmup, OR mark 1D L3 fields as NaN until enough
# bars accumulate. We do the latter to keep startup cheap.

# Per-TF cache key for compute_L0/L1/L2/L3 results. Avoids recomputing
# every bar when only the anchor needs a new lookup.
# cache[tf] = (cache_key, l0_df_or_None, l1_df, l2_df, l3_df)
# cache_key = (n_bars, latest_ts)


class LiveFeatureEngineV2(LiveFeatureEngine):
    """V2 streaming feature engine -- 185D vector on demand."""

    def __init__(self, atlas_root: str = 'DATA/ATLAS_NT8',
                 live_features_root: str = 'DATA/ATLAS_LIVE/FEATURES_5s_v2',
                 v2_only: bool = False):
        """
        Args:
            v2_only: when True, on_bar() does NOT compute the V1 91D vector
                     (the legacy SFE pass). It still ingests + aggregates
                     bars (needed for get_v2_vector). The L5 stack uses
                     V2 exclusively, so this skips ~all the per-bar SFE
                     cost and removes the unused V1 path. on_bar returns
                     a zeros(N_V1) placeholder for new bars (the ledger's
                     update_bar only touches features[12], guarded).
        """
        super().__init__(atlas_root=atlas_root)
        self._sfe_v2 = SFE_V2()
        self._live_features_root = live_features_root
        self._v2_only = v2_only
        # Per-TF SFE-v2 cache: tf -> (cache_key, l1_df, l2_df, l3_df)
        self._v2_cache: dict[str, tuple] = {}
        # Last computed V2 vector + the timestamp it was computed for
        # (so the engine can dump it without re-running the computation)
        self._last_v2_vector: Optional[np.ndarray] = None
        self._last_v2_ts: Optional[float] = None

        # V2 needs 4h TF that V1 parent doesn't track. Add its accumulator
        # so _aggregate_bar('4h', ...) works.
        if '4h' not in self._accumulators:
            self._accumulators['4h'] = {
                'tf_seconds': V2_TF_SECONDS['4h'],
                'current_start': 0,
                'open': 0, 'high': -1e18, 'low': 1e18,
                'close': 0, 'volume': 0, 'count': 0,
            }

    # ──────────────────────────────────────────────────────────────────
    # OVERRIDES -- extend V1 machinery for V2 TF coverage
    # ──────────────────────────────────────────────────────────────────

    def load_history(self, exclude_day: Optional[str] = None):
        """Load V2 TF history. Extends V1 + adds 4h + dedup-before-sort.

        Issue with V1's load_history: pandas `sort_values('timestamp')` is
        unstable (uses quicksort by default), so when two parquets have
        overlapping timestamps (1D especially: each file has 2 rows, one
        being yesterday's session close), the post-sort order of those
        duplicates is non-deterministic. A subsequent `drop_duplicates`
        could pick either row.

        The V2 batch builder (`_load_tf_all_days` in build_dataset_v2.py)
        DEDUPES BEFORE SORTING, which preserves the concat order (later
        file wins). We replicate that here.

        Result: bar_counts dict tracks the per-TF row counts after dedup.
        `_day_ends` is rebuilt from scratch based on the test-day boundary
        of each file's source timestamps.
        """
        # Reset state
        self._last_loaded_ts = {}
        self._day_ends = {}

        # All TFs to load (V2 set = ['5s'] + V2_TF_ORDER minus 5s itself)
        all_tfs = ['5s'] + [tf for tf in V2_TF_ORDER if tf != '5s']

        bar_counts = {}
        for tf in all_tfs:
            tf_dir = os.path.join(self._atlas_root, tf)
            if not os.path.exists(tf_dir):
                continue
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if not files:
                continue

            dfs = []
            file_day_for_row: list[str] = []
            for f in files:
                day_name = os.path.basename(f).replace('.parquet', '')
                if exclude_day and day_name >= exclude_day:
                    continue
                df = pd.read_parquet(f)
                if df.empty:
                    continue
                dfs.append(df)
                file_day_for_row.extend([day_name] * len(df))

            if not dfs:
                continue

            concat = pd.concat(dfs, ignore_index=True)
            concat['_src_day'] = file_day_for_row
            # Dedupe BEFORE sort (preserves concat order; later file wins)
            deduped = concat.drop_duplicates(subset='timestamp', keep='last')
            # Stable sort by timestamp
            sorted_df = (deduped.sort_values('timestamp', kind='mergesort')
                                  .reset_index(drop=True))
            src_days = sorted_df['_src_day'].values
            full = sorted_df.drop(columns='_src_day')

            self._bars[tf] = full
            bar_counts[tf] = len(full)
            self._sfe_bar_count[tf] = 0
            self._last_loaded_ts[tf] = float(full['timestamp'].iloc[-1])

            # Rebuild day_ends: scan src_days to find where each day ENDS
            day_ends: dict[str, int] = {}
            for i, d in enumerate(src_days):
                day_ends[d] = i + 1
            self._day_ends[tf] = day_ends

        return bar_counts

    def on_bar(self, bar: dict) -> Optional[np.ndarray]:
        """Feed one 5s bar. Extends V1 to also aggregate to 4h.

        Returns V1 91D vector as before (so the engine_v2 ledger path
        is unaffected). V2 vector is computed on-demand via
        `get_v2_vector(ts)`.

        IMPORTANT: ATLAS_NT8 stores bar timestamps as bin_END (a 5m bar
        at ts=T covers (T-300, T]). V1 parent's `_aggregate_bar` produces
        bin_START timestamps, which would create a convention mismatch
        when concatenated with bin_END history. We normalize aggregated
        bars to bin_END here so `self._bars[tf]` is consistent.
        """
        ts = bar['timestamp']
        if '5s' in self._bars and len(self._bars['5s']) > 0:
            highest_seen = float(self._bars['5s']['timestamp'].iloc[-1])
        else:
            highest_seen = 0
        is_new = ts > highest_seen

        if is_new:
            self._append_bar('5s', bar)
            v2_higher_tfs = [tf for tf in V2_TF_ORDER if tf != '5s']
            for tf in v2_higher_tfs:
                if tf in self._accumulators:
                    completed = self._aggregate_bar(tf, bar)
                    if completed:
                        # Convert bin_START -> bin_END (ATLAS_NT8 convention)
                        period = self._accumulators[tf]['tf_seconds']
                        completed['timestamp'] = (
                            int(completed['timestamp']) + period - 1
                        )
                        self._append_bar(tf, completed)

        if not is_new:
            return None

        if self._v2_only:
            # V1 PURGED. The L5 stack reasons entirely off V2 (B7/B9/B10
            # via get_v2_vector). Skip the legacy 91D SFE pass entirely.
            # Return a zeros placeholder so the engine's ledger.update_bar
            # signature is satisfied -- update_bar only reads features[12]
            # and is guarded for short/zero vectors.
            return np.zeros(N_V1, dtype=np.float32)

        # Compute V1 91D for the ledger path (legacy / blended parity)
        return self._compute_features(ts)

    # ──────────────────────────────────────────────────────────────────
    # PRIMARY ACCESSORS
    # ──────────────────────────────────────────────────────────────────

    def get_v2_vector(self, ts: float) -> Optional[np.ndarray]:
        """Return the 185D V2 vector at-or-before `ts`. None on warmup.

        Uses the lookahead-safe `searchsorted(... ts - period ...)` pattern
        for each TF. Caches per-TF L1/L2/L3 DataFrames between calls;
        recomputes a TF only when its bar count changes.

        Args:
            ts: anchor timestamp (Unix seconds).
        Returns:
            ndarray shape (185,) with column order = FEATURE_NAMES.
            None if any required TF lacks enough warmup.
        """
        row = self._compute_v2_row(ts)
        if row is None:
            return None
        vec = row[V2_FEATURE_NAMES].to_numpy(dtype=np.float32)
        self._last_v2_vector = vec
        self._last_v2_ts = float(ts)
        return vec

    def get_v2_row(self, ts: float) -> Optional[pd.Series]:
        """Same as get_v2_vector but returns a named pd.Series (for tests)."""
        return self._compute_v2_row(ts)

    # ──────────────────────────────────────────────────────────────────
    # CORE: assemble single V2 row from cached per-TF layer DataFrames
    # ──────────────────────────────────────────────────────────────────

    def _compute_v2_row(self, ts: float) -> Optional[pd.Series]:
        """Build a single 185D row aligned to anchor `ts`. None on warmup."""
        out = pd.Series(np.nan, index=V2_FEATURE_NAMES, dtype=np.float64)

        # L0 (global time-of-day) -- trivial, no TF dependency
        out['L0_time_of_day'] = (float(ts) % 86400.0) / 86400.0

        for tf in V2_TF_ORDER:
            if tf not in self._bars or len(self._bars[tf]) < SFE_MIN_BARS:
                # Warmup: leave NaN for this TF's columns
                continue

            df = self._bars[tf]
            period = V2_TF_SECONDS[tf]

            # Compute (or reuse cached) L1/L2/L3 DataFrames for this TF
            tf_features = self._get_tf_layer_features(tf, df, 0)
            if tf_features is None:
                continue

            l1, l2, l3 = tf_features

            # The cache stored the deduped timestamp array; use it for the
            # lookahead-safe alignment. Same formula as the V2 batch builder
            # `_last_closed_idx`:
            #     idx = searchsorted(tf_ts, anchor_ts - period, side='right') - 1
            cached = self._v2_cache[tf]
            tf_ts_dedup = cached[4]
            idx = int(np.searchsorted(tf_ts_dedup, ts - period, side='right') - 1)
            if idx < 0 or idx >= len(l1):
                continue

            # Pull the single row at `idx` for each layer
            for col in _l1_names(tf):
                if col in l1.columns:
                    out[col] = float(l1[col].iloc[idx])
            for col in _l2_names(tf):
                if col in l2.columns:
                    out[col] = float(l2[col].iloc[idx])
            for col in _l3_names(tf):
                if col in l3.columns:
                    out[col] = float(l3[col].iloc[idx])

        return out

    def _get_tf_layer_features(self, tf: str, df: pd.DataFrame,
                                 needed_idx: int) -> Optional[tuple]:
        """Return (l1_df, l2_df, l3_df) for a TF. Caches on (n_bars, latest_ts).

        Re-runs SFE-v2 only when bars have been added for this TF since the
        last call.

        Dedupes by timestamp before SFE (ATLAS_NT8 1D parquets overlap
        across files; without dedup the SFE sees duplicate consecutive
        rows and produces wrong velocity/accel).

        Args:
            tf: TF label.
            df: cached per-TF OHLCV DataFrame.
            needed_idx: the row index we need features for (we map it
                        back to the deduped DataFrame via timestamp).
        Returns:
            (l1, l2, l3) DataFrames each of length len(deduped_df).
            None on failure. The returned `needed_idx` for the caller
            (via instance state `self._last_deduped_idx`) reflects the
            position of the original row in the deduped DataFrame.
        """
        n = len(df)
        if n < SFE_MIN_BARS:
            return None
        latest_ts = float(df['timestamp'].iloc[-1])
        cache_key = (n, latest_ts)

        cached = self._v2_cache.get(tf)
        if cached and cached[0] == cache_key:
            return cached[1], cached[2], cached[3]

        # Dedupe by timestamp (matches V2 batch builder _load_tf_all_days).
        df_dd = (df.drop_duplicates(subset='timestamp', keep='last')
                   .sort_values('timestamp')
                   .reset_index(drop=True))

        # Recompute on deduped frame. SFE v2 is stateless and pure.
        try:
            l1 = self._sfe_v2.compute_L1(df_dd, tf=tf)
            l2 = self._sfe_v2.compute_L2(df_dd, tf=tf)
            l3 = self._sfe_v2.compute_L3(df_dd, tf=tf)
        except Exception as e:
            logger.warning(f'V2 SFE failed for tf={tf} n={n}: {e}')
            return None

        # Cache: keyed on the original (non-deduped) cache_key but stores
        # deduped layer DataFrames + the deduped timestamp array (so the
        # row-resolver can map anchor->idx in deduped space).
        self._v2_cache[tf] = (
            cache_key, l1, l2, l3, df_dd['timestamp'].values
        )
        return l1, l2, l3

    # ──────────────────────────────────────────────────────────────────
    # CHUNK DUMP -- write the V2 row to DATA/ATLAS_LIVE/FEATURES_5s_v2/
    # ──────────────────────────────────────────────────────────────────

    def dump_v2_row(self, ts: float, day_label: str) -> bool:
        """Append the V2 vector for `ts` to today's per-family parquets.

        Mirrors the batch builder's layout: one parquet per layer-family
        per day. Each parquet has timestamp + the family's feature cols.

        Args:
            ts: anchor timestamp.
            day_label: 'YYYY_MM_DD' day key.
        Returns:
            True if a row was appended. False if vector unavailable (warmup).
        """
        row = self._compute_v2_row(ts)
        if row is None:
            return False

        for family, meta in LAYER_FAMILIES.items():
            family_path = os.path.join(self._live_features_root, family,
                                          f'{day_label}.parquet')
            os.makedirs(os.path.dirname(family_path), exist_ok=True)
            cols = ['timestamp'] + list(meta['features'])
            family_row = pd.DataFrame(
                [[ts] + [float(row[c]) for c in meta['features']]],
                columns=cols,
            )
            if os.path.exists(family_path):
                existing = pd.read_parquet(family_path)
                merged = pd.concat([existing, family_row], ignore_index=True)
                # Dedupe on timestamp -- keep latest entry per ts
                merged = (merged.sort_values('timestamp')
                                  .drop_duplicates('timestamp', keep='last')
                                  .reset_index(drop=True))
                merged.to_parquet(family_path, index=False)
            else:
                family_row.to_parquet(family_path, index=False)
        return True
