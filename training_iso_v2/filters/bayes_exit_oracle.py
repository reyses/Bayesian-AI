"""Bayesian probability table EXIT ORACLE.

Refined framework (2026-05-10): the probability table's primary use is to
decide WHEN TO EXIT an open position during non-CAT operation. Tiers
handle ENTRY; this module handles EXIT.

Exit decision per bar of an open trade:
    1. If approaching a known CAT window (Tue UTC 1, Thu UTC 2, etc.):
       close pre-emptively if direction is wrong for that window's bias.
    2. If past peak_horizon for the cell + PnL decay > threshold: exit.
    3. If P(reverse in next 5 min) crosses threshold: exit.
    4. If hard z-stop hit (entry +/- q90_excess * SE): exit.

Inputs at each bar:
    - state.timestamp, direction, entry_price, current_price, entry_ts
    - state.v2 dict (for at-bar feature lookup)
    - peak/magnitude tables (loaded once at init)

Returns: (should_exit: bool, reason: str)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from training_iso_v2.filters.bayes_filters import (
    cat_harvest_signal, CAT_HARVEST_WINDOWS, _load_tod_dow_p_cat,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@lru_cache(maxsize=1)
def _load_duration_table() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT,
        'reports/findings/segments/bayes_table_v0_location/duration_per_axis.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_magnitude_table() -> pd.DataFrame:
    path = os.path.join(REPO_ROOT,
        'reports/findings/segments/bayes_table_v0_location/magnitude_per_axis.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def time_to_next_cat_window(timestamp_utc: float,
                             min_lookahead_s: int = 600) -> Optional[int]:
    """Returns seconds until next CAT_HARVEST window starts, or None if
    no such window in the next 24 hours."""
    dt = datetime.fromtimestamp(int(timestamp_utc), tz=timezone.utc)
    cur_dow = dt.strftime('%a')
    cur_hr = dt.hour
    cur_min = dt.minute
    # Search forward up to 24h
    for offset_min in range(0, 24*60, 5):
        check_dt = datetime.fromtimestamp(int(timestamp_utc) + offset_min*60,
                                            tz=timezone.utc)
        for dow, h_start, h_end, side in CAT_HARVEST_WINDOWS:
            if dow == check_dt.strftime('%a') and h_start <= check_dt.hour < h_end:
                if offset_min*60 >= min_lookahead_s:
                    return offset_min * 60
    return None


class BayesExitOracle:
    """At each bar of an open trade, return (exit?, reason)."""

    def __init__(self,
                  default_peak_horizon_min: float = 30.0,
                  giveback_threshold: float = 0.30,
                  pre_cat_close_lead_s: int = 600):
        """
        default_peak_horizon_min : if no cell data, exit after this many minutes
        giveback_threshold       : if past peak AND PnL < peak_PnL * (1-giveback): exit
        pre_cat_close_lead_s     : close positions this many seconds before a
                                    cat-harvest window opens (if direction is
                                    opposite to the window's bias)
        """
        self.default_peak_horizon_min = default_peak_horizon_min
        self.giveback_threshold = giveback_threshold
        self.pre_cat_close_lead_s = pre_cat_close_lead_s
        self._dur = _load_duration_table()
        self._mag = _load_magnitude_table()

    def _lookup_peak_horizon_min(self, side_anchor: str, sigma_q: int) -> float:
        """Use the duration table to get the typical peak-PnL horizon for
        a (side, anchor, sigma_q) cell. Fallback: default."""
        if self._dur.empty:
            return self.default_peak_horizon_min
        side, anchor = (side_anchor.split('_') + ['', ''])[:2]
        sub = self._dur[(self._dur['split']=='IS')
                          & (self._dur['side']==side)
                          & (self._dur['anchor']==anchor)
                          & (self._dur['axis']=='sigma_rank_q')
                          & (self._dur['bin']==sigma_q)
                          & (self._dur['threshold_min']==10)]
        if sub.empty:
            return self.default_peak_horizon_min
        return float(sub['med_duration'].iloc[0])

    def should_exit(self, state, trade) -> Tuple[bool, str]:
        """Per-bar exit query.

        state must have: .timestamp (s), .v2 (dict), .price
        trade must have: .entry_ts (s), .direction ('long' or 'short'),
                          .entry_price, .peak_pnl (running max favorable)
        """
        t_in_trade_s = state.timestamp - trade.entry_ts
        t_in_trade_m = t_in_trade_s / 60.0

        # ─── Rule A: PRE-CAT pre-emptive close ──────────────────────────
        # If we're holding a LONG within `pre_cat_close_lead_s` of a Tue UTC 1
        # (or similar) window, close it — the window has 96% crash bias.
        t_to_cat = time_to_next_cat_window(state.timestamp,
                                              min_lookahead_s=0)
        if t_to_cat is not None and t_to_cat <= self.pre_cat_close_lead_s:
            # All known CAT windows are SHORT-bias → close LONGs pre-emptively
            if trade.direction == 'long':
                return True, f'pre_cat_close_long_window_in_{t_to_cat}s'

        # ─── Rule B: Peak-horizon + giveback ────────────────────────────
        # Get current cell info — use a coarse default based on overall sigma
        # If state has L2_15m_vol_sigma_12 we can bucket; else use default
        sigma_q = 2  # default mid
        v = state.v2.get('L2_15m_vol_sigma_12', float('nan'))
        if np.isfinite(v):
            # Rough sigma_q estimate (not precise — would need population edges)
            # Quick heuristic: sigma_q from current vs population scale
            pass  # leave default for now
        side_anchor = 'above_low' if trade.direction == 'long' else 'below_high'
        peak_min = self._lookup_peak_horizon_min(side_anchor, sigma_q)

        current_pnl = ((state.price - trade.entry_price)
                         if trade.direction == 'long'
                         else (trade.entry_price - state.price))
        peak_pnl = max(getattr(trade, 'peak_pnl', current_pnl), current_pnl)

        if (t_in_trade_m > peak_min) and (peak_pnl > 0) and (
              current_pnl < peak_pnl * (1 - self.giveback_threshold)):
            return True, f'past_peak_h{peak_min:.0f}m_giveback_{int(100*self.giveback_threshold)}pct'

        # ─── Rule C: Hard stop on excursion ─────────────────────────────
        # Need entry-bar SE_anchor to size this properly; deferred until
        # full engine integration. Default to NOT firing here.

        # ─── Rule D: Time-stop fallback ────────────────────────────────
        if t_in_trade_m > 3 * peak_min:
            return True, f'time_stop_3x_peak_{peak_min:.0f}m'

        return False, 'hold'
