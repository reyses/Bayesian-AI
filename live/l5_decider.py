"""L5Decider -- zigzag + B7 + B9 + B10 at 1-contract cap.

Same `evaluate(state) -> DecisionBatch` interface as BlendedEngine, so
engine_v2's bar-loop can use it without per-call-site changes.

PHASE 1 LOGIC (1 contract per position, no scaling):

  Per bar:
    1. Update 1m aggregator + ATR(14) -> r_price
    2. Update zigzag state machine on 5s closes
    3. For each open position: track MAE/MFE
       - 5 five-second bars (25s) after entry: query B9; CUT if pred < threshold
    4. If R-trigger fires AND flat: query B7; ENTRY if pred_R >= threshold

  B10 day-mode (computed once at session start from cross_day_features):
    - if P(high) >= 0.5: 'normal' (boost dead at 1c)
    - if P(low)  >= 0.7: 'cautious' -> tighten B7 + B9 thresholds
    - else: 'normal'

  Cautious mode:
    - B7 skip threshold: 1.0 -> 1.5
    - B9 cut threshold:  -50  -> -25

NO scaling, no chains, no HALF, no PYRAMID, no daily-loss cap, no SL
tighten. All deferred to later phases. See docs/Active/LIVE_L5_ARCHITECTURE.md.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

import numpy as np
import pandas as pd

from core.engine_signals import (
    DecisionBatch,
    EntrySignal,
    PositionDecision,
)

logger = logging.getLogger('l5_decider')


# ─── Constants (named, no magic numbers) ──────────────────────────────────
TICK_SIZE = 0.25
DOLLAR_PER_POINT = 2.0       # MNQ: 1 point = $2 per contract
FRICTION_PER_LEG = 6.0        # $4 commission + $2 slippage (per leg, round trip approx)

# Zigzag ATR-based R-trigger -- MUST match tools/build_zigzag_pivot_dataset.py
# + tools/_viz/auto_swing_marker.py.detect_swings
ATR_PERIOD_1M = 14            # ATR window length (matches compute_atr in builder)
ATR_TR_WINDOW = ATR_PERIOD_1M * 3  # builder takes np.median(tr[-period*3:]) = 42 TRs
ATR_MULTIPLIER = 4.0          # R-trigger threshold ticks = ATR(14)/tick * 4
MIN_REV_TICKS_FLOOR = 4       # absolute floor on reversal threshold (in ticks)
MIN_BARS_BETWEEN_PIVOTS = 36  # 5s bars (= 3 minutes) -- production builder uses 36

# Phase-1 1-contract sizing thresholds.
# IS-calibrated by tools/forward_pass_1contract.py on 12,120 normal-mode
# IS legs (275d). MUST match those values to keep mock <-> OOS apples-to-apples.
B7_SKIP_THRESHOLD_NORMAL = 1.90      # take entry if pred_R >= this
B7_SKIP_THRESHOLD_CAUTIOUS = 2.10    # B10 cautious -> only take stronger setups (+0.20)
B9_K = 5                              # B9 fires 5 five-second bars (25s) after entry
B9_CUT_THRESHOLD_NORMAL = 5.0        # CUT if pred remaining_amp < this (in $)
B9_CUT_THRESHOLD_CAUTIOUS = 15.0     # B10 cautious -> cut weaker losers faster (+10)

# B10 day-mode thresholds
B10_P_HIGH_THRESHOLD = 0.5           # day-classifier: P(high-vol) >= this -> "normal" (boost dead at 1c)
B10_P_LOW_THRESHOLD = 0.7            # day-classifier: P(low-vol) >= this -> "cautious"

# B10 cross-day feature columns (must match training in train_b10_vol_regime_sizer.py)
B10_FEATS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc', 'dow',
]


@dataclass
class L5Context:
    """Loaded models + paths needed by L5Decider."""
    b7: dict
    b9: dict
    b10_high: dict
    b10_low: dict
    cross_day_features_path: str = 'DATA/CROSS_DAY/cross_day_features.parquet'
    # Pivot source: 'stream' = causal detector (production live), 'replay'
    # = inject pivots from a pre-built parquet (mock/SIM validation)
    pivot_source: str = 'stream'
    # When pivot_source='replay', this DataFrame holds (timestamp, pivot_dir,
    # pivot_price) for all days. L5Decider fires R-triggers at these
    # timestamps and skips its own zigzag state machine.
    replay_pivots_df: Optional[pd.DataFrame] = None

    @classmethod
    def load(cls,
              b7_path: str = 'reports/findings/regret_oracle/b7_leg_sizer.pkl',
              b9_path: str = 'reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl',
              b10_high_path: str = 'reports/findings/regret_oracle/b10_vol_regime_high.pkl',
              b10_low_path: str = 'reports/findings/regret_oracle/b10_vol_regime_low.pkl',
              pivot_source: str = 'stream',
              replay_pivot_parquet: Optional[str] = None):
        with open(b7_path, 'rb') as f:
            b7 = pickle.load(f)
        with open(b9_path, 'rb') as f:
            b9 = pickle.load(f)
        with open(b10_high_path, 'rb') as f:
            b10_high = pickle.load(f)
        with open(b10_low_path, 'rb') as f:
            b10_low = pickle.load(f)
        logger.info(f'L5 models loaded: B7 v2_cols={len(b7["v2_cols"])} '
                     f'B9 feat_cols={len(b9["feat_cols"])} '
                     f'B10 (high/low classifiers)')

        replay_pivots_df = None
        if pivot_source == 'replay':
            # Replay must use HARDENED LEGS (R-trigger fire timestamps),
            # not raw pivot timestamps. The pivot extreme bar is future
            # knowledge -- the R-trigger fire is causal (close crosses
            # pivot +- r_price). Hardened legs CSV stores fire_ts as
            # entry_ts and exit_ts.
            #
            # Schema we synthesize for the replay engine:
            #   day, timestamp, pivot_dir
            # One row per "trigger event" (entry of leg N is also the exit
            # of leg N-1, so we emit only ENTRY events here -- the engine's
            # zigzag-reverse logic handles the exit on subsequent ENTRY of
            # opposite direction).
            path = replay_pivot_parquet \
                or 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
            legs = pd.read_csv(path)
            # Each leg's entry_ts is an R-trigger fire. pivot_dir = leg_dir.
            # We also need to fire an exit-only trigger for the LAST leg's
            # exit_ts (otherwise the final position stays open).
            ev_rows = []
            for _, r in legs.iterrows():
                ev_rows.append({
                    'day':       r['day'],
                    'timestamp': float(r['entry_ts']),
                    'pivot_dir': r['leg_dir'],
                })
            # Per-day, add the last leg's exit_ts as a synthetic "close" trigger
            # in opposite direction of the last leg.
            for d, g in legs.groupby('day'):
                last_row = g.iloc[-1]
                close_dir = ('SHORT' if last_row['leg_dir'] == 'LONG' else 'LONG')
                ev_rows.append({
                    'day':       d,
                    'timestamp': float(last_row['exit_ts']),
                    'pivot_dir': close_dir,
                })
            replay_pivots_df = (pd.DataFrame(ev_rows)
                                  .sort_values(['day', 'timestamp'])
                                  .reset_index(drop=True))
            logger.info(f'Pivot REPLAY mode: loaded {len(replay_pivots_df)} '
                          f'R-trigger events from hardened legs at {path}')

        return cls(b7=b7, b9=b9, b10_high=b10_high, b10_low=b10_low,
                     pivot_source=pivot_source,
                     replay_pivots_df=replay_pivots_df)


@dataclass
class _PosTraj:
    """Per-open-position trajectory tracking for B9 + bookkeeping."""
    contract_id: str
    entry_ts: float
    entry_price: float
    leg_dir: str        # 'long' or 'short'
    peak_favorable: float = 0.0      # max favorable price excursion (points, >=0)
    peak_adverse: float = 0.0        # max adverse price excursion (points, >=0)
    b9_fired: bool = False
    # evaluate() runs once per 5-second bar, so self._bar_count counts 5s
    # bars. We snapshot it the first bar this position is seen; B9 then
    # fires B9_K bars (= B9_K * 5 seconds) later. The B9 model was trained
    # for K=5 in 5-SECOND-bar units (build_trade_trajectory_dataset.py:
    # bar_ts = entry_ts + K*5), so K=5 means entry + 25 seconds -- NOT the
    # 5 MINUTES that pos.bars_held (ledger // 60) would give.
    entry_bar_count: int = -1


class L5Decider:
    """Phase-1 1-contract engine. Same interface as BlendedEngine."""

    def __init__(self, ctx: L5Context):
        self._ctx = ctx

        # B10 day-mode state
        self._b10_day_label: Optional[str] = None
        self._b10_mode = 'normal'        # 'normal' | 'cautious'

        # 1m aggregator for ATR(14) -- need >=42 TRs in window for median
        self._bars_1m: deque = deque(maxlen=ATR_TR_WINDOW + 5)
        self._current_1m_acc: Optional[dict] = None
        self._min_rev_ticks: Optional[float] = None   # ATR*4 in TICKS (matches builder)
        self._r_price: Optional[float] = None         # min_rev_ticks * TICK_SIZE (points)

        # Zigzag state machine (5s closes, in tick space) -- mirrors
        # tools/_viz/auto_swing_marker.detect_swings causal pass.
        # direction:  0 = undecided (warmup)
        #             1 = trending up, tracking running HIGH
        #            -1 = trending down, tracking running LOW
        self._zz_direction: int = 0
        # Current candidate extreme (will become pivot when reversal confirmed)
        self._zz_extreme_val_ticks: Optional[float] = None
        self._zz_extreme_idx: int = -1
        self._zz_extreme_ts: Optional[float] = None
        self._zz_extreme_price: Optional[float] = None
        # Bar count since start of session (drives min_bars-between-pivots)
        self._zz_bar_idx: int = -1
        # Index of last confirmed pivot (for min_bars check)
        self._zz_last_pivot_idx: int = -1
        # Confirmed pivots this session: (ts, price, 'high'/'low').
        # Drives the zigzag line drawn by the NT8 companion.
        self._zz_pivots: deque = deque(maxlen=200)
        # Initial value (ct[0] -- the first bar's close in tick space)
        # for warmup-direction detection
        self._zz_init_val_ticks: Optional[float] = None
        # Last R-trigger fire (this bar only)
        self._this_bar_rtrig_dir: Optional[str] = None
        self._this_bar_rtrig_price: Optional[float] = None
        # Last B7/B9 predictions -- exposed for the live dashboard overlay
        self._last_b7_pred: Optional[float] = None
        self._last_b9_pred: Optional[float] = None
        # Skipped-signal tracking: an R-trigger fired but no entry was taken.
        # _this_bar_skip is reset each evaluate() and set by _record_skip.
        self._this_bar_skip: Optional[dict] = None
        self._skip_count: int = 0
        # Pending entry carryover: when a zigzag-reverse exit fires, the
        # entry order can't go out the same bar (ledger still shows old
        # position until exit FILL arrives). We remember the direction
        # here and fire the entry on the next evaluate() once flat.
        self._pending_flip_entry_dir: Optional[str] = None

        # Per-position trajectory (keyed by contract_id)
        self._pos_traj: dict[str, _PosTraj] = {}

        # Cache loaded cross-day-features dataframe (loaded lazily)
        self._cross_day_df: Optional[pd.DataFrame] = None

        # 5-second-bar count since start (one increment per evaluate()).
        # Drives B9's K=5 horizon: B9 fires B9_K bars after a position's
        # entry_bar_count snapshot.
        self._bar_count = 0

        logger.info('L5Decider initialized in mode=normal (will switch on '
                      'first session start when cross-day features load).')

    def prime_atr_from_history(self, bars_1m_df):
        """Lock min_rev_ticks at session start from yesterday's full-day
        1m bars. Matches the production builder's per-day approach:

            atr_pts = median(TR_full_day[-period*3:])
            min_rev_ticks = max(4, round(atr_pts / TICK_SIZE * ATR_MULT))

        We use YESTERDAY's full-day TRs as a causal proxy for today's
        (which we don't have yet at session start). This locks
        min_rev_ticks for the whole session -- no mid-day drift.

        Args:
            bars_1m_df: pd.DataFrame with columns timestamp, open, high, low,
                         close, volume. Must be the FULL prior day's 1m
                         bars (engine_v2 passes self._lfe._bars['1m'] which
                         is cumulative; we slice to yesterday below).
        """
        if bars_1m_df is None or len(bars_1m_df) == 0:
            logger.warning('L5Decider prime: no 1m history -- ATR cold start')
            return
        # Take the last 5 trading days of 1m bars (~5 * 1440 = 7200 1m bars).
        # Single-day-yesterday ATR is too noisy: a calm day followed by a
        # volatile day under-detects pivots by 3x (e.g., May 13 -> May 14
        # in the May 2026 OOS data). 5-day rolling smooths but adapts to
        # the recent volatility regime.
        tail = bars_1m_df.tail(7200)
        h = tail['high'].values.astype(np.float64)
        l = tail['low'].values.astype(np.float64)
        c = tail['close'].values.astype(np.float64)
        if len(c) < 2:
            return
        prev_c = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
        # Use median over the FULL 5-day tail (not just last 42 TRs). This
        # smooths single-day volatility shocks and gives a stable threshold
        # adapted to the recent regime. Production's 42-TR-of-current-day
        # median requires today's data we don't have causally.
        if len(tr) >= ATR_PERIOD_1M:
            atr_pts = float(np.median(tr))
        else:
            atr_pts = float(tr.mean())
        min_rev_ticks = max(MIN_REV_TICKS_FLOOR,
                              int(round(atr_pts / TICK_SIZE * ATR_MULTIPLIER)))
        self._min_rev_ticks = float(min_rev_ticks)
        self._r_price = min_rev_ticks * TICK_SIZE
        logger.info(f'L5Decider primed (yesterday-locked): '
                      f'tail={len(tail)} 1m bars, '
                      f'atr_pts={atr_pts:.2f}, '
                      f'min_rev_ticks={int(min_rev_ticks)}, '
                      f'r_price={self._r_price:.2f}')

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC API -- matches BlendedEngine.evaluate signature
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, state) -> DecisionBatch:
        """Same interface as BlendedEngine.evaluate.

        state expected keys:
            timestamp      float (bar timestamp, Unix seconds)
            price          float (bar close)
            high, low      float (optional; falls back to price)
            volume         int   (optional)
            v2_getter      callable(ts) -> np.ndarray[185]  (LiveFeatureEngineV2.get_v2_vector)
            positions      PositionsView snapshot
            features_79d   np.ndarray[91]  (V1 features; unused by L5)

        Returns DecisionBatch.
        """
        ts = float(state['timestamp'])
        price = float(state['price'])
        high = float(state.get('high', price))
        low = float(state.get('low', price))
        volume = int(state.get('volume', 0))
        v2_getter: Optional[Callable] = state.get('v2_getter')
        positions = state.get('positions')

        # 1. New-day rollover check
        day_label = self._day_label_from_ts(ts)
        if day_label != self._b10_day_label:
            self._on_session_start(day_label)

        # 2. 1m bar aggregation + ATR(14) -> r_price
        self._update_1m_aggregator(ts, price, high, low, volume)

        # 3. Update zigzag running extreme + detect R-trigger fire
        self._this_bar_rtrig_dir = None
        self._this_bar_rtrig_price = None
        self._this_bar_skip = None
        if self._ctx.pivot_source == 'replay':
            self._check_replay_pivots(ts, price)
        else:
            self._update_zigzag(ts, price)

        # 4. Update trajectories of open positions; query B9 at K=5.
        # evaluate() runs once per 5-second bar, so "B9_K 5s-bars elapsed
        # since entry" = B9_K * 5 seconds = the horizon B9 was trained on
        # (K=5 -> 25s). pos.bars_held is in MINUTES (ledger // 60) and must
        # NOT be used here -- it would fire B9 at the 5-MINUTE mark.
        position_decisions: list[PositionDecision] = []
        if positions is not None:
            for pos in self._iter_positions(positions):
                self._update_trajectory(pos, price, high, low)
                traj = self._pos_traj.get(pos.contract_id)
                # Fire B9 once, on the bar B9_K 5s-bars after entry.
                if (traj is not None
                        and not traj.b9_fired
                        and traj.entry_bar_count >= 0
                        and (self._bar_count - traj.entry_bar_count) >= B9_K):
                    pd_b9 = self._b9_query_full(pos, ts, price, v2_getter)
                    if pd_b9 is not None:
                        position_decisions.append(pd_b9)
                    traj.b9_fired = True

        # 5. R-trigger -> EXIT-AND-MAYBE-FLIP
        # The training data assumed each leg's exit IS the next R-trigger
        # fire (in the opposite direction). If there's an open position
        # and the new R-trigger is opposite-direction, close it.
        entry: Optional[EntrySignal] = None
        # NB: PositionsView exposes `is_flat` (property), NOT `has_any`.
        # Using getattr default would silently treat every bar as flat
        # and block exits.
        if positions is None:
            is_flat = True
        else:
            is_flat = bool(getattr(positions, 'is_flat', True))

        if self._this_bar_rtrig_dir is not None:
            # If we have an open position with OPPOSITE direction,
            # emit BOTH the exit (PositionDecision with exit_reason) AND
            # the new-direction entry (EntrySignal) in the SAME batch.
            # engine_v2 sees exit + entry together and sends both orders
            # back-to-back so they fill at the same bar's close (matches
            # the OOS hardened legs where exit_ts == entry_ts of next leg).
            if not is_flat:
                for pos in self._iter_positions(positions):
                    if pos.direction != self._this_bar_rtrig_dir:
                        already_exiting = any(
                            d.contract_id == pos.contract_id and d.exit_reason
                            for d in position_decisions)
                        if not already_exiting:
                            position_decisions.append(PositionDecision(
                                contract_id=pos.contract_id,
                                exit_reason='zigzag_reverse',
                            ))
                # Same-bar entry (engine_v2 bypasses is_flat check when
                # there's an exit in the same batch -- the "flip" pattern)
                entry = self._b7_query(ts, v2_getter)
            else:
                # Already flat -> open new position immediately
                entry = self._b7_query(ts, v2_getter)

        self._bar_count += 1
        return DecisionBatch(
            entry=entry,
            chain_entry=None,
            position_decisions=position_decisions,
            negative_exit=None,
        )

    # ──────────────────────────────────────────────────────────────────
    # SESSION / DAY HANDLING
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _day_label_from_ts(ts: float) -> str:
        """Format as 'YYYY_MM_DD' in UTC (matches ATLAS_NT8 file naming)."""
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return f'{dt.year:04d}_{dt.month:02d}_{dt.day:02d}'

    def _on_session_start(self, day_label: str):
        """Compute B10 day-mode for the new day, log, reset zigzag state."""
        prev = self._b10_day_label
        self._b10_day_label = day_label
        self._b10_mode = self._compute_b10_mode(day_label)
        # Reset zigzag state -- new session means new extreme baseline
        self._zz_direction = 0
        self._zz_extreme_val_ticks = None
        self._zz_extreme_idx = -1
        self._zz_extreme_ts = None
        self._zz_extreme_price = None
        self._zz_bar_idx = -1
        self._zz_last_pivot_idx = -1
        self._zz_init_val_ticks = None
        self._zz_pivots.clear()
        self._pending_flip_entry_dir = None
        # Reset 1m aggregator on session boundary so ATR doesn't carry
        # stale partial bars from prior session
        self._current_1m_acc = None
        logger.info(f'SESSION START {day_label}  (prev={prev})  '
                     f'B10 mode={self._b10_mode.upper()}  '
                     f'(B7_thr={self._b7_skip_threshold()}  '
                     f'B9_thr={self._b9_cut_threshold()})')

    def _compute_b10_mode(self, day_label: str) -> str:
        """Use B10 high/low classifiers on cross-day features to set mode.

        Cautious if P(low) >= B10_P_LOW_THRESHOLD; else normal.
        BOOST wing is dead at 1c (cannot size up) so we don't act on it.
        """
        try:
            if self._cross_day_df is None:
                if os.path.exists(self._ctx.cross_day_features_path):
                    self._cross_day_df = pd.read_parquet(
                        self._ctx.cross_day_features_path)
                    logger.info(f'Loaded cross-day features: '
                                  f'{self._ctx.cross_day_features_path}  '
                                  f'({len(self._cross_day_df)} rows)')
                else:
                    logger.warning(f'Cross-day features file missing: '
                                     f'{self._ctx.cross_day_features_path}  '
                                     f'-- defaulting to normal mode')
                    return 'normal'
            row = self._cross_day_df[
                self._cross_day_df['date_label'] == day_label]
            if row.empty:
                logger.warning(f'No cross-day features row for {day_label}  '
                                 f'-- defaulting to normal mode')
                return 'normal'
            X = row[B10_FEATS].fillna(0.0).values.astype(np.float32)
            p_high = float(self._ctx.b10_high['model'].predict_proba(X)[0, 1])
            p_low = float(self._ctx.b10_low['model'].predict_proba(X)[0, 1])
            logger.info(f'B10 for {day_label}: P(high)={p_high:.3f}  '
                          f'P(low)={p_low:.3f}')
            # Mutually exclusive: cautious wins over normal-boost
            if p_low >= B10_P_LOW_THRESHOLD:
                return 'cautious'
            return 'normal'
        except Exception as e:
            logger.exception(f'B10 inference failed: {e}')
            return 'normal'

    def _b7_skip_threshold(self) -> float:
        return (B7_SKIP_THRESHOLD_CAUTIOUS
                if self._b10_mode == 'cautious'
                else B7_SKIP_THRESHOLD_NORMAL)

    def _b9_cut_threshold(self) -> float:
        return (B9_CUT_THRESHOLD_CAUTIOUS
                if self._b10_mode == 'cautious'
                else B9_CUT_THRESHOLD_NORMAL)

    # ──────────────────────────────────────────────────────────────────
    # 1M AGGREGATION + ATR(14)
    # ──────────────────────────────────────────────────────────────────

    def _update_1m_aggregator(self, ts: float, close: float,
                                high: float, low: float, volume: int):
        """Accumulate 5s bars into 1m. On 1m close, append to deque.

        IMPORTANT: ATR + min_rev_ticks are LOCKED at session start from
        prime_atr_from_history() and NOT recomputed during the trading day.
        This matches production behaviour: the offline builder computes ONE
        atr_pts per day (median of 42 TRs from the full day's 1m bars), so
        the threshold is constant intraday. For live we use yesterday's
        full-day median (causal) -- same constant-threshold behaviour but
        based on yesterday's volatility, since today's full data isn't
        available yet.

        Mid-day re-computation drifts (rolling window mixes today's volatile
        bars into the median), producing thresholds that differ from any
        single static value. We don't do that.
        """
        boundary = int(ts) // 60 * 60      # 1m bin (bin_start convention)
        acc = self._current_1m_acc
        if acc is None or boundary != acc['boundary']:
            # Close previous bar (if any) and append to deque -- but do NOT
            # recompute ATR. The session-start lock is authoritative.
            if acc is not None and acc['count'] > 0:
                self._bars_1m.append({
                    'high': acc['high'], 'low': acc['low'],
                    'close': acc['close'], 'open': acc['open'],
                })
            # Open new 1m bar
            self._current_1m_acc = {
                'boundary': boundary,
                'open': close, 'high': high, 'low': low,
                'close': close, 'volume': volume, 'count': 1,
            }
        else:
            acc['high'] = max(acc['high'], high)
            acc['low'] = min(acc['low'], low)
            acc['close'] = close
            acc['volume'] += volume
            acc['count'] += 1

    def _recompute_atr(self):
        """Recompute ATR + min_rev_ticks. MUST match
        tools/build_zigzag_pivot_dataset.py:compute_atr exactly.

        Formula (from compute_atr):
            prev_c = concat([[c[0]], c[:-1]])
            tr = max(h - l, |h - prev_c|, |l - prev_c|)
            atr_pts = median(tr[-period*3:])    # = median of last 42 TRs
            min_rev_ticks = max(4, round(atr_pts / TICK_SIZE * ATR_MULT))
        """
        bars = list(self._bars_1m)
        if len(bars) < 2:
            return
        # Build TR array (skip first bar; prev_c = c[0] for index 1, c[i-1] otherwise)
        h = np.array([b['high'] for b in bars])
        l = np.array([b['low']  for b in bars])
        c = np.array([b['close'] for b in bars])
        prev_c = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
        # Builder: median of last ATR_PERIOD_1M * 3 (=42) TRs
        if len(tr) >= ATR_PERIOD_1M:
            atr_pts = float(np.median(tr[-ATR_TR_WINDOW:]))
        else:
            atr_pts = float(tr.mean())
        # min_rev_ticks in TICKS (note: builder works in tick space)
        min_rev_ticks = max(MIN_REV_TICKS_FLOOR,
                              int(round(atr_pts / TICK_SIZE * ATR_MULTIPLIER)))
        self._min_rev_ticks = float(min_rev_ticks)
        self._r_price = min_rev_ticks * TICK_SIZE

    # ──────────────────────────────────────────────────────────────────
    # ZIGZAG R-TRIGGER DETECTION
    # ──────────────────────────────────────────────────────────────────

    def _check_replay_pivots(self, ts: float, close: float):
        """Replay-mode pivot source: fire R-trigger if this bar's ts is
        within the production pivot dataset (used for SIM validation).

        Matches the OOS forward pass exactly: at each pivot timestamp,
        fire an R-trigger in the appropriate direction. Bypasses the
        streaming zigzag state machine entirely.
        """
        df = self._ctx.replay_pivots_df
        if df is None or len(df) == 0:
            return
        # Filter to current day for efficiency (build a per-day index lazily)
        if not hasattr(self, '_replay_today_idx') or self._replay_today_day != self._b10_day_label:
            day_df = df[df['day'] == self._b10_day_label].sort_values('timestamp')
            self._replay_today_pivots = day_df.reset_index(drop=True)
            self._replay_today_idx = 0
            self._replay_today_day = self._b10_day_label

        td = self._replay_today_pivots
        idx = self._replay_today_idx
        # Fire all pivots whose ts == this bar's ts (NT8 5s bars; ts is bin_END)
        while idx < len(td) and float(td.iloc[idx]['timestamp']) <= ts:
            piv = td.iloc[idx]
            piv_dir = str(piv['pivot_dir'])   # 'LONG' or 'SHORT'
            # Production: pivot_dir = direction of leg STARTING at this pivot
            self._this_bar_rtrig_dir = 'long' if piv_dir == 'LONG' else 'short'
            self._this_bar_rtrig_price = close
            idx += 1
        self._replay_today_idx = idx

    def _update_zigzag(self, ts: float, close: float):
        """Streaming zigzag pivot detector. CAUSAL port of
        tools/_viz/auto_swing_marker.detect_swings with the production
        params from tools/build_zigzag_pivot_dataset.py:
            min_reversal = self._min_rev_ticks   (= ATR median * 4)
            min_bars     = MIN_BARS_BETWEEN_PIVOTS (= 36 bars = 3 minutes)
            max_bars     = 0 (no duration cap -- not used in production)

        On each bar:
          1. If direction is undecided (warmup) and price has moved
             >= min_reversal from init: set initial direction + extreme.
          2. If trending up: update high if higher; else if reversal
             (extreme - price >= min_rev) AND we've held the extreme
             for >= min_bars: CONFIRM HIGH as pivot, flip direction,
             fire SHORT R-trigger.
          3. Mirror for trending down.

        R-trigger fires at the bar where the reversal threshold is hit
        (which IS the same bar the production R-trigger fires per
        detect_r_trigger_fire). Pivot timestamp is the prior extreme.
        """
        # Bar index (used by min_bars-between-pivots check)
        self._zz_bar_idx += 1
        i = self._zz_bar_idx

        if self._min_rev_ticks is None:
            # Warmup: ATR not yet available. Just remember init value.
            if self._zz_init_val_ticks is None:
                self._zz_init_val_ticks = close / TICK_SIZE
            return

        ct = close / TICK_SIZE
        min_rev = self._min_rev_ticks

        # First-time init (or after session reset)
        if self._zz_init_val_ticks is None:
            self._zz_init_val_ticks = ct
            self._zz_extreme_val_ticks = ct
            self._zz_extreme_idx = i
            self._zz_extreme_ts = ts
            self._zz_extreme_price = close
            return

        if self._zz_direction == 0:
            # Warmup -- waiting for first significant move from init
            if self._zz_extreme_val_ticks is None or ct > self._zz_extreme_val_ticks:
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
            init = self._zz_init_val_ticks
            if ct < init and (init - ct) >= min_rev:
                # First significant DOWN -- init was a HIGH pivot, now going down
                self._zz_direction = -1
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
                self._zz_last_pivot_idx = 0
            elif ct > init and (ct - init) >= min_rev:
                # First significant UP -- init was a LOW pivot, now going up
                self._zz_direction = 1
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
                self._zz_last_pivot_idx = 0
            return

        if self._zz_direction == 1:
            # Trending up -- track running HIGH
            if ct >= self._zz_extreme_val_ticks:
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
                return
            # Reversal candidate
            if (self._zz_extreme_val_ticks - ct) >= min_rev \
                    and (i - self._zz_extreme_idx) >= MIN_BARS_BETWEEN_PIVOTS:
                # CONFIRM HIGH as pivot. Next leg is SHORT.
                self._this_bar_rtrig_dir = 'short'
                self._this_bar_rtrig_price = close
                self._zz_last_pivot_idx = self._zz_extreme_idx
                self._zz_pivots.append(
                    (self._zz_extreme_ts, self._zz_extreme_price, 'high'))
                # Reset extreme to current bar; flip direction
                self._zz_direction = -1
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
            return

        if self._zz_direction == -1:
            # Trending down -- track running LOW
            if ct <= self._zz_extreme_val_ticks:
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
                return
            # Reversal candidate
            if (ct - self._zz_extreme_val_ticks) >= min_rev \
                    and (i - self._zz_extreme_idx) >= MIN_BARS_BETWEEN_PIVOTS:
                # CONFIRM LOW as pivot. Next leg is LONG.
                self._this_bar_rtrig_dir = 'long'
                self._this_bar_rtrig_price = close
                self._zz_last_pivot_idx = self._zz_extreme_idx
                self._zz_pivots.append(
                    (self._zz_extreme_ts, self._zz_extreme_price, 'low'))
                self._zz_direction = 1
                self._zz_extreme_val_ticks = ct
                self._zz_extreme_idx = i
                self._zz_extreme_ts = ts
                self._zz_extreme_price = close
            return

    # ──────────────────────────────────────────────────────────────────
    # TRAJECTORY TRACKING (per open position)
    # ──────────────────────────────────────────────────────────────────

    def _iter_positions(self, positions):
        """Yield all open positions (PositionView). Tolerates None."""
        if positions is None:
            return
        primary = getattr(positions, 'primary', None)
        if primary is not None:
            yield primary
        chains = getattr(positions, 'chains', [])
        for c in chains:
            yield c

    def _update_trajectory(self, pos, price: float,
                             high: float, low: float):
        """Track peak_favorable, peak_adverse since entry."""
        traj = self._pos_traj.get(pos.contract_id)
        if traj is None:
            # First time we see this position -- initialize. Snapshot the
            # current 5s-bar count so B9 can fire B9_K 5s-bars (25s) later.
            traj = _PosTraj(
                contract_id=pos.contract_id,
                entry_ts=pos.entry_ts,
                entry_price=pos.entry_price,
                leg_dir=pos.direction,
                peak_favorable=0.0,
                peak_adverse=0.0,
                b9_fired=False,
                entry_bar_count=self._bar_count,
            )
            self._pos_traj[pos.contract_id] = traj

        # Signed excursion vs entry, clamped at zero
        if traj.leg_dir == 'long':
            fav = high - traj.entry_price
            adv = traj.entry_price - low
        else:
            fav = traj.entry_price - low
            adv = high - traj.entry_price
        traj.peak_favorable = max(traj.peak_favorable, max(0.0, fav))
        traj.peak_adverse = max(traj.peak_adverse, max(0.0, adv))

    def cleanup_closed_position(self, contract_id: str):
        """Engine wrapper should call this when a position fully closes
        (post-FILL) so trajectory state doesn't leak."""
        self._pos_traj.pop(contract_id, None)

    def _record_skip(self, direction: Optional[str], reason: str,
                       pred_R: Optional[float], thr: Optional[float] = None):
        """Record a detected-but-skipped R-trigger signal for this bar."""
        self._skip_count += 1
        self._this_bar_skip = {
            'direction': direction or '?',
            'reason': reason,
            'pred_R': pred_R,
            'thr': thr,
            'skip_count': self._skip_count,
        }

    # ──────────────────────────────────────────────────────────────────
    # B7 INFERENCE -- entry sizing (binary at 1c: skip or take)
    # ──────────────────────────────────────────────────────────────────

    def _b7_query(self, ts: float,
                    v2_getter: Optional[Callable]) -> Optional[EntrySignal]:
        # This method is only called when an R-trigger fired this bar.
        # Any None return = a real entry signal was DETECTED but SKIPPED;
        # record it so the engine/dashboard can surface the reason.
        rtrig_dir = self._this_bar_rtrig_dir
        if v2_getter is None:
            logger.debug('B7: no V2 getter -- skipping entry')
            self._record_skip(rtrig_dir, 'no_v2_getter', None)
            return None
        v2 = v2_getter(ts)
        if v2 is None:
            logger.debug('B7: V2 warmup -- skipping entry')
            self._record_skip(rtrig_dir, 'v2_warmup', None)
            return None

        v2_cols = self._ctx.b7['v2_cols']
        from core_v2.features import FEATURE_NAMES as V2_NAMES
        # Build feature vector in the order B7 was trained on
        col_to_idx = {c: i for i, c in enumerate(V2_NAMES)}
        X = np.array([
            float(v2[col_to_idx[c]]) if not pd.isna(v2[col_to_idx[c]]) else 0.0
            for c in v2_cols
        ], dtype=np.float32).reshape(1, -1)
        try:
            pred_R = float(self._ctx.b7['model'].predict(X)[0])
            self._last_b7_pred = pred_R   # exposed for the dashboard
        except Exception as e:
            logger.exception(f'B7 inference failed: {e}')
            return None

        thr = self._b7_skip_threshold()
        if pred_R < thr:
            logger.info(f'B7 SKIP  pred_R={pred_R:.3f} < thr={thr}  '
                          f'dir={self._this_bar_rtrig_dir}')
            self._record_skip(rtrig_dir, 'b7_low_conviction', pred_R, thr)
            return None

        logger.info(f'B7 TAKE  pred_R={pred_R:.3f} >= thr={thr}  '
                      f'dir={self._this_bar_rtrig_dir}  mode={self._b10_mode}')
        return EntrySignal(
            tier='L5_ZIGZAG_RTRIG',
            direction=self._this_bar_rtrig_dir,
            cnn_flipped=False,
        )

    # ──────────────────────────────────────────────────────────────────
    # B9 INFERENCE -- during-trade CUT/HOLD at K=5
    # ──────────────────────────────────────────────────────────────────

    def _b9_query_full(self, pos, ts: float, current_close: float,
                         v2_getter: Optional[Callable]
                         ) -> Optional[PositionDecision]:
        if v2_getter is None:
            return None
        v2 = v2_getter(ts)
        if v2 is None:
            return None

        traj = self._pos_traj.get(pos.contract_id)
        if traj is None:
            return None

        feat_cols = self._ctx.b9['feat_cols']
        from core_v2.features import FEATURE_NAMES as V2_NAMES
        v2_idx = {c: i for i, c in enumerate(V2_NAMES)}

        leg_sign = +1.0 if traj.leg_dir == 'long' else -1.0
        pnl_pts = leg_sign * (current_close - traj.entry_price)
        mfe_pts = traj.peak_favorable
        mae_pts = traj.peak_adverse
        has_reached_R_against = float(
            (mae_pts >= (self._r_price or 0.0)) if self._r_price else 0.0)
        min_rev_ticks = float(
            (self._r_price / TICK_SIZE) if self._r_price else MIN_REV_TICKS_FLOOR)

        # Assemble in B9's expected column order
        traj_lookup = {
            'mae_pts_so_far': mae_pts,
            'mfe_pts_so_far': mfe_pts,
            'pnl_pts_so_far': pnl_pts,
            'pnl_usd_so_far': pnl_pts * DOLLAR_PER_POINT,
            'has_reached_R_against': has_reached_R_against,
            'min_rev_ticks': min_rev_ticks,
        }
        X_list = []
        for c in feat_cols:
            if c in traj_lookup:
                X_list.append(traj_lookup[c])
            elif c in v2_idx:
                v = v2[v2_idx[c]]
                X_list.append(float(v) if not pd.isna(v) else 0.0)
            else:
                X_list.append(0.0)
        X = np.array(X_list, dtype=np.float32).reshape(1, -1)

        try:
            pred_remaining = float(self._ctx.b9['model'].predict(X)[0])
            self._last_b9_pred = pred_remaining   # exposed for the dashboard
        except Exception as e:
            logger.exception(f'B9 inference failed: {e}')
            return None

        thr = self._b9_cut_threshold()
        if pred_remaining < thr:
            logger.info(f'B9 CUT  pred_remaining={pred_remaining:.1f} < '
                          f'thr={thr}  contract={pos.contract_id}  '
                          f'mae={mae_pts:.2f}  mfe={mfe_pts:.2f}  '
                          f'pnl={pnl_pts*DOLLAR_PER_POINT:.0f}$')
            return PositionDecision(
                contract_id=pos.contract_id,
                exit_reason='b9_cut',
            )
        logger.debug(f'B9 HOLD pred_remaining={pred_remaining:.1f} >= '
                       f'thr={thr}  contract={pos.contract_id}')
        return None
