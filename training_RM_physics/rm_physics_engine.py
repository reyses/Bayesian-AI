"""
Regression-Mean Physics Engine — Pivot-residual entry, asymmetric TP/SL exit.

Port of `tools/pivot_residual_forward.py`'s winning configuration into the
main forward-pass infrastructure (training_RM_physics/run_rm.py).

Winner from sweep: TP=$50, SL=$3, R_confirm=$5, min_res=0.5, inverse_thr=1.0
  → +$779/day combined (IS +$750, OOS +$897), 22% WR, RR 16.7:1

Signal chain:
  1. Zigzag on 1m price with R_confirm=$5 retracement detects pivots.
  2. At pivot confirmation bar T, read precomputed 1m_z_se at the PIVOT BAR
     (extreme_idx, historical — no lookahead).
  3. If |residual| < min_res_strength → skip (noise pivot).
  4. Direction: residual < 0 → LONG (price below RM → bet bounce up)
                residual > 0 → SHORT (price above RM → bet bounce down)
  5. Enter at T's close.

Exit (first trigger wins):
  - TP: pnl_pts ≥ tp_pts (→ +$100 at TP=$50)
  - SL: pnl_pts ≤ −sl_pts (→ −$6 at SL=$3... wait, pts × $2 = $)
  - INVERSE: live residual sign flipped from entry residual AND
            |live residual| ≥ inverse_threshold
  - EOD: 20:55 UTC force-close

One position at a time (chains can be enabled later).
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


# ── MNQ contract ──────────────────────────────────────────────────────
TICK = 0.25
TV = 0.50
DOLLAR_PER_POINT = 2.0

# ── Feature index ──────────────────────────────────────────────────────
N_CORE = 12
TF_1M = 1
_1M_Z_IDX = TF_1M * N_CORE + 0   # = 12 → 1m_z_se in feature vector

# ── Winning config (from TP/SL sweep) ─────────────────────────────────
# All in POINTS (MNQ: 1 pt = $2)
DEFAULT_R_CONFIRM_PTS = 2.5       # $5 retracement confirms a pivot
# TP/SL DISABLED by default — exits now driven by INVERSE (residual flip)
# and EOD / max-hold only. Using very large values so price never reaches
# these levels. Remove the direction-asymmetric geometry problem.
DEFAULT_TP_PTS = 9999.0           # effectively disabled
DEFAULT_SL_PTS = 9999.0           # effectively disabled
DEFAULT_MAX_HOLD_MIN = 120        # safety: force-close after 2 hours
# Slippage: adverse fill displacement as fraction of current 1s bar range.
# 0.5 = fill can be up to half the 1s bar range worse than theoretical.
DEFAULT_SLIPPAGE_FACTOR = 0.5
DEFAULT_SLIPPAGE_SEED = 42

# Flip rule (T34): if trade is adverse by > flip_threshold_pts within
# flip_window_sec of entry, close the position and open the opposite
# direction. Motivation: time-to-wrong diagnostic (2026-04-22) showed
# losers hit −$5 at median 14s vs winners at median 66s (~5× asymmetry).
DEFAULT_FLIP_ENABLED = False        # T36: flip hurt, disabling by default
DEFAULT_FLIP_WINDOW_SEC = 15
DEFAULT_FLIP_THRESHOLD_PTS = 2.5

# Phantom entry (T37): don't open at pivot confirmation — wait wait_bars 1m
# bars, then commit only if price moved favorably. Skips entries that
# immediately went adverse. If threshold = 0, any favorable tick confirms.
DEFAULT_PHANTOM_ENABLED = True
DEFAULT_PHANTOM_WAIT_BARS = 1       # wait 1 × 60s before deciding
DEFAULT_PHANTOM_MIN_FAVORABLE_PTS = 0.0  # any favorable move commits

# NN filter (T45): call a trained CNN at pivot confirmation with the 91D
# feature vector. Returns P(win) for the mean-reversion direction.
#   P(win) >= take_thr → TAKE direction as predicted (LONG at LOW pivot)
#   P(win) <= flip_thr → FLIP direction (trade opposite)
#   else              → SKIP (don't register an entry)
DEFAULT_NN_FILTER_ENABLED = True      # T45: enabled for end-to-end validation
DEFAULT_NN_MODEL_PATH = 'training_RM_physics/output/pivot_direction_cnn.pt'
DEFAULT_NN_TAKE_THR = 0.65        # high-confidence threshold (last try on CNN)
DEFAULT_NN_FLIP_THR = 0.0         # A: no-FLIP mode. Only TAKE (P≥0.65) or SKIP.
DEFAULT_MIN_RES = 0.5             # |residual| threshold for entry
DEFAULT_INVERSE_THR = 1.0         # |live residual| cross for inverse exit

# ── EOD ───────────────────────────────────────────────────────────────
EOD_UTC_SECONDS = 20 * 3600 + 55 * 60
ENTRY_CUTOFF_UTC_SECONDS = 20 * 3600 + 30 * 60


def _seconds_past_midnight(ts):
    return int(ts) % 86400


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════

class RMPhysicsEngine:
    """Pivot-residual entry + asymmetric TP/SL exit.

    Same signal chain as `tools/pivot_residual_forward.py`; wrapped to fit
    the run_rm.py harness (on_state API, chain-compatible position list,
    get_full_trades output schema)."""

    def __init__(self,
                 r_confirm_pts: float = DEFAULT_R_CONFIRM_PTS,
                 tp_pts: float = DEFAULT_TP_PTS,
                 sl_pts: float = DEFAULT_SL_PTS,
                 min_res_strength: float = DEFAULT_MIN_RES,
                 inverse_threshold: float = DEFAULT_INVERSE_THR,
                 slippage_factor: float = DEFAULT_SLIPPAGE_FACTOR,
                 slippage_seed: int = DEFAULT_SLIPPAGE_SEED,
                 flip_enabled: bool = DEFAULT_FLIP_ENABLED,
                 flip_window_sec: int = DEFAULT_FLIP_WINDOW_SEC,
                 flip_threshold_pts: float = DEFAULT_FLIP_THRESHOLD_PTS,
                 phantom_enabled: bool = DEFAULT_PHANTOM_ENABLED,
                 phantom_wait_bars: int = DEFAULT_PHANTOM_WAIT_BARS,
                 phantom_min_favorable_pts: float = DEFAULT_PHANTOM_MIN_FAVORABLE_PTS,
                 nn_filter_enabled: bool = DEFAULT_NN_FILTER_ENABLED,
                 nn_model_path: str = DEFAULT_NN_MODEL_PATH,
                 nn_take_thr: float = DEFAULT_NN_TAKE_THR,
                 nn_flip_thr: float = DEFAULT_NN_FLIP_THR,
                 max_hold_min: int = DEFAULT_MAX_HOLD_MIN,
                 max_chains: int = 1,
                 only_tier: str = None,              # API-parity kwarg
                 honor_per_tier_caps: bool = True): # API-parity kwarg
        import random as _random
        self.r_confirm_pts = float(r_confirm_pts)
        self.tp_pts = float(tp_pts)
        self.sl_pts = float(sl_pts)
        self.min_res_strength = float(min_res_strength)
        self.inverse_threshold = float(inverse_threshold)
        self.slippage_factor = float(slippage_factor)
        self._rng = _random.Random(int(slippage_seed))
        self.flip_enabled = bool(flip_enabled)
        self.flip_window_sec = int(flip_window_sec)
        self.flip_threshold_pts = float(flip_threshold_pts)
        self.phantom_enabled = bool(phantom_enabled)
        self.phantom_wait_bars = int(phantom_wait_bars)
        self.phantom_min_favorable_pts = float(phantom_min_favorable_pts)
        self.max_hold_min = int(max_hold_min)
        self.max_chains = max(1, int(max_chains))
        # Pending phantom state (one at a time): dict or None
        self._pending_entry = None

        # NN filter (lazy load: only when enabled)
        self.nn_filter_enabled = bool(nn_filter_enabled)
        self.nn_take_thr = float(nn_take_thr)
        self.nn_flip_thr = float(nn_flip_thr)
        self._nn_model = None
        self._nn_mean = None
        self._nn_std = None
        self._nn_device = None
        if self.nn_filter_enabled:
            from training_RM_physics.nn_direction import load_nn_filter
            (self._nn_model, self._nn_mean,
             self._nn_std, self._nn_device) = load_nn_filter(nn_model_path)

        # Position state
        self._positions: List[Dict] = []

        # 1m series tracked as each new 1m close arrives.
        # We need closes + residuals history so we can read residual at
        # the pivot bar (extreme_idx), which may be bars in the past.
        self._closes_1m: List[float] = []
        self._residuals_1m: List[float] = []
        self._ts_1m: List[int] = []
        self._last_1m_close_ts: Optional[int] = None

        # Zigzag pivot state on 1m closes
        self._leg_dir: Optional[str] = None   # None / 'up' / 'down'
        self._extreme_idx: int = 0
        self._extreme_price: Optional[float] = None

        # Trade accounting
        self._last_price = 0.0
        self._last_ts = 0
        self.trades: List[Dict] = []
        self.daily_pnl = 0.0

    # ── API-parity properties ────────────────────────────────────────
    @property
    def in_pos(self):
        return bool(self._positions)

    @property
    def direction(self):
        return self._positions[0]['direction'] if self._positions else None

    @property
    def entry_tier(self):
        return 'RM_PIVOT_TPSL'

    @property
    def n_positions(self):
        return len(self._positions)

    def set_sec_closes(self, sec_df):
        return

    def set_sec_prices(self, sec_df):
        return

    # ══════════════════════════════════════════════════════════════════
    # Main tick
    # ══════════════════════════════════════════════════════════════════

    def on_state(self, state: Dict):
        price = state['price']
        ts = int(state['timestamp'])
        feat = state['features']
        # 1s high/low for precise TP/SL; fall back to price if not provided.
        high = float(state.get('high', price))
        low = float(state.get('low', price))
        self._last_price = price
        self._last_ts = ts

        # Residual (1m_z_se) from the live feature row (nearest-past 5s).
        live_residual = float(feat[_1M_Z_IDX]) if feat is not None else float('nan')

        # EOD: close everything at the current price
        sec_today = _seconds_past_midnight(ts)
        if sec_today >= EOD_UTC_SECONDS and self._positions:
            self.force_close(reason='eod')
            return

        # Detect a new 1m close. bar_data refers to the MOST RECENTLY
        # COMPLETED 1m bar — no lookahead. When bar_data.timestamp
        # advances, a bar just completed and we pick up its close.
        bar_data = state.get('bar_data')
        new_1m_close = None
        if bar_data is not None:
            bd_ts = int(bar_data.get('timestamp', 0))
            if self._last_1m_close_ts is None or bd_ts > self._last_1m_close_ts:
                new_1m_close = float(bar_data['close'])
                self._last_1m_close_ts = bd_ts

        # Exits on open positions run EVERY tick with 1s high/low precision.
        self._check_exits(price, high, low, ts, live_residual, feat)

        # Phantom entry evaluation — if a signal is pending and we've
        # passed its wait window, decide whether to commit based on
        # favorable movement since the signal fired.
        if self._pending_entry is not None:
            p = self._pending_entry
            if ts >= p['decide_at_ts']:
                if p['direction'] == 'long':
                    favorable = price - p['signal_price']
                else:
                    favorable = p['signal_price'] - price
                if favorable >= self.phantom_min_favorable_pts \
                        and len(self._positions) < self.max_chains:
                    self._open(p['direction'], price, ts, feat, p['entry_residual'])
                self._pending_entry = None

        # Zigzag + entry only fire on a new 1m close.
        if new_1m_close is None:
            return

        # Append to our 1m history
        self._closes_1m.append(new_1m_close)
        self._residuals_1m.append(live_residual)
        self._ts_1m.append(ts)
        T = len(self._closes_1m) - 1  # index of this new close

        if self._extreme_price is None:
            self._extreme_price = new_1m_close
            self._extreme_idx = T
            return

        # Zigzag update + pivot confirmation detection
        pivot_confirmed = False
        pivot_idx = None

        if self._leg_dir is None:
            if new_1m_close - self._extreme_price >= self.r_confirm_pts:
                self._leg_dir = 'up'
                pivot_idx = self._extreme_idx   # the LOW we're leaving
                pivot_confirmed = True
                self._extreme_idx = T
                self._extreme_price = new_1m_close
            elif self._extreme_price - new_1m_close >= self.r_confirm_pts:
                self._leg_dir = 'down'
                pivot_idx = self._extreme_idx   # the HIGH we're leaving
                pivot_confirmed = True
                self._extreme_idx = T
                self._extreme_price = new_1m_close
            else:
                # Still tracking extreme
                if new_1m_close > self._extreme_price or new_1m_close < self._extreme_price:
                    self._extreme_idx = T
                    self._extreme_price = new_1m_close
        elif self._leg_dir == 'up':
            if new_1m_close > self._extreme_price:
                self._extreme_idx = T
                self._extreme_price = new_1m_close
            elif self._extreme_price - new_1m_close >= self.r_confirm_pts:
                pivot_idx = self._extreme_idx   # HIGH pivot confirmed
                pivot_confirmed = True
                self._leg_dir = 'down'
                self._extreme_idx = T
                self._extreme_price = new_1m_close
        elif self._leg_dir == 'down':
            if new_1m_close < self._extreme_price:
                self._extreme_idx = T
                self._extreme_price = new_1m_close
            elif new_1m_close - self._extreme_price >= self.r_confirm_pts:
                pivot_idx = self._extreme_idx   # LOW pivot confirmed
                pivot_confirmed = True
                self._leg_dir = 'up'
                self._extreme_idx = T
                self._extreme_price = new_1m_close

        # Entry check
        if (pivot_confirmed and pivot_idx is not None
                and len(self._positions) < self.max_chains
                and sec_today < ENTRY_CUTOFF_UTC_SECONDS):
            r_at_pivot = self._residuals_1m[pivot_idx] if pivot_idx < len(self._residuals_1m) else float('nan')
            if not np.isnan(r_at_pivot) and abs(r_at_pivot) >= self.min_res_strength:
                # Mean-reversion direction (original):
                #   residual < 0 (price below RM at pivot) → LONG
                #   residual > 0 (price above RM at pivot) → SHORT
                direction = 'long' if r_at_pivot < 0 else 'short'

                # NN filter: predict P(win) for the proposed direction.
                # Output regimes:  take / flip / skip
                if self.nn_filter_enabled and self._nn_model is not None:
                    from training_RM_physics.nn_direction import predict_pwin
                    try:
                        prob = predict_pwin(feat, self._nn_model,
                                             self._nn_mean, self._nn_std,
                                             self._nn_device)
                    except Exception:
                        prob = 0.5
                    if prob <= self.nn_flip_thr:
                        direction = 'short' if direction == 'long' else 'long'
                    elif prob < self.nn_take_thr:
                        # SKIP — don't register entry at all
                        return

                if self.phantom_enabled:
                    self._pending_entry = {
                        'direction': direction,
                        'signal_ts': ts,
                        'signal_price': new_1m_close,
                        'entry_residual': r_at_pivot,
                        'decide_at_ts': ts + self.phantom_wait_bars * 60,
                    }
                else:
                    self._open(direction, new_1m_close, ts, feat, r_at_pivot)

    def _check_exits(self, price, high, low, ts, live_residual, feat):
        """Evaluate TP / SL / inverse exit using 1s high/low for precision.

        For a LONG: TP fires when high ≥ entry + tp_pts; SL when low ≤
        entry − sl_pts. Symmetric for SHORT. Fills snap to the exact
        level (limit/stop order model, conservative)."""
        if not self._positions:
            return

        closed = []
        flip_requests = []  # (direction_to_open, entry_price, entry_ts, feat, residual)
        for i, pos in enumerate(self._positions):
            if pos['direction'] == 'long':
                pnl_pts_current = price - pos['entry_price']
                tp_hit = (high - pos['entry_price']) >= self.tp_pts
                sl_hit = (low - pos['entry_price']) <= -self.sl_pts
                # Flip trigger: low hit flip threshold in the window
                flip_adverse_hit = (low - pos['entry_price']) <= -self.flip_threshold_pts
            else:
                pnl_pts_current = pos['entry_price'] - price
                tp_hit = (pos['entry_price'] - low) >= self.tp_pts
                sl_hit = (pos['entry_price'] - high) >= self.sl_pts
                flip_adverse_hit = (pos['entry_price'] - high) >= self.flip_threshold_pts
            pnl_dollars = pnl_pts_current * DOLLAR_PER_POINT
            pos['bars_held'] = max(0, (ts - pos['entry_ts']) // 60)
            pos['peak_pnl'] = max(pos['peak_pnl'], pnl_dollars)

            # ── Flip check (runs BEFORE TP/SL so we act on the first
            #    adverse threshold cross inside the flip window) ──
            trade_age_sec = ts - pos['entry_ts']
            if (self.flip_enabled
                    and not pos.get('has_flipped', False)
                    and trade_age_sec <= self.flip_window_sec
                    and flip_adverse_hit):
                # Close at the flip-threshold price (model: stop-like fill
                # at the threshold level with adverse slippage).
                flip_close_price = (pos['entry_price'] - self.flip_threshold_pts
                                     if pos['direction'] == 'long'
                                     else pos['entry_price'] + self.flip_threshold_pts)
                bar_range_pts = max(0.0, high - low)
                slip = self._rng.uniform(0.0, self.slippage_factor * bar_range_pts)
                if pos['direction'] == 'long':
                    flip_close_price -= slip
                else:
                    flip_close_price += slip
                self._close(pos, flip_close_price, ts, feat, 'FLIP')
                closed.append(i)
                # Queue the flipped entry (opposite direction) at current price
                opposite = 'short' if pos['direction'] == 'long' else 'long'
                flip_requests.append((opposite, price, ts, feat,
                                       -pos.get('entry_residual', 0.0)))
                continue

            exit_reason = None
            exit_price = price
            # When BOTH TP and SL are hit in the same 1s bar, assume SL
            # fires first (conservative). Real fill order is unknowable
            # without sub-second data.
            # Max-hold safety: force-close if trade has been open too long.
            # Runs before TP/SL so it always fires when time expires,
            # regardless of price.
            if pos['bars_held'] >= self.max_hold_min:
                self._close(pos, price, ts, feat, 'MAX_HOLD')
                closed.append(i)
                continue

            if sl_hit:
                exit_reason = 'SL'
                exit_price = pos['entry_price'] + (-self.sl_pts if pos['direction'] == 'long' else self.sl_pts)
                # Adverse slippage: up to slippage_factor × current 1s
                # bar range, worse than theoretical fill
                bar_range_pts = max(0.0, high - low)
                slip = self._rng.uniform(0.0, self.slippage_factor * bar_range_pts)
                if pos['direction'] == 'long':
                    exit_price -= slip
                else:
                    exit_price += slip
            elif tp_hit:
                exit_reason = 'TP'
                exit_price = pos['entry_price'] + (self.tp_pts if pos['direction'] == 'long' else -self.tp_pts)
                # Adverse slippage on TP too (conservative — in reality
                # limit orders often fill at the limit, but we stress test)
                bar_range_pts = max(0.0, high - low)
                slip = self._rng.uniform(0.0, self.slippage_factor * bar_range_pts)
                if pos['direction'] == 'long':
                    exit_price -= slip
                else:
                    exit_price += slip
            else:
                # Inverse: live residual sign flipped from entry residual
                entry_r = pos['entry_residual']
                if (not np.isnan(live_residual)
                        and np.sign(live_residual) != np.sign(entry_r)
                        and abs(live_residual) >= self.inverse_threshold):
                    exit_reason = 'INVERSE'

            if exit_reason is not None:
                self._close(pos, exit_price, ts, feat, exit_reason)
                closed.append(i)
        for i in reversed(closed):
            self._positions.pop(i)
        # Open flipped positions (after closures to keep max_chains check clean).
        for direction, fp, fts, ffeat, fres in flip_requests:
            if len(self._positions) < self.max_chains:
                new_pos = {
                    'direction': direction,
                    'entry_price': float(fp),
                    'entry_ts': int(fts),
                    'entry_residual': float(fres),
                    'entry_tier': 'RM_PIVOT_TPSL_FLIPPED',
                    'entry_79d': ffeat.copy() if hasattr(ffeat, 'copy') else np.asarray(ffeat),
                    'bars_held': 0,
                    'peak_pnl': 0.0,
                    'chain_idx': len(self._positions),
                    'has_flipped': True,   # prevent re-flipping
                }
                self._positions.append(new_pos)

    # ══════════════════════════════════════════════════════════════════
    # Position lifecycle
    # ══════════════════════════════════════════════════════════════════

    def _open(self, direction, price, ts, feat, entry_residual):
        pos = {
            'direction': direction,
            'entry_price': price,
            'entry_ts': ts,
            'entry_residual': float(entry_residual),
            'entry_tier': 'RM_PIVOT_TPSL',
            'entry_79d': feat.copy() if hasattr(feat, 'copy') else np.asarray(feat),
            'bars_held': 0,
            'peak_pnl': 0.0,
            'chain_idx': len(self._positions),
        }
        self._positions.append(pos)

    def _close(self, pos, price, ts, feat, exit_reason):
        if pos['direction'] == 'long':
            pnl = (price - pos['entry_price']) * DOLLAR_PER_POINT
        else:
            pnl = (pos['entry_price'] - price) * DOLLAR_PER_POINT
        self.daily_pnl += pnl

        time_str = datetime.utcfromtimestamp(pos['entry_ts']).strftime('%H:%M')
        entry_79d = pos['entry_79d']
        self.trades.append({
            'trade_id': len(self.trades),
            'time': time_str,
            'timestamp': pos['entry_ts'],
            'dir': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'held': pos['bars_held'],
            'peak': pos['peak_pnl'],
            'entry_tier': 'RM_PIVOT_TPSL',
            'exit_reason': exit_reason,
            'chain_idx': pos.get('chain_idx', 0),
            'entry_79d': entry_79d.tolist() if hasattr(entry_79d, 'tolist') else list(entry_79d),
            'entry_residual': pos.get('entry_residual', 0.0),
        })

    def force_close(self, reason='end_of_day'):
        while self._positions:
            pos = self._positions[0]
            self._close(pos, self._last_price, self._last_ts, None, reason)
            self._positions.pop(0)

    def reset(self):
        self._positions = []
        self._pending_entry = None
        self._closes_1m = []
        self._residuals_1m = []
        self._ts_1m = []
        self._last_1m_close_ts = None
        self._leg_dir = None
        self._extreme_idx = 0
        self._extreme_price = None
        self._last_price = 0.0
        self._last_ts = 0
        self.daily_pnl = 0.0

    def get_full_trades(self):
        return self.trades

    def summary(self):
        n = len(self.trades)
        if n == 0:
            return 'No trades'
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = sum(t['pnl'] for t in self.trades)
        return f'{n} trades | WR={wins/n*100:.0f}% | ${total:+.0f}'


# ══════════════════════════════════════════════════════════════════════
# Compat shims for run_rm.py
# ══════════════════════════════════════════════════════════════════════

TIER_PRIORITY = ['RM_PIVOT_TPSL']
TIER_MAP = {'RM_PIVOT_TPSL': None}
MAX_CHAINS_PER_TIER = {'RM_PIVOT_TPSL': 1}

IsoEngine = RMPhysicsEngine
