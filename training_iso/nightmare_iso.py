"""
Isolated NightmareEngine — clean physics, 9 tiers, inverse-signal exits.

No trails, no stops, no 1s sensor, no giveback. Pure thesis in / thesis out:

  ENTRY (1m boundary, |z|>ROCHE):
    Priority cascade of 9 tier classifiers. First match wins.
      1. TREND_FOLLOWER — |1m_vel|>THR + 5m aligned + vr>1          → fade 1m_vel
      2. CASCADE        — wick rejection + 1h z aligned             → fade
      3. KILL_SHOT      — wick rejection, 1h NOT aligned            → fade
      4. RIDE_AGAINST   — 1h velocity opposes fade, 1h z not extreme→ ride 1h
      5. FADE_AGAINST   — 1h z extreme against fade + 5m quiet      → fade
      6. MTF_EXHAUSTION — 5m decel + 1m alive + z/vr/vol high       → ride 5m
      7. MTF_BREAKOUT   — 5m + 15m z aligned + dmi not against      → ride
      8. NMP_FADE       — default stable fade (vr<1)                → fade
      9. NMP_RIDE       — default chaotic ride (vr≥1), 1D-opposed   → ride

  EXIT (1m boundary):
    Inverse-signal rule. Each tier holds until its OWN entry condition
    fires for the opposite direction. KILL_SHOT long exits when KILL_SHOT
    short fires; FREIGHT_TRAIN long exits when FREIGHT_TRAIN short fires;
    etc. No protective stops. End-of-day force_close is the only fallback.

This is the bare physics baseline. Iterate gating/filtering per tier.
"""
import numpy as np
from typing import Dict
from datetime import datetime

TICK = 0.25
TV = 0.50

# ── Entry gate ─────────────────────────────────────────────────────────
ROCHE = 2.0       # |z| > ROCHE == Roche Limit == band edge
VR_ENTRY = 1.0    # vr < VR_ENTRY == stable; vr >= VR_ENTRY == chaotic

# ── Per-tier entry thresholds (basic form) ─────────────────────────────
# CASCADE / KILL_SHOT share wick thresholds; CASCADE adds 1h alignment
WICK_5M_MIN          = 0.83
WICK_15M_MIN         = 0.77
CASCADE_H1_Z_MIN     = 1.0    # CASCADE needs |1h_z| > this ALIGNED with fade

# TREND_FOLLOWER — strong 1m velocity + 5m aligned + chaotic regime (vr>1)
# Direction is FLIPPED: fade the velocity (exhaustion thesis), not ride it.
# ENTRY thresholds:
TF_VEL_MIN           = 10.0
TF_VR_MIN            = 1.0

# TREND_FOLLOWER peak-arrival exit (symmetric with entry thesis).
# Derived from the peak-signature EDA on 452 trades (2026-04-18):
#   1m_p_at_center:     entry 0.05 -> peak 0.40 (d/sigma = +10.32, THE signal)
#   1m_reversion_prob:  entry 0.49 -> peak 0.87 (d/sigma = +1.16)
#   1m_variance_ratio:  entry 1.32 -> peak 0.80 (d/sigma = -1.81, inverse of entry gate)
# All three fire together = "thesis complete": price at regression mean, OU
# model says reversion done, chaotic regime has settled.
TF_EXIT_P_CENTER_MIN   = 0.35
TF_EXIT_REVERSION_MIN  = 0.80
TF_EXIT_VR_MAX         = 1.0
# Peak-amplitude gate: the rule can only fire if a real peak has formed.
# Without this, the rule fires on 93% of trades at an average peak of $25
# with $33 giveback — intercepting trades that would otherwise run to
# their inverse exit at 91% WR / +$97/trade. Requiring peak_pnl >= $10
# means "we only exit AT peak if there WAS a peak."
TF_EXIT_MIN_PEAK_PNL   = 10.0
# Max hold time. Data on 733 trades shows the tier's entire edge lives in
# the first 60 minutes. Past that, every bucket is a loser (60-120m at
# -$32/tr, 300+m at -$309/tr on 19% WR). The fade thesis has a natural
# timescale of ~15-30 min; trades past 60 min are thesis-violated.
# Framed as physics, not safety: "the fade doesn't happen → thesis dead."
TF_EXIT_MAX_HOLD_MIN   = 60

# RIDE_AGAINST — 1h velocity opposes fade direction (sweep optimum = 2.0)
RA_H1_VEL_MIN        = 2.0
# Entry filter: require 5m_bar_range above threshold. EDA on 6,112 trades
# (2026-04-18) showed peak-reachers have 5m_bar_range median 67 vs 49 for
# non-reachers (Cohen d = +0.33 at 5m, +0.32 at 15m, similar across TFs).
# Physics: the 1h-velocity-reversal thesis needs volatile market to play
# out; quiet market = 1h_vel signal is spurious.
RA_ENTRY_MIN_5M_BAR_RANGE = 55.0

# Phantom entry config. When conditions fire we WAIT for price to move
# PHANTOM_CONFIRM_TICKS in our direction within PHANTOM_MAX_WAIT_MIN before
# committing capital. Trades off entry-price (worse) for fizzle protection
# (cut entries that would've never printed). Per-tier dict; tiers not in
# the dict enter immediately as before.
#   Key = tier name; Value = (confirm_ticks, max_wait_minutes)
PHANTOM_ENTRY = {
    'RIDE_AGAINST':  (4, 2),   # 4-tick ($2) confirm within 2 min (sweep optimum)
    'MTF_BREAKOUT':  (4, 2),   # flip+phantom: -$1,116 -> -$96 (+$1,020)
    'KILL_SHOT':     (4, 2),   # phantom: -$708 -> -$345 (+$363)
    # TREND_FOLLOWER: phantom HURT (+$495 -> -$1,814). When phantom-confirmed
    # trades fail, the 60m timeout accumulates -$166/tr of loss. Phantom
    # doesn't universally help — tiers with long natural timescales need
    # either (a) no phantom or (b) phantom with shorter timeout.
}

# RIDE_AGAINST peak-arrival exit (same physics as TREND_FOLLOWER — price
# returns to regression mean). Derived from Q3 EDA on 1,916 peak-bearing
# trades (2026-04-18):
#   1m_p_at_center:     entry 0.07 -> peak 0.46 (d/sigma = +12.38, dominant)
#   1m_reversion_prob:  entry 0.67 -> peak 0.93 (d/sigma = +1.01)
#   1m_variance_ratio:  entry ~mid -> peak lower (weaker signal here but
#                       kept symmetric with TREND_FOLLOWER rule template)
# Round 1: no amplitude gate (measure, add later if too eager).
RA_EXIT_P_CENTER_MIN   = 0.35
RA_EXIT_REVERSION_MIN  = 0.80
RA_EXIT_VR_MAX         = 1.0
# Amplitude gate (Round 2): without it the rule fires on 99.6% of trades
# at mean peak $13 with $13.5 giveback — same premature-exit bug
# TREND_FOLLOWER had. Winners peaked at $213 average, so peak>=$10 doesn't
# threaten winners; it just gates out the small-peak early exits.
RA_EXIT_MIN_PEAK_PNL   = 10.0
# Max-hold timeout (Round 3): hold-time cliff analysis shows RIDE_AGAINST's
# entire edge is in the first 15 minutes. 0-15m bucket: 84% WR, +$17/tr,
# +$2.28/min. 15-30m: near break-even (-$3). 30+m: every bucket a loser
# (-$47 at 30-60m, -$56 at 60-90m). Physics: if the 1h-velocity-reversal
# thesis doesn't pay off in 15 min, the reversal was noise, not signal.
# 4x faster timescale than TREND_FOLLOWER's 60min fade exhaustion.
RA_EXIT_MAX_HOLD_MIN   = 15

# KILL_SHOT peak-arrival exit. Same physics (price to center) as the other
# fade tiers. Q3 EDA on 435 peak-bearing trades (2026-04-18):
#   1m_p_at_center:    entry 0.07 -> peak 0.48 (d/sigma = +12.63, dominant)
#   1m_reversion_prob: entry 0.66 -> peak 0.93 (d/sigma = +1.03)
#   5m_wick_ratio:     entry 0.92 -> peak 0.59 (d/sigma = -6.40) — entry
#                      wick signal decays at peak (KILL_SHOT-specific, but
#                      we don't use it as exit since 1m_vr is already in
#                      template and Cohen-d at 15s/1m vr is close).
# No timeout: KILL_SHOT has BIMODAL hold distribution — 0-10m (70% WR,
# +$6/tr) AND 300+m (54% WR, +$25/tr) both win. Kill zone is 60-300m.
# A straight timeout would kill the 300+m recovery bucket. Let peak rule
# handle it at both timescales.
KS_EXIT_P_CENTER_MIN   = 0.35
KS_EXIT_REVERSION_MIN  = 0.80
KS_EXIT_VR_MAX         = 1.0
KS_EXIT_MIN_PEAK_PNL   = 10.0
# Max-hold timeout (Round 2): with peak rule on, the prior bimodal
# distribution collapses — distribution becomes monotonic decay past 30m.
# 0-10m: 87% WR, 10-30m: 77% WR (both above user's 70% quick-wins threshold).
# 30-60m: 57% WR -$8/tr. 60+m: losers. Cut at 30m — aligns with
# "quick-wins-above-70%" principle.
KS_EXIT_MAX_HOLD_MIN   = 30

# KILL_SHOT_INVERSE peak-cluster exits (GMM on 170 REAL-bucket winners,
# 2026-04-19). Two clusters surfaced — same physics signature mirrored on
# opposite directional sides. Winners peak at strong directional extension:
# |15s_dmi_diff| large + |5m_z_se| large + |1m_dmi_diff| large.
#
# Thresholds = cluster centroid values (mean feature value within cluster).
# Long side (N=97): dmi_15s > 12.5, z_5m > 0.79, dmi_1m > 10.0
# Short side (N=73): dmi_15s < -10.5, z_5m < -0.91, dmi_1m < -9.25
# Asymmetric thresholds reflect observed cluster centers, not symmetric
# by design — the short side clustered at slightly looser extension.
KSI_CLUSTER_LONG_DMI15S  = 12.5
KSI_CLUSTER_LONG_Z5M     = 0.79
KSI_CLUSTER_LONG_DMI1M   = 10.0
KSI_CLUSTER_SHORT_DMI15S = -10.5
KSI_CLUSTER_SHORT_Z5M    = -0.91
KSI_CLUSTER_SHORT_DMI1M  = -9.25

# KILL_SHOT no-progress cut (Q2 cliff, 2026-04-19 EDA).
# Winner p90 time-to-$5 = 12 bars. If trade hasn't shown $5 of peak progress
# by bar 15, thesis is dead — cut before the 30-bar timeout takes full damage.
# Data (current direction, 301 trades): 47% winners <$5 / 72% losers <$5 at
# bar 15 (25pp separation). Same rule applies to KILL_SHOT_INVERSE (parallel
# tier, opposite direction on same setup — whichever direction is wrong
# gets cut fast; whichever is right captures peak-rule exit).
KS_EXIT_CUT_BAR         = 15
KS_EXIT_CUT_PEAK_MAX    = 5.0

# NMP_FADE peak-arrival exit. Default catch-all fade tier; direction is
# mostly right (52.7% peak>$20, lowest selectivity across tiers). Q3 EDA
# on 7,118 peak-bearing trades (2026-04-18) gives universal template:
#   1m_p_at_center:     entry 0.07 -> peak 0.37  (d/sigma = +9.93, dominant)
#   1m_reversion_prob:  entry 0.71 -> peak 0.88  (d/sigma = +0.81)
#   1m_variance_ratio:  entry <1.0 (by entry gate), stays ~same at peak
#                       (weak signal for this tier, no-op as exit check).
# Q2 hold-time cliff at 15 min (same as RIDE_AGAINST): 5-15m bucket is
# 76% WR at +$22.79/tr; past 15m is break-even or losing.
NF_EXIT_P_CENTER_MIN   = 0.35
NF_EXIT_REVERSION_MIN  = 0.80
NF_EXIT_VR_MAX         = 1.0    # no-op (always true for NMP_FADE entries)
NF_EXIT_MIN_PEAK_PNL   = 10.0
NF_EXIT_MAX_HOLD_MIN   = 15

# MTF_BREAKOUT peak-arrival exit. Trend tier (rides multi-TF z-aligned
# breakouts) but the peak signature is UNIVERSAL — p_center spikes when
# price is about to revert, even in trends. Q3 on 807 peak-bearing trades:
#   1m_p_at_center:    entry 0.07 -> peak 0.48 (d/sigma = +12.83, dominant)
#   1m_reversion_prob: entry 0.67 -> peak 0.93 (d/sigma = +1.05)
# NO timeout: Q2 shows 0-60m losers, 120m+ winners. Trends need time.
# The inverse-signal exit is too slow (mean 109m, peak $98 -> give $110
# -> close -$12). Peak rule should fire mid-pullback, capturing peak $98.
MB_EXIT_P_CENTER_MIN   = 0.35
MB_EXIT_REVERSION_MIN  = 0.80
MB_EXIT_VR_MAX         = 1.0
MB_EXIT_MIN_PEAK_PNL   = 10.0

# FADE_AGAINST — 1h z extreme against fade + 5m quiet
FA_H1_Z_MIN          = 1.5
FA_5M_VEL_MAX        = 10.0

# MTF_EXHAUSTION — 5m turning while 1m still moving
MTFE_Z_MIN           = 2.0
MTFE_VR_MIN          = 1.0
MTFE_VOL_MIN         = 1.0
MTFE_5M_VEL_MIN      = 10.0
MTFE_1M_VEL_MIN      = 5.0

# MTF_BREAKOUT — multi-TF z aligned + dmi not strongly against
MTFB_Z_MIN           = 1.3
MTFB_DMI_AGAINST_MAX = 5.0

# NMP_RIDE 1D-alignment gate (drop trades aligned with yesterday's 1D)
NMP_RIDE_REQUIRE_OPPOSED_1D = True

# ── Entry time cutoff ──────────────────────────────────────────────────
# No new entries in the last 10 minutes of the session. Trades entered
# at 23:59 can't possibly see their inverse signal fire before EOD
# force-close, so they're pre-doomed. Global rule, applies to all tiers.
# Measured in UTC minutes-since-midnight.
ENTRY_CUTOFF_MINUTES_UTC = 23 * 60 + 50   # 23:50 UTC

# ── 91D feature indices ────────────────────────────────────────────────
# Layout: 6 TFs * 12 core + 6 TFs * 3 helper + 1 global = 91
# TF order: 15s(0), 1m(1), 5m(2), 15m(3), 1h(4), 1D(5)
N_CORE = 12
HELPER_START = 72
N_HELPER = 3

# Per-core slot
_Z         = 0
_DMI       = 1
_VR        = 2
_VELOCITY  = 3
_ACCEL     = 4
_VOL_REL   = 5
_HURST     = 7
_REVERSION = 8
_P_CENTER  = 9

def _core(tf, slot): return tf * N_CORE + slot
def _help(tf, slot): return HELPER_START + tf * N_HELPER + slot

TF_15S, TF_1M, TF_5M, TF_15M, TF_1H, TF_1D = 0, 1, 2, 3, 4, 5
HELPER_WICK = 2     # helper slot 2 = wick_ratio
HELPER_DIRVOL = 1   # helper slot 1 = dir_vol

# Pre-computed indices
_15S_DMI_IDX      = _core(TF_15S, _DMI)      # 1  (for KSI cluster exits)
_1M_Z_IDX         = _core(TF_1M, _Z)         # 12
_1M_DMI_IDX       = _core(TF_1M, _DMI)       # 13
_1M_VR_IDX        = _core(TF_1M, _VR)        # 14
_1M_VEL_IDX       = _core(TF_1M, _VELOCITY)  # 15
_1M_ACCEL_IDX     = _core(TF_1M, _ACCEL)     # 16
_1M_VOL_REL_IDX   = _core(TF_1M, _VOL_REL)   # 17
_1M_REVERSION_IDX = _core(TF_1M, _REVERSION) # 20
_1M_P_CENTER_IDX  = _core(TF_1M, _P_CENTER)  # 21
_5M_Z_IDX         = _core(TF_5M, _Z)         # 24
_5M_VEL_IDX       = _core(TF_5M, _VELOCITY)  # 27
_5M_ACCEL_IDX     = _core(TF_5M, _ACCEL)     # 28
_5M_BAR_RANGE_IDX = _core(TF_5M, 6)          # 30 (core[6] = bar_range)
_15M_Z_IDX        = _core(TF_15M, _Z)        # 36
_1H_Z_IDX         = _core(TF_1H, _Z)         # 48
_1H_VEL_IDX       = _core(TF_1H, _VELOCITY)  # 51
_1M_WICK_IDX      = _help(TF_1M, HELPER_WICK)    # 77
_5M_WICK_IDX      = _help(TF_5M, HELPER_WICK)    # 80
_15M_WICK_IDX     = _help(TF_15M, HELPER_WICK)   # 83
_1H_WICK_IDX      = _help(TF_1H, HELPER_WICK)    # 86
_1D_DIR_VOL_IDX   = _help(TF_1D, HELPER_DIRVOL)  # 88

# Tier priority (first match wins in cascade mode; isolated mode ignores)
TIER_PRIORITY = [
    'TREND_FOLLOWER',
    'CASCADE',
    'KILL_SHOT',
    'KILL_SHOT_INVERSE',
    'RIDE_AGAINST',
    'FADE_AGAINST',
    'MTF_EXHAUSTION',
    'MTF_BREAKOUT',
    'NMP_FADE',
    'NMP_RIDE',
]

TIER_MAP = {
    'TREND_FOLLOWER':    9,
    'CASCADE':           8,
    'KILL_SHOT':         7,
    'KILL_SHOT_INVERSE': 6,
    'RIDE_AGAINST':      5,
    'FADE_AGAINST':      4,
    'MTF_EXHAUSTION':    3,
    'MTF_BREAKOUT':      2,
    'NMP_RIDE':          1,
    'NMP_FADE':          0,
}


class IsoEngine:
    """Pure physics engine: 9-tier priority cascade, inverse-signal exits.

    Supports chain positions: up to `max_chains` concurrent positions per
    tier engine. Each position is independent — own entry price, own peak
    tracking, own exit rule check. A new chain opens when the tier's entry
    condition fires again while already in a position AND the direction
    matches existing positions AND we're under the cap. Inverse-signal
    direction (opposite of existing) doesn't chain — it naturally triggers
    exit logic on the open positions.
    """

    # 5s slope window (bars). 12 bars = last 60 seconds.
    # EDA (2026-04-18) showed d=-0.31 separator for TREND_FOLLOWER winners vs
    # tail losers at this window. Kept as a general tool available to any tier.
    SLOPE_WINDOW_BARS = 12

    def __init__(self, only_tier: str = None, max_chains: int = 4):
        self.only_tier = only_tier
        if only_tier is not None and only_tier not in TIER_MAP:
            raise ValueError(f'only_tier must be in {list(TIER_MAP)} or None')
        self.max_chains = max(1, int(max_chains))

        # Chain-aware state: list of open position dicts. Each has
        # direction, entry_price, entry_ts, entry_tier, entry_79d,
        # entry_abs_z, entry_approach, bars_held, peak_pnl, trade_path.
        self._positions = []

        # Phantom entry pending slot (at most one pending at a time).
        # None when no pending. Dict when waiting for confirmation:
        #   {tier, direction, signal_price, signal_ts, signal_z, entry_feat,
        #    confirm_ticks, max_wait_min}
        self._pending = None

        self._approach_buffer = []
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0

        # ── Slope infrastructure ──────────────────────────────────────
        # 5s close buffer for OLS-β (velocity of regression mean) computation.
        # Loaded per-day via set_sec_closes(). When absent, _slope_1m returns 0.
        # No tier consumes it yet — available for future gates when EDA shows
        # a tier's winners/losers separate on slope magnitude.
        self._sec_ts = None         # np.int64 array of 5s bar timestamps
        self._sec_close = None      # np.float64 array of aligned closes
        self._sec_cursor = 0        # monotonic search cursor (resets per day)

    # ── Backward-compat properties (legacy callers may still read these) ──
    @property
    def in_pos(self):
        return bool(self._positions)

    @property
    def direction(self):
        return self._positions[0]['direction'] if self._positions else None

    @property
    def entry_tier(self):
        return self._positions[0]['entry_tier'] if self._positions else None

    @property
    def n_positions(self):
        return len(self._positions)

    def set_sec_closes(self, sec_df):
        """Load the day's 5s OHLCV for slope (β) computation.

        Call once per day before ticking. Pass None or empty DataFrame to
        disable slope (the _slope_1m helper will return 0 cleanly).
        """
        if sec_df is None or len(sec_df) == 0:
            self._sec_ts = None
            self._sec_close = None
        else:
            df = sec_df.sort_values('timestamp')
            self._sec_ts = df['timestamp'].values.astype(np.int64)
            self._sec_close = df['close'].values.astype(np.float64)
        self._sec_cursor = 0

    # Legacy name kept for callers that may still reference it; no-op now.
    def set_sec_prices(self, sec_df):
        return

    def _slope_1m(self, entry_ts):
        """OLS β on the last SLOPE_WINDOW_BARS 5s closes ending at entry_ts.

        Units: price per 5s bar. Multiply by 12 for ticks/minute equivalent.
        Returns 0.0 if 5s data isn't loaded or there's insufficient history.

        Physical meaning: velocity of the regression mean at 1-minute horizon.
        |β| large = strong trend, sign(β) = direction. Entirely separate from
        the close-to-close `velocity` feature (which is a single-bar derivative
        — noisy). Slope is the OLS-fit drift across the full window.
        """
        if self._sec_ts is None or self._sec_close is None:
            return 0.0
        n_bars = self.SLOPE_WINDOW_BARS
        # Monotonic cursor: entry timestamps arrive in order across a day.
        ts_int = int(entry_ts)
        arr = self._sec_ts
        cursor = self._sec_cursor
        while cursor < len(arr) and arr[cursor] <= ts_int:
            cursor += 1
        self._sec_cursor = cursor
        # cursor now points to first bar strictly AFTER entry_ts.
        # Last bar at/before entry_ts is cursor-1.
        last_idx = cursor - 1
        if last_idx < n_bars - 1:
            return 0.0
        y = self._sec_close[last_idx - n_bars + 1: last_idx + 1]
        x = np.arange(n_bars, dtype=np.float64)
        xm = x.mean()
        ym = y.mean()
        dx = x - xm
        denom = float((dx * dx).sum())
        if denom < 1e-9:
            return 0.0
        return float((dx * (y - ym)).sum() / denom)

    # ══════════════════════════════════════════════════════════════════
    # Main state callback
    # ══════════════════════════════════════════════════════════════════

    def on_state(self, state: Dict):
        self._bar_count += 1
        feat = state['features']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price

        is_1m = (int(ts) % 60) < 5
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')

        z = feat[_1M_Z_IDX]
        vr = feat[_1M_VR_IDX]

        # Approach buffer when no open positions (for entry context).
        if not self._positions:
            self._approach_buffer.append({
                'timestamp': ts, 'price': price,
                'features': feat.copy(),
            })
            if len(self._approach_buffer) > 10:
                self._approach_buffer = self._approach_buffer[-10:]

        # ── IN-TRADE UPDATE + EXIT for each open position ───────────────
        closed_indices = []
        for i, pos in enumerate(self._positions):
            pos['bars_held'] = int((ts - pos['entry_ts']) // 60)
            if pos['direction'] == 'long':
                pnl = (price - pos['entry_price']) / TICK * TV
            else:
                pnl = (pos['entry_price'] - price) / TICK * TV
            pos['peak_pnl'] = max(pos['peak_pnl'], pnl)
            pos['trade_path'].append({
                'bar': pos['bars_held'], 'timestamp': ts, 'price': price,
                'pnl': pnl, 'peak_pnl': pos['peak_pnl'],
                'features': feat.copy(),
            })
            if is_1m:
                exit_reason = self._check_exit(pos, feat, z, vr)
                if exit_reason:
                    self._close_position(pos, price, ts, time_str, exit_reason, feat)
                    closed_indices.append(i)
        # Remove closed positions (reverse order preserves indices).
        for i in reversed(closed_indices):
            self._positions.pop(i)

        # ── PHANTOM PENDING: confirm or expire ──────────────────────────
        # Each tick after a signal fires, check if price has moved enough
        # in our direction to confirm. Runs on every bar (not just 1m)
        # so confirmation can fire fast once price starts to move.
        if self._pending is not None and len(self._positions) < self.max_chains:
            p = self._pending
            if p['direction'] == 'long':
                ticks_moved = (price - p['signal_price']) / TICK
            else:
                ticks_moved = (p['signal_price'] - price) / TICK
            if ticks_moved >= p['confirm_ticks']:
                # Confirmed! Enter at CURRENT price (not signal price).
                self._open_trade(p['direction'], price, ts, time_str,
                                 p['entry_feat'], p['tier'], p['signal_z'])
                self._pending = None
            elif (ts - p['signal_ts']) >= p['max_wait_min'] * 60:
                self._pending = None   # expired without confirmation

        # ── ENTRY / CHAIN (1m boundaries, |z|>ROCHE, under cap) ─────────
        if (is_1m and price > 100 and abs(z) > ROCHE
                and len(self._positions) < self.max_chains
                and self._pending is None):
            tier, direction = self._classify(feat, z, vr)
            if tier is not None:
                # Chain only if direction matches existing. Opposing
                # direction = inverse signal (already handled by exits).
                if (not self._positions
                        or direction == self._positions[0]['direction']):
                    phantom_cfg = PHANTOM_ENTRY.get(tier)
                    if phantom_cfg is None:
                        # Immediate entry (legacy behavior for tiers
                        # without phantom config).
                        self._open_trade(direction, price, ts, time_str,
                                         feat, tier, z)
                    else:
                        # Start phantom pending — watch for confirmation.
                        confirm_ticks, max_wait_min = phantom_cfg
                        self._pending = {
                            'tier': tier,
                            'direction': direction,
                            'signal_price': price,
                            'signal_ts': ts,
                            'signal_z': z,
                            'entry_feat': feat.copy(),
                            'confirm_ticks': confirm_ticks,
                            'max_wait_min': max_wait_min,
                        }

    # ══════════════════════════════════════════════════════════════════
    # Per-tier fire functions — return direction or None
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _trend_follower_fires(feat):
        """TREND_FOLLOWER — strong 1m velocity + 5m aligned + chaotic regime.

        Three conditions for the setup (user spec):
          A. |1m_velocity| >= TF_VEL_MIN (magnitude)
          B. sign(1m_vel) == sign(5m_vel) (multi-TF agreement)
          C. vr > TF_VR_MIN (chaotic regime)

        Direction: FADE the velocity (opposite of sign). 5-trade FREIGHT_TRAIN
        sample gave 20% WR held-to-EOD = 80% when flipped. Thesis: velocity
        extremes at band edge are exhaustion spikes; chasing the trend is
        buying the top. Under inverse-signal exit, velocity reversal (sign
        flip) auto-triggers exit because the tier's direction flips with it.
        """
        m1_vel = feat[_1M_VEL_IDX]
        m5_vel = feat[_5M_VEL_IDX]
        m1_vr = feat[_1M_VR_IDX]
        if abs(m1_vel) < TF_VEL_MIN:
            return None
        if m1_vel * m5_vel <= 0:
            return None
        if m1_vr <= TF_VR_MIN:
            return None
        # FLIPPED direction: fade the velocity, not ride it
        return 'short' if m1_vel > 0 else 'long'

    @staticmethod
    def _has_wick(feat):
        return (feat[_5M_WICK_IDX] > WICK_5M_MIN
                and feat[_15M_WICK_IDX] > WICK_15M_MIN)

    @classmethod
    def _cascade_fires(cls, feat, z):
        if not cls._has_wick(feat):
            return None
        h1_z = feat[_1H_Z_IDX]
        fade_dir = 'short' if z > 0 else 'long'
        h1_aligned = ((fade_dir == 'long' and h1_z < -CASCADE_H1_Z_MIN)
                      or (fade_dir == 'short' and h1_z > CASCADE_H1_Z_MIN))
        return fade_dir if h1_aligned else None

    @classmethod
    def _kill_shot_fires(cls, feat, z):
        if not cls._has_wick(feat):
            return None
        h1_z = feat[_1H_Z_IDX]
        fade_dir = 'short' if z > 0 else 'long'
        h1_aligned = ((fade_dir == 'long' and h1_z < -CASCADE_H1_Z_MIN)
                      or (fade_dir == 'short' and h1_z > CASCADE_H1_Z_MIN))
        return None if h1_aligned else fade_dir

    @classmethod
    def _kill_shot_inverse_fires(cls, feat, z):
        """KILL_SHOT_INVERSE — same setup, opposite direction.

        Thesis: KILL_SHOT identifies extreme reversal/continuation events.
        Current direction (fade) is correct 61% of the time. The 39% that
        run against us are BIG (avg -$33 vs winner +$18). Running the
        inverse direction in parallel extracts the edge regardless of which
        way the market actually resolves.

        Both tiers share the bar-15 peak<$5 no-progress cut in _check_exit.
        Whichever direction is wrong gets cut fast; whichever is right
        captures the peak-rule exit.
        """
        normal = cls._kill_shot_fires(feat, z)
        if normal is None:
            return None
        return 'short' if normal == 'long' else 'long'

    @staticmethod
    def _ride_against_fires(feat, z):
        """RIDE_AGAINST (FLIPPED direction — fade z despite h1_vel opposition).

        Setup: z at band edge + 1h velocity opposing the fade direction.
        Phantom entry (4-tick confirm) filters fizzles. All other entry
        filters (bar_range, h1_against_fade) removed — they were over-
        restrictive; phantom does the work now.
        """
        h1_vel = feat[_1H_VEL_IDX]
        fade_dir = 'short' if z > 0 else 'long'
        h1_vel_against = ((fade_dir == 'long' and h1_vel < -RA_H1_VEL_MIN)
                          or (fade_dir == 'short' and h1_vel > RA_H1_VEL_MIN))
        if not h1_vel_against:
            return None
        # FLIPPED: return fade direction (opposite of h1_vel).
        return fade_dir

    @staticmethod
    def _fade_against_fires(feat, z):
        h1_z = feat[_1H_Z_IDX]
        m5_vel = feat[_5M_VEL_IDX]
        fade_dir = 'short' if z > 0 else 'long'
        h1_against_fade = ((fade_dir == 'long' and h1_z > FA_H1_Z_MIN)
                           or (fade_dir == 'short' and h1_z < -FA_H1_Z_MIN))
        if not (h1_against_fade and abs(m5_vel) < FA_5M_VEL_MAX):
            return None
        return fade_dir

    @staticmethod
    def _mtf_exhaustion_fires(feat, z, vr):
        m5_vel = feat[_5M_VEL_IDX]
        m5_accel = feat[_5M_ACCEL_IDX]
        m1_vel = feat[_1M_VEL_IDX]
        vol_rel_1m = feat[_1M_VOL_REL_IDX]
        if not (m5_accel < 0
                and abs(m5_vel) > MTFE_5M_VEL_MIN
                and abs(m1_vel) > MTFE_1M_VEL_MIN
                and abs(z) > MTFE_Z_MIN
                and vr > MTFE_VR_MIN
                and vol_rel_1m > MTFE_VOL_MIN):
            return None
        return 'long' if m5_vel > 0 else 'short'

    @staticmethod
    def _mtf_breakout_fires(feat, z):
        """MTF_BREAKOUT (FLIPPED direction — fade the multi-TF z).

        Setup: 5m AND 15m z-extreme + 1m DMI not strongly against the z.
        Historical WR was 40% when we rode the breakout; the data said
        direction was inverted. Now FADE multi-TF extremes. Phantom
        confirmation filters fizzles. Testing RIDE_AGAINST-proven pattern.
        """
        z_5m = feat[_5M_Z_IDX]
        z_15m = feat[_15M_Z_IDX]
        if not (abs(z_5m) > MTFB_Z_MIN and abs(z_15m) > MTFB_Z_MIN):
            return None
        # Original direction (breakout = ride z)
        orig_dir = 'long' if z > 0 else 'short'
        dmi = feat[_1M_DMI_IDX]
        dmi_aligned = ((orig_dir == 'long' and dmi > -MTFB_DMI_AGAINST_MAX)
                       or (orig_dir == 'short' and dmi < MTFB_DMI_AGAINST_MAX))
        if not dmi_aligned:
            return None
        # FLIPPED: return fade direction (opposite of original)
        return 'short' if z > 0 else 'long'

    @staticmethod
    def _nmp_fade_fires(feat, z, vr):
        if abs(z) <= ROCHE or vr >= VR_ENTRY:
            return None
        return 'short' if z > 0 else 'long'

    @staticmethod
    def _nmp_ride_fires(feat, z, vr):
        if abs(z) <= ROCHE or vr < VR_ENTRY:
            return None
        ride_dir = 'long' if z > 0 else 'short'
        if NMP_RIDE_REQUIRE_OPPOSED_1D:
            d1_dir_vol = feat[_1D_DIR_VOL_IDX]
            dir_sign = 1 if ride_dir == 'long' else -1
            dv_sign = 1 if d1_dir_vol > 0 else (-1 if d1_dir_vol < 0 else 0)
            if dir_sign * dv_sign == 1:
                return None
        return ride_dir

    def _tier_fires(self, tier_name, feat, z, vr):
        if tier_name == 'TREND_FOLLOWER':    return self._trend_follower_fires(feat)
        if tier_name == 'CASCADE':           return self._cascade_fires(feat, z)
        if tier_name == 'KILL_SHOT':         return self._kill_shot_fires(feat, z)
        if tier_name == 'KILL_SHOT_INVERSE': return self._kill_shot_inverse_fires(feat, z)
        if tier_name == 'RIDE_AGAINST':      return self._ride_against_fires(feat, z)
        if tier_name == 'FADE_AGAINST':      return self._fade_against_fires(feat, z)
        if tier_name == 'MTF_EXHAUSTION':    return self._mtf_exhaustion_fires(feat, z, vr)
        if tier_name == 'MTF_BREAKOUT':      return self._mtf_breakout_fires(feat, z)
        if tier_name == 'NMP_FADE':          return self._nmp_fade_fires(feat, z, vr)
        if tier_name == 'NMP_RIDE':          return self._nmp_ride_fires(feat, z, vr)
        return None

    def _classify(self, feat, z, vr):
        """Classify entry tier for this bar.

        Isolated mode (only_tier set): check ONLY that tier's condition.
        No interference from higher-priority tiers — if our tier's
        conditions fire, we enter regardless of what other tiers would
        have done on this bar.

        Cascade mode (only_tier=None): priority order, first match wins.
        Kept for standalone/legacy usage; the main run_iso pipeline spins
        up one engine per tier in isolated mode.
        """
        if self.only_tier is not None:
            direction = self._tier_fires(self.only_tier, feat, z, vr)
            if direction is None:
                return None, None
            return self.only_tier, direction

        for tier_name in TIER_PRIORITY:
            direction = self._tier_fires(tier_name, feat, z, vr)
            if direction is not None:
                return tier_name, direction
        return None, None

    def _check_exit(self, pos, feat, z, vr):
        """Tier-specific exit logic for a single position (chain-aware).

        pos is a position dict with keys: direction, entry_tier, bars_held,
        peak_pnl, etc. Each open position is evaluated independently.

        TREND_FOLLOWER / RIDE_AGAINST / KILL_SHOT share the same peak-arrival
        rule (3-feature template from Q3 EDA) with per-tier amplitude gate
        and timeout thresholds. All other tiers fall through to inverse-signal.
        """
        entry_tier = pos['entry_tier']
        bars_held = pos['bars_held']
        peak_pnl = pos['peak_pnl']
        direction = pos['direction']
        if entry_tier is None:
            return None

        # TREND_FOLLOWER exits
        if entry_tier == 'TREND_FOLLOWER':
            if bars_held >= TF_EXIT_MAX_HOLD_MIN:
                return 'trend_follower_timeout'
            if peak_pnl >= TF_EXIT_MIN_PEAK_PNL:
                p_center = feat[_1M_P_CENTER_IDX]
                reversion = feat[_1M_REVERSION_IDX]
                m1_vr = feat[_1M_VR_IDX]
                if (p_center > TF_EXIT_P_CENTER_MIN
                        and reversion > TF_EXIT_REVERSION_MIN
                        and m1_vr < TF_EXIT_VR_MAX):
                    return 'trend_follower_peak'

        # RIDE_AGAINST exits (flipped direction — fade)
        # Thesis-dead removed: old check fired on h1_vel sign flip which
        # was DIRECTIONALLY RIGHT for riding but directionally BACKWARD
        # for fading. With flipped direction, h1_vel decay is the signal
        # reversion is winning (favorable). Peak rule + timeout + inverse
        # handle exits.
        if entry_tier == 'RIDE_AGAINST':
            if bars_held >= RA_EXIT_MAX_HOLD_MIN:
                return 'ride_against_timeout'
            if peak_pnl >= RA_EXIT_MIN_PEAK_PNL:
                p_center = feat[_1M_P_CENTER_IDX]
                reversion = feat[_1M_REVERSION_IDX]
                m1_vr = feat[_1M_VR_IDX]
                if (p_center > RA_EXIT_P_CENTER_MIN
                        and reversion > RA_EXIT_REVERSION_MIN
                        and m1_vr < RA_EXIT_VR_MAX):
                    return 'ride_against_peak'

        # KILL_SHOT / KILL_SHOT_INVERSE exits (shared rules, opposite directions).
        # Both tiers fire on the same wick-rejection setup; INVERSE just trades
        # the other way. Same exit logic applies — whichever direction is wrong
        # hits the no-progress cut fast, whichever is right runs to peak rule.
        if entry_tier in ('KILL_SHOT', 'KILL_SHOT_INVERSE'):
            # No-progress cut (Q2 cliff): if we haven't shown $5 of peak
            # progress by bar 15, the thesis is dead — cut before 30-bar
            # timeout takes full damage.
            if bars_held >= KS_EXIT_CUT_BAR and peak_pnl < KS_EXIT_CUT_PEAK_MAX:
                return 'kill_shot_no_progress'

            # ── KILL_SHOT_INVERSE cluster-signature exits ────────────────
            # REAL-bucket clustering (2026-04-19) surfaced two peak
            # signatures: long-extension and short-extension. Fire when
            # market reaches the signature's centroid state. Only applies
            # to INVERSE — normal KILL_SHOT's 301-trade sample didn't
            # produce clean clusters.
            if entry_tier == 'KILL_SHOT_INVERSE' and peak_pnl >= KS_EXIT_CUT_PEAK_MAX:
                dmi_15s = feat[_15S_DMI_IDX]
                z_5m    = feat[_5M_Z_IDX]
                dmi_1m  = feat[_1M_DMI_IDX]
                # Cluster 0: long-side directional extension
                if (dmi_15s > KSI_CLUSTER_LONG_DMI15S
                        and z_5m > KSI_CLUSTER_LONG_Z5M
                        and dmi_1m > KSI_CLUSTER_LONG_DMI1M):
                    return 'ksi_cluster_long_extension'
                # Cluster 1: short-side directional extension
                if (dmi_15s < KSI_CLUSTER_SHORT_DMI15S
                        and z_5m < KSI_CLUSTER_SHORT_Z5M
                        and dmi_1m < KSI_CLUSTER_SHORT_DMI1M):
                    return 'ksi_cluster_short_extension'

            if bars_held >= KS_EXIT_MAX_HOLD_MIN:
                return 'kill_shot_timeout'
            if peak_pnl >= KS_EXIT_MIN_PEAK_PNL:
                p_center = feat[_1M_P_CENTER_IDX]
                reversion = feat[_1M_REVERSION_IDX]
                m1_vr = feat[_1M_VR_IDX]
                if (p_center > KS_EXIT_P_CENTER_MIN
                        and reversion > KS_EXIT_REVERSION_MIN
                        and m1_vr < KS_EXIT_VR_MAX):
                    return 'kill_shot_peak'

        # MTF_BREAKOUT — inverse-only (no peak rule).
        # Tested with flipped direction + phantom + peak rule:
        #   flip+phantom alone:            661 tr, 60% WR, -$96
        #   flip+phantom + peak rule:      716 tr, 76% WR, -$2,934 (WORSE)
        # Peak rule steals inverse-bucket winners that had better $/trade
        # as inverse exits. This tier wants inverse-only like NMP_FADE.

        # NMP_FADE — NO tier-specific exit rule. Peak rule HURT this tier:
        # it captured winners at mean peak $23 (early), but winners' natural
        # peak is $69, so we were shortening winners more than saving losers.
        # As the least-selective catch-all tier (52.7% peak>$20, smallest
        # winner peaks across all tiers), NMP_FADE does best with just the
        # inverse-signal fallback — let winners run to their natural inverse.
        # Measured journey (chains=1):
        #   inverse-only:              +$40.13/day (BEST)
        #   + peak rule + 15m timeout: +$32.71/day
        #   + peak rule (no timeout):  +$31.46/day
        # Constants retained (NF_EXIT_*) for possible future use or A/B.

        # Inverse-signal (universal fallback)
        direction_fired = self._tier_fires(entry_tier, feat, z, vr)
        if direction_fired is not None and direction_fired != direction:
            return f'{entry_tier.lower()}_inverse'
        return None

    # ══════════════════════════════════════════════════════════════════
    # Trade lifecycle
    # ══════════════════════════════════════════════════════════════════

    def _open_trade(self, direction, price, ts, time_str, feat, tier, z):
        """Open a new position (primary or chain). Chains allowed up to
        self.max_chains total concurrent positions."""
        pos = {
            'direction': direction,
            'entry_price': price,
            'entry_ts': ts,
            'entry_tier': tier,
            'entry_79d': feat.copy(),
            'entry_abs_z': abs(z),
            'entry_approach': list(self._approach_buffer),
            'bars_held': 0,
            'peak_pnl': 0.0,
            'trade_path': [{
                'bar': 0, 'timestamp': ts, 'price': price,
                'pnl': 0.0, 'peak_pnl': 0.0,
                'features': feat.copy(),
            }],
            'chain_idx': len(self._positions),   # 0 = primary, 1+ = chain
        }
        self._positions.append(pos)

    def _close_position(self, pos, price, ts, time_str, exit_reason, feat):
        """Close a specific position (by reference). Caller removes it
        from self._positions (see on_state loop)."""
        if pos['direction'] == 'long':
            pnl = (price - pos['entry_price']) / TICK * TV
        else:
            pnl = (pos['entry_price'] - price) / TICK * TV
        self.daily_pnl += pnl

        entry_79d = pos['entry_79d']
        self.trades.append({
            'trade_id': len(self.trades),
            'time': time_str,
            'timestamp': pos['trade_path'][0]['timestamp'] if pos['trade_path'] else ts,
            'dir': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'held': pos['bars_held'],
            'peak': pos['peak_pnl'],
            'entry_tier': pos['entry_tier'],
            'exit_reason': exit_reason,
            'chain_idx': pos.get('chain_idx', 0),
            'entry_79d': entry_79d.tolist() if hasattr(entry_79d, 'tolist') else list(entry_79d),
            'exit_79d': feat.tolist() if hasattr(feat, 'tolist') else list(feat),
            'approach': pos['entry_approach'],
            'path': pos['trade_path'],
        })

    def force_close(self, reason='end_of_day'):
        """Close all open positions at the last-known price."""
        while self._positions:
            pos = self._positions[0]
            ts = pos['trade_path'][-1]['timestamp'] if pos['trade_path'] else 0
            feat = pos['trade_path'][-1]['features'] if pos['trade_path'] else pos['entry_79d']
            if feat is None:
                feat = np.zeros(91)
            time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M') if ts > 0 else '??:??'
            self._close_position(pos, self._last_price, ts, time_str, reason, feat)
            self._positions.pop(0)

    def reset(self):
        self._positions = []
        self._pending = None
        self._approach_buffer = []
        self.daily_pnl = 0.0
        # Invalidate 5s slope buffer — caller must call set_sec_closes() per day.
        # If they forget, _slope_1m returns 0.0 (safe fallback, not stale data).
        self._sec_ts = None
        self._sec_close = None
        self._sec_cursor = 0

    def get_full_trades(self):
        return self.trades

    def summary(self):
        n = len(self.trades)
        if n == 0:
            return 'No trades'
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = sum(t['pnl'] for t in self.trades)
        parts = []
        for tier in TIER_PRIORITY:
            k = sum(1 for t in self.trades if t['entry_tier'] == tier)
            if k > 0:
                parts.append(f'{tier}={k}')
        return (f'{n} trades ({" ".join(parts)}) | '
                f'WR={wins/n*100:.0f}% | ${total:+.0f}')
