"""Physics Engine — K-NN trajectory matching against enriched seeds.

Alternative to AdvanceEngine for live trading. Uses IS-learned seed
library with full physics (12 features x 10-bar trajectory) to match
current market state against 38K proven patterns.

Proven: $132/day OOS, no lookahead, 53% WR, $13.35/trade.

Usage:
    engine = PhysicsEngine.from_seeds('DATA/regime_seeds/auto_seeds_all.json')
    for bar in bars:
        result = engine.on_bar(price, high, low, timestamp, state)
        if result.action == 'ENTER':
            # enter result.direction, hold result.hold_bars
        elif result.action == 'EXIT':
            # exit at current price
"""

import json
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


TICK_SIZE = 0.25
TICK_VALUE = 0.50

# 12 core features (proven optimal — 22 features adds noise)
TRAJ_KEYS = [
    'fm', 'z', 'dmi_p', 'dmi_m', 'adx', 'vel', 'vol', 'hurst',
    'P_center', 'coherence', 'sigma', 'pid',
]
TRAJ_LEN = 10  # 10-bar lookback at 1m
N_FEAT = len(TRAJ_KEYS)


@dataclass
class EngineResult:
    action: str = 'HOLD'          # ENTER, EXIT, FLIP, HOLD
    direction: str = ''            # LONG, SHORT
    hold_bars: int = 0             # bars to hold
    consensus: float = 0.0         # K-NN direction consensus
    n_matched: int = 0             # number of seeds matched
    pnl_ticks: float = 0.0        # filled on EXIT
    reason: str = ''               # human-readable decision reason
    coherence: float = 0.0         # TF coherence at decision time
    magnitude: float = 0.0         # prior move magnitude (ticks)
    mag_pctile: float = 0.0        # magnitude percentile


class PhysicsEngine:
    """K-NN trajectory matching engine for live trading."""

    def __init__(
        self,
        seed_flat: np.ndarray,       # (n_seeds, TRAJ_LEN * N_FEAT) normalized
        seed_dir: np.ndarray,        # (n_seeds,) +1 LONG, -1 SHORT
        seed_bars: np.ndarray,       # (n_seeds,) hold duration per seed
        seed_mag: np.ndarray,        # (n_seeds,) |change_ticks| per seed
        feat_means: np.ndarray,      # (N_FEAT,) for normalization
        feat_stds: np.ndarray,       # (N_FEAT,) for normalization
        k: int = 20,
        min_consensus: float = 0.65,
        max_coherence: float = 0.60,
        min_mag_pctile: float = 0.25,
    ):
        self.seed_flat = seed_flat
        self.seed_dir = seed_dir
        self.seed_bars = seed_bars
        self.seed_mag = seed_mag
        self.feat_means = feat_means
        self.feat_stds = feat_stds

        self.k = k
        self.min_consensus = min_consensus
        self.max_coherence = max_coherence
        self.min_mag_pctile = min_mag_pctile

        # Rolling state
        self._traj_buffer = deque(maxlen=TRAJ_LEN)
        self._mag_window = deque(seed_mag[-200:], maxlen=200)
        self._move_start_price = 0.0
        self._in_trade = False
        self._entry_price = 0.0
        self._entry_bar = 0
        self._trade_dir = ''
        self._trade_hold = 0
        self._bar_count = 0

        # Funnel progression tracking
        self._consensus_history = deque(maxlen=20)
        self._direction_history = deque(maxlen=20)

        # Stats
        self.stats = {
            'bars': 0, 'entries': 0, 'exits': 0,
            'skipped_warmup': 0, 'skipped_coherence': 0,
            'skipped_magnitude': 0, 'skipped_consensus': 0,
            'skipped_fm': 0,
        }

    @classmethod
    def from_seeds(cls, seed_path: str, **kwargs) -> 'PhysicsEngine':
        """Load enriched seed JSON and build engine."""
        with open(seed_path) as f:
            raw = json.load(f)

        seeds_raw = raw.get('seeds', [])
        if not seeds_raw:
            for d in raw.get('days', {}).values():
                seeds_raw.extend(d.get('seeds', []))

        # Filter to seeds with full physics lookback
        seeds = [s for s in seeds_raw
                 if 'entry_fm' in s and len(s.get('lookback', [])) >= TRAJ_LEN]

        # Build trajectory matrix
        n = len(seeds)
        trajs = np.zeros((n, TRAJ_LEN, N_FEAT))
        for si, s in enumerate(seeds):
            lb = s['lookback'][-TRAJ_LEN:]
            for bi, bar in enumerate(lb):
                for fi, key in enumerate(TRAJ_KEYS):
                    trajs[si, bi, fi] = bar.get(key, 0.0)

        # Normalize
        flat = trajs.reshape(-1, N_FEAT)
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)
        stds[stds < 1e-8] = 1.0
        trajs_normed = (trajs - means) / stds
        seed_flat = trajs_normed.reshape(n, -1)

        seed_dir = np.array([1 if s['direction'] == 'LONG' else -1 for s in seeds])
        seed_bars = np.array([s['n_bars'] for s in seeds])
        seed_mag = np.array([abs(s['change_ticks']) for s in seeds])

        print(f'[PhysicsEngine] Loaded {n} seeds, trajectory: {seed_flat.shape}')
        return cls(seed_flat, seed_dir, seed_bars, seed_mag, means, stds, **kwargs)

    def _extract_features(self, state) -> list:
        """Extract 12 features from MarketState."""
        return [
            state.F_momentum,
            state.z_score,
            state.dmi_plus,
            state.dmi_minus,
            state.adx_strength,
            state.velocity,
            state.volume_delta,
            state.hurst_exponent,
            state.P_at_center,
            getattr(state, 'oscillation_entropy_normalized', 0),
            state.regression_sigma,
            state.term_pid,
        ]

    def _match_funnel(self) -> Optional[dict]:
        """Progressive funnel: match sliding window against seeds.

        Returns dict with direction, consensus, hold, distance if match found.
        Returns None if no match or funnel hasn't narrowed enough.
        """
        if len(self._traj_buffer) < TRAJ_LEN:
            return None

        current_traj = np.array(list(self._traj_buffer))
        current_normed = (current_traj - self.feat_means) / self.feat_stds
        current_flat = current_normed.reshape(1, -1)

        # Full distance against all seeds
        dist = np.linalg.norm(self.seed_flat - current_flat, axis=1)
        nearest_idx = np.argpartition(dist, self.k)[:self.k]
        nearest_dist = dist[nearest_idx]

        # Direction consensus
        nearest_dirs = self.seed_dir[nearest_idx]
        n_long = int((nearest_dirs > 0).sum())
        n_short = self.k - n_long
        consensus = max(n_long, n_short) / self.k
        direction = 'LONG' if n_long > n_short else 'SHORT'

        # Progressive narrowing: track how consensus evolves
        # Store for trend detection
        self._consensus_history.append(consensus)
        self._direction_history.append(direction)

        if consensus < self.min_consensus:
            return None

        hold = int(np.median(self.seed_bars[nearest_idx]))
        hold = max(3, min(hold, 20))

        return {
            'direction': direction,
            'consensus': consensus,
            'hold': hold,
            'avg_dist': float(nearest_dist.mean()),
        }

    def on_bar(
        self,
        price: float,
        high: float,
        low: float,
        timestamp: float,
        state,
    ) -> EngineResult:
        """Process one 1m bar. Returns EngineResult.

        The funnel runs continuously — every bar it matches the sliding
        window against seeds. Entry and exit are both driven by the funnel:
        - ENTER: funnel consensus exceeds threshold in a direction
        - EXIT: funnel flips direction (new seed dominates = old move done)
        """
        self._bar_count += 1
        self.stats['bars'] += 1

        if self._move_start_price == 0:
            self._move_start_price = price

        # Update trajectory buffer
        bar_feats = self._extract_features(state)
        self._traj_buffer.append(bar_feats)
        coherence = getattr(state, 'oscillation_entropy_normalized', 0)

        # Run funnel every bar (both in trade and flat)
        match = self._match_funnel()

        # ── IN TRADE: check for funnel flip (exit signal) ──
        _bars_held = self._bar_count - self._entry_bar if self._in_trade else 0
        _bars_left = (self._entry_bar + self._trade_hold - self._bar_count) if self._in_trade else 0

        if self._in_trade:
            _match_dir = match['direction'] if match else 'none'
            _match_cons = match['consensus'] if match else 0.0

            # Exit condition 1: funnel flipped direction
            if match and match['direction'] != self._trade_dir:
                if self._trade_dir == 'LONG':
                    pnl = (price - self._entry_price) / TICK_SIZE
                else:
                    pnl = (self._entry_price - price) / TICK_SIZE

                old_dir = self._trade_dir
                self._in_trade = False
                self._move_start_price = price
                self.stats['exits'] += 1

                # Immediately enter the new direction (funnel says flip)
                if coherence <= self.max_coherence:
                    self._entry_price = price
                    self._entry_bar = self._bar_count
                    self._trade_dir = match['direction']
                    self._trade_hold = match['hold']
                    self._in_trade = True
                    self.stats['entries'] += 1

                    return EngineResult(
                        action='FLIP',
                        direction=match['direction'],
                        hold_bars=match['hold'],
                        consensus=match['consensus'],
                        n_matched=self.k,
                        pnl_ticks=pnl,
                        reason=f'funnel_flip {old_dir}->{match["direction"]} cons={match["consensus"]:.2f} held={_bars_held}',
                        coherence=coherence,
                    )

                return EngineResult(
                    action='EXIT',
                    direction=old_dir,
                    pnl_ticks=pnl,
                    reason=f'flip_blocked coh={coherence:.2f}>{self.max_coherence} held={_bars_held}',
                    coherence=coherence,
                )

            # Exit condition 2: max hold reached
            if self._bar_count >= self._entry_bar + self._trade_hold:
                if self._trade_dir == 'LONG':
                    pnl = (price - self._entry_price) / TICK_SIZE
                else:
                    pnl = (self._entry_price - price) / TICK_SIZE

                self._in_trade = False
                self._move_start_price = price
                self.stats['exits'] += 1

                return EngineResult(
                    action='EXIT',
                    direction=self._trade_dir,
                    pnl_ticks=pnl,
                    reason=f'max_hold {self._trade_hold} bars',
                    coherence=coherence,
                )

            # Hold
            return EngineResult(
                reason=f'HOLD {self._trade_dir} bar {_bars_held}/{self._trade_hold} funnel={_match_dir}({_match_cons:.0%}) coh={coherence:.2f}',
                coherence=coherence,
            )

        # ── FLAT: check for entry ──

        if len(self._traj_buffer) < TRAJ_LEN:
            self.stats['skipped_warmup'] += 1
            return EngineResult(reason=f'warmup {len(self._traj_buffer)}/{TRAJ_LEN}')

        # Filter: coherence
        if coherence > self.max_coherence:
            self.stats['skipped_coherence'] += 1
            return EngineResult(
                reason=f'SKIP coh={coherence:.2f}>{self.max_coherence}',
                coherence=coherence,
            )

        # Filter: magnitude
        prior_move = abs(price - self._move_start_price) / TICK_SIZE
        mag_pctile = sum(1 for m in self._mag_window if m < prior_move) / max(len(self._mag_window), 1)
        if mag_pctile < self.min_mag_pctile:
            self.stats['skipped_magnitude'] += 1
            return EngineResult(
                reason=f'SKIP mag={prior_move:.1f}t p{mag_pctile:.0%}<p{self.min_mag_pctile:.0%}',
                coherence=coherence, magnitude=prior_move, mag_pctile=mag_pctile,
            )

        # Check funnel match
        if not match:
            self.stats['skipped_consensus'] += 1
            return EngineResult(
                reason=f'SKIP no_consensus mag={prior_move:.1f}t coh={coherence:.2f}',
                coherence=coherence, magnitude=prior_move, mag_pctile=mag_pctile,
            )

        # Entry
        self._entry_price = price
        self._entry_bar = self._bar_count
        self._trade_dir = match['direction']
        self._trade_hold = match['hold']
        self._in_trade = True
        self._mag_window.append(prior_move)
        self._move_start_price = price
        self.stats['entries'] += 1

        return EngineResult(
            action='ENTER',
            direction=match['direction'],
            hold_bars=match['hold'],
            consensus=match['consensus'],
            n_matched=self.k,
            reason=f'ENTER {match["direction"]} cons={match["consensus"]:.2f} hold={match["hold"]} mag={prior_move:.1f}t',
            coherence=coherence, magnitude=prior_move, mag_pctile=mag_pctile,
        )

    def report(self) -> str:
        """Return stats summary."""
        s = self.stats
        lines = [
            'PhysicsEngine Stats:',
            f'  Bars: {s["bars"]}',
            f'  Entries: {s["entries"]}',
            f'  Exits: {s["exits"]}',
            f'  Skipped: warmup={s["skipped_warmup"]} coh={s["skipped_coherence"]} '
            f'mag={s["skipped_magnitude"]} consensus={s["skipped_consensus"]}',
        ]
        return '\n'.join(lines)
