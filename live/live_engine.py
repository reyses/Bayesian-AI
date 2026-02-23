"""
LiveEngine — main live trading loop.

Mirrors the Phase 4 forward pass (orchestrator.py) but operates on real-time
bars from NinjaTrader 8 via the NT8Client TCP bridge.

Key simplifications vs Phase 4:
  - No oracle markers (no lookahead in live)
  - No score_loser tracking
  - No scan_day_cascade — state-to-centroid matching directly from latest bar
  - No equity sim — real equity from NT8 POSITION messages
  - Single "best candidate" per bar (no multi-TF cascade scan)
"""

import asyncio
import json
import logging
import os
import pickle
import time
import glob
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

from core.quantum_field_engine import QuantumFieldEngine
from core.bayesian_brain import QuantumBayesianBrain
from training.fractal_clustering import FractalClusteringEngine
from training.timeframe_belief_network import TimeframeBeliefNetwork
from training.wave_rider import WaveRider
from config.symbols import SYMBOL_MAP

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.bar_aggregator import LiveBarAggregator
from live.order_manager import OrderManager

logger = logging.getLogger(__name__)

# Timeframe seconds lookup (same as orchestrator)
TIMEFRAME_SECONDS = {
    '1D': 86400, '4h': 14400, '1h': 3600, '30m': 1800, '15m': 900,
    '5m': 300, '3m': 180, '2m': 120, '1m': 60, '30s': 30, '15s': 15,
    '5s': 5, '1s': 1,
}

# Gate 0 constants (match orchestrator.py)
_ADX_TREND_CONFIRMATION = 25.0
_HURST_TREND_CONFIRMATION = 0.6
_GATE1_DIST_THRESHOLD = 4.5
_WORKER_BYPASS_CONV = 0.65


@dataclass
class _LiveCandidate:
    """Lightweight stand-in for PatternEvent in live mode."""
    pattern_type: str
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    state: object         # ThreeBodyQuantumState
    depth: int = 10       # default: 15s depth level
    timeframe: str = '15s'
    parent_chain: list = None
    timestamp: float = 0.0
    price: float = 0.0
    idx: int = 0
    file_source: str = 'live'
    window_data: object = None
    oracle_marker: int = 0
    oracle_meta: dict = None

    def __post_init__(self):
        if self.parent_chain is None:
            self.parent_chain = []
        if self.oracle_meta is None:
            self.oracle_meta = {}


class LiveEngine:
    """Main live trading loop — replaces Phase 4 forward pass for real-time."""

    def __init__(self, config: LiveConfig, dry_run: bool = False):
        self._cfg = config
        self._dry_run = dry_run

        # Core components (loaded from checkpoints)
        self._asset = SYMBOL_MAP.get(config.asset_ticker)
        if self._asset is None:
            raise ValueError(f"Unknown asset ticker: {config.asset_ticker}")

        self._engine = QuantumFieldEngine()
        self._brain = QuantumBayesianBrain()
        self._wave_rider = WaveRider(self._asset)

        # These are loaded in _load_checkpoints()
        self._pattern_library: Dict = {}
        self._scaler = None
        self._valid_tids: List[str] = []
        self._centroids_scaled: np.ndarray = None
        self._template_tier_map: Dict = {}
        self._exception_tids: set = set()
        self._depth_score_adj: Dict[int, float] = {}
        self._depth_filter_out: set = set()
        self._tier_score_adj = {1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5}

        # Live components
        self._client = NT8Client(config)
        self._aggregator = LiveBarAggregator(self._engine, config)
        self._orders = OrderManager(config)
        self._belief_network: Optional[TimeframeBeliefNetwork] = None

        # Position tracking (mirrors orchestrator forward pass)
        self._position_open = False
        self._entry_price = 0.0
        self._entry_time = 0.0
        self._entry_bar = 0
        self._active_side = ''
        self._active_tid = None
        self._max_hold_bars = 960

        self._bar_i = 0   # running bar index

    # ── Public API ────────────────────────────────────────────────────

    async def run(self):
        """Main entry point — connect, load checkpoints, run loop."""
        logger.info("=" * 60)
        logger.info("LIVE ENGINE STARTING")
        logger.info(f"  Instrument: {self._cfg.instrument}")
        logger.info(f"  Account:    {self._cfg.account}")
        logger.info(f"  Dry run:    {self._dry_run}")
        logger.info("=" * 60)

        self._load_checkpoints()
        self._init_belief_network()

        connected = await self._client.connect()
        if not connected:
            logger.error("Failed to connect to NT8 bridge — exiting")
            return

        try:
            await self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        except Exception as e:
            logger.error(f"LiveEngine fatal error: {e}", exc_info=True)
        finally:
            # Close any open position on shutdown
            if self._position_open and not self._dry_run:
                msg = self._orders.build_exit_order(reason='shutdown')
                if msg:
                    await self._client.send(msg)
            await self._client.disconnect()
            logger.info("LiveEngine stopped")

    # ── Main Loop ─────────────────────────────────────────────────────

    async def _main_loop(self):
        """Process inbound messages from NT8."""
        while not self._client._stop:
            try:
                msg = await asyncio.wait_for(
                    self._client.inbound.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue

            mtype = msg.get('type', '')

            if mtype == 'BAR':
                await self._on_bar(msg)
            elif mtype == 'FILL':
                self._orders.on_fill(msg)
                self._sync_position_state()
            elif mtype == 'ORDER_STATUS':
                self._orders.on_order_status(msg)
            elif mtype == 'POSITION':
                self._orders.on_position(msg)
                self._sync_position_state()
            elif mtype == 'CONNECTED':
                logger.info(f"NT8 CONNECTED: account={msg.get('account')}")

    async def _on_bar(self, msg: dict):
        """Process a single inbound BAR message."""
        states = self._aggregator.add_bar(msg)
        if states is None:
            return  # still warming up

        price = float(msg['close'])
        ts = float(msg['timestamp'])
        self._bar_i += 1

        # Feed belief network
        df = self._aggregator.df
        if self._bar_i % 240 == 1:
            # Re-prepare day periodically (resamples higher TFs)
            self._belief_network.prepare_day(df, states_micro=states)

        self._belief_network.tick_all(self._bar_i)

        # ── Exit check (if position open) ─────────────────────────────
        if self._position_open:
            await self._check_exit(price, ts)

        # ── Entry check (if flat) ─────────────────────────────────────
        if not self._position_open and not self._orders.loss_limit_hit:
            await self._check_entry(price, ts, states)

    # ── Exit Logic ────────────────────────────────────────────────────

    async def _check_exit(self, price: float, ts: float):
        """Check for exit signals on the current bar."""
        # Max hold check
        bars_held = self._bar_i - self._entry_bar
        if bars_held >= self._max_hold_bars:
            logger.info(f"MAX_HOLD reached ({bars_held} bars) — closing")
            self._belief_network.stop_trade_tracking()
            await self._close_position('MAX_HOLD')
            return

        # Normal exit via wave rider + belief network
        pos = self._wave_rider.position
        if pos is None:
            return

        exit_sig = self._belief_network.get_exit_signal(pos.side)
        result = self._wave_rider.update_trail(price, None, ts, exit_signal=exit_sig)

        if result.get('should_exit', False):
            reason = result.get('exit_reason', 'trail_stop')
            logger.info(f"EXIT signal: {reason} (decay={exit_sig.get('decay_score', 0):.2f})")
            self._belief_network.stop_trade_tracking()
            await self._close_position(reason)

    async def _close_position(self, reason: str):
        """Send close order and reset position state."""
        self._position_open = False
        self._wave_rider.position = None

        if self._dry_run:
            logger.info(f"[DRY RUN] Would close position: {reason}")
            return

        msg = self._orders.build_exit_order(reason=reason)
        if msg:
            await self._client.send(msg)

    # ── Entry Logic (Gate 0 → 3) ─────────────────────────────────────

    async def _check_entry(self, price: float, ts: float, states: list):
        """Run the gate cascade on the current bar's quantum state."""
        if not states:
            return

        # Get the latest state
        latest = states[-1]
        state = latest['state']

        # Detect pattern type from state flags
        candidates = []
        if state.cascade_detected:
            candidates.append(self._build_candidate(
                'ROCHE_SNAP', state, price, ts))
        if state.structure_confirmed:
            candidates.append(self._build_candidate(
                'STRUCTURAL_DRIVE', state, price, ts))

        if not candidates:
            return

        # Run gates on each candidate
        best_candidate = None
        best_dist = 999.0
        best_tid = None

        for p in candidates:
            # ── Gate 0: Headroom ──────────────────────────────────────
            micro_z = abs(p.z_score)
            micro_pattern = p.pattern_type
            macro_z = 0.0  # no parent chain in live (single TF)

            should_skip = False

            # Data-quality override check
            _data_override = False
            if self._exception_tids and micro_pattern:
                _e_feat = np.array([FractalClusteringEngine.extract_features(p)])
                _e_scaled = self._scaler.transform(_e_feat)
                _e_dists = np.linalg.norm(self._centroids_scaled - _e_scaled, axis=1)
                _e_nearest = int(np.argmin(_e_dists))
                if (_e_dists[_e_nearest] < _GATE1_DIST_THRESHOLD
                        and self._valid_tids[_e_nearest] in self._exception_tids):
                    _data_override = True

            if not _data_override:
                if not micro_pattern:
                    should_skip = True
                elif micro_z < 0.5:
                    should_skip = True
                elif 0.5 <= micro_z < 2.0:
                    if micro_pattern == 'STRUCTURAL_DRIVE':
                        if (state.adx_strength < _ADX_TREND_CONFIRMATION
                                or state.hurst_exponent < _HURST_TREND_CONFIRMATION):
                            should_skip = True
                    elif micro_pattern == 'ROCHE_SNAP':
                        should_skip = True
                elif micro_z >= 2.0:
                    headroom = macro_z < 3.0
                    if micro_pattern == 'ROCHE_SNAP':
                        if not headroom and micro_z > 3.0:
                            should_skip = True
                    elif micro_pattern == 'STRUCTURAL_DRIVE':
                        if not headroom:
                            should_skip = True

            if should_skip:
                continue

            # ── Gate 0.5: Depth filter ────────────────────────────────
            if p.depth in self._depth_filter_out:
                continue

            # ── Gate 1: Cluster matching ──────────────────────────────
            features = np.array([FractalClusteringEngine.extract_features(p)])
            feat_scaled = self._scaler.transform(features)
            dists = np.linalg.norm(self._centroids_scaled - feat_scaled, axis=1)
            nearest_idx = np.argmin(dists)
            dist = float(dists[nearest_idx])
            tid = self._valid_tids[nearest_idx]

            if dist >= _GATE1_DIST_THRESHOLD:
                continue

            # ── Gate 2: Brain ─────────────────────────────────────────
            if not self._brain.should_fire(tid, min_prob=0.05, min_conf=0.0):
                continue

            # Score competition
            p_depth = p.depth
            tier_adj = self._tier_score_adj.get(
                self._template_tier_map.get(tid, 3), 0.0)
            depth_adj = self._depth_score_adj.get(p_depth, 0.0)
            score = p_depth + dist + tier_adj + depth_adj

            if score < best_dist:
                best_dist = score
                best_candidate = p
                best_tid = tid

        if best_candidate is None:
            return

        # ── Gate 3: Path conviction ───────────────────────────────────
        belief = self._belief_network.get_belief()
        side = self._determine_direction(best_candidate, best_tid)

        if belief is not None:
            if not belief.is_confident:
                return  # conviction too low
            if belief.direction != side:
                side = belief.direction
            _network_tp = (max(4, int(round(belief.predicted_mfe)))
                           if belief.predicted_mfe > 2.0 else None)
        else:
            _network_tp = None

        # ── Exit sizing ───────────────────────────────────────────────
        lib_entry = self._pattern_library.get(best_tid, {})
        params = lib_entry.get('params', {})
        sl_ticks, trail_ticks, trail_act, tp_ticks = self._compute_exit_params(
            lib_entry, params, _network_tp, best_candidate)

        # ── Execute entry ─────────────────────────────────────────────
        logger.info(f"ENTRY: {side.upper()} @ {price:.2f}  "
                    f"tid={best_tid}  dist={best_dist:.2f}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")

        self._wave_rider.open_position(
            entry_price=price, side=side,
            state=best_candidate.state,
            stop_distance_ticks=sl_ticks,
            profit_target_ticks=tp_ticks,
            trailing_stop_ticks=trail_ticks,
            trail_activation_ticks=trail_act,
            template_id=best_tid,
        )
        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = best_tid

        _tf_s = str(best_candidate.timeframe)
        _tf_sec = TIMEFRAME_SECONDS.get(_tf_s, 14400)
        self._max_hold_bars = max(20, _tf_sec // 15)

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars)

        if self._dry_run:
            logger.info("[DRY RUN] Entry logged but no order sent")
            return

        order_msg = self._orders.build_entry_order(
            'BUY' if side == 'long' else 'SELL')
        if order_msg:
            await self._client.send(order_msg)

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_candidate(self, pattern_type: str, state, price: float,
                         ts: float) -> _LiveCandidate:
        """Build a PatternEvent-like candidate from a quantum state."""
        return _LiveCandidate(
            pattern_type=pattern_type,
            z_score=state.z_score,
            velocity=state.particle_velocity,
            momentum=state.momentum_strength,
            coherence=state.coherence,
            state=state,
            depth=10,           # 15s resolution in live
            timeframe='15s',
            timestamp=ts,
            price=price,
            idx=self._bar_i,
        )

    def _determine_direction(self, candidate: _LiveCandidate,
                             tid: str) -> str:
        """Determine trade direction from template library + live state."""
        lib_entry = self._pattern_library.get(tid, {})
        long_bias = lib_entry.get('long_bias', 0.0)
        short_bias = lib_entry.get('short_bias', 0.0)
        _BIAS_THRESH = 0.55

        # Priority 1: per-cluster logistic regression
        _live_feat = np.array(FractalClusteringEngine.extract_features(candidate))
        _live_scaled = self._scaler.transform([_live_feat])[0]

        _dir_coeff = lib_entry.get('dir_coeff')
        if _dir_coeff is not None:
            _dir_logit = (np.dot(_live_scaled, np.array(_dir_coeff))
                          + lib_entry.get('dir_intercept', 0.0))
            _dir_prob = 1.0 / (1.0 + np.exp(-_dir_logit))
            if _dir_prob > _BIAS_THRESH:
                return 'long'
            elif _dir_prob < (1.0 - _BIAS_THRESH):
                return 'short'

        # Priority 2: template aggregate bias
        if long_bias >= _BIAS_THRESH:
            return 'long'
        elif short_bias >= _BIAS_THRESH:
            return 'short'
        elif long_bias + short_bias >= 0.10:
            return 'long' if long_bias >= short_bias else 'short'

        # Priority 3: live DMI (trend-following)
        s = candidate.state
        _dmi_diff = (getattr(s, 'dmi_plus', 0.0)
                     - getattr(s, 'dmi_minus', 0.0))
        if _dmi_diff > 0:
            return 'long'
        elif _dmi_diff < 0:
            return 'short'

        vel = getattr(s, 'particle_velocity', 0.0)
        return 'long' if vel >= 0 else 'short'

    def _compute_exit_params(self, lib_entry: dict, params: dict,
                             network_tp: Optional[int],
                             candidate: _LiveCandidate):
        """Compute SL, trail, trail activation, and TP ticks."""
        _reg_sigma = lib_entry.get('regression_sigma_ticks', 0.0)
        _mean_mae = lib_entry.get('mean_mae_ticks', 0.0)
        _p75_mfe = lib_entry.get('p75_mfe_ticks', 0.0)
        _p25_mae = lib_entry.get('p25_mae_ticks', 0.0)

        # Phase 1: hard stop
        if _p25_mae > 2.0:
            sl = max(4, int(round(_p25_mae * 3.0)))
        elif _mean_mae > 2.0:
            sl = max(4, int(round(_mean_mae * 2.0)))
        else:
            sl = params.get('stop_loss_ticks', 20)

        # Phase 2: trail
        if _reg_sigma > 2.0:
            trail = max(2, int(round(_reg_sigma * 1.1)))
        elif _mean_mae > 2.0:
            trail = max(2, int(round(_mean_mae * 1.1)))
        else:
            trail = params.get('trailing_stop_ticks', 10)

        # Trail activation
        trail_act = (max(2, int(round(_p25_mae * 0.3)))
                     if _p25_mae > 2.0 else None)

        # TP priority: network → OLS → p75 → fallback
        if network_tp is not None:
            tp = network_tp
        else:
            _live_feat = np.array(FractalClusteringEngine.extract_features(candidate))
            _live_scaled = self._scaler.transform([_live_feat])[0]
            _mfe_coeff = lib_entry.get('mfe_coeff')
            if _mfe_coeff is not None:
                _pred = (np.dot(_live_scaled, np.array(_mfe_coeff))
                         + lib_entry.get('mfe_intercept', 0.0))
                _pred_ticks = max(0.0, _pred / 0.25)
                tp = (max(4, int(round(_pred_ticks)))
                      if _pred_ticks > 2.0
                      else (max(4, int(round(_p75_mfe)))
                            if _p75_mfe > 2.0
                            else params.get('take_profit_ticks', 50)))
            elif _p75_mfe > 2.0:
                tp = max(4, int(round(_p75_mfe)))
            else:
                tp = params.get('take_profit_ticks', 50)

        return sl, trail, trail_act, tp

    def _sync_position_state(self):
        """Sync local position tracking with OrderManager's state."""
        if self._position_open and self._orders.is_flat:
            # Position closed (fill received)
            self._position_open = False
            self._wave_rider.position = None
        elif not self._position_open and not self._orders.is_flat:
            # Unexpected position (NT8 source of truth)
            logger.warning("NT8 has position but engine thinks flat — syncing")
            self._position_open = True

    # ── Checkpoint Loading ────────────────────────────────────────────

    def _load_checkpoints(self):
        """Load all training checkpoints needed for live trading."""
        cpdir = self._cfg.checkpoint_dir
        logger.info(f"Loading checkpoints from {cpdir}/")

        # Pattern library + scaler
        lib_path = os.path.join(cpdir, 'pattern_library.pkl')
        scaler_path = os.path.join(cpdir, 'clustering_scaler.pkl')
        if not os.path.exists(lib_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Missing pattern_library.pkl or clustering_scaler.pkl in {cpdir}")

        with open(lib_path, 'rb') as f:
            self._pattern_library = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self._scaler = pickle.load(f)
        logger.info(f"  Library: {len(self._pattern_library)} templates")

        # Valid template IDs (must have centroids)
        self._valid_tids = [
            tid for tid in self._pattern_library
            if 'centroid' in self._pattern_library[tid]
        ]
        if not self._valid_tids:
            raise ValueError("No valid templates with centroids found")

        # Template tiers
        tiers_path = os.path.join(cpdir, 'template_tiers.pkl')
        if os.path.exists(tiers_path):
            with open(tiers_path, 'rb') as f:
                self._template_tier_map = pickle.load(f)
            logger.info(f"  Tiers: {len(self._template_tier_map)} templates")

        # Depth weights
        dw_path = os.path.join(cpdir, 'depth_weights.json')
        if os.path.exists(dw_path):
            with open(dw_path) as f:
                dw_data = json.load(f)
            self._depth_score_adj = {
                int(k): float(v.get('score_adj', 0.0))
                for k, v in dw_data.items()
            }
            self._depth_filter_out = {
                int(k) for k, v in dw_data.items()
                if v.get('filter_out', False)
            }
            logger.info(f"  Depth weights: {len(self._depth_score_adj)} depths, "
                         f"{len(self._depth_filter_out)} filtered out")

        # Build scaled centroid index
        centroids = np.array([
            self._pattern_library[tid]['centroid']
            for tid in self._valid_tids
        ])
        self._centroids_scaled = self._scaler.transform(centroids)
        logger.info(f"  Centroids: {len(self._valid_tids)} ready for matching")

        # Exception templates (data-quality override)
        for tid in self._valid_tids:
            lib = self._pattern_library.get(tid, {})
            if (lib.get('n_members', 0) >= 10
                    and lib.get('stats_win_rate', 0.0) >= 0.55
                    and (lib.get('regression_sigma_ticks') or 999) <= 10.0):
                self._exception_tids.add(tid)
        logger.info(f"  Exception templates: {len(self._exception_tids)}")

        # Brain
        brain_files = sorted(glob.glob(os.path.join(cpdir, '*_brain.pkl')))
        if brain_files:
            self._brain.load(brain_files[-1])
            logger.info(f"  Brain: {os.path.basename(brain_files[-1])}")
        else:
            logger.warning("  No brain checkpoint found — using empty brain")

    def _init_belief_network(self):
        """Initialize the fractal belief network from loaded checkpoints."""
        self._belief_network = TimeframeBeliefNetwork(
            pattern_library=self._pattern_library,
            scaler=self._scaler,
            engine=self._engine,
            valid_tids=self._valid_tids,
            centroids_scaled=self._centroids_scaled,
        )
        logger.info(f"  Belief network: {len(TimeframeBeliefNetwork.TIMEFRAMES_SECONDS)} TF workers")
