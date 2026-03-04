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

# Anchor TF → depth mapping (from OOS depth distribution)
# depth numbers match the fractal DNA tree levels in training
_ANCHOR_TF_MAP = {
    '1s':  {'period_s': 1,   'depth': 12},
    '5s':  {'period_s': 5,   'depth': 10},
    '15s': {'period_s': 15,  'depth': 8},
    '30s': {'period_s': 30,  'depth': 7},
    '1m':  {'period_s': 60,  'depth': 5},
    '3m':  {'period_s': 180, 'depth': 4},
    '5m':  {'period_s': 300, 'depth': 4},
}


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
    parent_type: str = 'STRUCTURAL_DRIVE'  # needed by extract_features
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

    def __init__(self, config: LiveConfig, dry_run: bool = False,
                 client=None, gui_queue=None, shared_state=None):
        self._cfg = config
        self._dry_run = dry_run
        self._client_override = client  # None = use NT8Client
        self._gui_queue = gui_queue     # None = headless
        self._shared_state = shared_state or {}  # mutable dict from launcher

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

        # Screening gates (loaded in _load_checkpoints)
        self._fission_map: Dict = {}
        self._good_hours_utc: set = set()

        # Anchor TF resolution
        _anchor = _ANCHOR_TF_MAP.get(config.anchor_tf, _ANCHOR_TF_MAP['15s'])
        self._anchor_period = _anchor['period_s']
        self._anchor_depth = _anchor['depth']
        self._anchor_tf = config.anchor_tf

        # Live components
        self._client = self._client_override if self._client_override is not None else NT8Client(config)
        self._aggregator = LiveBarAggregator(self._engine, config,
                                             target_period=self._anchor_period)
        self._orders = OrderManager(config)
        self._belief_network: Optional[TimeframeBeliefNetwork] = None

        # Position tracking (mirrors orchestrator forward pass)
        self._position_open = False
        self._entry_price = 0.0
        self._entry_time = 0.0
        self._entry_bar = 0
        self._active_side = ''
        self._active_tid = None
        self._entry_depth = '?'
        self._max_hold_bars = 960
        self._last_exit_reason = 'unknown'
        self._last_high_water = 0.0

        self._bar_i = 0   # running 15s bar index (NOT 1s ticks)
        self._last_states = []  # latest quantum states (refreshed every 15s)
        self._last_exit_time = 0.0  # cooldown between trades
        self._live_trade_count = 0  # for periodic brain save
        self._brain_save_interval = 5  # save brain every N trades

        # Ping-pong mode (continuous wave-riding with direction refinement)
        self._ping_pong_mode = self._shared_state.get('ping_pong', False)
        self._live_dir_bias: Dict[str, dict] = {}  # tid → {long_w, long_l, short_w, short_l}
        self._last_exit_side = ''  # side we just exited (for flip)
        self._pp_min_conviction = config.pp_min_conviction
        self._pp_agree_veto = config.pp_agree_veto
        self._pp_bias_min_trades = config.pp_bias_min_trades
        self._pp_bias_wr_good = config.pp_bias_wr_good
        self._pp_bias_wr_bad = config.pp_bias_wr_bad
        self._pp_sl_override = config.pp_sl_override
        self._pp_tp_override = config.pp_tp_override
        self._pp_trail_override = config.pp_trail_override
        self._pp_max_hold_override = config.pp_max_hold_bars
        self._pp_flip_count = 0
        self._pp_pending_flip = None  # deferred flip: {'side', 'price', 'ts'}
        self._pp_last_exit_params = None  # TF-scaled params from exited trade

        # GUI stats (for popup)
        self._session_pnl = 0.0
        self._session_wins = 0
        self._session_trades = 0
        self._gross_win = 0.0
        self._gross_loss = 0.0
        self._trade_log = []  # per-trade records for session report
        self._session_start = time.time()

        # Gate rejection counters (for session report)
        self._gate_stats = {
            'bars_seen': 0,
            'candidates': 0,
            'gate0_skip': 0,
            'gate0_5_skip': 0,
            'gate1_skip': 0,
            'gate2_skip': 0,
            'gate3_skip': 0,
            'gate3_5_skip': 0,
            'gate4_skip': 0,
            'traded': 0,
        }

        # Exit capture buckets (Optimal>=80%, Partial 20-80%, Early<20%, Reversed<=0%)
        self._exit_buckets = {'optimal': 0, 'partial': 0, 'early': 0, 'reversed': 0}

        # NT8 account equity (from ACCOUNT_UPDATE messages)
        self._nt8_cash_value = 0.0
        self._nt8_realized_pnl = 0.0
        self._nt8_unrealized_pnl = 0.0
        self._nt8_net_liquidation = 0.0

    # ── Public API ────────────────────────────────────────────────────

    async def run(self):
        """Main entry point — connect, load checkpoints, run loop."""
        from core.keep_awake import keep_awake

        logger.info("=" * 60)
        logger.info("LIVE ENGINE STARTING")
        logger.info(f"  Instrument: {self._cfg.instrument}")
        logger.info(f"  Account:    {self._cfg.account}")
        logger.info(f"  Anchor TF:  {self._anchor_tf}  (depth={self._anchor_depth}, period={self._anchor_period}s)")
        logger.info(f"  Dry run:    {self._dry_run}")
        logger.info(f"  Ping-pong:  {self._ping_pong_mode}")
        logger.info("=" * 60)

        self._load_checkpoints()
        self._init_belief_network()

        connected = await self._client.connect()
        if not connected:
            logger.error("Failed to connect to NT8 bridge — exiting")
            return

        with keep_awake(display=True):
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
                # Save brain on exit (preserve learning)
                if self._live_trade_count > 0:
                    brain_path = os.path.join(
                        self._cfg.checkpoint_dir, 'live_brain.pkl')
                    self._brain.save(brain_path)
                    logger.info(f"Live brain saved on exit ({self._live_trade_count} trades)")

    # ── Main Loop ─────────────────────────────────────────────────────

    async def _main_loop(self):
        """Process inbound messages from NT8."""
        while not self._client._stop:
            # Check if GUI requested shutdown (popup closed)
            if self._shared_state.get('shutdown'):
                logger.info("Shutdown requested by GUI — stopping engine")
                break

            # Prepare-for-shutdown: save brain + advise on position
            if self._shared_state.pop('prepare_shutdown', False):
                self._prepare_shutdown()

            try:
                msg = await asyncio.wait_for(
                    self._client.inbound.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue

            mtype = msg.get('type', '')

            if mtype == 'BAR':
                await self._on_bar(msg)
            elif mtype == 'FILL':
                pnl = self._orders.on_fill(msg)
                if pnl is not None:
                    self._brain_learn(pnl)
                self._sync_position_state()
            elif mtype == 'ORDER_STATUS':
                self._orders.on_order_status(msg)
            elif mtype == 'POSITION':
                self._orders.on_position(msg)
                self._sync_position_state()
            elif mtype == 'CONNECTED':
                bridge_ver = msg.get('version', '???')
                logger.info(f"NT8 CONNECTED: account={msg.get('account')}  bridge={bridge_ver}")
                if bridge_ver != '6.1.1':
                    logger.warning(f"BRIDGE VERSION MISMATCH: expected 6.1.1, got {bridge_ver}")
                self._gui_push({
                    'type': 'PHASE_PROGRESS',
                    'phase': 'LIVE',
                    'step': 'CONNECTED — warming up',
                    'pct': 0,
                })
            elif mtype == 'HISTORY_DONE':
                count = int(msg.get('bar_count', 0))
                logger.info(f"History dump complete: {count} bars from NT8")
                # Run heavy recompute in thread to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._aggregator.finish_history)
                # Drain any remaining stale BAR messages from duplicate
                # history requests that arrived before HISTORY_DONE
                drained = 0
                while not self._client.inbound.empty():
                    try:
                        stale = self._client.inbound.get_nowait()
                        if stale.get('type') != 'BAR':
                            # Re-queue non-BAR messages (CONNECTED, etc.)
                            await self._client.inbound.put(stale)
                            break
                        drained += 1
                    except Exception:
                        break
                if drained:
                    logger.info(f"Drained {drained} stale BAR messages from queue")
            elif mtype == 'ACCOUNT_UPDATE':
                self._on_account_update(msg)
            elif mtype == 'DOM':
                self._gui_push({
                    'type': 'DOM_UPDATE',
                    'bid': msg.get('bid'),
                    'ask': msg.get('ask'),
                })

    async def _on_bar(self, msg: dict):
        """Process a single inbound BAR message.

        Two-tier compute:
          - Every 1s bar: price ticker, exit checks (trail/SL/TP)
          - Every 15s bar: full state recompute + entry evaluation
        This ensures exits fire within 1 second of trigger, not up to 15s late.
        """
        # Only feed 15s and 1s bars to the aggregator.  Higher-TF bars
        # (30s, 1m, … 1D) from the NT8 history dump have different
        # timestamps that trigger spurious session-gap resets.
        bar_period = int(msg.get('bar_period_s', 1))
        # Accept 1s bars (for aggregation + exit checks) and anchor-TF bars
        if bar_period != 1 and bar_period != self._anchor_period:
            return

        price = float(msg['close'])
        ts = float(msg['timestamp'])

        # Run add_bar (which may trigger recompute) in thread to avoid
        # blocking the event loop for 1-3s during batch_compute_states
        loop = asyncio.get_event_loop()
        states = await loop.run_in_executor(None, self._aggregator.add_bar, msg)

        # 15s bar completed → save fresh states, advance bar counter
        new_bar = states is not None
        if new_bar:
            self._last_states = states
            self._bar_i += 1

        # ── Still warming up — show progress, skip evaluation ─────────
        if not self._aggregator.is_warmed_up:
            if new_bar and self._bar_i % 10 == 0:
                pct = self._aggregator.bar_count / max(1, self._cfg.warmup_bars) * 100
                self._gui_push({
                    'type': 'PHASE_PROGRESS',
                    'phase': 'LIVE',
                    'step': (f'warmup {self._aggregator.bar_count}'
                             f'/{self._cfg.warmup_bars}'),
                    'pct': min(99, pct),
                })
            return

        # ══ 1-SECOND PROCESSING (runs on every inbound bar) ══════════

        # Push live price to GUI ticker
        self._gui_push({
            'type': 'TICK_UPDATE',
            'price': price,
            'bars': self._bar_i,
        })

        # Safety: skip trading on stale data (>2 min old)
        age = time.time() - ts
        if age > 120:
            return  # stale bar from history leak, skip

        # Sync ping-pong toggle from GUI
        self._ping_pong_mode = self._shared_state.get('ping_pong', False)

        # Manual order check (from popup buttons)
        manual = self._shared_state.pop('manual_order', None)
        if manual:
            await self._handle_manual_order(manual, price, ts, self._last_states or [])
            return  # skip gate cascade this tick

        # Exit check EVERY SECOND (trail stop, SL, TP are price-dependent)
        if self._position_open:
            await self._check_exit(price, ts)

        # ══ 15-SECOND PROCESSING (only on fresh state recompute) ══════
        if not new_bar:
            return

        # Feed belief network (time-constants calibrated to 15s bars)
        df = self._aggregator.df
        if self._bar_i % 240 == 1:
            # Re-prepare day periodically (resamples higher TFs)
            self._belief_network.prepare_day(df, states_micro=states)

        self._belief_network.tick_all(self._bar_i)

        # Ping-pong deferred flip (scheduled by _check_exit, executes on next 15s bar)
        if self._pp_pending_flip and not self._position_open:
            flip = self._pp_pending_flip
            self._pp_pending_flip = None
            if not self._orders.loss_limit_hit:
                await self._enter_ping_pong(flip['side'], price, ts, states)
                return  # skip normal entry cascade this bar

        # Entry check (if flat + cooldown expired — one anchor bar minimum)
        _cooldown_ok = (time.time() - self._last_exit_time) > float(self._anchor_period)
        if not self._position_open and not self._orders.loss_limit_hit and _cooldown_ok:
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
            exited_side = pos.side
            self._belief_network.stop_trade_tracking()
            await self._close_position(reason)

            # Ping-pong: schedule flip entry on next anchor-TF bar
            if self._ping_pong_mode:
                self._schedule_ping_pong_flip(exited_side, price, ts)

    async def _close_position(self, reason: str):
        """Send close order and reset position state."""
        self._last_exit_side = self._active_side  # for ping-pong flip
        self._position_open = False
        self._last_exit_reason = reason  # for trade log
        # Snapshot high_water_mark before clearing position (for capture bucket)
        pos = self._wave_rider.position
        self._last_high_water = pos.high_water_mark if pos else self._entry_price
        self._wave_rider.position = None
        self._last_exit_time = time.time()

        if self._dry_run:
            logger.info(f"[DRY RUN] Would close position: {reason}")
            return

        msg = self._orders.build_exit_order(reason=reason)
        if msg:
            await self._client.send(msg)

    async def _handle_manual_order(self, action: str, price: float,
                                   ts: float, states: list):
        """Process a manual BUY/SELL/FLATTEN from the popup buttons."""
        if action == 'FLATTEN':
            if self._position_open:
                logger.info(f"MANUAL FLATTEN @ {price:.2f}")
                self._belief_network.stop_trade_tracking()
                await self._close_position('MANUAL_FLATTEN')
            else:
                logger.info("MANUAL FLATTEN — already flat")
            return

        if action not in ('BUY', 'SELL'):
            return

        # If already in a position, flatten first
        if self._position_open:
            logger.info(f"MANUAL {action}: flattening existing position first")
            self._belief_network.stop_trade_tracking()
            await self._close_position('MANUAL_REVERSE')

        side = 'long' if action == 'BUY' else 'short'

        # Use default exit params (generous for manual trades)
        sl_ticks = 20
        tp_ticks = 50
        trail_ticks = 12
        trail_act = 8

        logger.info(f"MANUAL ENTRY: {side.upper()} @ {price:.2f}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")

        # Get latest state for wave rider
        state = states[-1]['state'] if states else None

        self._wave_rider.open_position(
            entry_price=price, side=side,
            state=state,
            stop_distance_ticks=sl_ticks,
            profit_target_ticks=tp_ticks,
            trailing_stop_ticks=trail_ticks,
            trail_activation_ticks=trail_act,
            template_id='MANUAL',
        )
        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = 'MANUAL'
        self._entry_depth = self._anchor_depth
        self._max_hold_bars = 960  # 4 hours max for manual trades

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars)

        if self._dry_run:
            logger.info("[DRY RUN] Manual entry logged but no order sent")
            return

        order_msg = self._orders.build_entry_order(
            'BUY' if side == 'long' else 'SELL')
        if order_msg:
            await self._client.send(order_msg)

    # ── Ping-Pong Mode ──────────────────────────────────────────────

    def _schedule_ping_pong_flip(self, exited_side: str, price: float,
                                  ts: float):
        """After exit, check belief network and schedule a flip entry."""
        belief = self._belief_network.get_belief()
        if belief is None:
            logger.debug("PING-PONG: no belief available, skip flip")
            return

        flip_side = 'short' if exited_side == 'long' else 'long'

        # Side lock override (--long-only / --short-only)
        _side_lock = self._shared_state.get('side_lock')
        if _side_lock:
            flip_side = _side_lock

        # Use belief direction if it agrees with flip; otherwise trust the flip
        if belief.direction == exited_side and belief.conviction > self._pp_agree_veto:
            # Belief still agrees with old direction — don't flip
            logger.info(f"PING-PONG: belief still {belief.direction} "
                        f"(conv={belief.conviction:.2f}), skip flip")
            return

        if belief.conviction < self._pp_min_conviction:
            logger.info(f"PING-PONG: conviction {belief.conviction:.2f} "
                        f"< {self._pp_min_conviction}, skip flip")
            return

        # If belief has a strong opinion, use it
        if belief.conviction > 0.6:
            flip_side = belief.direction

        logger.info(f"PING-PONG: scheduling flip {exited_side}→{flip_side} "
                    f"(belief={belief.direction}, conv={belief.conviction:.2f})")
        self._pp_pending_flip = {'side': flip_side, 'price': price, 'ts': ts}

    async def _enter_ping_pong(self, side: str, price: float, ts: float,
                                states: list):
        """Lightweight entry for ping-pong flip — skip full gate cascade."""
        if not states:
            logger.debug("PING-PONG: no states for flip entry, skip")
            self._pp_pending_flip = None
            return

        # Check live direction bias (refinement cycle)
        tid = self._active_tid or 'MANUAL'
        bias = self._live_dir_bias.get(tid)
        if bias:
            dir_key = 'long' if side == 'long' else 'short'
            my_total = bias.get(f'{dir_key}_w', 0) + bias.get(f'{dir_key}_l', 0)
            if my_total >= self._pp_bias_min_trades:
                my_wr = bias.get(f'{dir_key}_w', 0) / my_total
                if my_wr < self._pp_bias_wr_bad:
                    # This direction loses — don't flip into it
                    logger.info(f"PING-PONG: live bias rejects {side.upper()} "
                                f"(WR={my_wr:.0%} on {my_total} trades)")
                    return

        latest = states[-1]
        state = latest['state']

        # Inherit TF-scaled exit params, with config overrides
        ep = self._pp_last_exit_params
        if ep:
            sl_ticks = self._pp_sl_override or ep['sl']
            tp_ticks = self._pp_tp_override or ep['tp']
            trail_ticks = self._pp_trail_override or ep['trail']
            trail_act = ep['trail_act']
            max_hold = self._pp_max_hold_override or ep['max_hold']
            tf_label = ep.get('tf', '?')
        else:
            sl_ticks = self._pp_sl_override or 15
            tp_ticks = self._pp_tp_override or 30
            trail_ticks = self._pp_trail_override or 10
            trail_act = 6
            max_hold = self._pp_max_hold_override or max(20, 3600 // self._anchor_period)
            tf_label = '?'

        self._pp_flip_count += 1
        logger.info(f"PING-PONG FLIP #{self._pp_flip_count}: {side.upper()} "
                    f"@ {price:.2f}  TF={tf_label}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")

        self._wave_rider.open_position(
            entry_price=price, side=side,
            state=state,
            stop_distance_ticks=sl_ticks,
            profit_target_ticks=tp_ticks,
            trailing_stop_ticks=trail_ticks,
            trail_activation_ticks=trail_act,
            template_id=f'PP_{tid}',
        )
        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = f'PP_{tid}'
        self._entry_depth = self._entry_depth  # carry depth from original entry
        self._max_hold_bars = max_hold

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars)

        if self._dry_run:
            logger.info("[DRY RUN] Ping-pong flip logged but no order sent")
            return

        order_msg = self._orders.build_entry_order(
            'BUY' if side == 'long' else 'SELL')
        if order_msg:
            await self._client.send(order_msg)

    def _direction_learn(self, tid, side: str, pnl: float):
        """Record direction-specific outcome for live refinement."""
        if tid is None:
            return
        # Strip PP_ prefix to aggregate with parent template
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid

        if base_tid not in self._live_dir_bias:
            self._live_dir_bias[base_tid] = {
                'long_w': 0, 'long_l': 0, 'short_w': 0, 'short_l': 0}

        bias = self._live_dir_bias[base_tid]
        key = side.lower()
        if pnl > 0:
            bias[f'{key}_w'] += 1
        else:
            bias[f'{key}_l'] += 1

        # Log the running tally
        lw, ll = bias['long_w'], bias['long_l']
        sw, sl = bias['short_w'], bias['short_l']
        lt = lw + ll
        st = sw + sl
        l_wr = f"{lw/lt:.0%}" if lt > 0 else "n/a"
        s_wr = f"{sw/st:.0%}" if st > 0 else "n/a"
        logger.info(f"DIR LEARN: tid={base_tid}  "
                    f"LONG {lw}W/{ll}L ({l_wr}) | SHORT {sw}W/{sl}L ({s_wr})")

    def _prepare_shutdown(self):
        """Save brain + write session report + advise GUI."""
        # Save brain
        if self._live_trade_count > 0:
            brain_path = os.path.join(
                self._cfg.checkpoint_dir, 'live_brain.pkl')
            self._brain.save(brain_path)
            logger.info(f"Brain saved ({self._live_trade_count} trades)")

        # Write session report
        self._write_session_report()

        # Check position status
        if self._position_open:
            unreal = self._nt8_unrealized_pnl
            side = self._active_side or '?'
            sign = '+' if unreal >= 0 else ''
            status = (f"OPEN {side.upper()} ({sign}${unreal:,.0f}) "
                      f"— FLATTEN first or close to lock in")
            logger.info(f"Prepare shutdown: {status}")
        else:
            status = "FLAT — saved — safe to close"
            logger.info("Prepare shutdown: no open position, safe to close")

        self._gui_push({'type': 'SHUTDOWN_READY', 'status': status})

    def _write_session_report(self):
        """Write session summary to reports/live/ (OOS-level detail)."""
        from collections import defaultdict

        report_dir = os.path.join('reports', 'live')
        os.makedirs(report_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(report_dir, f'session_{ts}.txt')

        dur = time.time() - self._session_start
        dur_h = int(dur // 3600)
        dur_m = int((dur % 3600) // 60)
        wr = (self._session_wins / self._session_trades * 100
              if self._session_trades > 0 else 0.0)
        pf = (self._gross_win / abs(self._gross_loss)
              if self._gross_loss != 0 else 0.0)
        avg = (self._session_pnl / self._session_trades
               if self._session_trades > 0 else 0.0)

        L = []
        L.append("=" * 72)
        L.append(f"LIVE SESSION REPORT  (run: {time.strftime('%Y-%m-%d %H:%M:%S')})")
        L.append(f"  Account:    {self._cfg.account}")
        L.append(f"  Instrument: {self._cfg.instrument}")
        L.append(f"  Anchor TF:  {self._anchor_tf}  (depth={self._anchor_depth})")
        L.append(f"  Duration:   {dur_h}h {dur_m}m  ({self._bar_i} bars)")
        L.append(f"Total Trades: {self._session_trades}")
        L.append(f"Win Rate: {wr:.1f}%")
        L.append(f"Total PnL: ${self._session_pnl:+,.2f}")
        L.append("=" * 72)

        # ── Per-depth PnL breakdown ──
        L.append("")
        L.append("=" * 72)
        L.append("PER-DEPTH PnL BREAKDOWN")
        L.append("=" * 72)
        if self._trade_log:
            depth_stats = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
            for t in self._trade_log:
                d = t.get('depth', '?')
                depth_stats[d]['n'] += 1
                depth_stats[d]['pnl'] += t['pnl']
                if t['pnl'] > 0:
                    depth_stats[d]['wins'] += 1
            L.append(f"  {'Depth':<10} {'Trades':>7} {'Win%':>6} {'Total PnL':>12} {'Avg/trade':>10}")
            L.append(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*12} {'-'*10}")
            for d in sorted(depth_stats.keys()):
                s = depth_stats[d]
                _wr = s['wins'] / s['n'] * 100 if s['n'] > 0 else 0
                _avg = s['pnl'] / s['n'] if s['n'] > 0 else 0
                L.append(f"  depth {str(d):<5} {s['n']:>7} {_wr:>5.0f}% ${s['pnl']:>10,.2f} ${_avg:>9,.2f}")

        # ── Exit reason breakdown ──
        L.append("")
        L.append("=" * 72)
        L.append("EXIT REASON BREAKDOWN")
        L.append("=" * 72)
        if self._trade_log:
            reason_stats = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
            for t in self._trade_log:
                r = t.get('reason', '?')
                reason_stats[r]['n'] += 1
                reason_stats[r]['pnl'] += t['pnl']
                if t['pnl'] > 0:
                    reason_stats[r]['wins'] += 1
            L.append(f"  {'Reason':<18} {'Trades':>7} {'Win%':>6} {'Total PnL':>12} {'Avg/trade':>10}")
            L.append(f"  {'-'*18} {'-'*7} {'-'*6} {'-'*12} {'-'*10}")
            for r in sorted(reason_stats.keys()):
                s = reason_stats[r]
                _wr = s['wins'] / s['n'] * 100 if s['n'] > 0 else 0
                _avg = s['pnl'] / s['n'] if s['n'] > 0 else 0
                L.append(f"  {r:<18} {s['n']:>7} {_wr:>5.0f}% ${s['pnl']:>10,.2f} ${_avg:>9,.2f}")

        # ── Direction breakdown ──
        L.append("")
        L.append("=" * 72)
        L.append("DIRECTION BREAKDOWN")
        L.append("=" * 72)
        if self._trade_log:
            for d in ('LONG', 'SHORT'):
                dt = [t for t in self._trade_log if t['side'] == d]
                if dt:
                    _dw = sum(1 for t in dt if t['pnl'] > 0)
                    _dp = sum(t['pnl'] for t in dt)
                    _dwr = _dw / len(dt) * 100
                    L.append(f"  {d:<8} {len(dt):>4} trades  WR={_dwr:.0f}%  "
                             f"PnL=${_dp:+,.2f}  Avg=${_dp/len(dt):+,.2f}")

        # ── Ping-pong direction refinement ──
        if self._pp_flip_count > 0 or self._live_dir_bias:
            L.append("")
            L.append("=" * 72)
            L.append("PING-PONG DIRECTION REFINEMENT")
            L.append("=" * 72)
            L.append(f"  Flip count: {self._pp_flip_count}")
            if self._live_dir_bias:
                L.append(f"  {'Template':<20} {'LONG WR':>10} {'LONG N':>8} "
                         f"{'SHORT WR':>10} {'SHORT N':>8}")
                L.append("  " + "-" * 58)
                for tid, b in sorted(self._live_dir_bias.items(),
                                     key=lambda x: sum(x[1].values()),
                                     reverse=True):
                    lw, ll = b['long_w'], b['long_l']
                    sw, sl = b['short_w'], b['short_l']
                    lt, st = lw + ll, sw + sl
                    l_wr = f"{lw/lt:.0%}" if lt > 0 else "n/a"
                    s_wr = f"{sw/st:.0%}" if st > 0 else "n/a"
                    L.append(f"  {str(tid):<20} {l_wr:>10} {lt:>8} "
                             f"{s_wr:>10} {st:>8}")

        # ── Gate rejection funnel ──
        L.append("")
        L.append("=" * 72)
        L.append("GATE REJECTION FUNNEL")
        L.append("=" * 72)
        gs = self._gate_stats
        _total = gs['candidates'] or 1
        L.append(f"  Bars with candidates:           {gs['bars_seen']:>8,}")
        L.append(f"  Total candidates evaluated:     {gs['candidates']:>8,}")
        _pct = lambda n: f"{n/_total*100:.1f}%"
        L.append(f"    Gate 0 (headroom/physics):     {gs['gate0_skip']:>8,}  ({_pct(gs['gate0_skip'])})")
        L.append(f"    Gate 0.5 (depth filter):       {gs['gate0_5_skip']:>8,}  ({_pct(gs['gate0_5_skip'])})")
        L.append(f"    Gate 1 (cluster dist):         {gs['gate1_skip']:>8,}  ({_pct(gs['gate1_skip'])})")
        L.append(f"    Gate 2 (brain rejected):       {gs['gate2_skip']:>8,}  ({_pct(gs['gate2_skip'])})")
        L.append(f"    Gate 3 (conviction):           {gs['gate3_skip']:>8,}  ({_pct(gs['gate3_skip'])})")
        L.append(f"    Gate 3.5 (screening/hours):    {gs['gate3_5_skip']:>8,}  ({_pct(gs['gate3_5_skip'])})")
        L.append(f"    Gate 4 (direction conf):       {gs['gate4_skip']:>8,}  ({_pct(gs['gate4_skip'])})")
        L.append(f"    Passed all gates -> traded:    {gs['traded']:>8,}  ({_pct(gs['traded'])})")

        # ── Session equity ──
        L.append("")
        L.append("=" * 72)
        L.append("ACCOUNT SNAPSHOT")
        L.append("=" * 72)
        L.append(f"  Cash Value:      ${self._nt8_cash_value:>12,.2f}")
        L.append(f"  Unrealized PnL:  ${self._nt8_unrealized_pnl:>+12,.2f}")
        L.append(f"  Net Liquidation: ${self._nt8_net_liquidation:>12,.2f}")
        L.append(f"  Profit Factor:   {pf:.2f}")
        L.append(f"  Avg PnL/trade:   ${avg:+,.2f}")
        L.append(f"  Gross Win:       ${self._gross_win:+,.2f}")
        L.append(f"  Gross Loss:      ${abs(self._gross_loss):,.2f}")

        # ── Trade log ──
        L.append("")
        L.append("=" * 72)
        L.append("TRADE LOG")
        L.append("=" * 72)
        if self._trade_log:
            L.append(f"  {'#':>3}  {'Time':<10} {'Side':<6} {'Entry':>10} "
                     f"{'Exit':>10} {'PnL':>10} {'Reason':<14} {'Bars':>5}")
            L.append("  " + "-" * 68)
            cum = 0.0
            for i, t in enumerate(self._trade_log, 1):
                cum += t['pnl']
                L.append(
                    f"  {i:>3}  {t['time']:<10} {t['side']:<6} "
                    f"{t['entry']:>10,.2f} {t['exit']:>10,.2f} "
                    f"${t['pnl']:>+9,.2f} {t['reason']:<14} {t['bars']:>5}")
            L.append("  " + "-" * 68)
            L.append(f"  {'':>3}  {'':10} {'':6} {'':10} {'TOTAL':>10} "
                     f"${self._session_pnl:>+9,.2f}")
        else:
            L.append("  No trades this session.")

        L.append("")
        L.append("=" * 72)

        with open(path, 'w') as f:
            f.write('\n'.join(L) + '\n')
        logger.info(f"Session report saved: {path}")

    def _gui_push(self, msg: dict):
        """Non-blocking push to GUI queue. Drop if full."""
        if self._gui_queue is None:
            return
        try:
            self._gui_queue.put_nowait(msg)
        except Exception:
            pass

    def _gui_push_stats(self):
        """Push current session stats to the popup."""
        wr = (self._session_wins / self._session_trades * 100
              if self._session_trades > 0 else 0.0)
        pf = (self._gross_win / abs(self._gross_loss)
              if self._gross_loss != 0 else 0.0)
        eb = self._exit_buckets
        self._gui_push({
            'type': 'PHASE_PROGRESS',
            'phase': 'LIVE',
            'step': f'trade {self._session_trades}',
            'pct': min(100, self._bar_i / max(1, self._cfg.warmup_bars) * 100),
            'pnl': self._session_pnl,
            'wr': wr,
            'trades': self._session_trades,
            'pf': pf,
            'exit_optimal': eb['optimal'],
            'exit_partial': eb['partial'],
            'exit_early': eb['early'],
            'exit_reversed': eb['reversed'],
            'gross_w': self._gross_win,
            'gross_l': abs(self._gross_loss),
        })

    def _on_account_update(self, msg: dict):
        """Handle ACCOUNT_UPDATE from NT8 — push equity to GUI."""
        self._nt8_cash_value = float(msg.get('cash_value', 0))
        self._nt8_realized_pnl = float(msg.get('realized_pnl', 0))
        self._nt8_unrealized_pnl = float(msg.get('unrealized_pnl', 0))
        self._nt8_net_liquidation = float(msg.get('net_liquidation', 0))

        self._gui_push({
            'type': 'ACCOUNT_UPDATE',
            'cash_value': self._nt8_cash_value,
            'realized_pnl': self._nt8_realized_pnl,
            'unrealized_pnl': self._nt8_unrealized_pnl,
            'net_liquidation': self._nt8_net_liquidation,
        })

    def _brain_learn(self, pnl: float):
        """Feed trade outcome to brain, update GUI stats, save periodically."""
        from core.bayesian_brain import TradeOutcome

        result = 'WIN' if pnl > 0 else 'LOSS'

        # Use actual fill prices from OrderManager when available
        exi = self._orders.last_exit_info
        entry_px = exi.get('entry_px', self._entry_price)
        exit_px = exi.get('exit_px', 0.0)
        if not exit_px:
            # Fallback: compute from PnL (side-aware)
            tick_val = getattr(self, '_cfg_tick_value', 5.0) or 5.0
            if self._active_side == 'short':
                exit_px = entry_px - pnl / tick_val
            else:
                exit_px = entry_px + pnl / tick_val

        outcome = TradeOutcome(
            state=self._active_tid or 'UNKNOWN',
            entry_price=entry_px,
            exit_price=exit_px,
            pnl=pnl,
            result=result,
            timestamp=time.time(),
            exit_reason=self._last_exit_reason,
            direction='LONG' if self._active_side == 'long' else 'SHORT',
            template_id=self._active_tid,
        )
        self._brain.update(outcome)
        self._direction_learn(self._active_tid, self._active_side, pnl)
        self._live_trade_count += 1

        # Update session stats for GUI
        self._session_pnl += pnl
        self._session_trades += 1
        if pnl > 0:
            self._session_wins += 1
            self._gross_win += pnl
        else:
            self._gross_loss += pnl

        # Capture bucket (MFE-based exit quality)
        hwm = getattr(self, '_last_high_water', entry_px)
        if self._active_side == 'long':
            mfe_ticks = (hwm - entry_px) / 0.25 if entry_px else 0
        else:
            mfe_ticks = (entry_px - hwm) / 0.25 if entry_px else 0
        pnl_ticks = pnl / 5.0  # MNQ $5/tick
        if mfe_ticks > 0:
            capture = pnl_ticks / mfe_ticks * 100
        else:
            capture = 0.0 if pnl <= 0 else 100.0
        if capture >= 80:
            self._exit_buckets['optimal'] += 1
        elif capture >= 20:
            self._exit_buckets['partial'] += 1
        elif capture > 0:
            self._exit_buckets['early'] += 1
        else:
            self._exit_buckets['reversed'] += 1

        # Record for session report
        self._trade_log.append({
            'time': time.strftime('%H:%M:%S'),
            'side': 'LONG' if self._active_side == 'long' else 'SHORT',
            'entry': entry_px,
            'exit': exit_px,
            'pnl': pnl,
            'result': result,
            'reason': self._last_exit_reason,
            'bars': self._bar_i - self._entry_bar,
            'tid': self._active_tid,
            'depth': self._entry_depth,
            'capture': capture,
        })

        logger.info(f"Brain learned: tid={self._active_tid} {result} "
                    f"${pnl:+.2f}  capture={capture:.0f}%  "
                    f"(table size: {len(self._brain.table)})")

        # Push stats to GUI popup
        self._gui_push_stats()
        self._gui_push({
            'type': 'DAY_PNL',
            'day': time.strftime('%H:%M'),
            'pnl': pnl,
            'trades': 1,
            'wins': 1 if pnl > 0 else 0,
        })

        # Save every N trades
        if self._live_trade_count % self._brain_save_interval == 0:
            brain_path = os.path.join(
                self._cfg.checkpoint_dir, 'live_brain.pkl')
            self._brain.save(brain_path)
            logger.info(f"Live brain saved ({self._live_trade_count} trades)")

    # ── Entry Logic (Gate 0 → 3) ─────────────────────────────────────

    async def _check_entry(self, price: float, ts: float, states: list):
        """Run the gate cascade on the current bar's quantum state."""
        if not states:
            return

        # Aggression scaling (0.0=SNIPER … 1.0=YOLO)
        agg = self._shared_state.get('aggression', 0.5)
        _yolo = agg >= 0.99
        _g1_dist = (float('inf') if _yolo
                    else _GATE1_DIST_THRESHOLD + agg * 10.0)  # YOLO=∞, else 4.5→14.5
        _g2_prob = 0.05 * (1.0 - agg)                    # 0.05 → 0.0
        _g4_dir  = 0.05 * (1.0 - agg)                    # 0.05 → 0.0
        _skip_screening = agg > 0.75                      # AGGRESSIVE+ skips 3.5

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
        # YOLO: if no physics triggers found, force a candidate anyway
        if not candidates and _yolo:
            candidates.append(self._build_candidate(
                'STRUCTURAL_DRIVE', state, price, ts))
            logger.debug(f"YOLO forced candidate @ {price:.2f}  "
                         f"z={state.z_score:.3f} v={state.particle_velocity:.3f}")

        if not candidates:
            return

        self._gate_stats['bars_seen'] += 1
        self._gate_stats['candidates'] += len(candidates)
        logger.debug(f"Gate cascade: {len(candidates)} candidate(s), agg={agg:.2f}")

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

            if not _data_override and not _yolo:
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
                self._gate_stats['gate0_skip'] += 1
                continue

            # ── Gate 0.5: Depth filter ────────────────────────────────
            if not _yolo and p.depth in self._depth_filter_out:
                self._gate_stats['gate0_5_skip'] += 1
                continue

            # ── Gate 1: Cluster matching ──────────────────────────────
            features = np.array([FractalClusteringEngine.extract_features(p)])
            feat_scaled = self._scaler.transform(features)
            dists = np.linalg.norm(self._centroids_scaled - feat_scaled, axis=1)
            nearest_idx = np.argmin(dists)
            dist = float(dists[nearest_idx])
            tid = self._valid_tids[nearest_idx]

            if dist >= _g1_dist:
                self._gate_stats['gate1_skip'] += 1
                logger.debug(f"Gate 1 reject: dist={dist:.2f} >= {_g1_dist:.2f}")
                continue

            logger.debug(f"Gate 1 pass: tid={tid} dist={dist:.2f}/{_g1_dist:.2f}")

            # ── Gate 2: Brain ─────────────────────────────────────────
            if not self._brain.should_fire(tid, min_prob=_g2_prob, min_conf=0.0):
                self._gate_stats['gate2_skip'] += 1
                logger.debug(f"Gate 2 reject: tid={tid} "
                             f"prob={self._brain.get_probability(tid):.3f} "
                             f"conf={self._brain.get_confidence(tid):.3f}")
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
        side, _p_long, _dir_src = self._determine_direction(best_candidate, best_tid)

        # Side lock override (--long-only / --short-only)
        _side_lock = self._shared_state.get('side_lock')
        if _side_lock:
            side = _side_lock
            _dir_src = f'locked_{_side_lock}'

        if belief is not None:
            if not belief.is_confident and agg < 0.75:
                self._gate_stats['gate3_skip'] += 1
                return  # conviction too low (skipped at AGGRESSIVE+)
            if belief.direction != side:
                side = belief.direction
            _network_tp = (max(4, int(round(belief.predicted_mfe)))
                           if belief.predicted_mfe > 2.0 else None)
        else:
            _network_tp = None

        # ── Gate 4: Direction confidence ─────────────────────────────
        _dir_conf = abs(_p_long - 0.5)
        if _dir_conf < _g4_dir:
            self._gate_stats['gate4_skip'] += 1
            logger.debug(f"Gate 4 reject: dir_conf={_dir_conf:.3f} < {_g4_dir:.3f} "
                         f"(src={_dir_src}, tid={best_tid})")
            return

        # ── Gate 3.5: Screening fission + hour filter ────────────────
        if not _skip_screening:
            if self._fission_map:
                _fission_rule = self._fission_map.get(best_tid)
                if _fission_rule and _fission_rule.get('action') == 'reject':
                    self._gate_stats['gate3_5_skip'] += 1
                    logger.debug(f"Gate 3.5 reject: fission rule for tid={best_tid}")
                    return
            if self._good_hours_utc:
                import datetime as _dt
                _hour_utc = _dt.datetime.utcnow().hour
                if _hour_utc not in self._good_hours_utc:
                    self._gate_stats['gate3_5_skip'] += 1
                    logger.debug(f"Gate 3.5 reject: hour {_hour_utc} not in good_hours_utc")
                    return

        # ── Exit sizing ───────────────────────────────────────────────
        lib_entry = self._pattern_library.get(best_tid, {})
        params = lib_entry.get('params', {})
        sl_ticks, trail_ticks, trail_act, tp_ticks = self._compute_exit_params(
            lib_entry, params, _network_tp, best_candidate)

        # ── Execute entry ─────────────────────────────────────────────
        self._gate_stats['traded'] += 1
        logger.info(f"ENTRY: {side.upper()} @ {price:.2f}  "
                    f"tid={best_tid}  dist={best_dist:.2f}  "
                    f"dir_src={_dir_src}  conf={_dir_conf:.3f}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}  "
                    f"agg={agg:.0%}")

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
        self._entry_depth = best_candidate.depth

        _tf_s = str(best_candidate.timeframe)
        _tf_sec = TIMEFRAME_SECONDS.get(_tf_s, 14400)
        self._max_hold_bars = max(20, _tf_sec // self._anchor_period)

        # Store TF-scaled exit params for ping-pong reuse
        self._pp_last_exit_params = {
            'sl': sl_ticks, 'tp': tp_ticks,
            'trail': trail_ticks, 'trail_act': trail_act,
            'max_hold': self._max_hold_bars, 'tf': _tf_s,
        }

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
            depth=self._anchor_depth,
            timeframe=self._anchor_tf,
            parent_type=pattern_type,  # same as self in live (no tree yet)
            timestamp=ts,
            price=price,
            idx=self._bar_i,
        )

    def _determine_direction(self, candidate: _LiveCandidate,
                             tid: str):
        """Determine trade direction from template library + live state.

        Returns (side, p_long, source) where p_long is the probability of
        long direction (0..1) and source is a debug label.
        """
        lib_entry = self._pattern_library.get(tid, {})
        long_bias = lib_entry.get('long_bias', 0.0)
        short_bias = lib_entry.get('short_bias', 0.0)
        _BIAS_THRESH = 0.55

        # Priority 0: live direction bias (refinement cycle)
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid
        bias = self._live_dir_bias.get(base_tid)
        if bias:
            lw, ll = bias.get('long_w', 0), bias.get('long_l', 0)
            sw, sl = bias.get('short_w', 0), bias.get('short_l', 0)
            lt, st = lw + ll, sw + sl
            _min_t = self._pp_bias_min_trades
            _wr_good = self._pp_bias_wr_good
            _wr_bad = self._pp_bias_wr_bad
            if lt >= _min_t or st >= _min_t:
                l_wr = lw / lt if lt > 0 else 0.5
                s_wr = sw / st if st > 0 else 0.5
                if l_wr > _wr_good and (st < 3 or s_wr < _wr_bad):
                    return 'long', 0.5 + l_wr * 0.4, 'live_bias'
                if s_wr > _wr_good and (lt < 3 or l_wr < _wr_bad):
                    return 'short', 0.5 - s_wr * 0.4, 'live_bias'

        # Priority 1: signed MFE regression (sign=direction, |val|=confidence)
        _live_feat = np.array(FractalClusteringEngine.extract_features(candidate))
        _live_scaled = self._scaler.transform([_live_feat])[0]

        _smfe_coeff = lib_entry.get('signed_mfe_coeff')
        if _smfe_coeff is not None:
            _pred_smfe = float(np.dot(_live_scaled, np.array(_smfe_coeff))
                               + lib_entry.get('signed_mfe_intercept', 0.0))
            side = 'long' if _pred_smfe > 0 else 'short'
            _p_long = 0.5 + min(abs(_pred_smfe) / 20.0, 0.45) * (1 if _pred_smfe > 0 else -1)
            return side, _p_long, 'signed_mfe'

        # Priority 2: balanced direction logistic regression
        _dir_coeff = lib_entry.get('dir_coeff')
        if _dir_coeff is not None:
            _dir_logit = (np.dot(_live_scaled, np.array(_dir_coeff))
                          + lib_entry.get('dir_intercept', 0.0))
            _p_long = 1.0 / (1.0 + np.exp(-np.clip(_dir_logit, -20, 20)))
            side = 'long' if _p_long > 0.5 else 'short'
            return side, _p_long, 'balanced_dir'

        # Priority 3: template aggregate bias
        if long_bias >= _BIAS_THRESH:
            return 'long', long_bias, 'template_bias'
        elif short_bias >= _BIAS_THRESH:
            return 'short', 1.0 - short_bias, 'template_bias'
        elif long_bias + short_bias >= 0.10:
            side = 'long' if long_bias >= short_bias else 'short'
            _p_long = long_bias / (long_bias + short_bias)
            return side, _p_long, 'template_bias'

        # Priority 4: live DMI (trend-following)
        s = candidate.state
        _dmi_diff = (getattr(s, 'dmi_plus', 0.0)
                     - getattr(s, 'dmi_minus', 0.0))
        if _dmi_diff != 0:
            side = 'long' if _dmi_diff > 0 else 'short'
            _p_long = 0.6 if side == 'long' else 0.4
            return side, _p_long, 'dmi_live'

        vel = getattr(s, 'particle_velocity', 0.0)
        side = 'long' if vel >= 0 else 'short'
        return side, 0.55 if side == 'long' else 0.45, 'velocity'

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

        # Phase 2: trail — sigma*2.5 captures normal noise in trending move;
        # floor at 8 ticks ($10/tick MNQ = $80 minimum breathing room).
        if _reg_sigma > 2.0:
            trail = max(8, int(round(_reg_sigma * 2.5)))
        elif _mean_mae > 2.0:
            trail = max(8, int(round(_mean_mae * 1.5)))
        else:
            trail = max(8, params.get('trailing_stop_ticks', 12))

        # Trail activation: 0.6× p25_mae ensures trade is meaningfully
        # in profit before trail takes over from hard SL.
        trail_act = (max(4, int(round(_p25_mae * 0.6)))
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

        # Pattern library
        lib_path = os.path.join(cpdir, 'pattern_library.pkl')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Missing pattern_library.pkl in {cpdir}")
        with open(lib_path, 'rb') as f:
            self._pattern_library = pickle.load(f)
        logger.info(f"  Library: {len(self._pattern_library)} templates")

        # Clustering scaler (for scaling live features before L2 matching)
        scaler_path = os.path.join(cpdir, 'clustering_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
            logger.info(f"  Scaler: loaded from clustering_scaler.pkl")
        else:
            # Fallback: fit scaler on library centroids (already scaled space)
            # This gives identity-like transform; works for matching but
            # won't perfectly standardize new features. Retrain with --fresh
            # to generate proper scaler.
            from sklearn.preprocessing import StandardScaler
            logger.warning("  clustering_scaler.pkl not found — reconstructing from centroids")
            _cents = [v['centroid'] for v in self._pattern_library.values()
                      if 'centroid' in v]
            if _cents:
                self._scaler = StandardScaler().fit(np.array(_cents))
            else:
                raise FileNotFoundError("No centroids in library and no scaler")

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

        # Build centroid index — centroids stored RAW, must scale for L2 matching
        _raw_centroids = np.array([
            self._pattern_library[tid]['centroid']
            for tid in self._valid_tids
        ])
        self._centroids_scaled = self._scaler.transform(_raw_centroids)
        logger.info(f"  Centroids: {len(self._valid_tids)} ready for matching")

        # Exception templates (data-quality override)
        for tid in self._valid_tids:
            lib = self._pattern_library.get(tid, {})
            if (lib.get('n_members', 0) >= 10
                    and lib.get('stats_win_rate', 0.0) >= 0.55
                    and (lib.get('regression_sigma_ticks') or 999) <= 10.0):
                self._exception_tids.add(tid)
        logger.info(f"  Exception templates: {len(self._exception_tids)}")

        # Brain — prefer live_brain.pkl (has live learning), then training brain
        live_brain_path = os.path.join(cpdir, 'live_brain.pkl')
        training_brains = sorted(glob.glob(os.path.join(cpdir, 'pattern_*_brain.pkl')))

        if os.path.exists(live_brain_path):
            self._brain.load(live_brain_path)
            logger.info(f"  Brain: live_brain.pkl ({len(self._brain.table)} states)")
        elif training_brains:
            self._brain.load(training_brains[-1])
            logger.info(f"  Brain: {os.path.basename(training_brains[-1])} (training base)")
        else:
            logger.warning("  No brain checkpoint found — starting fresh "
                          "(will learn from live trades)")

        # Screening gates (Gate 3.5 fission + temporal filter)
        sg_path = os.path.join(cpdir, 'screening_gates.json')
        if os.path.exists(sg_path):
            with open(sg_path) as f:
                sg_data = json.load(f)
            self._fission_map = sg_data.get('fission_map', {})
            self._good_hours_utc = set(sg_data.get('good_hours_utc', []))
            logger.info(f"  Screening gates: {len(self._fission_map)} fission rules, "
                         f"{len(self._good_hours_utc)} good hours")
        else:
            logger.warning("  No screening_gates.json — all signals unfiltered")

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
