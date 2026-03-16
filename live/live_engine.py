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
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from core.statistical_field_engine import StatisticalFieldEngine
from core.bayesian_brain import MarketBayesianBrain
from core.exit_engine import ExitEngine, ExitAction, PositionState
from core.execution_engine import (ExecutionEngine, ActionType, TradeAction)
from core.timeframe_belief_network import TimeframeBeliefNetwork
from core.checkpoint_loader import load_checkpoints
from core.engine_factory import create_belief_network, create_execution_engine
from core.bar_processor import BarProcessor
from live.exit_watcher import ExitWatcher
from live.gui_bridge import GUIBridge
from live.session_tracker import SessionTracker
from live.ping_pong import PingPongManager
from config.symbols import SYMBOL_MAP

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.bar_aggregator import LiveBarAggregator
from live.order_manager import OrderManager
from live.protocol import close_position
from live.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

# Timeframe seconds lookup (same as orchestrator)
TIMEFRAME_SECONDS = {
    '1D': 86400, '4h': 14400, '1h': 3600, '30m': 1800, '15m': 900,
    '5m': 300, '3m': 180, '2m': 120, '1m': 60, '30s': 30, '15s': 15,
    '5s': 5, '1s': 1,
}

# Pattern Quality constants
_ADX_TREND_CONFIRMATION = 25.0
_HURST_TREND_CONFIRMATION = 0.6
_GATE1_DIST_THRESHOLD = 4.5  # Template Match distance threshold
_WORKER_BYPASS_CONV = 0.65

# Anchor TF → depth mapping (from OOS depth distribution)
# depth numbers match the fractal DNA tree levels in training
_TUNING_DEFAULTS = {
    '_comment': 'Edit while live — engine hot-reloads every 20 bars (~5 min)',
    'max_hold_seconds': 300,
    'manual_sl': 0,
    'manual_tp': 0,
    'manual_trail': 0,
    'manual_trail_act': 0,
    'sl_offset': 0,
    'tp_offset': 0,
    'trail_offset': 0,
    'trail_act_offset': 0,
    'pp_sl': 0,
    'pp_tp': 0,
    'pp_trail': 0,
    'pp_max_hold_seconds': 0,
    'gate1_dist': 4.5,
    'gate0_adx': 25.0,
    'gate0_hurst': 0.6,
    'exit_sl_mult': 3.0,
    'exit_trail_mult': 2.5,
    'exit_trail_act_mult': 0.6,
    'exit_tp_mult': 5.0,
    'min_tick_floor': 4,
    'envelope_halflife_bars': 20,
    'envelope_floor_ticks': 4,
    'envelope_accel_sensitivity': 1.0,
    'auto_tp_reentry': True,
}

_ANCHOR_TF_MAP = {
    '1s':  {'period_s': 1,   'depth': 12},
    '5s':  {'period_s': 5,   'depth': 10},
    '15s': {'period_s': 15,  'depth': 8},
    '30s': {'period_s': 30,  'depth': 7},
    '1m':  {'period_s': 60,  'depth': 5},
    '3m':  {'period_s': 180, 'depth': 4},
    '5m':  {'period_s': 300, 'depth': 4},
}


# _live_features() DELETED — use BarProcessor._build_features() instead


class LiveEngine:
    """Main live trading loop — replaces Phase 4 forward pass for real-time."""

    def __init__(self, config: LiveConfig,
                 client=None, gui_queue=None, shared_state=None):
        self._cfg = config
        self._client_override = client  # None = use NT8Client
        self._shared_state = shared_state or {}  # mutable dict from launcher

        # Core components (loaded from checkpoints)
        self._asset = SYMBOL_MAP.get(config.asset_ticker)
        if self._asset is None:
            raise ValueError(f"Unknown asset ticker: {config.asset_ticker}")

        self._engine = StatisticalFieldEngine()
        self._brain = MarketBayesianBrain()
        self._position = None  # PositionState or None

        # Unified exit engine (same logic as training — training/live parity)
        self._exit_engine = ExitEngine(
            mode='live',
            tick_size=self._asset.tick_size,
            tick_value=self._asset.tick_size * self._asset.point_value,
        )
        self._pos_state = None  # ExitEngine PositionState for current trade

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

        # Multi-TF bar buffers for TBN workers (populated from NT8 bridge)
        self._tf_bars: Dict[int, list] = {5: [], 14400: []}

        # Position tracking
        self._position_open = False
        self._entry_price = 0.0
        self._entry_time = 0.0
        self._entry_bar = 0
        self._active_side = ''
        self._predicted_mfe_ticks = 0.0
        self._price_expected = 0.0
        self._active_tid = None
        self._entry_depth = '?'
        self._max_hold_bars = 960
        self._last_exit_reason = 'unknown'
        self._last_high_water = 0.0

        self._primary_period = config.base_resolution_s
        self._bar_i = 0
        self._last_states = []
        self._last_price = 0.0
        self._last_ts = 0.0
        self._last_exit_time = 0.0
        self._last_1s_tick = 0.0
        self._order_send_ts = 0.0
        self._live_trade_count = 0
        self._brain_save_interval = 5
        self._shutting_down = False
        self._instrument_mismatch = False
        self._entry_belief_pct = 0
        self._exit_belief_pct = 100

        # Hot-reloadable tuning (loaded in _load_checkpoints, refreshed every 20 bars)
        self._tuning = dict(_TUNING_DEFAULTS)
        self._tuning_mtime = 0.0

        # Live ATR — computed from actual bar data
        self._live_atr_ticks = 0.0

        # ── Extracted subsystems ──
        self._gui = GUIBridge(gui_queue)
        self._session = SessionTracker(config)
        self._exit_watcher = ExitWatcher(self._asset.tick_size,
                                         self._asset.point_value)
        self._pp = PingPongManager(config, self._tuning)
        self._trade_logger = TradeLogger()

        # Ping-pong state (kept on LiveEngine — guards NT8 order lifecycle)
        self._ping_pong_mode = self._shared_state.get('ping_pong', False)
        self._last_exit_side = ''
        self._pp_min_conviction = config.pp_min_conviction
        self._pp_agree_veto = config.pp_agree_veto
        self._flip_in_progress = False
        self._pending_manual_entry = None
        self._pp_last_exit_params = None

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
        logger.info(f"  Ping-pong:  {self._ping_pong_mode}")
        logger.info("=" * 60)

        self._load_checkpoints()
        self._init_belief_network()
        self._init_exec_engine()
        self._init_bar_processor()

        # ── Connect to NT8 ────────────────────────────────────────────────
        # Brain warm from OOS3 (live_brain.pkl).
        # TBN warmed from NT8's 10k bar history dump (in HISTORY_DONE handler).
        last_ts = self._aggregator.load_from_parquet()
        if last_ts > 0:
            self._client.set_resume_timestamp(last_ts)
            logger.info(f"Delta sync enabled: will request bars after ts={last_ts:.0f}")

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
                # Safety: close any open position (no-op if graceful shutdown already flattened)
                if self._position_open:
                    msg = self._orders.build_exit_order(reason='shutdown')
                    if msg:
                        await self._client.send(msg)
                # Safety: save brain (no-op if _prepare_shutdown already saved)
                if self._live_trade_count > 0:
                    brain_path = os.path.join(
                        self._cfg.checkpoint_dir, 'live_brain.pkl')
                    self._brain.save(brain_path)
                    logger.info(f"Live brain saved on exit ({self._live_trade_count} trades)")
                # Safety: persist bars for delta sync on next start
                self._aggregator.save_to_parquet()
                # Disconnect from NT8
                await self._client.disconnect()
                logger.info("NT8 disconnected — shutdown complete")
                # NOW signal GUI that everything is done
                self._shared_state['shutdown_confirmed'] = True

    # ── Main Loop ─────────────────────────────────────────────────────

    async def _main_loop(self):
        """Process inbound messages from NT8."""
        while not self._client._stop:
            # Check if GUI requested shutdown (popup closed)
            if self._shared_state.get('shutdown'):
                logger.info("Shutdown requested by GUI -- stopping engine")
                break

            # Graceful shutdown: flatten → disable trading → confirm → exit
            if self._shared_state.pop('shutdown_flatten', False):
                self._shutting_down = True
                self._ping_pong_mode = False  # kill PP immediately
                self._belief_network.stop_trade_tracking()
                if self._position_open:
                    logger.info("SHUTDOWN: flattening position before close...")
                    await self._close_position('SHUTDOWN')
                else:
                    logger.info("SHUTDOWN: already flat")
                    self._prepare_shutdown()
                    break  # exit main loop -> finally: disconnect -> confirmed

            # Prepare-for-shutdown: save brain + advise on position
            if self._shared_state.pop('prepare_shutdown', False):
                self._prepare_shutdown()

            # Sync ping-pong toggle from GUI
            self._ping_pong_mode = self._shared_state.get('ping_pong', False)

            # Unlock daily loss limit (from GUI button)
            if self._shared_state.pop('unlock_loss_limit', False):
                self._orders.reset_loss_limit()
                logger.warning("DAILY LOSS LIMIT UNLOCKED by user")
                self._gui.push({'type': 'LOSS_LIMIT', 'locked': False,
                                'daily_pnl': self._orders.daily_pnl})

            # Manual order — checked every loop cycle (instant response)
            manual = self._shared_state.pop('manual_order', None)
            if manual:
                _last_px = getattr(self, '_last_price', 0.0)
                _last_ts = getattr(self, '_last_ts', time.time())
                await self._handle_manual_order(
                    manual, _last_px, _last_ts, self._last_states or [])

            # Every ~1s: exit protection + belief compute + GUI push
            _now = time.time()
            if _now - self._last_1s_tick >= 1.0 and self._aggregator.is_warmed_up:
                self._last_1s_tick = _now
                # Exit check between bars — catches SL/TP/trail within 1s
                if self._position_open and self._last_price > 0:
                    try:
                        await self._check_exit(self._last_price, _now)
                    except Exception as _exit_err:
                        logger.error(f"_check_exit CRASHED (1s loop): {_exit_err} — emergency flatten")
                        await self._close_position('EXIT_CRASH')
                # Safety net: NT8 has position but engine thinks flat — emergency exit calc
                elif (not self._position_open and not self._orders.is_flat
                      and not self._flip_in_progress and self._last_price > 0):
                    _om_pos = self._orders.position
                    _om_side = _om_pos.side if _om_pos else '?'
                    _om_px = _om_pos.avg_price if _om_pos else 0
                    _tick = self._cfg.tick_size
                    _pv = self._cfg.point_value
                    if _om_side == 'LONG':
                        _unreal = (self._last_price - _om_px) * _pv
                    elif _om_side == 'SHORT':
                        _unreal = (_om_px - self._last_price) * _pv
                    else:
                        _unreal = 0
                    # Emergency flatten if orphan position loses > $20
                    if _unreal < -20:
                        logger.error(f"ORPHAN POSITION: {_om_side} @ {_om_px} "
                                     f"unreal=${_unreal:+.0f} — emergency flatten")
                        await self._close_position('ORPHAN_FLATTEN')
                    elif not hasattr(self, '_orphan_warn_ts') or _now - self._orphan_warn_ts > 30:
                        self._orphan_warn_ts = _now
                        logger.warning(f"ORPHAN POSITION detected: {_om_side} @ {_om_px} "
                                       f"unreal=${_unreal:+.0f} — monitoring")
                # Legacy PP deferred flip fallback (instant flip handles most cases)
                _pf = self._pp.consume_pending()
                if _pf and not self._position_open and self._orders.is_flat:
                    await self._enter_ping_pong(
                        _pf['exited_side'], self._last_price, _now,
                        self._last_states or [])
                self._compute_life_pct()
                self._gui.push_stats(
                    session_pnl=self._session.stats.pnl,
                    session_wins=self._session.stats.wins,
                    session_trades=self._session.stats.trades,
                    gross_win=self._session.stats.gross_win,
                    gross_loss=self._session.stats.gross_loss,
                    exit_buckets=self._session.stats.exit_buckets,
                    belief_pct=self._exit_belief_pct if self._position_open else self._entry_belief_pct,
                    in_position=self._position_open,
                    daily_pnl=self._orders.daily_pnl,
                )

            try:
                msg = await asyncio.wait_for(
                    self._client.inbound.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            mtype = msg.get('type', '')

            if mtype == 'BAR':
                await self._on_bar(msg)
            elif mtype == 'PARTIAL_BAR':
                self._on_partial_bar(msg)
            elif mtype == 'FILL':
                if self._order_send_ts:
                    _rt_ms = (time.perf_counter() - self._order_send_ts) * 1000
                    logger.info(f"LATENCY: fill_rtt={_rt_ms:.1f}ms  (order sent->fill)")
                    self._order_send_ts = 0.0
                pnl = self._orders.on_fill(msg)
                if pnl is not None:
                    exi = self._orders.last_exit_info
                    self._gui.push({
                        'type': 'TRADE_MARKER', 'action': 'exit',
                        'side': exi.get('side', self._active_side),
                        'price': exi.get('exit_px', 0),
                        'pnl': pnl,
                    })
                    self._brain_learn(pnl)
                    # Start post-exit counterfactual watcher
                    self._exit_watcher.add(
                        tid=self._active_tid,
                        side=exi.get('side', self._active_side),
                        entry_px=exi.get('entry_px', self._entry_price),
                        exit_px=exi.get('exit_px', 0),
                        exit_pnl=pnl,
                        reason=self._last_exit_reason,
                    )
                self._sync_position_state()
                # Fire deferred manual entry now that position is flat
                if self._pending_manual_entry and self._orders.is_flat:
                    _pm = self._pending_manual_entry
                    self._pending_manual_entry = None
                    if self._orders.loss_limit_hit:
                        logger.warning("Deferred manual entry cancelled — daily loss limit hit")
                    else:
                        fill_px = msg.get('fill_price', _pm['price'])
                        logger.info(f"Deferred manual entry firing (flat confirmed @ {fill_px})")
                        await self._execute_manual_entry(
                            _pm['action'], fill_px, _pm['ts'], _pm['states'])
                # Graceful shutdown: confirm flat to GUI, then exit loop
                if self._shutting_down and self._orders.is_flat:
                    logger.info("SHUTDOWN: NT8 confirmed flat")
                    self._prepare_shutdown()
                    break  # exit main loop -> finally: disconnect -> confirmed
            elif mtype == 'ORDER_STATUS':
                self._orders.on_order_status(msg)
                # Retry close if exit was rejected (position still open in NT8)
                if self._orders.exit_rejected:
                    self._orders.exit_rejected = False
                    logger.warning("Retrying CLOSE_POSITION after rejection...")
                    await asyncio.sleep(0.5)
                    await self._client.send(
                        close_position(self._cfg.instrument, self._cfg.account))
            elif mtype == 'POSITION':
                self._orders.on_position(msg)
                self._sync_position_state()
                # Auto-flatten leftover position from previous session
                if not self._aggregator.is_warmed_up and not self._orders.is_flat:
                    logger.warning("STALE POSITION detected during warmup — auto-flattening")
                    await self._client.send(
                        close_position(self._cfg.instrument, self._cfg.account))
            elif mtype == 'CONNECTED':
                bridge_ver = msg.get('version', '???')
                bridge_inst = msg.get('instrument', '')
                # Read primary chart period from bridge (v6.4+)
                self._primary_period = int(msg.get('primary_period_s',
                                                   self._cfg.base_resolution_s))
                logger.info(f"NT8 CONNECTED: account={msg.get('account')}  "
                            f"instrument={bridge_inst}  bridge={bridge_ver}  "
                            f"primary={self._primary_period}s")
                # Prepare aggregator for history ingestion (handles NT8 restarts)
                if self._aggregator.bar_count > 0:
                    # Delta sync: keep existing bars, just enter history mode
                    # for the incoming delta bars
                    self._aggregator._history_mode = True
                    self._client.set_resume_timestamp(self._aggregator.last_timestamp)
                    logger.info(f"Delta sync: keeping {self._aggregator.bar_count:,} bars, "
                                f"requesting from ts={self._aggregator.last_timestamp:.0f}")
                else:
                    # Full reset — no persisted data
                    self._aggregator.reset()
                    self._aggregator._history_mode = True
                    logger.info("Aggregator reset for full history ingestion")
                # Instrument handshake — compare root symbol (MNQ, ES, etc.)
                # NT8 may send "MNQ MAR26" while config has "MNQ 03-26"
                _cfg_root = self._cfg.asset_ticker.upper()  # "MNQ"
                _bridge_root = bridge_inst.split()[0].upper() if bridge_inst else ""
                if _bridge_root and _cfg_root != _bridge_root:
                    logger.error(
                        f"INSTRUMENT MISMATCH: engine expects '{_cfg_root}' "
                        f"but NT8 chart is '{bridge_inst}' -- REFUSING TO TRADE")
                    self._instrument_mismatch = True
                    self._gui.push({
                        'type': 'PHASE_PROGRESS',
                        'phase': 'LIVE',
                        'step': f'WRONG INSTRUMENT: {bridge_inst}',
                        'pct': 0,
                    })
                else:
                    self._instrument_mismatch = False
                self._gui.push({
                    'type': 'PHASE_PROGRESS',
                    'phase': 'LIVE',
                    'step': 'CONNECTED — warming up',
                    'pct': 0,
                })
            elif mtype == 'HISTORY_DONE':
                count = int(msg.get('bar_count', 0))
                logger.info(f"History dump complete: {count} bars from NT8")
                self._gui.push({
                    'type': 'PHASE_PROGRESS',
                    'phase': 'LIVE',
                    'step': f'computing states ({self._aggregator.bar_count:,} bars)',
                    'pct': 50,
                })
                # Run heavy recompute in thread to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._aggregator.finish_history)
                self._update_live_atr()
                logger.info(f"Live ATR: {self._live_atr_ticks:.1f} ticks")
                # Bootstrap TBN from history — native NT8 bars for each TF
                df = self._aggregator.df
                states = self._aggregator.states
                df_5s = pd.DataFrame(self._tf_bars.get(5, [])) if self._tf_bars.get(5) else pd.DataFrame()
                df_4h = pd.DataFrame(self._tf_bars.get(14400, [])) if self._tf_bars.get(14400) else pd.DataFrame()
                _n5 = len(df_5s)
                _n4h = len(df_4h)
                logger.info(f"TBN bootstrap: 5s={_n5} bars, 4h={_n4h} bars (native from NT8)")
                self._belief_network.prepare_day(
                    df, states_micro=states, df_5s=df_5s, df_4h=df_4h)
                # Tick through all history bars so workers form beliefs
                for _hist_i in range(len(states)):
                    self._belief_network.tick_all(_hist_i)
                logger.info(f"TBN warmed: ticked {len(states)} bars from NT8 history")
                # Update ATLAS with NT8 bars so OOS data stays current
                self._aggregator.update_atlas()
                self._gui.push({
                    'type': 'PHASE_PROGRESS',
                    'phase': 'LIVE',
                    'step': f'READY — ATR {self._live_atr_ticks:.0f}t  ({self._aggregator.bar_count:,} bars)',
                    'pct': 100,
                })
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
                self._gui.push({
                    'type': 'DOM_UPDATE',
                    'bid': msg.get('bid'),
                    'ask': msg.get('ask'),
                })

    def _on_partial_bar(self, msg: dict):
        """Handle PARTIAL_BAR from NT8 — update TBN worker with forming bar."""
        if not self._belief_network:
            return
        bar_period = int(msg.get('bar_period_s', 0))
        if bar_period == 0:
            return
        bar_data = {
            'timestamp': float(msg['timestamp']),
            'open':      float(msg['open']),
            'high':      float(msg['high']),
            'low':       float(msg['low']),
            'close':     float(msg['close']),
            'volume':    float(msg.get('volume', 0)),
        }
        self._belief_network.update_partial(bar_period, bar_data)

    async def _on_bar(self, msg: dict):
        """Route inbound BAR to 1s or 15s processing."""
        bar_period = int(msg.get('bar_period_s', 1))

        # Capture multi-TF bars for TBN workers (5s, 4h, etc.)
        if bar_period in self._tf_bars:
            self._tf_bars[bar_period].append({
                'timestamp': float(msg['timestamp']),
                'open':      float(msg['open']),
                'high':      float(msg['high']),
                'low':       float(msg['low']),
                'close':     float(msg['close']),
                'volume':    float(msg.get('volume', 0)),
            })

        # Only feed primary chart bars and anchor-TF bars to the aggregator
        if bar_period != self._primary_period and bar_period != self._anchor_period:
            return

        price = float(msg['close'])
        ts = float(msg['timestamp'])
        self._last_price = price
        self._last_ts = ts
        self._last_bar_high = float(msg.get('high', price))
        self._last_bar_low = float(msg.get('low', price))

        # Run add_bar (may trigger state recompute) in thread
        loop = asyncio.get_event_loop()
        states = await loop.run_in_executor(None, self._aggregator.add_bar, msg)

        new_bar = states is not None
        if new_bar:
            self._last_states = states
            self._bar_i += 1
            self._update_live_atr()
            if self._bar_i % 20 == 0:
                self._load_tuning()
                self._orders.cleanup_stale_orders()

        # History ingestion — progress bar only
        if self._aggregator._history_mode:
            _bc = self._aggregator.bar_count
            if _bc % 1000 == 0 and _bc > 0:
                self._gui.push({
                    'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
                    'step': f'loading history {_bc:,} bars',
                    'pct': min(45, _bc / 500),
                })
            return

        # Warmup — show progress, skip evaluation
        if not self._aggregator.is_warmed_up:
            if new_bar and self._bar_i % 10 == 0:
                pct = self._aggregator.bar_count / max(1, self._cfg.warmup_bars) * 100
                self._gui.push({
                    'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
                    'step': (f'warmup {self._aggregator.bar_count}'
                             f'/{self._cfg.warmup_bars}'),
                    'pct': min(99, pct),
                })
            return

        # Every bar: exits + GUI tick
        await self._process_1s(price, ts)

        # 15s bars only: entries + TBN
        if new_bar:
            await self._process_15s(price, ts, states)

    async def _process_1s(self, price: float, ts: float):
        """Sub-second processing: GUI tick, staleness check, exit checks."""
        self._gui.push({
            'type': 'TICK_UPDATE', 'price': price, 'bars': self._bar_i,
        })

        if time.time() - ts > 120:
            return  # stale bar from history leak

        if self._position_open:
            try:
                await self._check_exit(price, ts)
            except Exception as _exit_err:
                logger.error(f"_check_exit CRASHED: {_exit_err} — emergency flatten")
                await self._close_position('EXIT_CRASH')

    async def _process_15s(self, price: float, ts: float, states: list):
        """Per-anchor-bar processing: BarProcessor handles entry + exit."""
        if self._bar_i % 240 == 1:
            df_1s = self._aggregator.df_1s
            df_5s = pd.DataFrame(self._tf_bars.get(5, [])) if self._tf_bars.get(5) else pd.DataFrame()
            df_4h = pd.DataFrame(self._tf_bars.get(14400, [])) if self._tf_bars.get(14400) else pd.DataFrame()
            self._belief_network.prepare_day(
                self._aggregator.df, states_micro=states,
                df_5s=df_5s, df_1s=df_1s, df_4h=df_4h)

        # Block entries during maintenance or loss limit
        _cooldown_ok = (time.time() - self._last_exit_time) > float(self._anchor_period)
        _can_enter = (not self._position_open
                      and not self._orders.loss_limit_hit
                      and _cooldown_ok
                      and not self._instrument_mismatch
                      and not self._exit_engine.is_maintenance_window(ts))

        # Aggression scaling
        agg = self._shared_state.get('aggression', 0.5)
        _yolo = agg >= 0.99
        _side_lock = self._shared_state.get('side_lock')

        # Get latest state
        state = states[-1]['state'] if states else None
        if state is None:
            return

        # ── UNIFIED: BarProcessor handles BOTH entry and exit ──
        result = self._processor.process_bar(
            bar_index=self._bar_i,
            price=price,
            bar_high=getattr(self, '_last_bar_high', price),
            bar_low=getattr(self, '_last_bar_low', price),
            timestamp=ts,
            state=state,
            pp_dir_override=_side_lock if _side_lock else None,
            yolo=_yolo,
        )

        # Route BarResult to live order management
        if result.action.type == ActionType.ENTER and _can_enter:
            await self._execute_entry(result.action, price, ts, time.perf_counter())
        elif result.action.type == ActionType.EXIT and self._position_open:
            reason = getattr(result.action, 'exit_reason', 'unknown')
            exited_side = self._position.side if self._position else 'flat'
            self._belief_network.stop_trade_tracking()

            if (reason == 'profit_target'
                    and self._tuning.get('auto_tp_reentry', False)
                    and not self._shutting_down):
                await self._auto_tp_reentry(exited_side, price, ts)
            elif self._ping_pong_mode and not self._shutting_down:
                await self._flip_position(reason, exited_side, price, ts)
            else:
                await self._close_position(reason)

        self._exit_watcher.tick(price)

    # ── Exit Logic ────────────────────────────────────────────────────

    def _compute_life_pct(self):
        """Compute trade life % (100%=fresh, 0%=exit imminent). Cheap — runs every second.

        PnL-anchored: unrealized PnL dominates. A losing trade can't show high life.
          - PnL health (50%): 100% at TP, 0% at SL, linear between
          - Trail health (20%): how far from current trail stop
          - Conviction (15%): belief network strength
          - Alignment (15%): belief direction matches trade side
        """
        pos = self._position
        if pos is None or not self._position_open:
            return
        price = self._last_price
        if price <= 0:
            return

        _tick = self._cfg.tick_size

        # PnL health — where are we between SL and TP?
        if pos.side == 'long':
            profit_ticks = (price - pos.entry_price) / _tick
        else:
            profit_ticks = (pos.entry_price - price) / _tick
        sl_ticks = abs(pos.entry_price - pos.stop_loss) / _tick if pos.stop_loss else 80
        tp_ticks = abs(pos.profit_target - pos.entry_price) / _tick if pos.profit_target else 200
        # Scale: -1 at SL, 0 at entry, +1 at TP
        _range = sl_ticks + tp_ticks
        _pnl_health = max(0, min(1, (profit_ticks + sl_ticks) / max(1, _range)))

        # Trail health — distance from current stop (tightened by trail)
        if pos.stop_loss and pos.side == 'long':
            _trail_dist = (price - pos.stop_loss) / _tick
        elif pos.stop_loss:
            _trail_dist = (pos.stop_loss - price) / _tick
        else:
            _trail_dist = sl_ticks
        _trail_health = max(0, min(1, _trail_dist / max(1, sl_ticks)))

        # Conviction + alignment from belief network
        exit_sig = self._belief_network.get_exit_signal(pos.side, pos.entry_price)
        _conviction = exit_sig.get('conviction', 0.5)
        _aligned = 1.0
        belief = self._belief_network.get_belief()
        if belief and belief.direction != pos.side:
            _aligned = 0.3

        _life_pct = (
            _pnl_health * 50
            + _trail_health * 20
            + _conviction * 15
            + _aligned * 15
        )
        self._exit_belief_pct = max(0, min(100, _life_pct))

    async def _check_exit(self, price: float, ts: float):
        """Check for exit signals on the current bar — Unified ExitEngine."""
        pos = self._position
        if pos is None or self._pos_state is None:
            return

        # Update trade pace cache for TBN exit signals
        _tp = self._belief_network.get_trade_progress(
            price, tick_size=self._cfg.tick_size)
        self._belief_network._trade_pace_cache = _tp
        self._belief_network._trade_pace_blend = _tp.get('pace', 1.0) - 1.0

        # CME maintenance cutoff: flatten before daily halt (16:45-18:00 ET)
        if self._exit_engine.is_maintenance_window(ts):
            logger.warning("MAINTENANCE FLAT: closing position before CME halt")
            await self._close_position('maintenance_flat')
            return

        exit_sig = self._belief_network.get_exit_signal(pos.side, pos.entry_price)

        # Gather inputs for unified exit engine
        _band_ctx = (self._belief_network.get_band_confluence()
                     if hasattr(self._belief_network, 'get_band_confluence') else None)
        _st = self._last_states[-1]['state'] if self._last_states else None
        _f_net = float(getattr(_st, 'net_force', 0.0)) if _st else 0.0
        _noise = float(getattr(_st, 'swing_noise_ticks', 0.0)) if _st else 0.0
        _bar_high = getattr(self, '_last_bar_high', price)
        _bar_low = getattr(self, '_last_bar_low', price)

        # Sync tuning params to exit engine
        self._exit_engine.envelope_half_life_bars = self._tuning.get('envelope_halflife_bars', 40)
        self._exit_engine.envelope_min_bars = self._tuning.get('envelope_min_bars', 5)

        _exit_result = self._exit_engine.evaluate(
            pos=self._pos_state,
            bar_high=_bar_high, bar_low=_bar_low, bar_close=price,
            current_bar_index=self._bar_i,
            band_context=_band_ctx,
            net_force=_f_net,
            exit_signal=exit_sig,
            noise_ticks=_noise,
            belief_network=self._belief_network,
        )

        # Per-trade diagnostic capture
        _tick = self._cfg.tick_size
        _pnl_t = self._exit_engine._calc_pnl_ticks(self._pos_state, price)
        _peak = self._pos_state.peak_favorable
        if self._pos_state.side == 'long':
            _hwm_t = (_peak - self._pos_state.entry_price) / _tick
        else:
            _hwm_t = (self._pos_state.entry_price - _peak) / _tick
        self._trade_logger.log_bar({
            'ts': ts,
            'price': price,
            'bars_held': self._pos_state.bars_held,
            'pnl_ticks': round(_pnl_t, 2),
            'hwm_ticks': round(_hwm_t, 2),
            'net_force': round(_f_net, 4),
            'envelope_tol': round(self._pos_state.envelope_level, 2),
            't_half_eff': round(self._exit_engine.envelope_half_life_bars, 1),
            'stop_level': self._exit_engine._get_stop_price(self._pos_state),
            'trail_ticks': round(self._pos_state.trailing_stop_ticks, 1),
            'conviction': round(exit_sig.get('conviction', 0), 3),
            'direction': exit_sig.get('direction', ''),
            'wave_maturity': round(exit_sig.get('wave_maturity', 0), 3),
            'decay_score': round(exit_sig.get('decay_score', 0), 3),
            'z_score': round(getattr(_st, 'z_score', 0.0), 3) if _st else 0,
            'velocity': round(getattr(_st, 'velocity', 0.0), 4)
                        if _st else 0,
        })

        if _exit_result.action != ExitAction.HOLD:
            reason = _exit_result.action.value
            logger.info(f"EXIT signal: {reason} (decay={exit_sig.get('decay_score', 0):.2f})")
            exited_side = pos.side
            self._belief_network.stop_trade_tracking()

            # Auto-TP re-entry: bank profit, re-enter same side if belief agrees
            if (reason == 'profit_target'
                    and self._tuning.get('auto_tp_reentry', False)
                    and not self._shutting_down):
                await self._auto_tp_reentry(exited_side, price, ts)
            # Ping-pong: send 2-contract flip (close + open opposite) in one order
            elif self._ping_pong_mode and not self._shutting_down:
                await self._flip_position(reason, exited_side, price, ts)
            else:
                await self._close_position(reason)

    async def _flip_position(self, reason: str, exited_side: str,
                             price: float, ts: float):
        """Ping-pong flip/continuation after exit trigger.

        Opposite side: single 2-contract flip order (BUY 2 when SHORT 1).
        Same side: no orders — keep NT8 position, reset exits internally.
        """
        # Determine new direction before clearing state
        _fresh = self._last_states or []
        if not _fresh:
            logger.info("FLIP: no states — falling back to close only")
            await self._close_position(reason)
            return

        state = _fresh[-1]['state']
        flip = self._pp.determine_flip(
            exited_side=exited_side, state=state,
            exec_engine=self._exec_engine,
            anchor_depth=self._anchor_depth, anchor_tf=self._anchor_tf,
            ts=ts, active_tid=self._active_tid,
            side_lock=self._shared_state.get('side_lock'),
            atr_ticks=self._live_atr_ticks,
        )
        side = flip.side
        sl_ticks, tp_ticks = flip.sl_ticks, flip.tp_ticks
        trail_ticks, trail_act = flip.trail_ticks, flip.trail_act
        tid = self._active_tid or 'MANUAL'
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid

        # Reset exit state (same as _close_position)
        self._last_exit_side = self._active_side
        self._last_exit_reason = reason
        self._trade_logger.finish_trade(reason, price)
        pos = self._position
        self._last_high_water = (self._pos_state.peak_favorable if self._pos_state
                                 else (pos.peak_favorable if pos else self._entry_price))
        self._position = None
        self._last_exit_time = time.time()

        logger.info(f"INSTANT FLIP #{self._pp.flip_count}: {exited_side}->{side.upper()} "
                    f"@ {price:.2f}  dir_src={flip.dir_source}  p_long={flip.p_long:.2f}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")

        self._position = self._exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=self._bar_i,
            template_id=f'PP_{base_tid}',
            sl_ticks=sl_ticks, tp_ticks=tp_ticks,
            trail_ticks=trail_ticks, trail_activation_ticks=trail_act,
        )
        self._init_exit_state(side, price, sl_ticks, tp_ticks, f'PP_{base_tid}')
        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = f'PP_{base_tid}'
        self._max_hold_bars = 960
        self._trade_logger.start_trade(
            self._session.stats.trades + 1, side, price, ts)

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars)

        same_side = flip.same_side
        if same_side:
            # Same-side continuation: keep NT8 position open, reset exits.
            # No orders = zero latency, zero commission, zero slippage.
            # Track PnL internally for the "closed" trade segment.
            pos = self._orders.position
            if pos and pos.avg_price > 0:
                if pos.side == 'LONG':
                    seg_pnl = (price - pos.avg_price) * pos.qty * self._cfg.point_value
                else:
                    seg_pnl = (pos.avg_price - price) * pos.qty * self._cfg.point_value
                self._orders._daily_pnl += seg_pnl
                self._orders._trade_count += 1
                self._orders._log_trade(price, seg_pnl, reason='pp_continuation')
                self._orders.last_exit_info = {
                    'entry_px': pos.avg_price, 'exit_px': price, 'side': pos.side,
                }
                # Reset entry price to current for next segment
                pos.avg_price = price
                pos.entry_time = time.time()
                logger.info(f"PP CONTINUATION: {side.upper()} @ {price:.2f}  "
                            f"seg_pnl=${seg_pnl:+.2f}  (no orders, exits reset)")
        else:
            # Opposite side: single 2-contract flip order
            self._flip_in_progress = True
            msg = self._orders.build_flip_order(reason=reason)
            if msg:
                self._order_send_ts = time.perf_counter()
                await self._client.send(msg)
                logger.info("INSTANT FLIP: 2-contract order sent")

        self._gui.push({'type': 'TRADE_MARKER', 'action': 'entry',
                        'side': side, 'price': price})

    def _init_exit_state(self, side: str, price: float, sl_ticks: float,
                         tp_ticks: float, template_id, lib_entry: dict = None):
        """Create ExitEngine PositionState for a new trade."""
        self._pos_state = self._exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=self._bar_i,
            template_id=template_id,
            sl_ticks=sl_ticks, tp_ticks=tp_ticks,
            max_hold_bars=self._max_hold_bars,
            lib_entry=lib_entry,
        )

    async def _close_position(self, reason: str):
        """Send close order and reset position state."""
        self._last_exit_side = self._active_side  # for ping-pong flip
        self._position_open = False
        self._last_exit_reason = reason  # for trade log
        # Finish per-trade diagnostic CSV
        self._trade_logger.finish_trade(reason, self._last_price)
        # Snapshot peak before clearing position (for capture bucket + self-tune)
        pos = self._position
        _ps = self._pos_state
        self._last_high_water = (_ps.peak_favorable if _ps
                                 else (pos.peak_favorable if pos else self._entry_price))
        # Self-tune envelope halflife
        if _ps is not None and self._entry_price > 0:
            _tick = self._asset.tick_size
            _tv = self._asset.tick_value
            if self._active_side == 'long':
                _tmfe = (_ps.peak_favorable - self._entry_price) / _tick
            else:
                _tmfe = (self._entry_price - _ps.peak_favorable) / _tick
            _pnl_ticks = (self._last_price - self._entry_price) / _tick if self._active_side == 'long' \
                else (self._entry_price - self._last_price) / _tick
            _cap = _pnl_ticks / _tmfe if _tmfe > 0 else 0.0
            self._exit_engine.record_trade_outcome(_tmfe, _pnl_ticks, _cap)
        self._position = None
        self._pos_state = None  # reset ExitEngine position
        if hasattr(self, '_exec_engine'):
            self._exec_engine.position_closed()
        self._last_exit_time = time.time()

        msg = self._orders.build_exit_order(reason=reason)
        if msg:
            self._order_send_ts = time.perf_counter()
            await self._client.send(msg)
            logger.info(f"LATENCY: exit order sent  (reason={reason})")

    async def _auto_tp_reentry(self, exited_side: str, price: float, ts: float):
        """Auto take-profit re-entry: close, bank profit, re-enter same side if belief agrees."""
        belief = self._belief_network.get_belief()
        _side_lock = self._shared_state.get('side_lock')

        # Re-entry gate: belief must agree with exited side + conviction threshold
        reenter = (belief is not None
                   and belief.direction == exited_side
                   and belief.is_confident
                   and (not _side_lock or _side_lock == exited_side))

        if reenter:
            logger.info(f"AUTO-TP RE-ENTRY: {exited_side.upper()} @ {price:.2f} "
                        f"(conv={belief.conviction:.2f})")
            # Close current position first (banks the profit)
            await self._close_position('profit_target')
            # Re-enter same side with fresh stops
            atr = self._live_atr_ticks if self._live_atr_ticks > 0 else 8.0
            _floor = max(4, self._tuning.get('min_tick_floor', 4))
            sl_ticks = max(_floor, int(round(atr * self._tuning.get('exit_sl_mult', 3.0))))
            tp_ticks = max(_floor, int(round(atr * self._tuning.get('exit_tp_mult', 5.0))))
            trail_ticks = max(_floor, int(round(atr * self._tuning.get('exit_trail_mult', 2.5))))
            trail_act = max(_floor, int(round(atr * self._tuning.get('exit_trail_act_mult', 0.6))))

            _fresh = self._last_states or []
            state = _fresh[-1]['state'] if _fresh else None
            tid = self._active_tid or 'REENTRY'

            self._position = self._exit_engine.open_position(
                side=exited_side, entry_price=price, entry_bar_index=self._bar_i,
                template_id=f'RE_{tid}',
                sl_ticks=sl_ticks, tp_ticks=tp_ticks,
                trail_ticks=trail_ticks, trail_activation_ticks=trail_act,
            )
            self._init_exit_state(exited_side, price, sl_ticks, tp_ticks, f'RE_{tid}')
            self._position_open = True
            self._entry_price = price
            self._entry_time = ts
            self._entry_bar = self._bar_i
            self._active_side = exited_side
            self._active_tid = f'RE_{tid}'
            self._max_hold_bars = 960
            self._trade_logger.start_trade(
                self._session.stats.trades + 1, exited_side, price, ts)

            self._belief_network.start_trade_tracking(
                side=exited_side, entry_bar=self._bar_i,
                pattern_horizon_bars=self._max_hold_bars)

            order_msg = self._orders.build_entry_order(
                'BUY' if exited_side == 'long' else 'SELL')
            if order_msg:
                self._order_send_ts = time.perf_counter()
                await self._client.send(order_msg)
            self._gui.push({'type': 'TRADE_MARKER', 'action': 'entry',
                            'side': exited_side, 'price': price})
        else:
            _reason = 'no belief' if belief is None else f'dir={belief.direction} conv={belief.conviction:.2f}'
            logger.info(f"AUTO-TP: no re-entry ({_reason}) — closing flat")
            await self._close_position('profit_target')

    async def _handle_manual_order(self, action: str, price: float,
                                   ts: float, states: list):
        """Process a manual BUY/SELL/FLATTEN from the popup buttons."""
        if self._instrument_mismatch:
            logger.error("BLOCKED: instrument mismatch — fix NT8 chart first")
            return
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

        # If already in a position, flatten first then defer entry until FILL
        if self._position_open:
            logger.info(f"MANUAL {action}: flattening existing position first")
            self._belief_network.stop_trade_tracking()
            self._pending_manual_entry = {
                'action': action, 'price': price, 'ts': ts, 'states': states,
            }
            await self._close_position('MANUAL_REVERSE')
            return  # entry will fire from _execute_pending_manual on FILL

        await self._execute_manual_entry(action, price, ts, states)

    async def _execute_manual_entry(self, action: str, price: float,
                                    ts: float, states: list):
        """Execute the manual entry (called directly or after deferred FILL)."""
        side = 'long' if action == 'BUY' else 'short'

        # Belief warning — check if workers disagree with manual direction
        belief = self._belief_network.get_belief()
        if belief and belief.direction != side:
            _warn = (f"WARNING: belief says {belief.direction.upper()} "
                     f"(conv={belief.conviction:.2f}) — you're going {side.upper()}")
            logger.warning(_warn)
            self._gui.push({
                'type': 'PHASE_PROGRESS', 'phase': 'LIVE',
                'step': f'WARN: belief={belief.direction.upper()}',
                'pct': self._entry_belief_pct,
            })

        # Exit params: ATR-based defaults, manual overrides if non-zero
        atr = self._live_atr_ticks if self._live_atr_ticks > 0 else 8.0
        _floor = max(4, self._tuning.get('min_tick_floor', 4))
        sl_ticks = self._tuning.get('manual_sl', 0) or max(_floor, int(round(atr * self._tuning.get('exit_sl_mult', 3.0))))
        tp_ticks = self._tuning.get('manual_tp', 0) or max(_floor, int(round(atr * self._tuning.get('exit_tp_mult', 5.0))))
        trail_ticks = self._tuning.get('manual_trail', 0) or max(_floor, int(round(atr * self._tuning.get('exit_trail_mult', 2.5))))
        trail_act = max(_floor, int(round(atr * self._tuning.get('exit_trail_act_mult', 0.6))))

        logger.info(f"MANUAL ENTRY: {side.upper()} @ {price:.2f}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")
        self._gui.push({'type': 'TRADE_MARKER', 'action': 'entry',
                        'side': side, 'price': price})

        # Always use freshest market state — a trade is a trade
        _fresh_states = self._last_states or states
        if not _fresh_states:
            # Force recompute if nothing cached (rare: manual during warmup)
            _fresh_states = self._aggregator._recompute() or []
            if _fresh_states:
                self._last_states = _fresh_states
                logger.info(f"Forced state recompute for manual entry: {len(_fresh_states)} states")
        state = _fresh_states[-1]['state'] if _fresh_states else None
        if state is None:
            logger.warning("No market state available — manual trade will have limited exit protection")

        self._position = self._exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=self._bar_i,
            template_id='MANUAL',
            sl_ticks=sl_ticks, tp_ticks=tp_ticks,
            trail_ticks=trail_ticks, trail_activation_ticks=trail_act,
        )
        self._init_exit_state(side, price, sl_ticks, tp_ticks, 'MANUAL')
        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = 'MANUAL'
        self._entry_depth = self._anchor_depth
        self._max_hold_bars = 960  # 4 hours max for manual trades
        self._trade_logger.start_trade(
            self._session.stats.trades + 1, side, price, ts)

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars)

        order_msg = self._orders.build_entry_order(
            'BUY' if side == 'long' else 'SELL')
        if order_msg:
            self._order_send_ts = time.perf_counter()
            await self._client.send(order_msg)
            logger.info("LATENCY: manual entry order sent")

    # ── Ping-Pong Mode ──────────────────────────────────────────────

    async def _enter_ping_pong(self, side_hint: str, price: float,
                                ts: float, states: list):
        """Full-context flip entry — uses direction model + learns outcomes."""
        # Always use freshest market state
        _fresh = self._last_states or states
        if not _fresh:
            logger.debug("PING-PONG: no states for flip entry, skip")
            self._pp.pending_flip = None
            return

        state = _fresh[-1]['state']
        flip = self._pp.determine_flip(
            exited_side=side_hint, state=state,
            exec_engine=self._exec_engine,
            anchor_depth=self._anchor_depth, anchor_tf=self._anchor_tf,
            ts=ts, active_tid=self._active_tid,
            side_lock=self._shared_state.get('side_lock'),
            atr_ticks=self._live_atr_ticks,
        )
        side = flip.side
        sl_ticks, tp_ticks = flip.sl_ticks, flip.tp_ticks
        trail_ticks, trail_act = flip.trail_ticks, flip.trail_act
        tid = self._active_tid or 'MANUAL'
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid

        logger.info(f"PING-PONG FLIP #{self._pp.flip_count}: {side.upper()} "
                    f"@ {price:.2f}  dir_src={flip.dir_source}  p_long={flip.p_long:.2f}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}")
        self._gui.push({'type': 'TRADE_MARKER', 'action': 'entry',
                        'side': side, 'price': price})

        self._position = self._exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=self._bar_i,
            template_id=f'PP_{base_tid}',
            sl_ticks=sl_ticks, tp_ticks=tp_ticks,
            trail_ticks=trail_ticks, trail_activation_ticks=trail_act,
        )
        self._init_exit_state(side, price, sl_ticks, tp_ticks, f'PP_{base_tid}')
        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = f'PP_{base_tid}'
        self._entry_depth = self._entry_depth
        self._max_hold_bars = 960  # no forced exit — exhaustion only
        self._trade_logger.start_trade(
            self._session.stats.trades + 1, side, price, ts)

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars)

        order_msg = self._orders.build_entry_order(
            'BUY' if side == 'long' else 'SELL')
        if order_msg:
            self._order_send_ts = time.perf_counter()
            await self._client.send(order_msg)
            logger.info("LATENCY: ping-pong flip order sent")

    def _log_direction_bias(self, tid, side: str, pnl: float):
        """Log direction bias after record_trade() has already updated the brain."""
        bias = self._brain.get_dir_bias(tid)
        if bias is None:
            return
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid
        lw, ll = bias['long_w'], bias['long_l']
        sw, sl = bias['short_w'], bias['short_l']
        lt, st = lw + ll, sw + sl
        l_wr = f"{lw/lt:.0%}" if lt > 0 else "n/a"
        s_wr = f"{sw/st:.0%}" if st > 0 else "n/a"
        alt_key = 'short' if side.lower() == 'long' else 'long'
        _verdict = "CONFIRMED" if pnl > 0 else f"WRONG (alt {alt_key.upper()} would be ${-pnl:+.0f})"
        logger.info(f"DIR LEARN: tid={base_tid}  {side.upper()} ${pnl:+.0f} -> {_verdict}  |  "
                    f"LONG {lw}W/{ll}L ({l_wr})  SHORT {sw}W/{sl}L ({s_wr})")

    def _prepare_shutdown(self):
        """Save brain + bars + write session report + advise GUI."""
        # Save brain
        if self._live_trade_count > 0:
            brain_path = os.path.join(
                self._cfg.checkpoint_dir, 'live_brain.pkl')
            self._brain.save(brain_path)
            logger.info(f"Brain saved ({self._live_trade_count} trades)")

        # Persist bars to parquet for delta sync on next start
        self._aggregator.save_to_parquet()

        # Write session report
        self._session.write_report(
            gate_stats=(self._exec_engine.gate_stats
                        if hasattr(self, '_exec_engine') else {}),
            brain_dir_bias=self._brain.dir_bias if self._brain else {},
            account_snapshot={
                'cash': self._nt8_cash_value,
                'unrealized': self._nt8_unrealized_pnl,
                'net_liq': self._nt8_net_liquidation,
            },
            pp_flip_count=self._pp.flip_count,
            anchor_tf=self._anchor_tf,
            anchor_depth=self._anchor_depth,
            bar_count=self._bar_i,
        )

        # Replay parity report (replay mode only)
        if self._shared_state.get('replay_mode'):
            self._write_replay_parity_report()

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

        self._gui.push({'type': 'SHUTDOWN_READY', 'status': status})

    def _write_replay_parity_report(self):
        """Write detailed parity report comparing replay results vs OOS inline."""
        import re
        from collections import Counter, defaultdict
        from datetime import datetime, timezone

        trades = self._session.trade_log
        s = self._session.stats

        if not trades:
            logger.warning("Replay parity: no trades to report")
            return

        # ── Parse OOS daily ledger from reports/oos_report.txt ────────────
        oos_daily = {}  # date_str -> {trades, wins, pnl}
        oos_report_path = os.path.join('reports', 'oos_report.txt')
        try:
            with open(oos_report_path, 'r') as f:
                in_ledger = False
                for line in f:
                    if 'DAILY SESSION LEDGER' in line:
                        in_ledger = True
                        continue
                    if in_ledger and re.match(r'\s+20\d\d-\d\d-\d\d', line):
                        parts = line.split()
                        if len(parts) >= 7:
                            date_str = parts[0]
                            n_trades = int(parts[1])
                            n_wins = int(parts[2])
                            # PnL has $ and +/- signs, find it
                            pnl_str = parts[5].replace(',', '').replace('+', '')
                            try:
                                pnl_val = float(pnl_str)
                            except ValueError:
                                pnl_val = 0.0
                            oos_daily[date_str] = {
                                'trades': n_trades, 'wins': n_wins, 'pnl': pnl_val}
                    elif in_ledger and line.strip().startswith('──') and oos_daily:
                        break  # end of ledger
        except FileNotFoundError:
            logger.warning(f"OOS report not found: {oos_report_path}")

        # ── Build replay daily ledger from trade log ──────────────────────
        # Derive date from entry price timestamp approximation
        # Use bar index and anchor period to estimate timestamps
        replay_daily = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0,
            'gross_win': 0.0, 'gross_loss': 0.0, 'exits': Counter()})

        for t in trades:
            _date = t.get('date', 'unknown')
            d = replay_daily[_date]
            d['trades'] += 1
            d['pnl'] += t['pnl']
            if t['pnl'] > 0:
                d['wins'] += 1
                d['gross_win'] += t['pnl']
            elif t['pnl'] < 0:
                d['losses'] += 1
                d['gross_loss'] += t['pnl']
            d['exits'][t.get('reason', '?')] += 1

        # ── Identify last 5 OOS days for comparison ───────────────────────
        oos_dates_sorted = sorted(oos_daily.keys())
        last_5_oos = oos_dates_sorted[-5:] if len(oos_dates_sorted) >= 5 else oos_dates_sorted
        oos_5d_trades = sum(oos_daily[d]['trades'] for d in last_5_oos)
        oos_5d_wins = sum(oos_daily[d]['wins'] for d in last_5_oos)
        oos_5d_pnl = sum(oos_daily[d]['pnl'] for d in last_5_oos)
        oos_5d_wr = oos_5d_wins / oos_5d_trades * 100 if oos_5d_trades else 0
        oos_5d_avg = oos_5d_pnl / oos_5d_trades if oos_5d_trades else 0

        # Replay totals
        rp_wr = s.wins / s.trades * 100 if s.trades else 0
        rp_avg = s.pnl / s.trades if s.trades else 0
        pf = s.gross_win / abs(s.gross_loss) if s.gross_loss else float('inf')
        n_days = max(1, len(replay_daily))

        # ── Exit reason breakdown ─────────────────────────────────────────
        exit_stats = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0,
                                          'gross_win': 0.0, 'gross_loss': 0.0})
        for t in trades:
            r = t.get('reason', '?')
            exit_stats[r]['n'] += 1
            exit_stats[r]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                exit_stats[r]['wins'] += 1
                exit_stats[r]['gross_win'] += t['pnl']
            elif t['pnl'] < 0:
                exit_stats[r]['gross_loss'] += t['pnl']

        # ── Direction breakdown ───────────────────────────────────────────
        dir_stats = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
        for t in trades:
            side = t.get('side', '?')
            dir_stats[side]['n'] += 1
            dir_stats[side]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                dir_stats[side]['wins'] += 1

        # ── Build report ──────────────────────────────────────────────────
        L = []
        L.append("=" * 80)
        L.append(f"  REPLAY PARITY REPORT  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        L.append(f"  Engine: LiveEngine + ReplayBridge (same code path as live)")
        L.append(f"  Bars processed: {self._bar_i:,}  |  Anchor: {self._anchor_tf}")
        L.append("=" * 80)

        # ── Key Metrics (OOS vs Replay) ───────────────────────────────────
        L.append("")
        L.append(f"  {'Metric':<30} {'OOS (last 5d)':>14} {'Replay':>14} {'Delta':>14}")
        L.append(f"  {'-'*72}")
        L.append(f"  {'Trades':<30} {oos_5d_trades:>14} {s.trades:>14} {s.trades - oos_5d_trades:>+14}")
        L.append(f"  {'Win Rate':<30} {oos_5d_wr:>13.1f}% {rp_wr:>13.1f}% {rp_wr - oos_5d_wr:>+13.1f}%")
        L.append(f"  {'Total PnL':<30} ${oos_5d_pnl:>12,.2f} ${s.pnl:>12,.2f} ${s.pnl - oos_5d_pnl:>+12,.2f}")
        L.append(f"  {'Avg Trade':<30} ${oos_5d_avg:>12,.2f} ${rp_avg:>12,.2f} ${rp_avg - oos_5d_avg:>+12,.2f}")
        L.append(f"  {'Gross Profit':<30} {'':>14} ${s.gross_win:>12,.2f}")
        L.append(f"  {'Gross Loss':<30} {'':>14} ${abs(s.gross_loss):>12,.2f}")
        L.append(f"  {'Profit Factor':<30} {'':>14} {pf:>14.2f}")
        L.append(f"  {'Max Drawdown':<30} {'':>14} ${s.max_session_drawdown:>12,.2f}")
        L.append(f"  {'Max Consec Losses':<30} {'':>14} {s.max_consec_losses:>14}")
        L.append(f"  {'Trades/Day':<30} {oos_5d_trades/max(1,len(last_5_oos)):>14.1f} "
                 f"{s.trades/n_days:>14.1f}")

        # ── Exit Reason Breakdown ─────────────────────────────────────────
        L.append("")
        L.append("  EXIT REASON BREAKDOWN")
        L.append(f"  {'Reason':<20} {'Trades':>7} {'Win%':>7} {'PnL':>12} "
                 f"{'Avg':>9} {'GrossWin':>10} {'GrossLoss':>10}")
        L.append(f"  {'-'*75}")
        for r in sorted(exit_stats.keys(), key=lambda k: -exit_stats[k]['pnl']):
            es = exit_stats[r]
            wr = es['wins'] / es['n'] * 100 if es['n'] else 0
            avg = es['pnl'] / es['n'] if es['n'] else 0
            L.append(f"  {r:<20} {es['n']:>7} {wr:>6.1f}% ${es['pnl']:>10,.2f} "
                     f"${avg:>8,.2f} ${es['gross_win']:>9,.2f} "
                     f"${abs(es['gross_loss']):>9,.2f}")

        # ── Direction Breakdown ───────────────────────────────────────────
        L.append("")
        L.append("  DIRECTION BREAKDOWN")
        L.append(f"  {'Side':<10} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Avg':>9}")
        L.append(f"  {'-'*50}")
        for side in sorted(dir_stats.keys()):
            ds = dir_stats[side]
            wr = ds['wins'] / ds['n'] * 100 if ds['n'] else 0
            avg = ds['pnl'] / ds['n'] if ds['n'] else 0
            L.append(f"  {side:<10} {ds['n']:>8} {wr:>7.1f}% ${ds['pnl']:>10,.2f} ${avg:>8,.2f}")

        # ── Gate Rejection Funnel ─────────────────────────────────────────
        gs = self._exec_engine.gate_stats if hasattr(self, '_exec_engine') else {}
        if gs:
            L.append("")
            L.append("  GATE REJECTION FUNNEL")
            _total = gs.get('candidates', 1) or 1
            _pct = lambda n: f"{n/_total*100:.1f}%"
            L.append(f"    Total candidates:     {_total:>8,}")
            L.append(f"    Pattern Quality:      {gs.get('gate0_skip', 0):>8,}  ({_pct(gs.get('gate0_skip', 0))})")
            L.append(f"    Template Match:       {gs.get('gate1_nomatch', gs.get('gate1_skip', 0)):>8,}  "
                     f"({_pct(gs.get('gate1_nomatch', gs.get('gate1_skip', 0)))})")
            L.append(f"    Brain Reject:         {gs.get('gate2_brain', gs.get('gate2_skip', 0)):>8,}  "
                     f"({_pct(gs.get('gate2_brain', gs.get('gate2_skip', 0)))})")
            L.append(f"    Low Conviction:       {gs.get('gate3_conviction', gs.get('gate3_skip', 0)):>8,}  "
                     f"({_pct(gs.get('gate3_conviction', gs.get('gate3_skip', 0)))})")
            L.append(f"    Momentum Misalign:    {gs.get('gate4_direction', gs.get('gate4_skip', 0)):>8,}  "
                     f"({_pct(gs.get('gate4_direction', gs.get('gate4_skip', 0)))})")
            L.append(f"    -> Traded:            {gs.get('traded', 0):>8,}  ({_pct(gs.get('traded', 0))})")

        # ── Per-Day Comparison ────────────────────────────────────────────
        L.append("")
        L.append("  PER-DAY COMPARISON")
        L.append(f"  {'Date':<12} {'OOS_T':>6} {'OOS_PnL':>11} {'RP_T':>6} {'RP_PnL':>11} "
                 f"{'RP_GW':>10} {'RP_GL':>10} {'Delta':>11}")
        L.append(f"  {'-'*78}")
        all_dates = sorted(set(list(last_5_oos) + list(replay_daily.keys())))
        for d in all_dates:
            if d == 'unknown':
                continue
            oos_d = oos_daily.get(d, {'trades': 0, 'pnl': 0})
            rp_d = replay_daily.get(d, {'trades': 0, 'pnl': 0, 'gross_win': 0, 'gross_loss': 0})
            delta = rp_d['pnl'] - oos_d['pnl']
            L.append(f"  {d:<12} {oos_d['trades']:>6} ${oos_d['pnl']:>9,.2f} "
                     f"{rp_d['trades']:>6} ${rp_d['pnl']:>9,.2f} "
                     f"${rp_d.get('gross_win', 0):>9,.2f} "
                     f"${abs(rp_d.get('gross_loss', 0)):>9,.2f} "
                     f"${delta:>+9,.2f}")
        # Unknown date bucket
        if 'unknown' in replay_daily:
            ud = replay_daily['unknown']
            L.append(f"  {'(undated)':<12} {'':>6} {'':>11} "
                     f"{ud['trades']:>6} ${ud['pnl']:>9,.2f} "
                     f"${ud.get('gross_win', 0):>9,.2f} "
                     f"${abs(ud.get('gross_loss', 0)):>9,.2f}")

        # ── Trade-by-Trade Log ────────────────────────────────────────────
        L.append("")
        L.append("  TRADE LOG (all replay trades)")
        L.append(f"  {'#':>4} {'Side':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} "
                 f"{'Reason':<18} {'Bars':>5} {'MFE':>6} {'Capture':>8}")
        L.append(f"  {'-'*83}")
        cum = 0.0
        for i, t in enumerate(trades, 1):
            cum += t['pnl']
            mfe = t.get('mfe_ticks', 0)
            cap = t.get('capture', 0)
            L.append(f"  {i:>4} {t.get('side', '?'):<6} "
                     f"{t.get('entry', 0):>10,.2f} {t.get('exit', 0):>10,.2f} "
                     f"${t['pnl']:>+9,.2f} {t.get('reason', '?'):<18} "
                     f"{t.get('bars', 0):>5} {mfe:>6.1f} {cap:>7.0f}%")
        L.append(f"  {'-'*83}")
        L.append(f"  {'':>4} {'':6} {'':10} {'TOTAL':>10} ${s.pnl:>+9,.2f}")

        # ── Parity Verdict ────────────────────────────────────────────────
        L.append("")
        _score = 1.0
        warnings = []
        if oos_5d_trades > 0:
            trade_ratio = s.trades / oos_5d_trades
            if abs(trade_ratio - 1.0) > 0.3:
                _score -= 0.2
                warnings.append(f"Trade count: {s.trades} vs OOS {oos_5d_trades} "
                                f"(ratio {trade_ratio:.2f})")
        if abs(rp_wr - oos_5d_wr) > 5:
            _score -= 0.2
            warnings.append(f"Win rate: {rp_wr:.1f}% vs OOS {oos_5d_wr:.1f}%")
        if oos_5d_pnl > 0:
            pnl_ratio = s.pnl / oos_5d_pnl
            if pnl_ratio < 0.5:
                _score -= 0.3
                warnings.append(f"PnL capture: ${s.pnl:,.2f} / ${oos_5d_pnl:,.2f} "
                                f"= {pnl_ratio:.1%}")
            elif pnl_ratio < 0.8:
                _score -= 0.15
                warnings.append(f"PnL gap: ${s.pnl:,.2f} vs OOS ${oos_5d_pnl:,.2f} "
                                f"({pnl_ratio:.1%})")
        if s.gross_loss != 0 and pf < 1.5:
            _score -= 0.1
            warnings.append(f"Low profit factor: {pf:.2f}")
        _score = max(0, min(1, _score))
        _status = "PASSED" if _score >= 0.7 else "FAILED"

        L.append("  PARITY VERDICT")
        L.append(f"    Score:  {_score:.2f}")
        L.append(f"    Status: {_status}")
        if not warnings:
            L.append("    No warnings — replay matches OOS")
        for w in warnings:
            L.append(f"    WARNING: {w}")

        L.append("")
        L.append("=" * 80)

        report_text = '\n'.join(L)

        # Save to file
        os.makedirs(os.path.join('reports', 'live'), exist_ok=True)
        ts_str = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join('reports', 'live', f'replay_parity_{ts_str}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text + '\n')

        # Print to console
        print("\n" + report_text)
        logger.info(f"Replay parity report: {path}")

    def _on_account_update(self, msg: dict):
        """Handle ACCOUNT_UPDATE from NT8 — push equity to GUI."""
        self._nt8_cash_value = float(msg.get('cash_value', 0))
        self._nt8_realized_pnl = float(msg.get('realized_pnl', 0))
        self._nt8_unrealized_pnl = float(msg.get('unrealized_pnl', 0))
        self._nt8_net_liquidation = float(msg.get('net_liquidation', 0))

        self._gui.push_account(self._nt8_cash_value, self._nt8_realized_pnl,
                               self._nt8_unrealized_pnl, self._nt8_net_liquidation)

    def _brain_learn(self, pnl: float):
        """Feed trade outcome to brain, update GUI stats, save periodically."""
        from core.bayesian_brain import record_trade

        result = 'WIN' if pnl > 0 else 'LOSS'

        # Use actual fill prices from OrderManager when available
        exi = self._orders.last_exit_info
        entry_px = exi.get('entry_px', self._entry_price)
        exit_px = exi.get('exit_px', 0.0)
        if not exit_px:
            # Fallback: compute from PnL (side-aware)
            tick_val = self._cfg.point_value
            if self._active_side == 'short':
                exit_px = entry_px - pnl / tick_val
            else:
                exit_px = entry_px + pnl / tick_val

        outcome = record_trade(
            self._brain, tid=self._active_tid,
            entry_price=entry_px, exit_price=exit_px,
            pnl=pnl, side=self._active_side,
            exit_reason=self._last_exit_reason,
            timestamp=time.time(),
            tick_value=self._cfg.tick_size * self._cfg.point_value,
        )
        self._log_direction_bias(self._active_tid, self._active_side, pnl)
        self._live_trade_count += 1

        # Compute MFE for capture bucket
        hwm = getattr(self, '_last_high_water', entry_px)
        if self._active_side == 'long':
            mfe_ticks = (hwm - entry_px) / 0.25 if entry_px else 0
        else:
            mfe_ticks = (entry_px - hwm) / 0.25 if entry_px else 0
        pnl_ticks = pnl / (self._cfg.point_value * self._cfg.tick_size)
        capture = (pnl_ticks / mfe_ticks * 100) if mfe_ticks > 0 else 0.0
        _pe = getattr(self, '_price_expected', entry_px)
        _pe_err = round((exit_px - _pe) / self._asset.tick_size, 2) if _pe != entry_px else 0.0

        # Derive trade date from aggregator bar timestamp
        _trade_date = time.strftime('%Y-%m-%d')
        if self._entry_bar < len(self._aggregator._rows):
            _bar_ts = self._aggregator._rows[self._entry_bar].get('timestamp', 0)
            if _bar_ts:
                from datetime import datetime as _dt, timezone as _tz
                _trade_date = _dt.fromtimestamp(_bar_ts, tz=_tz.utc).strftime('%Y-%m-%d')

        # Delegate all stat tracking + trade log to SessionTracker
        self._session.record_trade(pnl, {
            'time': time.strftime('%H:%M:%S'),
            'date': _trade_date,
            'side': 'LONG' if self._active_side == 'long' else 'SHORT',
            'entry': entry_px, 'exit': exit_px,
            'pnl': pnl, 'result': result,
            'reason': self._last_exit_reason,
            'bars': self._bar_i - self._entry_bar,
            'entry_bar': self._entry_bar,
            'tid': self._active_tid, 'depth': self._entry_depth,
            'mfe_ticks': mfe_ticks, 'pnl_ticks': pnl_ticks,
            'predicted_mfe': getattr(self, '_predicted_mfe_ticks', 0.0),
            'price_expected': _pe, 'price_expected_error': _pe_err,
        })

        logger.info(f"Brain learned: tid={self._active_tid} {result} "
                    f"${pnl:+.2f}  capture={capture:.0f}%  "
                    f"(table size: {len(self._brain.table)})")

        # Push stats to GUI popup
        self._gui.push_stats(
                    session_pnl=self._session.stats.pnl,
                    session_wins=self._session.stats.wins,
                    session_trades=self._session.stats.trades,
                    gross_win=self._session.stats.gross_win,
                    gross_loss=self._session.stats.gross_loss,
                    exit_buckets=self._session.stats.exit_buckets,
                    belief_pct=self._exit_belief_pct if self._position_open else self._entry_belief_pct,
                    in_position=self._position_open,
                    daily_pnl=self._orders.daily_pnl,
                )
        self._gui.push({
            'type': 'DAY_PNL',
            'day': time.strftime('%H:%M'),
            'pnl': pnl,
            'trades': 1,
            'wins': 1 if pnl > 0 else 0,
        })
        if self._orders.loss_limit_hit:
            self._gui.push({'type': 'LOSS_LIMIT', 'locked': True,
                            'daily_pnl': self._orders.daily_pnl})

        # Save every N trades
        if self._live_trade_count % self._brain_save_interval == 0:
            brain_path = os.path.join(
                self._cfg.checkpoint_dir, 'live_brain.pkl')
            self._brain.save(brain_path)
            logger.info(f"Live brain saved ({self._live_trade_count} trades)")

    # ── Entry Logic (gate cascade) ───────────────────────────────────

    async def _check_entry(self, price: float, ts: float, states: list):
        """Thin wrapper: delegates gate cascade to ExecutionEngine."""
        _t0 = time.perf_counter()
        if not states:
            self._entry_belief_pct = 0
            return
        if self._instrument_mismatch:
            return
        # Block entries during CME maintenance window
        if self._exit_engine.is_maintenance_window(ts):
            return

        # Aggression scaling → set on exec_engine before each call
        agg = self._shared_state.get('aggression', 0.5)
        _yolo = agg >= 0.99
        _g1_base = self._tuning.get('gate1_dist', _GATE1_DIST_THRESHOLD)
        self._exec_engine.gate1_dist = (
            float('inf') if _yolo else _g1_base + agg * 10.0)
        self._exec_engine.set_live_atr(self._live_atr_ticks)

        # Build Candidate objects via shared BarProcessor (same as OOS/replay)
        latest = states[-1]
        state = latest['state']
        ee_candidates = self._processor._build_candidates(state, ts, yolo=_yolo)

        if not ee_candidates:
            self._entry_belief_pct = 0
            return

        # Side lock → ping-pong override
        _side_lock = self._shared_state.get('side_lock')
        _pp_dir = _side_lock if _side_lock else None

        # Ask ExecutionEngine for decision
        action = self._exec_engine.on_bar(
            price=price, bar_high=price, bar_low=price,
            bar_index=self._bar_i,
            candidates=ee_candidates,
            pp_dir_override=_pp_dir,
        )

        if action.type != ActionType.ENTER:
            # Map gate_label to belief bar percentage
            self._entry_belief_pct = self._gate_label_to_pct(
                action.gate_label if hasattr(action, 'gate_label') else '')
            return

        # ── All gates passed — execute entry ──────────────────────────
        await self._execute_entry(action, price, ts, _t0)

    async def _execute_entry(self, action: TradeAction, price: float,
                             ts: float, t0: float):
        """Execute an ENTER action from ExecutionEngine."""
        side = action.side
        best_tid = action.template_id
        lib_entry = self._pattern_library.get(best_tid, {})
        _dir_src = action.dir_source

        # Tuning offsets
        _floor = max(4, self._tuning.get('min_tick_floor', 4))
        sl_ticks = max(_floor, action.sl_ticks + self._tuning.get('sl_offset', 0))
        tp_ticks = max(_floor, action.tp_ticks + self._tuning.get('tp_offset', 0))
        trail_ticks = max(_floor, action.trail_ticks + self._tuning.get('trail_offset', 0))
        trail_act = max(_floor, action.trail_activation_ticks
                        + self._tuning.get('trail_act_offset', 0))

        self._entry_belief_pct = 100
        logger.info(f"ENTRY: {side.upper()} @ {price:.2f}  "
                    f"tid={best_tid}  dist={action.dist:.2f}  "
                    f"dir_src={_dir_src}  "
                    f"SL={sl_ticks} TP={tp_ticks} trail={trail_ticks}  "
                    f"ATR={self._live_atr_ticks:.1f}")
        _band = self._belief_network.get_band_confluence()
        if _band:
            logger.info(f"  BANDS: {_band.get('band_summary', '')}")
        self._gui.push({'type': 'TRADE_MARKER', 'action': 'entry',
                        'side': side, 'price': price})

        self._position = self._exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=self._bar_i,
            template_id=best_tid,
            sl_ticks=sl_ticks, tp_ticks=tp_ticks,
            trail_ticks=trail_ticks, trail_activation_ticks=trail_act,
            lib_entry=lib_entry,
        )
        self._init_exit_state(side, price, sl_ticks, tp_ticks, best_tid, lib_entry)

        # Sync ExecutionEngine position state
        self._exec_engine.position_opened(
            side=side, price=price, bar_index=self._bar_i,
            template_id=best_tid, lib_entry=lib_entry,
            sl_ticks=sl_ticks, tp_ticks=tp_ticks,
            max_hold_bars=action.max_hold_bars or 960,
        )

        self._position_open = True
        self._entry_price = price
        self._entry_time = ts
        self._entry_bar = self._bar_i
        self._active_side = side
        self._active_tid = best_tid
        self._entry_depth = action.depth

        belief = self._belief_network.get_belief()
        self._predicted_mfe_ticks = round(belief.predicted_mfe, 2) if belief and belief.predicted_mfe > 0 else 0.0
        self._price_expected = round(
            price + ((belief.predicted_mfe if side == 'long' else -belief.predicted_mfe)
                     * self._asset.tick_size), 6) if belief and belief.predicted_mfe > 0 else price
        self._trade_logger.start_trade(
            self._session.stats.trades + 1, side, price, ts)

        _hold_sec = self._tuning.get('max_hold_seconds', 300) or 14400
        self._max_hold_bars = max(20, _hold_sec // self._anchor_period)

        self._pp_last_exit_params = {
            'sl': sl_ticks, 'tp': tp_ticks,
            'trail': trail_ticks, 'trail_act': trail_act,
            'max_hold': self._max_hold_bars,
        }

        _avg_mfe_bar = lib_entry.get('avg_mfe_bar', 0.0)
        _p75_mfe_bar = lib_entry.get('p75_mfe_bar', 0.0)
        _p75_mfe_ticks = lib_entry.get('p75_mfe_ticks', 0.0)

        self._belief_network.start_trade_tracking(
            side=side, entry_bar=self._bar_i,
            pattern_horizon_bars=self._max_hold_bars,
            target_mfe_ticks=_p75_mfe_ticks,
            resolve_bars=_avg_mfe_bar,
            entry_price=price)

        # Per-template exit timescale
        if _avg_mfe_bar > 0:
            self._belief_network.set_active_trade_timescale(
                _avg_mfe_bar, _p75_mfe_bar)

        order_msg = self._orders.build_entry_order(
            'BUY' if side == 'long' else 'SELL')
        if order_msg:
            self._order_send_ts = time.perf_counter()
            await self._client.send(order_msg)
            _decision_ms = (self._order_send_ts - t0) * 1000
            logger.info(f"LATENCY: decision={_decision_ms:.1f}ms  (bar->order sent)")

    @staticmethod
    def _gate_label_to_pct(gate_label: str) -> int:
        """Map ExecutionEngine rejection labels to GUI belief bar %."""
        _MAP = {
            'gate0_skip': 10, 'gate0_noise': 10, 'gate0_hurst': 15,
            'gate0_momentum': 15, 'gate0_tunnel': 15,
            'gate0_5_skip': 20, 'gate1_nomatch': 35,
            'gate2_brain': 55, 'gate3_conviction': 70,
            'gate4_direction': 80, 'gate3_5_screening': 85,
        }
        return min(99, _MAP.get(gate_label, 0))

    # ── Helpers ───────────────────────────────────────────────────────

    def _update_live_atr(self):
        """Compute ATR in ticks from the live bar buffer (rolling 14-period)."""
        df = self._aggregator.df
        if len(df) < 2:
            return
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        # True Range: max(H-L, |H-prevC|, |L-prevC|)
        hl = highs[1:] - lows[1:]
        hpc = np.abs(highs[1:] - closes[:-1])
        lpc = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(hl, np.maximum(hpc, lpc))
        # EMA-14 ATR (use last 14 bars, or all if fewer)
        n = min(14, len(tr))
        atr = float(np.mean(tr[-n:]))
        self._live_atr_ticks = max(1.0, atr / self._cfg.tick_size)
        # ATR logged silently — only on significant change (>20% shift)
        if abs(self._live_atr_ticks - getattr(self, '_last_logged_atr', 0)) > self._live_atr_ticks * 0.2:
            logger.debug(f"Live ATR: {self._live_atr_ticks:.1f} ticks "
                         f"({atr:.2f} points, {len(df)} bars)")
            self._last_logged_atr = self._live_atr_ticks

    def _sync_position_state(self):
        """Sync local position tracking with OrderManager's state."""
        # During a 2-contract flip, on_fill creates new position atomically;
        # guard stays until order_manager confirms non-flat (fill received)
        if self._flip_in_progress:
            if not self._orders.is_flat:
                # Entry fill received — flip complete
                self._flip_in_progress = False
            return
        if self._position_open and self._orders.is_flat:
            # Position closed (fill received)
            self._position_open = False
            self._position = None
        elif not self._position_open and not self._orders.is_flat:
            # Unexpected position (NT8 source of truth)
            logger.warning("NT8 has position but engine thinks flat — syncing")
            self._position_open = True

    # ── Checkpoint Loading ────────────────────────────────────────────

    def _load_tuning(self, force: bool = False):
        """Hot-reload live_tuning.json if changed (mtime check)."""
        path = os.path.join(self._cfg.checkpoint_dir, 'live_tuning.json')
        if not os.path.exists(path):
            # Auto-create with defaults so user has a template to edit
            with open(path, 'w') as f:
                json.dump(_TUNING_DEFAULTS, f, indent=2)
            logger.info(f"Created default live_tuning.json in {self._cfg.checkpoint_dir}/")

        try:
            mt = os.path.getmtime(path)
        except OSError:
            return
        if not force and mt == self._tuning_mtime:
            return  # unchanged

        try:
            with open(path) as f:
                data = json.load(f)
            # Merge with defaults (missing keys get default values)
            merged = dict(_TUNING_DEFAULTS)
            merged.update(data)
            self._tuning = merged
            self._tuning_mtime = mt
            logger.info(f"Tuning reloaded: max_hold={merged['max_hold_seconds']}s  "
                        f"template_dist={merged['gate1_dist']}  "
                        f"sl_mult={merged['exit_sl_mult']}  "
                        f"trail_mult={merged['exit_trail_mult']}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to reload live_tuning.json: {e}")

    def _load_checkpoints(self):
        """Load all training checkpoints needed for live trading."""
        cpdir = self._cfg.checkpoint_dir
        logger.info(f"Loading checkpoints from {cpdir}/")

        # Shared checkpoint loading (same as trainer.py)
        _bundle = load_checkpoints(cpdir, verbose=False)
        self._pattern_library = _bundle.pattern_library
        self._scaler = _bundle.scaler
        self._valid_tids = _bundle.valid_tids
        self._centroids_scaled = _bundle.centroids_scaled
        self._template_tier_map = _bundle.template_tier_map
        self._depth_score_adj = _bundle.depth_score_adj
        self._depth_filter_out = _bundle.depth_filter_out
        self._exception_tids = _bundle.exception_tids

        # Brain — prefer live > forward_pass > training
        live_brain_path = os.path.join(cpdir, 'live_brain.pkl')
        forward_brain_path = os.path.join(cpdir, 'pattern_forward_brain.pkl')
        training_brains = sorted(glob.glob(os.path.join(cpdir, 'pattern_*_brain.pkl')))

        if os.path.exists(live_brain_path):
            self._brain.load(live_brain_path)
            logger.info(f"  Brain: live_brain.pkl ({len(self._brain.table)} states, "
                        f"{len(self._brain.dir_table)} dir pairs)")
        elif os.path.exists(forward_brain_path):
            self._brain.load(forward_brain_path)
            logger.info(f"  Brain: pattern_forward_brain.pkl ({len(self._brain.table)} states, "
                        f"{len(self._brain.dir_table)} dir pairs) — IS-learned directions")
        elif training_brains:
            self._brain.load(training_brains[-1])
            logger.info(f"  Brain: {os.path.basename(training_brains[-1])} (training base)")
        else:
            logger.warning("  No brain checkpoint found — starting fresh "
                          "(will learn from live trades)")

        # Screening Filter (fission + temporal filter)
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

        # Live tuning config (hot-reloadable)
        self._load_tuning(force=True)

    def _init_belief_network(self):
        """Initialize the fractal belief network from loaded checkpoints."""
        _bundle = self._checkpoint_bundle()
        self._belief_network = create_belief_network(_bundle, self._engine)

    def _init_exec_engine(self):
        """Initialize ExecutionEngine — same mode as OOS for parity."""
        _bundle = self._checkpoint_bundle()
        self._exec_engine = create_execution_engine(
            bundle=_bundle,
            brain=self._brain,
            belief_network=self._belief_network,
            exit_engine=self._exit_engine,
            tick_size=self._asset.tick_size,
            point_value=self._asset.point_value,
            mode='oos',
            tier_preference=True,
        )

    def _init_bar_processor(self):
        """Create shared BarProcessor — same decision logic as OOS/replay."""
        self._processor = BarProcessor(
            exec_engine=self._exec_engine,
            belief_network=self._belief_network,
            exit_engine=self._exit_engine,
            brain=self._brain,
            pattern_library=self._pattern_library,
            anchor_tf=self._anchor_tf,
            anchor_depth=self._anchor_depth,
            tick_size=self._asset.tick_size,
            point_value=self._asset.point_value,
        )

    def _checkpoint_bundle(self):
        """Build a CheckpointBundle from already-loaded instance attrs."""
        from core.checkpoint_loader import CheckpointBundle
        return CheckpointBundle(
            pattern_library=self._pattern_library,
            scaler=self._scaler,
            valid_tids=self._valid_tids,
            centroids_scaled=self._centroids_scaled,
            template_tier_map=self._template_tier_map,
            depth_score_adj=self._depth_score_adj,
            depth_filter_out=self._depth_filter_out,
            exception_tids=self._exception_tids,
        )
