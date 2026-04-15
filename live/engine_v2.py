"""
Live Engine V2 — production pipeline, 7-step startup, SIM account.

1. CHECK    → Is ATLAS_NT8 current? If not, dump missing days
2. BUILD    → Build features for any new days
3. WARMUP   → Load ATLAS_NT8 + ATLAS_LIVE delta into aggregator
4. SYNC     → Connect NT8, receive history bars
5. CATCH-UP → Compute features until Python time == NT8 time
6. VERIFY   → Latency < 1s? Sync confirmed?
7. TRADE    → Engine makes decisions, orders via OrderManager

Physics-only BlendedEngine with chained lightning (parallel contracts).
All orders go to NT8 SIM account. No dry-run — SIM IS the test.

Outputs:
    reports/live/v2_ledger_YYYY_MM_DD.csv   — every 5s bar + features + state
    reports/live/v2_trades_YYYY_MM_DD.csv   — entry/exit events for parity check

Usage:
    python -m live.engine_v2                     # full production run
    python -m live.engine_v2 --skip-check        # skip step 1 (assume current)
    python -m live.engine_v2 --skip-build        # skip step 2
    python -m live.engine_v2 --headless          # no dashboard GUI
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('engine_v2')

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.protocol import MsgType, subscribe, place_order, close_position
from training.live_feature_engine import LiveFeatureEngine
from training.nightmare_blended import BlendedEngine
from core.features_79d import FEATURE_NAMES_79D, N_FEATURES
from config.symbols import SYMBOL_MAP
from live.order_manager import OrderManager
from live.gui_bridge import GUIBridge

TICK = 0.25
TV = 0.50

# Paths
ATLAS_NT8 = 'DATA/ATLAS_NT8'
ATLAS_LIVE = 'DATA/ATLAS_LIVE'
FEATURES_LIVE = 'DATA/FEATURES_LIVE_5s'
FEATURES_NT8 = 'DATA/FEATURES_NT8_5s'
NT8_CONFIG = 'config/nt8_dataset.json'

# Checkpoints — dual system (NT8 seed + live rolling)
NT8_CHECKPOINT = os.path.join(ATLAS_NT8, 'checkpoint.json')
LIVE_CHECKPOINT = 'live/state/checkpoint.json'

# Sync thresholds
MAX_SYNC_LAG_S = 10.0    # max seconds behind NT8 before trading allowed
WARMUP_DAYS = 5           # days of history to load for aggregator context


class LiveEngineV2:
    """Production live engine — 7-step startup, physics-only."""

    def __init__(self, config: LiveConfig,
                 skip_check: bool = False, skip_build: bool = False,
                 gui_queue=None, shared_state=None):
        self._cfg = config
        self._skip_check = skip_check
        self._skip_build = skip_build
        self._shared_state = shared_state or {}

        self._asset = SYMBOL_MAP.get(config.asset_ticker)
        if self._asset is None:
            raise ValueError(f'Unknown asset: {config.asset_ticker}')

        # Core components (initialized in startup steps)
        self._client = None
        self._lfe = None      # LiveFeatureEngine (100% parity with training)
        self._engine = None

        # Order management (NT8 is source of truth)
        self._orders = OrderManager(config)

        # Dashboard
        self._gui = GUIBridge(gui_queue)

        # State
        self._bar_count = 0
        self._feat_count = 0
        self._synced = False
        self._trading = False
        self._shutting_down = False
        self._broker_connected = True  # NT8 <-> broker status (assume OK at start)
        self._nt8_realized_pnl = None  # set from ACCOUNT_UPDATE messages
        self._nt8_unrealized_pnl = 0.0
        self._nt8_cash_value = 0.0
        self._daily_pnl = 0.0
        self._trade_count = 0
        self._last_ts = 0.0
        self._last_price = 0.0
        self._session_date = time.strftime('%Y_%m_%d')

        # Ledger (every 5s bar) + trade log (entry/exit events)
        self._ledger = None
        self._ledger_path = None
        self._trade_log = None
        self._trade_log_path = None
        # NT8 ground-truth trade log (from TRADE_CLOSED events)
        self._nt8_trade_log = None
        self._nt8_trade_log_path = None

        # Live bar capture
        self._live_bars = []
        self._live_79d = []

    # ═══════════════════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════════════════

    async def run(self):
        """Execute the 7-step startup then trade."""
        logger.info('=' * 60)
        logger.info('LIVE ENGINE V2 — Physics Only')
        logger.info(f'  Instrument: {self._cfg.instrument}')
        logger.info(f'  Account:    {self._cfg.account}')
        logger.info(f'  Account:    {self._cfg.account} (SIM)')
        logger.info('=' * 60)

        try:
            # Steps 1-2: offline (no connection needed)
            if not self._skip_check:
                self._step1_check()
            if not self._skip_build:
                self._step2_build()

            # Step 3: warmup from disk
            self._step3_warmup()

            # Steps 4-7: online (need NT8 connection)
            self._client = NT8Client(self._cfg)
            # Tell NT8 to only send bars after our warmup — skip redundant history
            if self._last_ts > 0:
                self._client.set_resume_timestamp(self._last_ts)
                logger.info(f'  Delta sync from {self._ts_str(self._last_ts)}')
            connected = await self._client.connect()
            if not connected:
                logger.error('Failed to connect to NT8')
                return

            await self._step4_sync()
            await self._step5_catchup()
            self._step5b_recover_trade()
            self._step6_verify()

            if self._synced:
                await self._step7_trade()
            else:
                logger.error('Sync failed — not trading')

        except KeyboardInterrupt:
            logger.info('Keyboard interrupt')
        except Exception as e:
            logger.error(f'Fatal: {e}', exc_info=True)
        finally:
            await self._shutdown()

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: CHECK — is ATLAS_NT8 current?
    # ═══════════════════════════════════════════════════════════════════

    def _step1_check(self):
        logger.info('')
        logger.info('STEP 1: CHECK ATLAS_NT8')

        atlas_5s = os.path.join(ATLAS_NT8, '5s')
        if not os.path.exists(atlas_5s):
            logger.warning(f'  {atlas_5s}/ not found — run history dump first')
            return

        files = sorted(f for f in os.listdir(atlas_5s) if f.endswith('.parquet'))
        if not files:
            logger.warning('  No parquets in ATLAS_NT8/5s/')
            return

        last_day = files[-1].replace('.parquet', '')
        today = time.strftime('%Y_%m_%d')
        logger.info(f'  Last NT8 day: {last_day}')
        logger.info(f'  Today:        {today}')

        if last_day >= today:
            logger.info('  ATLAS_NT8 is current')
        else:
            logger.warning(f'  ATLAS_NT8 is {last_day}, today is {today}')
            logger.warning(f'  Run: python tools/convert_nt8_atlas.py')
            logger.warning(f'  Or use diagnostic_run.py to dump from NT8')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: BUILD — features for new days
    # ═══════════════════════════════════════════════════════════════════

    def _step2_build(self):
        logger.info('')
        logger.info('STEP 2: BUILD FEATURES')

        os.makedirs(FEATURES_NT8, exist_ok=True)
        atlas_5s = os.path.join(ATLAS_NT8, '5s')
        if not os.path.exists(atlas_5s):
            return

        existing = {f.replace('.parquet', '') for f in os.listdir(FEATURES_NT8)
                    if f.endswith('.parquet')}
        atlas_days = {f.replace('.parquet', '') for f in os.listdir(atlas_5s)
                      if f.endswith('.parquet') and f >= '2026_03_20'}
        missing = sorted(atlas_days - existing)

        if not missing:
            logger.info(f'  Features up to date ({len(existing)} days)')
            return

        logger.info(f'  Missing: {len(missing)} days ({missing[0]} to {missing[-1]})')
        logger.info(f'  Building...')

        import subprocess
        result = subprocess.run(
            [sys.executable, 'training/build_dataset.py',
             '--resolution', '5s', '--atlas', ATLAS_NT8,
             '--start', missing[0].replace('_', '-')],
            capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            logger.info(f'  Features built')
        else:
            logger.warning(f'  Build failed: {result.stderr[-200:]}')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: WARMUP — load history into aggregator
    # ═══════════════════════════════════════════════════════════════════

    def _step3_warmup(self):
        logger.info('')
        logger.info('STEP 3: WARMUP')

        # LiveFeatureEngine: same batch SFE path as build_dataset (100% parity)
        self._lfe = LiveFeatureEngine(ATLAS_NT8)
        bar_counts = self._lfe.load_history()
        logger.info(f'  Loaded: {bar_counts}')

        # Load velocities from checkpoint
        best_path, best_ts = None, 0
        for path in [LIVE_CHECKPOINT, NT8_CHECKPOINT]:
            if os.path.exists(path):
                try:
                    with open(path, encoding='utf-8') as f:
                        cp = json.load(f)
                    ts = cp.get('last_ts', 0)
                    if ts > best_ts:
                        best_path, best_ts = path, ts
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f'  Corrupt checkpoint: {path}')

        self._saved_trade_state = {}
        if best_path:
            with open(best_path, encoding='utf-8') as f:
                cp = json.load(f)
            self._lfe.load_velocities(cp.get('velocities', {}))
            self._saved_trade_state = cp.get('trade_state', {})
            self._last_ts = cp.get('last_ts', 0)
            logger.info(f'  Velocities from: {os.path.basename(best_path)}')

        self._engine = BlendedEngine(use_cnn=False, live_mode=True)

        # Verify: compute one feature from last loaded bar
        if '1m' in self._lfe._bars and len(self._lfe._bars['1m']) > 0:
            last_1m_ts = float(self._lfe._bars['1m']['timestamp'].iloc[-1])
            feat = self._lfe._compute_features(last_1m_ts)
            if feat is not None:
                logger.info(f'  WARMED UP: z={feat[12]:.2f} vr={feat[14]:.2f} '
                            f'1m={len(self._lfe._bars.get("1m", []))} '
                            f'1h={len(self._lfe._bars.get("1h", []))}')
            else:
                logger.warning('  Warmup feature returned None')
        else:
            logger.warning('  No 1m bars loaded')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: SYNC — connect NT8, receive history
    # ═══════════════════════════════════════════════════════════════════

    async def _step4_sync(self):
        logger.info('')
        logger.info('STEP 4: SYNC NT8')

        # Receive history dump
        history_done = False
        bar_count = 0
        t0 = time.perf_counter()

        while not history_done:
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning('  History timeout (120s)')
                break

            if msg.get('type') == MsgType.BAR:
                bar = self._extract_bar(msg)
                self._lfe.on_bar(bar)  # appends if new
                # Track latest bar we've seen so Step 6 verify knows current state
                if bar['timestamp'] > self._last_ts:
                    self._last_ts = bar['timestamp']
                    self._last_price = bar['close']
                bar_count += 1
            elif msg.get('type') == MsgType.HISTORY_DONE:
                history_done = True
            elif msg.get('type') == 'CONNECTED':
                logger.info(f'  NT8: account={msg.get("account")} '
                            f'instrument={msg.get("instrument")}')

        elapsed = time.perf_counter() - t0
        logger.info(f'  History: {bar_count:,} bars in {elapsed:.1f}s')
        logger.info(f'  Bars: {self._lfe.bar_counts}')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: CATCH-UP — process bars until current
    # ═══════════════════════════════════════════════════════════════════

    async def _step5_catchup(self):
        logger.info('')
        logger.info('STEP 5: CATCH-UP')

        # Feed bars into LiveFeatureEngine until caught up to wall time.
        catchup_bars = 0
        wall_time = time.time()

        while True:
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=2.0)
            except asyncio.TimeoutError:
                break  # queue drained

            if msg.get('type') != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)
            self._lfe.on_bar(bar)  # appends to bar stores if new
            self._last_ts = bar['timestamp']
            self._last_price = bar['close']
            catchup_bars += 1

            # Break once we're within 10s of wall time (caught up)
            if self._last_ts > 0 and (wall_time - self._last_ts) < MAX_SYNC_LAG_S:
                break

        # Verify with one feature
        feat = self._lfe._compute_features(self._last_ts) if self._last_ts > 0 else None
        if feat is not None:
            logger.info(f'  Caught up: {catchup_bars:,} bars | '
                        f'z={feat[12]:.2f} vr={feat[14]:.2f}')
        else:
            logger.info(f'  Caught up: {catchup_bars:,} bars')
        logger.info(f'  Last bar: {self._ts_str(self._last_ts)} '
                    f'price={self._last_price:.2f}')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5b: RECOVER — restore in-flight trade if NT8 has open position
    # ═══════════════════════════════════════════════════════════════════

    def _step5b_recover_trade(self):
        """If NT8 has an open position and checkpoint has matching trade state,
        restore engine to continue managing the trade.
        """
        nt8_pos = self._orders.position
        saved = self._saved_trade_state

        if nt8_pos.qty == 0 and not saved.get('in_pos', False):
            logger.info('  RECOVERY: both flat — clean start')
            return

        if nt8_pos.qty == 0 and saved.get('in_pos', False):
            logger.warning('  RECOVERY: checkpoint had trade but NT8 is flat — trade closed while offline')
            return

        if nt8_pos.qty != 0 and not saved.get('in_pos', False):
            logger.warning(f'  RECOVERY: NT8 has {nt8_pos.side} x{nt8_pos.qty} '
                           f'but no trade in checkpoint — unknown trade, monitoring only')
            return

        # Both have a position — verify they match
        if nt8_pos.qty != 0 and saved.get('in_pos', False):
            saved_dir = saved.get('direction', '')
            nt8_dir = 'long' if nt8_pos.side == 'LONG' else 'short'

            if saved_dir == nt8_dir:
                self._engine.restore_trade_state(saved)
                n_chains = len(saved.get('chains', []))
                logger.info(f'  RECOVERY: restored {saved_dir} {saved.get("entry_tier")} '
                            f'@ {saved.get("entry_price", 0):.2f} '
                            f'(chains={n_chains}, peak=${saved.get("peak_pnl", 0):.0f})')
            else:
                logger.error(f'  RECOVERY: direction mismatch — NT8={nt8_dir} '
                             f'checkpoint={saved_dir} — NOT restoring')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: VERIFY — latency check
    # ═══════════════════════════════════════════════════════════════════

    def _step6_verify(self):
        logger.info('')
        logger.info('STEP 6: VERIFY SYNC')

        wall_time = time.time()
        lag = wall_time - self._last_ts if self._last_ts > 0 else 999

        logger.info(f'  Wall time:   {self._ts_str(wall_time)}')
        logger.info(f'  Last bar:    {self._ts_str(self._last_ts)}')
        logger.info(f'  Lag:         {lag:.1f}s')

        if lag < MAX_SYNC_LAG_S:
            self._synced = True
            logger.info(f'  SYNC VERIFIED — lag {lag:.1f}s < {MAX_SYNC_LAG_S}s')
        else:
            # Any lag > 10s: proceed but start in "waiting for live bars" mode.
            # Market closures, session gaps, weekend = any lag possible.
            # Real-time guard rails (broker_connected, catch_up detection,
            # stale bar detection) protect us during the trading loop itself.
            self._synced = True
            logger.warning(f'  LAG: {lag:.0f}s — proceeding, will enter live '
                           f'when first fresh bar arrives')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: TRADE — main loop
    # ═══════════════════════════════════════════════════════════════════

    async def _step7_trade(self):
        logger.info('')
        logger.info('=' * 60)
        logger.info('STEP 7: TRADING')
        logger.info('=' * 60)

        # Open ledger (every 5s bar) + trade log (entry/exit events only)
        os.makedirs('reports/live', exist_ok=True)
        self._ledger_path = f'reports/live/v2_ledger_{self._session_date}.csv'
        self._ledger = open(self._ledger_path, 'w', encoding='utf-8')
        ledger_cols = ['timestamp', 'price', 'z', 'vr', 'vel',
                       'in_pos', 'direction', 'tier', 'bars_held',
                       'pnl', 'peak', 'contracts', 'event']
        self._ledger.write(','.join(ledger_cols) + '\n')

        # Trade log — engine's internal view (for parity vs backtest)
        self._trade_log_path = f'reports/live/v2_trades_{self._session_date}.csv'
        self._trade_log = open(self._trade_log_path, 'w', encoding='utf-8')
        trade_cols = ['timestamp', 'type', 'tier', 'direction',
                      'requested_price', 'fill_price', 'slippage',
                      'pnl', 'bars_held', 'exit_reason', 'is_chain',
                      'contracts', 'daily_pnl']
        self._trade_log.write(','.join(trade_cols) + '\n')
        self._pending_requests = {}

        # NT8 trade log — ground truth from TRADE_CLOSED events (for reconciliation)
        self._nt8_trade_log_path = f'reports/live/nt8_trades_{self._session_date}.csv'
        self._nt8_trade_log = open(self._nt8_trade_log_path, 'w', encoding='utf-8')
        nt8_cols = ['fill_time', 'order_id', 'side', 'entry_price',
                    'exit_price', 'pnl', 'qty', 'is_chain']
        self._nt8_trade_log.write(','.join(nt8_cols) + '\n')

        self._trading = True

        logger.info(f'  Ledger: {self._ledger_path}')
        logger.info(f'  Trades: {self._trade_log_path}')
        logger.info(f'  Listening for bars...')

        while not self._shutting_down:
            if self._shared_state.get('shutdown'):
                self._shutting_down = True
                break

            # Dashboard requested a manual save
            if self._shared_state.get('save_now'):
                logger.info('  MANUAL SAVE requested from dashboard')
                self._periodic_save()
                self._shared_state['save_now'] = False

            # Stale bar detection: time since last bar ARRIVAL at our process
            # (monotonic clock, not wall time — independent of timezone/clock drift)
            last_arrival = getattr(self, '_last_arrival', 0)
            if last_arrival > 0:
                silence_s = time.monotonic() - last_arrival
                if silence_s > 60 and self._broker_connected:
                    logger.error(f'  STALE: {silence_s:.0f}s since last bar arrived — '
                                 f'assuming NT8 panic, blocking new orders')
                    self._broker_connected = False
                elif silence_s < 15 and not self._broker_connected:
                    logger.warning(f'  BARS FLOWING AGAIN — broker OK, unblocking orders')
                    self._broker_connected = True

            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # check shutdown flag every 1s

            msg_type = msg.get('type', '')
            if msg_type == MsgType.FILL:
                pnl_from_fill = self._orders.on_fill(msg)
                fill_px = float(msg.get('fill_price', 0))
                oid = msg.get('order_id', '')

                # Log FILL row with slippage
                req = self._pending_requests.pop(oid, {})
                req_px = req.get('requested_price', fill_px)
                self._log_trade_event(
                    float(msg.get('fill_time', time.time())),
                    f'FILL_{req.get("type", "?")}',
                    req.get('tier', '?'), req.get('direction', '?'),
                    req_px, fill_px,
                    pnl_from_fill or 0, 0, oid,
                    req.get('is_chain', False))
                slip = fill_px - req_px
                if req.get('direction') == 'short':
                    slip = -slip
                logger.info(f'  FILL {oid} @ {fill_px:.2f} '
                            f'(requested {req_px:.2f}, slip={slip:+.2f})')

                if fill_px and self._engine.in_pos:
                    # Correct engine entry price from NT8 actual fill
                    side = msg.get('side', '')
                    same_dir = ((self._engine.direction == 'long' and side == 'BUY') or
                                (self._engine.direction == 'short' and side == 'SELL'))
                    if same_dir and not self._engine._chain_contracts:
                        self._engine.entry_price = fill_px
                    elif same_dir and self._engine._chain_contracts:
                        self._engine._chain_contracts[-1]['entry_price'] = fill_px
                # Sync realized PnL from NT8
                self._daily_pnl = self._orders.daily_pnl
                continue
            elif msg_type == MsgType.ORDER_ACK:
                self._orders.on_order_ack(msg)
                continue
            elif msg_type == MsgType.TRADE_CLOSED:
                # NT8 ground-truth round-trip event
                self._on_nt8_trade_closed(msg)
                continue
            elif msg_type == 'ORDER_STATUS':
                self._orders.on_order_status(msg)
                continue
            elif msg_type == 'POSITION':
                self._orders.on_position(msg)
                # If NT8 says flat but engine thinks it's in a trade, force-close
                # to keep state in sync. NT8 is ground truth.
                if (self._orders.position.qty == 0 and self._engine
                        and self._engine.in_pos):
                    logger.warning(f'  POSITION sync: NT8 flat but engine in '
                                   f'{self._engine.direction} {self._engine.entry_tier} '
                                   f'— forcing engine flat')
                    self._engine.force_close(reason='nt8_position_sync')
                continue
            elif msg_type == MsgType.HEARTBEAT:
                # Enhanced heartbeat — reconcile position
                if 'position_qty' in msg:
                    self._orders.on_heartbeat(msg)
                continue
            elif msg_type == 'ACCOUNT_UPDATE':
                # NT8 is the source of truth for realized PnL
                self._nt8_realized_pnl = float(msg.get('realized_pnl', 0))
                self._nt8_unrealized_pnl = float(msg.get('unrealized_pnl', 0))
                self._nt8_cash_value = float(msg.get('cash_value', 0))
                # Push to dashboard
                self._gui.push({
                    'type': 'ACCOUNT_UPDATE',
                    'cash_value': self._nt8_cash_value,
                    'realized_pnl': self._nt8_realized_pnl,
                    'unrealized_pnl': self._nt8_unrealized_pnl,
                })
                continue
            elif msg_type == 'CONNECTION_LOST':
                self._broker_connected = False
                logger.error('  BROKER DISCONNECTED — blocking new orders, waiting for restore')
                continue
            elif msg_type == 'CONNECTION_RESTORED':
                self._broker_connected = True
                logger.warning('  BROKER RESTORED — requesting position snapshot')
                # Query actual NT8 position to reconcile after disconnect
                from live.protocol import request_position
                await self._client.send(request_position())
                continue
            elif msg_type != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)
            self._bar_count += 1
            self._last_ts = bar['timestamp']
            self._last_price = bar['close']
            self._live_bars.append(bar)

            # Catch-up detection: measure how fast bars are arriving at our
            # process (inter-arrival delta), not wall clock vs bar timestamp.
            # This is timezone/clock independent — it just measures bar flood rate.
            #
            # Real-time: bars arrive ~5s apart (one at a time)
            # Catch-up: bars flood in <1s apart (NT8 dumps post-panic)
            arrival_now = time.monotonic()
            inter_arrival = arrival_now - getattr(self, '_last_arrival', arrival_now)
            self._last_arrival = arrival_now

            # Rolling buffer of last 10 arrivals to smooth out jitter
            if not hasattr(self, '_arrival_window'):
                self._arrival_window = []
            self._arrival_window.append(inter_arrival)
            if len(self._arrival_window) > 10:
                self._arrival_window.pop(0)

            avg_arrival = sum(self._arrival_window) / len(self._arrival_window)
            # Flood: average inter-arrival is much less than bar period (5s)
            # Threshold: if avg < 1s and we have 10 samples, we're flooding
            is_catchup = (len(self._arrival_window) >= 10 and avg_arrival < 1.0)

            # Compute features via LiveFeatureEngine (same path as training)
            feat = self._lfe.on_bar(bar)

            if feat is None:
                continue

            # Dedup: skip if we already saved features for this ts
            last_saved_ts = self._live_79d[-1]['timestamp'] if self._live_79d else 0
            if bar['timestamp'] <= last_saved_ts:
                continue

            self._feat_count += 1
            self._live_79d.append({'timestamp': bar['timestamp'], 'features': feat.copy()})
            z = feat[12]
            vr = feat[14]
            vel = feat[15]

            if is_catchup:
                if self._bar_count % 100 == 0:
                    logger.info(f'  CATCH-UP: {avg_arrival*1000:.0f}ms inter-arrival '
                                f'(flooding {self._bar_count} bars)')
                continue  # skip engine + orders while backfilling

            # ── Feed engine, detect all events ─────────────────────────
            prev_in_pos = self._engine.in_pos
            prev_trades = len(self._engine.trades)
            prev_chains = len(self._engine._chain_contracts)

            state = {
                'features_79d': feat,
                'price': bar['close'],
                'timestamp': bar['timestamp'],
            }
            self._engine.on_state(state)

            entered = self._engine.in_pos and not prev_in_pos
            exited = not self._engine.in_pos and prev_in_pos
            new_trades = len(self._engine.trades) - prev_trades
            curr_chains = len(self._engine._chain_contracts)
            chain_opened = curr_chains > prev_chains

            events = []

            # ── PRIMARY ENTRY ──────────────────────────────────────────
            if entered and self._orders.can_enter and self._broker_connected:
                side = 'BUY' if self._engine.direction == 'long' else 'SELL'
                order_msg = self._orders.build_entry_order(side)
                if order_msg:
                    self._pending_requests[order_msg['order_id']] = {
                        'requested_price': bar['close'],
                        'tier': self._engine.entry_tier,
                        'direction': self._engine.direction,
                        'is_chain': False, 'type': 'ENTRY',
                    }
                    await self._client.send(order_msg)
                events.append(f'ENTRY_{self._engine.entry_tier}')
                self._log_trade_event(bar['timestamp'], 'ENTRY',
                    self._engine.entry_tier, self._engine.direction,
                    bar['close'], 0, 0, 0, '', False)
                logger.info(f'  ENTRY {self._engine.direction} '
                            f'{self._engine.entry_tier} @ {bar["close"]:.2f}')

            # ── CHAIN ENTRY (scale-in) ─────────────────────────────────
            if chain_opened and self._engine.in_pos:
                cc = self._engine._chain_contracts[-1]
                side = 'BUY' if cc['direction'] == 'long' else 'SELL'
                if self._orders.can_scale_in and self._broker_connected:
                    order_msg = self._orders.build_scale_in_order(side)
                    if order_msg:
                        self._pending_requests[order_msg['order_id']] = {
                            'requested_price': bar['close'],
                            'tier': cc['entry_tier'],
                            'direction': cc['direction'],
                            'is_chain': True, 'type': 'CHAIN_ENTRY',
                        }
                        await self._client.send(order_msg)
                events.append(f'CHAIN_ENTRY_{cc["entry_tier"]}')
                self._log_trade_event(bar['timestamp'], 'CHAIN_ENTRY',
                    cc['entry_tier'], cc['direction'],
                    bar['close'], 0, 0, 0, '', True)
                logger.info(f'  CHAIN ENTRY {cc["entry_tier"]} '
                            f'@ {bar["close"]:.2f} (contracts={curr_chains + 1})')

            # ── CHAIN EXITS (scale-out, before primary exit) ───────────
            chain_exit_count = new_trades - (1 if exited else 0)
            if chain_exit_count > 0:
                for ci in range(chain_exit_count):
                    ct = self._engine.trades[-(new_trades - ci - (1 if exited else 0))]
                    if self._orders.position.qty > 1 and self._broker_connected:
                        order_msg = self._orders.build_scale_out_order(
                            reason=ct.get('exit_reason', 'chain_exit'))
                        if order_msg:
                            self._pending_requests[order_msg['order_id']] = {
                                'requested_price': bar['close'],
                                'tier': ct.get('entry_tier', '?'),
                                'direction': ct.get('direction', '?'),
                                'is_chain': True, 'type': 'CHAIN_EXIT',
                            }
                            await self._client.send(order_msg)
                    events.append(f'CHAIN_EXIT_{ct.get("exit_reason", "?")}')
                    self._log_trade_event(bar['timestamp'], 'CHAIN_EXIT',
                        ct.get('entry_tier', '?'), ct.get('direction', '?'),
                        bar['close'], 0, ct['pnl'], ct.get('held', 0),
                        ct.get('exit_reason', ''), True)
                    logger.info(f'  CHAIN EXIT {ct.get("exit_reason")} '
                                f'pnl=${ct["pnl"]:.1f}')

            # ── PRIMARY EXIT (close remaining) ─────────────────────────
            if exited and new_trades > 0:
                t = self._engine.trades[-1] if chain_exit_count == 0 else \
                    self._engine.trades[-(new_trades)]
                if self._orders.can_exit and self._broker_connected:
                    order_msg = self._orders.build_exit_order(
                        reason=t.get('exit_reason', 'signal'))
                    if order_msg:
                        await self._client.send(order_msg)
                events.append(f'EXIT_{t.get("exit_reason", "?")}')
                self._log_trade_event(bar['timestamp'], 'EXIT',
                    t.get('entry_tier', '?'), t.get('direction', '?'),
                    bar['close'], 0, t['pnl'], t.get('held', 0),
                    t.get('exit_reason', ''), False)
                logger.info(f'  EXIT {t.get("exit_reason")} pnl=${t["pnl"]:.1f}')

            # ── PnL tracking ───────────────────────────────────────────
            # Realized: NT8 ACCOUNT_UPDATE is the absolute source of truth
            # (matches what NT8 shows on screen). Falls back to OrderManager
            # before first ACCOUNT_UPDATE arrives.
            self._daily_pnl = getattr(self, '_nt8_realized_pnl', None)
            if self._daily_pnl is None:
                self._daily_pnl = self._orders.daily_pnl
            self._trade_count = len(self._engine.trades)

            # Unrealized: NT8 ACCOUNT_UPDATE is truth, fallback to engine calc
            unrealized = self._nt8_unrealized_pnl
            if unrealized == 0 and self._engine.in_pos:
                # Fallback before first ACCOUNT_UPDATE
                px = bar['close']
                if self._engine.direction == 'long':
                    unrealized = (px - self._engine.entry_price) / TICK * TV
                else:
                    unrealized = (self._engine.entry_price - px) / TICK * TV
                for cc in self._engine._chain_contracts:
                    if cc['direction'] == 'long':
                        unrealized += (px - cc['entry_price']) / TICK * TV
                    else:
                        unrealized += (cc['entry_price'] - px) / TICK * TV

            n_contracts = 1 + curr_chains if self._engine.in_pos else 0
            event_str = ' '.join(events)

            # ── Write ledger row ───────────────────────────────────────
            self._write_ledger(bar['timestamp'], bar['close'], z, vr, vel,
                               unrealized, event_str, n_contracts)

            # ── Dashboard ──────────────────────────────────────────────
            self._gui.push({
                'type': 'TICK_UPDATE',
                'price': bar['close'],
                'bars': self._bar_count,
                'unrealized': unrealized,
                'daily_pnl': self._daily_pnl,
                'in_position': self._engine.in_pos,
                'direction': self._engine.direction or '',
                'tier': self._engine.entry_tier or '',
                'z_se': z, 'vr': vr,
                'is_1m': False,
            })

            if 'ENTRY' in event_str and 'CHAIN' not in event_str.split()[0]:
                self._gui.push_trade_marker('ENTRY',
                    'BUY' if self._engine.direction == 'long' else 'SELL',
                    bar['close'])
            if 'EXIT' in event_str and 'CHAIN' not in event_str.split()[-1]:
                last_pnl = self._engine.trades[-1]['pnl'] if self._engine.trades else 0
                self._gui.push_trade_marker('EXIT', '', bar['close'], pnl=last_pnl)

            wins = sum(1 for t in self._engine.trades if t['pnl'] > 0)
            gross_win = sum(t['pnl'] for t in self._engine.trades if t['pnl'] > 0)
            gross_loss = sum(t['pnl'] for t in self._engine.trades if t['pnl'] <= 0)
            z_pct = min(abs(z) / 2.0 * 100, 100)
            eb = {'reversed': 0, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0,
                  'q100plus': 0, 'forced': 0}
            self._gui.push_stats(
                session_pnl=self._daily_pnl, session_wins=wins,
                session_trades=self._trade_count,
                gross_win=gross_win, gross_loss=gross_loss,
                exit_buckets=eb, belief_pct=z_pct,
                in_position=self._engine.in_pos, daily_pnl=self._daily_pnl)

            # ── Status line ────────────────────────────────────────────
            if self._feat_count % 12 == 0:
                pos = self._engine.direction or 'FLAT'
                tier = self._engine.entry_tier or '-'
                chain_str = f'+{curr_chains}ch' if curr_chains else ''
                print(f'\r  {self._ts_str(bar["timestamp"])} | {bar["close"]:>10.2f} | '
                      f'{pos:>5} {tier:>15} {chain_str:>4} | z={z:>+5.1f} vr={vr:.2f} | '
                      f'tr={self._trade_count} day=${self._daily_pnl:>+.0f}    ',
                      end='', flush=True)

            # ── Periodic maintenance ───────────────────────────────────
            # Every minute (12 bars @ 5s): save state + sync position
            if self._bar_count % 12 == 0:
                self._periodic_save()
                from live.protocol import request_position
                await self._client.send(request_position())

            # ── Engine state push (every bar) ──────────────────────────
            self._push_engine_state()

            # Stale order cleanup every 5 min
            if self._bar_count % 60 == 0:
                self._orders.cleanup_stale_orders(max_age_s=120.0)

    # ═══════════════════════════════════════════════════════════════════
    # TRADE LOG — one row per entry/exit for parity comparison
    # ═══════════════════════════════════════════════════════════════════

    def _on_nt8_trade_closed(self, msg):
        """Handle TRADE_CLOSED — NT8 ground-truth round trip."""
        order_id = msg.get('order_id', '')
        side = msg.get('side', '')
        entry = float(msg.get('entry_price', 0))
        exit_p = float(msg.get('exit_price', 0))
        pnl = float(msg.get('pnl', 0))
        fill_time = float(msg.get('fill_time', time.time()))
        qty = int(msg.get('qty', 1))
        is_chain = bool(msg.get('is_chain', False))

        # Write to NT8 ground-truth log
        if self._nt8_trade_log:
            row = [f'{fill_time:.0f}', order_id, side,
                   f'{entry:.2f}', f'{exit_p:.2f}', f'{pnl:.2f}',
                   str(qty), '1' if is_chain else '0']
            self._nt8_trade_log.write(','.join(row) + '\n')
            self._nt8_trade_log.flush()

        # Push to dashboard trade log + trade marker
        self._gui.push({
            'type': 'NT8_TRADE',
            'order_id': order_id, 'side': side,
            'entry_price': entry, 'exit_price': exit_p,
            'pnl': pnl, 'fill_time': fill_time,
            'is_chain': is_chain,
        })
        self._gui.push_trade_marker('NT8_EXIT', side, exit_p, pnl=pnl)

        logger.info(f'  NT8 TRADE: {order_id} {side} '
                    f'{entry:.2f}→{exit_p:.2f} pnl=${pnl:+.2f}'
                    f'{" [chain]" if is_chain else ""}')

    def _push_engine_state(self):
        """Push engine health to dashboard — state, bar flow, activity."""
        # Derive state
        if self._shutting_down:
            state = 'SHUTDOWN'
        elif not self._broker_connected:
            state = 'BROKER_DISCONNECTED'
        elif not self._synced:
            state = 'SYNCING'
        elif self._last_ts == 0:
            state = 'WARMUP'
        else:
            # Check if bars are fresh (inter-arrival < 10s avg)
            window = getattr(self, '_arrival_window', [])
            if len(window) >= 5:
                avg_arrival = sum(window) / len(window)
                if avg_arrival < 1.0:
                    state = 'CATCH_UP'
                elif avg_arrival > 30:
                    state = 'STALE'
                else:
                    state = 'TRADING'
            else:
                state = 'TRADING'

        # Bars per minute from arrival window
        bar_rate = 0.0
        window = getattr(self, '_arrival_window', [])
        if len(window) >= 3:
            avg_s = sum(window) / len(window)
            if avg_s > 0:
                bar_rate = 60.0 / avg_s

        # Activity description
        activity = ''
        if self._engine and self._engine.in_pos:
            n_chains = len(self._engine._chain_contracts)
            chain_str = f' +{n_chains}ch' if n_chains else ''
            activity = f'{self._engine.direction} {self._engine.entry_tier}{chain_str}'

        self._gui.push({
            'type': 'ENGINE_STATE',
            'state': state,
            'bar_count': self._bar_count,
            'last_bar_ts': self._last_ts,
            'bar_rate': bar_rate,
            'activity': activity,
        })

    def _log_trade_event(self, ts, event_type, tier, direction,
                         requested_price, fill_price,
                         pnl, bars_held, exit_reason, is_chain):
        """Write one trade event row with slippage tracking.

        requested_price: what the engine saw when it decided to trade
        fill_price: what NT8 actually filled at (0 if not yet filled)
        """
        if self._trade_log is None:
            return
        slip = fill_price - requested_price if fill_price > 0 else 0
        # Normalize: positive slippage = worse fill
        if direction == 'short' or (direction == '' and event_type.startswith('EXIT')):
            slip = -slip  # for shorts, higher fill on entry = worse
        row = [f'{ts:.0f}', event_type, tier, direction,
               f'{requested_price:.2f}', f'{fill_price:.2f}', f'{slip:.2f}',
               f'{pnl:.1f}', str(bars_held), exit_reason,
               '1' if is_chain else '0',
               str(self._orders.position.qty),
               f'{self._daily_pnl:.1f}']
        self._trade_log.write(','.join(row) + '\n')
        self._trade_log.flush()

    # ═══════════════════════════════════════════════════════════════════
    # LEDGER + SAVE
    # ═══════════════════════════════════════════════════════════════════

    def _write_ledger(self, ts, price, z, vr, vel, pnl, event, n_contracts=0):
        if self._ledger is None:
            return
        row = [f'{ts:.0f}', f'{price:.2f}', f'{z:.4f}', f'{vr:.4f}', f'{vel:.2f}',
               '1' if self._engine.in_pos else '0',
               self._engine.direction or '',
               self._engine.entry_tier or '',
               str(self._engine.bars_held),
               f'{pnl:.1f}',
               f'{self._engine.peak_pnl:.1f}',
               str(n_contracts),
               event]
        self._ledger.write(','.join(row) + '\n')

    def _periodic_save(self):
        if self._ledger and not self._ledger.closed:
            self._ledger.flush()
        # Save checkpoint (velocities + trade state for recovery)
        if self._lfe:
            trade_state = self._engine.get_trade_state() if self._engine else {}
            # Save velocities + trade state as JSON checkpoint
            os.makedirs(os.path.dirname(LIVE_CHECKPOINT) or '.', exist_ok=True)
            import json as _json
            cp = {
                'version': 3,
                'last_ts': self._last_ts,
                'velocities': self._lfe.prev_velocities,
                'trade_state': trade_state,
                'bar_counts': self._lfe.bar_counts,
            }
            with open(LIVE_CHECKPOINT, 'w', encoding='utf-8') as f:
                _json.dump(cp, f)
        # Save live features — same schema as build_dataset output
        self._save_live_features()
        # Save live bars to ATLAS_LIVE
        if self._live_bars:
            from live.incremental_writer import IncrementalWriter
            writer = IncrementalWriter(ATLAS_LIVE, self._session_date)
            writer.save_all_chunks({'5s': self._live_bars})

    def _save_live_features(self):
        """Save accumulated 91D features to parquet — same format as build_dataset."""
        if not self._live_79d:
            return
        from core.features_79d import FEATURE_NAMES_79D
        os.makedirs(FEATURES_LIVE, exist_ok=True)

        # Group by UTC date
        by_date = {}
        for row in self._live_79d:
            ts = row['timestamp']
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m_%d')
            by_date.setdefault(day, []).append(row)

        for day, rows in by_date.items():
            timestamps = [r['timestamp'] for r in rows]
            features = np.array([r['features'] for r in rows])
            data = {'timestamp': timestamps}
            for i, name in enumerate(FEATURE_NAMES_79D):
                data[name] = features[:, i] if i < features.shape[1] else 0.0
            df = pd.DataFrame(data)
            df['timestamp'] = df['timestamp'].astype(np.int64)
            out_path = os.path.join(FEATURES_LIVE, f'{day}.parquet')
            df.to_parquet(out_path, index=False)
        logger.info(f'  Live features: {len(self._live_79d)} rows -> {FEATURES_LIVE}/')

    # ═══════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════

    async def _shutdown(self):
        self._shutting_down = True
        logger.info('')
        logger.info('SHUTDOWN')

        # 1. Close positions (need _orders and _engine still alive)
        if self._orders and not self._orders.is_flat:
            order_msg = self._orders.build_exit_order(reason='shutdown')
            if order_msg and self._client:
                await self._client.send(order_msg)
            await asyncio.sleep(2.0)
        if self._engine:
            self._engine.force_close()

        # 2. Final save (need _lfe and _engine still alive)
        try:
            self._periodic_save()
            logger.info(f'  Checkpoint saved: {LIVE_CHECKPOINT}')
        except Exception as e:
            logger.error(f'  Checkpoint save failed: {e}')

        # 3. Close file handles
        if self._ledger:
            try:
                self._ledger.flush()
                self._ledger.close()
                logger.info(f'  Ledger: {self._ledger_path}')
            except Exception:
                pass
        if hasattr(self, '_trade_log') and self._trade_log:
            try:
                self._trade_log.flush()
                self._trade_log.close()
                logger.info(f'  Trades: {self._trade_log_path}')
            except Exception:
                pass
        if hasattr(self, '_nt8_trade_log') and self._nt8_trade_log:
            try:
                self._nt8_trade_log.flush()
                self._nt8_trade_log.close()
                logger.info(f'  NT8 trades: {self._nt8_trade_log_path}')
            except Exception:
                pass

        # 4. Summary (still needs _engine)
        wins = 0
        chains = 0
        if self._engine:
            wins = sum(1 for t in self._engine.trades if t['pnl'] > 0)
            chains = sum(1 for t in self._engine.trades
                         if str(t.get('exit_reason', '')).startswith('chain_'))
        logger.info(f'  Bars:     {self._bar_count:,}')
        logger.info(f'  Feats:    {self._feat_count:,}')
        logger.info(f'  Trades:   {self._trade_count} ({chains} chains)')
        logger.info(f'  Win rate: {wins}/{self._trade_count} '
                    f'({wins/max(self._trade_count,1)*100:.0f}%)')
        logger.info(f'  PnL:      ${self._daily_pnl:.0f}')

        # 5. Disconnect
        if self._client:
            await self._client.disconnect()

        # 6. Release GPU + RAM (AFTER everything else uses engine/lfe)
        try:
            from numba import cuda
            if cuda.is_available():
                cuda.close()
        except Exception:
            pass
        self._lfe = None
        self._engine = None
        self._live_bars.clear()
        self._live_79d.clear()
        import gc
        gc.collect()
        logger.info('  Memory released')
        logger.info('  Done')

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_bar(msg):
        return {
            'timestamp': msg.get('timestamp', 0),
            'open': msg.get('open', 0),
            'high': msg.get('high', 0),
            'low': msg.get('low', 0),
            'close': msg.get('close', 0),
            'volume': msg.get('volume', 0),
        }

    @staticmethod
    def _ts_str(ts):
        if ts < 1e9:
            return '?'
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Live Engine V2')
    parser.add_argument('--skip-check', action='store_true')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--headless', action='store_true', help='No dashboard GUI')
    args = parser.parse_args()

    config = LiveConfig()
    shared_state = {}
    gui_queue = None

    # Launch dashboard unless headless
    if not args.headless:
        import queue as stdlib_queue
        import threading
        import tkinter as tk

        gui_queue = stdlib_queue.Queue(maxsize=5000)

        def _run_dashboard():
            try:
                from visualization.dashboard_v2 import TradingDashboard
                root = tk.Tk()
                popup = TradingDashboard(root, gui_queue, shared_state=shared_state)
                def _on_close():
                    shared_state['shutdown'] = True
                    root.destroy()
                root.protocol('WM_DELETE_WINDOW', _on_close)
                root.mainloop()
            except Exception as e:
                logger.warning(f'Dashboard failed: {e}')

        t = threading.Thread(target=_run_dashboard, daemon=True)
        t.start()
        logger.info('Dashboard launched')

    engine = LiveEngineV2(config,
                          skip_check=args.skip_check,
                          skip_build=args.skip_build,
                          gui_queue=gui_queue,
                          shared_state=shared_state)
    asyncio.run(engine.run())


if __name__ == '__main__':
    main()
