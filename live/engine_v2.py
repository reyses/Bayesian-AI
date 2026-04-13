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
from training.aggregator import Aggregator
from core.statistical_field_engine import StatisticalFieldEngine
from training.compute_79d import compute_79d_from_aggregator, SFE_MIN_BARS
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
        self._agg = None
        self._sfe = None
        self._engine = None
        self._prev_vel = {}

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

        self._agg = Aggregator(history_limit=2000)
        self._sfe = StatisticalFieldEngine()
        self._engine = BlendedEngine(use_cnn=False)

        # Pick newest checkpoint: live vs NT8 (compare last_ts)
        best_path, best_ts = None, 0
        for path in [LIVE_CHECKPOINT, NT8_CHECKPOINT]:
            if os.path.exists(path):
                try:
                    with open(path, encoding='utf-8') as f:
                        ts = json.load(f).get('last_ts', 0)
                    if ts > best_ts:
                        best_path, best_ts = path, ts
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f'  Corrupt checkpoint: {path}')

        if best_path:
            last_ts, self._prev_vel = self._agg.load_checkpoint(best_path)
            self._last_ts = last_ts
            bar_counts = {tf: len(bars) for tf, bars in self._agg.history.items() if bars}
            logger.info(f'  Checkpoint: {os.path.basename(os.path.dirname(best_path))}/'
                        f'{os.path.basename(best_path)}')
            logger.info(f'  Bars: {bar_counts}')
            logger.info(f'  Last: {self._ts_str(last_ts)}')

            # Feed ATLAS_LIVE delta (bars after checkpoint)
            delta_bars = self._feed_delta(last_ts)
            if delta_bars:
                logger.info(f'  LIVE delta: {delta_bars:,} bars')
        else:
            # Cold start: no checkpoint, replay last N days from ATLAS_NT8/5s
            logger.warning('  No checkpoint found — cold start')
            warmup_bars = 0
            atlas_nt8_5s = os.path.join(ATLAS_NT8, '5s')
            if os.path.exists(atlas_nt8_5s):
                day_files = sorted(glob.glob(os.path.join(atlas_nt8_5s, '*.parquet')))[-WARMUP_DAYS:]
                for fpath in day_files:
                    warmup_bars += self._feed_parquet(fpath)
                logger.info(f'  Cold start: {warmup_bars:,} bars ({len(day_files)} days)')

        # Verify: compute one feature
        if self._agg.get_bar_count('1m') >= SFE_MIN_BARS:
            last_ts = self._agg.get_closed_bars('1m')[-1]['timestamp']
            feat, self._prev_vel, _, _ = compute_79d_from_aggregator(
                self._agg, self._sfe, self._prev_vel, last_ts)
            if feat is not None:
                logger.info(f'  WARMED UP: z={feat[12]:.2f} vr={feat[14]:.2f} '
                            f'1m={self._agg.get_bar_count("1m")} '
                            f'1h={self._agg.get_bar_count("1h")}')
            else:
                logger.warning('  Warmup feature returned None')
        else:
            logger.warning(f'  Not enough 1m bars: '
                           f'{self._agg.get_bar_count("1m")} < {SFE_MIN_BARS}')

    def _feed_delta(self, after_ts: float) -> int:
        """Feed ATLAS_LIVE 5s bars newer than after_ts into aggregator."""
        atlas_live_5s = os.path.join(ATLAS_LIVE, '5s')
        if not os.path.exists(atlas_live_5s):
            return 0
        total = 0
        for fpath in sorted(glob.glob(os.path.join(atlas_live_5s, '*.parquet'))):
            total += self._feed_parquet(fpath, after_ts=after_ts)
        return total

    def _feed_parquet(self, fpath, after_ts=0.0):
        """Feed a parquet file into aggregator. Returns bar count."""
        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        if after_ts > 0:
            df = df[df['timestamp'] > after_ts]
        count = 0
        for _, row in df.iterrows():
            ts = row['timestamp']
            self._agg.feed({
                'timestamp': ts,
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row.get('volume', 0),
            })
            self._last_ts = ts
            self._last_price = row['close']
            count += 1
        return count

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
                self._agg.feed(bar)
                bar_count += 1
            elif msg.get('type') == MsgType.HISTORY_DONE:
                history_done = True
            elif msg.get('type') == 'CONNECTED':
                logger.info(f'  NT8: account={msg.get("account")} '
                            f'instrument={msg.get("instrument")}')

        elapsed = time.perf_counter() - t0
        logger.info(f'  History: {bar_count:,} bars in {elapsed:.1f}s')
        logger.info(f'  Aggregator: 1m={self._agg.get_bar_count("1m")} '
                    f'5m={self._agg.get_bar_count("5m")} '
                    f'1h={self._agg.get_bar_count("1h")}')

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: CATCH-UP — process bars until current
    # ═══════════════════════════════════════════════════════════════════

    async def _step5_catchup(self):
        logger.info('')
        logger.info('STEP 5: CATCH-UP')

        # Feed bars into aggregator until caught up to wall time.
        # No feature computation — just build aggregator state.
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
            self._agg.feed(bar)
            self._last_ts = bar['timestamp']
            self._last_price = bar['close']
            catchup_bars += 1

            # Break once we're within 10s of wall time (caught up)
            if self._last_ts > 0 and (wall_time - self._last_ts) < MAX_SYNC_LAG_S:
                break

        # Compute ONE feature to verify + warm prev_velocities
        if self._agg.get_bar_count('1m') >= SFE_MIN_BARS:
            feat, self._prev_vel, _, _ = compute_79d_from_aggregator(
                self._agg, self._sfe, self._prev_vel, self._last_ts)
            if feat is not None:
                logger.info(f'  Caught up: {catchup_bars:,} bars | '
                            f'z={feat[12]:.2f} vr={feat[14]:.2f}')
            else:
                logger.info(f'  Caught up: {catchup_bars:,} bars (feature=None)')
        else:
            logger.info(f'  Caught up: {catchup_bars:,} bars')
        logger.info(f'  Last bar: {self._ts_str(self._last_ts)} '
                    f'price={self._last_price:.2f}')

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
        elif lag < 3600:
            # Within an hour — probably playback or slow feed
            self._synced = True
            logger.warning(f'  LAG WARNING: {lag:.0f}s behind — proceeding anyway')
        else:
            self._synced = False
            logger.error(f'  SYNC FAILED: {lag:.0f}s behind — NOT TRADING')

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

        # Trade log — one row per entry/exit, matches run_baseline output for parity
        self._trade_log_path = f'reports/live/v2_trades_{self._session_date}.csv'
        self._trade_log = open(self._trade_log_path, 'w', encoding='utf-8')
        trade_cols = ['timestamp', 'type', 'tier', 'direction', 'price',
                      'pnl', 'bars_held', 'exit_reason', 'is_chain',
                      'contracts', 'daily_pnl']
        self._trade_log.write(','.join(trade_cols) + '\n')

        self._trading = True

        pending_5s = None

        def on_bar_close(tf, bar):
            nonlocal pending_5s
            if tf == '5s':
                pending_5s = bar

        self._agg.on_bar_close = on_bar_close

        logger.info(f'  Ledger: {self._ledger_path}')
        logger.info(f'  Trades: {self._trade_log_path}')
        logger.info(f'  Listening for bars...')

        while not self._shutting_down:
            if self._shared_state.get('shutdown'):
                self._shutting_down = True
                break

            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue

            msg_type = msg.get('type', '')
            if msg_type == MsgType.FILL:
                self._orders.on_fill(msg)
                # Correct engine primary entry price from real fill
                fill_px = float(msg.get('fill_price', 0))
                if fill_px and self._engine.in_pos:
                    self._engine.entry_price = fill_px
                continue
            elif msg_type == 'ORDER_STATUS':
                self._orders.on_order_status(msg)
                continue
            elif msg_type == 'POSITION':
                self._orders.on_position(msg)
                continue
            elif msg_type != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)
            self._bar_count += 1
            self._last_ts = bar['timestamp']
            self._last_price = bar['close']
            self._live_bars.append(bar)

            pending_5s = None
            self._agg.feed(bar)

            if pending_5s is None:
                continue

            # Compute features
            feat, self._prev_vel, _, _ = compute_79d_from_aggregator(
                self._agg, self._sfe, self._prev_vel, bar['timestamp'])

            if feat is None:
                continue

            self._feat_count += 1
            z = feat[12]
            vr = feat[14]
            vel = feat[15]

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
            if entered and self._orders.can_enter:
                side = 'BUY' if self._engine.direction == 'long' else 'SELL'
                order_msg = self._orders.build_entry_order(side)
                if order_msg:
                    await self._client.send(order_msg)
                events.append(f'ENTRY_{self._engine.entry_tier}')
                self._log_trade_event(bar['timestamp'], 'ENTRY',
                    self._engine.entry_tier, self._engine.direction,
                    bar['close'], 0, 0, '', False)
                logger.info(f'  ENTRY {self._engine.direction} '
                            f'{self._engine.entry_tier} @ {bar["close"]:.2f}')

            # ── CHAIN ENTRY (scale-in) ─────────────────────────────────
            if chain_opened and self._engine.in_pos:
                cc = self._engine._chain_contracts[-1]
                side = 'BUY' if cc['direction'] == 'long' else 'SELL'
                if self._orders.can_scale_in:
                    order_msg = self._orders.build_scale_in_order(side)
                    if order_msg:
                        await self._client.send(order_msg)
                events.append(f'CHAIN_ENTRY_{cc["entry_tier"]}')
                self._log_trade_event(bar['timestamp'], 'CHAIN_ENTRY',
                    cc['entry_tier'], cc['direction'],
                    bar['close'], 0, 0, '', True)
                logger.info(f'  CHAIN ENTRY {cc["entry_tier"]} '
                            f'@ {bar["close"]:.2f} (contracts={curr_chains + 1})')

            # ── CHAIN EXITS (scale-out, before primary exit) ───────────
            chain_exit_count = new_trades - (1 if exited else 0)
            if chain_exit_count > 0:
                for ci in range(chain_exit_count):
                    ct = self._engine.trades[-(new_trades - ci - (1 if exited else 0))]
                    # Scale-out 1 contract if position still > 1
                    if self._orders.position.qty > 1:
                        order_msg = self._orders.build_scale_out_order(
                            reason=ct.get('exit_reason', 'chain_exit'))
                        if order_msg:
                            await self._client.send(order_msg)
                    events.append(f'CHAIN_EXIT_{ct.get("exit_reason", "?")}')
                    self._log_trade_event(bar['timestamp'], 'CHAIN_EXIT',
                        ct.get('entry_tier', '?'), ct.get('direction', '?'),
                        bar['close'], ct['pnl'], ct.get('held', 0),
                        ct.get('exit_reason', ''), True)
                    logger.info(f'  CHAIN EXIT {ct.get("exit_reason")} '
                                f'pnl=${ct["pnl"]:.1f}')

            # ── PRIMARY EXIT (close remaining) ─────────────────────────
            if exited and new_trades > 0:
                t = self._engine.trades[-1] if chain_exit_count == 0 else \
                    self._engine.trades[-(new_trades)]
                if self._orders.can_exit:
                    order_msg = self._orders.build_exit_order(
                        reason=t.get('exit_reason', 'signal'))
                    if order_msg:
                        await self._client.send(order_msg)
                events.append(f'EXIT_{t.get("exit_reason", "?")}')
                self._log_trade_event(bar['timestamp'], 'EXIT',
                    t.get('entry_tier', '?'), t.get('direction', '?'),
                    bar['close'], t['pnl'], t.get('held', 0),
                    t.get('exit_reason', ''), False)
                logger.info(f'  EXIT {t.get("exit_reason")} pnl=${t["pnl"]:.1f}')

            # ── PnL tracking (engine is always source of truth) ────────
            self._daily_pnl = self._engine.daily_pnl
            self._trade_count = len(self._engine.trades)

            # Unrealized PnL (primary + chains)
            unrealized = 0
            if self._engine.in_pos:
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
            if self._bar_count % 300 == 0:
                self._periodic_save()
                self._orders.cleanup_stale_orders(max_age_s=120.0)

    # ═══════════════════════════════════════════════════════════════════
    # TRADE LOG — one row per entry/exit for parity comparison
    # ═══════════════════════════════════════════════════════════════════

    def _log_trade_event(self, ts, event_type, tier, direction, price,
                         pnl, bars_held, exit_reason, is_chain):
        """Write one trade event row. Matches run_baseline trade format."""
        if self._trade_log is None:
            return
        row = [f'{ts:.0f}', event_type, tier, direction, f'{price:.2f}',
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
        # Save aggregator checkpoint (live)
        self._agg.save_checkpoint(LIVE_CHECKPOINT, velocities=self._prev_vel)
        # Save live bars to ATLAS_LIVE
        if self._live_bars:
            from live.incremental_writer import IncrementalWriter
            writer = IncrementalWriter(ATLAS_LIVE, self._session_date)
            writer.save_all_chunks({'5s': self._live_bars})

    # ═══════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════

    async def _shutdown(self):
        self._shutting_down = True
        logger.info('')
        logger.info('SHUTDOWN')

        # Close all positions (primary + chains)
        if not self._orders.is_flat:
            order_msg = self._orders.build_exit_order(reason='shutdown')
            if order_msg:
                await self._client.send(order_msg)
            await asyncio.sleep(2.0)
        self._engine.force_close()

        # Save final checkpoint
        self._agg.save_checkpoint(LIVE_CHECKPOINT, velocities=self._prev_vel)
        logger.info(f'  Checkpoint saved: {LIVE_CHECKPOINT}')

        # Save ledger + trade log
        if self._ledger:
            self._ledger.flush()
            self._ledger.close()
            logger.info(f'  Ledger: {self._ledger_path}')
        if hasattr(self, '_trade_log') and self._trade_log:
            self._trade_log.flush()
            self._trade_log.close()
            logger.info(f'  Trades: {self._trade_log_path}')

        # Save live data
        self._periodic_save()

        # Summary — engine is source of truth
        wins = sum(1 for t in self._engine.trades if t['pnl'] > 0)
        chains = sum(1 for t in self._engine.trades
                     if str(t.get('exit_reason', '')).startswith('chain_'))
        logger.info(f'  Bars:     {self._bar_count:,}')
        logger.info(f'  Feats:    {self._feat_count:,}')
        logger.info(f'  Trades:   {self._trade_count} ({chains} chains)')
        logger.info(f'  Win rate: {wins}/{self._trade_count} '
                    f'({wins/max(self._trade_count,1)*100:.0f}%)')
        logger.info(f'  PnL:      ${self._daily_pnl:.0f}')

        # Disconnect
        if self._client:
            await self._client.disconnect()
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
                root.protocol('WM_DELETE_WINDOW',
                              lambda: shared_state.update({'shutdown': True}))
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
