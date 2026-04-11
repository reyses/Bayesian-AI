"""
Live Engine V2 — clean production pipeline, 7-step startup.

1. CHECK    → Is ATLAS_NT8 current? If not, dump missing days
2. BUILD    → Build features for any new days
3. WARMUP   → Load last N days into aggregator + SFE
4. SYNC     → Connect NT8, receive bars into ATLAS_LIVE
5. CATCH-UP → Compute features until Python time == NT8 time
6. VERIFY   → Latency < 1s? Parity confirmed?
7. TRADE    → Engine starts making decisions

No legacy CNN. No GUI coupling. Ledger is the monitoring tool.
Physics-only BlendedEngine with rarity-ordered waterfall.

Usage:
    python -m live.engine_v2                     # full production run
    python -m live.engine_v2 --dry-run           # no orders sent to NT8
    python -m live.engine_v2 --skip-check        # skip step 1 (assume current)
    python -m live.engine_v2 --skip-build        # skip step 2
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

TICK = 0.25
TV = 0.50

# Paths
ATLAS_NT8 = 'DATA/ATLAS_NT8'
ATLAS_LIVE = 'DATA/ATLAS_LIVE'
FEATURES_NT8 = 'DATA/FEATURES_NT8_5s'
NT8_CONFIG = 'config/nt8_dataset.json'

# Sync thresholds
MAX_SYNC_LAG_S = 10.0    # max seconds behind NT8 before trading allowed
WARMUP_DAYS = 5           # days of history to load for aggregator context


class LiveEngineV2:
    """Production live engine — 7-step startup, physics-only."""

    def __init__(self, config: LiveConfig, dry_run: bool = False,
                 skip_check: bool = False, skip_build: bool = False):
        self._cfg = config
        self._dry_run = dry_run
        self._skip_check = skip_check
        self._skip_build = skip_build

        self._asset = SYMBOL_MAP.get(config.asset_ticker)
        if self._asset is None:
            raise ValueError(f'Unknown asset: {config.asset_ticker}')

        # Core components (initialized in startup steps)
        self._client = None
        self._agg = None
        self._sfe = None
        self._engine = None
        self._prev_vel = {}

        # State
        self._bar_count = 0
        self._feat_count = 0
        self._synced = False
        self._trading = False
        self._shutting_down = False
        self._position_open = False
        self._order_pending = False   # block new actions while waiting for fill
        self._daily_pnl = 0.0
        self._trade_count = 0
        self._last_ts = 0.0
        self._last_price = 0.0
        self._session_date = time.strftime('%Y_%m_%d')

        # Ledger
        self._ledger = None
        self._ledger_path = None

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
        logger.info(f'  Dry run:    {self._dry_run}')
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

        # Load last N days of 1s bars from ATLAS (Databento) for context
        atlas_1s = 'DATA/ATLAS/1s'
        if os.path.exists(atlas_1s):
            day_files = sorted(glob.glob(os.path.join(atlas_1s, '*.parquet')))[-WARMUP_DAYS:]
            warmup_bars = 0
            for fpath in day_files:
                df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
                for _, row in df.iterrows():
                    self._agg.feed({
                        'timestamp': row['timestamp'],
                        'open': row['open'], 'high': row['high'],
                        'low': row['low'], 'close': row['close'],
                        'volume': row.get('volume', 0),
                    })
                    warmup_bars += 1
            logger.info(f'  ATLAS warmup: {warmup_bars:,} bars')

        # Compute one feature to verify warmup
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
            logger.warning(f'  Not enough 1m bars: {self._agg.get_bar_count("1m")} < {SFE_MIN_BARS}')

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

        pending_5s = None

        def on_bar_close(tf, bar):
            nonlocal pending_5s
            if tf == '5s':
                pending_5s = bar

        self._agg.on_bar_close = on_bar_close
        catchup_feats = 0

        while True:
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # No more bars queued — we've caught up
                break

            if msg.get('type') != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)
            pending_5s = None
            self._agg.feed(bar)
            self._last_ts = bar['timestamp']
            self._last_price = bar['close']

            if pending_5s is None:
                continue

            feat, self._prev_vel, _, _ = compute_79d_from_aggregator(
                self._agg, self._sfe, self._prev_vel, bar['timestamp'])
            if feat is not None:
                catchup_feats += 1

        logger.info(f'  Catch-up: {catchup_feats} features computed')
        logger.info(f'  Last bar: {self._ts_str(self._last_ts)} price={self._last_price:.2f}')

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

        # Open ledger
        os.makedirs('reports/live', exist_ok=True)
        self._ledger_path = f'reports/live/v2_ledger_{self._session_date}.csv'
        self._ledger = open(self._ledger_path, 'w')
        cols = ['timestamp', 'price', 'z', 'vr', 'vel',
                'in_pos', 'direction', 'tier', 'bars_held', 'pnl', 'peak',
                'event']
        self._ledger.write(','.join(cols) + '\n')
        self._trading = True

        pending_5s = None

        def on_bar_close(tf, bar):
            nonlocal pending_5s
            if tf == '5s':
                pending_5s = bar

        self._agg.on_bar_close = on_bar_close

        logger.info(f'  Ledger: {self._ledger_path}')
        logger.info(f'  Listening for bars...')

        while not self._shutting_down:
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue

            if msg.get('type') != MsgType.BAR:
                if msg.get('type') == 'POSITION':
                    qty = int(msg.get('qty', msg.get('quantity', 0)))
                    if qty == 0 and self._position_open:
                        logger.warning('  NT8 says FLAT — syncing local state')
                        self._position_open = False
                    elif qty != 0 and not self._position_open:
                        logger.warning(f'  NT8 says POSITION qty={qty} — orphan detected')
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

            # Feed engine
            prev_in_pos = self._engine.in_pos
            prev_trades = len(self._engine.trades)

            state = {
                'features_79d': feat,
                'price': bar['close'],
                'timestamp': bar['timestamp'],
            }
            self._engine.on_state(state)

            # Detect events
            entered = self._engine.in_pos and not prev_in_pos
            exited = not self._engine.in_pos and prev_in_pos
            new_trade = len(self._engine.trades) > prev_trades

            event = ''
            if entered and not self._order_pending:
                event = f'ENTRY_{self._engine.entry_tier}'
                if not self._dry_run:
                    self._order_pending = True
                    await self._send_entry()
                    self._order_pending = False
                else:
                    logger.info(f'  [DRY] ENTRY {self._engine.direction} '
                                f'{self._engine.entry_tier} @ {bar["close"]:.2f}')

            if exited and new_trade and not self._order_pending:
                t = self._engine.trades[-1]
                event = f'EXIT_{t.get("exit_reason", "?")}'
                self._daily_pnl += t['pnl']
                self._trade_count += 1
                if not self._dry_run:
                    self._order_pending = True
                    await self._send_exit()
                    self._order_pending = False
                logger.info(f'  {"[DRY] " if self._dry_run else ""}'
                            f'EXIT {t.get("exit_reason")} pnl=${t["pnl"]:.1f} '
                            f'daily=${self._daily_pnl:.0f} #{self._trade_count}')

            # Ledger
            pnl = 0
            if self._engine.in_pos:
                if self._engine.direction == 'long':
                    pnl = (bar['close'] - self._engine.entry_price) / TICK * TV
                else:
                    pnl = (self._engine.entry_price - bar['close']) / TICK * TV

            self._write_ledger(bar['timestamp'], bar['close'], z, vr, vel, pnl, event)

            # Status line
            if self._feat_count % 12 == 0:  # every minute
                pos = self._engine.direction or 'FLAT'
                tier = self._engine.entry_tier or '-'
                print(f'\r  {self._ts_str(bar["timestamp"])} | {bar["close"]:>10.2f} | '
                      f'{pos:>5} {tier:>15} | z={z:>+5.1f} vr={vr:.2f} | '
                      f'tr={self._trade_count} day=${self._daily_pnl:>+.0f}    ',
                      end='', flush=True)

            # Periodic save
            if self._bar_count % 300 == 0:
                self._periodic_save()

    # ═══════════════════════════════════════════════════════════════════
    # TRADE EXECUTION — wait for NT8 confirmation
    # ═══════════════════════════════════════════════════════════════════

    async def _send_entry(self):
        """Send entry order and WAIT for fill confirmation from NT8."""
        side = 'BUY' if self._engine.direction == 'long' else 'SELL'
        order_id = f'v2_{self._trade_count}_{int(time.time())}'
        msg = place_order(
            order_id=order_id,
            instrument=self._cfg.instrument,
            account=self._cfg.account,
            side=side, qty=1)
        await self._client.send(msg)
        logger.info(f'  ORDER SENT: {side} {self._engine.entry_tier} (waiting for fill...)')

        # Wait for FILL confirmation
        fill = await self._wait_for_fill(order_id, timeout=10.0)
        if fill:
            fill_price = float(fill.get('fill_price', self._last_price))
            self._engine.entry_price = fill_price
            self._position_open = True
            logger.info(f'  FILLED: {side} @ {fill_price:.2f} '
                        f'(engine had {self._last_price:.2f}, slippage={abs(fill_price - self._last_price):.2f})')
        else:
            logger.warning(f'  NO FILL after 10s — position state uncertain')
            self._position_open = True  # assume filled, will reconcile on POSITION msg

    async def _send_exit(self):
        """Send close order and WAIT for confirmation."""
        msg = close_position(
            instrument=self._cfg.instrument,
            account=self._cfg.account)
        await self._client.send(msg)
        logger.info(f'  CLOSE SENT (waiting for confirmation...)')

        # Wait for POSITION with qty=0 or FILL confirmation
        confirmed = await self._wait_for_flat(timeout=10.0)
        if confirmed:
            self._position_open = False
            logger.info(f'  CLOSE CONFIRMED')
        else:
            logger.warning(f'  CLOSE NOT CONFIRMED after 10s — will reconcile')
            self._position_open = False  # assume closed

    async def _wait_for_fill(self, order_id, timeout=10.0):
        """Wait for FILL message matching order_id. Returns fill msg or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.get('type') == MsgType.FILL:
                return msg
            elif msg.get('type') == MsgType.BAR:
                # Keep processing bars while waiting
                bar = self._extract_bar(msg)
                self._agg.feed(bar)
                self._bar_count += 1
                self._last_ts = bar['timestamp']
                self._last_price = bar['close']
                self._live_bars.append(bar)
            elif msg.get('type') == 'POSITION':
                qty = int(msg.get('qty', 0))
                if qty != 0:
                    return msg  # position opened = fill happened

        return None

    async def _wait_for_flat(self, timeout=10.0):
        """Wait for POSITION with qty=0 confirming flat. Returns True/False."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.get('type') == 'POSITION':
                qty = int(msg.get('qty', msg.get('quantity', 0)))
                if qty == 0:
                    return True
            elif msg.get('type') == MsgType.FILL:
                # Exit fill received
                return True
            elif msg.get('type') == MsgType.BAR:
                bar = self._extract_bar(msg)
                self._agg.feed(bar)
                self._bar_count += 1
                self._last_ts = bar['timestamp']
                self._last_price = bar['close']
                self._live_bars.append(bar)

        return False

    # ═══════════════════════════════════════════════════════════════════
    # LEDGER + SAVE
    # ═══════════════════════════════════════════════════════════════════

    def _write_ledger(self, ts, price, z, vr, vel, pnl, event):
        if self._ledger is None:
            return
        row = [f'{ts:.0f}', f'{price:.2f}', f'{z:.4f}', f'{vr:.4f}', f'{vel:.2f}',
               '1' if self._engine.in_pos else '0',
               self._engine.direction or '',
               self._engine.entry_tier or '',
               str(self._engine.bars_held),
               f'{pnl:.1f}',
               f'{self._engine.peak_pnl:.1f}',
               event]
        self._ledger.write(','.join(row) + '\n')

    def _periodic_save(self):
        if self._ledger:
            self._ledger.flush()
        # Save live bars to ATLAS_LIVE
        if self._live_bars:
            from live.incremental_writer import IncrementalWriter
            writer = IncrementalWriter(ATLAS_LIVE, self._session_date)
            writer.save_all_chunks({'1s': self._live_bars})

    # ═══════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════

    async def _shutdown(self):
        self._shutting_down = True
        logger.info('')
        logger.info('SHUTDOWN')

        # Close position if open
        if self._position_open and not self._dry_run:
            await self._send_exit()
            await asyncio.sleep(2.0)
        self._engine.force_close()

        # Save ledger
        if self._ledger:
            self._ledger.flush()
            self._ledger.close()
            logger.info(f'  Ledger: {self._ledger_path}')

        # Save live data
        self._periodic_save()

        # Summary
        logger.info(f'  Bars:   {self._bar_count:,}')
        logger.info(f'  Feats:  {self._feat_count:,}')
        logger.info(f'  Trades: {self._trade_count}')
        logger.info(f'  PnL:    ${self._daily_pnl:.0f}')

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
    parser.add_argument('--dry-run', action='store_true', help='No orders to NT8')
    parser.add_argument('--skip-check', action='store_true')
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()

    config = LiveConfig()
    engine = LiveEngineV2(config, dry_run=args.dry_run,
                          skip_check=args.skip_check,
                          skip_build=args.skip_build)
    asyncio.run(engine.run())


if __name__ == '__main__':
    main()
