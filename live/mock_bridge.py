"""
Mock NT8 Bridge -- replays ATLAS bars as if they came from NT8.

Drop-in replacement for NT8Client. The live engine sees the same
message types (BAR, FILL, HISTORY_DONE, ORDER_ACK, HEARTBEAT) and
can't tell the difference. Orders get instant-filled at bar close.

Usage:
    python -m live.engine_v2 --mock                        # replay today from ATLAS_NT8
    python -m live.engine_v2 --mock --mock-day 2026_04_16  # replay specific day
    python -m live.engine_v2 --mock --mock-speed 0         # instant (no delay)

The mock bridge replays bars from ATLAS_NT8/5s for the specified day.
Bars before the warmup cutoff go as HISTORY, bars after go as LIVE
(with optional inter-bar delay to simulate real-time).
"""
import asyncio
import logging
import os
import time
import glob
import numpy as np
import pandas as pd

logger = logging.getLogger('mock_bridge')


class MockBridge:
    """Replays ATLAS bars as mock NT8 messages."""

    def __init__(self, config, atlas_root='DATA/ATLAS_NT8',
                 day=None, speed=1.0, warmup_cutoff_ts=0):
        """
        Args:
            config: LiveConfig (for instrument/account names)
            atlas_root: path to ATLAS with 5s/ subdirectory
            day: day name to replay (e.g. '2026_04_16'). None = latest.
            speed: playback speed multiplier. 0 = instant, 1 = real-time.
            warmup_cutoff_ts: bars before this ts go as history dump,
                              bars after go as live (with delay).
        """
        self._cfg = config
        self._atlas_root = atlas_root
        self._speed = speed
        self._warmup_cutoff_ts = warmup_cutoff_ts
        self._stop = False
        # _day_to_replay set in _load_bars (resolves 'latest' to actual day name)

        # Same interface as NT8Client, but SMALL queue for backpressure.
        # Small queue forces the engine to consume each bar before the mock
        # puts the next. Without this, the mock floods 50K bars ahead and
        # order ACK/FILL responses arrive far behind their orders.
        self.inbound: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._resume_ts: float = 0.0

        # Priority buffer for order responses (ACK, FILL) — drained
        # before each bar so the engine processes them immediately.
        self._priority_msgs = []

        # Handshake: Step 7 signals ready before we send live bars
        self._live_ready = asyncio.Event()

        # Position tracking for fill responses
        self._position_qty = 0
        self._position_side = ''
        self._last_price = 0.0

        # Load bars
        self._bars, self._day_to_replay = self._load_bars(day)
        logger.info(f'MockBridge: {len(self._bars)} bars loaded for {self._day_to_replay}')

    def _load_bars(self, day):
        """Load 5s bars for replay. Returns (df, day_name)."""
        src = os.path.join(self._atlas_root, '5s')
        if day:
            path = os.path.join(src, f'{day}.parquet')
            if os.path.exists(path):
                return pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True), day
        # Default: latest day
        files = sorted(glob.glob(os.path.join(src, '*.parquet')))
        if files:
            last_f = files[-1]
            day_name = os.path.basename(last_f).replace('.parquet', '')
            return pd.read_parquet(last_f).sort_values('timestamp').reset_index(drop=True), day_name
        return pd.DataFrame(), None

    def set_resume_timestamp(self, ts: float):
        self._resume_ts = ts

    async def connect(self) -> bool:
        """Simulate connection -- push CONNECTED message."""
        await self.inbound.put({
            'type': 'CONNECTED',
            'account': self._cfg.account,
            'instrument': self._cfg.instrument,
        })
        # Start the bar replay in background
        asyncio.create_task(self._replay_bars())
        return True

    async def disconnect(self):
        self._stop = True

    def signal_live_ready(self):
        """Called by engine when Step 7 starts listening."""
        self._live_ready.set()

    async def send(self, msg: dict):
        """Handle outbound messages from the engine (orders).

        Fills are queued in a PRIORITY buffer that gets drained before
        the next bar is pumped. This prevents the watchdog from timing
        out orders while bars are flooding the main queue.
        """
        msg_type = msg.get('type', '')

        if msg_type == 'PLACE_ORDER':
            order_id = msg.get('order_id', '')
            side = msg.get('side', '')
            qty = int(msg.get('qty', 1))

            self._priority_msgs.append({
                'type': 'ORDER_ACK',
                'order_id': order_id,
            })

            fill_price = self._last_price
            position_effect = msg.get('position_effect', 'OPEN')

            if position_effect == 'OPEN':
                if self._position_qty == 0:
                    self._position_side = 'LONG' if side == 'BUY' else 'SHORT'
                    self._position_qty = qty
                else:
                    self._position_qty += qty
            elif position_effect == 'REDUCE':
                self._position_qty = max(0, self._position_qty - qty)
                if self._position_qty == 0:
                    self._position_side = ''

            self._priority_msgs.append({
                'type': 'FILL',
                'order_id': order_id,
                'side': side,
                'fill_price': fill_price,
                'fill_time': time.time(),
                'qty': qty,
            })

        elif msg_type == 'CLOSE_POSITION':
            if self._position_qty > 0:
                close_side = 'SELL' if self._position_side == 'LONG' else 'BUY'
                close_qty = self._position_qty
                self._priority_msgs.append({
                    'type': 'ORDER_ACK',
                    'order_id': 'BAY_CLOSE',
                })
                self._priority_msgs.append({
                    'type': 'FILL',
                    'order_id': 'BAY_CLOSE',
                    'side': close_side,
                    'fill_price': self._last_price,
                    'fill_time': time.time(),
                    'qty': close_qty,
                })
                self._position_qty = 0
                self._position_side = ''

        elif msg_type == 'REQUEST_POSITION':
            self._priority_msgs.append({
                'type': 'POSITION',
                'qty': self._position_qty,
                'side': self._position_side,
                'avg_price': self._last_price,
            })

        elif msg_type == 'HEARTBEAT':
            self._priority_msgs.append({
                'type': 'HEARTBEAT',
                'position_qty': self._position_qty,
                'position_side': self._position_side or 'FLAT',
                'position_avg_price': self._last_price,
            })

    async def _replay_bars(self):
        """Feed bars to inbound queue as BAR messages.

        Split: first 10 bars as HISTORY (for Steps 4/5 to consume),
        then HISTORY_DONE, then remaining bars as LIVE (Step 7 processes
        them through the engine with full evaluate + orders).
        """
        if len(self._bars) == 0:
            await self.inbound.put({'type': 'HISTORY_DONE'})
            return

        bars = self._bars
        cutoff = self._warmup_cutoff_ts or self._resume_ts

        # Filter to non-skipped bars
        replay_bars = []
        for _, row in bars.iterrows():
            if cutoff > 0 and row['timestamp'] <= cutoff:
                continue
            if row['timestamp'] <= self._resume_ts:
                continue
            replay_bars.append(row)

        if not replay_bars:
            await self.inbound.put({'type': 'HISTORY_DONE'})
            logger.info('MockBridge: no bars after cutoff')
            return

        # Phase 1: Send first 10 bars as history (enough for Steps 4-6 to sync)
        HISTORY_COUNT = 10
        history_bars = replay_bars[:HISTORY_COUNT]
        live_bars = replay_bars[HISTORY_COUNT:]

        for row in history_bars:
            if self._stop:
                return
            self._last_price = float(row['close'])
            await self.inbound.put(self._row_to_msg(row))

        await self.inbound.put({'type': 'HISTORY_DONE'})
        logger.info(f'MockBridge: history done ({len(history_bars)} bars), '
                    f'{len(live_bars)} bars queued for live replay')

        # Wait for Step 7 to signal ready before sending live bars
        logger.info('MockBridge: waiting for LIVE_READY signal...')
        await self._live_ready.wait()
        logger.info('MockBridge: LIVE_READY received, sending bars')

        # Phase 2: Feed remaining bars as "live" — engine processes each
        # through evaluate() + orders. No delay (compressed time).
        from tqdm import tqdm
        pbar = tqdm(live_bars, desc='Mock replay', unit='bar')
        for row in pbar:
            if self._stop:
                break
            # Drain priority buffer first — ACK/FILL must reach engine
            # before the next bar so watchdog doesn't time out orders
            while self._priority_msgs:
                await self.inbound.put(self._priority_msgs.pop(0))
                await asyncio.sleep(0)

            self._last_price = float(row['close'])
            await self.inbound.put(self._row_to_msg(row))
            # Yield so the engine can process this bar (and any order it sends)
            await asyncio.sleep(0)
        pbar.close()

        # Final drain
        while self._priority_msgs:
            await self.inbound.put(self._priority_msgs.pop(0))

        logger.info(f'MockBridge: replay complete ({len(replay_bars)} total bars)')

        # Signal end of session — engine should shut down
        # Small delay to let the last bars process through the engine
        await asyncio.sleep(0.5)
        await self.inbound.put({'type': 'MOCK_DONE'})

    @staticmethod
    def _row_to_msg(row):
        return {
            'type': 'BAR',
            'timestamp': float(row['timestamp']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
        }
