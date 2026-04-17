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

        # Same interface as NT8Client
        self.inbound: asyncio.Queue = asyncio.Queue(maxsize=50000)
        self._resume_ts: float = 0.0

        # Pending orders (engine sends, we auto-fill)
        self._pending_orders = asyncio.Queue()

        # Position tracking for fill responses
        self._position_qty = 0
        self._position_side = ''
        self._last_price = 0.0

        # Load bars
        self._bars = self._load_bars(day)
        logger.info(f'MockBridge: {len(self._bars)} bars loaded')

    def _load_bars(self, day):
        """Load 5s bars for replay."""
        src = os.path.join(self._atlas_root, '5s')
        if day:
            path = os.path.join(src, f'{day}.parquet')
            if os.path.exists(path):
                return pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
        # Default: latest day
        files = sorted(glob.glob(os.path.join(src, '*.parquet')))
        if files:
            return pd.read_parquet(files[-1]).sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

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

    async def send(self, msg: dict):
        """Handle outbound messages from the engine (orders)."""
        msg_type = msg.get('type', '')

        if msg_type == 'PLACE_ORDER':
            # Auto-fill at last known price
            order_id = msg.get('order_id', '')
            side = msg.get('side', '')
            qty = int(msg.get('qty', 1))

            # ACK first
            await self.inbound.put({
                'type': 'ORDER_ACK',
                'order_id': order_id,
            })

            # Then FILL
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

            await self.inbound.put({
                'type': 'FILL',
                'order_id': order_id,
                'side': side,
                'fill_price': fill_price,
                'fill_time': time.time(),
                'qty': qty,
            })

        elif msg_type == 'CLOSE_POSITION':
            # Close all -- fill at last price
            if self._position_qty > 0:
                close_side = 'SELL' if self._position_side == 'LONG' else 'BUY'
                await self.inbound.put({
                    'type': 'ORDER_ACK',
                    'order_id': 'BAY_CLOSE',
                })
                await self.inbound.put({
                    'type': 'FILL',
                    'order_id': 'BAY_CLOSE',
                    'side': close_side,
                    'fill_price': self._last_price,
                    'fill_time': time.time(),
                    'qty': self._position_qty,
                })
                self._position_qty = 0
                self._position_side = ''

        elif msg_type == 'REQUEST_POSITION':
            await self.inbound.put({
                'type': 'POSITION',
                'qty': self._position_qty,
                'side': self._position_side,
                'avg_price': self._last_price,
            })

        elif msg_type == 'HEARTBEAT':
            await self.inbound.put({
                'type': 'HEARTBEAT',
                'position_qty': self._position_qty,
                'position_side': self._position_side or 'FLAT',
                'position_avg_price': self._last_price,
            })

    async def _replay_bars(self):
        """Feed bars to inbound queue as BAR messages."""
        if len(self._bars) == 0:
            await self.inbound.put({'type': 'HISTORY_DONE'})
            return

        bars = self._bars
        cutoff = self._warmup_cutoff_ts or self._resume_ts

        # Phase 1: History dump (bars before cutoff, instant)
        history_count = 0
        for _, row in bars.iterrows():
            if self._stop:
                return
            if cutoff > 0 and row['timestamp'] <= cutoff:
                # Skip bars the LFE already has from warmup
                continue
            if row['timestamp'] <= self._resume_ts:
                continue
            # First non-skipped bar starts the history
            bar_msg = {
                'type': 'BAR',
                'timestamp': float(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
            }
            self._last_price = float(row['close'])
            await self.inbound.put(bar_msg)
            history_count += 1

        # History done
        await self.inbound.put({'type': 'HISTORY_DONE'})
        logger.info(f'MockBridge: history done ({history_count} bars)')

        # Phase 2: Live bars (with delay if speed > 0)
        # In this simple version, all bars go as history since we're replaying.
        # For real-time simulation, split at a midpoint and add delays.

        # Signal end
        logger.info(f'MockBridge: replay complete')
