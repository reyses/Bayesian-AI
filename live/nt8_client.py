"""
NT8Client — asyncio TCP client that talks to the BayesianBridge indicator
running inside NinjaTrader 8.

Handles:
  - Connection / reconnection with exponential backoff
  - Heartbeat monitoring
  - Inbound message dispatching via asyncio.Queue
  - Outbound message encoding + send
"""

import asyncio
import logging
import time
from typing import Optional

from live.config import LiveConfig
from live.protocol import (
    encode, MessageReader, subscribe, heartbeat as hb_msg, validate,
)

logger = logging.getLogger(__name__)


class NT8Client:
    """Bidirectional TCP client for the NinjaTrader 8 bridge."""

    def __init__(self, config: LiveConfig):
        self._cfg = config
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._stop = False

        # Inbound messages land here for the LiveEngine to consume
        self.inbound: asyncio.Queue = asyncio.Queue(maxsize=500)

        self._read_task: Optional[asyncio.Task] = None
        self._hb_task: Optional[asyncio.Task] = None
        self._last_hb_recv: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to NT8 bridge with retries.  Returns True on success."""
        attempt = 0
        delay = self._cfg.reconnect_delay_s

        while not self._stop and attempt < self._cfg.max_reconnect_attempts:
            attempt += 1
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self._cfg.nt8_host, self._cfg.nt8_port)
                self._connected = True
                self._last_hb_recv = time.time()
                logger.info(f"Connected to NT8 bridge at "
                            f"{self._cfg.nt8_host}:{self._cfg.nt8_port}")

                # Send subscription request
                sub = subscribe(self._cfg.instrument,
                                self._cfg.base_resolution_s,
                                self._cfg.account)
                await self.send(sub)

                # Start background tasks
                self._read_task = asyncio.create_task(self._read_loop())
                self._hb_task = asyncio.create_task(self._heartbeat_loop())
                return True

            except (ConnectionRefusedError, OSError) as e:
                logger.warning(f"Connect attempt {attempt}/{self._cfg.max_reconnect_attempts}"
                               f" failed: {e}  (retry in {delay:.1f}s)")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 30.0)  # exponential backoff, cap 30s

        logger.error("Max reconnect attempts reached — giving up.")
        return False

    async def disconnect(self):
        """Graceful shutdown."""
        self._stop = True
        self._connected = False
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        if self._hb_task and not self._hb_task.done():
            self._hb_task.cancel()
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        logger.info("Disconnected from NT8 bridge.")

    async def send(self, msg: dict):
        """Encode and send a message to NT8."""
        if not self._connected or self._writer is None:
            logger.warning(f"Cannot send — not connected: {msg.get('type')}")
            return
        try:
            self._writer.write(encode(msg))
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            logger.error(f"Send failed: {e}")
            await self._handle_disconnect()

    def stop(self):
        """Signal the client to shut down."""
        self._stop = True

    # ── Internal ──────────────────────────────────────────────────────

    async def _read_loop(self):
        """Continuously read messages from NT8 and enqueue them."""
        reader = MessageReader(self._reader)
        try:
            async for msg in reader:
                if self._stop:
                    break
                if msg.get('type') == 'HEARTBEAT':
                    self._last_hb_recv = time.time()
                    continue
                if validate(msg):
                    await self.inbound.put(msg)
                else:
                    logger.warning(f"Invalid message dropped: {msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Read loop error: {e}")

        if not self._stop:
            await self._handle_disconnect()

    async def _heartbeat_loop(self):
        """Send heartbeats and monitor for stale connection."""
        try:
            while not self._stop and self._connected:
                await asyncio.sleep(self._cfg.heartbeat_interval_s)
                await self.send(hb_msg())

                elapsed = time.time() - self._last_hb_recv
                if elapsed > self._cfg.heartbeat_interval_s * 3:
                    logger.warning(f"No heartbeat from NT8 for {elapsed:.0f}s")
        except asyncio.CancelledError:
            pass

    async def _handle_disconnect(self):
        """Clean up and attempt reconnection."""
        if not self._connected:
            return
        self._connected = False
        logger.warning("Connection lost — attempting reconnect...")
        if self._writer:
            self._writer.close()
        if not self._stop:
            await self.connect()
