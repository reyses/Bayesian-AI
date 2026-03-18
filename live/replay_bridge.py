"""
ReplayBridge  -- TCP server that feeds ATLAS parquet data through the same
protocol as NT8_BayesianBridge.cs.

Eliminates OOS/live parity by construction: LiveEngine sees the exact same
messages whether connected to NT8 or to this replay server.

Usage:
    # From launcher (preferred):
    python -m live.launcher --replay DATA/ATLAS_OOS --speed 50

    # Standalone:
    python -m live.replay_bridge DATA/ATLAS_OOS --speed 50

Protocol: Length-prefixed JSON over TCP (port 5199)  -- identical to NT8 bridge.
"""

import asyncio
import glob
import json
import logging
import os
import struct
import time
from typing import Dict, Optional

import pandas as pd

from live.protocol import HEADER_FMT

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PORT = 5199
DEFAULT_SPEED_MS = 50          # ms per anchor bar (50ms -> 57 days in ~35 min)
BRIDGE_VERSION = "REPLAY-1.0"

# TFs available in ATLAS, ordered by period (seconds)
ATLAS_TFS = {
    '1s': 1, '5s': 5, '15s': 15, '30s': 30,
    '1m': 60, '3m': 180, '5m': 300, '15m': 900,
    '30m': 1800, '1h': 3600, '4h': 14400,
}


def _encode(msg: dict) -> bytes:
    """Encode dict to length-prefixed JSON bytes (NT8 wire format)."""
    payload = json.dumps(msg, separators=(',', ':')).encode('utf-8')
    return struct.pack(HEADER_FMT, len(payload)) + payload


class ReplayBridge:
    """TCP server that replays ATLAS parquet data as NT8 BAR messages."""

    def __init__(self, atlas_dir: str, port: int = DEFAULT_PORT,
                 speed_ms: int = DEFAULT_SPEED_MS,
                 instrument: str = 'MNQ 03-26',
                 account: str = 'REPLAY',
                 anchor_tf: str = '15s',
                 warmup_bars: int = 2400):
        self._atlas_dir = atlas_dir
        self._port = port
        self._speed_ms = speed_ms
        self._instrument = instrument
        self._account = account
        self._anchor_tf = anchor_tf
        self._anchor_period = ATLAS_TFS[anchor_tf]
        self._warmup_bars = warmup_bars

        # Loaded data
        self._data: Dict[str, pd.DataFrame] = {}  # tf_label -> DataFrame
        self._anchor_df: Optional[pd.DataFrame] = None

        # Connection state
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._connected = False
        self._stop = False

        # Simulated position for fill responses
        self._sim_position = ''  # '', 'LONG', 'SHORT'
        self._sim_entry_price = 0.0
        self._sim_qty = 0
        self._current_price = 0.0

        # Stats
        self._bars_sent = 0
        self._fills_sent = 0
        self._start_time = 0.0

    # ── Data Loading ──────────────────────────────────────────────────────

    def _load_atlas(self):
        """Load all available TF parquet files from the ATLAS directory."""
        logger.info(f"Loading ATLAS data from {self._atlas_dir}")
        total_bars = 0

        for tf_label, period_s in ATLAS_TFS.items():
            tf_dir = os.path.join(self._atlas_dir, tf_label)
            if not os.path.isdir(tf_dir):
                continue
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if not files:
                continue
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values('timestamp').drop_duplicates(
                subset='timestamp', keep='last').reset_index(drop=True)
            self._data[tf_label] = df
            total_bars += len(df)
            logger.info(f"  {tf_label}: {len(df):,} bars from {len(files)} files")

        if self._anchor_tf not in self._data:
            raise FileNotFoundError(
                f"No {self._anchor_tf} data in {self._atlas_dir}")

        self._anchor_df = self._data[self._anchor_tf]
        logger.info(f"Total: {total_bars:,} bars across {len(self._data)} TFs")
        logger.info(f"Anchor: {len(self._anchor_df):,} {self._anchor_tf} bars")

        # Time range
        ts_min = self._anchor_df['timestamp'].iloc[0]
        ts_max = self._anchor_df['timestamp'].iloc[-1]
        from datetime import datetime, timezone
        d0 = datetime.fromtimestamp(ts_min, tz=timezone.utc).strftime('%Y-%m-%d')
        d1 = datetime.fromtimestamp(ts_max, tz=timezone.utc).strftime('%Y-%m-%d')
        logger.info(f"Date range: {d0} to {d1}")

    # ── TCP Server ────────────────────────────────────────────────────────

    async def start(self):
        """Start TCP server and wait for LiveEngine to connect."""
        self._load_atlas()

        server = await asyncio.start_server(
            self._on_client_connect, '127.0.0.1', self._port)
        addr = server.sockets[0].getsockname()
        logger.info(f"ReplayBridge listening on {addr[0]}:{addr[1]}")
        logger.info(f"Speed: {self._speed_ms}ms/bar  -- start live.launcher to connect")

        async with server:
            await server.serve_forever()

    async def run_with_engine(self):
        """Start server, wait for one client, replay, then shut down."""
        self._load_atlas()

        server = await asyncio.start_server(
            self._on_client_connect, '127.0.0.1', self._port)
        logger.info(f"ReplayBridge ready on port {self._port}")

        # Wait for client connection + replay completion
        self._stop = False
        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                pass

    async def _on_client_connect(self, reader: asyncio.StreamReader,
                                  writer: asyncio.StreamWriter):
        """Handle incoming connection from LiveEngine."""
        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected from {addr}")
        self._reader = reader
        self._writer = writer
        self._connected = True

        # Start read loop for inbound commands (SUBSCRIBE, PLACE_ORDER, etc.)
        read_task = asyncio.create_task(self._read_loop())

        try:
            # Wait for SUBSCRIBE before starting replay
            await self._wait_for_subscribe()
            # Send CONNECTED
            await self._send_connected()
            # Send position snapshot (flat)
            await self._send_position(qty=0, side='', avg_price=0)
            # Wait for REQUEST_HISTORY
            await self._wait_for_history_request()
            # Send history + live bars
            await self._replay()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            logger.warning("Client disconnected during replay")
        finally:
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
            writer.close()
            self._connected = False
            self._stop = True
            # Stop the server
            raise asyncio.CancelledError()

    # ── Inbound Message Handling ──────────────────────────────────────────

    async def _read_loop(self):
        """Read commands from LiveEngine (PLACE_ORDER, CLOSE_POSITION, etc.)."""
        try:
            while self._connected and not self._stop:
                try:
                    header = await asyncio.wait_for(
                        self._reader.readexactly(4), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.IncompleteReadError:
                    break

                length = struct.unpack(HEADER_FMT, header)[0]
                payload = await self._reader.readexactly(length)
                msg = json.loads(payload.decode('utf-8'))

                mtype = msg.get('type', '')

                if mtype == 'PLACE_ORDER':
                    await self._handle_place_order(msg)
                elif mtype == 'CLOSE_POSITION':
                    await self._handle_close_position(msg)
                elif mtype == 'CANCEL_ORDER':
                    pass  # no-op in replay
                elif mtype == 'HEARTBEAT':
                    await self._send({'type': 'HEARTBEAT',
                                      'server_time': time.time()})
                elif mtype == 'SUBSCRIBE':
                    self._subscribe_event.set()
                elif mtype in ('REQUEST_HISTORY', 'RESUME_FROM'):
                    self._history_event.set()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Read loop error: {e}")

    async def _wait_for_subscribe(self):
        """Block until SUBSCRIBE received."""
        self._subscribe_event = asyncio.Event()
        await asyncio.wait_for(self._subscribe_event.wait(), timeout=30.0)
        logger.info("SUBSCRIBE received")

    async def _wait_for_history_request(self):
        """Block until REQUEST_HISTORY or RESUME_FROM received."""
        self._history_event = asyncio.Event()
        await asyncio.wait_for(self._history_event.wait(), timeout=30.0)
        logger.info("History request received")

    async def _handle_place_order(self, msg: dict):
        """Simulate instant fill at current price."""
        order_id = msg.get('order_id', 'unknown')
        side = msg.get('side', 'BUY')
        qty = int(msg.get('qty', 1))
        fill_price = self._current_price

        # Update simulated position
        if side == 'BUY':
            self._sim_position = 'LONG'
            self._sim_entry_price = fill_price
            self._sim_qty = qty
        else:
            self._sim_position = 'SHORT'
            self._sim_entry_price = fill_price
            self._sim_qty = qty

        # Send ORDER_STATUS (Working -> Filled)
        await self._send({
            'type': 'ORDER_STATUS',
            'order_id': order_id,
            'status': 'FILLED',
        })

        # Send FILL
        await self._send({
            'type': 'FILL',
            'order_id': order_id,
            'side': side,
            'qty': qty,
            'fill_price': fill_price,
            'fill_time': time.time(),
            'commission': 0.0,
        })
        self._fills_sent += 1

        # Send POSITION update
        await self._send_position(
            qty=qty, side=self._sim_position,
            avg_price=fill_price)

        logger.debug(f"FILL: {side} {qty} @ {fill_price:.2f} (order={order_id})")

    async def _handle_close_position(self, msg: dict):
        """Simulate closing at current price."""
        if not self._sim_position:
            return

        # Determine exit side (opposite of position)
        exit_side = 'SELL' if self._sim_position == 'LONG' else 'BUY'
        fill_price = self._current_price
        order_id = f"close_{int(time.time()*1000)}"

        await self._send({
            'type': 'ORDER_STATUS',
            'order_id': order_id,
            'status': 'FILLED',
        })

        await self._send({
            'type': 'FILL',
            'order_id': order_id,
            'side': exit_side,
            'qty': self._sim_qty,
            'fill_price': fill_price,
            'fill_time': time.time(),
            'commission': 0.0,
        })
        self._fills_sent += 1

        # Reset position
        self._sim_position = ''
        self._sim_entry_price = 0.0
        self._sim_qty = 0

        await self._send_position(qty=0, side='', avg_price=0)
        logger.debug(f"CLOSE: {exit_side} @ {fill_price:.2f}")

    # ── Outbound Messages ─────────────────────────────────────────────────

    async def _send(self, msg: dict):
        """Send a message to the connected client."""
        if not self._connected or self._writer is None:
            return
        try:
            self._writer.write(_encode(msg))
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            self._connected = False

    async def _send_connected(self):
        """Send CONNECTED handshake (mirrors NT8 bridge)."""
        await self._send({
            'type': 'CONNECTED',
            'account': self._account,
            'instrument': self._instrument,
            'primary_period_s': self._anchor_period,
            'version': BRIDGE_VERSION,
        })
        logger.info("Sent CONNECTED")

    async def _send_position(self, qty: int, side: str, avg_price: float):
        """Send POSITION update."""
        await self._send({
            'type': 'POSITION',
            'instrument': self._instrument,
            'qty': qty,
            'side': side,
            'avg_price': avg_price,
        })

    async def _send_bar(self, tf_label: str, period_s: int, row: dict):
        """Send a single BAR message."""
        await self._send({
            'type': 'BAR',
            'instrument': self._instrument,
            'tf': tf_label,
            'bar_period_s': period_s,
            'timestamp': float(row['timestamp']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
        })

    async def _send_history_done(self, count: int):
        """Send HISTORY_DONE marker."""
        await self._send({
            'type': 'HISTORY_DONE',
            'bar_count': count,
        })

    # ── Replay Engine ─────────────────────────────────────────────────────

    async def _replay(self):
        """Feed all bars through the protocol  -- history warmup + live stream."""
        anchor_df = self._anchor_df
        n_total = len(anchor_df)

        # Split: first warmup_bars go as history dump, rest as live stream
        n_warmup = min(self._warmup_bars, n_total)

        logger.info(f"Replay plan: {n_warmup:,} warmup bars (history) "
                    f"+ {n_total - n_warmup:,} live bars")
        logger.info(f"Speed: {self._speed_ms}ms/bar "
                    f"(ETA: {(n_total - n_warmup) * self._speed_ms / 60000:.1f} min)")

        self._start_time = time.time()

        # ── Phase 1: History dump (fast, no delay) ────────────────────────
        # Send all available TF bars up to the warmup timestamp as history
        warmup_ts = float(anchor_df.iloc[n_warmup - 1]['timestamp'])
        history_count = 0

        # Send sub-TF bars first (1s, 5s) for TBN workers
        for tf_label in ['1s', '5s']:
            if tf_label not in self._data:
                continue
            period_s = ATLAS_TFS[tf_label]
            df = self._data[tf_label]
            hist = df[df['timestamp'] <= warmup_ts]
            for _, row in hist.iterrows():
                await self._send_bar(tf_label, period_s, row.to_dict())
                history_count += 1
            if history_count % 5000 == 0:
                await asyncio.sleep(0)  # yield to event loop

        # Send anchor TF warmup bars
        for i in range(n_warmup):
            row = anchor_df.iloc[i]
            await self._send_bar(self._anchor_tf, self._anchor_period,
                                 row.to_dict())
            history_count += 1
            if history_count % 1000 == 0:
                await asyncio.sleep(0)

        # Send higher-TF bars as history too (for TBN workers)
        for tf_label, period_s in ATLAS_TFS.items():
            if period_s <= self._anchor_period:
                continue  # already sent or is anchor
            if tf_label not in self._data:
                continue
            df = self._data[tf_label]
            hist = df[df['timestamp'] <= warmup_ts]
            for _, row in hist.iterrows():
                await self._send_bar(tf_label, period_s, row.to_dict())
                history_count += 1

        await self._send_history_done(history_count)
        logger.info(f"History phase complete: {history_count:,} bars sent")

        # ── Phase 2: Live stream (paced) ──────────────────────────────────
        # Send remaining anchor bars at configured speed, interleaving
        # sub-TF and higher-TF bars that fall within each anchor window
        delay_s = self._speed_ms / 1000.0
        live_count = 0
        n_live = n_total - n_warmup

        # Pre-build timestamp indices for sub/higher TFs
        tf_cursors: Dict[str, int] = {}  # tf -> next row index to send
        for tf_label, df in self._data.items():
            if tf_label == self._anchor_tf:
                continue
            # Find first bar after warmup timestamp
            mask = df['timestamp'] > warmup_ts
            idx = mask.idxmax() if mask.any() else len(df)
            tf_cursors[tf_label] = idx

        for i in range(n_warmup, n_total):
            if not self._connected:
                break

            row = anchor_df.iloc[i]
            bar_ts = float(row['timestamp'])
            self._current_price = float(row['close'])

            # Send any sub-TF and higher-TF bars up to this anchor timestamp
            for tf_label, cursor in list(tf_cursors.items()):
                if tf_label not in self._data:
                    continue
                df = self._data[tf_label]
                period_s = ATLAS_TFS[tf_label]
                while cursor < len(df) and float(df.iloc[cursor]['timestamp']) <= bar_ts:
                    await self._send_bar(
                        tf_label, period_s, df.iloc[cursor].to_dict())
                    cursor += 1
                tf_cursors[tf_label] = cursor

            # Send anchor bar
            await self._send_bar(self._anchor_tf, self._anchor_period,
                                 row.to_dict())
            self._bars_sent += 1
            live_count += 1

            # Progress logging
            if live_count % 500 == 0:
                pct = live_count / n_live * 100
                elapsed = time.time() - self._start_time
                eta_min = (n_live - live_count) * delay_s / 60
                logger.info(f"Replay: {live_count:,}/{n_live:,} bars "
                            f"({pct:.1f}%)  -- {self._fills_sent} fills  -- "
                            f"ETA {eta_min:.1f}min")

            # Pace control
            await asyncio.sleep(delay_s)

        # ── Done ──────────────────────────────────────────────────────────
        elapsed = time.time() - self._start_time
        logger.info("=" * 60)
        logger.info(f"REPLAY COMPLETE")
        logger.info(f"  Bars sent:  {self._bars_sent:,} live + {history_count:,} history")
        logger.info(f"  Fills:      {self._fills_sent}")
        logger.info(f"  Elapsed:    {elapsed:.1f}s ({elapsed/60:.1f}min)")
        logger.info(f"  Sim pos:    {self._sim_position or 'FLAT'}")
        logger.info("=" * 60)

        # Give engine time to process final bars
        await asyncio.sleep(2.0)

        # If position still open, force close
        if self._sim_position:
            logger.warning("Forcing position close at end of replay")
            await self._handle_close_position({})

        # Keep connection alive briefly for final messages
        await asyncio.sleep(3.0)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='ReplayBridge  -- feed ATLAS parquet through NT8 protocol')
    parser.add_argument('atlas_dir',
                        help='Path to ATLAS directory (e.g. DATA/ATLAS_OOS)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'TCP port (default: {DEFAULT_PORT})')
    parser.add_argument('--speed', type=int, default=DEFAULT_SPEED_MS,
                        help=f'ms per anchor bar (default: {DEFAULT_SPEED_MS})')
    parser.add_argument('--instrument', default='MNQ 03-26',
                        help='Instrument name (default: MNQ 03-26)')
    parser.add_argument('--anchor-tf', default='15s',
                        help='Anchor timeframe (default: 15s)')
    parser.add_argument('--warmup-bars', type=int, default=2400,
                        help='Bars to send as history (default: 2400 = 10h)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    bridge = ReplayBridge(
        atlas_dir=args.atlas_dir,
        port=args.port,
        speed_ms=args.speed,
        instrument=args.instrument,
        anchor_tf=args.anchor_tf,
        warmup_bars=args.warmup_bars,
    )
    asyncio.run(bridge.start())


if __name__ == '__main__':
    main()
