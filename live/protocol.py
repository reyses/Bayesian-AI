"""
NT8 Bridge Protocol — length-prefixed JSON over TCP.

Wire format:
    [4 bytes: uint32 big-endian payload length] [N bytes: UTF-8 JSON]

Message types:
    NT8 → Python:  BAR, FILL, ORDER_STATUS, POSITION, CONNECTED, HEARTBEAT
    Python → NT8:  PLACE_ORDER, CLOSE_POSITION, CANCEL_ORDER, SUBSCRIBE, HEARTBEAT
"""

import json
import struct
import asyncio
from enum import Enum
from typing import Optional, Dict, Any

# ── Constants ─────────────────────────────────────────────────────────────────
HEADER_SIZE = 4                # uint32 big-endian
MAX_MSG_SIZE = 1_048_576       # 1 MB safety limit
HEADER_FMT = '>I'              # big-endian unsigned int


class MsgType(str, Enum):
    """All valid message types on the wire."""
    # NT8 → Python
    BAR          = 'BAR'
    FILL         = 'FILL'
    ORDER_STATUS = 'ORDER_STATUS'
    POSITION     = 'POSITION'
    CONNECTED    = 'CONNECTED'
    HEARTBEAT    = 'HEARTBEAT'

    # Python → NT8
    PLACE_ORDER    = 'PLACE_ORDER'
    CLOSE_POSITION = 'CLOSE_POSITION'
    CANCEL_ORDER   = 'CANCEL_ORDER'
    SUBSCRIBE      = 'SUBSCRIBE'
    # HEARTBEAT is shared


# Required fields per inbound message type (minimal validation)
_REQUIRED: Dict[str, tuple] = {
    'BAR':          ('instrument', 'timestamp', 'open', 'high', 'low', 'close', 'volume'),
    'FILL':         ('order_id', 'side', 'qty', 'fill_price', 'fill_time'),
    'ORDER_STATUS': ('order_id', 'status'),
    'POSITION':     ('instrument', 'qty'),
    'CONNECTED':    ('account',),
    'HEARTBEAT':    (),
}


# ── Encode / Decode ──────────────────────────────────────────────────────────

def encode(msg: dict) -> bytes:
    """Serialize a dict to length-prefixed JSON bytes."""
    payload = json.dumps(msg, separators=(',', ':')).encode('utf-8')
    return struct.pack(HEADER_FMT, len(payload)) + payload


def decode(payload_bytes: bytes) -> dict:
    """Deserialize UTF-8 JSON bytes to dict."""
    return json.loads(payload_bytes.decode('utf-8'))


def validate(msg: dict) -> bool:
    """Check that inbound message has required fields."""
    mtype = msg.get('type', '')
    required = _REQUIRED.get(mtype)
    if required is None:
        return False  # unknown type
    return all(k in msg for k in required)


# ── Async Message Reader ─────────────────────────────────────────────────────

class MessageReader:
    """
    Reads length-prefixed JSON messages from an asyncio StreamReader.

    Usage:
        reader = MessageReader(stream_reader)
        async for msg in reader:
            handle(msg)
    """

    def __init__(self, stream: asyncio.StreamReader):
        self._stream = stream

    async def read_one(self) -> Optional[dict]:
        """Read a single message. Returns None on EOF or protocol error."""
        try:
            header = await self._stream.readexactly(HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        length = struct.unpack(HEADER_FMT, header)[0]
        if length > MAX_MSG_SIZE:
            return None  # reject oversized messages

        try:
            payload = await self._stream.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        return decode(payload)

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        msg = await self.read_one()
        if msg is None:
            raise StopAsyncIteration
        return msg


# ── Message Builders (Python → NT8) ──────────────────────────────────────────

def subscribe(instrument: str, bar_period_s: int, account: str) -> dict:
    return {
        'type': MsgType.SUBSCRIBE,
        'instrument': instrument,
        'bar_period_s': bar_period_s,
        'account': account,
    }


def place_order(order_id: str, instrument: str, account: str,
                side: str, qty: int = 1) -> dict:
    return {
        'type': MsgType.PLACE_ORDER,
        'order_id': order_id,
        'instrument': instrument,
        'account': account,
        'side': side,
        'qty': qty,
        'order_type': 'MARKET',
    }


def close_position(instrument: str, account: str) -> dict:
    return {
        'type': MsgType.CLOSE_POSITION,
        'instrument': instrument,
        'account': account,
    }


def cancel_order(order_id: str) -> dict:
    return {
        'type': MsgType.CANCEL_ORDER,
        'order_id': order_id,
    }


def heartbeat() -> dict:
    import time
    return {'type': MsgType.HEARTBEAT, 'client_time': time.time()}
