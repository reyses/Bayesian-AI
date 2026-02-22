"""
OrderManager — tracks order lifecycle, position state, and logs trades to CSV.

NT8 is the source of truth for position state.  The OrderManager keeps a
local shadow that updates on FILL / POSITION messages from the bridge.
"""

import csv
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict

from live.config import LiveConfig
from live.protocol import place_order, close_position

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    PENDING   = 'PENDING'
    WORKING   = 'WORKING'
    FILLED    = 'FILLED'
    CANCELLED = 'CANCELLED'
    REJECTED  = 'REJECTED'


@dataclass
class OrderRecord:
    order_id: str
    side: str          # 'BUY' or 'SELL'
    qty: int
    state: OrderState = OrderState.PENDING
    submit_time: float = 0.0
    fill_price: float = 0.0
    fill_time: float = 0.0


@dataclass
class PositionState:
    """Local shadow of NT8 position."""
    side: str = ''         # 'LONG', 'SHORT', or '' (flat)
    qty: int = 0
    avg_price: float = 0.0
    entry_time: float = 0.0
    unrealized_pnl: float = 0.0


class OrderManager:
    """Manage orders, position tracking, and trade logging."""

    def __init__(self, config: LiveConfig):
        self._cfg = config
        self._orders: Dict[str, OrderRecord] = {}
        self.position = PositionState()
        self._daily_pnl: float = 0.0
        self._trade_count: int = 0
        self._daily_loss_limit_hit = False

        # CSV trade log
        self._log_dir = os.path.join(config.checkpoint_dir, 'live_logs')
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_path = os.path.join(
            self._log_dir,
            f"trades_{time.strftime('%Y%m%d')}.csv"
        )
        self._log_headers = [
            'timestamp', 'order_id', 'side', 'qty', 'fill_price',
            'entry_price', 'pnl', 'daily_pnl', 'exit_reason',
        ]
        self._init_csv()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        return self.position.qty == 0

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def loss_limit_hit(self) -> bool:
        return self._daily_loss_limit_hit

    def build_entry_order(self, side: str) -> Optional[dict]:
        """
        Build a PLACE_ORDER message for a new entry.
        Returns None if risk limits prevent it.
        """
        if self._daily_loss_limit_hit:
            logger.warning("Daily loss limit hit — no new orders")
            return None
        if not self.is_flat:
            logger.warning(f"Already in position ({self.position.side}) — skipping entry")
            return None
        if side not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None

        oid = f"BAY_{uuid.uuid4().hex[:8]}"
        rec = OrderRecord(order_id=oid, side=side,
                          qty=self._cfg.max_position_size,
                          submit_time=time.time())
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, side,
                          self._cfg.max_position_size)
        logger.info(f"ORDER → {side} {self._cfg.max_position_size} {self._cfg.instrument}  id={oid}")
        return msg

    def build_exit_order(self, reason: str = 'signal') -> Optional[dict]:
        """Build a CLOSE_POSITION message.  Returns None if already flat."""
        if self.is_flat:
            return None
        logger.info(f"EXIT → close {self.position.side} ({reason})")
        return close_position(self._cfg.instrument, self._cfg.account)

    def on_fill(self, msg: dict):
        """Handle a FILL message from NT8."""
        oid = msg.get('order_id', '')
        rec = self._orders.get(oid)
        if rec:
            rec.state = OrderState.FILLED
            rec.fill_price = float(msg['fill_price'])
            rec.fill_time = float(msg.get('fill_time', time.time()))

        side = msg.get('side', '')
        fill_px = float(msg['fill_price'])
        qty = int(msg.get('qty', 1))

        # Determine if this is an entry or exit fill
        if self.is_flat:
            # Entry fill
            self.position = PositionState(
                side='LONG' if side == 'BUY' else 'SHORT',
                qty=qty,
                avg_price=fill_px,
                entry_time=time.time(),
            )
            logger.info(f"FILL entry: {self.position.side} @ {fill_px}")
        else:
            # Exit fill — compute PnL
            entry_px = self.position.avg_price
            if self.position.side == 'LONG':
                pnl = (fill_px - entry_px) * qty * 5.0  # MNQ = $5/pt
            else:
                pnl = (entry_px - fill_px) * qty * 5.0

            self._daily_pnl += pnl
            self._trade_count += 1

            self._log_trade(fill_px, pnl, reason='fill')

            logger.info(f"FILL exit: {self.position.side} @ {fill_px}  "
                        f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}")

            self.position = PositionState()  # flat

            # Check daily loss limit
            if self._daily_pnl <= -self._cfg.max_daily_loss_usd:
                self._daily_loss_limit_hit = True
                logger.warning(f"DAILY LOSS LIMIT HIT: ${self._daily_pnl:.2f}")

    def on_order_status(self, msg: dict):
        """Handle an ORDER_STATUS message from NT8."""
        oid = msg.get('order_id', '')
        status = msg.get('status', '')
        rec = self._orders.get(oid)
        if rec:
            if status in ('Cancelled', 'Rejected'):
                rec.state = OrderState(status.upper())
                logger.warning(f"Order {oid} {status}")
            elif status in ('Accepted', 'Working'):
                rec.state = OrderState.WORKING

    def on_position(self, msg: dict):
        """Handle a POSITION snapshot from NT8 (source of truth)."""
        nt8_qty = int(msg.get('qty', 0))
        if nt8_qty == 0:
            if not self.is_flat:
                logger.warning("NT8 says flat — syncing local state")
            self.position = PositionState()
        else:
            side = 'LONG' if nt8_qty > 0 else 'SHORT'
            self.position = PositionState(
                side=side,
                qty=abs(nt8_qty),
                avg_price=float(msg.get('avg_price', 0)),
            )
            self.position.unrealized_pnl = float(msg.get('unrealized_pnl', 0))
            logger.info(f"POSITION sync: {side} {abs(nt8_qty)} @ {self.position.avg_price}")

    def reset_daily(self):
        """Reset daily counters at session boundary."""
        self._daily_pnl = 0.0
        self._trade_count = 0
        self._daily_loss_limit_hit = False
        self._log_path = os.path.join(
            self._log_dir,
            f"trades_{time.strftime('%Y%m%d')}.csv"
        )
        self._init_csv()
        logger.info("Daily counters reset")

    # ── Internal ──────────────────────────────────────────────────────

    def _init_csv(self):
        if not os.path.exists(self._log_path):
            with open(self._log_path, 'w', newline='') as f:
                csv.writer(f).writerow(self._log_headers)

    def _log_trade(self, exit_price: float, pnl: float, reason: str = ''):
        row = [
            time.strftime('%Y-%m-%d %H:%M:%S'),
            '',  # order_id (latest)
            self.position.side,
            self.position.qty,
            exit_price,
            self.position.avg_price,
            f'{pnl:.2f}',
            f'{self._daily_pnl:.2f}',
            reason,
        ]
        with open(self._log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
