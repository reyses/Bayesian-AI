"""
OrderManager  -- tracks order lifecycle, position state, and logs trades to CSV.

NT8 is the source of truth for position state.  The OrderManager keeps a
local shadow that updates on FILL / POSITION messages from the bridge.
"""

import csv
import logging
import os
import time
import uuid
from dataclasses import dataclass
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
        self._awaiting_fill: bool = False  # True from order sent until FILL/REJECT received
        self._trade_count: int = 0
        self._daily_loss_limit_hit = False
        self.last_exit_info: dict = {}  # filled on exit FILL (entry_px, exit_px, side)
        self.exit_rejected = False     # set True when a close order is rejected

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
        if self.position.qty != 0:
            return False
        # Block if awaiting fill confirmation from NT8
        if self._awaiting_fill:
            return False
        return True

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def loss_limit_hit(self) -> bool:
        return self._daily_loss_limit_hit

    def reset_loss_limit(self):
        """Unlock trading after user override. PnL counter is NOT reset."""
        self._daily_loss_limit_hit = False
        logger.warning(f"Loss limit unlocked (daily PnL still ${self._daily_pnl:+.2f})")

    def build_entry_order(self, side: str) -> Optional[dict]:
        """
        Build a PLACE_ORDER message for a new entry.
        Returns None if risk limits prevent it.
        """
        if self._daily_loss_limit_hit:
            logger.warning("Daily loss limit hit  -- no new orders")
            return None
        if not self.is_flat:
            logger.warning(f"Already in position ({self.position.side})  -- skipping entry")
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
        self._awaiting_fill = True
        logger.info(f"ORDER -> {side} {self._cfg.max_position_size} {self._cfg.instrument}  id={oid}")
        return msg

    def build_exit_order(self, reason: str = 'signal') -> Optional[dict]:
        """Build a CLOSE_POSITION message.  Returns None if already flat."""
        if self.is_flat:
            return None
        self.exit_rejected = False  # reset before new attempt
        logger.info(f"EXIT -> close {self.position.side} ({reason})")
        return close_position(self._cfg.instrument, self._cfg.account)

    def build_flip_order(self, reason: str = 'pp_flip') -> Optional[dict]:
        """Build a 2-contract PLACE_ORDER that closes + opens opposite in one shot.

        SHORT 1 -> BUY 2 = cover short + open long (instant flip).
        Returns None if already flat.
        """
        if self.is_flat:
            return None
        flip_side = 'BUY' if self.position.side == 'SHORT' else 'SELL'
        qty = self.position.qty * 2  # 1 to close + 1 to open
        oid = f"BAY_{uuid.uuid4().hex[:8]}"
        rec = OrderRecord(order_id=oid, side=flip_side, qty=qty,
                          submit_time=time.time())
        self._orders[oid] = rec
        logger.info(f"FLIP -> {flip_side} {qty} {self._cfg.instrument}  id={oid}  ({reason})")
        return place_order(oid, self._cfg.instrument, self._cfg.account,
                           flip_side, qty)

    def on_fill(self, msg: dict) -> Optional[float]:
        """Handle a FILL message from NT8.

        Returns PnL (float) on exit fills, None on entry fills.
        """
        self._awaiting_fill = False  # got confirmation
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
                qty=min(qty, self._cfg.max_position_size),
                avg_price=fill_px,
                entry_time=time.time(),
            )
            logger.info(f"FILL entry: {self.position.side} @ {fill_px}")
            return None
        else:
            # Exit fill  -- PnL uses position qty (1), NOT fill qty (may be 2 for flip)
            entry_px = self.position.avg_price
            exit_qty = self.position.qty
            if self.position.side == 'LONG':
                pnl = (fill_px - entry_px) * exit_qty * self._cfg.point_value
            else:
                pnl = (entry_px - fill_px) * exit_qty * self._cfg.point_value

            # Preserve fill info before clearing position (for trade log)
            self.last_exit_info = {
                'entry_px': entry_px, 'exit_px': fill_px,
                'side': self.position.side,
            }

            self._daily_pnl += pnl
            self._trade_count += 1

            self._log_trade(fill_px, pnl, reason='fill')

            # Flip order: qty > position.qty means close + open opposite
            if qty > self.position.qty:
                new_side = 'LONG' if side == 'BUY' else 'SHORT'
                new_qty = qty - self.position.qty
                logger.info(f"FILL flip: {self.position.side}->{new_side} @ {fill_px}  "
                            f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}")
                self.position = PositionState(
                    side=new_side, qty=new_qty,
                    avg_price=fill_px, entry_time=time.time(),
                )
            else:
                logger.info(f"FILL exit: {self.position.side} @ {fill_px}  "
                            f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}")
                self.position = PositionState()  # flat

            # Check daily loss limit (0 = disabled)
            if self._cfg.max_daily_loss_usd > 0 and self._daily_pnl <= -self._cfg.max_daily_loss_usd:
                self._daily_loss_limit_hit = True
                logger.warning(f"DAILY LOSS LIMIT HIT: ${self._daily_pnl:.2f}")

            return pnl

    def on_order_status(self, msg: dict):
        """Handle an ORDER_STATUS message from NT8."""
        oid = msg.get('order_id', '')
        status = msg.get('status', '')
        rec = self._orders.get(oid)
        if rec:
            if status in ('Cancelled', 'Rejected'):
                rec.state = OrderState(status.upper())
                self._awaiting_fill = False  # order won't fill
                logger.warning(f"Order {oid} {status}")
            elif status in ('Accepted', 'Working'):
                rec.state = OrderState.WORKING
        else:
            # Untracked order (e.g. CLOSE_POSITION uses NT8-generated IDs)
            if status in ('Cancelled', 'Rejected'):
                logger.warning(f"Untracked order {oid} {status}")

        # Any rejected/cancelled order while we think we have a position = danger
        if status in ('Cancelled', 'Rejected') and not self.is_flat:
            self.exit_rejected = True
            logger.error(f"ORDER REJECTED while in position: {oid}  -- will retry close")

    def on_position(self, msg: dict):
        """Handle a POSITION snapshot from NT8 (source of truth)."""
        nt8_qty = int(msg.get('qty', 0))
        if nt8_qty == 0:
            if not self.is_flat:
                logger.warning("NT8 says flat  -- syncing local state")
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

    def cleanup_stale_orders(self, max_age_s: float = 60.0):
        """Remove orders stuck in PENDING/WORKING for too long."""
        now = time.time()
        stale = [oid for oid, rec in self._orders.items()
                 if rec.state in (OrderState.PENDING, OrderState.WORKING)
                 and now - rec.submit_time > max_age_s]
        for oid in stale:
            self._orders[oid].state = OrderState.CANCELLED
            logger.warning(f"Stale order pruned: {oid} "
                           f"(age={now - self._orders[oid].submit_time:.0f}s)")
        if stale:
            self._awaiting_fill = False  # stale = no fill coming
            logger.info(f"Pruned {len(stale)} stale orders")

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
