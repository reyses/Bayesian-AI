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
        self._awaiting_entry_fill: bool = False   # True from entry sent until FILL/REJECT
        self._awaiting_exit_fill: bool = False    # True from exit sent until FILL/REJECT
        self._max_position_size: int = config.max_position_size
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
        """True only when NO position AND NO pending orders of any kind."""
        if self.position.qty != 0:
            return False
        if self._awaiting_entry_fill:
            return False
        if self._awaiting_exit_fill:
            return False
        return True

    @property
    def can_enter(self) -> bool:
        """True only when flat AND no pending entry or exit."""
        return self.is_flat and not self._daily_loss_limit_hit

    @property
    def can_exit(self) -> bool:
        """True when in a position AND not already waiting for exit fill."""
        return self.position.qty != 0 and not self._awaiting_exit_fill

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
        Returns None if ANY order is pending or risk limits hit.
        """
        # Hard safety: never exceed max position size
        MAX_OPEN_ORDERS = 3  # absolute max pending orders before lockout
        _pending = sum(1 for r in self._orders.values()
                       if r.state in (OrderState.PENDING, OrderState.WORKING))
        if _pending >= MAX_OPEN_ORDERS:
            logger.error(f"SAFETY LOCKOUT: {_pending} pending orders (max={MAX_OPEN_ORDERS})")
            return None

        if not self.can_enter:
            if self._awaiting_entry_fill:
                logger.warning("Entry blocked: awaiting entry fill")
            elif self._awaiting_exit_fill:
                logger.warning("Entry blocked: awaiting exit fill")
            elif self.position.qty != 0:
                logger.warning(f"Entry blocked: in position ({self.position.side})")
            elif self._daily_loss_limit_hit:
                logger.warning("Entry blocked: daily loss limit")
            return None
        if side not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None

        oid = self._make_order_id(side, tag='ENTRY')
        rec = OrderRecord(order_id=oid, side=side,
                          qty=self._cfg.max_position_size,
                          submit_time=time.time())
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, side,
                          self._cfg.max_position_size)
        self._awaiting_entry_fill = True
        logger.info(f"ORDER -> {side} {self._cfg.max_position_size} {self._cfg.instrument}  id={oid}")
        return msg

    @property
    def can_scale_in(self) -> bool:
        """True when in position, same direction scale-in is safe, and under max contracts."""
        if self.position.qty == 0:
            return False
        if self.position.qty >= self._cfg.max_contracts:
            return False
        if self._awaiting_entry_fill or self._awaiting_exit_fill:
            return False
        if self._daily_loss_limit_hit:
            return False
        return True

    def build_scale_in_order(self, side: str) -> Optional[dict]:
        """Add 1 contract to existing position (chain entry at OrderManager level).

        Same direction as current position. Returns None if blocked.
        """
        if not self.can_scale_in:
            logger.warning(f"Scale-in blocked: qty={self.position.qty} "
                           f"max={self._cfg.max_contracts} "
                           f"pending_entry={self._awaiting_entry_fill} "
                           f"pending_exit={self._awaiting_exit_fill}")
            return None

        # Safety: side must match current position
        expected_side = 'BUY' if self.position.side == 'LONG' else 'SELL'
        if side != expected_side:
            logger.error(f"Scale-in side mismatch: {side} vs position {self.position.side}")
            return None

        oid = self._make_order_id(side, tag='CHAIN')
        rec = OrderRecord(order_id=oid, side=side, qty=1,
                          submit_time=time.time())
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, side, 1)
        self._awaiting_entry_fill = True
        logger.info(f"SCALE-IN -> {side} +1 {self._cfg.instrument} "
                     f"(total will be {self.position.qty + 1})  id={oid}")
        return msg

    def build_scale_out_order(self, reason: str = 'chain_exit') -> Optional[dict]:
        """Remove 1 contract from position (chain exit at OrderManager level).

        If this is the last contract, use build_exit_order instead.
        Returns None if blocked or position qty <= 1.
        """
        if self.position.qty <= 1:
            logger.warning(f"Scale-out blocked: qty={self.position.qty} (use build_exit_order)")
            return None
        if self._awaiting_entry_fill or self._awaiting_exit_fill:
            logger.warning(f"Scale-out blocked: pending order")
            return None

        # Opposite side to close 1 contract
        close_side = 'SELL' if self.position.side == 'LONG' else 'BUY'

        oid = self._make_order_id(close_side, tag='CHEXIT')
        rec = OrderRecord(order_id=oid, side=close_side, qty=1,
                          submit_time=time.time())
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, close_side, 1)
        self._awaiting_exit_fill = True
        logger.info(f"SCALE-OUT -> {close_side} -1 {self._cfg.instrument} "
                     f"(total will be {self.position.qty - 1})  id={oid}  ({reason})")
        return msg

    def build_exit_order(self, reason: str = 'signal') -> Optional[dict]:
        """Build a CLOSE_POSITION message. Closes ALL contracts. Blocks if already exiting or flat."""
        if not self.can_exit:
            if self._awaiting_exit_fill:
                logger.warning(f"Exit blocked: already awaiting exit fill ({reason})")
            return None
        self.exit_rejected = False
        self._awaiting_exit_fill = True
        logger.info(f"EXIT -> close ALL {self.position.side} x{self.position.qty} ({reason})")
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

        Returns PnL (float) on exit/scale-out fills, None on entry/scale-in fills.
        """
        was_awaiting_exit = self._awaiting_exit_fill
        self._awaiting_entry_fill = False
        self._awaiting_exit_fill = False

        oid = msg.get('order_id', '')
        rec = self._orders.get(oid)
        if rec:
            rec.state = OrderState.FILLED
            rec.fill_price = float(msg['fill_price'])
            rec.fill_time = float(msg.get('fill_time', time.time()))

        side = msg.get('side', '')
        fill_px = float(msg['fill_price'])
        qty = int(msg.get('qty', 1))

        # Classify fill: entry/scale-in (adds to position) vs exit/scale-out (reduces)
        if self.position.qty == 0:
            # Was flat → this is a fresh entry
            self.position = PositionState(
                side='LONG' if side == 'BUY' else 'SHORT',
                qty=qty,
                avg_price=fill_px,
                entry_time=time.time(),
            )
            logger.info(f"FILL entry: {self.position.side} x{qty} @ {fill_px}")
            return None

        # Already in position — is this adding or reducing?
        same_direction = ((self.position.side == 'LONG' and side == 'BUY') or
                          (self.position.side == 'SHORT' and side == 'SELL'))

        if same_direction:
            # Scale-in (chain entry): add contracts, update avg price
            old_qty = self.position.qty
            new_qty = old_qty + qty
            # Weighted average entry price
            new_avg = (self.position.avg_price * old_qty + fill_px * qty) / new_qty
            self.position.qty = min(new_qty, self._cfg.max_contracts)
            self.position.avg_price = new_avg
            logger.info(f"FILL scale-in: {self.position.side} +{qty} @ {fill_px} "
                        f"(total={self.position.qty}, avg={new_avg:.2f})")
            return None

        else:
            # Opposite direction = reducing/closing
            if qty >= self.position.qty or was_awaiting_exit:
                # Full close (or close_position which closes all)
                exit_qty = self.position.qty
                entry_px = self.position.avg_price
                if self.position.side == 'LONG':
                    pnl = (fill_px - entry_px) * exit_qty * self._cfg.point_value
                else:
                    pnl = (entry_px - fill_px) * exit_qty * self._cfg.point_value

                self.last_exit_info = {
                    'entry_px': entry_px, 'exit_px': fill_px,
                    'side': self.position.side, 'qty': exit_qty,
                }
                self._daily_pnl += pnl
                self._trade_count += 1
                self._log_trade(fill_px, pnl, reason='fill')

                # Flip: qty > position = close + open opposite
                remaining = qty - self.position.qty
                if remaining > 0:
                    new_side = 'LONG' if side == 'BUY' else 'SHORT'
                    logger.info(f"FILL flip: {self.position.side}->{new_side} @ {fill_px} "
                                f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}")
                    self.position = PositionState(
                        side=new_side, qty=remaining,
                        avg_price=fill_px, entry_time=time.time(),
                    )
                else:
                    logger.info(f"FILL close-all: {self.position.side} x{exit_qty} @ {fill_px} "
                                f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}")
                    self.position = PositionState()  # flat

            else:
                # Scale-out (chain exit): reduce by qty, PnL on those contracts
                entry_px = self.position.avg_price
                if self.position.side == 'LONG':
                    pnl = (fill_px - entry_px) * qty * self._cfg.point_value
                else:
                    pnl = (entry_px - fill_px) * qty * self._cfg.point_value

                self.position.qty -= qty
                self._daily_pnl += pnl
                self._trade_count += 1
                self._log_trade(fill_px, pnl, reason='scale_out')

                logger.info(f"FILL scale-out: -{qty} @ {fill_px} "
                            f"PnL=${pnl:+.2f} (remaining={self.position.qty})  "
                            f"daily=${self._daily_pnl:+.2f}")

            # Check daily loss limit
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
                self._awaiting_entry_fill = False
                self._awaiting_exit_fill = False
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
        """Handle a POSITION snapshot from NT8 (source of truth).

        NT8 POSITION is authoritative. If NT8 says flat, we're flat —
        clear all pending flags regardless of local state.
        """
        nt8_qty = int(msg.get('qty', 0))
        if nt8_qty == 0:
            if self.position.qty != 0 or self._awaiting_entry_fill or self._awaiting_exit_fill:
                logger.warning(f"NT8 says flat -- syncing (was: pos={self.position.side}, "
                              f"entry_pending={self._awaiting_entry_fill}, "
                              f"exit_pending={self._awaiting_exit_fill})")
            self.position = PositionState()
            self._awaiting_entry_fill = False
            self._awaiting_exit_fill = False
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
            self._awaiting_entry_fill = False
            self._awaiting_exit_fill = False
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

    def _make_order_id(self, side: str, tag: str = '') -> str:
        """Generate a readable order ID: BAY_ENTRY_S_001
        Format: BAY_{tag}_{L/S}_{sequence}
        """
        self._trade_count += 0  # don't increment here, just read
        dir_char = 'L' if side == 'BUY' else 'S'
        seq = len(self._orders) + 1
        return f"{tag}_{dir_char}_{seq:03d}"

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
