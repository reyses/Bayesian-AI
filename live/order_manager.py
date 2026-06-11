"""
OrderManager  -- tracks order lifecycle, position state, and logs trades to CSV.

NT8 is the source of truth for position state. The OrderManager keeps a local
shadow that updates on FILL / POSITION messages from the bridge.

Per-order handshake (v2, 2026-04-15):
    1. build_*_order()           -> OrderRecord(state=PENDING,  intent=OPEN/REDUCE)
    2. await client.send(msg)    -> state=SENT,    sent_time set
    3. ORDER_ACK from bridge     -> state=ACKED,   ack_time set
    4. ORDER_STATUS Working      -> state=WORKING
    5. FILL                      -> state=FILLED,  fill_time/fill_price set
       (validate position transition matches expected_position_after)
    6. ORDER_STATUS Cancelled    -> state=CANCELLED
       ORDER_STATUS Rejected     -> state=REJECTED
    *  watchdog (check_pending_timeouts) -> state=TIMED_OUT after deadlines

Every entry, chain entry, scale-out, and manual close registers an OrderRecord
so on_fill can reconcile against the rec.intent rather than guessing from the
ambient _awaiting_*_fill flag (which was the source of the chain-exit bug).
"""

import csv
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple

from live.config import LiveConfig
from live.protocol import place_order, close_position

logger = logging.getLogger(__name__)


# Handshake timeouts — orders stuck longer than this trigger watchdog alerts.
ACK_TIMEOUT_S        = 5.0    # bridge ACK after PLACE_ORDER
WORKING_TIMEOUT_S    = 10.0   # NT8 OrderState.Working after ACK
FILL_TIMEOUT_S       = 30.0   # actual fill after Working


class OrderState(str, Enum):
    PENDING   = 'PENDING'      # built locally, not yet sent
    SENT      = 'SENT'         # written to socket, awaiting ACK
    ACKED     = 'ACKED'        # bridge confirmed receipt
    WORKING   = 'WORKING'      # NT8 accepted/working (OrderState.Working)
    FILLED    = 'FILLED'       # execution received
    CANCELLED = 'CANCELLED'
    REJECTED  = 'REJECTED'
    TIMED_OUT = 'TIMED_OUT'    # watchdog gave up


# Order intent (independent of NT8 state — what we MEANT the order to do).
class OrderIntent(str, Enum):
    OPEN   = 'OPEN'    # open new position OR scale-in (same direction)
    REDUCE = 'REDUCE'  # scale-out OR full close (opposite direction)
    FLIP   = 'FLIP'    # close + reopen opposite (single 2x order)


@dataclass
class OrderRecord:
    order_id: str
    side: str                                    # 'BUY' or 'SELL'
    qty: int
    intent: OrderIntent = OrderIntent.OPEN
    is_chain: bool = False
    expected_position_after: Tuple[str, int] = ('', 0)  # (side, qty)
    state: OrderState = OrderState.PENDING
    submit_time: float = 0.0   # time the rec was built
    sent_time: float = 0.0     # time we wrote the wire message
    ack_time: float = 0.0      # bridge confirmed receipt
    working_time: float = 0.0  # NT8 OrderState.Working
    fill_time: float = 0.0     # execution
    fill_price: float = 0.0
    fill_qty: int = 0
    reject_reason: str = ''


class OrderManager:
    """Manage orders, position tracking, and trade logging.

    Position state lives in the shared Ledger (core/ledger.py), not here.
    The OrderManager owns the NT8 wire handshake and translates fills into
    ledger mutations. It also tracks a lightweight NT8 shadow (nt8_qty,
    nt8_side) for reconciliation with heartbeats/position messages.
    """

    # States that count as "in flight" for handshake bookkeeping
    _IN_FLIGHT_STATES = (
        OrderState.PENDING, OrderState.SENT,
        OrderState.ACKED,   OrderState.WORKING,
    )

    def __init__(self, config: LiveConfig, ledger=None):
        self._cfg = config
        self._ledger = ledger  # core.ledger.Ledger — single source of truth
        self._orders: Dict[str, OrderRecord] = {}
        # Lightweight NT8 shadow for heartbeat/position reconciliation.
        # NOT the position source of truth — the ledger is.
        self._nt8_qty: int = 0
        self._nt8_side: str = ''     # 'LONG', 'SHORT', ''
        self._nt8_avg_price: float = 0.0
        self._daily_pnl: float = 0.0
        # NOTE: _awaiting_*_fill are now derived from order rec states (see
        # is_awaiting_open / is_awaiting_reduce). Kept as cached booleans so
        # external callers can read a stable snapshot without iterating.
        self._awaiting_entry_fill: bool = False
        self._awaiting_exit_fill: bool = False
        self._max_position_size: int = config.max_position_size
        self._trade_count: int = 0
        self._daily_loss_limit_hit = False
        self.last_exit_info: dict = {}  # filled on exit FILL (entry_px, exit_px, side)
        self.exit_rejected = False     # set True when a close order is rejected
        self._close_seq = 0            # sequence counter for BAY_CLOSE records
        # Most-recent reconciliation mismatch (for dashboard surfacing)
        self.last_reconcile_error: str = ''

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

    # ── NT8 shadow accessors ─────────────────────────────────────────

    @property
    def nt8_qty(self) -> int:
        return self._nt8_qty

    @property
    def nt8_side(self) -> str:
        return self._nt8_side

    def _resolve_rec(self, oid: str) -> Optional[OrderRecord]:
        """Map order ID to OrderRecord, resolving BAY_CLOSE to the latest sequence."""
        if oid == 'BAY_CLOSE' and self._close_seq > 0:
            return self._orders.get(f'BAY_CLOSE#{self._close_seq}')
        return self._orders.get(oid)

    # ── Public API ────────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        """True only when ledger is flat AND NO orders in flight."""
        if self._ledger and not self._ledger.is_flat:
            return False
        if self.is_awaiting_open() or self.is_awaiting_reduce():
            return False
        return True

    @property
    def can_enter(self) -> bool:
        """True only when flat AND no pending entry or exit."""
        return self.is_flat and not self._daily_loss_limit_hit

    @property
    def can_exit(self) -> bool:
        """True when ledger has positions AND not already waiting for a REDUCE fill."""
        if self._ledger:
            return not self._ledger.is_flat and not self.is_awaiting_reduce()
        return self._nt8_qty != 0 and not self.is_awaiting_reduce()

    @property
    def can_scale_out(self) -> bool:
        """True when multiple contracts open and no in-flight orders."""
        if self._ledger:
            return self._ledger.n_contracts > 1 and not self.is_awaiting_reduce()
        return self._nt8_qty > 1 and not self.is_awaiting_reduce()

    # ── Handshake state queries ───────────────────────────────────────

    def is_awaiting_open(self) -> bool:
        """True if any OPEN-intent (or FLIP) order is in flight."""
        for rec in self._orders.values():
            if rec.state in self._IN_FLIGHT_STATES and rec.intent in (
                    OrderIntent.OPEN, OrderIntent.FLIP):
                return True
        return False

    def is_awaiting_reduce(self) -> bool:
        """True if any REDUCE-intent (or FLIP) order is in flight."""
        for rec in self._orders.values():
            if rec.state in self._IN_FLIGHT_STATES and rec.intent in (
                    OrderIntent.REDUCE, OrderIntent.FLIP):
                return True
        return False

    def in_flight_orders(self) -> List[OrderRecord]:
        """All orders not yet in a terminal state."""
        return [r for r in self._orders.values()
                if r.state in self._IN_FLIGHT_STATES]

    def _refresh_pending_flags(self):
        """Recompute the cached _awaiting_* booleans from rec states."""
        self._awaiting_entry_fill = self.is_awaiting_open()
        self._awaiting_exit_fill  = self.is_awaiting_reduce()

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

    def build_entry_order(self, side: str, allow_flip: bool = False) -> Optional[dict]:
        """
        Build a PLACE_ORDER message for a new entry.
        Returns None if ANY order is pending or risk limits hit.

        Args:
            side: 'BUY' or 'SELL'
            allow_flip: when True, bypass the `can_enter` check (which
                requires is_flat). Used when engine signals an exit + entry
                in the same bar (L5 zigzag-reverse flip). The exit order
                must have been sent immediately prior.
        """
        # Hard safety: never exceed max in-flight orders
        MAX_OPEN_ORDERS = 3
        in_flight = len(self.in_flight_orders())
        if in_flight >= MAX_OPEN_ORDERS:
            logger.error(f"SAFETY LOCKOUT: {in_flight} in-flight orders (max={MAX_OPEN_ORDERS})")
            return None

        if not allow_flip and not self.can_enter:
            if self.is_awaiting_open():
                logger.warning("Entry blocked: awaiting OPEN fill")
            elif self.is_awaiting_reduce():
                logger.warning("Entry blocked: awaiting REDUCE fill")
            elif self._ledger and not self._ledger.is_flat:
                prim = self._ledger.primary
                logger.warning(f"Entry blocked: in position ({prim.direction if prim else '?'})")
            elif self._daily_loss_limit_hit:
                logger.warning("Entry blocked: daily loss limit")
            return None
        if side not in ('BUY', 'SELL'):
            logger.error(f"Invalid side: {side}")
            return None

        oid = self._make_order_id(side, tag='ENTRY')
        qty = self._cfg.max_position_size
        expected_side = 'LONG' if side == 'BUY' else 'SHORT'
        rec = OrderRecord(
            order_id=oid, side=side, qty=qty,
            intent=OrderIntent.OPEN, is_chain=False,
            expected_position_after=(expected_side, qty),
            state=OrderState.PENDING,
            submit_time=time.time(),
        )
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, side, qty,
                          position_effect='OPEN')
        self._refresh_pending_flags()
        logger.info(f"ORDER -> {side} {qty} {self._cfg.instrument}  id={oid}  effect=OPEN")
        return msg

    @property
    def can_scale_in(self) -> bool:
        """True when in position, same direction scale-in is safe, and under max contracts."""
        n = self._ledger.n_contracts if self._ledger else self._nt8_qty
        if n == 0:
            return False
        if n >= self._cfg.max_contracts:
            return False
        if self.is_awaiting_open() or self.is_awaiting_reduce():
            return False
        if self._daily_loss_limit_hit:
            return False
        return True

    def build_scale_in_order(self, side: str) -> Optional[dict]:
        """Add 1 contract to existing position (chain entry at OrderManager level).

        Same direction as current position. Returns None if blocked.
        """
        if not self.can_scale_in:
            n = self._ledger.n_contracts if self._ledger else self._nt8_qty
            logger.warning(f"Scale-in blocked: contracts={n} "
                           f"max={self._cfg.max_contracts} "
                           f"pending_open={self.is_awaiting_open()} "
                           f"pending_reduce={self.is_awaiting_reduce()}")
            return None

        # Safety: side must match current position direction
        prim = self._ledger.primary if self._ledger else None
        if prim:
            expected_side = 'BUY' if prim.direction == 'long' else 'SELL'
        else:
            expected_side = 'BUY' if self._nt8_side == 'LONG' else 'SELL'
        if side != expected_side:
            logger.error(f"Scale-in side mismatch: {side} vs ledger {prim.direction if prim else self._nt8_side}")
            return None

        n = self._ledger.n_contracts if self._ledger else self._nt8_qty
        oid = self._make_order_id(side, tag='CHAIN')
        rec = OrderRecord(
            order_id=oid, side=side, qty=1,
            intent=OrderIntent.OPEN, is_chain=True,
            expected_position_after=(self._nt8_side or ('LONG' if side == 'BUY' else 'SHORT'), n + 1),
            state=OrderState.PENDING,
            submit_time=time.time(),
        )
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, side, 1,
                          position_effect='OPEN')
        self._refresh_pending_flags()
        logger.info(f"SCALE-IN -> {side} +1 {self._cfg.instrument} "
                     f"(total will be {n + 1})  id={oid}  effect=OPEN")
        return msg

    def build_scale_out_order(self, reason: str = 'chain_exit') -> Optional[dict]:
        """Remove 1 contract from position (chain exit at OrderManager level).

        If this is the last contract, use build_exit_order instead.
        Returns None if blocked or position qty <= 1.

        Sends position_effect=REDUCE so the bridge uses Sell/BuyToCover (closing
        action) rather than SellShort/Buy (which would open a new opposing position).
        """
        n = self._ledger.n_contracts if self._ledger else self._nt8_qty
        if n <= 1:
            logger.warning(f"Scale-out blocked: contracts={n} (use build_exit_order)")
            return None
        if self.is_awaiting_open() or self.is_awaiting_reduce():
            logger.warning(f"Scale-out blocked: pending order in flight")
            return None

        # Opposite side to close 1 contract
        prim = self._ledger.primary if self._ledger else None
        if prim:
            close_side = 'SELL' if prim.direction == 'long' else 'BUY'
        else:
            close_side = 'SELL' if self._nt8_side == 'LONG' else 'BUY'

        oid = self._make_order_id(close_side, tag='CHEXIT')
        nt8_side = self._nt8_side or ('LONG' if close_side == 'SELL' else 'SHORT')
        rec = OrderRecord(
            order_id=oid, side=close_side, qty=1,
            intent=OrderIntent.REDUCE, is_chain=True,
            expected_position_after=(nt8_side, max(n - 1, 0)),
            state=OrderState.PENDING,
            submit_time=time.time(),
        )
        self._orders[oid] = rec

        msg = place_order(oid, self._cfg.instrument,
                          self._cfg.account, close_side, 1,
                          position_effect='REDUCE')
        self._refresh_pending_flags()
        logger.info(f"SCALE-OUT -> {close_side} -1 {self._cfg.instrument} "
                     f"(total will be {max(n - 1, 0)})  id={oid}  effect=REDUCE  ({reason})")
        return msg

    def build_exit_order(self, reason: str = 'signal') -> Optional[dict]:
        """Build a CLOSE_POSITION message. Closes ALL contracts.

        Blocks if already exiting or flat. Pre-registers a BAY_CLOSE OrderRecord
        so on_fill can reconcile the resulting fill against an intent=REDUCE entry.
        """
        if not self.can_exit:
            if self.is_awaiting_reduce():
                logger.warning(f"Exit blocked: already awaiting REDUCE fill ({reason})")
            return None

        # Pre-register the BAY_CLOSE order so on_fill knows the intent.
        # The bridge generates fills with order_id="BAY_CLOSE" for CLOSE_POSITION.
        prim = self._ledger.primary if self._ledger else None
        if prim:
            close_side = 'SELL' if prim.direction == 'long' else 'BUY'
        else:
            close_side = 'SELL' if self._nt8_side == 'LONG' else 'BUY'
        n = self._ledger.n_contracts if self._ledger else self._nt8_qty
        self._close_seq += 1
        seq_key = f'BAY_CLOSE#{self._close_seq}'
        rec = OrderRecord(
            order_id='BAY_CLOSE',
            side=close_side, qty=n,
            intent=OrderIntent.REDUCE, is_chain=False,
            expected_position_after=('', 0),  # FLAT
            state=OrderState.PENDING,
            submit_time=time.time(),
        )
        self._orders[seq_key] = rec

        self.exit_rejected = False
        self._refresh_pending_flags()
        side_str = prim.direction if prim else self._nt8_side
        logger.info(f"EXIT -> close ALL {side_str} x{n} ({reason})")
        return close_position(self._cfg.instrument, self._cfg.account)

    def build_flip_order(self, reason: str = 'pp_flip') -> Optional[dict]:
        """Build a 2-contract PLACE_ORDER that closes + opens opposite in one shot.

        SHORT 1 -> BUY 2 = cover short + open long (instant flip).
        Returns None if already flat.

        NOTE: This uses position_effect=OPEN because NT8 OrderAction.Buy/SellShort
        with qty > position handles the close-then-open atomically. REDUCE wouldn't
        work for the "open" half. The bridge will accept this because it never
        rejects OPEN.
        """
        if self.is_flat:
            return None
        flip_side = 'BUY' if self._nt8_side == 'SHORT' else 'SELL'
        qty = self._nt8_qty * 2  # 1 to close + 1 to open
        oid = f"BAY_{uuid.uuid4().hex[:8]}"
        new_side = 'LONG' if flip_side == 'BUY' else 'SHORT'
        rec = OrderRecord(
            order_id=oid, side=flip_side, qty=qty,
            intent=OrderIntent.FLIP, is_chain=False,
            expected_position_after=(new_side, self._nt8_qty),
            state=OrderState.PENDING,
            submit_time=time.time(),
        )
        self._orders[oid] = rec
        self._refresh_pending_flags()
        logger.info(f"FLIP -> {flip_side} {qty} {self._cfg.instrument}  id={oid}  ({reason})")
        return place_order(oid, self._cfg.instrument, self._cfg.account,
                           flip_side, qty, position_effect='OPEN')

    # ── Wire send tracking ────────────────────────────────────────────

    def mark_sent(self, order_id: str):
        """Called by the engine immediately after writing the order to the wire.

        Advances the rec from PENDING -> SENT so the watchdog knows the bridge
        has had the order since this moment (for ACK timeout).
        """
        rec = self._resolve_rec(order_id)
        if rec and rec.state == OrderState.PENDING:
            rec.state = OrderState.SENT
            rec.sent_time = time.time()

    def on_fill(self, msg: dict) -> Optional[float]:
        """Handle a FILL message from NT8.

        Returns PnL (float) on REDUCE fills, None on OPEN fills.

        Classification rules (in priority order):
          1. If we have an OrderRecord for this order_id, use rec.intent —
             the order was built locally and we know what it was for.
          2. Otherwise (untracked: manual NT8 order, recovery, etc.) infer
             from the side vs current position direction.

        After applying the fill, validate the resulting position against
        rec.expected_position_after and surface any mismatch via
        last_reconcile_error so the engine can alert the user.
        """
        oid = msg.get('order_id', '')
        rec = self._resolve_rec(oid)

        side = msg.get('side', '')
        fill_px = float(msg['fill_price'])
        qty = int(msg.get('qty', 1))
        fill_time = float(msg.get('fill_time', time.time()))

        # Determine intent
        if rec is not None:
            intent = rec.intent
            rec.state = OrderState.FILLED
            rec.fill_price = fill_px
            rec.fill_time = fill_time
            rec.fill_qty = qty
            tracked = True
        else:
            tracked = False
            if self._nt8_qty == 0:
                intent = OrderIntent.OPEN
            else:
                same_direction = ((self._nt8_side == 'LONG' and side == 'BUY') or
                                  (self._nt8_side == 'SHORT' and side == 'SELL'))
                intent = OrderIntent.OPEN if same_direction else OrderIntent.REDUCE
            logger.warning(f"FILL untracked order_id={oid} — inferred intent={intent}")

        pnl: Optional[float] = None

        # ── Apply position transition via ledger ─────────────────────
        if intent == OrderIntent.OPEN:
            if self._ledger:
                # Find the pending request context from engine_v2's _pending_requests
                # The caller (engine_v2) stores entry context on _pending_requests[oid].
                # on_fill receives the raw msg; entry context is passed through rec fields.
                is_chain = rec.is_chain if rec else False
                # Ledger mutation happens in engine_v2.py's fill handler via
                # _pending_requests context. Here we just update the NT8 shadow.
                pass

            # Update NT8 shadow
            if self._nt8_qty == 0:
                self._nt8_side = 'LONG' if side == 'BUY' else 'SHORT'
                self._nt8_qty = qty
                self._nt8_avg_price = fill_px
                logger.info(f"FILL entry: {self._nt8_side} x{qty} @ {fill_px}  id={oid}")
            else:
                old_qty = self._nt8_qty
                new_qty = old_qty + qty
                new_avg = (self._nt8_avg_price * old_qty + fill_px * qty) / new_qty
                self._nt8_qty = min(new_qty, self._cfg.max_contracts)
                self._nt8_avg_price = new_avg
                logger.info(f"FILL scale-in: {self._nt8_side} +{qty} @ {fill_px} "
                            f"(total={self._nt8_qty}, avg={new_avg:.2f})  id={oid}")

        elif intent == OrderIntent.REDUCE:
            if self._nt8_qty == 0:
                logger.error(f"FILL REDUCE on FLAT position! id={oid} side={side} qty={qty} @ {fill_px}")
                self.last_reconcile_error = f"REDUCE fill {oid} on flat position"
                self._refresh_pending_flags()
                return None

            close_qty = min(qty, self._nt8_qty)
            entry_px = self._nt8_avg_price

            if self._nt8_side == 'LONG':
                pnl = (fill_px - entry_px) * close_qty * self._cfg.point_value
            else:
                pnl = (entry_px - fill_px) * close_qty * self._cfg.point_value

            self._daily_pnl += pnl
            self._trade_count += 1

            if close_qty >= self._nt8_qty:
                self.last_exit_info = {
                    'entry_px': entry_px, 'exit_px': fill_px,
                    'side': self._nt8_side, 'qty': close_qty,
                }
                self._log_trade(fill_px, pnl, reason='close')
                logger.info(f"FILL close-all: {self._nt8_side} x{close_qty} @ {fill_px} "
                            f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}  id={oid}")
                self._nt8_qty = 0
                self._nt8_side = ''
                self._nt8_avg_price = 0.0
            else:
                self._nt8_qty -= close_qty
                self._log_trade(fill_px, pnl, reason='scale_out')
                logger.info(f"FILL scale-out: -{close_qty} @ {fill_px} "
                            f"PnL=${pnl:+.2f} (remaining={self._nt8_qty})  "
                            f"daily=${self._daily_pnl:+.2f}  id={oid}")

            if self._cfg.max_daily_loss_usd > 0 and self._daily_pnl <= -self._cfg.max_daily_loss_usd:
                self._daily_loss_limit_hit = True
                logger.warning(f"DAILY LOSS LIMIT HIT: ${self._daily_pnl:.2f}")

        elif intent == OrderIntent.FLIP:
            if self._nt8_qty == 0:
                logger.error(f"FILL FLIP on FLAT position! id={oid}")
                self.last_reconcile_error = f"FLIP fill {oid} on flat position"
                self._refresh_pending_flags()
                return None
            close_qty = self._nt8_qty
            entry_px = self._nt8_avg_price
            if self._nt8_side == 'LONG':
                pnl = (fill_px - entry_px) * close_qty * self._cfg.point_value
            else:
                pnl = (entry_px - fill_px) * close_qty * self._cfg.point_value
            self.last_exit_info = {
                'entry_px': entry_px, 'exit_px': fill_px,
                'side': self._nt8_side, 'qty': close_qty,
            }
            self._daily_pnl += pnl
            self._trade_count += 1
            self._log_trade(fill_px, pnl, reason='flip')
            remaining = qty - close_qty
            new_side = 'LONG' if side == 'BUY' else 'SHORT'
            self._nt8_side = new_side
            self._nt8_qty = max(remaining, 0)
            self._nt8_avg_price = fill_px
            logger.info(f"FILL flip: -> {new_side} x{remaining} @ {fill_px} "
                        f"PnL=${pnl:+.2f}  daily=${self._daily_pnl:+.2f}  id={oid}")

        # ── Reconcile against expected position ───────────────────────
        if tracked and rec.expected_position_after != ('', 0) or (
                tracked and intent == OrderIntent.REDUCE
                and rec.expected_position_after == ('', 0)):
            exp_side, exp_qty = rec.expected_position_after
            actual_side = self._nt8_side or ''
            actual_qty = self._nt8_qty
            if (exp_side or '') != actual_side or exp_qty != actual_qty:
                msg_str = (f"RECONCILE MISMATCH {oid}: expected={exp_side} x{exp_qty} "
                           f"actual={actual_side or 'FLAT'} x{actual_qty}")
                logger.error(msg_str)
                self.last_reconcile_error = msg_str

        self._refresh_pending_flags()
        return pnl

    def on_order_ack(self, msg: dict):
        """Handle ORDER_ACK — bridge confirms receipt of our order.

        Advances the rec from SENT -> ACKED. This is the first step of the
        handshake; if we never see this within ACK_TIMEOUT_S the watchdog
        marks the order TIMED_OUT and refuses dependent orders.
        """
        oid = msg.get('order_id', '')
        rec = self._resolve_rec(oid)
        if rec:
            if rec.state in (OrderState.PENDING, OrderState.SENT):
                rec.state = OrderState.ACKED
                rec.ack_time = time.time()
            logger.info(f"ACK received: {oid} (bridge has it)  state={rec.state}")
        else:
            logger.warning(f"ACK for unknown order: {oid}")

    def on_heartbeat(self, msg: dict):
        """Handle enhanced heartbeat — reconcile NT8 shadow.

        If bridge says FLAT but our shadow disagrees, bridge wins.
        The ledger is reconciled separately by engine_v2 (via POSITION messages).
        """
        nt8_qty = int(msg.get('position_qty', 0))
        nt8_side = msg.get('position_side', 'FLAT')

        if nt8_qty == 0 and self._nt8_qty != 0:
            logger.warning(f"HEARTBEAT DRIFT: NT8=FLAT but shadow={self._nt8_side} x{self._nt8_qty} — syncing")
            self._nt8_qty = 0
            self._nt8_side = ''
            self._nt8_avg_price = 0.0
            self._cancel_in_flight('heartbeat_flat_drift')
        elif nt8_qty != 0 and self._nt8_qty == 0 and not self.is_awaiting_open():
            logger.warning(f"HEARTBEAT DRIFT: NT8={nt8_side} x{nt8_qty} but shadow=FLAT — syncing")
            self._nt8_side = nt8_side
            self._nt8_qty = nt8_qty
            self._nt8_avg_price = float(msg.get('position_avg_price', 0))
        elif nt8_qty != self._nt8_qty and nt8_qty != 0:
            logger.warning(f"HEARTBEAT DRIFT: NT8={nt8_side} x{nt8_qty} vs shadow={self._nt8_side} x{self._nt8_qty}")
            self._nt8_qty = nt8_qty

    def on_order_status(self, msg: dict):
        """Handle an ORDER_STATUS message from NT8."""
        oid = msg.get('order_id', '')
        status = msg.get('status', '')
        rec = self._resolve_rec(oid)
        if rec:
            if status in ('Cancelled', 'Rejected'):
                rec.state = OrderState(status.upper())
                rec.reject_reason = msg.get('reason', '') or status
                logger.warning(f"Order {oid} {status} reason={rec.reject_reason}")
            elif status in ('Accepted', 'Working', 'Submitted'):
                if rec.state in (OrderState.PENDING, OrderState.SENT, OrderState.ACKED):
                    rec.state = OrderState.WORKING
                    rec.working_time = time.time()
            elif status == 'Filled':
                # Belt-and-suspenders: status=Filled may arrive before/with FILL.
                # The actual position transition still happens in on_fill.
                pass
        else:
            # Untracked order (e.g. NT8-side manual cancels)
            if status in ('Cancelled', 'Rejected'):
                logger.warning(f"Untracked order {oid} {status}")

        # Any rejected/cancelled order while we think we have a position = danger
        if status in ('Cancelled', 'Rejected') and not self.is_flat:
            self.exit_rejected = True
            logger.error(f"ORDER REJECTED while in position: {oid}  -- will retry close")

        self._refresh_pending_flags()

    def on_position(self, msg: dict):
        """Handle a POSITION snapshot from NT8 — update the NT8 shadow.

        NT8 POSITION is authoritative for the shadow. Ledger reconciliation
        is handled by engine_v2.py (which may force-close the ledger if NT8
        reports flat but the ledger has positions).
        """
        nt8_qty = int(msg.get('qty', 0))
        if nt8_qty == 0:
            if self._nt8_qty != 0 or self.is_awaiting_open() or self.is_awaiting_reduce():
                logger.warning(f"NT8 says flat -- syncing shadow (was: {self._nt8_side} x{self._nt8_qty})")
            self._nt8_qty = 0
            self._nt8_side = ''
            self._nt8_avg_price = 0.0
            self._cancel_in_flight('position_flat_sync')
        else:
            side = 'LONG' if nt8_qty > 0 else 'SHORT'
            self._nt8_side = side
            self._nt8_qty = abs(nt8_qty)
            self._nt8_avg_price = float(msg.get('avg_price', 0))
            logger.info(f"POSITION sync: {side} {abs(nt8_qty)} @ {self._nt8_avg_price}")

    def check_pending_timeouts(self) -> List[OrderRecord]:
        """Watchdog — return list of orders that have exceeded their deadlines.

        Mark them TIMED_OUT (terminal) so dependent orders can proceed. This is
        the safety net for the case where the bridge crashes or NT8 silently
        drops an order: the engine wouldn't know without this poll.

        Called from the engine's main loop every iteration.
        """
        now = time.time()
        timed_out: List[OrderRecord] = []
        for rec in self._orders.values():
            if rec.state == OrderState.SENT and (now - rec.sent_time) > ACK_TIMEOUT_S:
                logger.error(f"ACK TIMEOUT {rec.order_id}: {now - rec.sent_time:.1f}s "
                             f"since send (limit={ACK_TIMEOUT_S}s)")
                rec.state = OrderState.TIMED_OUT
                rec.reject_reason = 'ack_timeout'
                timed_out.append(rec)
            elif rec.state == OrderState.ACKED and (now - rec.ack_time) > WORKING_TIMEOUT_S:
                logger.error(f"WORKING TIMEOUT {rec.order_id}: {now - rec.ack_time:.1f}s "
                             f"since ACK (limit={WORKING_TIMEOUT_S}s)")
                rec.state = OrderState.TIMED_OUT
                rec.reject_reason = 'working_timeout'
                timed_out.append(rec)
            elif rec.state == OrderState.WORKING and (now - rec.working_time) > FILL_TIMEOUT_S:
                logger.error(f"FILL TIMEOUT {rec.order_id}: {now - rec.working_time:.1f}s "
                             f"since Working (limit={FILL_TIMEOUT_S}s)")
                rec.state = OrderState.TIMED_OUT
                rec.reject_reason = 'fill_timeout'
                timed_out.append(rec)
        if timed_out:
            self._refresh_pending_flags()
            self.exit_rejected = any(r.intent == OrderIntent.REDUCE for r in timed_out)
        return timed_out

    def _cancel_in_flight(self, reason: str):
        """Mark any non-terminal orders as CANCELLED — used when NT8 forces sync."""
        for rec in self._orders.values():
            if rec.state in self._IN_FLIGHT_STATES:
                rec.state = OrderState.CANCELLED
                rec.reject_reason = reason
        self._refresh_pending_flags()

    # Legacy alias — kept for any callers still using the old name.
    def cleanup_stale_orders(self, max_age_s: float = 60.0):
        """Deprecated: use check_pending_timeouts() instead."""
        return self.check_pending_timeouts()

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
            self._nt8_side,
            self._nt8_qty,
            exit_price,
            self._nt8_avg_price,
            f'{pnl:.2f}',
            f'{self._daily_pnl:.2f}',
            reason,
        ]
        with open(self._log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
