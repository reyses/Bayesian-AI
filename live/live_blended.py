"""
Live Blended Engine — connects BlendedEngine + 3 CNNs to NT8.

Receives 1s bars from NT8 → aggregates into TFs → SFE → 79D → BlendedEngine.
Orders sent back to NT8 via existing bridge.

Architecture:
  NT8 → 1s bars → Aggregator (15s, 1m, 5m, 15m, 1h, 1D)
                → SFE per TF
                → extract_79d (6×13 grid)
                → BlendedEngine.on_state()
                → CNN flip (direction) + CNN hold (exit) + CNN risk (cut)
                → OrderManager → NT8

Usage:
    python -m live.live_blended                     # connect to NT8 SIM
    python -m live.live_blended --account Sim101     # specific account
"""
import asyncio
import logging
import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import extract_79d, FEATURE_NAMES_79D, TF_ORDER, N_FEATURES

from nn_v2.nightmare_blended import BlendedEngine
from nn_v2.aggregator import Aggregator

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.order_manager import OrderManager
from live.protocol import MsgType, place_order, close_position

logger = logging.getLogger(__name__)

SFE_MIN_BARS = 21


class LiveBlendedEngine:
    """Main live trading loop with BlendedEngine + 3 CNNs."""

    def __init__(self, config: LiveConfig):
        self.cfg = config
        self.client = NT8Client(config)
        self.order_mgr = OrderManager(config)

        # SFE + Aggregator
        self.sfe = StatisticalFieldEngine()
        self.agg = Aggregator(history_limit=2000)
        self.prev_velocities = {}

        # Blended engine with all 3 CNNs
        self.engine = BlendedEngine(use_cnn=True)

        # State
        self._bar_count = 0
        self._last_79d_ts = 0
        self._warmed_up = False
        self._history_done = False
        self._position = 'flat'  # track what NT8 says
        self._daily_pnl = 0.0

    async def run(self):
        """Main event loop."""
        logger.info('LiveBlendedEngine starting...')
        logger.info(f'  Instrument: {self.cfg.instrument}')
        logger.info(f'  Account: {self.cfg.account}')

        if not await self.client.connect():
            logger.error('Failed to connect to NT8')
            return

        logger.info('Connected to NT8. Waiting for bars...')

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(
                        self.client.inbound.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue

                msg_type = msg.get('type', '')

                if msg_type == MsgType.BAR:
                    self._on_bar(msg)
                elif msg_type == MsgType.HISTORY_DONE:
                    self._history_done = True
                    logger.info(f'History done. {self._bar_count} bars loaded.')
                elif msg_type == MsgType.FILL:
                    self.order_mgr.on_fill(msg)
                elif msg_type == MsgType.ORDER_STATUS:
                    self.order_mgr.on_order_status(msg)
                elif msg_type == MsgType.POSITION:
                    self._on_position(msg)

        except KeyboardInterrupt:
            logger.info('Shutting down...')
        finally:
            # Force close any open position
            if self.engine.in_pos:
                logger.info('Closing open position on shutdown')
                await self._send_close()
            await self.client.disconnect()

    def _on_bar(self, msg: dict):
        """Process one 1s bar from NT8."""
        bar = {
            'timestamp': msg.get('timestamp', 0),
            'open': msg.get('open', 0),
            'high': msg.get('high', 0),
            'low': msg.get('low', 0),
            'close': msg.get('close', 0),
            'volume': msg.get('volume', 0),
        }

        self._bar_count += 1
        self.agg.feed(bar)

        # Only compute 79D at 1m bar close
        ts = bar['timestamp']
        if (int(ts) % 60) >= 5:
            return  # not a 1m boundary

        # Build 79D from aggregator state
        states_by_tf = {}
        ohlcv_by_tf = {}

        for tf in TF_ORDER:
            df = self.agg.get_closed_bars_df(tf)
            partial = self.agg.get_partial_bar(tf)
            if partial is not None:
                partial_df = pd.DataFrame([partial])
                full_df = pd.concat([df, partial_df], ignore_index=True) if len(df) > 0 else partial_df
            else:
                full_df = df

            if len(full_df) < SFE_MIN_BARS:
                continue

            ohlcv_by_tf[tf] = full_df
            sfe_input = full_df.tail(300).reset_index(drop=True) if len(full_df) > 300 else full_df
            states = self.sfe.batch_compute_states(sfe_input)
            if states:
                states_by_tf[tf] = states[-1]

        if '1m' not in states_by_tf:
            return

        # Extract 79D
        feat, self.prev_velocities = extract_79d(
            states_by_tf, ohlcv_by_tf, self.prev_velocities, ts)

        # Warmup check
        if not self._warmed_up:
            if self._bar_count < self.cfg.warmup_bars:
                return
            self._warmed_up = True
            logger.info(f'Warmed up after {self._bar_count} bars')

        # Don't trade during history replay
        if not self._history_done:
            return

        # Feed BlendedEngine
        state = {
            'features_79d': feat,
            'price': bar['close'],
            'timestamp': ts,
        }

        prev_in_pos = self.engine.in_pos
        prev_dir = self.engine.direction
        prev_n_trades = len(self.engine.trades)

        self.engine.on_state(state)

        # Check for trade events
        new_trade = len(self.engine.trades) > prev_n_trades
        entered = self.engine.in_pos and not prev_in_pos
        exited = not self.engine.in_pos and prev_in_pos

        if entered:
            asyncio.create_task(self._send_entry(
                self.engine.direction, self.engine.entry_tier))

        if exited and new_trade:
            t = self.engine.trades[-1]
            self._daily_pnl += t['pnl']
            logger.info(f'TRADE CLOSED: {t["dir"]} | tier={t["entry_tier"]} | '
                        f'exit={t["exit_reason"]} | pnl=${t["pnl"]:.1f} | '
                        f'daily=${self._daily_pnl:.0f}')

        if new_trade and not entered:
            # Trade closed, need to close NT8 position
            asyncio.create_task(self._send_close())

    async def _send_entry(self, direction: str, tier: str):
        """Send entry order to NT8."""
        side = 'BUY' if direction == 'long' else 'SELL'
        order_id = f'blended_{tier}_{uuid.uuid4().hex[:8]}'

        msg = place_order(
            order_id=order_id,
            instrument=self.cfg.instrument,
            account=self.cfg.account,
            side=side,
            qty=1,
        )
        await self.client.send(msg)
        self.order_mgr.on_order_sent(order_id, side, 1)
        logger.info(f'ORDER SENT: {side} | tier={tier} | id={order_id}')

    async def _send_close(self):
        """Send close position to NT8."""
        msg = close_position(
            instrument=self.cfg.instrument,
            account=self.cfg.account,
        )
        await self.client.send(msg)
        logger.info('CLOSE SENT')

    def _on_position(self, msg: dict):
        """Update position state from NT8."""
        qty = msg.get('quantity', 0)
        if qty > 0:
            self._position = 'long'
        elif qty < 0:
            self._position = 'short'
        else:
            self._position = 'flat'


def main():
    import argparse
    p = argparse.ArgumentParser(description='Live Blended Engine')
    p.add_argument('--account', type=str, default=None)
    p.add_argument('--instrument', type=str, default=None)
    p.add_argument('--port', type=int, default=5199)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    kwargs = {}
    if args.account:
        kwargs['account'] = args.account
    if args.instrument:
        kwargs['instrument'] = args.instrument
    if args.port != 5199:
        kwargs['nt8_port'] = args.port

    config = LiveConfig(**kwargs)
    engine = LiveBlendedEngine(config)

    asyncio.run(engine.run())


if __name__ == '__main__':
    main()
