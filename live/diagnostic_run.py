"""
Diagnostic Run — block by block validation of the live pipeline.

Block 1: Connect to NT8, receive bars, verify OHLCV
Block 2: Feed aggregator, verify TF bar closes
Block 3: Compute 79D features, verify values
Block 4: Feed BlendedEngine, verify tier classification
Block 5: Full run with ledger

Run with --block N to test up to that block.

Usage:
    python live/diagnostic_run.py --block 1   # just connect + bars
    python live/diagnostic_run.py --block 2   # + aggregator
    python live/diagnostic_run.py --block 3   # + features
    python live/diagnostic_run.py --block 4   # + engine (no orders)
    python live/diagnostic_run.py --block 5   # full run
"""
import asyncio
import argparse
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.protocol import MsgType

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('diagnostic')


async def block1_bars(config, max_bars=100):
    """Block 1: Connect and receive raw bars."""
    logger.info('BLOCK 1: Connect + receive bars')

    client = NT8Client(config)
    connected = await client.connect()
    if not connected:
        logger.error('Failed to connect')
        return

    count = 0
    while count < max_bars:
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning('Timeout waiting for bar')
            break

        if msg.get('type') == MsgType.BAR:
            count += 1
            ts = msg.get('timestamp', 0)
            price = msg.get('close', 0)
            vol = msg.get('volume', 0)
            if count <= 5 or count % 20 == 0:
                logger.info(f'  BAR {count}: ts={ts:.0f} close={price:.2f} vol={vol}')
        elif msg.get('type') == MsgType.HISTORY_DONE:
            logger.info(f'  HISTORY_DONE: {msg.get("bar_count", "?")} bars')
        elif msg.get('type') == MsgType.CONNECTED:
            logger.info(f'  CONNECTED: {msg}')

    logger.info(f'Block 1 PASS: received {count} bars')
    await client.disconnect()


async def block2_aggregator(config, max_bars=500):
    """Block 2: Bars -> Aggregator -> verify TF closes."""
    from training.aggregator import Aggregator

    logger.info('BLOCK 2: Aggregator TF bar closes')

    client = NT8Client(config)
    agg = Aggregator(history_limit=2000)

    tf_counts = {}

    def on_bar_close(tf, bar):
        tf_counts[tf] = tf_counts.get(tf, 0) + 1

    agg.on_bar_close = on_bar_close

    connected = await client.connect()
    if not connected:
        return

    count = 0
    history_done = False
    while count < max_bars:
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=30.0)
        except asyncio.TimeoutError:
            break

        if msg.get('type') == MsgType.BAR:
            bar = {
                'timestamp': msg.get('timestamp', 0),
                'open': msg.get('open', 0),
                'high': msg.get('high', 0),
                'low': msg.get('low', 0),
                'close': msg.get('close', 0),
                'volume': msg.get('volume', 0),
            }
            agg.feed(bar)
            count += 1

            if count % 100 == 0:
                logger.info(f'  {count} bars fed | TF closes: {tf_counts}')

        elif msg.get('type') == MsgType.HISTORY_DONE:
            history_done = True
            logger.info(f'  HISTORY_DONE at bar {count}')

    logger.info(f'Block 2 PASS: {count} bars, TF closes: {tf_counts}')
    for tf in ['5s', '15s', '1m', '5m', '15m', '1h']:
        n = agg.get_bar_count(tf)
        logger.info(f'  {tf}: {n} closed bars')
    await client.disconnect()


async def block3_features(config, max_bars=2000):
    """Block 3: Aggregator -> SFE -> 79D features."""
    from training.aggregator import Aggregator
    from core.statistical_field_engine import StatisticalFieldEngine
    from training.compute_79d import compute_79d_from_aggregator, SFE_MIN_BARS
    from core.features_79d import FEATURE_NAMES_79D

    logger.info('BLOCK 3: 79D feature computation')

    client = NT8Client(config)
    agg = Aggregator(history_limit=2000)
    sfe = StatisticalFieldEngine()
    prev_vel = {}

    pending_5s = None

    def on_bar_close(tf, bar):
        nonlocal pending_5s
        if tf == '5s':
            pending_5s = bar

    agg.on_bar_close = on_bar_close

    connected = await client.connect()
    if not connected:
        return

    count = 0
    feat_count = 0
    history_done = False

    while count < max_bars:
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=30.0)
        except asyncio.TimeoutError:
            break

        if msg.get('type') == MsgType.BAR:
            bar = {
                'timestamp': msg.get('timestamp', 0),
                'open': msg.get('open', 0),
                'high': msg.get('high', 0),
                'low': msg.get('low', 0),
                'close': msg.get('close', 0),
                'volume': msg.get('volume', 0),
            }
            pending_5s = None
            agg.feed(bar)
            count += 1

            if pending_5s is None:
                continue

            feat, prev_vel, states_by_tf, _ = compute_79d_from_aggregator(
                agg, sfe, prev_vel, bar['timestamp'])

            if feat is not None:
                feat_count += 1
                z = feat[12]   # 1m z
                vr = feat[14]  # 1m vr
                vel = feat[15] # 1m vel

                if feat_count <= 3 or feat_count % 50 == 0:
                    tfs = list(states_by_tf.keys())
                    logger.info(f'  FEAT {feat_count}: z={z:.2f} vr={vr:.2f} vel={vel:.1f} '
                                f'TFs={tfs} price={bar["close"]:.2f}')

        elif msg.get('type') == MsgType.HISTORY_DONE:
            history_done = True
            logger.info(f'  HISTORY_DONE at bar {count}, {feat_count} features computed')

    logger.info(f'Block 3 PASS: {count} bars, {feat_count} features')
    await client.disconnect()


async def block4_engine(config, max_bars=5000):
    """Block 4: Features -> BlendedEngine -> tier classification (no orders)."""
    from training.aggregator import Aggregator
    from core.statistical_field_engine import StatisticalFieldEngine
    from training.compute_79d import compute_79d_from_aggregator, SFE_MIN_BARS
    from training.nightmare_blended import BlendedEngine

    logger.info('BLOCK 4: BlendedEngine tier classification (dry run)')

    client = NT8Client(config)
    agg = Aggregator(history_limit=2000)
    sfe = StatisticalFieldEngine()
    engine = BlendedEngine(use_cnn=False)
    prev_vel = {}

    pending_5s = None

    def on_bar_close(tf, bar):
        nonlocal pending_5s
        if tf == '5s':
            pending_5s = bar

    agg.on_bar_close = on_bar_close

    connected = await client.connect()
    if not connected:
        return

    count = 0
    feat_count = 0
    system_ready = False

    while count < max_bars:
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=30.0)
        except asyncio.TimeoutError:
            break

        if msg.get('type') == MsgType.BAR:
            bar = {
                'timestamp': msg.get('timestamp', 0),
                'open': msg.get('open', 0),
                'high': msg.get('high', 0),
                'low': msg.get('low', 0),
                'close': msg.get('close', 0),
                'volume': msg.get('volume', 0),
            }
            pending_5s = None
            agg.feed(bar)
            count += 1

            if pending_5s is None:
                continue

            feat, prev_vel, _, _ = compute_79d_from_aggregator(
                agg, sfe, prev_vel, bar['timestamp'])

            if feat is None:
                continue

            feat_count += 1

            if not system_ready:
                continue

            prev_trades = len(engine.trades)
            prev_in_pos = engine.in_pos

            state = {
                'features_79d': feat,
                'price': bar['close'],
                'timestamp': bar['timestamp'],
            }
            engine.on_state(state)

            # Log events
            if engine.in_pos and not prev_in_pos:
                logger.info(f'  ENTRY: {engine.direction} {engine.entry_tier} @ {bar["close"]:.2f}')
            elif not engine.in_pos and prev_in_pos and len(engine.trades) > prev_trades:
                t = engine.trades[-1]
                logger.info(f'  EXIT: {t.get("exit_reason", "?")} pnl=${t["pnl"]:.1f}')

            if count % 500 == 0:
                logger.info(f'  {count} bars | {feat_count} feats | {len(engine.trades)} trades | '
                            f'pnl=${engine.daily_pnl:.0f}')

        elif msg.get('type') == MsgType.HISTORY_DONE:
            system_ready = True
            logger.info(f'  HISTORY_DONE — system ready. {count} bars, {feat_count} features')

    engine.force_close()
    logger.info(f'Block 4 DONE: {len(engine.trades)} trades, ${engine.daily_pnl:.0f} PnL')
    for t in engine.trades[-5:]:
        logger.info(f'  {t.get("dir", "?")} {t.get("entry_tier", "?")} pnl=${t["pnl"]:.1f} '
                    f'exit={t.get("exit_reason", "?")}')
    await client.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', type=int, default=4, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--bars', type=int, default=None)
    args = parser.parse_args()

    config = LiveConfig()

    blocks = {
        1: (block1_bars, args.bars or 100),
        2: (block2_aggregator, args.bars or 500),
        3: (block3_features, args.bars or 2000),
        4: (block4_engine, args.bars or 50000),
    }

    if args.block == 5:
        print('Block 5: use python -m live.launcher for full run')
        return

    fn, max_bars = blocks[args.block]
    asyncio.run(fn(config, max_bars))


if __name__ == '__main__':
    main()
