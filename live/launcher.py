"""
CLI launcher for the live trading connector.

Usage:
    python -m live.launcher                     # paper trade (Sim101)
    python -m live.launcher --dry-run           # log signals, no orders
    python -m live.launcher --account MyAccount  # real account
    python -m live.launcher --checkpoint-dir checkpoints_v2
"""

import argparse
import asyncio
import logging
import sys

from live.config import LiveConfig
from live.live_engine import LiveEngine


def main():
    parser = argparse.ArgumentParser(
        description='Bayesian-AI NinjaTrader 8 Live Connector')
    parser.add_argument('--host', default='127.0.0.1',
                        help='NT8 bridge host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5199,
                        help='NT8 bridge port (default: 5199)')
    parser.add_argument('--account', default='Sim101',
                        help='NT8 account name (default: Sim101)')
    parser.add_argument('--instrument', default='MNQ 03-26',
                        help='NT8 instrument name (default: MNQ 03-26)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help='Training checkpoint directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run full pipeline but send no orders')
    parser.add_argument('--max-daily-loss', type=float, default=200.0,
                        help='Max daily loss in USD before stopping (default: 200)')
    parser.add_argument('--warmup-bars', type=int, default=240,
                        help='Bars to accumulate before first signal (default: 240 = 1h)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('live_trading.log', mode='a'),
        ]
    )

    config = LiveConfig(
        nt8_host=args.host,
        nt8_port=args.port,
        account=args.account,
        instrument=args.instrument,
        checkpoint_dir=args.checkpoint_dir,
        warmup_bars=args.warmup_bars,
        max_daily_loss_usd=args.max_daily_loss,
    )

    engine = LiveEngine(config, dry_run=args.dry_run)
    asyncio.run(engine.run())


if __name__ == '__main__':
    main()
