"""
CLI launcher for the live trading connector.

Usage:
    python -m live.launcher                         # connect to NT8 SIM
    python -m live.launcher --account Sim101        # specific account
    python -m live.launcher --instrument "MNQ 06-26" # specific contract
    python -m live.launcher --no-gui                # headless
    python -m live.launcher --max-daily-loss 100    # custom daily limit
"""

import argparse
import asyncio
import logging
import os
import queue as stdlib_queue
import sys
import time
import threading

from live.config import LiveConfig
from live.live_engine import LiveEngine


def _kill_stale_live_engines():
    """Kill any leftover Python live engine processes from previous runs."""
    try:
        import psutil
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                if (proc.info['pid'] != current_pid and
                    'python' in (proc.info.get('name') or '').lower() and
                    any('live' in (c or '') for c in cmdline)):
                    print(f"[cleanup] Killing stale process PID={proc.info['pid']}")
                    proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    except ImportError:
        pass  # psutil not installed, skip


def _run_popup(gui_queue, shared_state):
    """Launch ProgressPopup in its own Tk mainloop (daemon thread)."""
    import tkinter as tk
    from visualization.dashboard import ProgressPopup

    root = tk.Tk()
    popup = ProgressPopup(root, gui_queue, shared_state=shared_state)
    root.title("Bayesian-AI  BLENDED LIVE")

    _close_start = [0]

    def _on_close():
        if shared_state.get('shutdown'):
            root.quit()
            return
        shared_state['shutdown_flatten'] = True
        shared_state['shutdown'] = True
        _close_start[0] = time.time()
        root.title("Bayesian-AI  LIVE  [CLOSING...]")
        _check_shutdown()

    def _check_shutdown():
        if shared_state.get('shutdown_confirmed'):
            root.quit()
            return
        # Force close after 5 seconds
        if time.time() - _close_start[0] > 5.0:
            root.quit()
            return
        root.after(200, _check_shutdown)

    root.protocol("WM_DELETE_WINDOW", _on_close)
    try:
        root.mainloop()
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass
    del popup, root
    import gc
    gc.collect()

    shared_state['shutdown'] = True


def main():
    """Entry point for live trading."""
    parser = argparse.ArgumentParser(
        description='Bayesian-AI Live Trading — BlendedEngine + 3 CNNs')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5199)
    parser.add_argument('--account', default='Sim101',
                        help='NT8 account name')
    parser.add_argument('--instrument', default='MNQ 06-26',
                        help='NT8 instrument name (front month)')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run headless without dashboard popup')
    parser.add_argument('--max-daily-loss', type=float, default=200.0,
                        help='Max daily loss in USD before stopping')
    parser.add_argument('--warmup-bars', type=int, default=60,
                        help='Bars before first signal (default: 60 = 1 min)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    _kill_stale_live_engines()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('live_trading.log', mode='a'),
        ]
    )
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('numba.cuda').setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings('ignore', module='numba')

    config = LiveConfig(
        nt8_host=args.host,
        nt8_port=args.port,
        account=args.account,
        instrument=args.instrument,
        warmup_bars=args.warmup_bars,
        max_daily_loss_usd=args.max_daily_loss,
    )

    # Shared state between GUI and engine
    shared_state = {}

    # Launch GUI
    gui_queue = None
    if not args.no_gui:
        gui_queue = stdlib_queue.Queue(maxsize=5000)
        gui_thread = threading.Thread(
            target=_run_popup, args=(gui_queue, shared_state),
            daemon=True, name='LivePopup')
        gui_thread.start()

    # Create and run engine
    engine = LiveEngine(config, gui_queue=gui_queue, shared_state=shared_state)
    asyncio.run(engine.run())


if __name__ == '__main__':
    main()
