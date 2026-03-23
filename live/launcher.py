"""
CLI launcher for the live trading connector.

Usage:
    python -m live.launcher                     # connect to NT8 (account controls sim/real)
    python -m live.launcher --account MyAccount  # specific NT8 account
    python -m live.launcher --no-gui            # headless (no popup)
    python -m live.launcher --checkpoint-dir checkpoints_v2
    python -m live.launcher --physics                       # PhysicsEngine (K-NN trajectory matching)
    python -m live.launcher --physics --seed-path path.json # custom seed file
"""

import argparse
import asyncio
import logging
import os
import queue as stdlib_queue
import sys
import threading

from live.config import LiveConfig
from live.live_engine import LiveEngine


def _kill_stale_live_engines():
    """Kill any leftover Python live engine processes from previous runs.

    Without this, a stale Python process holds the C# bridge connection
    and new connections go into the OS backlog unserviced.
    """
    import subprocess
    my_pid = os.getpid()
    try:
        # Find all python processes with 'live.launcher' in their command line
        result = subprocess.run(
            ['powershell', '-Command',
             'Get-CimInstance Win32_Process -Filter "Name like \'python%\'" '
             '| Select-Object ProcessId, CommandLine '
             '| ConvertTo-Json'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return
        import json
        procs = json.loads(result.stdout or '[]')
        if isinstance(procs, dict):
            procs = [procs]  # single result comes as dict
        for p in procs:
            pid = int(p.get('ProcessId', 0))
            cmd = p.get('CommandLine', '') or ''
            if pid != my_pid and 'live.launcher' in cmd:
                logging.getLogger(__name__).warning(
                    f"Killing stale live engine PID {pid}")
                subprocess.run(
                    ['powershell', '-Command', f'Stop-Process -Id {pid} -Force'],
                    timeout=3,
                )
    except Exception:
        pass  # best-effort; don't crash if cleanup fails


def _run_popup(gui_queue, shared_state):
    """Launch ProgressPopup in its own Tk mainloop (daemon thread)."""
    import tkinter as tk
    from visualization.dashboard import ProgressPopup

    root = tk.Tk()
    popup = ProgressPopup(root, gui_queue, shared_state=shared_state)
    root.title("Bayesian-AI  LIVE")

    def _on_close():
        # If already shutting down, force-close
        if shared_state.get('shutdown'):
            root.quit()  # break mainloop; cleanup below
            return
        # First close: flatten + wait for confirmation
        shared_state['shutdown_flatten'] = True
        root.title("Bayesian-AI  LIVE  [CLOSING...]")
        _check_shutdown()  # start polling for confirmation

    def _check_shutdown():
        """Poll until engine confirms flat, then quit mainloop."""
        if shared_state.get('shutdown_confirmed'):
            shared_state['shutdown'] = True
            root.quit()  # break mainloop; cleanup below
            return
        root.after(200, _check_shutdown)

    root.protocol("WM_DELETE_WINDOW", _on_close)
    try:
        root.mainloop()
    except Exception:
        pass

    # ── Post-mainloop cleanup (still in GUI thread) ──────────────
    # Must GC all tkinter/matplotlib objects HERE, not in the main thread.
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass
    # Release all references so GC collects tkinter objects in THIS thread
    del popup, root
    import gc
    gc.collect()

    shared_state['shutdown'] = True


def _clean_nt8_cache():
    """Delete NinjaTrader.sqlite to prevent stale connection issues."""
    import pathlib
    db_path = pathlib.Path.home() / 'OneDrive' / 'Documents' / 'NinjaTrader 8' / 'db' / 'NinjaTrader.sqlite'
    if db_path.exists():
        try:
            db_path.unlink()
            print(f"[cleanup] Deleted {db_path}")
        except PermissionError:
            print(f"[cleanup] Cannot delete {db_path}  -- NT8 may be running. Close NT8 first.")
        except Exception as e:
            print(f"[cleanup] Failed to delete {db_path}: {e}")


def main():
    """Entry point for live trading. Connects to NT8 and trades."""
    parser = argparse.ArgumentParser(
        description='Bayesian-AI NinjaTrader 8 Live Connector')
    parser.add_argument('--host', default='127.0.0.1',
                        help='NT8 bridge host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5199,
                        help='NT8 bridge port (default: 5199)')
    parser.add_argument('--account', default='DEMO6872628',
                        help='NT8 account name (default: DEMO6872628)')
    parser.add_argument('--instrument', default='MNQ 03-26',
                        help='NT8 instrument name (default: MNQ 03-26)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help='Training checkpoint directory')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run headless without progress popup')
    parser.add_argument('--max-daily-loss', type=float, default=200.0,
                        help='Max daily loss in USD before stopping (default: 200)')
    parser.add_argument('--warmup-bars', type=int, default=240,
                        help='Bars to accumulate before first signal (default: 240 = 1h)')
    parser.add_argument('--anchor-tf', default='15s',
                        choices=['1s', '5s', '15s', '30s', '1m', '3m', '5m'],
                        help='Primary signal timeframe (default: 15s)')
    parser.add_argument('--ping-pong', action='store_true',
                        help='Continuous wave-riding with direction refinement')
    parser.add_argument('--physics', action='store_true',
                        help='Use PhysicsEngine (K-NN trajectory matching) instead of AdvanceEngine')
    parser.add_argument('--seed-path', default=None,
                        help='Path to enriched seed JSON for PhysicsEngine')
    parser.add_argument('--yolo', action='store_true',
                        help='Max aggression, minimal warmup  -- force trades fast')
    parser.add_argument('--long-only', action='store_true',
                        help='Force all trades LONG (brain learns long side)')
    parser.add_argument('--short-only', action='store_true',
                        help='Force all trades SHORT (brain learns short side)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    args = parser.parse_args()

    _kill_stale_live_engines()
    _clean_nt8_cache()

    # Physics mode: force 1m anchor, auto-find seeds
    if args.physics:
        args.anchor_tf = '1m'
        if not args.seed_path:
            import glob as _glob
            _candidates = sorted(_glob.glob('DATA/regime_seeds/auto_seeds_all_*.json'))
            if _candidates:
                args.seed_path = _candidates[-1]  # most recent
            else:
                print("ERROR: --physics requires seed JSON. Use --seed-path or place in DATA/regime_seeds/")
                sys.exit(1)
        print(f"[physics] Engine: PhysicsEngine (K-NN trajectory matching)")
        print(f"[physics] Seeds:  {args.seed_path}")
        print(f"[physics] Anchor: 1m")

    # YOLO mode: override warmup + aggression
    if args.yolo:
        args.warmup_bars = 20   # ~5 minutes of 15s bars
        args.log_level = 'DEBUG'

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
    # Silence numba CUDA debug spam
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('numba.cuda').setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings('ignore', module='numba')

    config = LiveConfig(
        nt8_host=args.host,
        nt8_port=args.port,
        account=args.account,
        instrument=args.instrument,
        checkpoint_dir=args.checkpoint_dir,
        warmup_bars=args.warmup_bars,
        max_daily_loss_usd=args.max_daily_loss,
        anchor_tf=args.anchor_tf,
        ping_pong=args.ping_pong,
    )

    # Validate side lock
    if args.long_only and args.short_only:
        print("ERROR: --long-only and --short-only are mutually exclusive")
        sys.exit(1)

    # Shared mutable state between GUI and engine (thread-safe via GIL)
    shared_state = {
        'aggression': 1.0 if args.yolo else 0.5,
        'ping_pong': args.ping_pong,
        'side_lock': 'long' if args.long_only else ('short' if args.short_only else None),
        'physics_mode': args.physics,
        'seed_path': getattr(args, 'seed_path', None),
    }

    # Launch GUI popup (unless --no-gui)
    gui_queue = None
    if not args.no_gui:
        gui_queue = stdlib_queue.Queue(maxsize=5000)
        gui_thread = threading.Thread(
            target=_run_popup, args=(gui_queue, shared_state),
            daemon=True, name='LivePopup')
        gui_thread.start()

    engine = LiveEngine(config, gui_queue=gui_queue, shared_state=shared_state)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass
    finally:
        # Signal GUI to exit cleanly (avoids Tcl_AsyncDelete crash)
        shared_state['shutdown'] = True
        if gui_queue is not None:
            gui_queue.put({'type': 'SHUTDOWN'})
        if not args.no_gui:
            gui_thread.join(timeout=3)


if __name__ == '__main__':
    main()
