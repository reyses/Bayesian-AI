"""
LiveEngine — BlendedEngine + 3 CNNs on NT8 bridge.

Production trading loop with full infrastructure:
  - BlendedEngine (NMP + CNN flip + CNN hold + CNN risk + hard stop)
  - OrderManager (fills, position sync, stale cleanup)
  - SessionTracker (PnL, drawdown, win/loss stats)
  - TradeLogger (per-trade diagnostic CSV)
  - ExitWatcher (post-exit counterfactual)
  - GUIBridge (Tk dashboard)
  - Daily loss limit
  - Keep awake
  - Hot-reload tuning
  - Reconnection handling

Architecture:
  NT8 → 1s bars → nn_v2.Aggregator → SFE → extract_79d → BlendedEngine
  BlendedEngine → entry/exit decisions → OrderManager → NT8

Usage (via launcher):
    python -m live.launcher --account Sim101
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
from typing import Optional, Dict

# Suppress numba CUDA debug spam
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import extract_79d, FEATURE_NAMES_79D, TF_ORDER, N_FEATURES

from nn_v2.nightmare_blended import BlendedEngine
from nn_v2.aggregator import Aggregator

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.order_manager import OrderManager
from live.protocol import MsgType, place_order, close_position
from live.gui_bridge import GUIBridge
from live.session_tracker import SessionTracker
from live.trade_logger import TradeLogger
from nn_v2.regret import compute_regret
from config.symbols import SYMBOL_MAP

logger = logging.getLogger(__name__)

SFE_MIN_BARS = 21

# Daily loss limit
DEFAULT_DAILY_LOSS_LIMIT = 200.0  # USD — stop trading if daily loss exceeds this

# Tuning file for hot-reload
TUNING_FILE = 'live_tuning.json'
TUNING_DEFAULTS = {
    'daily_loss_limit': DEFAULT_DAILY_LOSS_LIMIT,
    'hard_stop': -150.0,
    'enabled': True,
}


class LiveEngine:
    """Production live trading with BlendedEngine + 3 CNNs."""

    def __init__(self, config: LiveConfig,
                 client=None, gui_queue=None, shared_state=None):
        self._cfg = config
        self._shared_state = shared_state or {}

        # Asset info
        self._asset = SYMBOL_MAP.get(config.asset_ticker)
        if self._asset is None:
            raise ValueError(f"Unknown asset ticker: {config.asset_ticker}")

        # Core SFE
        self._sfe = StatisticalFieldEngine()
        self._prev_velocities = {}

        # Aggregator — try warm state from maintenance
        from live.maintenance import load_state
        warm = load_state()
        if warm:
            self._agg, self._prev_velocities, self._warmup_info = warm
            self._warmed_up = True
            logger.info('Loaded warm aggregator state from maintenance')
        else:
            self._agg = Aggregator(history_limit=2000)
            self._warmup_info = None
            self._warmed_up = False
            logger.warning('No warm state — cold start')

        # BlendedEngine + 3 CNNs
        self._engine = BlendedEngine(use_cnn=True)

        # NT8 connection
        self._client = client if client is not None else NT8Client(config)

        # Infrastructure
        self._orders = OrderManager(config)
        self._gui = GUIBridge(gui_queue)
        self._session = SessionTracker(config)
        # Post-trade regret analysis (nn_v2 regret engine)
        self._regret_buffer = []  # stores recent closes for regret computation
        self._live_trades_for_brain = []  # accumulate trades for CNN retraining
        self._regret_log = []  # daily regret CSV data

        # Live brain paths (separate from backtest brain)
        self._live_brain_dir = os.path.join('live', 'brains')
        os.makedirs(self._live_brain_dir, exist_ok=True)
        self._trade_logger = TradeLogger(
            os.path.join('reports', 'live', 'trades'))

        # State
        self._bar_count = 0
        self._history_done = False
        self._system_ready = False
        self._position_open = False
        self._closing_position = False
        self._daily_pnl = 0.0
        self._daily_loss_limit = DEFAULT_DAILY_LOSS_LIMIT
        self._daily_loss_limit_hit = False
        self._last_price = 0.0
        self._last_ts = 0.0
        self._live_trade_count = 0
        self._shutting_down = False

        # Tuning
        self._tuning = dict(TUNING_DEFAULTS)
        self._tuning_mtime = 0.0

        # Reports
        self._reports_dir = os.path.join('reports', 'live')
        os.makedirs(self._reports_dir, exist_ok=True)

    # ── Main Loop ──────────────────────────────────────────────────────

    async def run(self):
        """Main entry point."""
        from core.keep_awake import keep_awake

        logger.info("=" * 60)
        logger.info("LIVE ENGINE — BLENDED + 3 CNNs")
        logger.info(f"  Instrument: {self._cfg.instrument}")
        logger.info(f"  Account:    {self._cfg.account}")
        logger.info(f"  Daily loss: ${self._daily_loss_limit:.0f}")
        logger.info("=" * 60)

        connected = await self._client.connect()
        if not connected:
            logger.error("Failed to connect to NT8 — exiting")
            return

        with keep_awake(display=True):
            try:
                await self._main_loop()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — shutting down")
            except Exception as e:
                logger.error(f"Fatal error: {e}", exc_info=True)
            finally:
                await self._shutdown()

    async def _main_loop(self):
        """Dispatch NT8 messages."""
        while not self._shutting_down:
            # Check GUI shutdown request
            if self._shared_state.get('shutdown') or self._shared_state.get('shutdown_flatten'):
                if self._position_open:
                    await self._close_position('gui_shutdown')
                self._shutting_down = True
                break

            try:
                msg = await asyncio.wait_for(
                    self._client.inbound.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue

            msg_type = msg.get('type', '')

            if msg_type == MsgType.BAR:
                await self._on_bar(msg)
            elif msg_type == MsgType.HISTORY_DONE:
                self._history_done = True
                self._system_ready = True
                logger.info(f'History done. {self._bar_count} bars. System READY.')
            elif msg_type == MsgType.FILL:
                self._orders.on_fill(msg)
            elif msg_type == MsgType.ORDER_STATUS:
                self._orders.on_order_status(msg)
            elif msg_type == MsgType.POSITION:
                self._on_position(msg)
            elif msg_type == MsgType.ACCOUNT_UPDATE:
                self._on_account_update(msg)

    # ── Bar Processing ─────────────────────────────────────────────────

    async def _on_bar(self, msg: dict):
        """Process one 1s bar."""
        bar = {
            'timestamp': msg.get('timestamp', 0),
            'open': msg.get('open', 0),
            'high': msg.get('high', 0),
            'low': msg.get('low', 0),
            'close': msg.get('close', 0),
            'volume': msg.get('volume', 0),
        }

        self._bar_count += 1
        self._last_price = bar['close']
        self._last_ts = bar['timestamp']
        self._agg.feed(bar)

        # Log trade path if in position
        if self._position_open:
            self._trade_logger.log_bar(bar)

        # Buffer prices for post-trade regret (keep last 1000 bars)
        self._regret_buffer.append(bar['close'])
        if len(self._regret_buffer) > 5000:
            self._regret_buffer = self._regret_buffer[-5000:]

        # Per-bar: push tick + unrealized PnL to dashboard
        unrealized = 0.0
        if self._engine.in_pos:
            if self._engine.direction == 'long':
                unrealized = (bar['close'] - self._engine.entry_price) / 0.25 * 0.50
            else:
                unrealized = (self._engine.entry_price - bar['close']) / 0.25 * 0.50
        self._gui.push({
            'type': 'TICK_UPDATE',
            'price': bar['close'],
            'bars': self._bar_count,
            'unrealized': unrealized,
            'daily_pnl': self._daily_pnl,
            'in_position': self._engine.in_pos,
            'direction': self._engine.direction or '',
            'tier': self._engine.entry_tier or '',
        })

        # Status line every 60 bars
        if self._bar_count % 60 == 0:
            self._print_status(bar)

        # Hot-reload tuning every ~5 min (300 bars)
        if self._bar_count % 300 == 0:
            self._load_tuning()

        # Only compute 79D at 1m boundary
        ts = bar['timestamp']
        if (int(ts) % 60) >= 5:
            return

        # Build 79D
        states_by_tf = {}
        ohlcv_by_tf = {}
        for tf in TF_ORDER:
            df = self._agg.get_closed_bars_df(tf)
            partial = self._agg.get_partial_bar(tf)
            if partial is not None:
                partial_df = pd.DataFrame([partial])
                full_df = pd.concat([df, partial_df], ignore_index=True) if len(df) > 0 else partial_df
            else:
                full_df = df
            if len(full_df) < SFE_MIN_BARS:
                continue
            ohlcv_by_tf[tf] = full_df
            sfe_input = full_df.tail(300).reset_index(drop=True) if len(full_df) > 300 else full_df
            states = self._sfe.batch_compute_states(sfe_input)
            if states:
                states_by_tf[tf] = states[-1]

        if '1m' not in states_by_tf:
            return

        feat, self._prev_velocities = extract_79d(
            states_by_tf, ohlcv_by_tf, self._prev_velocities, ts)

        # Warmup check
        if not self._warmed_up:
            if self._bar_count < self._cfg.warmup_bars:
                return
            self._warmed_up = True
            logger.info(f'Warmed up after {self._bar_count} bars')

        # Don't trade during history replay
        if not self._system_ready:
            return

        # Daily loss limit
        if self._daily_loss_limit_hit:
            return

        # Feed BlendedEngine
        state = {
            'features_79d': feat,
            'price': bar['close'],
            'timestamp': ts,
        }

        prev_in_pos = self._engine.in_pos
        prev_n_trades = len(self._engine.trades)

        self._engine.on_state(state)

        # Detect trade events
        new_trade = len(self._engine.trades) > prev_n_trades
        entered = self._engine.in_pos and not prev_in_pos
        exited = not self._engine.in_pos and prev_in_pos

        if entered and not self._closing_position:
            await self._send_entry(
                self._engine.direction, self._engine.entry_tier)

        if exited and new_trade:
            t = self._engine.trades[-1]
            await self._on_trade_closed(t, bar)

    # ── Trade Execution ────────────────────────────────────────────────

    async def _send_entry(self, direction: str, tier: str):
        """Send entry order to NT8."""
        side = 'BUY' if direction == 'long' else 'SELL'
        order_id = f'blended_{tier}_{uuid.uuid4().hex[:8]}'

        msg = place_order(
            order_id=order_id,
            instrument=self._cfg.instrument,
            account=self._cfg.account,
            side=side,
            qty=1,
        )
        await self._client.send(msg)
        self._orders.on_order_sent(order_id, side, 1)
        self._position_open = True
        self._trade_logger.start_trade(side, self._last_price, self._last_ts)

        logger.info(f'ENTRY: {side} | tier={tier} | price={self._last_price:.2f}')
        self._gui.push({'type': 'TRADE_MARKER', 'side': side,
                        'price': self._last_price, 'action': 'ENTRY'})

    async def _close_position(self, reason: str):
        """Close current position."""
        if not self._position_open or self._closing_position:
            return
        self._closing_position = True
        msg = close_position(
            instrument=self._cfg.instrument,
            account=self._cfg.account,
        )
        await self._client.send(msg)
        logger.info(f'CLOSE: reason={reason}')

    async def _on_trade_closed(self, trade: dict, bar: dict):
        """Handle trade completion."""
        pnl = trade['pnl']
        self._daily_pnl += pnl
        self._live_trade_count += 1
        self._position_open = False
        self._closing_position = False

        # Session tracker
        self._session.record_trade(pnl, trade.get('peak', 0))

        # Trade logger
        self._trade_logger.end_trade(
            pnl, trade.get('exit_reason', ''), self._last_price)

        # Post-trade regret analysis + accumulate for live brain
        if len(self._regret_buffer) > 50:
            closes = np.array(self._regret_buffer)
            entry_idx = max(0, len(closes) - trade.get('held', 10) - 1)
            try:
                r = compute_regret(trade, closes, entry_idx)
                regret_pnl = r['best_pnl'] - r['actual_pnl']
                logger.info(f'  REGRET: best={r["best_action"]} ${r["best_pnl"]:.0f} '
                           f'(left ${regret_pnl:.0f} on table)')

                # Accumulate for daily regret CSV
                self._regret_log.append({
                    'timestamp': self._last_ts,
                    'dir': trade.get('dir', ''),
                    'tier': trade.get('entry_tier', ''),
                    'exit_reason': trade.get('exit_reason', ''),
                    'pnl': pnl,
                    'best_action': r['best_action'],
                    'best_pnl': r['best_pnl'],
                    'regret': regret_pnl,
                })

                # Accumulate full trade for brain retraining
                self._live_trades_for_brain.append({
                    **trade,
                    'regret': r,
                })
            except Exception as e:
                logger.warning(f'Regret computation failed: {e}')

        # Close NT8 position
        await self._close_position(trade.get('exit_reason', 'cnn'))

        # Log
        logger.info(
            f'TRADE: {trade["dir"]} | tier={trade.get("entry_tier", "?")} | '
            f'exit={trade.get("exit_reason", "?")} | pnl=${pnl:.1f} | '
            f'peak=${trade.get("peak", 0):.1f} | daily=${self._daily_pnl:.0f} | '
            f'#{self._live_trade_count}')

        # Dashboard
        self._gui.push({
            'type': 'TRADE',
            'dir': trade['dir'],
            'tier': trade.get('entry_tier', '?'),
            'exit': trade.get('exit_reason', '?'),
            'pnl': pnl,
            'daily_pnl': self._daily_pnl,
            'n_trades': self._live_trade_count,
        })

        # Daily loss limit check
        if self._daily_pnl <= -self._daily_loss_limit:
            self._daily_loss_limit_hit = True
            logger.warning(f'DAILY LOSS LIMIT HIT: ${self._daily_pnl:.0f} '
                          f'(limit=${self._daily_loss_limit:.0f})')
            self._gui.push({'type': 'LOSS_LIMIT', 'locked': True,
                           'daily_pnl': self._daily_pnl})

    # ── Position & Account ─────────────────────────────────────────────

    def _on_position(self, msg: dict):
        """Sync position state from NT8."""
        qty = msg.get('quantity', 0)
        if qty == 0 and self._position_open:
            # NT8 says flat but we think we're in — sync
            self._position_open = False
            self._closing_position = False

    def _on_account_update(self, msg: dict):
        """Track account equity from NT8."""
        self._gui.push({
            'type': 'ACCOUNT_UPDATE',
            'cash': msg.get('cash_value', 0),
            'unrealized': msg.get('unrealized_pnl', 0),
            'net_liq': msg.get('net_liquidation', 0),
        })

    # ── Status & Monitoring ────────────────────────────────────────────

    def _print_status(self, bar: dict):
        """Print console status line."""
        from datetime import datetime
        pos = self._engine.direction or 'FLAT'
        pnl_str = ''
        if self._engine.in_pos:
            if self._engine.direction == 'long':
                unrealized = (bar['close'] - self._engine.entry_price) / 0.25 * 0.50
            else:
                unrealized = (self._engine.entry_price - bar['close']) / 0.25 * 0.50
            pnl_str = f'unrl=${unrealized:>+.0f} pk=${self._engine.peak_pnl:>.0f}'
        tier = self._engine.entry_tier or '-'
        t_str = datetime.utcfromtimestamp(bar['timestamp']).strftime('%H:%M:%S')
        print(f'\r  {t_str} | {bar["close"]:>10.2f} | {pos:>5} {tier:>10} | '
              f'tr={self._live_trade_count} day=${self._daily_pnl:>+.0f} {pnl_str}    ',
              end='', flush=True)

        # Push to GUI
        self._gui.push_tick(bar['close'], self._bar_count)

        # Push stats to dashboard
        wins = sum(1 for t in self._engine.trades if t['pnl'] > 0)
        gross_win = sum(t['pnl'] for t in self._engine.trades if t['pnl'] > 0)
        gross_loss = sum(t['pnl'] for t in self._engine.trades if t['pnl'] <= 0)
        self._gui.push_stats(
            session_pnl=self._daily_pnl,
            session_wins=wins,
            session_trades=self._live_trade_count,
            gross_win=gross_win,
            gross_loss=gross_loss,
            exit_buckets={'reversed': 0, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0, 'q100plus': 0},
            belief_pct=0,
            in_position=self._engine.in_pos,
            daily_pnl=self._daily_pnl,
        )

    # ── Tuning Hot Reload ──────────────────────────────────────────────

    def _load_tuning(self, force=False):
        """Hot-reload tuning from JSON file."""
        if not os.path.exists(TUNING_FILE):
            return
        try:
            mtime = os.path.getmtime(TUNING_FILE)
            if mtime == self._tuning_mtime and not force:
                return
            with open(TUNING_FILE, 'r') as f:
                new_tuning = json.load(f)
            self._tuning_mtime = mtime
            self._tuning.update(new_tuning)
            self._daily_loss_limit = self._tuning.get('daily_loss_limit',
                                                       DEFAULT_DAILY_LOSS_LIMIT)
            logger.info(f'Tuning reloaded: {new_tuning}')
        except Exception as e:
            logger.warning(f'Tuning load failed: {e}')

    # ── Shutdown ───────────────────────────────────────────────────────

    async def _shutdown(self):
        """Graceful shutdown: flatten, save regret, retrain brain, disconnect."""
        self._shutting_down = True

        # Flatten if in position
        if self._position_open:
            logger.info('Shutdown: closing open position')
            await self._close_position('shutdown')
            await asyncio.sleep(2.0)  # wait for fill

        # Force close engine
        self._engine.force_close()

        # Session report
        try:
            report_path = self._session.write_report(
                gate_stats={}, brain_dir_bias={},
                account_snapshot={},
                bar_count=self._bar_count)
            logger.info(f'Session report saved: {report_path}')
        except Exception as e:
            logger.warning(f'Session report failed: {e}')
            logger.info(f'Session: {self._live_trade_count} trades, ${self._daily_pnl:.0f}')

        # Save daily regret CSV
        if self._regret_log:
            import pandas as pd
            date_str = time.strftime('%Y%m%d')
            regret_path = os.path.join(self._reports_dir, f'regret_{date_str}.csv')
            pd.DataFrame(self._regret_log).to_csv(regret_path, index=False)
            logger.info(f'Regret log saved: {regret_path} ({len(self._regret_log)} trades)')

        # Retrain live brain from accumulated trades
        if self._live_trades_for_brain:
            self._retrain_live_brain()

        # Disconnect
        await self._client.disconnect()
        logger.info('Shutdown complete')
        self._shared_state['shutdown_confirmed'] = True

    def _retrain_live_brain(self):
        """Retrain CNNs on live trades → save as live brain (separate from backtest)."""
        import pickle

        n_trades = len(self._live_trades_for_brain)
        logger.info(f'Retraining live brain from {n_trades} trades...')

        if n_trades < 10:
            logger.info('Too few trades for retraining — skipping')
            return

        # Save raw live trades for offline analysis
        trades_path = os.path.join(self._live_brain_dir, 'live_trades.pkl')
        with open(trades_path, 'wb') as f:
            pickle.dump(self._live_trades_for_brain, f)
        logger.info(f'Live trades saved: {trades_path}')

        # For now: save trades only. Full retraining requires more data.
        # The maintenance script can retrain from accumulated live trades.
        # This prevents slow shutdown from GPU training.
        logger.info('Live brain retraining deferred to maintenance window')
        logger.info(f'  Run: python -m live.maintenance --retrain')
