"""
Compressed History Replay Engine.

Replays ATLAS parquet data through the SAME per-bar compressed path that
live trading uses. No pre-computed discovery  -- just MarketState features
matched against library centroids, identical to live_engine.py.

This ensures replay/live/OOS parity.

Usage:
    replay = HistoryReplayEngine(checkpoint_dir='checkpoints', n_days=5, context_days=21)
    result = replay.run()
    # result.brain, result.belief_network, result.exit_engine, etc.
"""

import glob
import json
import os
import time
from dataclasses import dataclass, field

import pandas as pd

from core.advance_engine import AdvanceEngine, AdvanceEngineHooks
from core.bayesian_brain import MarketBayesianBrain
from core.checkpoint_loader import load_checkpoints
from core.engine_factory import create_belief_network, create_execution_engine
from core.execution_engine import ExecutionEngine
from core.exit_engine import ExitEngine
from core.statistical_field_engine import StatisticalFieldEngine
from core.timeframe_belief_network import TimeframeBeliefNetwork
from live.atlas_loader import load_multi_tf, split_trading_days, _slice_day


@dataclass
class ValidationReport:
    replay_trades: int
    replay_wr: float
    replay_pnl: float
    replay_avg_trade: float
    oos_trades: int = 0
    oos_wr: float = 0.0
    oos_pnl: float = 0.0
    oos_avg_trade: float = 0.0
    parity_score: float = 0.5
    gate_stats: dict = field(default_factory=dict)
    direction_source_dist: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    passed: bool = False


@dataclass
class ReplayResult:
    brain: MarketBayesianBrain
    belief_network: TimeframeBeliefNetwork
    exit_engine: ExitEngine
    execution_engine: ExecutionEngine
    last_timestamp: float
    validation: ValidationReport
    states_micro: list
    df_micro: pd.DataFrame
    trades: list = field(default_factory=list)


class HistoryReplayEngine:
    """Compressed per-bar forward pass over historical ATLAS data.

    Uses the SAME compressed path as live trading:
      - MarketState features (no parent chain, no discovery)
      - Centroid matching with live-equivalent gate1_dist
      - Brain learning from trade outcomes

    Produces fully warmed state for live handoff.
    """

    # Anchor depth mapping (same as live_engine)
    _TF_DEPTH = {'1s': 10, '5s': 9, '15s': 8, '30s': 7, '1m': 6,
                 '3m': 5, '5m': 4, '15m': 3, '30m': 2, '1h': 1}

    def __init__(self, checkpoint_dir: str = 'checkpoints',
                 n_days: int = 5, context_days: int = 0,
                 atlas_root: str = 'DATA/ATLAS',
                 anchor_tf: str = '15s', validate_against_oos: bool = True,
                 aggression: float = 0.5):
        self.checkpoint_dir = checkpoint_dir
        self.n_days = n_days
        self.context_days = context_days
        self.atlas_root = atlas_root
        self.anchor_tf = anchor_tf
        self.validate_against_oos = validate_against_oos
        self.aggression = aggression

        self.anchor_depth = self._TF_DEPTH.get(anchor_tf, 8)

        # Loaded in _load_checkpoints (via shared checkpoint_loader)
        self._bundle = None
        self.brain = MarketBayesianBrain()

    def run(self) -> ReplayResult:
        """Execute compressed replay and return warmed state."""
        t0 = time.perf_counter()
        print("\n" + "=" * 72)
        print("  COMPRESSED HISTORY REPLAY")
        print("=" * 72)

        # 1. Load checkpoints
        self._load_checkpoints()

        # 2. Load ATLAS data
        total_days = self.n_days + self.context_days
        print(f"\n  Loading {total_days} days of ATLAS data "
              f"({self.context_days} context + {self.n_days} trading)...")
        tf_data = load_multi_tf(self.atlas_root, total_days)
        df_15s = tf_data['15s']

        # 3. Split into trading days
        all_days = split_trading_days(df_15s)
        print(f"  Split into {len(all_days)} trading days")

        if self.context_days > 0 and len(all_days) > self.n_days:
            n_trade = min(self.n_days, len(all_days))
            context_days_list = all_days[:-n_trade]
            trade_days_list = all_days[-n_trade:]
        else:
            context_days_list = []
            trade_days_list = all_days

        # 4. Initialize components  -- SAME factory as trainer (parity)
        engine = StatisticalFieldEngine()

        exit_engine = ExitEngine(
            mode='training',
            tick_size=0.25,
            tick_value=0.25 * 2.0,
        )

        belief_network = create_belief_network(self._bundle, engine)

        exec_engine = create_execution_engine(
            bundle=self._bundle,
            brain=self.brain,
            belief_network=belief_network,
            exit_engine=exit_engine,
            tick_size=0.25,
            point_value=2.0,
            mode='oos',
            tier_preference=True,
        )

        # OOS compressed mode: widen gate1_dist (same as trainer OOS)
        _g1 = 4.5 + self.aggression * 10.0
        exec_engine.gate1_dist = _g1
        print(f"  gate1_dist={_g1:.1f} (aggression={self.aggression})")

        # 4b. AdvanceEngine  -- shared per-bar decision loop
        all_trades = []

        def _on_exit(trade, outcome):
            all_trades.append(trade)

        processor = AdvanceEngine(
            exec_engine=exec_engine,
            belief_network=belief_network,
            exit_engine=exit_engine,
            brain=self.brain,
            pattern_library=self._bundle.pattern_library,
            anchor_tf=self.anchor_tf,
            anchor_depth=self.anchor_depth,
            hooks=AdvanceEngineHooks(on_exit=_on_exit),
        )

        # 5a. Context warmup  -- TBN state only, no trading
        if context_days_list:
            print(f"  Context warmup: {len(context_days_list)} days")
        for i, day_df in enumerate(context_days_list):
            states = engine.batch_compute_states(day_df, use_cuda=True)
            if states:
                df_5s = _slice_day(tf_data.get('5s'), day_df)
                df_4h = _slice_day(tf_data.get('4h'), day_df)
                belief_network.prepare_day(day_df, states_micro=states,
                                           df_5s=df_5s, df_4h=df_4h)
                for bar_i in range(len(states)):
                    belief_network.tick_all(bar_i)
            print(f"    Context {i+1}/{len(context_days_list)}: "
                  f"{len(states) if states else 0} bars")

        # 5b. Trading days  -- shared AdvanceEngine (same as trainer OOS)
        last_day_states = []
        for i, day_df in enumerate(trade_days_list):
            day_start = len(all_trades)
            day_states = self._replay_day(day_df, tf_data, engine, processor)
            last_day_states = day_states

            day_trades = all_trades[day_start:]
            n_w = sum(1 for t in day_trades if t['pnl'] > 0)
            n_t = len(day_trades)
            pnl = sum(t['pnl'] for t in day_trades)
            wr = n_w / n_t * 100 if n_t > 0 else 0
            print(f"    Day {i+1}/{len(trade_days_list)}: {n_t} trades, "
                  f"{wr:.0f}% WR, ${pnl:+.2f}")

        # 6. Validation
        report = self._build_validation(all_trades, exec_engine)

        elapsed = time.perf_counter() - t0
        print(f"\n  Replay complete: {len(all_trades)} trades in {elapsed:.1f}s")
        print(f"  WR={report.replay_wr:.1%}  PnL=${report.replay_pnl:+,.2f}  "
              f"Avg=${report.replay_avg_trade:+.2f}")
        if report.warnings:
            for w in report.warnings:
                print(f"  WARNING: {w}")
        print(f"  Parity score: {report.parity_score:.2f} "
              f"({'PASSED' if report.passed else 'FAILED'})")
        print("=" * 72)

        # Write parity report to disk
        self.write_parity_report(all_trades, report)

        return ReplayResult(
            brain=self.brain,
            belief_network=belief_network,
            exit_engine=exit_engine,
            execution_engine=exec_engine,
            last_timestamp=float(df_15s['timestamp'].iloc[-1]),
            validation=report,
            states_micro=last_day_states,
            df_micro=trade_days_list[-1] if trade_days_list else pd.DataFrame(),
            trades=all_trades,
        )

    def _replay_day(self, day_df, tf_data, engine, processor):
        """Compressed per-bar forward pass via shared AdvanceEngine."""
        states = engine.batch_compute_states(day_df, use_cuda=True)
        if not states:
            return []

        rp = engine.regression_period

        # Prepare TBN for this day
        df_5s = _slice_day(tf_data.get('5s'), day_df)
        df_4h = _slice_day(tf_data.get('4h'), day_df)
        processor.belief_network.prepare_day(
            day_df, states_micro=states, df_5s=df_5s, df_4h=df_4h)

        for bar_i, result in enumerate(states):
            row_idx = bar_i + rp
            row = day_df.iloc[row_idx]
            _cur_state = result['state']
            # Same state for entry + exit (matches inline OOS behavior)
            processor.process_bar(
                bar_index=bar_i,
                price=float(row['close']),
                bar_high=float(row['high']),
                bar_low=float(row['low']),
                timestamp=float(row['timestamp']),
                state=_cur_state,
            )

        # Force-close any open position at EOD
        if processor.in_position:
            last_row = day_df.iloc[-1]
            processor.force_close(
                price=float(last_row['close']),
                timestamp=float(last_row['timestamp']),
                bar_index=len(states) - 1,
            )

        return states

    # ── Validation ────────────────────────────────────────────────────────

    def _build_validation(self, trades, exec_engine) -> ValidationReport:
        """Compare replay results to OOS checkpoint numbers."""
        n = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        wr = wins / n if n > 0 else 0
        avg = total_pnl / n if n > 0 else 0

        dir_dist = {}
        for t in trades:
            src = t.get('dir_source', 'unknown')
            dir_dist[src] = dir_dist.get(src, 0) + 1

        oos = self._load_oos_reference()
        warnings = []
        parity = 1.0

        if oos and oos.get('trades', 0) > 0:
            wr_delta = abs(wr - oos['wr'])
            if wr_delta > 0.05:
                warnings.append(
                    f"WR diverged: replay={wr:.1%} vs OOS={oos['wr']:.1%}")
                parity -= min(0.3, wr_delta * 3)

            if oos.get('avg_trade', 0) > 0:
                avg_ratio = avg / oos['avg_trade']
                if avg_ratio < 0.5 or avg_ratio > 2.0:
                    warnings.append(
                        f"Avg trade diverged: ${avg:.2f} vs OOS ${oos['avg_trade']:.2f}")
                    parity -= 0.2

            replay_days = max(1, self.n_days)
            oos_days = max(1, oos.get('days', 1))
            replay_rate = n / replay_days
            oos_rate = oos['trades'] / oos_days
            if oos_rate > 0:
                rate_ratio = replay_rate / oos_rate
                if rate_ratio < 0.7 or rate_ratio > 1.3:
                    warnings.append(
                        f"Trade frequency diverged: {replay_rate:.1f}/day "
                        f"vs OOS {oos_rate:.1f}/day")
                    parity -= 0.15
        else:
            warnings.append("No OOS reference available  -- cannot validate")
            parity = 0.5

        parity = max(0.0, min(1.0, parity))

        return ValidationReport(
            replay_trades=n, replay_wr=wr,
            replay_pnl=total_pnl, replay_avg_trade=avg,
            oos_trades=oos.get('trades', 0) if oos else 0,
            oos_wr=oos.get('wr', 0) if oos else 0,
            oos_pnl=oos.get('pnl', 0) if oos else 0,
            oos_avg_trade=oos.get('avg_trade', 0) if oos else 0,
            parity_score=parity,
            gate_stats=exec_engine.get_skip_counts(),
            direction_source_dist=dir_dist,
            warnings=warnings,
            passed=parity >= 0.80,
        )

    def _load_oos_reference(self) -> dict:
        """Load OOS metrics from checkpoint for comparison."""
        snap_path = os.path.join(self.checkpoint_dir, 'run_snapshot.json')
        if os.path.exists(snap_path):
            try:
                with open(snap_path) as f:
                    data = json.load(f)
                oos = data.get('oos', {})
                if oos:
                    return {
                        'trades': oos.get('trades', 0),
                        'wr': oos.get('win_rate', 0),
                        'pnl': oos.get('total_pnl', 0),
                        'avg_trade': oos.get('avg_pnl', 0),
                        'days': oos.get('days', 1),
                    }
            except (json.JSONDecodeError, KeyError):
                pass

        log_path = os.path.join(self.checkpoint_dir, 'oos_trade_log.csv')
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                _pnl_col = 'pnl_dollars' if 'pnl_dollars' in df.columns else (
                    'actual_pnl' if 'actual_pnl' in df.columns else None)
                if len(df) > 0 and _pnl_col:
                    _days = 1
                    if 'entry_day' in df.columns:
                        _days = max(1, df['entry_day'].nunique())
                    elif 'entry_time' in df.columns:
                        _ts = pd.to_datetime(df['entry_time'], unit='s',
                                             errors='coerce')
                        _days = max(1, _ts.dt.date.nunique())
                    return {
                        'trades': len(df),
                        'wr': (df[_pnl_col] > 0).mean(),
                        'pnl': df[_pnl_col].sum(),
                        'avg_trade': df[_pnl_col].mean(),
                        'days': _days,
                    }
            except Exception:
                pass

        return None

    def write_parity_report(self, trades: list, report: ValidationReport,
                            out_dir: str = 'reports/live') -> str:
        """Write OOS vs Replay parity report to disk.

        Returns path to the written report file.
        """
        from datetime import datetime
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out_dir, f'parity_report_{ts}.txt')

        oos = self._load_oos_reference() or {}
        lines = []
        lines.append("=" * 72)
        lines.append("  OOS vs LIVE REPLAY  -- PARITY REPORT")
        lines.append("=" * 72)
        lines.append("")

        # ── Side-by-side summary ─────────────────────────────────────
        lines.append(f"  {'Metric':<30} {'OOS':>12} {'Replay':>12} {'Delta':>12}")
        lines.append("  " + "-" * 66)

        oos_trades = oos.get('trades', 0)
        oos_wr = oos.get('wr', 0)
        oos_pnl = oos.get('pnl', 0)
        oos_avg = oos.get('avg_trade', 0)
        oos_days = oos.get('days', 1)

        rep_n = report.replay_trades
        rep_wr = report.replay_wr
        rep_pnl = report.replay_pnl
        rep_avg = report.replay_avg_trade

        lines.append(f"  {'Trades':<30} {oos_trades:>12} {rep_n:>12} "
                     f"{rep_n - oos_trades:>+12}")
        lines.append(f"  {'Win Rate':<30} {oos_wr:>11.1%} {rep_wr:>11.1%} "
                     f"{(rep_wr - oos_wr) * 100:>+11.1f}%")
        lines.append(f"  {'Total PnL':<30} ${oos_pnl:>10,.2f} ${rep_pnl:>10,.2f} "
                     f"${rep_pnl - oos_pnl:>+10,.2f}")
        lines.append(f"  {'Avg Trade':<30} ${oos_avg:>10,.2f} ${rep_avg:>10,.2f} "
                     f"${rep_avg - oos_avg:>+10,.2f}")

        # Normalize to per-day
        rep_days = max(1, self.n_days)
        oos_rate = oos_trades / max(1, oos_days)
        rep_rate = rep_n / rep_days
        lines.append(f"  {'Trades/Day':<30} {oos_rate:>12.1f} {rep_rate:>12.1f} "
                     f"{rep_rate - oos_rate:>+12.1f}")
        oos_pnl_day = oos_pnl / max(1, oos_days)
        rep_pnl_day = rep_pnl / rep_days
        lines.append(f"  {'PnL/Day':<30} ${oos_pnl_day:>10,.2f} ${rep_pnl_day:>10,.2f} "
                     f"${rep_pnl_day - oos_pnl_day:>+10,.2f}")
        lines.append("")

        # ── Exit reason breakdown ────────────────────────────────────
        exit_dist = {}
        for t in trades:
            r = t.get('exit_reason', 'unknown')
            if r not in exit_dist:
                exit_dist[r] = {'n': 0, 'pnl': 0, 'wins': 0}
            exit_dist[r]['n'] += 1
            exit_dist[r]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                exit_dist[r]['wins'] += 1

        if exit_dist:
            lines.append("  EXIT REASON BREAKDOWN (Replay)")
            lines.append(f"  {'Reason':<25} {'Trades':>8} {'WR%':>8} {'PnL':>12} {'Avg':>10}")
            lines.append("  " + "-" * 63)
            for reason in sorted(exit_dist, key=lambda r: exit_dist[r]['pnl'], reverse=True):
                d = exit_dist[reason]
                wr_pct = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
                avg_t = d['pnl'] / d['n'] if d['n'] > 0 else 0
                lines.append(f"  {reason:<25} {d['n']:>8} {wr_pct:>7.1f}% "
                             f"${d['pnl']:>10,.2f} ${avg_t:>8,.2f}")
            lines.append("")

        # ── Direction breakdown ──────────────────────────────────────
        dir_stats = {}
        for t in trades:
            s = t.get('side', 'unknown')
            if s not in dir_stats:
                dir_stats[s] = {'n': 0, 'pnl': 0, 'wins': 0}
            dir_stats[s]['n'] += 1
            dir_stats[s]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                dir_stats[s]['wins'] += 1

        if dir_stats:
            lines.append("  DIRECTION BREAKDOWN (Replay)")
            lines.append(f"  {'Side':<12} {'Trades':>8} {'WR%':>8} {'PnL':>12} {'Avg':>10}")
            lines.append("  " + "-" * 50)
            for side in sorted(dir_stats):
                d = dir_stats[side]
                wr_pct = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
                avg_t = d['pnl'] / d['n'] if d['n'] > 0 else 0
                lines.append(f"  {side:<12} {d['n']:>8} {wr_pct:>7.1f}% "
                             f"${d['pnl']:>10,.2f} ${avg_t:>8,.2f}")
            lines.append("")

        # ── Gate rejection funnel ────────────────────────────────────
        gs = report.gate_stats
        if gs:
            lines.append("  GATE REJECTION FUNNEL")
            for gate, count in sorted(gs.items()):
                lines.append(f"    {gate}: {count:,}")
            lines.append("")

        # ── Direction source distribution ────────────────────────────
        ds = report.direction_source_dist
        if ds:
            lines.append("  DIRECTION SOURCE DISTRIBUTION")
            total_src = sum(ds.values())
            for src in sorted(ds, key=ds.get, reverse=True):
                pct = ds[src] / total_src * 100 if total_src > 0 else 0
                lines.append(f"    {src}: {ds[src]} ({pct:.1f}%)")
            lines.append("")

        # ── Parity verdict ───────────────────────────────────────────
        lines.append("  PARITY VERDICT")
        lines.append(f"    Score: {report.parity_score:.2f}")
        lines.append(f"    Status: {'PASSED' if report.passed else 'FAILED'}")
        if report.warnings:
            for w in report.warnings:
                lines.append(f"    WARNING: {w}")
        lines.append("")
        lines.append("=" * 72)

        text = "\n".join(lines)
        with open(path, 'w') as f:
            f.write(text)
        print(f"  Parity report: {path}")
        return path

    def _load_checkpoints(self):
        """Load checkpoints via shared loader (same as trainer)."""
        cpdir = self.checkpoint_dir
        print(f"  Loading checkpoints from {cpdir}/")

        # Same loader the trainer uses  -- single source of truth
        self._bundle = load_checkpoints(cpdir, verbose=True)

        # Brain: forward_pass (OOS weights) > training > live
        forward_brain = os.path.join(cpdir, 'pattern_forward_brain.pkl')
        training_brains = sorted(glob.glob(
            os.path.join(cpdir, 'pattern_*_brain.pkl')))

        if os.path.exists(forward_brain):
            self.brain.load(forward_brain)
            print(f"  Brain: pattern_forward_brain.pkl "
                  f"({len(self.brain.table)} states)")
        elif training_brains:
            self.brain.load(training_brains[-1])
            print(f"  Brain: {os.path.basename(training_brains[-1])}")
        else:
            print("  Brain: starting fresh")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='History replay (compressed)')
    parser.add_argument('--days', type=int, default=5, help='Trading days')
    parser.add_argument('--context', type=int, default=0,
                        help='Context warmup days (no trading)')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--atlas', default='DATA/ATLAS')
    parser.add_argument('--aggression', type=float, default=0.5,
                        help='Gate aggression 0-1 (matches live)')
    args = parser.parse_args()

    replay = HistoryReplayEngine(
        checkpoint_dir=args.checkpoint_dir,
        n_days=args.days,
        context_days=args.context,
        atlas_root=args.atlas,
        aggression=args.aggression,
    )
    result = replay.run()
    print(f"\nResult: {result.validation.replay_trades} trades, "
          f"parity={result.validation.parity_score:.2f}")
