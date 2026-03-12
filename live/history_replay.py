"""
Compressed History Replay Engine.

Replays ATLAS parquet data through the SAME per-bar compressed path that
live trading uses. No pre-computed discovery — just MarketState features
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
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.bayesian_brain import MarketBayesianBrain, record_trade
from core.execution_engine import (ActionType, Candidate, ExecutionEngine,
                                   TradeAction)
from core.exit_engine import ExitEngine
from core.feature_extraction import extract_feature_vector
from core.fractal_clustering import FractalClusteringEngine
from core.statistical_field_engine import StatisticalFieldEngine
from core.timeframe_belief_network import TimeframeBeliefNetwork
from live.atlas_loader import load_multi_tf, split_trading_days, _slice_day

# TF seconds for feature extraction
_TF_SECS = {'1s': 1, '5s': 5, '15s': 15, '30s': 30, '1m': 60, '2m': 120,
             '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600}


def _compressed_features(state, tf_seconds: int, depth: int) -> np.ndarray:
    """Build 16D feature vector from MarketState (no parent chain).

    Identical to live_engine._live_features() — single source of truth
    for the compressed per-bar path used by live, replay, and OOS.
    """
    return np.array([extract_feature_vector(
        z_score=getattr(state, 'z_score', 0.0),
        velocity=getattr(state, 'velocity', 0.0),
        momentum=getattr(state, 'momentum_strength',
                         getattr(state, 'momentum', 0.0)),
        entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
        tf_seconds=tf_seconds,
        depth=float(depth),
        parent_is_band_reversal=0.0,
        adx=getattr(state, 'adx_strength', 0.0) / 100.0,
        hurst=getattr(state, 'hurst_exponent', 0.5),
        dmi_diff=(getattr(state, 'dmi_plus', 0.0)
                  - getattr(state, 'dmi_minus', 0.0)) / 100.0,
        parent_z=0.0, parent_dmi_diff=0.0,
        root_is_roche=0.0, tf_alignment=0.0,
        pid=getattr(state, 'term_pid', 0.0),
        osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0),
    )])


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
        self._tf_seconds = _TF_SECS.get(anchor_tf, 15)

        # Loaded in _load_checkpoints
        self.pattern_library = {}
        self.scaler = None
        self.valid_tids = []
        self.centroids_scaled = None
        self.brain = MarketBayesianBrain()
        self.template_tier_map = {}
        self.depth_score_adj = {}
        self.depth_filter_out = set()
        self.exception_tids = set()

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

        # 4. Initialize components
        engine = StatisticalFieldEngine()

        exit_engine = ExitEngine(
            mode='training',
            tick_size=0.25,
            tick_value=0.25 * 2.0,
        )

        belief_network = TimeframeBeliefNetwork(
            pattern_library=self.pattern_library,
            scaler=self.scaler,
            engine=engine,
            valid_tids=self.valid_tids,
            centroids_scaled=self.centroids_scaled,
        )

        exec_engine = ExecutionEngine(
            brain=self.brain,
            belief_network=belief_network,
            exit_engine=exit_engine,
            pattern_library=self.pattern_library,
            scaler=self.scaler,
            centroids_scaled=self.centroids_scaled,
            valid_tids=self.valid_tids,
            tick_size=0.25,
            point_value=2.0,
            mode='replay',
            tier_score_adj={},
            depth_score_adj=self.depth_score_adj,
            template_tier_map=self.template_tier_map,
            exception_tids=self.exception_tids,
            depth_filter_out=self.depth_filter_out,
            feature_extractor=FractalClusteringEngine.extract_features,
        )

        # Match live engine's aggression-scaled gate1_dist
        _g1 = 4.5 + self.aggression * 10.0
        exec_engine.gate1_dist = _g1
        print(f"  gate1_dist={_g1:.1f} (aggression={self.aggression})")

        # 5a. Context warmup — TBN state only, no trading
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

        # 5b. Trading days — compressed per-bar (same as live)
        all_trades = []
        last_day_states = []
        for i, day_df in enumerate(trade_days_list):
            day_trades, day_states = self._replay_day(
                day_df, tf_data, engine, exec_engine, belief_network)
            all_trades.extend(day_trades)
            last_day_states = day_states
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

    def _replay_day(self, day_df, tf_data, engine, exec_engine, tbn):
        """Compressed per-bar forward pass — identical to live trading path.

        1. batch_compute_states (all 15s bars)
        2. TBN prepare_day
        3. Per-bar: build Candidate from MarketState features (no discovery)
        4. ExecutionEngine gate cascade
        5. Brain learning from outcomes
        """
        states = engine.batch_compute_states(day_df, use_cuda=True)
        if not states:
            return [], []

        rp = engine.regression_period

        # Prepare TBN
        df_5s = _slice_day(tf_data.get('5s'), day_df)
        df_4h = _slice_day(tf_data.get('4h'), day_df)
        tbn.prepare_day(day_df, states_micro=states,
                        df_5s=df_5s, df_4h=df_4h)

        trades = []
        _current_entry = None
        tick_size = 0.25
        point_value = 2.0

        for bar_i, result in enumerate(states):
            state = result['state']
            row_idx = bar_i + rp
            price = float(day_df.iloc[row_idx]['close'])
            bar_high = float(day_df.iloc[row_idx]['high'])
            bar_low = float(day_df.iloc[row_idx]['low'])
            ts = float(day_df.iloc[row_idx]['timestamp'])

            tbn.tick_all(bar_i)

            # Build candidates from MarketState (compressed path — same as live)
            candidates = []
            if not exec_engine.in_position:
                _pt = getattr(state, 'pattern_type', '')
                _z = getattr(state, 'z_score', 0.0)
                _cascade = getattr(state, 'cascade_detected', False)
                _struct = getattr(state, 'structure_confirmed', False)

                if _pt and _pt != 'NONE' and (_cascade or _struct):
                    _feat = _compressed_features(
                        state, self._tf_seconds, self.anchor_depth)
                    candidates.append(Candidate(
                        state=state,
                        depth=self.anchor_depth,
                        timeframe=self.anchor_tf,
                        timestamp=ts,
                        pattern_type=_pt,
                        z_score=_z,
                        features=_feat,
                    ))

            # Exit signal from TBN
            _exit_sig = None
            if exec_engine.in_position:
                _exit_sig = tbn.get_exit_signal(
                    side=exec_engine.active_side,
                    entry_price=exec_engine.entry_price,
                )

            # ExecutionEngine
            action = exec_engine.on_bar(
                price=price,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_index=bar_i,
                candidates=candidates if candidates else None,
                exit_signal=_exit_sig,
            )

            if action.type == ActionType.ENTER:
                _lib_entry = self.pattern_library.get(action.template_id, {})
                exec_engine.position_opened(
                    side=action.side,
                    price=action.price,
                    bar_index=bar_i,
                    template_id=action.template_id,
                    lib_entry=_lib_entry,
                    sl_ticks=action.sl_ticks,
                    tp_ticks=action.tp_ticks,
                    max_hold_bars=getattr(action, 'max_hold_bars', 960),
                )
                _current_entry = {
                    'side': action.side,
                    'entry_price': price,
                    'entry_bar': bar_i,
                    'tid': action.template_id,
                    'dir_source': getattr(action, 'dir_source', 'unknown'),
                }

            elif action.type == ActionType.EXIT and _current_entry is not None:
                pnl_ticks = action.pnl_ticks if hasattr(action, 'pnl_ticks') else 0
                pnl_dollars = pnl_ticks * tick_size * point_value

                trades.append({
                    **_current_entry,
                    'exit_price': price,
                    'exit_bar': bar_i,
                    'pnl': pnl_dollars,
                    'pnl_ticks': pnl_ticks,
                    'exit_reason': getattr(action, 'exit_reason', 'unknown'),
                    'bars_held': bar_i - _current_entry['entry_bar'],
                })

                record_trade(
                    self.brain,
                    tid=_current_entry['tid'],
                    entry_price=_current_entry['entry_price'],
                    exit_price=price,
                    pnl=pnl_dollars,
                    side=_current_entry['side'],
                    exit_reason=getattr(action, 'exit_reason', 'unknown'),
                    timestamp=ts,
                    tick_value=tick_size * point_value,
                    hold_bars=bar_i - _current_entry['entry_bar'],
                )

                exec_engine.position_closed()
                _current_entry = None

        return trades, states

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
            warnings.append("No OOS reference available — cannot validate")
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
        lines.append("  OOS vs LIVE REPLAY — PARITY REPORT")
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
        """Load all training checkpoints."""
        cpdir = self.checkpoint_dir
        print(f"  Loading checkpoints from {cpdir}/")

        lib_path = os.path.join(cpdir, 'pattern_library.pkl')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Missing pattern_library.pkl in {cpdir}")
        with open(lib_path, 'rb') as f:
            self.pattern_library = pickle.load(f)
        print(f"  Library: {len(self.pattern_library)} templates")

        scaler_path = os.path.join(cpdir, 'clustering_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            from sklearn.preprocessing import StandardScaler
            _cents = [v['centroid'] for v in self.pattern_library.values()
                      if 'centroid' in v]
            self.scaler = StandardScaler().fit(np.array(_cents))

        self.valid_tids = [
            tid for tid in self.pattern_library
            if 'centroid' in self.pattern_library[tid]
        ]

        centroids = np.array([
            self.pattern_library[tid]['centroid']
            for tid in self.valid_tids
        ])
        self.centroids_scaled = self.scaler.transform(centroids)

        tiers_path = os.path.join(cpdir, 'template_tiers.pkl')
        if os.path.exists(tiers_path):
            with open(tiers_path, 'rb') as f:
                self.template_tier_map = pickle.load(f)

        dw_path = os.path.join(cpdir, 'depth_weights.json')
        if os.path.exists(dw_path):
            with open(dw_path) as f:
                dw_data = json.load(f)
            self.depth_score_adj = {
                int(k): float(v.get('score_adj', 0.0))
                for k, v in dw_data.items()
            }
            self.depth_filter_out = {
                int(k) for k, v in dw_data.items()
                if v.get('filter_out', False)
            }

        for tid in self.valid_tids:
            lib = self.pattern_library.get(tid, {})
            if (lib.get('member_count', 0) >= 10
                    and lib.get('stats_win_rate', 0.0) >= 0.55
                    and (lib.get('regression_sigma_ticks') or 999) <= 10.0):
                self.exception_tids.add(tid)

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

        print(f"  Centroids: {len(self.valid_tids)} ready")
        print(f"  Exception templates: {len(self.exception_tids)}")


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
