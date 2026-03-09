"""
Compressed History Replay Engine.

Replays ATLAS parquet data through the full gate cascade to produce
a warmed brain, TBN, and exit engine for live trading handoff.

Usage:
    replay = HistoryReplayEngine(checkpoint_dir='checkpoints', n_days=5)
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

from core.bayesian_brain import MarketBayesianBrain
from core.execution_engine import (ActionType, Candidate, ExecutionEngine,
                                   TradeAction)
from core.exit_engine import ExitEngine
from core.fractal_clustering import FractalClusteringEngine
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


class HistoryReplayEngine:
    """Compressed forward pass over historical ATLAS data.

    Produces fully warmed state for live handoff.
    Uses ExecutionEngine (same as training/OOS) — one source of truth.
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints',
                 n_days: int = 5, atlas_root: str = 'DATA/ATLAS',
                 anchor_tf: str = '15s', validate_against_oos: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.n_days = n_days
        self.atlas_root = atlas_root
        self.anchor_tf = anchor_tf
        self.validate_against_oos = validate_against_oos

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

        # Anchor depth mapping
        _TF_DEPTH = {'1s': 10, '5s': 9, '15s': 8, '30s': 7, '1m': 6,
                     '3m': 5, '5m': 4, '15m': 3, '30m': 2, '1h': 1}
        self.anchor_depth = _TF_DEPTH.get(anchor_tf, 8)

    def run(self) -> ReplayResult:
        """Execute compressed replay and return warmed state."""
        t0 = time.perf_counter()
        print("\n" + "=" * 72)
        print("  COMPRESSED HISTORY REPLAY")
        print("=" * 72)

        # 1. Load checkpoints
        self._load_checkpoints()

        # 2. Load ATLAS data
        print(f"\n  Loading {self.n_days} days of ATLAS data...")
        tf_data = load_multi_tf(self.atlas_root, self.n_days)
        df_15s = tf_data['15s']

        # 3. Split into trading days
        days = split_trading_days(df_15s)
        print(f"  Split into {len(days)} trading days")

        # 4. Initialize components
        engine = StatisticalFieldEngine()

        exit_engine = ExitEngine(
            mode='training',
            tick_size=0.25,
            tick_value=0.25 * 2.0,  # MNQ: tick_size * point_value
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

        # 5. Forward pass per day
        all_trades = []
        last_day_states = []
        for i, day_df in enumerate(days):
            day_trades, day_states = self._replay_day(
                day_df, tf_data, engine, exec_engine, belief_network)
            all_trades.extend(day_trades)
            last_day_states = day_states
            n_w = sum(1 for t in day_trades if t['pnl'] > 0)
            n_t = len(day_trades)
            pnl = sum(t['pnl'] for t in day_trades)
            wr = n_w / n_t * 100 if n_t > 0 else 0
            print(f"    Day {i+1}/{len(days)}: {n_t} trades, "
                  f"{wr:.0f}% WR, ${pnl:+.2f}")

        # 6. Build validation report
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

        return ReplayResult(
            brain=self.brain,
            belief_network=belief_network,
            exit_engine=exit_engine,
            execution_engine=exec_engine,
            last_timestamp=float(df_15s['timestamp'].iloc[-1]),
            validation=report,
            states_micro=last_day_states,
            df_micro=days[-1] if days else pd.DataFrame(),
        )

    def _replay_day(self, day_df, tf_data, engine, exec_engine, tbn):
        """Compressed forward pass for one trading day.

        Mirrors trainer Phase 4 logic:
          1. batch_compute_states (all bars at once)
          2. TBN prepare_day (resample + compute TF states)
          3. Per-bar loop: TBN tick + ExecutionEngine.on_bar()
          4. Record trades for validation
        """
        # 1. Compute states (bulk)
        states = engine.batch_compute_states(day_df, use_cuda=True)
        if not states:
            return [], []

        rp = engine.regression_period

        # 2. Prepare TBN workers
        df_5s = _slice_day(tf_data.get('5s'), day_df)
        df_4h = _slice_day(tf_data.get('4h'), day_df)
        tbn.prepare_day(day_df, states_micro=states,
                        df_5s=df_5s, df_4h=df_4h)

        # 3. Per-bar loop
        trades = []
        _current_entry = None
        tick_size = 0.25
        point_value = 2.0

        for bar_i, result in enumerate(states):
            state = result['state']
            price = float(day_df.iloc[bar_i + rp]['close'])
            bar_high = float(day_df.iloc[bar_i + rp]['high'])
            bar_low = float(day_df.iloc[bar_i + rp]['low'])

            # Tick TBN
            tbn.tick_all(bar_i)

            # Build candidates
            candidates = []
            if not exec_engine.in_position:
                _pt = getattr(state, 'pattern_type', '')
                _z = getattr(state, 'z_score', 0.0)
                _cascade = getattr(state, 'cascade_detected', False)
                _struct = getattr(state, 'structure_confirmed', False)

                if _pt and (_cascade or _struct):
                    candidates.append(Candidate(
                        state=state,
                        depth=self.anchor_depth,
                        timeframe=self.anchor_tf,
                        timestamp=float(day_df.iloc[bar_i + rp]['timestamp']),
                        pattern_type=_pt,
                        z_score=_z,
                    ))

            # Get exit signal from TBN
            _exit_sig = None
            if exec_engine.in_position:
                _exit_sig = tbn.get_exit_signal(
                    side=exec_engine.active_side,
                    bars_held=bar_i - exec_engine.entry_bar if exec_engine.entry_bar else 0,
                )

            # ExecutionEngine handles everything
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

                # Brain learning
                self.brain.record(
                    _current_entry['tid'],
                    1 if pnl_dollars > 0 else 0,
                )
                self.brain.direction_learn(
                    _current_entry['tid'],
                    _current_entry['side'],
                    pnl_dollars,
                )

                exec_engine.position_closed()
                _current_entry = None

        return trades, states

    def _build_validation(self, trades, exec_engine) -> ValidationReport:
        """Compare replay results to OOS checkpoint numbers."""
        n = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        wr = wins / n if n > 0 else 0
        avg = total_pnl / n if n > 0 else 0

        # Direction source distribution
        dir_dist = {}
        for t in trades:
            src = t.get('dir_source', 'unknown')
            dir_dist[src] = dir_dist.get(src, 0) + 1

        # Load OOS reference
        oos = self._load_oos_reference()

        # Parity scoring
        warnings = []
        parity = 1.0

        if oos and oos.get('trades', 0) > 0:
            # WR within 5 percentage points
            wr_delta = abs(wr - oos['wr'])
            if wr_delta > 0.05:
                warnings.append(
                    f"WR diverged: replay={wr:.1%} vs OOS={oos['wr']:.1%}")
                parity -= min(0.3, wr_delta * 3)

            # Avg trade within 50%
            if oos.get('avg_trade', 0) > 0:
                avg_ratio = avg / oos['avg_trade']
                if avg_ratio < 0.5 or avg_ratio > 2.0:
                    warnings.append(
                        f"Avg trade diverged: ${avg:.2f} vs OOS ${oos['avg_trade']:.2f}")
                    parity -= 0.2

            # Trade frequency within 30%
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
            replay_trades=n,
            replay_wr=wr,
            replay_pnl=total_pnl,
            replay_avg_trade=avg,
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
        # Try run_snapshot.json first
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

        # Try oos_trade_log.csv
        log_path = os.path.join(self.checkpoint_dir, 'oos_trade_log.csv')
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                if len(df) > 0 and 'pnl_dollars' in df.columns:
                    return {
                        'trades': len(df),
                        'wr': (df['pnl_dollars'] > 0).mean(),
                        'pnl': df['pnl_dollars'].sum(),
                        'avg_trade': df['pnl_dollars'].mean(),
                        'days': max(1, df['entry_day'].nunique()
                                    if 'entry_day' in df.columns else 1),
                    }
            except Exception:
                pass

        return None

    def _load_checkpoints(self):
        """Load all training checkpoints (mirrors LiveEngine._load_checkpoints)."""
        cpdir = self.checkpoint_dir
        print(f"  Loading checkpoints from {cpdir}/")

        # Pattern library
        lib_path = os.path.join(cpdir, 'pattern_library.pkl')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Missing pattern_library.pkl in {cpdir}")
        with open(lib_path, 'rb') as f:
            self.pattern_library = pickle.load(f)
        print(f"  Library: {len(self.pattern_library)} templates")

        # Scaler
        scaler_path = os.path.join(cpdir, 'clustering_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            from sklearn.preprocessing import StandardScaler
            _cents = [v['centroid'] for v in self.pattern_library.values()
                      if 'centroid' in v]
            self.scaler = StandardScaler().fit(np.array(_cents))

        # Valid template IDs
        self.valid_tids = [
            tid for tid in self.pattern_library
            if 'centroid' in self.pattern_library[tid]
        ]

        # Centroids
        centroids = np.array([
            self.pattern_library[tid]['centroid']
            for tid in self.valid_tids
        ])
        self.centroids_scaled = self.scaler.transform(centroids)

        # Template tiers
        tiers_path = os.path.join(cpdir, 'template_tiers.pkl')
        if os.path.exists(tiers_path):
            with open(tiers_path, 'rb') as f:
                self.template_tier_map = pickle.load(f)

        # Depth weights
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

        # Exception templates
        for tid in self.valid_tids:
            lib = self.pattern_library.get(tid, {})
            if (lib.get('member_count', 0) >= 10
                    and lib.get('stats_win_rate', 0.0) >= 0.55
                    and (lib.get('regression_sigma_ticks') or 999) <= 10.0):
                self.exception_tids.add(tid)

        # Brain — prefer live > forward_pass > training
        live_brain = os.path.join(cpdir, 'live_brain.pkl')
        forward_brain = os.path.join(cpdir, 'pattern_forward_brain.pkl')
        training_brains = sorted(glob.glob(os.path.join(cpdir, 'pattern_*_brain.pkl')))

        if os.path.exists(live_brain):
            self.brain.load(live_brain)
            print(f"  Brain: live_brain.pkl ({len(self.brain.table)} states)")
        elif os.path.exists(forward_brain):
            self.brain.load(forward_brain)
            print(f"  Brain: pattern_forward_brain.pkl ({len(self.brain.table)} states)")
        elif training_brains:
            self.brain.load(training_brains[-1])
            print(f"  Brain: {os.path.basename(training_brains[-1])}")
        else:
            print("  Brain: starting fresh")

        print(f"  Centroids: {len(self.valid_tids)} ready")
        print(f"  Exception templates: {len(self.exception_tids)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='History replay test')
    parser.add_argument('--days', type=int, default=5)
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--atlas', default='DATA/ATLAS')
    args = parser.parse_args()

    replay = HistoryReplayEngine(
        checkpoint_dir=args.checkpoint_dir,
        n_days=args.days,
        atlas_root=args.atlas,
    )
    result = replay.run()
    print(f"\nResult: {result.validation.replay_trades} trades, "
          f"parity={result.validation.parity_score:.2f}")
