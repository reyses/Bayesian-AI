"""
BAYESIAN-AI TRAINING ORCHESTRATOR
Single entry point for all training operations

Integrates:
- Walk-forward training (Pattern-Adaptive)
- Live dashboard (real-time visualization)
- Pattern analysis (strongest states)
- Progress reporting (terminal output)
- Batch regret analysis (end-of-day evaluation)
- Checkpoint management (resume capability)
"""
import os
import sys
import pickle
import argparse
import threading
import tempfile
import multiprocessing
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import time
from numba import cuda

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Core components
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.quantum_field_engine import QuantumFieldEngine
from core.context_detector import ContextDetector
from core.adaptive_confidence import AdaptiveConfidenceManager
from core.multi_timeframe_context import MultiTimeframeContext
from core.dynamic_binner import DynamicBinner
from core.three_body_state import ThreeBodyQuantumState
from core.exploration_mode import UnconstrainedExplorer, ExplorationConfig

# Training components
from training.doe_parameter_generator import DOEParameterGenerator
from training.pattern_analyzer import PatternAnalyzer
from training.progress_reporter import ProgressReporter, DayMetrics
from training.databento_loader import DatabentoLoader
from training.fractal_discovery_agent import FractalDiscoveryAgent, PatternEvent, TIMEFRAME_SECONDS
from training.fractal_clustering import FractalClusteringEngine, PatternTemplate
from training.pipeline_checkpoint import PipelineCheckpoint

# Execution components
from training.integrated_statistical_system import IntegratedStatisticalEngine
from training.batch_regret_analyzer import BatchRegretAnalyzer
from training.wave_rider import WaveRider
from training.orchestrator_worker import simulate_trade_standalone, _optimize_pattern_task, _optimize_template_task, _process_template_job, _audit_trade
from training.orchestrator_worker import FISSION_SUBSET_SIZE, INDIVIDUAL_OPTIMIZATION_ITERATIONS, DEFAULT_BASE_SLIPPAGE, DEFAULT_VELOCITY_SLIPPAGE_FACTOR

# Monte Carlo Pipeline
from training.monte_carlo_engine import MonteCarloEngine, simulate_template_tf_combo
from training.anova_analyzer import ANOVAAnalyzer
from training.thompson_refiner import ThompsonRefiner

INITIAL_CLUSTER_DIVISOR = 100
_ADX_TREND_CONFIRMATION = 25.0
_HURST_TREND_CONFIRMATION = 0.6

# Visualization
try:
    from visualization.live_training_dashboard import launch_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("WARNING: Live dashboard not available")

# Configuration
from config.symbols import MNQ

PRECOMPUTE_DEBUG_LOG_FILENAME = 'precompute_debug.log'

TIMEFRAME_MAP = {
    0: '5s',
    1: '15s',
    2: '60s',
    3: '5m',
    4: '15m',
    5: '1h'
}

def verify_cuda_availability():
    """Fail fast if CUDA is not available"""
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-capable GPU and drivers installed.")

    try:
        current_device = cuda.get_current_device()
        print(f"CUDA: AVAILABLE | Device: {current_device.name}")
    except Exception as e:
        print(f"CUDA: AVAILABLE (but failed to get device name: {e})")

# --------------------------------------------------

@dataclass
class DayResults:
    """Results from one day of training"""
    day_number: int
    date: str
    total_iterations: int
    best_iteration: int
    best_params: Dict[str, Any]
    best_sharpe: float
    best_win_rate: float
    best_pnl: float
    total_trades: int
    states_learned: int
    high_confidence_states: int
    execution_time_seconds: float
    avg_duration: float
    real_pnl: float = 0.0
    real_trades_count: int = 0
    real_win_rate: float = 0.0
    real_sharpe: float = 0.0


class BayesianTrainingOrchestrator:
    """
    UNIFIED TRAINING ORCHESTRATOR

    Runs complete walk-forward training with:
    - Pattern-Adaptive Optimization Loop
    - Live visualization dashboard
    - Real-time terminal progress
    - Pattern analysis and reporting
    - Batch regret analysis (fully integrated)
    - Automatic checkpointing
    """

    def __init__(self, config):
        self.config = config
        self.asset = MNQ  # Default asset
        self.checkpoint_dir = config.checkpoint_dir

        # Slippage configuration
        self.BASE_SLIPPAGE = DEFAULT_BASE_SLIPPAGE
        self.VELOCITY_SLIPPAGE_FACTOR = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize core components
        self.brain = QuantumBayesianBrain()
        self.engine = QuantumFieldEngine()
        self.context_detector = ContextDetector()
        self.param_generator = DOEParameterGenerator(self.context_detector)
        self.confidence_manager = AdaptiveConfidenceManager(self.brain)
        self.stat_validator = IntegratedStatisticalEngine(self.asset)
        self.wave_rider = WaveRider(self.asset)
        self.discovery_agent = FractalDiscoveryAgent()

        # Multi-timeframe context engine
        self.mtf_context = MultiTimeframeContext()
        self.all_tf_data = None  # Populated in train()

        # Dynamic histogram binner (fitted from first day's data)
        self.dynamic_binner = None  # Populated on first _precompute_day_states()

        # Analysis components
        self.pattern_analyzer = PatternAnalyzer()
        self.progress_reporter = ProgressReporter()
        self.regret_analyzer = BatchRegretAnalyzer()

        # Exploration Mode (optional)
        self.exploration_mode = getattr(config, 'exploration_mode', False)
        self.explorer = UnconstrainedExplorer(ExplorationConfig(max_trades=5000, fire_probability=1.0)) if self.exploration_mode else None
        if self.exploration_mode:
            print("WARNING: UNCONSTRAINED EXPLORATION MODE ENABLED (Entry filters bypassed)")

        # Training state
        self.day_results: List[DayResults] = []
        self.todays_trades: List[TradeOutcome] = []
        self._best_trades_today: List[TradeOutcome] = []
        self._cumulative_best_trades: List[TradeOutcome] = []
        self.dashboard = None
        self.dashboard_thread = None
        self.dashboard_queue = multiprocessing.Manager().Queue()

        # Pattern Library (Bayesian Priors)
        self.pattern_library = {}

        # Slippage parameters
        self.BASE_SLIPPAGE = DEFAULT_BASE_SLIPPAGE
        self.VELOCITY_SLIPPAGE_FACTOR = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

    def calculate_optimal_workers(self):
        try:
            return max(1, multiprocessing.cpu_count() - 2)
        except NotImplementedError:
            return 1

    def run_forward_pass(self, data_source: str):
        """
        Phase 4: Forward pass — replay full year using playbook.
        Scans fractal cascade per day, matches templates, trades via WaveRider.
        Brain learns from outcomes.
        """
        print("\n" + "="*80)
        print("PHASE 4: FORWARD PASS (EXECUTION MODE)")
        print("="*80)

        # 1. Load Prerequisites
        lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
        scaler_path = os.path.join(self.checkpoint_dir, 'clustering_scaler.pkl')

        if not os.path.exists(lib_path) or not os.path.exists(scaler_path):
            print("ERROR: pattern_library.pkl or clustering_scaler.pkl not found.")
            print("Run with --fresh to build from scratch.")
            return

        with open(lib_path, 'rb') as f:
            self.pattern_library = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"  Loaded library: {len(self.pattern_library)} templates")

        if hasattr(self.scaler, 'mean_'):
            print(f"  Loaded scaler: {self.scaler.mean_.shape[0]} features")
        else:
            print(f"  Loaded scaler: Not fitted (no patterns found)")
            if not self.pattern_library:
                print("  No patterns/templates to simulate. Exiting Phase 4.")
                return

        # Build centroid index for fast matching
        template_ids = list(self.pattern_library.keys())
        # Filter only templates with valid centroids (some might be empty/invalid if manually edited)
        valid_template_ids = [tid for tid in template_ids if 'centroid' in self.pattern_library[tid]]

        if not valid_template_ids:
            print("ERROR: No valid templates in library.")
            return

        centroids = np.array([self.pattern_library[tid]['centroid'] for tid in valid_template_ids])
        # Centroids are already in raw space (PatternTemplate stores raw), so we scale them
        # Wait, PatternTemplate stores raw_centroid. But clustering scaler was fit on X (features).
        # We need to scale features before matching, OR scale centroids.
        # Usually we scale input features to match the scaler's space.
        # But here we want to find nearest neighbor.
        # Ideally we scale BOTH into the normalized space.
        # So we transform centroids once.
        centroids_scaled = self.scaler.transform(centroids)

        print(f"  Prepared {len(centroids)} centroids for matching.")

        # 2. Iterate Days
        # Use 15s daily files for position management (daily granularity).
        # Pattern entry timestamps are snapped to 1m for signal quality.
        daily_files_15s = sorted(glob.glob(os.path.join(data_source, '15s', '*.parquet')))
        if not daily_files_15s:
            print(f"  No 15s data found in {data_source}/15s/")
            return

        print(f"  Found {len(daily_files_15s)} days to simulate.")

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0

        # Audit counters
        audit_tp = 0
        audit_fp_noise = 0
        audit_fp_wrong = 0
        audit_tn = 0
        audit_fn = 0

        # Per-trade oracle tracking
        _ORACLE_LABEL_NAMES = {2: 'MEGA_LONG', 1: 'SCALP_LONG', 0: 'NOISE', -1: 'SCALP_SHORT', -2: 'MEGA_SHORT'}
        oracle_trade_records = []  # completed per-trade oracle dicts
        pending_oracle = None      # oracle facts for currently open trade
        fn_potential_pnl = 0.0    # dollar potential of real moves we skipped

        for day_idx, day_file in enumerate(daily_files_15s):
            day_date = os.path.basename(day_file).replace('.parquet', '')
            print(f"\n  Day {day_idx+1}/{len(daily_files_15s)}: {day_date} ... ", end='', flush=True)

            # A. Fractal Cascade Scan (get actionable patterns with chains)
            # This uses the discovery agent logic but focused on this day
            actionable_patterns = self.discovery_agent.scan_day_cascade(data_source, day_date)

            # Sort by timestamp to simulate real-time feed
            actionable_patterns.sort(key=lambda x: x.timestamp)

            day_trades = []

            # Load 15s data for position management
            # We need the full dataframe to step through it
            try:
                df_15s = pd.read_parquet(day_file)
                # Ensure timestamps are floats for comparison
                if 'timestamp' in df_15s.columns and not np.issubdtype(df_15s['timestamp'].dtype, np.number):
                     df_15s['timestamp'] = df_15s['timestamp'].apply(lambda x: x.timestamp())
            except Exception as e:
                print(f"FAILED to load {day_file}: {e}")
                continue

            # Map patterns to bar indices for efficient processing
            # Or just iterate bars and check if a pattern triggered?
            # Better: Iterate patterns, attempt entry.
            # But we also need to manage exits tick-by-tick (or bar-by-bar).
            # Hybrid approach:
            # 1. Patterns queue.
            # 2. Iterate bars.
            # 3. If bar matches pattern timestamp, try entry.
            # 4. Always update open position.

            # Snap pattern timestamps to nearest 1m bar for matching
            # Patterns come from all TFs (4h down to 15s), snap to 60s boundary
            pattern_map = defaultdict(list)
            for p in actionable_patterns:
                snapped_ts = int(p.timestamp) // 60 * 60  # Floor to nearest minute
                pattern_map[snapped_ts].append(p)

            # B. Simulation Loop
            t_sim_start = time.perf_counter()

            # We iterate through the dataframe row by row
            # To speed up, we can convert to list of namedtuples or similar?
            # Or just iterate row tuples.
            # WaveRider expects: price, state (optional for trail?), timestamp

            # Pre-compute states for the day?
            # _scan_day_cascade already computed states for the patterns.
            # For 15s bars that are NOT patterns, we might need state for exit logic?
            # WaveRider usually needs price and timestamp. Some logic might use state.
            # Let's assume exit logic primarily uses price action (trail, stop).

            current_position_open = False
            active_entry_price = 0.0
            active_entry_time = 0.0
            active_side = 'long'
            active_template_id = None

            for row in df_15s.itertuples():
                ts_raw = row.timestamp
                # Snap to 60s boundary to match pattern_map keys
                ts = int(ts_raw) // 60 * 60
                price = getattr(row, 'close', getattr(row, 'price', 0.0))

                # 1. Manage existing position
                if self.wave_rider.position is not None:
                    res = self.wave_rider.update_trail(price, None, ts_raw)
                    if res['should_exit']:
                        outcome = TradeOutcome(
                            state=active_template_id,
                            entry_price=active_entry_price,
                            exit_price=res['exit_price'],
                            pnl=res['pnl'],
                            result='WIN' if res['pnl'] > 0 else 'LOSS',
                            timestamp=ts_raw,
                            exit_reason=res['exit_reason'],
                            entry_time=active_entry_time,
                            exit_time=ts_raw,
                            duration=ts_raw - active_entry_time,
                            direction='LONG' if active_side == 'long' else 'SHORT',
                            template_id=active_template_id
                        )
                        self.brain.update(outcome)
                        day_trades.append(outcome)
                        current_position_open = False

                        # Complete oracle record for this trade
                        if pending_oracle is not None:
                            o_mfe = pending_oracle['oracle_mfe']
                            o_mae = pending_oracle['oracle_mae']
                            oracle_favorable = o_mfe if pending_oracle['direction'] == 'LONG' else o_mae
                            oracle_potential = oracle_favorable * self.asset.point_value
                            capture = outcome.pnl / oracle_potential if oracle_potential > 0 else 0.0
                            oracle_trade_records.append({
                                **pending_oracle,
                                'exit_price': outcome.exit_price,
                                'exit_reason': outcome.exit_reason,
                                'actual_pnl': outcome.pnl,
                                'oracle_potential_pnl': oracle_potential,
                                'capture_rate': round(min(capture, 9.99), 4),
                                'result': outcome.result,
                            })
                            pending_oracle = None

                # 2. Check for entries (if no position)
                if not current_position_open and ts in pattern_map:
                    candidates = pattern_map[ts]
                    best_candidate = None
                    best_dist = 999.0
                    best_tid = None

                    for p in candidates:
                        # --- Gate 0: Headroom Gate (Nightmare Field Equation) ---
                        micro_z = abs(p.z_score)
                        micro_pattern = p.pattern_type

                        # Macro context
                        chain = getattr(p, 'parent_chain', [])
                        root_entry = chain[-1] if chain else None
                        macro_z = abs(root_entry['z']) if root_entry else 0.0

                        should_skip = False

                        # RULE 1: No pattern = no trade
                        if not micro_pattern:
                            should_skip = True

                        # RULE 2: Noise zone (<0.5 sigma)
                        elif micro_z < 0.5:
                            should_skip = True

                        # RULE 3: Approach zone (0.5 - 2.0 sigma)
                        elif 0.5 <= micro_z < 2.0:
                            if micro_pattern == 'STRUCTURAL_DRIVE':
                                # Only trade if strong trend confirmed
                                if p.state.adx_strength < _ADX_TREND_CONFIRMATION or p.state.hurst_exponent < _HURST_TREND_CONFIRMATION:
                                    should_skip = True
                            elif micro_pattern == 'ROCHE_SNAP':
                                # Snap hasn't reached tradeable zone
                                should_skip = True

                        # RULE 4: Mean Reversion / Extreme zone (>= 2.0 sigma)
                        elif micro_z >= 2.0:
                            headroom = macro_z < 2.0

                            if micro_pattern == 'ROCHE_SNAP':
                                # Extreme + Wall = Nightmare Field (Skip)
                                if not headroom and micro_z > 3.0:
                                    should_skip = True
                            elif micro_pattern == 'STRUCTURAL_DRIVE':
                                # Momentum into a wall = don't chase
                                if not headroom:
                                    should_skip = True

                        if should_skip:
                            continue

                        # Extract 14D features using shared logic
                        features = np.array([FractalClusteringEngine.extract_features(p)])

                        # Scale
                        feat_scaled = self.scaler.transform(features)

                        # Match
                        dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)
                        nearest_idx = np.argmin(dists)
                        dist = dists[nearest_idx]
                        tid = valid_template_ids[nearest_idx]

                        if dist < 3.0: # Threshold
                            # Brain gate — use low threshold for exploration
                            if self.brain.should_fire(tid, min_prob=0.05, min_conf=0.0):
                                # Score: lower depth (higher TF) gets priority, then closer centroid distance
                                # depth: 1=4h, 2=1h, 3=15m, 4=5m, 5=1m, 6=15s
                                p_depth = getattr(p, 'depth', 6)
                                score = p_depth + dist  # Lower is better
                                if score < best_dist:
                                    best_dist = score
                                    best_candidate = p
                                    best_tid = tid

                    if best_candidate:
                        # FIRE
                        params = self.pattern_library[best_tid]['params']
                        # Direction from template centroid, not current bar
                        template_z = self.pattern_library[best_tid]['centroid'][0]  # z_score is feature[0]
                        side = 'short' if template_z > 0 else 'long'
                        self.wave_rider.open_position(
                            entry_price=price,
                            side=side,
                            state=best_candidate.state,
                            stop_distance_ticks=params.get('stop_loss_ticks', 15),
                            profit_target_ticks=params.get('take_profit_ticks', 50),
                            trailing_stop_ticks=params.get('trailing_stop_ticks', 10),
                            template_id=best_tid
                        )
                        current_position_open = True
                        active_entry_price = price
                        active_entry_time = ts
                        active_side = side
                        active_template_id = best_tid

                        # Store oracle facts for this trade (linked at exit)
                        pending_oracle = {
                            'template_id': best_tid,
                            'direction': 'LONG' if side == 'long' else 'SHORT',
                            'entry_price': price,
                            'oracle_label': best_candidate.oracle_marker,
                            'oracle_label_name': _ORACLE_LABEL_NAMES.get(best_candidate.oracle_marker, 'UNKNOWN'),
                            'oracle_mfe': best_candidate.oracle_meta.get('mfe', 0.0),
                            'oracle_mae': best_candidate.oracle_meta.get('mae', 0.0),
                        }

                        # AUDIT: True Positive or False Positive
                        audit_outcome = TradeOutcome(
                            state=best_candidate.state,
                            entry_price=price,
                            exit_price=0.0,
                            pnl=0.0,
                            result='PENDING',
                            timestamp=ts,
                            exit_reason='PENDING',
                            direction='LONG' if side == 'long' else 'SHORT'
                        )
                        audit_res = _audit_trade(audit_outcome, best_candidate)
                        cls = audit_res['classification']
                        if cls == 'TP': audit_tp += 1
                        elif cls == 'FP_NOISE': audit_fp_noise += 1
                        elif cls == 'FP_WRONG': audit_fp_wrong += 1

                        # Audit other candidates as SKIPPED
                        for p in candidates:
                            if p == best_candidate: continue
                            audit_res = _audit_trade(None, p)
                            if audit_res['classification'] == 'TN':
                                audit_tn += 1
                            elif audit_res['classification'] == 'FN':
                                audit_fn += 1
                                _om = getattr(p, 'oracle_marker', 0)
                                _meta = getattr(p, 'oracle_meta', {})
                                fn_potential_pnl += (_meta.get('mfe', 0.0) if _om > 0 else _meta.get('mae', 0.0)) * self.asset.point_value
                    else:
                        # Audit all candidates as SKIPPED
                        for p in candidates:
                            audit_res = _audit_trade(None, p)
                            if audit_res['classification'] == 'TN':
                                audit_tn += 1
                            elif audit_res['classification'] == 'FN':
                                audit_fn += 1
                                _om = getattr(p, 'oracle_marker', 0)
                                _meta = getattr(p, 'oracle_meta', {})
                                fn_potential_pnl += (_meta.get('mfe', 0.0) if _om > 0 else _meta.get('mae', 0.0)) * self.asset.point_value

            # End of day cleanup — force close any open position
            if self.wave_rider.position is not None:
                pos = self.wave_rider.position
                if pos.side == 'short':
                    eod_pnl = (pos.entry_price - price) * self.asset.point_value
                else:
                    eod_pnl = (price - pos.entry_price) * self.asset.point_value
                self.wave_rider.position = None

                outcome = TradeOutcome(
                    state=active_template_id,
                    entry_price=active_entry_price,
                    exit_price=price,
                    pnl=eod_pnl,
                    result='WIN' if eod_pnl > 0 else 'LOSS',
                    timestamp=ts,
                    exit_reason='TIME_EXIT',
                    entry_time=active_entry_time,
                    exit_time=ts,
                    duration=ts - active_entry_time,
                    direction='LONG' if active_side == 'long' else 'SHORT',
                    template_id=active_template_id
                )
                self.brain.update(outcome)
                day_trades.append(outcome)

                # Complete oracle record for EOD-forced close
                if pending_oracle is not None:
                    o_mfe = pending_oracle['oracle_mfe']
                    o_mae = pending_oracle['oracle_mae']
                    oracle_favorable = o_mfe if pending_oracle['direction'] == 'LONG' else o_mae
                    oracle_potential = oracle_favorable * self.asset.point_value
                    capture = outcome.pnl / oracle_potential if oracle_potential > 0 else 0.0
                    oracle_trade_records.append({
                        **pending_oracle,
                        'exit_price': outcome.exit_price,
                        'exit_reason': 'TIME_EXIT',
                        'actual_pnl': outcome.pnl,
                        'oracle_potential_pnl': oracle_potential,
                        'capture_rate': round(min(capture, 9.99), 4),
                        'result': outcome.result,
                    })
                    pending_oracle = None

            # Analyze day
            if day_trades:
                # Regret analysis (optional, or just stats)
                d_pnl = sum(t.pnl for t in day_trades)
                d_wins = sum(1 for t in day_trades if t.result == 'WIN')
                total_pnl += d_pnl
                total_trades += len(day_trades)
                total_wins += d_wins
                print(f"Trades: {len(day_trades)}, Wins: {d_wins}, PnL: ${d_pnl:.2f} ({time.perf_counter() - t_sim_start:.1f}s)")
            else:
                print("No trades.")

        # Final Report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FORWARD PASS COMPLETE")
        report_lines.append(f"Total Trades: {total_trades}")
        report_lines.append(f"Win Rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "Win Rate: N/A")
        report_lines.append(f"Total PnL: ${total_pnl:.2f}")
        report_lines.append("=" * 80)

        # ── ORACLE PROFIT ATTRIBUTION ────────────────────────────────────────────
        import csv as _csv
        from collections import defaultdict

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("ORACLE PROFIT ATTRIBUTION")
        report_lines.append("=" * 80)

        # ── 1. Opportunity landscape ─────────────────────────────────────────────
        total_real_opps = audit_tp + audit_fp_wrong + audit_fn  # oracle said real move
        total_noise_opps = audit_fp_noise + audit_tn              # oracle said noise
        tp_potential  = sum(r['oracle_potential_pnl'] for r in oracle_trade_records if r['oracle_label'] != 0 and r['oracle_label_name'] not in ('NOISE',))
        ideal_profit  = tp_potential + fn_potential_pnl           # perfect execution on everything

        report_lines.append("")
        report_lines.append(f"  TOTAL SIGNALS SEEN BY ORACLE: {total_real_opps + total_noise_opps:,}")
        report_lines.append(f"    Real moves (MEGA/SCALP):  {total_real_opps:>6,}   — worth ${ideal_profit:>10,.2f} if perfectly traded")
        report_lines.append(f"    Noise (no real move):     {total_noise_opps:>6,}")

        # ── 2. What we did ───────────────────────────────────────────────────────
        n_traded   = len(oracle_trade_records)
        n_skipped  = audit_fn + audit_tn
        report_lines.append("")
        report_lines.append(f"  WHAT WE DID:")
        report_lines.append(f"    Traded:  {n_traded:>6,}  ({n_traded/(total_real_opps+total_noise_opps)*100:.1f}% of all signals)")
        report_lines.append(f"    Skipped: {n_skipped:>6,}  ({n_skipped/(total_real_opps+total_noise_opps)*100:.1f}% of all signals)")

        # ── 3. Of trades taken ───────────────────────────────────────────────────
        tp_recs       = [r for r in oracle_trade_records if r['oracle_label'] != 0 and
                         ((r['direction']=='LONG' and r['oracle_label']>0) or
                          (r['direction']=='SHORT' and r['oracle_label']<0))]
        fp_wrong_recs = [r for r in oracle_trade_records if r['oracle_label'] != 0 and r not in tp_recs]
        fp_noise_recs = [r for r in oracle_trade_records if r['oracle_label'] == 0]

        tp_pnl       = sum(r['actual_pnl'] for r in tp_recs)
        fp_wrong_pnl = sum(r['actual_pnl'] for r in fp_wrong_recs)
        fp_noise_pnl = sum(r['actual_pnl'] for r in fp_noise_recs)

        report_lines.append("")
        report_lines.append(f"  OF {n_traded:,} TRADES TAKEN:")
        report_lines.append(f"    Correct direction:  {len(tp_recs):>6,}  ({len(tp_recs)/n_traded*100:.1f}%)  →  actual: ${tp_pnl:>10,.2f}")
        report_lines.append(f"    Wrong direction:    {len(fp_wrong_recs):>6,}  ({len(fp_wrong_recs)/n_traded*100:.1f}%)  →  losses: ${fp_wrong_pnl:>10,.2f}")
        report_lines.append(f"    Traded noise:       {len(fp_noise_recs):>6,}  ({len(fp_noise_recs)/n_traded*100:.1f}%)  →  losses: ${fp_noise_pnl:>10,.2f}")

        # ── 4. Exit quality on correct-direction trades ──────────────────────────
        if tp_recs:
            optimal   = [r for r in tp_recs if r['capture_rate'] >= 0.80]
            partial   = [r for r in tp_recs if 0.20 <= r['capture_rate'] < 0.80]
            too_early = [r for r in tp_recs if 0 < r['capture_rate'] < 0.20]
            reversed_ = [r for r in tp_recs if r['capture_rate'] <= 0]

            left_on_table = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl']) for r in tp_recs)

            report_lines.append("")
            report_lines.append(f"  EXIT QUALITY (correct-direction trades):")
            report_lines.append(f"    Optimal  (≥80% of move captured): {len(optimal):>6,}  →  ${sum(r['actual_pnl'] for r in optimal):>10,.2f}")
            report_lines.append(f"    Partial  (20-80% captured):        {len(partial):>6,}  →  ${sum(r['actual_pnl'] for r in partial):>10,.2f}")
            report_lines.append(f"    Too early (<20% captured):         {len(too_early):>6,}  →  ${sum(r['actual_pnl'] for r in too_early):>10,.2f}")
            report_lines.append(f"    Reversed (went wrong after entry): {len(reversed_):>6,}  →  ${sum(r['actual_pnl'] for r in reversed_):>10,.2f}")
            report_lines.append(f"    Left on table (TP underperform):                      ${left_on_table:>10,.2f}")

        # ── 5. Profit gap summary ────────────────────────────────────────────────
        left_on_table_val = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl']) for r in tp_recs) if tp_recs else 0.0

        report_lines.append("")
        report_lines.append(f"  PROFIT GAP ANALYSIS:")
        report_lines.append(f"    Ideal (all real moves, perfect exits):  ${ideal_profit:>12,.2f}")
        report_lines.append(f"    ─────────────────────────────────────────────────────")
        report_lines.append(f"    Lost — missed opportunities (skipped):  ${fn_potential_pnl:>12,.2f}  ({fn_potential_pnl/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost — wrong direction trades:          ${abs(fp_wrong_pnl):>12,.2f}  ({abs(fp_wrong_pnl)/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost — noise trades:                    ${abs(fp_noise_pnl):>12,.2f}  ({abs(fp_noise_pnl)/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost — exited too early/late:           ${left_on_table_val:>12,.2f}  ({left_on_table_val/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    ─────────────────────────────────────────────────────")
        report_lines.append(f"    Actual profit:                          ${total_pnl:>12,.2f}  ({total_pnl/ideal_profit*100:.1f}% of ideal)" if ideal_profit else f"    Actual profit: ${total_pnl:.2f}")

        # ── 6. Save CSV ──────────────────────────────────────────────────────────
        if oracle_trade_records:
            csv_path = os.path.join(self.checkpoint_dir, 'oracle_trade_log.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = _csv.DictWriter(f, fieldnames=list(oracle_trade_records[0].keys()))
                writer.writeheader()
                writer.writerows(oracle_trade_records)
            report_lines.append("")
            report_lines.append(f"  Per-trade oracle log saved: {csv_path}")

        for line in report_lines:
            print(line)

        # Save report
        report_path = os.path.join(self.checkpoint_dir, 'phase4_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"  Report saved to {report_path}")

    def run_final_validation(self, top_strategies):
        """
        Walk-forward: train on first 70% of months, validate on last 30%.
        Only strategies that are profitable in BOTH periods survive.
        """
        print("\n" + "="*80)
        print("PHASE 6: FINAL VALIDATION (WALK-FORWARD)")
        print("="*80)

        # Determine data split
        # We need to find all months available in ATLAS
        # Assume standard structure DATA/ATLAS/{tf}/{month}.parquet
        # We can just check '15m' or '1h' to find months
        # Or check all timeframes involved in top_strategies

        # Find unique timeframes in strategies
        tfs = set(s['timeframe'] for s in top_strategies)
        if not tfs:
            print("No strategies to validate.")
            return []

        sample_tf = list(tfs)[0]
        data_root = os.path.join("DATA", "ATLAS") # Default
        tf_dir = os.path.join(data_root, sample_tf)

        # Fallback if specific data path provided to orchestrator
        if hasattr(self.config, 'data') and os.path.isdir(self.config.data):
             # check if it has TF subdirs
             if os.path.isdir(os.path.join(self.config.data, sample_tf)):
                 data_root = self.config.data
                 tf_dir = os.path.join(data_root, sample_tf)

        monthly_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
        if not monthly_files:
            print(f"No monthly files found in {tf_dir}")
            return []

        split_idx = int(len(monthly_files) * 0.7)
        # train_months = monthly_files[:split_idx] # Not used, we rely on previous phases for "training" stats
        val_months = monthly_files[split_idx:]

        print(f"Validation Set: {len(val_months)} months ({os.path.basename(val_months[0])} - {os.path.basename(val_months[-1])})")

        validated = []

        # Load scaler
        scaler_path = os.path.join(self.checkpoint_dir, 'clustering_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None

        for strategy in top_strategies:
            tid = strategy['template_id']
            tf = strategy['timeframe']
            params = strategy['params']
            is_pnl = strategy.get('train_pnl', 0.0) # From Thompson Refiner

            # Out-of-sample performance
            # Run single iteration with fixed params on validation months
            oos_result = simulate_template_tf_combo(
                tid, tf, 1, data_root,
                self.pattern_library[tid], self.asset,
                original_scaler=scaler,
                mutation_base=None, # Fixed params
                month_filter=val_months
            )

            # Since we didn't pass mutation_base/scale, it generated random params?
            # Wait, simulate_template_tf_combo generates random params if mutation_base is None.
            # We want FIXED params.
            # We need to modify simulate_template_tf_combo to accept fixed_params?
            # Or use mutation_base with mutation_scale=0?
            # Let's use mutation_base=params, mutation_scale=0.0

            oos_result = simulate_template_tf_combo(
                tid, tf, 1, data_root,
                self.pattern_library[tid], self.asset,
                original_scaler=scaler,
                mutation_base=params,
                mutation_scale=0.0,
                month_filter=val_months
            )

            # Extract best (and only) iteration
            if not oos_result.iterations:
                continue

            iter_res = oos_result.iterations[0]
            oos_pnl = iter_res.total_pnl
            oos_win_rate = iter_res.win_rate
            oos_trades = iter_res.num_trades

            # Compute Sharpe manually for OOS
            pnl_per_trade = [t.pnl for t in iter_res.trades]
            if len(pnl_per_trade) > 1 and np.std(pnl_per_trade) > 0:
                oos_sharpe = np.mean(pnl_per_trade) / np.std(pnl_per_trade)
            else:
                oos_sharpe = 0.0

            # Tier classification
            if (oos_trades >= 20 and oos_win_rate > 0.45 and oos_pnl > 0 and oos_sharpe > 0.3):
                tier = 1  # PRODUCTION
            elif oos_trades >= 10 and oos_pnl > 0:
                tier = 2  # CANDIDATE
            elif oos_trades >= 5:
                tier = 3  # UNPROVEN
            else:
                tier = 4  # TOXIC

            validated.append({
                'template_id': tid,
                'timeframe': tf,
                'params': params,
                'is_pnl': is_pnl,
                'oos_pnl': oos_pnl,
                'oos_win_rate': oos_win_rate,
                'oos_trades': oos_trades,
                'oos_sharpe': oos_sharpe,
                'tier': tier
            })

        # Report
        print("\nFINAL VALIDATION REPORT")
        print(f"{'ID':<10} | {'TF':<5} | {'Tier':<4} | {'Trades':<6} | {'Win%':<5} | {'Sharpe':<6} | {'OOS PnL':<10}")
        print("-" * 80)
        for v in sorted(validated, key=lambda x: (x['tier'], -x['oos_sharpe'])):
             print(f"{v['template_id']:<10} | {v['timeframe']:<5} | {v['tier']:<4} | {v['oos_trades']:<6} | {v['oos_win_rate']*100:5.1f} | {v['oos_sharpe']:6.2f} | ${v['oos_pnl']:<9.2f}")

        # Save Tier 1 strategies to production playbook
        tier1 = [v for v in validated if v['tier'] == 1]

        # Convert to playbook format
        playbook = {}
        for v in tier1:
            key = f"{v['template_id']}_{v['timeframe']}" # Use unique key for combo
            playbook[key] = {
                'template_id': v['template_id'],
                'timeframe': v['timeframe'],
                'params': v['params'],
                'stats': {
                    'win_rate': v['oos_win_rate'],
                    'sharpe': v['oos_sharpe'],
                    'total_trades': v['oos_trades']
                }
            }

        with open(os.path.join(self.checkpoint_dir, 'production_playbook_mc.pkl'), 'wb') as f:
            pickle.dump(playbook, f)

        print(f"\nSaved {len(playbook)} Tier 1 strategies to production_playbook_mc.pkl")

        return validated

    def run_strategy_selection(self):
        """
        Phase 5: Analyze brain data + regret history to rank strategies.
        """
        print("\n" + "="*80)
        print("PHASE 5: STRATEGY SELECTION & RISK SCORING")
        print("="*80)

        lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
        if not os.path.exists(lib_path):
            print("ERROR: pattern_library.pkl not found.")
            return

        with open(lib_path, 'rb') as f:
            self.pattern_library = pickle.load(f)

        # Try to load brain from latest checkpoint or assume it's loaded
        # The orchestrator init loads a fresh brain. If we ran forward pass in same process, it's populated.
        # If running separately, we need to load the brain.
        # Look for the latest brain checkpoint.
        brain_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*_brain.pkl')))
        if brain_files:
            latest_brain = brain_files[-1]
            self.brain.load(latest_brain)
            print(f"  Loaded brain state from {os.path.basename(latest_brain)}")
        else:
            print("  WARNING: No brain checkpoint found. Using empty brain (or current memory if chained).")

        tier1_templates = []
        report_data = []

        print(f"\nAnalyzing {len(self.pattern_library)} strategies...")

        # Pre-group history by template_id for O(1) lookup
        history_by_template = defaultdict(list)
        for trade in self.brain.trade_history:
            if trade.template_id is not None:
                history_by_template[trade.template_id].append(trade)

        for tid in self.pattern_library:
            stats = self.brain.get_stats(tid)
            prob = stats['probability']
            conf = stats['confidence']
            total = stats['total']

            # Calculate Risk Metrics from history
            history = history_by_template.get(tid, [])

            if not history:
                sharpe = 0.0
                max_dd = 0.0
                win_rate = 0.0
                risk_score = 1.0 # High risk if unknown
                avg_pnl = 0.0
            else:
                pnls = [t.pnl for t in history]
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]

                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                avg_pnl = np.mean(pnls)

                std_pnl = np.std(pnls)
                sharpe = avg_pnl / (std_pnl + 1e-6) if std_pnl > 0 else 0.0

                # Max Drawdown
                cum_pnl = np.cumsum(pnls)
                peak = np.maximum.accumulate(cum_pnl)
                dd = peak - cum_pnl
                max_dd = np.max(dd) if len(dd) > 0 else 0.0

                win_rate = len(wins) / len(pnls)

                # Consec losses
                max_consec_loss = 0
                curr_consec = 0
                for p in pnls:
                    if p <= 0:
                        curr_consec += 1
                        max_consec_loss = max(max_consec_loss, curr_consec)
                    else:
                        curr_consec = 0

                # Risk Score Formula
                # 0.3 * (1 - win_rate) +
                # 0.3 * (abs(avg_loss) / (avg_win + 1e-6)) +
                # 0.2 * (max_consec_loss / 10.0) +
                # 0.2 * (abs(max_dd) / (sum(wins) + 1e-6))  # vs total gains? or total pnl?
                # Prompt said: abs(max_dd) / (total_pnl + 1e-6)

                total_gain = sum(wins)
                dd_ratio = abs(max_dd) / (total_gain + 1e-6) if total_gain > 0 else 1.0
                loss_ratio = abs(avg_loss) / (avg_win + 1e-6)

                risk_score = (
                    0.3 * (1.0 - win_rate) +
                    0.3 * min(loss_ratio, 2.0) + # Cap ratio
                    0.2 * min(max_consec_loss / 10.0, 1.0) +
                    0.2 * min(dd_ratio, 1.0)
                )

            # Determine Tier (use actual win_rate from history, not brain's Bayesian prob)
            tier = 3  # Default: UNPROVEN
            if total >= 20 and win_rate > 0.45 and avg_pnl > 0 and sharpe > 0.3:
                tier = 1  # PRODUCTION
            elif total >= 10 and win_rate > 0.40 and avg_pnl > 0:
                tier = 2  # PROMISING
            elif total >= 10 and (win_rate < 0.35 or avg_pnl < 0):
                tier = 4  # TOXIC

            report_data.append({
                'id': tid,
                'tier': tier,
                'trades': total,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'pnl': sum(t.pnl for t in history),
                'max_dd': max_dd,
                'risk': risk_score
            })

            if tier == 1:
                # Add to production playbook
                entry = self.pattern_library[tid].copy()
                entry['tier'] = 1
                entry['stats'] = {
                    'win_rate': win_rate,
                    'sharpe': sharpe,
                    'risk_score': risk_score,
                    'total_trades': total
                }
                tier1_templates.append((tid, entry))

        # Sort report
        report_data.sort(key=lambda x: (x['tier'], -x['sharpe']))

        # Build report
        rpt = []
        rpt.append("")
        rpt.append("STRATEGY PERFORMANCE REPORT")
        header = f"{'ID':<10} | {'Tier':<4} | {'Trades':<6} | {'Win%':<5} | {'Sharpe':<6} | {'PnL':<10} | {'MaxDD':<10} | {'Risk':<5}"
        rpt.append(header)
        rpt.append("-" * 85)
        for r in report_data:
            rpt.append(f"{r['id']:<10} | {r['tier']:<4} | {r['trades']:<6} | {r['win_rate']*100:5.1f} | {r['sharpe']:6.2f} | ${r['pnl']:<9.2f} | ${r['max_dd']:<9.2f} | {r['risk']:.2f}")

        # Save Playbook
        playbook = {tid: data for tid, data in tier1_templates}
        pb_path = os.path.join(self.checkpoint_dir, 'production_playbook.pkl')
        with open(pb_path, 'wb') as f:
            pickle.dump(playbook, f)

        rpt.append(f"\nSaved {len(playbook)} Tier 1 strategies to {pb_path}")

        # Tier summary
        from collections import Counter
        tier_counts = Counter(r['tier'] for r in report_data)
        rpt.append("")
        rpt.append("TIER SUMMARY:")
        for t in sorted(tier_counts.keys()):
            label = {1: 'PRODUCTION', 2: 'PROMISING', 3: 'UNPROVEN', 4: 'TOXIC'}.get(t, '?')
            rpt.append(f"  Tier {t} ({label}): {tier_counts[t]} templates")

        # Ancestry Analysis
        roche_roots = 0
        struct_roots = 0
        for tid, data in tier1_templates:
            centroid = data['centroid']
            if centroid[-1] > 0.5:
                roche_roots += 1
            else:
                struct_roots += 1

        rpt.append("")
        rpt.append("ANCESTRY ANALYSIS (Tier 1):")
        rpt.append(f"  Roche-backed: {roche_roots}")
        rpt.append(f"  Structure-backed: {struct_roots}")

        # Print to console
        for line in rpt:
            print(line)

        # Save to file
        report_path = os.path.join(self.checkpoint_dir, 'phase5_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(rpt) + '\n')
        print(f"\n  Report saved to {report_path}")


    def train(self, data_source: Any):
        """
        Master training loop (Pattern-Adaptive Walk-Forward)
        Supports full resume from any phase via PipelineCheckpoint.

        Args:
            data_source: Path to data (str) or list of files.
        """
        print("\n" + "="*80)
        print("BAYESIAN-AI TRAINING ORCHESTRATOR (PATTERN-ADAPTIVE)")
        print("="*80)

        # 0. Pipeline Checkpoint
        ckpt = PipelineCheckpoint(self.checkpoint_dir)

        # Handle --fresh flag
        if getattr(self.config, 'fresh', False):
            print("--fresh flag: clearing all pipeline checkpoints...")
            ckpt.clear()

        print(ckpt.summary())

        # 1. Pre-Flight Checks
        print("\nPerforming pre-flight checks...")
        verify_cuda_availability()

        print(f"Asset: {self.asset.ticker}")
        print(f"Checkpoint Dir: {self.checkpoint_dir}")
        print(f"Data Source: {data_source}")
        if os.path.isdir(str(data_source)):
            # Check if ATLAS root with TF subdirs
            subdirs = [d for d in os.listdir(str(data_source))
                       if os.path.isdir(os.path.join(str(data_source), d))]
            if subdirs:
                print(f"  Mode: ATLAS root ({len(subdirs)} timeframe dirs: {', '.join(sorted(subdirs))})")
            else:
                import glob as _glob
                _files = sorted(_glob.glob(os.path.join(str(data_source), "*.parquet")))
                print(f"  Mode: ATLAS directory ({len(_files)} parquet files)")
        elif os.path.isfile(str(data_source)):
            _sz = os.path.getsize(str(data_source)) / (1024*1024)
            print(f"  Mode: Single file ({_sz:.1f} MB)")
        else:
            print(f"  WARNING: Path does not exist!")

        # Launch Dashboard
        if not self.config.no_dashboard and DASHBOARD_AVAILABLE:
            self.launch_dashboard()

        # ===================================================================
        # PHASE 2: Pattern Discovery (with checkpoint/resume)
        # ===================================================================
        manifest = None
        templates = None

        if ckpt.has_discovery():
            cached_manifest, cached_levels = ckpt.load_discovery()
            if cached_manifest is not None:
                print(f"\n[RESUME] Phase 2: Loaded {len(cached_manifest)} patterns "
                      f"from {len(cached_levels)} completed levels")
                manifest = cached_manifest

        if manifest is None:
            print("\nPhase 2: Fractal Top-Down Discovery...")
            ckpt.update_phase('discovery', 'in_progress')

            # Check for partial resume (some levels done)
            partial_manifest, partial_levels = ckpt.load_discovery()

            manifest = self._run_discovery(
                data_source,
                checkpoint_callback=lambda lvl, tf, patterns, levels:
                    ckpt.save_discovery_level(patterns, levels),
                resume_manifest=partial_manifest,
                resume_levels=partial_levels
            )

            # Save completed discovery
            from collections import Counter
            completed_levels = list(set(p.timeframe for p in manifest))
            ckpt.save_discovery(manifest, completed_levels)

        # Print manifest summary
        roche = sum(1 for p in manifest if p.pattern_type == 'ROCHE_SNAP')
        struct = sum(1 for p in manifest if p.pattern_type == 'STRUCTURAL_DRIVE')
        print(f"Discovery: {len(manifest)} patterns (ROCHE: {roche}, STRUCT: {struct})")

        manifest.sort(key=lambda x: x.timestamp)
        if len(manifest) > 0:
            from datetime import datetime as _dt
            from collections import Counter
            first_ts = manifest[0].timestamp
            last_ts = manifest[-1].timestamp
            print(f"  Time range: {_dt.fromtimestamp(first_ts).strftime('%Y-%m-%d %H:%M')} -> {_dt.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M')}")

            tf_counts = Counter(p.timeframe for p in manifest)
            depth_counts = Counter(p.depth for p in manifest)
            print(f"  By timeframe:")
            for tf, count in sorted(tf_counts.items(), key=lambda x: TIMEFRAME_SECONDS.get(x[0], 0), reverse=True):
                r = sum(1 for p in manifest if p.timeframe == tf and p.pattern_type == 'ROCHE_SNAP')
                s = sum(1 for p in manifest if p.timeframe == tf and p.pattern_type == 'STRUCTURAL_DRIVE')
                print(f"    [{tf:>4s}] {count:>7,} (R:{r:,} S:{s:,})")
            print(f"  By depth: {dict(sorted(depth_counts.items()))}")

        # ===================================================================
        # PHASE 2.5: Recursive Clustering (with checkpoint)
        # ===================================================================
        if ckpt.has_templates():
            templates = ckpt.load_templates()
            if templates is not None:
                print(f"\n[RESUME] Phase 2.5: Loaded {len(templates)} templates from checkpoint")

        if templates is None:
            print("\nPhase 2.5: Generating Physically Tight Templates...")
            ckpt.update_phase('clustering', 'in_progress')

            n_initial = max(10, len(manifest) // INITIAL_CLUSTER_DIVISOR)
            print(f"  Initial clusters: {n_initial} (from {len(manifest)} patterns / {INITIAL_CLUSTER_DIVISOR})")

            clustering_engine = FractalClusteringEngine(n_clusters=n_initial, max_variance=0.5)
            templates = clustering_engine.create_templates(manifest)
            print(f"  Condensed {len(manifest)} raw patterns into {len(templates)} Tight Templates.")

            # Save scaler for Phase 4
            import pickle as _pickle
            scaler_path = os.path.join(self.checkpoint_dir, 'clustering_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                _pickle.dump(clustering_engine.scaler, f)
            print(f"  Saved clustering scaler to {scaler_path}")

            ckpt.save_templates(templates)

        # ===================================================================
        # PHASE 3: Template Optimization & Fission (with persistent scheduler)
        # ===================================================================
        num_workers = self.calculate_optimal_workers()

        # Check for Phase 3 resume
        completed_results = {}
        fissioned_ids = set()
        template_queue = []
        resumed_phase3 = False

        if ckpt.has_scheduler_state():
            prev_completed, prev_fissioned, prev_pending, prev_metrics = ckpt.load_scheduler_state()
            if prev_completed is not None:
                completed_results = prev_completed
                fissioned_ids = prev_fissioned or set()
                template_queue = prev_pending or []
                resumed_phase3 = True

                print(f"\n[RESUME] Phase 3: {len(completed_results)} done, "
                      f"{len(fissioned_ids)} fissioned, {len(template_queue)} pending")

                # Restore pattern library from completed results
                for tmpl_id, result in completed_results.items():
                    if result.get('status') == 'DONE' and 'template' in result:
                        self.register_template_logic(result['template'], result['best_params'])

        if not resumed_phase3:
            template_queue = templates.copy()

        print(f"\nPhase 3: Template Optimization & Fission Loop...")
        print(f"  Templates to process: {len(template_queue)}")
        print(f"  Already completed: {len(completed_results)}")
        print(f"  Workers: {num_workers}")
        print(f"  Iterations per template: {self.config.iterations}")

        ckpt.update_phase('optimization', 'in_progress')

        # We need a clustering engine for fission checks (may not exist if we resumed past Phase 2.5)
        try:
            clustering_engine
        except NameError:
            n_initial = max(10, len(manifest) // INITIAL_CLUSTER_DIVISOR)
            clustering_engine = FractalClusteringEngine(n_clusters=n_initial, max_variance=0.5)

        self.pattern_library = self.pattern_library or {}
        processed_count = len(completed_results)
        optimized_count = sum(1 for r in completed_results.values() if r.get('status') == 'DONE')
        fission_count = len(fissioned_ids)
        total_val_pnl = sum(r.get('val_pnl', 0.0) for r in completed_results.values() if r.get('status') == 'DONE')
        batch_size = num_workers * 2
        batch_number = 0
        t_phase3_start = time.perf_counter()

        with multiprocessing.Pool(processes=num_workers) as pool:
            while template_queue:
                # Prepare Batch
                current_batch = []
                for _ in range(batch_size):
                    if not template_queue:
                        break
                    current_batch.append(template_queue.pop(0))

                if not current_batch:
                    break

                batch_number += 1
                t_batch = time.perf_counter()
                total_in_queue = len(template_queue)
                print(f"\n  Batch {batch_number}: processing {len(current_batch)} templates ({total_in_queue} remaining in queue)...")

                tasks = []
                for tmpl in current_batch:
                    # Pass arguments as a dictionary to _process_template_job
                    tasks.append({
                        'template': tmpl,
                        'clustering_engine': clustering_engine,
                        'iterations': self.config.iterations,
                        'generator': self.param_generator,
                        'point_value': self.asset.point_value,
                        'pattern_library': self.pattern_library
                    })

                results = pool.map(_process_template_job, tasks)

                batch_done = 0
                batch_split = 0
                batch_pnl = 0.0

                for j, result in enumerate(results):
                    processed_count += 1
                    status = result['status']
                    tmpl_id = result['template_id']
                    original_tmpl = current_batch[j]

                    if status == 'SPLIT':
                        new_sub_templates = result['new_templates']
                        batch_split += 1
                        fission_count += 1
                        fissioned_ids.add(tmpl_id)
                        timing = result.get('timing', '')
                        print(f"    [{processed_count}] Template {tmpl_id}: FISSION -> {len(new_sub_templates)} subsets | {timing}")
                        template_queue.extend(new_sub_templates)

                        if self.dashboard_queue:
                            self.dashboard_queue.put({
                                'type': 'FISSION_EVENT',
                                'parent_id': tmpl_id,
                                'children_count': len(new_sub_templates),
                                'reason': 'Regret Divergence'
                            })

                    elif status == 'DONE':
                        tmpl = result['template']
                        best_params = result['best_params']
                        val_pnl = result['val_pnl']
                        member_count = result['member_count']

                        batch_done += 1
                        optimized_count += 1
                        batch_pnl += val_pnl
                        total_val_pnl += val_pnl

                        completed_results[tmpl_id] = result

                        # Set Risk & Reward Metrics
                        val_count = result.get('val_count', 0)
                        if val_count > 0:
                            tmpl.expected_value = val_pnl / val_count
                        else:
                            tmpl.expected_value = 0.0

                        tmpl.outcome_variance = result.get('outcome_variance', 0.0)
                        tmpl.avg_drawdown = result.get('avg_drawdown', 0.0)
                        tmpl.risk_score = result.get('risk_score', 0.0)

                        self.register_template_logic(tmpl, best_params)
                        timing = result.get('timing', '')
                        print(f"    [{processed_count}] Template {tmpl_id}: DONE ({member_count} members) -> PnL: ${val_pnl:.2f} | {timing}")

                        if self.dashboard_queue:
                            centroid = original_tmpl.centroid
                            self.dashboard_queue.put({
                                'type': 'TEMPLATE_UPDATE',
                                'id': tmpl_id,
                                'z': centroid[0],
                                'mom': centroid[2],
                                'pnl': val_pnl,
                                'count': member_count,
                                'transitions': tmpl.transition_probs
                            })

                batch_elapsed = time.perf_counter() - t_batch
                print(
                    f"  Batch {batch_number} complete: "
                    f"{batch_done} optimized, {batch_split} fissioned, "
                    f"batch PnL: ${batch_pnl:.2f} | {batch_elapsed:.1f}s"
                )

                # CHECKPOINT after each batch
                ckpt.save_scheduler_state(
                    completed_results, fissioned_ids, template_queue,
                    metrics={
                        'processed': processed_count,
                        'optimized': optimized_count,
                        'fission_count': fission_count,
                        'total_val_pnl': total_val_pnl,
                        'batch_number': batch_number
                    }
                )

        phase3_elapsed = time.perf_counter() - t_phase3_start
        ckpt.update_phase('optimization', 'complete', {
            'optimized': optimized_count,
            'fissioned': fission_count,
            'total_val_pnl': total_val_pnl
        })

        print(f"\n  Phase 3 Summary:")
        print(f"    Batches: {batch_number}")
        print(f"    Templates processed: {processed_count}")
        print(f"    Optimized: {optimized_count} | Fissioned: {fission_count}")
        print(f"    Library size: {len(self.pattern_library)} entries")
        print(f"    Total validated PnL: ${total_val_pnl:.2f}")
        print(f"    Time: {phase3_elapsed:.1f}s")

        # Save pattern library for Phase 4
        lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
        with open(lib_path, 'wb') as f:
            pickle.dump(self.pattern_library, f)
        print(f"  Saved pattern_library.pkl ({len(self.pattern_library)} entries)")

        print("\n=== Training Complete ===")
        self.print_final_summary()
        return self.day_results

    def _optimize_pattern_task(self, args):
        """Wrapper for standalone _optimize_pattern_task"""
        return _optimize_pattern_task(args)

    def _optimize_template_batch(self, subset):
        """Wrapper for standalone _optimize_template_task (Consensus Optimization)"""
        # We pass None for template as it is not used in the optimization logic
        best_params, best_sharpe = _optimize_template_task((None, subset, self.config.iterations, self.param_generator, self.asset.point_value))
        return best_params

    def _run_discovery(self, data_source: Any,
                       checkpoint_callback=None,
                       resume_manifest=None,
                       resume_levels=None) -> List[PatternEvent]:
        """
        Run top-down fractal discovery across ATLAS timeframes.
        If data_source points to the ATLAS root (contains TF subdirectories),
        uses hierarchical top-down scanning. Otherwise falls back to flat scan.
        """
        manifest = []

        if isinstance(data_source, str):
            is_atlas_root = os.path.isdir(data_source) and any(
                os.path.isdir(os.path.join(data_source, tf))
                for tf in TIMEFRAME_SECONDS
            )

            if is_atlas_root:
                manifest = self.discovery_agent.scan_atlas_topdown(
                    data_source,
                    on_level_complete=checkpoint_callback,
                    resume_manifest=resume_manifest,
                    resume_levels=resume_levels
                )
            else:
                manifest = self.discovery_agent.scan_atlas_parallel(data_source)

        elif isinstance(data_source, list):
            for path in data_source:
                manifest.extend(self.discovery_agent.scan_atlas_parallel(path))

        return manifest

    def register_template_logic(self, template: PatternTemplate, params: Dict):
        """
        Saves the centroid and params to the pattern_library.
        """
        self.pattern_library[template.template_id] = {
            'centroid': template.centroid,
            'params': params,
            'member_count': template.member_count,
            'transition_map': template.transition_map or template.transition_probs, # Support alias
            'expected_value': template.expected_value,
            'outcome_variance': template.outcome_variance,
            'avg_drawdown': template.avg_drawdown,
            'risk_score': template.risk_score
        }

    def validate_template_group(self, patterns: List[PatternEvent], params: Dict) -> float:
        """
        Validates a group of patterns with fixed params. Returns total PnL.
        """
        total_pnl = 0.0
        for p in patterns:
            outcome = simulate_trade_standalone(
                entry_price=p.price,
                data=p.window_data,
                state=p.state,
                params=params,
                point_value=self.asset.point_value
            )
            if outcome:
                total_pnl += outcome.pnl
        return total_pnl

    def update_library(self, pattern: PatternEvent, params: Dict, result: Dict):
        """
        Update Pattern_Library centroid.
        """
        ptype = pattern.pattern_type
        if ptype not in self.pattern_library:
            self.pattern_library[ptype] = {'count': 0, 'params': {}}

        lib = self.pattern_library[ptype]
        n = lib['count']

        current_avg = lib['params']
        for k, v in params.items():
            if isinstance(v, (int, float)):
                old_val = current_avg.get(k, v)
                new_val = (old_val * n + v) / (n + 1)
                current_avg[k] = new_val
            else:
                current_avg[k] = v

        lib['count'] += 1
        lib['params'] = current_avg

    def validate_pattern(self, pattern: PatternEvent, params: Dict) -> Optional[TradeOutcome]:
        """
        Run validation (Walk-Forward) using fixed params.
        """
        if not params:
            return None

        window = pattern.window_data
        if window is None or window.empty:
            return None

        outcome = simulate_trade_standalone(
            entry_price=pattern.price,
            data=window,
            state=pattern.state,
            params=params,
            point_value=self.asset.point_value
        )
        return outcome

    # Helpers
    def launch_dashboard(self):
        """Launch dashboard in background thread"""
        self.dashboard_thread = threading.Thread(target=launch_dashboard, args=(self.dashboard_queue,), daemon=True)
        self.dashboard_thread.start()
        print("Dashboard launching in background...")
        time.sleep(2)

    def shutdown_dashboard(self):
        """Cleanly stop the dashboard before process exit to avoid tkinter GC errors."""
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            try:
                self.dashboard_queue.put({'type': 'SHUTDOWN'})
                self.dashboard_thread.join(timeout=5)
            except Exception:
                pass

    def _prepare_dashboard_history(self):
        self._history_trades_data = []
        for t in self._cumulative_best_trades:
            self._history_trades_data.append({
                'pnl': t.pnl,
                'result': t.result,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'duration': t.duration,
                'timestamp': t.timestamp,
            })
        self._history_pnls = [t.pnl for t in self._cumulative_best_trades]
        self._history_durations = [t.duration for t in self._cumulative_best_trades]
        self._history_wins = sum(1 for t in self._cumulative_best_trades if t.result == 'WIN')
        self._history_day_summaries = []

    def _update_dashboard_with_current(self, day_result: DayResults, total_days: int, current_day_trades: List[TradeOutcome] = None):
        import json as _json
        json_path = os.path.join(os.path.dirname(__file__), 'training_progress.json')

        current_trades_data = []
        current_pnls = []
        current_durations = []
        current_wins = 0

        if current_day_trades:
            current_pnls = [t.pnl for t in current_day_trades]
            current_durations = [t.duration for t in current_day_trades]
            current_wins = sum(1 for t in current_day_trades if t.result == 'WIN')
            for t in current_day_trades:
                current_trades_data.append({
                    'pnl': t.pnl,
                    'result': t.result,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'exit_reason': t.exit_reason,
                    'duration': t.duration,
                    'timestamp': t.timestamp,
                })

        if not hasattr(self, '_history_pnls'):
            self._prepare_dashboard_history()

        trades_data = self._history_trades_data + current_trades_data
        all_pnls = self._history_pnls + current_pnls
        all_durations = self._history_durations + current_durations
        total_wins = self._history_wins + current_wins

        total_pnl = sum(all_pnls)
        total_trades = len(all_pnls)
        cum_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
        cum_sharpe = (np.mean(all_pnls) / (np.std(all_pnls) + 1e-6)) if total_trades >= 2 else 0.0
        avg_duration = np.mean(all_durations) if total_trades > 0 else 0.0

        cum_pnl = np.cumsum(all_pnls) if all_pnls else np.array([0.0])
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        day_summaries = self._history_day_summaries

        elapsed = time.time() - self.progress_reporter.start_time

        payload = {
            'iteration': day_result.day_number,
            'total_iterations': total_days,
            'elapsed_seconds': elapsed,
            'current_date': day_result.date,
            'states_learned': day_result.states_learned,
            'high_confidence_states': day_result.high_confidence_states,
            'trades': trades_data,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'cumulative_win_rate': cum_win_rate,
            'cumulative_sharpe': cum_sharpe,
            'avg_duration': avg_duration,
            'max_drawdown': max_drawdown,
            'best_params': str(day_result.best_params),
            'day_summaries': day_summaries,
            'today_trades': day_result.total_trades,
            'today_pnl': day_result.best_pnl,
            'today_win_rate': day_result.best_win_rate,
            'today_sharpe': day_result.best_sharpe,
        }

        try:
            with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(json_path), delete=False) as tmp:
                _json.dump(payload, tmp, default=str)
                tmp_path = tmp.name
            os.replace(tmp_path, json_path)
        except Exception:
            pass

    def save_checkpoint(self, day_number: int, date: str, day_result: DayResults):
        """Save brain, params, and pattern library to checkpoint"""
        # Save brain
        brain_path = os.path.join(self.checkpoint_dir, f"pattern_{day_number:04d}_brain.pkl")
        self.brain.save(brain_path)

        # Save pattern library
        lib_path = os.path.join(self.checkpoint_dir, f"pattern_library.pkl")
        with open(lib_path, 'wb') as f:
            pickle.dump(self.pattern_library, f)

        # Save best params
        if day_result.best_params:
            params_path = os.path.join(self.checkpoint_dir, f"pattern_{day_number:04d}_params.pkl")
            with open(params_path, 'wb') as f:
                pickle.dump(day_result.best_params, f)

        # Save day results
        results_path = os.path.join(self.checkpoint_dir, f"pattern_{day_number:04d}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(day_result, f)

    def print_final_summary(self):
        print(f"Total PnL: ${sum(t.pnl for t in self._cumulative_best_trades):.2f}")


def check_and_install_requirements():
    """Auto-install requirements.txt if missing packages detected"""
    import subprocess
    requirements_path = os.path.join(PROJECT_ROOT, 'requirements.txt')
    if not os.path.exists(requirements_path):
        return

    print("Checking dependencies...")
    try:
        # pip install
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--prefer-binary', '-r', requirements_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"WARNING: Some dependencies failed to install:\n{result.stderr[:500]}")
        else:
            # Check CUDA status after install
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"Dependencies OK | CUDA ready: {torch.cuda.get_device_name(0)}")
                else:
                    print("Dependencies OK | WARNING: CUDA not available — GPU acceleration disabled")
                    print("      (To enable CUDA, run: python scripts/fix_cuda.py)")
            except ImportError:
                print("Dependencies OK | WARNING: PyTorch not installed.")
    except subprocess.TimeoutExpired:
        print("WARNING: pip install timed out, continuing anyway...")
    except Exception as e:
        print(f"WARNING: Could not check dependencies: {e}")


def main():
    """Single entry point - command line interface"""
    parser = argparse.ArgumentParser(
        description="Bayesian-AI Training Orchestrator (Pattern-Adaptive)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', default=os.path.join("DATA", "ATLAS"), help="Path to ATLAS root, single TF directory, or parquet file")
    parser.add_argument('--iterations', type=int, default=1000, help="Iterations per pattern (default: 1000)")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--no-dashboard', action='store_true', help="Disable live dashboard")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")
    parser.add_argument('--exploration-mode', action='store_true', help="Enable unconstrained exploration mode")
    parser.add_argument('--fresh', action='store_true', help="Clear all pipeline checkpoints and start fresh")
    parser.add_argument('--forward-pass', action='store_true', help="Run Phase 4 forward pass using existing playbook")
    parser.add_argument('--strategy-report', action='store_true', help="Run Phase 5 strategy selection report")

    # Monte Carlo Flags (opt-in with --mc)
    parser.add_argument('--mc', action='store_true', help='Enable Monte Carlo sweep after Bayesian Phase 3')
    parser.add_argument('--mc-iters', type=int, default=2000, help='Monte Carlo iterations per (template, timeframe) combo')
    parser.add_argument('--mc-only', action='store_true', help='Skip discovery, just run Monte Carlo from existing templates')
    parser.add_argument('--anova-only', action='store_true', help='Skip MC sweep, just run ANOVA on existing results')
    parser.add_argument('--refine-only', action='store_true', help='Skip MC+ANOVA, just run Thompson refinement')

    args = parser.parse_args()

    # ── Tee stdout → checkpoints/training_log.txt (append, one file per project) ──
    import io
    class _Tee(io.TextIOWrapper):
        def __init__(self, log_path):
            self._file = open(log_path, 'a', encoding='utf-8', buffering=1)
            self._stdout = sys.stdout
        def write(self, data):
            self._stdout.write(data)
            self._file.write(data)
            return len(data)
        def flush(self):
            self._stdout.flush()
            self._file.flush()
        def isatty(self):
            return self._stdout.isatty()
        def fileno(self):
            return self._stdout.fileno()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, 'training_log.txt')
    _tee = _Tee(log_path)
    sys.stdout = _tee

    # Print a run separator so phases from different runs are easy to distinguish
    import datetime as _dt
    print(f"\n{'='*80}")
    print(f"RUN STARTED: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    if not args.skip_deps:
        check_and_install_requirements()

    orchestrator = BayesianTrainingOrchestrator(args)

    try:
        if args.anova_only:
             # Just ANOVA
            mc = MonteCarloEngine(
                checkpoint_dir=orchestrator.checkpoint_dir,
                asset=orchestrator.asset,
                pattern_library=orchestrator.pattern_library, # May be empty if not loaded
                brain=orchestrator.brain
            )
            # Try to load existing results
            mc._load_checkpoint()
            anova = ANOVAAnalyzer()
            factor_results, top_combos = anova.analyze(mc.results_db)
            return 0

        if args.refine_only:
             # Just Thompson
             # We need top_combos... usually from ANOVA.
             # Assume we can load them or re-run ANOVA on existing MC results.
            mc = MonteCarloEngine(
                checkpoint_dir=orchestrator.checkpoint_dir,
                asset=orchestrator.asset,
                pattern_library=orchestrator.pattern_library,
                brain=orchestrator.brain
            )
            mc._load_checkpoint()
            anova = ANOVAAnalyzer()
            _, top_combos = anova.analyze(mc.results_db)

            refiner = ThompsonRefiner(
                brain=orchestrator.brain,
                asset=orchestrator.asset,
                top_combos=top_combos,
                pattern_library=orchestrator.pattern_library,
                checkpoint_dir=orchestrator.checkpoint_dir
            )
            refined_strategies = refiner.refine()
            orchestrator.run_final_validation(refined_strategies)
            return 0

        if args.forward_pass and not args.fresh:
            # Phase 4 only (using existing playbook)
            orchestrator.run_forward_pass(args.data)
            if args.strategy_report:
                orchestrator.run_strategy_selection()
        elif args.strategy_report and not args.forward_pass:
            orchestrator.run_strategy_selection()
        else:
            # Full pipeline
            if args.mc_only:
                # MC-only: skip discovery, load existing library
                print("Skipping Phase 2/2.5, loading existing library...")
                lib_path = os.path.join(orchestrator.checkpoint_dir, 'pattern_library.pkl')
                if os.path.exists(lib_path):
                    with open(lib_path, 'rb') as f:
                        orchestrator.pattern_library = pickle.load(f)
                else:
                    print("ERROR: pattern_library.pkl not found for --mc-only")
                    return 1
            else:
                # Phase 2 (Discovery) + 2.5 (Clustering) + 3 (Bayesian DOE Optimization)
                orchestrator.train(args.data)

            if args.mc or args.mc_only:
                # Optional: Monte Carlo Sweep → ANOVA → Thompson → Validation
                mc = MonteCarloEngine(
                    checkpoint_dir=orchestrator.checkpoint_dir,
                    asset=orchestrator.asset,
                    pattern_library=orchestrator.pattern_library,
                    brain=orchestrator.brain
                )
                mc.run_sweep(data_root=args.data, iterations_per_combo=args.mc_iters)

                anova = ANOVAAnalyzer()
                factor_results, top_combos = anova.analyze(mc.results_db)

                refiner = ThompsonRefiner(
                    brain=orchestrator.brain,
                    asset=orchestrator.asset,
                    top_combos=top_combos,
                    pattern_library=orchestrator.pattern_library,
                    checkpoint_dir=orchestrator.checkpoint_dir
                )
                refined_strategies = refiner.refine()
                orchestrator.run_final_validation(refined_strategies)
            else:
                # Default: Bayesian path → Forward Pass → Strategy Report
                orchestrator.run_forward_pass(args.data)
                orchestrator.run_strategy_selection()

        return 0
    except KeyboardInterrupt:
        print("\n\nWARNING: Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        orchestrator.shutdown_dashboard()
        # Restore stdout and close log file
        if isinstance(sys.stdout, _Tee):
            sys.stdout = _tee._stdout
            _tee._file.close()


if __name__ == "__main__":
    sys.exit(main())
