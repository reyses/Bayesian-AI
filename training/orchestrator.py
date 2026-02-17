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
from training.orchestrator_worker import simulate_trade_standalone, _optimize_pattern_task, _optimize_template_task, _process_template_job
from training.orchestrator_worker import FISSION_SUBSET_SIZE, INDIVIDUAL_OPTIMIZATION_ITERATIONS, DEFAULT_BASE_SLIPPAGE, DEFAULT_VELOCITY_SLIPPAGE_FACTOR

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
        print(f"  Loaded scaler: {self.scaler.mean_.shape[0]} features")

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
        # Assume data_source is ATLAS root. Look for 15s files to define "days".
        daily_files_15s = sorted(glob.glob(os.path.join(data_source, '15s', '*.parquet')))
        if not daily_files_15s:
            print(f"  No 15s data found in {data_source}/15s/")
            return

        print(f"  Found {len(daily_files_15s)} days to simulate.")

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0

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

            # Create a lookup for patterns by timestamp (or index)
            # Round timestamp to nearest 15s? They should match exactly if from same source.
            pattern_map = defaultdict(list)
            for p in actionable_patterns:
                pattern_map[p.timestamp].append(p)

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
                ts = row.timestamp
                price = getattr(row, 'close', getattr(row, 'price', 0.0))

                # 1. Manage existing position
                if self.wave_rider.position is not None:
                    res = self.wave_rider.update_trail(price, None, ts)
                    if res['should_exit']:
                        outcome = TradeOutcome(
                            state=active_template_id,
                            entry_price=active_entry_price,
                            exit_price=res['exit_price'],
                            pnl=res['pnl'],
                            result='WIN' if res['pnl'] > 0 else 'LOSS',
                            timestamp=ts,
                            exit_reason=res['exit_reason'],
                            entry_time=active_entry_time,
                            exit_time=ts,
                            duration=ts - active_entry_time,
                            direction='LONG' if active_side == 'long' else 'SHORT',
                            template_id=active_template_id
                        )
                        self.brain.update(outcome)
                        day_trades.append(outcome)
                        current_position_open = False

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

                        # RULE 2: Noise zone (<1.0 sigma)
                        elif micro_z < 1.0:
                            should_skip = True

                        # RULE 3: Approach zone (1.0 - 2.0 sigma)
                        elif 1.0 <= micro_z < 2.0:
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
                            # (Brain starts with pessimistic Beta(1,10) prior,
                            #  so 80% default would block everything)
                            if self.brain.should_fire(tid, min_prob=0.05, min_conf=0.0):
                                if dist < best_dist:
                                    best_dist = dist
                                    best_candidate = p
                                    best_tid = tid

                    if best_candidate:
                        # FIRE
                        params = self.pattern_library[best_tid]['params']
                        side = 'short' if best_candidate.z_score > 0 else 'long'
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

        for line in report_lines:
            print(line)

        # Save report
        report_path = os.path.join(self.checkpoint_dir, 'phase4_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"  Report saved to {report_path}")

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

            # Determine Tier
            tier = 4 # Toxic/Unknown
            if total > 30 and prob > 0.55 and conf > 0.30 and sharpe > 0.5:
                tier = 1
            elif total > 15 and prob > 0.50 and conf > 0.20:
                tier = 2
            elif total < 15:
                tier = 3
            elif prob < 0.45 and total > 20:
                tier = 4

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
                    tasks.append((tmpl, clustering_engine, self.config.iterations, self.param_generator, self.asset.point_value))

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
                                'count': member_count
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
            'member_count': template.member_count
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

    args = parser.parse_args()

    if not args.skip_deps:
        check_and_install_requirements()

    orchestrator = BayesianTrainingOrchestrator(args)

    try:
        if args.forward_pass and not args.fresh:
            # Phase 4 only (using existing playbook)
            orchestrator.run_forward_pass(args.data)
            if args.strategy_report:
                orchestrator.run_strategy_selection()
        elif args.strategy_report and not args.forward_pass:
            orchestrator.run_strategy_selection()
        else:
            # Full pipeline: Phase 2 → 3 → (optionally 4 → 5)
            results = orchestrator.train(args.data)
            if args.forward_pass:
                orchestrator.run_forward_pass(args.data)
            if args.strategy_report:
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


if __name__ == "__main__":
    sys.exit(main())
