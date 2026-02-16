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
import asyncio
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
from training.fractal_discovery_agent import FractalDiscoveryAgent, PatternEvent
from training.fractal_clustering import FractalClusteringEngine, PatternTemplate

# Execution components
from training.integrated_statistical_system import IntegratedStatisticalEngine
from training.batch_regret_analyzer import BatchRegretAnalyzer
from training.wave_rider import WaveRider


# Visualization
try:
    from visualization.live_training_dashboard import LiveDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("WARNING: Live dashboard not available")

# Configuration
from config.symbols import MNQ

PRECOMPUTE_DEBUG_LOG_FILENAME = 'precompute_debug.log'

DEFAULT_BASE_SLIPPAGE = 0.25
DEFAULT_VELOCITY_SLIPPAGE_FACTOR = 0.1
REPRESENTATIVE_SUBSET_SIZE = 20

INITIAL_CLUSTER_DIVISOR = 100
FISSION_SUBSET_SIZE = 50
INDIVIDUAL_OPTIMIZATION_ITERATIONS = 20

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

# --- Standalone Helpers for Multiprocessing ---

def simulate_trade_standalone(entry_price: float, data: pd.DataFrame, state: Any,
                              params: Dict[str, Any], point_value: float) -> Optional[TradeOutcome]:
    """
    Simulate single trade with lookahead — direction-aware
    Uses params for stop loss and take profit
    """
    stop_loss = params.get('stop_loss_ticks', 15) * 0.25
    take_profit = params.get('take_profit_ticks', 40) * 0.25
    max_hold = params.get('max_hold_seconds', 600)
    trading_cost = params.get('trading_cost_points', 0.50)

    # Dynamic Slippage
    base_slippage = DEFAULT_BASE_SLIPPAGE
    velocity_factor = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

    velocity = state.particle_velocity
    slippage = base_slippage + velocity_factor * abs(velocity)
    total_slippage = slippage * 2.0

    # Direction from Archetype/State
    if hasattr(state, 'cascade_detected') and state.cascade_detected: # Roche Snap
        direction = 'SHORT' if state.z_score > 0 else 'LONG'
    elif hasattr(state, 'structure_confirmed') and state.structure_confirmed: # Structural Drive
        direction = 'LONG' if state.momentum_strength > 0 else 'SHORT'
    else:
        direction = 'LONG' # Fallback

    dir_sign = -1.0 if direction == 'SHORT' else 1.0

    entry_time = state.timestamp

    for i in range(1, len(data)):
        row = data.iloc[i]
        price = row['price'] if 'price' in row else row['close']
        pnl = (price - entry_price) * dir_sign - trading_cost - total_slippage

        exit_time = row['timestamp']
        # Handle timestamp types safely
        if hasattr(exit_time, 'timestamp'):
            exit_time = exit_time.timestamp()

        duration = exit_time - entry_time

        # Check TP/SL
        if pnl >= take_profit:
            return TradeOutcome(
                state=state, entry_price=entry_price, exit_price=price,
                pnl=take_profit * point_value, result='WIN', timestamp=exit_time,
                exit_reason='TP', entry_time=entry_time, exit_time=exit_time,
                duration=duration, direction=direction
            )
        elif pnl <= -stop_loss:
            return TradeOutcome(
                state=state, entry_price=entry_price, exit_price=price,
                pnl=-stop_loss * point_value, result='LOSS', timestamp=exit_time,
                exit_reason='SL', entry_time=entry_time, exit_time=exit_time,
                duration=duration, direction=direction
            )
        elif duration >= max_hold:
            return TradeOutcome(
                state=state, entry_price=entry_price, exit_price=price,
                pnl=pnl * point_value, result='WIN' if pnl > 0 else 'LOSS', timestamp=exit_time,
                exit_reason='TIME', entry_time=entry_time, exit_time=exit_time,
                duration=duration, direction=direction
            )

    return None

def _optimize_pattern_task(args):
    """
    Task function for multiprocessing.
    args: (pattern, iterations, param_generator, point_value)
    Returns: (best_params, best_result_dict)
    """
    pattern, iterations, generator, point_value = args

    window = pattern.window_data
    if window is None or window.empty:
        return {}, {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}

    # Generate parameters
    all_param_sets = []
    for i in range(iterations):
        ps = generator.generate_parameter_set(iteration=i, day=pattern.idx, context='PATTERN')
        all_param_sets.append(ps.parameters)

    best_pnl = -float('inf')
    best_params = {}
    best_result_dict = {'trades': [], 'sharpe': 0.0, 'win_rate': 0.0, 'pnl': 0.0}

    entry_price = pattern.price
    state = pattern.state

    for params in all_param_sets:
        outcome = simulate_trade_standalone(
            entry_price=entry_price,
            data=window,
            state=state,
            params=params,
            point_value=point_value
        )

        if outcome:
            if outcome.pnl > best_pnl:
                best_pnl = outcome.pnl
                best_params = params
                best_result_dict = {
                    'trades': [outcome],
                    'sharpe': 1.0 if outcome.pnl > 0 else -1.0,
                    'win_rate': 1.0 if outcome.result == 'WIN' else 0.0,
                    'pnl': outcome.pnl
                }

    return best_params, best_result_dict

def _optimize_template_task(args):
    """
    Optimizes a Template Group.
    args: (template, subset, iterations, generator, point_value)
    Returns: (best_params, best_sharpe)
    """
    template, subset, iterations, generator, point_value = args

    # 1. Generate Parameter Sets (DOE)
    # We use the first pattern in subset to drive the generator context,
    # but params are applicable to all.
    ref_pattern = subset[0]
    param_sets = []
    for i in range(iterations):
        ps = generator.generate_parameter_set(iteration=i, day=ref_pattern.idx, context='TEMPLATE')
        param_sets.append(ps.parameters)

    best_sharpe = -float('inf')
    best_params = {}

    # 2. Iterate through Parameter Sets
    for params in param_sets:
        pnls = []

        # 3. Test Params on ALL members of the subset
        for pattern in subset:
            window = pattern.window_data
            if window is None or window.empty:
                continue

            outcome = simulate_trade_standalone(
                entry_price=pattern.price,
                data=window,
                state=pattern.state,
                params=params,
                point_value=point_value
            )

            if outcome:
                pnls.append(outcome.pnl)
            else:
                pnls.append(0.0) # No trade = 0 PnL

        if not pnls:
            continue

        # 4. Calculate Combined Metric (Sharpe)
        # combined_pnl = sum(pnls)
        pnl_array = np.array(pnls)
        if len(pnl_array) > 1 and np.std(pnl_array) > 1e-9:
            sharpe = np.mean(pnl_array) / np.std(pnl_array)
        else:
            sharpe = 0.0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

    return best_params, best_sharpe

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

    def train(self, data_source: Any):
        """
        Master training loop (Pattern-Adaptive Walk-Forward)

        Args:
            data_source: Path to data (str) or list of files.
        """
        print("\n" + "="*80)
        print("BAYESIAN-AI TRAINING ORCHESTRATOR (PATTERN-ADAPTIVE)")
        print("="*80)

        # 1. Pre-Flight Checks
        print("Performing pre-flight checks...")
        verify_cuda_availability()

        print(f"Asset: {self.asset.ticker}")
        print(f"Checkpoint Dir: {self.checkpoint_dir}")

        # Launch Dashboard
        if not self.config.no_dashboard and DASHBOARD_AVAILABLE:
            self.launch_dashboard()

        # 2. Pattern Discovery (Phase 2)
        print("\nPhase 2: Scanning Atlas for Physics Archetypes...")
        manifest = self._run_discovery(data_source)
        print(f"Discovery Complete. Found {len(manifest)} patterns.")

        # Sort manifest chronologically
        manifest.sort(key=lambda x: x.timestamp)

        # --- PHASE 2.5: RECURSIVE CLUSTERING ---
        print("\nPhase 2.5: Generating Physically Tight Templates...")

        # Start with fewer clusters, let recursion expand them
        n_initial = max(10, len(manifest) // INITIAL_CLUSTER_DIVISOR)

        # Max Variance 0.5 means Z-Score spread of +/- 0.5 sigma is allowed.
        clustering_engine = FractalClusteringEngine(n_clusters=n_initial, max_variance=0.5)

        templates = clustering_engine.create_templates(manifest)
        print(f"Condensed {len(manifest)} raw patterns into {len(templates)} Tight Templates.")

        # --- PHASE 3: TEMPLATE OPTIMIZATION (THE FOUNDRY) ---
        print("\nPhase 3: Template Optimization & Fission Loop...")

        # Initialize Pattern Library
        self.pattern_library = {}

        template_queue = templates.copy()
        processed_count = 0
        num_workers = self.calculate_optimal_workers()
        batch_size = num_workers * 2

        # Import worker function here to avoid circular import
        from training.orchestrator_worker import _process_template_job

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

                # Prepare Tasks
                tasks = []
                for tmpl in current_batch:
                    tasks.append((tmpl, clustering_engine, self.config.iterations, self.param_generator, self.asset.point_value))

                # Run Batch
                results = pool.map(_process_template_job, tasks)

                for result in results:
                    processed_count += 1
                    status = result['status']
                    tmpl_id = result['template_id']

                    if status == 'SPLIT':
                        new_sub_templates = result['new_templates']
                        print(f"Template {tmpl_id}: FISSION -> Shattered into {len(new_sub_templates)} subsets.")
                        template_queue.extend(new_sub_templates)

                    elif status == 'DONE':
                        tmpl = result['template'] # Modified template object might be returned or just ID if unchanged
                        best_params = result['best_params']
                        val_pnl = result['val_pnl']
                        member_count = result['member_count']

                        self.register_template_logic(tmpl, best_params)
                        print(f"Template {tmpl_id}: OPTIMIZED ({member_count} members) -> Validated PnL: ${val_pnl:.2f}")

                        # Update Dashboard
                        if self.dashboard_queue:
                            self.dashboard_queue.put({
                                'type': 'TEMPLATE_UPDATE',
                                'id': tmpl_id,
                                'count': member_count,
                                'pnl': val_pnl
                            })

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

    def _run_discovery(self, data_source: Any) -> List[PatternEvent]:
        """Wrapper to run async discovery"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        manifest = []

        async def collect():
            if isinstance(data_source, str):
                async for pattern in self.discovery_agent.scan_atlas_async(data_source):
                    manifest.append(pattern)
            elif isinstance(data_source, list):
                for path in data_source:
                    async for pattern in self.discovery_agent.scan_atlas_async(path):
                        manifest.append(pattern)

        loop.run_until_complete(collect())
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
        def run_dashboard():
            try:
                import tkinter as tk
                root = tk.Tk()
                self.dashboard = LiveDashboard(root, queue=self.dashboard_queue)
                root.mainloop()
            except Exception as e:
                print(f"WARNING: Dashboard failed to launch: {e}")

        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
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

    parser.add_argument('--data', default=r"DATA\glbx-mdp3-20250101-20260209.ohlcv-1s.parquet", help="Path to parquet data file")
    parser.add_argument('--iterations', type=int, default=1000, help="Iterations per pattern (default: 1000)")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--no-dashboard', action='store_true', help="Disable live dashboard")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")
    parser.add_argument('--exploration-mode', action='store_true', help="Enable unconstrained exploration mode")

    args = parser.parse_args()

    if not args.skip_deps:
        check_and_install_requirements()

    orchestrator = BayesianTrainingOrchestrator(args)

    try:
        results = orchestrator.train(args.data)
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
