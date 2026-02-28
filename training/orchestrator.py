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
import warnings
warnings.filterwarnings("ignore", message=".*Grid size.*will likely result in GPU under-utilization.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*",               category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar divide.*", category=RuntimeWarning)
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

# Reports directory structure: reports/{mode}/ for all human-readable outputs
REPORTS_ROOT = os.path.join(PROJECT_ROOT, 'reports')

def _get_reports_dir(mode: str) -> str:
    """Return reports/{mode}/ path, creating it if needed."""
    d = os.path.join(REPORTS_ROOT, mode)
    os.makedirs(d, exist_ok=True)
    return d

# Core components
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.quantum_field_engine import QuantumFieldEngine
from core.dynamic_binner import DynamicBinner
from core.three_body_state import ThreeBodyQuantumState

# Training components
from training.doe_parameter_generator import DOEParameterGenerator
from training.pattern_analyzer import PatternAnalyzer
from training.progress_reporter import ProgressReporter, DayMetrics
from training.databento_loader import DatabentoLoader
from training.fractal_discovery_agent import FractalDiscoveryAgent, PatternEvent, TIMEFRAME_SECONDS, TIMEFRAME_HIERARCHY
from training.fractal_clustering import FractalClusteringEngine, PatternTemplate, HypervolumeTree, HypervolumeNode
from training.fractal_dna_tree import FractalDNATree
from training.pipeline_checkpoint import PipelineCheckpoint
from training.timeframe_belief_network import TimeframeBeliefNetwork, BeliefState

# Execution components
from training.batch_regret_analyzer import BatchRegretAnalyzer
from training.wave_rider import WaveRider
from training.orchestrator_worker import simulate_trade_standalone, _validate_template_consistency, _audit_trade, _init_pool_worker, _analytical_exits
from training.orchestrator_worker import FISSION_SUBSET_SIZE, INDIVIDUAL_OPTIMIZATION_ITERATIONS, DEFAULT_BASE_SLIPPAGE, DEFAULT_VELOCITY_SLIPPAGE_FACTOR

# Monte Carlo Pipeline
from training.monte_carlo_engine import MonteCarloEngine, simulate_template_tf_combo
from training.anova_analyzer import ANOVAAnalyzer
from training.thompson_refiner import ThompsonRefiner
from training.pid_oscillation_analyzer import PIDOscillationAnalyzer, PIDSignal

INITIAL_CLUSTER_DIVISOR = 100
_ADX_TREND_CONFIRMATION = 25.0
_HURST_TREND_CONFIRMATION = 0.6

DIRECTION_CONFIDENCE_THRESHOLD = 0.15
CONVICTION_SL_MULTIPLIER = 0.5
CONVICTION_SL_THRESHOLD = 0.5
DEFAULT_DECAY_HORIZON = 40
CONSENSUS_CONFIDENCE_THRESHOLD = 0.60

# Visualization
try:
    from visualization.live_training_dashboard import launch_dashboard, launch_popup
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("WARNING: Live dashboard not available")

# Configuration
from config.symbols import MNQ

PRECOMPUTE_DEBUG_LOG_FILENAME = 'precompute_debug.log'

MAX_CLUSTER_DISTANCE = 4.5

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
        print("WARNING: CUDA is not available. Proceeding with CPU fallback (slow).")
        return

    try:
        current_device = cuda.get_current_device()
        print(f"CUDA: AVAILABLE | Device: {current_device.name}")
    except Exception as e:
        print(f"CUDA: AVAILABLE (but failed to get device name: {e})")

# Golden Path Constants
BASE_RESOLUTION_SECONDS = 15
FALLBACK_HOLD_BARS = 5
MIN_ESTIMATED_HOLD_BARS = 2
ESTIMATED_TICKS_PER_BAR = 2.0  # Speed estimate for hold time calc

def _compute_golden_path_ideal(
    oracle_trade_records: list,   # actual trades taken (have entry_time, exit_time, oracle_potential_pnl)
    fn_oracle_records: list,      # gate-blocked real moves (have timestamp, fn_potential_pnl)
    bar_seconds: int = BASE_RESOLUTION_SECONDS,
    point_value: float = 2.0,     # MNQ = $2/point
) -> dict:
    """
    Compute the maximum achievable PnL via sequential non-overlapping trades.
    Returns a dict with the ideal total and a decomposition of where value was lost.

    Returns:
        {
            'ideal':       float,  # golden path total PnL
            'gp_traded':   float,  # value from golden-path signals we actually traded
            'gp_missed':   float,  # value from golden-path signals that were gate-blocked
            'gp_n_traded': int,    # count of golden-path signals we traded
            'gp_n_missed': int,    # count of golden-path signals that were gate-blocked
            'gp_traded_entry_times': set,  # entry timestamps of traded signals on golden path
        }
    """
    # Build unified candidate list: (entry_ts, exit_ts, value_usd, source)
    # source: 'traded' or 'fn' (gate-blocked)
    candidates = []

    # Traded signals — use actual times
    for r in oracle_trade_records:
        om = r.get('oracle_label', 0)
        if om == 0:
            continue  # noise trade
        val = r.get('oracle_potential_pnl', 0.0)
        if val <= 0:
            continue

        # Phase 2: Use DNA expectancy if available
        dna_exp = r.get('dna_expectancy', 0.0)
        if dna_exp > 0:
            val = max(val * 0.5, dna_exp)

        entry_ts = r.get('entry_time', 0)
        exit_ts  = r.get('exit_time',  entry_ts + bar_seconds * FALLBACK_HOLD_BARS)
        candidates.append((entry_ts, exit_ts, val, 'traded'))

    # Gate-blocked FN signals — estimate hold from MFE
    tick_value = point_value * 0.25  # MNQ: $0.50/tick
    for r in fn_oracle_records:
        val = r.get('fn_potential_pnl', 0.0)
        if val <= 0:
            continue

        # Phase 2: Use DNA expectancy if available
        dna_exp = r.get('dna_expectancy', 0.0)
        if dna_exp > 0:
            val = max(val * 0.5, dna_exp)

        entry_ts = r.get('timestamp', 0)
        # Estimate hold: MFE_ticks / estimated speed, minimum bars
        mfe_ticks = val / tick_value if tick_value > 0 else 8.0
        est_hold_bars = max(MIN_ESTIMATED_HOLD_BARS, int(mfe_ticks / ESTIMATED_TICKS_PER_BAR))
        exit_ts = entry_ts + est_hold_bars * bar_seconds
        candidates.append((entry_ts, exit_ts, val, 'fn'))

    if not candidates:
        return {'ideal': 0.0, 'gp_traded': 0.0, 'gp_missed': 0.0,
                'gp_n_traded': 0, 'gp_n_missed': 0,
                'gp_traded_entry_times': set()}

    # Sort by exit_time (greedy interval scheduling: earliest finish first)
    candidates.sort(key=lambda x: x[1])

    # Greedy: take each candidate if we are free at its entry time
    free_at    = 0
    total_pnl  = 0.0
    gp_traded  = 0.0
    gp_missed  = 0.0
    gp_n_traded = 0
    gp_n_missed = 0
    gp_traded_entry_times = set()
    for entry_ts, exit_ts, val, source in candidates:
        if entry_ts >= free_at:
            total_pnl += val
            free_at = exit_ts
            if source == 'traded':
                gp_traded += val
                gp_n_traded += 1
                gp_traded_entry_times.add(entry_ts)
            else:
                gp_missed += val
                gp_n_missed += 1

    return {
        'ideal':       total_pnl,
        'gp_traded':   gp_traded,
        'gp_missed':   gp_missed,
        'gp_n_traded': gp_n_traded,
        'gp_n_missed': gp_n_missed,
        'gp_traded_entry_times': gp_traded_entry_times,
    }

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
        self.param_generator = DOEParameterGenerator(None)
        self.wave_rider = WaveRider(self.asset)
        self.discovery_agent = FractalDiscoveryAgent()
        self.all_tf_data = None  # Populated in train()

        # Dynamic histogram binner (fitted from first day's data)
        self.dynamic_binner = None  # Populated on first _precompute_day_states()

        # Analysis components
        self.pattern_analyzer = PatternAnalyzer()
        self.progress_reporter = ProgressReporter()
        self.regret_analyzer = BatchRegretAnalyzer()

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

        # Bottom-line accumulators -- populated by run_forward_pass / run_strategy_selection
        self._fp_summary   = {}   # Phase 4 key metrics (IS)
        self._oos_summary  = {}   # Phase 6 key metrics (OOS)
        self._tier_summary = {}   # Phase 5 tier counts + top templates

        # PID Analyzer (Shadow Mode)
        self.pid_analyzer = PIDOscillationAnalyzer()

        # Slippage parameters
        self.BASE_SLIPPAGE = DEFAULT_BASE_SLIPPAGE
        self.VELOCITY_SLIPPAGE_FACTOR = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

    def calculate_optimal_workers(self):
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            mem_gb = 16.0
        # Pool-initializer mode: heavy objects loaded once per worker (~2GB each).
        # Per-task pickle payload is <1KB (just template + iterations).
        max_by_mem = max(1, int(mem_gb // 3))
        max_by_cpu = max(1, multiprocessing.cpu_count() - 2)
        return min(max_by_cpu, max_by_mem)

    def _calculate_cst_analytics(self, pending_oracle: dict, exit_idx: int) -> dict:
        """
        Helper to compute CST analytics (bar of death, bleed, exit integrity).
        Refactored to avoid duplication.
        """
        _struc_list = pending_oracle.get('structural_integrity', [])
        _thresh = pending_oracle.get('basin_threshold', 999.0)
        # Fallback constant should match WaveRider but here we use the logged threshold
        if _thresh < 1e-6: _thresh = 4.5

        _bar_death = -1
        for i, d in enumerate(_struc_list):
            if d > _thresh:
                _bar_death = i
                break

        _bleed_bars = 0
        _bleed_cost = 0.0 # Placeholder

        if _bar_death >= 0 and exit_idx > _bar_death:
            _bleed_bars = exit_idx - _bar_death

        _si_exit = _struc_list[exit_idx] if 0 <= exit_idx < len(_struc_list) else -1.0

        return {
            'structural_integrity_at_exit': _si_exit,
            'bar_of_structural_death': _bar_death,
            'bleed_bars': _bleed_bars,
            'bleed_cost': _bleed_cost
        }

    def _create_cst_exit_result(self, exit_price, entry_price, side, point_value):
        """Helper to construct the exit result dictionary for a structural break."""
        pnl = ((exit_price - entry_price) if side == 'long' else (entry_price - exit_price)) * point_value
        return {
            'should_exit': True,
            'exit_price': exit_price,
            'exit_reason': 'structural_break',
            'pnl': pnl,
            'adjustment_reason': 'structure_break'
        }

    def _audit_pid_signal(self, sig: PIDSignal, day_bars: pd.DataFrame,
                          bar_idx: int, point_value: float) -> dict:
        """
        Scan forward from bar_idx to see if price hit target or stop first.
        Uses high/low of each subsequent bar to check touch.
        Lookahead cap: 40 bars (10 minutes at 15s) — PID oscillations are fast.
        """
        lookahead  = min(40, len(day_bars) - bar_idx - 1)
        hit_target = False
        hit_stop   = False

        # Use iterrows for sequential access if dataframe, or iloc loop
        for i in range(bar_idx + 1, bar_idx + 1 + lookahead):
            # Access by position for speed
            row = day_bars.iloc[i]
            hi = row['high']
            lo = row['low']

            if sig.direction == 'LONG':
                if lo <= sig.stop_price:
                    hit_stop = True
                    break
                if hi >= sig.target_price:
                    hit_target = True
                    break
            else:  # SHORT
                if hi >= sig.stop_price:
                    hit_stop = True
                    break
                if lo <= sig.target_price:
                    hit_target = True
                    break

        if hit_target:
            theo_pnl = abs(sig.target_price - sig.entry_price) * point_value
        elif hit_stop:
            theo_pnl = -abs(sig.stop_price - sig.entry_price) * point_value
        else:
            # Neither hit within 10 min — use last bar's close vs entry
            last_row = day_bars.iloc[min(bar_idx + lookahead, len(day_bars)-1)]
            last_close = last_row['close']

            diff = (last_close - sig.entry_price) if sig.direction == 'LONG' \
                   else (sig.entry_price - last_close)
            theo_pnl = diff * point_value

        return {
            'would_have_hit_target': hit_target,
            'would_have_hit_stop':   hit_stop,
            'theoretical_pnl':       round(theo_pnl, 2),
        }

    def _navigate_hypervolume_tree(self, tree: HypervolumeTree,
                                   candidate: PatternEvent) -> Optional[Tuple[PatternTemplate, HypervolumeNode]]:
        """Walk the hypervolume tree to find the best matching template."""
        matrix = FractalClusteringEngine.build_hypervolume_matrix(candidate)
        if matrix is None or matrix.shape[0] < 2:
            return None

        current_nodes = tree.roots
        matched_path = []  # for logging

        for depth in range(matrix.shape[0]):
            if not current_nodes:
                return None

            feat_16d = matrix[depth]  # (16,)

            # ── Primary: cell membership test (is vector inside bounding box?) ──
            containing_nodes = []
            for node_id, node in current_nodes.items():
                if (np.all(feat_16d >= node.cell_min_16d) and
                        np.all(feat_16d <= node.cell_max_16d)):
                    dist = np.linalg.norm(feat_16d - node.centroid_16d)
                    containing_nodes.append((dist, node))

            if containing_nodes:
                containing_nodes.sort(key=lambda x: x[0])
                best_node = containing_nodes[0][1]
            else:
                # ── Fallback: nearest centroid within soft margin ──
                best_node = None
                best_dist = float('inf')
                for node_id, node in current_nodes.items():
                    dist = np.linalg.norm(feat_16d - node.centroid_16d)
                    # Soft margin: accept if within mean cell radius
                    cell_radius = np.mean(node.cell_max_16d - node.cell_min_16d) / 2
                    if dist < cell_radius * 2.0 and dist < best_dist:
                        best_dist = dist
                        best_node = node
                if best_node is None:
                    return None  # no match at this depth

            matched_path.append(best_node.node_id)

            if best_node.template is not None:
                return (best_node.template, best_node)

            current_nodes = best_node.children

        return None

    def run_forward_pass(self, data_source: str,
                         start_date: str = None, end_date: str = None,
                         min_tier: int = None,
                         bias_threshold: float = None,
                         dmi_threshold: float = None,
                         oos_mode: bool = False,
                         account_size: float = 0.0,
                         telemetry: bool = False):
        """
        Phase 4: Forward pass -- replay full year using playbook.
        Scans fractal cascade per day, matches templates, trades via WaveRider.
        Brain learns from outcomes.

        Args:
            start_date: Inclusive lower bound YYYYMMDD (e.g. '20260101').
                        If None, no lower bound -- all days included.
            end_date:   Inclusive upper bound YYYYMMDD (e.g. '20260209').
                        If None, no upper bound -- all days included.
            oos_mode:     When True, writes reports to reports/oos/ instead of reports/is/
                          and preserves training depth_weights.json unchanged.
                          Use with --forward-start to run blind out-of-sample simulation.
            account_size: Starting account equity in USD (0 = disabled, no equity gate).
                          When > 0, simulates a funded account: gates new entries if
                          running equity drops below NinjaTrader MNQ intraday margin
                          ($50/contract). Report shows equity curve + max drawdown.
        """
        # Launch UI — popup by default, full dashboard with --dashboard, nothing with --no-dashboard
        # (run_forward_pass is called directly for --forward-pass so we launch here too)
        if not getattr(self.config, 'no_dashboard', False) and DASHBOARD_AVAILABLE:
            if not self.dashboard_thread or not self.dashboard_thread.is_alive():
                if getattr(self.config, 'dashboard', False):
                    self.launch_dashboard()
                else:
                    self.launch_popup()

        print("\n" + "="*80)
        if oos_mode:
            print("PHASE 6: OOS BLIND VALIDATION (templates/scaler frozen from training)")
        else:
            print("PHASE 4: FORWARD PASS (IN-SAMPLE EXECUTION)")
        if start_date or end_date:
            _lo = start_date or "start"
            _hi = end_date   or "end"
            print(f"  Date slice: {_lo} -> {_hi}")
        print("="*80)
        if self.dashboard_queue:
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                      'step': 'FORWARD_PASS', 'pct': 0})

        # Reports directory: reports/oos/ or reports/is/
        _rpt_dir = _get_reports_dir('oos' if oos_mode else 'is')
        print(f"  Reports dir: {_rpt_dir}")

        # ── Rotate previous run: rename current reports → _prev ──────────────
        for _prev_name in ('oracle_trade_log.csv', 'signal_log.csv',
                           'fn_oracle_log.csv', 'pid_oracle_log.csv',
                           'phase4_report.txt'):
            _prev_path = os.path.join(_rpt_dir, _prev_name)
            if os.path.exists(_prev_path):
                _prev_dest = os.path.join(_rpt_dir,
                    _prev_name.replace('.', '_prev.', 1))
                try:
                    os.replace(_prev_path, _prev_dest)
                except OSError:
                    pass
        # Also rotate shards/
        _shards_dir = os.path.join(_rpt_dir, 'shards')
        _shards_prev = os.path.join(_rpt_dir, 'shards_prev')
        if os.path.isdir(_shards_dir):
            import shutil as _shutil_rotate
            if os.path.isdir(_shards_prev):
                try:
                    _shutil_rotate.rmtree(_shards_prev)
                except PermissionError:
                    # OneDrive/antivirus may hold locks — retry after brief pause
                    import time as _t_rotate
                    _t_rotate.sleep(1)
                    try:
                        _shutil_rotate.rmtree(_shards_prev)
                    except PermissionError:
                        print(f"  WARNING: Could not remove {_shards_prev} (locked), skipping rotation")
                        _shards_dir = None  # skip the rename below
            if _shards_dir:
                try:
                    os.rename(_shards_dir, _shards_prev)
                except OSError:
                    pass

        # 1. Load Prerequisites (Hypervolume Tree)
        tree_path = os.path.join(self.checkpoint_dir, 'hypervolume_tree.pkl')
        if not os.path.exists(tree_path):
            print("ERROR: hypervolume_tree.pkl not found. Run with --fresh.")
            return

        with open(tree_path, 'rb') as f:
            self.hypervolume_tree = pickle.load(f)

        # Flatten templates from tree for legacy compatibility (pattern_library lookups)
        # Assuming we can rebuild self.pattern_library or we just rely on tree nodes
        # But downstream logic uses self.pattern_library[tid].

        # We need to populate self.pattern_library from the tree templates
        self.pattern_library = {}
        # Simple recursion to collect all templates
        def _collect_templates(nodes):
            tmpls = []
            for node in nodes.values():
                if node.template:
                    tmpls.append(node.template)
                else:
                    tmpls.extend(_collect_templates(node.children))
            return tmpls

        _templates = _collect_templates(self.hypervolume_tree.roots)
        for t in _templates:
            # Templates carry best_params from DOE Phase 3 validation
            params = getattr(t, 'best_params', None) or {}
            self.register_template_logic(t, params)

        print(f"  Loaded Hypervolume Tree: {len(_templates)} templates")

        # Dummy scaler for legacy code that expects it (though it shouldn't be used)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.zeros(16)
        self.scaler.scale_ = np.ones(16)

        self.dna_tree = None # Disabled as per spec

        valid_template_ids = list(self.pattern_library.keys())
        if not valid_template_ids:
            print("ERROR: No templates in pattern library. Tree may have produced no leaves.")
            print("  Check Phase 2.5 output for '0 leaf templates'.")
            return
        centroids_scaled = np.array([self.pattern_library[tid]['centroid'] for tid in valid_template_ids])

        # Load tier map before centroid build so we can pre-filter by min_tier
        _TIER_SCORE_ADJ = {1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5}
        tiers_path = os.path.join(self.checkpoint_dir, 'template_tiers.pkl')
        template_tier_map = {}
        if os.path.exists(tiers_path):
            with open(tiers_path, 'rb') as f:
                template_tier_map = pickle.load(f)
            t1 = sum(1 for v in template_tier_map.values() if v == 1)
            print(f"  Loaded tier map: {len(template_tier_map)} templates ({t1} Tier 1)")
        else:
            print("  No tier map found -- all templates weighted equally (run strategy report first)")

        # Load per-depth PnL weights computed from the previous forward pass.
        # depth_weights.json is written at the end of each run so the NEXT run
        # can use data-driven depth scoring and filtering.
        _DEPTH_SCORE_ADJ  = {}   # depth (int) -> score adjustment (float, lower=better)
        _DEPTH_FILTER_OUT = set()  # depths whose avg_pnl/trade was negative last run
        _depth_weights_path = os.path.join(self.checkpoint_dir, 'depth_weights.json')
        if os.path.exists(_depth_weights_path):
            import json as _json
            with open(_depth_weights_path) as _dw_f:
                _dw_data = _json.load(_dw_f)
            _DEPTH_SCORE_ADJ  = {int(k): float(v.get('score_adj', 0.0)) for k, v in _dw_data.items()}
            _DEPTH_FILTER_OUT = {int(k) for k, v in _dw_data.items() if v.get('filter_out', False)}
            print(f"  Loaded depth weights ({len(_DEPTH_SCORE_ADJ)} depths, {len(_DEPTH_FILTER_OUT)} filtered out):")
            for _dk in sorted(_dw_data.keys(), key=int):
                _dv = _dw_data[_dk]
                print(f"    depth {_dk}: avg_pnl=${_dv.get('avg_pnl',0):.1f}/trade  "
                      f"score_adj={_dv.get('score_adj',0):+.2f}  filter={_dv.get('filter_out',False)}")
        else:
            print("  No depth weights found -- uniform depth scoring (run forward pass first to build weights)")

        # Load screening gates (model fission + temporal filters from waveform_screening)
        import json as _json
        _screening_gates_path = os.path.join(self.checkpoint_dir, 'screening_gates.json')
        _screening_fission = {}       # key: "{tid}_{direction}" -> "KEEP"/"SPLIT"/"DROP"
        _screening_good_hours = None  # set of UTC hours, or None = no filter
        _screening_default = None     # default class for unmatched (tid, dir) pairs
        if os.path.exists(_screening_gates_path):
            with open(_screening_gates_path) as _sgf:
                _sg = _json.load(_sgf)
            _screening_fission = _sg.get('fission_map', {})
            _hrs = _sg.get('good_hours_utc')
            _screening_good_hours = set(_hrs) if _hrs else None
            _screening_default = _sg.get('default_class', 'DROP')
            _n_keep = sum(1 for v in _screening_fission.values() if v == 'KEEP')
            _n_split = sum(1 for v in _screening_fission.values() if v == 'SPLIT')
            print(f"  Loaded screening gates: {_n_keep} KEEP, {_n_split} SPLIT, "
                  f"hours={sorted(_screening_good_hours) if _screening_good_hours else 'all'}")
        else:
            print("  No screening gates found -- all templates pass (run waveform_screening first)")

        # Apply min_tier filter -- removes losing tiers from the centroid index entirely
        # (oracle_trade_log shows Tier 4 = -$52K drag; min_tier=3 -> +$96K vs $44K baseline)
        if min_tier is not None and template_tier_map:
            _before = len(valid_template_ids)
            valid_template_ids = [tid for tid in valid_template_ids
                                  if template_tier_map.get(tid, 4) <= min_tier]
            print(f"  Min-tier filter (tier <= {min_tier}): {_before} -> {len(valid_template_ids)} active templates")

        # Rebuild centroids_scaled after tier filtering (already in scaled space)
        if not valid_template_ids:
            print("ERROR: All templates filtered out by min_tier. Try --min-tier 4 or omit it.")
            return
        centroids_scaled = np.array([self.pattern_library[tid]['centroid'] for tid in valid_template_ids])
        print(f"  Prepared {len(valid_template_ids)} centroids for matching.")

        # Pre-compute templates that earned a Gate 0 Rule 3 exception via data quality.
        # A template earns an exception if: enough members + positive win rate + low residuals.
        _EXCEPTION_MIN_MEMBERS  = 10
        _EXCEPTION_MIN_WIN_RATE = 0.55
        _EXCEPTION_MAX_SIGMA    = 10.0   # ticks; low = consistent behaviour
        _exception_tids = set()
        for _etid in valid_template_ids:
            _elib = self.pattern_library.get(_etid, {})
            _en   = _elib.get('n_members', 0)
            _ewr  = _elib.get('stats_win_rate', 0.0)
            _esig = _elib.get('regression_sigma_ticks', None)
            if (_en >= _EXCEPTION_MIN_MEMBERS
                    and _ewr >= _EXCEPTION_MIN_WIN_RATE
                    and _esig is not None
                    and _esig <= _EXCEPTION_MAX_SIGMA):
                _exception_tids.add(_etid)
        print(f"  Gate 0 exception templates: {len(_exception_tids)} / {len(valid_template_ids)} "
              f"(>={_EXCEPTION_MIN_MEMBERS} members, WR>={_EXCEPTION_MIN_WIN_RATE:.0%}, "
              f"sigma<={_EXCEPTION_MAX_SIGMA} ticks)")

        # Build per-TF DNA index for parent-anchor matching + multi-TF verification
        dna_index = {}  # tf_seconds → list of (tid, centroid_16d, bounds_min_10d, bounds_max_10d)
        for tid in valid_template_ids:
            lib = self.pattern_library[tid]
            dna_c = lib.get('dna_centroids', {})
            dna_bmin = lib.get('dna_bounds_min', {})
            dna_bmax = lib.get('dna_bounds_max', {})
            for tf_label, centroid in dna_c.items():
                tf_secs = TIMEFRAME_SECONDS.get(tf_label)
                if tf_secs and tf_label in dna_bmin and tf_label in dna_bmax:
                    dna_index.setdefault(tf_secs, []).append((
                        tid,
                        np.array(centroid),
                        np.array(dna_bmin[tf_label]),
                        np.array(dna_bmax[tf_label])
                    ))
        n_15m = len(dna_index.get(900, []))
        print(f"  DNA index: {n_15m} templates with 15m anchor, "
              f"{sum(len(v) for v in dna_index.values())} total TF entries")

        # Fractal belief network: 10 TF workers (1h -> 1s), path conviction
        # 15m worker = anchor (parent signature matching via cell containment)
        # Upper workers (30m, 1h) = macro DNA verification
        # Lower workers (5m -> 15s) = micro DNA verification
        belief_network = TimeframeBeliefNetwork(
            pattern_library  = self.pattern_library,
            scaler           = self.scaler,
            engine           = self.engine,
            valid_tids       = valid_template_ids,
            centroids_scaled = centroids_scaled,
            dna_index        = dna_index,
        )
        print(f"  Belief network: {len(TimeframeBeliefNetwork.TIMEFRAMES_SECONDS)} TF workers "
              f"(conviction threshold: {TimeframeBeliefNetwork.MIN_CONVICTION:.2f})")

        # 2. Iterate files (monthly YYYY_MM.parquet or daily YYYYMMDD.parquet)
        daily_files_15s = sorted(glob.glob(os.path.join(data_source, '15s', '*.parquet')))
        if not daily_files_15s:
            print(f"  No 15s data found in {data_source}/15s/")
            return

        def _file_sort_key(fpath):
            """Normalise filename to YYYYMMDD for date filtering/sorting.
            Monthly YYYY_MM -> YYYYMM01; daily YYYYMMDD stays as-is."""
            name = os.path.basename(fpath).replace('.parquet', '')
            if '_' in name:
                y, m = name.split('_', 1)
                return f"{y}{m.zfill(2)}01"
            return name

        # ── Time-slice filter ────────────────────────────────────────────────
        # start_date: files BEFORE this date are kept as warmup (workers tick,
        # no trades). This gives belief network workers full context before
        # trading begins.  end_date: files AFTER this date are removed.
        _warmup_cutoff = None   # file sort key; files < this are warmup-only
        if start_date or end_date:
            _before = len(daily_files_15s)
            if end_date:
                daily_files_15s = [f for f in daily_files_15s
                                   if _file_sort_key(f) <= end_date]
            if start_date:
                _warmup_cutoff = start_date
                _n_warmup = sum(1 for f in daily_files_15s
                                if _file_sort_key(f) < start_date)
                print(f"  Warmup: {_n_warmup} files (context only, no trades) "
                      f"before {start_date}")
            print(f"  Date filter: {_before} -> {len(daily_files_15s)} files "
                  f"({start_date or 'start'} -> {end_date or 'end'})")

        print(f"  Found {len(daily_files_15s)} files to simulate.")

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        _worker_total_states   = {}   # {tf_label: total states across all days}
        _worker_days_with_data = {}   # {tf_label: days that had > 0 states}
        decision_matrix_records = []  # per-candidate gate decision log (for root cause)

        # ── Account equity tracking (active when account_size > 0) ───────────
        # Simulates a NinjaTrader MNQ funded account with $50 intraday margin
        # per contract. Gates new entries when equity < margin. Tracks ruin.
        _NINJATRADER_MNQ_MARGIN = 50.0    # USD per contract, NinjaTrader intraday margin
        _equity_enabled  = account_size > 0.0
        running_equity   = account_size if _equity_enabled else 0.0
        peak_equity      = running_equity
        trough_equity    = running_equity
        account_ruined   = False    # True if equity drops below margin (cannot trade)
        ruin_day         = None     # Date string when ruin occurred
        skipped_ruin     = 0        # Trade entries skipped due to insufficient equity

        if _equity_enabled:
            print(f"  Account constraint: start=${account_size:.2f}  margin/contract=${_NINJATRADER_MNQ_MARGIN:.2f}")

        # Audit counters
        audit_tp = 0
        audit_fp_noise = 0
        audit_fp_wrong = 0
        audit_tn = 0
        audit_fn = 0

        def _effective_oracle(p) -> int:
            """Return the strongest oracle marker across the full macro-to-leaf chain.
            If the leaf pattern is NOISE but a macro ancestor says MEGA_LONG, use that.
            Chain is ordered leaf-first (index 0) -> root (last), so we scan all entries
            and keep the one with the highest |oracle_marker|.
            """
            leaf_om = getattr(p, 'oracle_marker', 0)
            best = leaf_om
            for ce in (getattr(p, 'parent_chain', None) or []):
                macro_om = ce.get('oracle_marker', 0)
                if abs(macro_om) > abs(best):
                    best = macro_om
            return best

        def _get_dna_expectancy(p):
            if self.dna_tree:
                dna, node, conf = self.dna_tree.match(p)
                if node and node.member_count >= 10:
                    return node.expectancy * 0.25 * self.asset.point_value # ticks -> USD
            return 0.0

        def _dm_rec(p, gate, day, ts_val, micro_z_val, macro_z_val, pattern_val,
                    dist=0.0, conviction=0.0, template_id='', tier='', pattern_dna=''):
            """Build one signal-log record. Trade outcome fields default to empty."""
            om   = _effective_oracle(p)
            meta = getattr(p, 'oracle_meta', None) or {}
            mfe  = float(meta.get('mfe', 0.0)) if isinstance(meta, dict) else 0.0
            olbl = {2:'MEGA', 1:'SCALP', 0:'NOISE', -1:'SCALP', -2:'MEGA'}.get(om, 'NOISE')
            opnl = round(abs(mfe) * self.asset.point_value, 2)
            return {
                # Detection context
                'ts': ts_val, 'day': day, 'depth': getattr(p, 'depth', 6),
                'pattern': pattern_val or '', 'micro_z': round(micro_z_val, 2),
                'macro_z': round(macro_z_val, 2),
                # Gate decision
                'gate': gate,
                'gate1_dist': round(dist, 3), 'gate3_conv': round(conviction, 3),
                # Oracle ground truth
                'oracle_label': olbl,
                'oracle_dir': 'LONG' if om > 0 else ('SHORT' if om < 0 else 'NONE'),
                'oracle_pnl': opnl,
                'template_id': str(template_id), 'tier': str(tier), 'pattern_dna': pattern_dna,
                # Trade outcome (filled in later if gate='traded')
                'trade_direction': '', 'trade_result': '', 'trade_pnl': 0.0,
                'exit_reason': '', 'exit_signal_reason': '',
                'exit_conviction': 0.0, 'exit_wave_maturity': 0.0,
            }

        # Per-trade oracle tracking
        _ORACLE_LABEL_NAMES = {2: 'MEGA_LONG', 1: 'SCALP_LONG', 0: 'NOISE', -1: 'SCALP_SHORT', -2: 'MEGA_SHORT'}
        oracle_trade_records = []  # completed per-trade oracle dicts
        pending_oracle = None      # oracle facts for currently open trade
        _pending_dm_idx = None     # index into decision_matrix_records for open trade
        fn_potential_pnl    = 0.0  # dollar potential of real moves we missed (gate-blocked)
        score_loser_pnl     = 0.0  # dollar potential of real moves we correctly passed over (took better trade same bar)

        # PID Shadow Log
        pid_oracle_records = []

        # Skip reason counters (per-candidate across all days)
        skip_headroom    = 0   # Gate 0: no pattern or noise zone / structural rules
        skip_dist        = 0   # No cluster match within distance 3.0
        skip_brain       = 0   # Brain gate: template probability too low
        skip_conviction  = 0   # Belief network: path conviction below MIN_CONVICTION
        skip_screening   = 0   # Gate 3.5: screening gates (fission DROP or bad temporal window)
        skip_direction   = 0   # Gate 5: direction consensus unclear
        n_signals_seen   = 0   # Total candidate signals evaluated (all gates combined)
        depth_traded     = defaultdict(int)  # depth -> trade count (1=high TF, 6=15s)

        # FN oracle records: per-signal log of missed real moves with worker snapshots.
        # Answers: "when we missed a profitable move, what did the workers think?"
        # If workers agreed with oracle direction on FN signals, a gate is too strict.
        fn_oracle_records = []

        n_days = len(daily_files_15s)
        for day_idx, day_file in enumerate(daily_files_15s):
            day_date = os.path.basename(day_file).replace('.parquet', '')
            # Normalise: monthly YYYY_MM -> YYYY_MM kept as-is for scan_day_cascade
            # (discovery agent matches by substring so both formats work)
            _is_warmup = (_warmup_cutoff is not None
                          and _file_sort_key(day_file) < _warmup_cutoff)
            _warmup_tag = ' [WARMUP]' if _is_warmup else ''
            print(f"\n  Day {day_idx+1}/{n_days}: {day_date}{_warmup_tag} ... ", end='', flush=True)
            if self.dashboard_queue:
                pct = (day_idx / n_days) * 100
                _wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                          'step': f'FORWARD_PASS  day {day_idx+1}/{n_days}',
                                          'pct': round(pct, 1),
                                          'pnl': total_pnl,
                                          'trades': total_trades,
                                          'wr': round(_wr, 1)})

            # A. Fractal Cascade Scan (get actionable patterns with chains)
            # This uses the discovery agent logic but focused on this day
            actionable_patterns = self.discovery_agent.scan_day_cascade(data_source, day_date)

            # Sort by timestamp to simulate real-time feed
            actionable_patterns.sort(key=lambda x: x.timestamp)

            day_trades = []
            eod_review_trades = []  # For worker EOD self-review

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

            # Load 5s and 1s ATLAS files for sub-resolution workers.
            # Monthly files: {data_source}/5s/YYYY_MM.parquet (same key as day_date)
            _atlas_root = data_source

            def _load_fine(tf_label):
                # Both training and OOS use monthly format: YYYY_MM.parquet
                # day_date is the file stem (e.g. '2025_11'); load the whole file.
                p = os.path.join(_atlas_root, tf_label, f"{day_date}.parquet")
                if not os.path.exists(p):
                    return None
                try:
                    _df = pd.read_parquet(p)
                    if 'timestamp' in _df.columns and not np.issubdtype(_df['timestamp'].dtype, np.number):
                        _df['timestamp'] = _df['timestamp'].apply(lambda x: x.timestamp())
                    return _df if not _df.empty else None
                except Exception:
                    return None

            _df_5s = _load_fine('5s')
            _df_1s  = _load_fine('1s')

            # Pre-extract 1s numpy arrays for wick-aware inner loop
            if _df_1s is not None and not _df_1s.empty:
                _1s_ts    = _df_1s['timestamp'].values.astype(np.float64)
                _1s_highs = _df_1s['high'].values.astype(np.float64)
                _1s_lows  = _df_1s['low'].values.astype(np.float64)
                _has_1s   = True
            else:
                _has_1s = False

            # Belief network: Task 1 for all 10 TF workers (1h -> 1s)
            # 1h->15s resampled from df_15s; 5s/1s from monthly ATLAS files.
            try:
                _states_15s = self.engine.batch_compute_states(df_15s, use_cuda=True)
                belief_network.prepare_day(df_15s, states_micro=_states_15s,
                                           df_5s=_df_5s, df_1s=_df_1s)
            except Exception as _bn_err:
                _states_15s = []
                belief_network.prepare_day(df_15s, states_micro=[],
                                           df_5s=_df_5s, df_1s=_df_1s)

            # Accumulate worker state counts for report diagnostics
            for _wlbl, _wcnt in belief_network.get_worker_state_counts().items():
                _worker_total_states[_wlbl]   = _worker_total_states.get(_wlbl, 0) + _wcnt
                _worker_days_with_data[_wlbl] = _worker_days_with_data.get(_wlbl, 0) + (1 if _wcnt > 0 else 0)

            # Reset PID analyzer for the day
            _day_sigmas = [s['state'].sigma_fractal for s in _states_15s if s['state'].sigma_fractal > 0]
            day_sigma = np.nanmean(_day_sigmas) if _day_sigmas else 1.0
            self.pid_analyzer.reset(sigma=day_sigma)

            # Map states for fast access by bar_idx
            _states_map = {s['bar_idx']: s['state'] for s in _states_15s}

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
            active_entry_depth = 5  # depth of triggering pattern (used for hold-time scaling)

            _bar_i = 0  # 15s bar index for belief network worker ticks

            # Per-calendar-day PnL tracking for dashboard
            import datetime as _dt_chart
            _chart_cur_day = None   # current calendar date string "MM/DD"
            _chart_day_pnl = 0.0
            _chart_day_trades = 0
            _chart_day_wins = 0

            for row in df_15s.itertuples():
                ts_raw = row.timestamp
                # Snap to 60s boundary to match pattern_map keys
                ts = int(ts_raw) // 60 * 60
                price = getattr(row, 'close', getattr(row, 'price', 0.0))

                # Detect calendar day change → emit daily PnL to dashboard
                _bar_day = _dt_chart.datetime.utcfromtimestamp(ts_raw).strftime('%m/%d')
                if _chart_cur_day is not None and _bar_day != _chart_cur_day:
                    if self.dashboard_queue:
                        self.dashboard_queue.put({
                            'type': 'DAY_PNL',
                            'day': _chart_cur_day,
                            'pnl': _chart_day_pnl,
                            'trades': _chart_day_trades,
                            'wins': _chart_day_wins,
                        })
                    _chart_day_pnl = 0.0
                    _chart_day_trades = 0
                    _chart_day_wins = 0
                _chart_cur_day = _bar_day

                # Belief network: tick all workers (event-driven by TF bar change)
                # 1h worker updates once per 240 bars; 15s worker updates every bar
                belief_network.tick_all(_bar_i)

                # PID ANALYZER TICK
                _pid_state = _states_map.get(_bar_i)
                if _pid_state:
                     pid_signal = self.pid_analyzer.tick(_pid_state)
                     if pid_signal:
                          # Audit
                          audit_res = self._audit_pid_signal(pid_signal, df_15s, _bar_i, self.asset.point_value)

                          # Find oracle label if any pattern exists at this timestamp
                          _candidates_here = pattern_map.get(ts, [])
                          _best_om = 0
                          _best_meta = {}
                          if _candidates_here:
                               for _c in _candidates_here:
                                    _om = _effective_oracle(_c)
                                    if abs(_om) > abs(_best_om):
                                         _best_om = _om
                                         _best_meta = getattr(_c, 'oracle_meta', {})

                          pid_oracle_records.append({
                                'timestamp': pid_signal.timestamp,
                                'direction': pid_signal.direction,
                                'entry_price': pid_signal.entry_price,
                                'target_price': pid_signal.target_price,
                                'stop_price': pid_signal.stop_price,
                                'z_score': pid_signal.z_score,
                                'band_touched': pid_signal.band_touched,
                                'regime_bars': pid_signal.regime_bars,
                                'osc_coherence': pid_signal.osc_coherence,
                                'term_pid': pid_signal.term_pid,
                                'pid_class': pid_signal.pid_class,
                                'tension_reason': pid_signal.tension_reason,
                                'oracle_label': _best_om,
                                'oracle_label_name': _ORACLE_LABEL_NAMES.get(_best_om, 'NOISE'),
                                'oracle_mfe': _best_meta.get('mfe', 0.0),
                                'oracle_mae': _best_meta.get('mae', 0.0),
                                **audit_res
                          })

                _bar_i += 1

                # 1. Manage existing position
                # Depth-scaled max hold — each TF has a natural resolution window.
                # Deeper = shorter TF trigger = must resolve faster.
                # Two-tier per depth: losing (underwater) gets half the normal limit.
                #   depth 1-2 (4h/1h patterns): 1440 bars normal / 480 losing
                #   depth 3-4 (15m/5m):          960 bars normal / 240 losing
                #   depth 5   (1m):              400 bars normal / 120 losing
                #   depth 6+  (sub-minute):      blocked from trading (Gate 0.5)
                _DEPTH_HOLD_NORMAL = {1:1440,2:1440,3:960,4:960,5:400,6:20,7:12,8:12,9:8,10:8,11:8,12:8}
                _DEPTH_HOLD_LOSING = {1:480, 2:480, 3:240,4:240,5:120,6:12,7:8, 8:8, 9:6, 10:6, 11:6, 12:6}
                if self.wave_rider.position is not None:
                    res = {'should_exit': False}   # default; overwritten by update_trail if called
                    belief_network.tick_trade_bar()
                    _bars_held   = max(1, int((ts_raw - active_entry_time) / 15))
                    _unrealized  = ((price - active_entry_price) if active_side == 'long'
                                    else (active_entry_price - price)) * self.asset.point_value
                    _max_hold    = (_DEPTH_HOLD_LOSING.get(active_entry_depth, 120) if _unrealized < 0
                                    else _DEPTH_HOLD_NORMAL.get(active_entry_depth, 400))
                    _hold_reason = 'emergency_close' if _unrealized < 0 else 'max_hold'
                    if _bars_held >= _max_hold:
                        _mh_pnl = ((price - active_entry_price) if active_side == 'long'
                                   else (active_entry_price - price)) * self.asset.point_value
                        outcome = TradeOutcome(
                            state=active_template_id,
                            entry_price=active_entry_price,
                            exit_price=price,
                            pnl=_mh_pnl,
                            result='WIN' if _mh_pnl > 0 else 'LOSS',
                            timestamp=ts_raw,
                            exit_reason=_hold_reason,
                            entry_time=active_entry_time,
                            exit_time=ts_raw,
                            duration=ts_raw - active_entry_time,
                            direction='LONG' if active_side == 'long' else 'SHORT',
                            template_id=active_template_id
                        )
                        self.wave_rider.position = None
                        self.brain.update(outcome)
                        day_trades.append(outcome)
                        if telemetry:
                            _hold_bars_t = _bar_i - getattr(self, '_telem_entry_bar', _bar_i)
                            _oracle_mfe_t = pending_oracle.get('oracle_mfe', 0) if pending_oracle else 0
                            _oracle_lbl_t = pending_oracle.get('oracle_label_name', '?') if pending_oracle else '?'
                            print(f"      <<< EXIT: {outcome.result} ${outcome.pnl:+.2f} "
                                  f"reason={outcome.exit_reason} hold={_hold_bars_t}bars "
                                  f"oracle_mfe={_oracle_mfe_t:.0f}t oracle={_oracle_lbl_t}")
                        _chart_day_pnl += outcome.pnl
                        _chart_day_trades += 1
                        _chart_day_wins += 1 if outcome.result == 'WIN' else 0
                        if pending_oracle and 'entry_workers' in pending_oracle:
                            eod_review_trades.append({
                                'side': active_side,
                                'pnl':  outcome.pnl,
                                'worker_snapshots': __import__('json').loads(pending_oracle['entry_workers'])
                            })
                        current_position_open = False
                        if _equity_enabled:
                            running_equity += outcome.pnl
                            peak_equity   = max(peak_equity, running_equity)
                            trough_equity = min(trough_equity, running_equity)
                            if running_equity < _NINJATRADER_MNQ_MARGIN:
                                account_ruined = True
                                ruin_day = ruin_day or day_date
                        if pending_oracle is not None:
                            _mh_pot = pending_oracle['oracle_mfe'] * self.asset.point_value
                            _mh_cap = outcome.pnl / _mh_pot if _mh_pot > 0 else 0.0
                            _mh_sig = belief_network.get_exit_signal(active_side)

                            # CST Analytics
                            _struc_list = pending_oracle.get('structural_integrity', [])
                            _thresh = pending_oracle.get('basin_threshold', 999.0)
                            if _thresh < 1e-6: _thresh = 4.5
                            _bar_death = -1
                            for i, d in enumerate(_struc_list):
                                if d > _thresh:
                                    _bar_death = i
                                    break
                            _bleed_bars = 0
                            _bleed_cost = 0.0
                            if _bar_death >= 0 and _bars_held > _bar_death:
                                _bleed_bars = _bars_held - _bar_death
                            _exit_idx = _bars_held - 1
                            _si_exit = _struc_list[_exit_idx] if 0 <= _exit_idx < len(_struc_list) else -1.0

                            oracle_trade_records.append({
                                **pending_oracle,
                                'exit_price':  price,
                                'exit_time':   ts_raw,
                                'hold_bars':   _bars_held,
                                'exit_reason': _hold_reason,
                                'actual_pnl':  outcome.pnl,
                                'oracle_potential_pnl': _mh_pot,
                                'capture_rate': round(min(_mh_cap, 9.99), 4),
                                'result': outcome.result,
                                'exit_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                                'exit_conviction':    _mh_sig.get('conviction', 0.0),
                                'exit_wave_maturity': _mh_sig.get('wave_maturity', 0.0),
                                'exit_signal_reason': _hold_reason,
                                # CST
                                'structural_integrity_at_exit': _si_exit,
                                'bar_of_structural_death': _bar_death,
                                'bleed_bars': _bleed_bars,
                                'bleed_cost': _bleed_cost,
                            })
                            if _pending_dm_idx is not None:
                                decision_matrix_records[_pending_dm_idx].update({
                                    'trade_result': outcome.result,
                                    'trade_pnl':    round(outcome.pnl, 2),
                                    'exit_reason':  _hold_reason,
                                })
                            pending_oracle = None
                            _pending_dm_idx = None
                            belief_network.clear_active_trade_timescale()
                            belief_network.clear_trade_context()
                            belief_network.stop_trade_tracking()
                    else:
                        # Build trade context for belief network + CST
                        _pos = self.wave_rider.position
                        _profit_ticks = ((price - active_entry_price) if active_side == 'long'
                                         else (active_entry_price - price)) / self.asset.tick_size

                        # Propagate trade side + P&L to all workers
                        belief_network.set_trade_context(active_side, _profit_ticks)

                        _trade_ctx = None
                        if _pos and _pos.tmpl_expected_mfe_ticks > 0:
                            _trade_ctx = {
                                'profit_ticks':      _profit_ticks,
                                'running_mfe_ticks': _pos.running_mfe_ticks,
                                'running_mae_ticks': _pos.running_mae_ticks,
                                'pct_mfe_captured':  _pos.running_mfe_ticks / _pos.tmpl_expected_mfe_ticks,
                                'pct_hold_elapsed':  (_pos.bars_in_trade / _pos.tmpl_expected_hold_bars
                                                      if _pos.tmpl_expected_hold_bars > 0 else 0.0),
                                'template_win_rate': _pos.tmpl_win_rate,
                            }

                        # Get exit signal with trade context (includes net_pressure)
                        _exit_sig = belief_network.get_exit_signal(
                            _pos.side, trade_context=_trade_ctx)

                        # C2. Decay Cascade Exit
                        _cascade = belief_network.get_decay_cascade()
                        if _cascade['should_exit']:
                            _exit_sig['decay_exit'] = True
                            _exit_sig['reason'] = f"decay_cascade_{_cascade['cascade_score']:.2f}"

                        # CST Check — pressure-controlled (grace + sigma modulation)
                        current_state = _states_map.get(_bar_i)
                        cst_broken = False
                        if current_state and _pos:
                            cst_broken = not self.wave_rider.check_structural_integrity(
                                current_state, profit_ticks=_profit_ticks,
                                net_pressure=_exit_sig.get('net_pressure', 0.0))

                        if cst_broken:
                            # Structural Abort (CST)
                            res = self._create_cst_exit_result(price, active_entry_price, active_side, self.asset.point_value)
                        else:
                            # Normal Trail/Gate logic
                            res = self.wave_rider.update_trail(price, current_state, ts_raw, exit_signal=_exit_sig)

                        # -- 1s inner loop: check wicks within this 15s bar --
                        if not res['should_exit'] and _has_1s:
                            _s0 = np.searchsorted(_1s_ts, ts_raw, side='left')
                            _s1 = np.searchsorted(_1s_ts, ts_raw + 15, side='left')
                            for _1s_i in range(_s0, _s1):
                                belief_network.tick_sub_resolution(
                                    tf_bar_idx_1s=_1s_i,
                                    tf_bar_idx_5s=_1s_i // 5
                                )
                                res_1s = self.wave_rider.check_stops_hilo(
                                    _1s_highs[_1s_i], _1s_lows[_1s_i], _1s_ts[_1s_i]
                                )
                                if res_1s['should_exit']:
                                    res = res_1s
                                    break

                    if res['should_exit']:
                        _exit_ts = res.get('exit_time', ts_raw)
                        outcome = TradeOutcome(
                            state=active_template_id,
                            entry_price=active_entry_price,
                            exit_price=res['exit_price'],
                            pnl=res['pnl'],
                            result='WIN' if res['pnl'] > 0 else 'LOSS',
                            timestamp=ts_raw,
                            exit_reason=res['exit_reason'],
                            entry_time=active_entry_time,
                            exit_time=_exit_ts,
                            duration=_exit_ts - active_entry_time,
                            direction='LONG' if active_side == 'long' else 'SHORT',
                            template_id=active_template_id
                        )
                        self.brain.update(outcome)
                        day_trades.append(outcome)
                        _chart_day_pnl += outcome.pnl
                        _chart_day_trades += 1
                        _chart_day_wins += 1 if outcome.result == 'WIN' else 0
                        if pending_oracle and 'entry_workers' in pending_oracle:
                            eod_review_trades.append({
                                'side': active_side,
                                'pnl':  outcome.pnl,
                                'worker_snapshots': __import__('json').loads(pending_oracle['entry_workers'])
                            })
                        current_position_open = False
                        self.wave_rider.position = None  # Bug fix: prevent ghost trades

                        # Update running equity after trade close
                        if _equity_enabled:
                            running_equity += outcome.pnl
                            peak_equity   = max(peak_equity, running_equity)
                            trough_equity = min(trough_equity, running_equity)
                            if running_equity < _NINJATRADER_MNQ_MARGIN:
                                account_ruined = True
                                ruin_day = ruin_day or day_date

                        # Complete oracle record for this trade
                        if pending_oracle is not None:
                            o_mfe = pending_oracle['oracle_mfe']
                            o_mae = pending_oracle['oracle_mae']
                            oracle_favorable = o_mfe if pending_oracle['direction'] == 'LONG' else o_mae
                            oracle_potential = oracle_favorable * self.asset.point_value
                            capture = outcome.pnl / oracle_potential if oracle_potential > 0 else 0.0
                            _exit_t = outcome.exit_time
                            _entry_t = pending_oracle['entry_time']

                            _exit_idx = max(0, int((_exit_t - _entry_t) / 15))
                            cst_stats = self._calculate_cst_analytics(pending_oracle, _exit_idx)

                            oracle_trade_records.append({
                                **pending_oracle,
                                'exit_price':  outcome.exit_price,
                                'exit_time':   _exit_t,
                                'hold_bars':   max(1, int((_exit_t - _entry_t) / 15)),  # 15s bars held
                                'exit_reason': outcome.exit_reason,
                                'actual_pnl':  outcome.pnl,
                                'oracle_potential_pnl': oracle_potential,
                                'capture_rate': round(min(capture, 9.99), 4),
                                'result': outcome.result,
                                # Worker snapshot at exit: compare vs entry_workers to find direction flips
                                'exit_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                                # Dynamic Exit Fields
                                # adjustment_reason = the belief signal that LAST tightened/widened
                                # the trail (root cause). Falls back to current-bar signal if no
                                # adjustment happened (standard trail, PT, urgent flip).
                                'exit_conviction':    _exit_sig.get('conviction', 0.0),
                                'exit_wave_maturity': _exit_sig.get('wave_maturity', 0.0),
                                'exit_signal_reason': (res.get('adjustment_reason') or
                                                       _exit_sig.get('reason', '')),
                                # CST
                                **cst_stats
                            })
                            # Update signal-log record with trade outcome
                            if _pending_dm_idx is not None:
                                decision_matrix_records[_pending_dm_idx].update({
                                    'trade_result':       outcome.result,
                                    'trade_pnl':          round(outcome.pnl, 2),
                                    'exit_reason':        outcome.exit_reason,
                                    'exit_signal_reason': (res.get('adjustment_reason') or
                                                           _exit_sig.get('reason', '')),
                                    'exit_conviction':    _exit_sig.get('conviction', 0.0),
                                    'exit_wave_maturity': _exit_sig.get('wave_maturity', 0.0),
                                })
                            pending_oracle = None
                            _pending_dm_idx = None
                            belief_network.clear_active_trade_timescale()
                            belief_network.clear_trade_context()
                            belief_network.stop_trade_tracking()

                # 2. Check for entries (if no position)
                # Equity ruin check: simulation ends when equity hits 0 (no money to trade).
                if _equity_enabled and account_ruined:
                    break   # stop processing this day's bars entirely
                if _is_warmup:
                    pass  # Warmup: workers tick but no trades
                elif not current_position_open and ts in pattern_map:
                    candidates = pattern_map[ts]
                    if telemetry:
                        _t_dt = _dt_chart.datetime.utcfromtimestamp(ts_raw)
                        _t_bel = belief_network.get_belief()
                        _t_snap = belief_network.get_worker_snapshot()
                        print(f"\n    [{_t_dt.strftime('%Y-%m-%d %H:%M')} UTC] "
                              f"{len(candidates)} candidate(s) @ ${price:.2f}")
                        if _t_bel:
                            print(f"      BELIEF: dir={_t_bel.direction} conv={_t_bel.conviction:.3f} "
                                  f"confident={_t_bel.is_confident} mfe={_t_bel.predicted_mfe:.1f} "
                                  f"wave_mat={_t_bel.wave_maturity:.2f}")
                        else:
                            print(f"      BELIEF: None (network cold)")
                        if _t_snap and isinstance(_t_snap, dict):
                            _tf_order = ['1h','30m','15m','5m','3m','1m','30s','15s','5s','1s']
                            for _tl in _tf_order:
                                _tw = _t_snap.get(_tl)
                                if not _tw or not isinstance(_tw, dict):
                                    continue
                                _td = 'LONG' if _tw.get('d', 0.5) > 0.5 else 'SHORT'
                                _tp = _tw.get('d', 0.5)
                                _tc = _tw.get('c', 0)
                                _tm = _tw.get('mfe', 0)
                                _twm = _tw.get('m', 0)
                                print(f"        {_tl:>4s}: {_td:<5s} p={_tp:.2f} "
                                      f"conv={_tc:.3f} mfe={_tm:.1f} mat={_twm:.2f}")
                    best_candidate = None
                    best_dist = 999.0
                    best_tid = None
                    _candidate_gate = {}    # id(p) -> gate that blocked it (for FN audit)
                    _bypass_candidate = None  # best Gate-1 reject for worker-conviction bypass
                    _bypass_dist      = 999.0
                    _gate_passers = {}      # id(p) -> record dict (for score_loser tracking)

                    best_leaf_node = None # For CST cell bounds

                    for p in candidates:
                        n_signals_seen += 1
                        # --- Gate 0: Headroom Gate (Nightmare Field Equation) ---
                        micro_z = abs(p.z_score)
                        micro_pattern = p.pattern_type

                        # Macro context
                        chain = getattr(p, 'parent_chain', [])
                        root_entry = chain[-1] if chain else None
                        macro_z = abs(root_entry.get('z', 0.0)) if root_entry else 0.0

                        should_skip = False
                        _skip_label = 'gate0'

                        # DATA-QUALITY OVERRIDE
                        _data_override = False
                        # Exception TIDs check requires matching a template first - moved to Gate 1

                        if not _data_override:
                            # RULE 1: No pattern = no trade
                            if not micro_pattern:
                                should_skip = True

                            # RULE 2: Noise zone (<0.5 sigma)
                            elif micro_z < 0.5:
                                should_skip = True
                                _skip_label = 'gate0_noise'

                            # RULE 3: Approach zone (0.5 - 2.0 sigma)
                            elif 0.5 <= micro_z < 2.0:
                                if micro_pattern == 'STRUCTURAL_DRIVE':
                                    if p.state.adx_strength < _ADX_TREND_CONFIRMATION or p.state.hurst_exponent < _HURST_TREND_CONFIRMATION:
                                        should_skip = True
                                        _skip_label = 'gate0_r3_struct'
                                elif micro_pattern == 'ROCHE_SNAP':
                                    should_skip = True
                                    _skip_label = 'gate0_r3_snap'

                            # RULE 4: Mean Reversion / Extreme zone (>= 2.0 sigma)
                            elif micro_z >= 2.0:
                                headroom = macro_z < 3.0
                                if micro_pattern == 'ROCHE_SNAP':
                                    if not headroom and micro_z > 3.0:
                                        should_skip = True
                                        _skip_label = 'gate0_r4_nightmare'
                                elif micro_pattern == 'STRUCTURAL_DRIVE':
                                    if not headroom:
                                        should_skip = True
                                        _skip_label = 'gate0_r4_struct'

                        if should_skip:
                            skip_headroom += 1
                            _candidate_gate[id(p)] = _skip_label
                            if telemetry:
                                print(f"      GATE0 BLOCK: {_skip_label} z={micro_z:.2f} pat={micro_pattern}")
                            decision_matrix_records.append(_dm_rec(
                                p, _skip_label, day_date, ts, micro_z, macro_z, micro_pattern))
                            continue

                        # Gate 0.5 REMOVED (Depth filter handled by tree structure)

                        # Gate 1: Hypervolume Tree Navigation
                        matched = self._navigate_hypervolume_tree(self.hypervolume_tree, p)

                        dist = 0.0 # No longer meaningful L2 distance
                        tid = None

                        if matched:
                            tmpl, leaf = matched
                            tid = tmpl.template_id

                            # Brain Gate
                            if self.brain.should_fire(tid, min_prob=0.05, min_conf=0.0):
                                p_depth = getattr(p, 'depth', 6)
                                tier_adj = _TIER_SCORE_ADJ.get(template_tier_map.get(tid, 3), 0.0)
                                depth_adj = _DEPTH_SCORE_ADJ.get(p_depth, 0.0)
                                score = p_depth + tier_adj + depth_adj # dist removed

                                if score < best_dist:
                                    best_dist = score
                                    best_candidate = p
                                    best_tid = tid
                                    best_leaf_node = leaf

                                _gate_passers[id(p)] = _dm_rec(
                                    p, 'score_loser', day_date, ts, micro_z, macro_z,
                                    micro_pattern, dist=dist,
                                    template_id=tid, tier=template_tier_map.get(tid, 3))
                            else:
                                skip_brain += 1
                                _candidate_gate[id(p)] = 'gate2'
                                if telemetry:
                                    print(f"      GATE2 BLOCK: brain reject tid={tid} tier={template_tier_map.get(tid,3)}")
                                decision_matrix_records.append(_dm_rec(
                                    p, 'gate2', day_date, ts, micro_z, macro_z,
                                    micro_pattern, dist=dist,
                                    template_id=tid, tier=template_tier_map.get(tid, 3)))
                        else:
                            # No match found in tree
                            # Track for worker bypass
                            if 1.0 < _bypass_dist: # Just pick first valid no-match as bypass candidate
                                _bypass_dist = 1.0
                                _bypass_candidate = p

                            skip_dist += 1
                            _candidate_gate[id(p)] = 'gate1'
                            decision_matrix_records.append(_dm_rec(
                                p, 'gate1', day_date, ts, micro_z, macro_z,
                                micro_pattern, dist=dist))

                    # ── Emit score_loser records (gate-passers that lost on score) ──
                    for _pid, _prec in _gate_passers.items():
                        if best_candidate is None or _pid != id(best_candidate):
                            decision_matrix_records.append(_prec)

                    # ── Worker-conviction bypass (Gate 1 override) ───────────────
                    _WORKER_BYPASS_CONV = 0.65
                    _bypass_belief = None
                    if best_candidate is None and _bypass_candidate is not None:
                        _bypass_belief = belief_network.get_belief()
                        if _bypass_belief is None or _bypass_belief.conviction < _WORKER_BYPASS_CONV:
                            _bypass_belief = None

                    if best_candidate:
                        # FIRE
                        pattern_dna = ''   # DNA tree disabled (self.dna_tree=None)
                        _belief = None     # assigned at belief_network.get_belief() below
                        params = self.pattern_library[best_tid]['params']
                        lib_entry = self.pattern_library[best_tid]

                        # ── Live feature vector ──
                        # Used for direction/MFE models.
                        # Need to scale it using the leaf node scaler (if we had it exposed).
                        # But PatternTemplate stores coefficients fitted on SCALED features?
                        # The HypervolumeNode has a 'scaler'.
                        # The leaf_node.scaler should be used to transform live features for regression.

                        _live_feat_raw = np.array(FractalClusteringEngine.extract_features(best_candidate))

                        # Use leaf node scaler if available, else dummy
                        _leaf_scaler = best_leaf_node.scaler if best_leaf_node and best_leaf_node.scaler else self.scaler
                        try:
                            _live_scaled = _leaf_scaler.transform([_live_feat_raw])[0]
                        except Exception:
                            _live_scaled = _live_feat_raw # Fallback

                        # Direction determination
                        # Use template direction (bias) or predicted direction
                        side = lib_entry.get('direction', '')
                        if not side:
                            # Fallback to z_score sign if template is ambiguous
                            side = 'long' if best_candidate.z_score <= 0 else 'short'

                        # ── Direction gate ──────────────────────────────────────────
                        _live_s   = best_candidate.state
                        _dmi_diff = (getattr(_live_s, 'dmi_plus',  0.0)
                                   - getattr(_live_s, 'dmi_minus', 0.0))

                        long_bias  = lib_entry.get('long_bias',  0.0)
                        short_bias = lib_entry.get('short_bias', 0.0)
                        _nn_marker = _effective_oracle(best_candidate)

                        _dir_source = 'template_bias'

                        # ── Direction gate ──────────────────────────────────────────
                        # Snowflake: Branch determines direction (Z<=0 -> Long, Z>0 -> Short)
                        # We retain DMI calc only for logging/diagnostics
                        _live_s   = best_candidate.state
                        _dmi_diff = (getattr(_live_s, 'dmi_plus',  0.0)
                                   - getattr(_live_s, 'dmi_minus', 0.0))

                        long_bias  = lib_entry.get('long_bias',  0.0)
                        short_bias = lib_entry.get('short_bias', 0.0)
                        _nn_marker = _effective_oracle(best_candidate)

                        _dir_source = 'snowflake_z'

                        # ── Gate 4: Direction Confidence ──────────────────────────
                        # Calculate P(LONG) from logistic model
                        _dir_coeff = lib_entry.get('dir_coeff')
                        _p_long = 0.5
                        if _dir_coeff is not None:
                            _logit = np.dot(_live_scaled, np.array(_dir_coeff)) + lib_entry.get('dir_intercept', 0.0)
                            _p_long = 1.0 / (1.0 + np.exp(-np.clip(_logit, -20, 20)))

                        _dir_conf = abs(_p_long - 0.5)

                        # Filter uncertain trades (unless bypass/exception? No, apply to all template trades)
                        if _dir_conf < DIRECTION_CONFIDENCE_THRESHOLD:
                            _candidate_gate[id(best_candidate)] = 'gate4_confidence'
                            _bc_mz = abs(best_candidate.z_score)
                            _bc_mac = abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0))
                            decision_matrix_records.append(_dm_rec(
                                best_candidate, 'gate4_confidence', day_date, ts,
                                _bc_mz, _bc_mac,
                                getattr(best_candidate, 'pattern_type', ''),
                                dist=best_dist,
                                conviction=_belief.conviction if _belief else 0.0,
                                template_id=best_tid,
                                tier=template_tier_map.get(best_tid, 3),
                                pattern_dna=str(pattern_dna) if pattern_dna else ''))
                            continue

                        # ── Path conviction gate (fractal belief network) ─────────
                        # Collect all 8 TF workers' current beliefs.
                        # If the fractal tree is uncertain (conviction < threshold) -> skip.
                        # If tree agrees but disagrees with our leaf direction -> flip.
                        # If tree agrees and TP from decision-level worker is better -> use it.
                        _belief = belief_network.get_belief()
                        if _belief is not None:
                            if not _belief.is_confident:
                                # Tree uncertain across scales -- skip this bar
                                skip_conviction += 1
                                _candidate_gate[id(best_candidate)] = 'gate3'
                                _bc_mz = abs(best_candidate.z_score)
                                _bc_mac = abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0))
                                decision_matrix_records.append(_dm_rec(
                                    best_candidate, 'gate3', day_date, ts,
                                    _bc_mz, _bc_mac,
                                    getattr(best_candidate, 'pattern_type', ''),
                                    dist=best_dist,
                                    conviction=_belief.conviction if _belief else 0.0,
                                    template_id=best_tid,
                                    tier=template_tier_map.get(best_tid, 3),
                                    pattern_dna=str(pattern_dna) if pattern_dna else ''))
                                continue
                            # Path direction override: belief network wins over z_score branch
                            # OOS: workers have +0.20 edge at 15m, +0.09 at 15s.
                            # IS forward pass: z_score sign was ~56% accurate (near random).
                            # Belief direction is a stronger signal than z_score sign alone.
                            if _belief.direction != side:
                                side = _belief.direction
                                _dir_source = 'belief_path'
                            # Use network's predicted MFE if better than leaf-level estimate
                            _network_tp = max(4, int(round(_belief.predicted_mfe))) if _belief.predicted_mfe > 2.0 else None
                        else:
                            _network_tp = None

                        # ── TELEMETRY: Direction + conviction reasoning ──
                        if telemetry:
                            _oracle_dir = 'LONG' if _nn_marker > 0 else ('SHORT' if _nn_marker < 0 else 'NOISE')
                            _our_dir = side.upper()
                            _dir_match = 'MATCH' if ((_nn_marker > 0 and side == 'long') or (_nn_marker < 0 and side == 'short')) else ('NOISE' if _nn_marker == 0 else 'WRONG')
                            _fk = f"{best_tid}_{side}"
                            _fc = _screening_fission.get(_fk, _screening_default) if _screening_fission else 'N/A'
                            print(f"      >>> CANDIDATE PASSED CONVICTION tid={best_tid} fission={_fc}")
                            print(f"          Oracle: {_ORACLE_LABEL_NAMES.get(_nn_marker,'?')} ({_oracle_dir}) | "
                                  f"We say: {_our_dir} (src={_dir_source}) | {_dir_match}")
                            print(f"          z={best_candidate.z_score:.2f} depth={getattr(best_candidate,'depth',0)} "
                                  f"p_long={_p_long:.3f} dir_conf={_dir_conf:.3f} "
                                  f"conv={_belief.conviction if _belief else 0:.3f} "
                                  f"dmi={_dmi_diff:+.2f}")
                            if _belief:
                                print(f"          belief_dir={_belief.direction} wave_mat={_belief.wave_maturity:.2f} "
                                      f"pred_mfe={_belief.predicted_mfe:.1f} levels={_belief.active_levels}")

                        # ── Gate 3.5: Screening gates (model fission + temporal) ──
                        if _screening_fission:
                            # Direction enforcement: check BOTH directions, pick the best
                            _FISSION_RANK = {'KEEP': 2, 'SPLIT': 1, 'DROP': 0}
                            _class_long  = _screening_fission.get(f"{best_tid}_long",  _screening_default)
                            _class_short = _screening_fission.get(f"{best_tid}_short", _screening_default)
                            _rank_long  = _FISSION_RANK.get(_class_long, 0)
                            _rank_short = _FISSION_RANK.get(_class_short, 0)

                            if _rank_long > _rank_short:
                                _fission_side = 'long'
                            elif _rank_short > _rank_long:
                                _fission_side = 'short'
                            else:
                                _fission_side = side  # tied — keep current direction

                            _fission_key = f"{best_tid}_{_fission_side}"
                            _fission_class = _screening_fission.get(_fission_key, _screening_default)

                            # Override direction if fission map disagrees with belief
                            if _fission_side != side:
                                if telemetry:
                                    print(f"          FISSION DIR OVERRIDE: {side}→{_fission_side} "
                                          f"(long={_class_long} short={_class_short})")
                                side = _fission_side
                                _dir_source = 'fission_map'

                            # Template gate: DROP = noise, gate out
                            if _fission_class == 'DROP':
                                skip_screening += 1
                                if telemetry:
                                    print(f"          GATE3.5 DROP: {_fission_key} not in KEEP/SPLIT")
                                _candidate_gate[id(best_candidate)] = 'gate3.5_drop'
                                _bc_mz = abs(best_candidate.z_score)
                                _bc_mac = abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0))
                                decision_matrix_records.append(_dm_rec(
                                    best_candidate, 'gate3.5_drop', day_date, ts,
                                    _bc_mz, _bc_mac,
                                    getattr(best_candidate, 'pattern_type', ''),
                                    dist=best_dist,
                                    conviction=_belief.conviction if _belief else 0.0,
                                    template_id=best_tid,
                                    tier=template_tier_map.get(best_tid, 3),
                                    pattern_dna=str(pattern_dna) if pattern_dna else ''))
                                continue

                            # Temporal gate: only trade during good hours
                            if _screening_good_hours is not None:
                                _utc_hour = _dt_chart.datetime.utcfromtimestamp(ts_raw).hour
                                if _utc_hour not in _screening_good_hours:
                                    skip_screening += 1
                                    _candidate_gate[id(best_candidate)] = 'gate3.5_temporal'
                                    _bc_mz = abs(best_candidate.z_score)
                                    _bc_mac = abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0))
                                    decision_matrix_records.append(_dm_rec(
                                        best_candidate, 'gate3.5_temporal', day_date, ts,
                                        _bc_mz, _bc_mac,
                                        getattr(best_candidate, 'pattern_type', ''),
                                        dist=best_dist,
                                        conviction=_belief.conviction if _belief else 0.0,
                                        template_id=best_tid,
                                        tier=template_tier_map.get(best_tid, 3),
                                        pattern_dna=str(pattern_dna) if pattern_dna else ''))
                                    continue

                        # ── Gate 4: P(profitable) from live DMI/momentum + template WR ──
                        _tmpl_wr = lib_entry.get('stats_win_rate', 0.5)
                        _p_prof = belief_network.compute_p_profitable(side, _tmpl_wr)
                        if _p_prof < 0.70:
                            _candidate_gate[id(best_candidate)] = 'gate4_probability'
                            _bc_mz = abs(best_candidate.z_score)
                            _bc_mac = abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0))
                            decision_matrix_records.append(_dm_rec(
                                best_candidate, 'gate4_probability', day_date, ts,
                                _bc_mz, _bc_mac,
                                getattr(best_candidate, 'pattern_type', ''),
                                dist=best_dist,
                                conviction=_belief.conviction if _belief else 0.0,
                                template_id=best_tid,
                                tier=template_tier_map.get(best_tid, 3),
                                pattern_dna=str(pattern_dna) if pattern_dna else ''))
                            continue

                        # ── Gate 5: Direction consensus ──────────────────────────
                        _consensus = belief_network.get_direction_consensus(side)
                        # Raised threshold from 0.55 to 0.60 for higher quality
                        if _consensus['confidence'] < CONSENSUS_CONFIDENCE_THRESHOLD:
                            skip_direction += 1
                            _candidate_gate[id(best_candidate)] = 'gate5_direction'
                            _bc_mz = abs(best_candidate.z_score)
                            _bc_mac = abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0))
                            decision_matrix_records.append(_dm_rec(
                                best_candidate, 'gate5_direction', day_date, ts,
                                _bc_mz, _bc_mac,
                                getattr(best_candidate, 'pattern_type', ''),
                                dist=best_dist,
                                conviction=_belief.conviction if _belief else 0.0,
                                template_id=best_tid,
                                tier=template_tier_map.get(best_tid, 3),
                                pattern_dna=str(pattern_dna) if pattern_dna else ''))
                            continue
                        # Consensus may flip direction
                        if _consensus['direction'] != side:
                            side = _consensus['direction']
                            _dir_source = 'consensus_flip'

                        # ── Exit sizing from per-cluster regression models ────────
                        # TWO-PHASE EXIT DESIGN
                        # Phase 1 (initial hard stop): wide enough to survive entry
                        #   noise at the Roche limit before mean reversion kicks in.
                        #   = mean_mae_ticks * 2.0  (cluster's avg adverse excursion x2)
                        # Phase 2 (trailing stop): activates once price has moved
                        #   trail_activation_ticks in our favour, then trails HWM.
                        #   = regression_sigma_ticks * 1.1  (OLS breathing room)
                        # Trail activation threshold = p25_mae_ticks * 0.5
                        #   (half the 25th-pct adverse excursion -- modest confirmation)
                        _reg_sigma = lib_entry.get('regression_sigma_ticks', 0.0)
                        _mean_mae  = lib_entry.get('mean_mae_ticks', 0.0)
                        _p75_mfe   = lib_entry.get('p75_mfe_ticks',  0.0)
                        _p25_mae   = lib_entry.get('p25_mae_ticks',  0.0)

                        # Phase 1: initial hard stop (wide)
                        # Anchor to p25_mae (25th-pct adverse excursion) * 3.0 so that
                        # outlier clusters with huge mean_mae don't produce runaway stops.
                        # Falls back to mean_mae * 2.0 when p25 is unavailable.
                        if _p25_mae > 2.0:
                            _sl_ticks = max(4, int(round(_p25_mae * 3.0)))
                        elif _mean_mae > 2.0:
                            _sl_ticks = max(4, int(round(_mean_mae * 2.0)))
                        else:
                            _sl_ticks = params.get('stop_loss_ticks', 20)

                        # A2. Widen Initial Stop-Loss for High-Conviction Entries
                        # Scale SL by entry conviction — high conviction = wider leash
                        if _belief is not None:
                            _conv = _belief.conviction
                            _sl_multiplier = 1.0 + CONVICTION_SL_MULTIPLIER * max(0, _conv - CONVICTION_SL_THRESHOLD)
                            _sl_ticks = int(_sl_ticks * _sl_multiplier)

                        # Phase 2: trailing stop distance (from HWM).
                        # Pre-fix used sigma*1.1 -> 3-tick trip-wire that collapsed to 2t via tightening.
                        # Now: sigma*2.5 (captures normal price noise within a trending move);
                        # floor at 8 ticks ($10/tick on MNQ = $80 minimum breathing room).
                        if _reg_sigma > 2.0:
                            _trail_ticks = max(8, int(round(_reg_sigma * 2.5)))
                        elif _mean_mae > 2.0:
                            _trail_ticks = max(8, int(round(_mean_mae * 1.5)))
                        else:
                            _trail_ticks = max(8, params.get('trailing_stop_ticks', 12))

                        # Trail activation: needs p25_mae * 0.6 profit ticks to engage.
                        # Pre-fix: 0.3 -> trail activated with only 3 ticks profit -> tight trail
                        # immediately fired on any oscillation. Now: 0.6 ensures trade is
                        # meaningfully in profit before trail takes over from hard SL.
                        _trail_act_ticks = (max(4, int(round(_p25_mae * 0.6)))
                                            if _p25_mae > 2.0
                                            else None)  # None = immediate (legacy)

                        # TP: network path prediction (highest priority, sees all scales)
                        #     -> per-bar OLS (leaf cluster model)
                        #     -> template p75 (historical average)
                        #     -> DOE param (last resort)
                        if _network_tp is not None:
                            _tp_ticks = _network_tp
                        else:
                            _mfe_coeff = lib_entry.get('mfe_coeff')
                            if _mfe_coeff is not None:
                                _pred_mfe_pts   = (np.dot(_live_scaled, np.array(_mfe_coeff))
                                                   + lib_entry.get('mfe_intercept', 0.0))
                                _pred_mfe_ticks = max(0.0, _pred_mfe_pts / 0.25)
                                _tp_ticks = max(4, int(round(_pred_mfe_ticks))) if _pred_mfe_ticks > 2.0 else (
                                    max(4, int(round(_p75_mfe))) if _p75_mfe > 2.0
                                    else params.get('take_profit_ticks', 50))
                            elif _p75_mfe > 2.0:
                                _tp_ticks = max(4, int(round(_p75_mfe)))
                            else:
                                _tp_ticks = params.get('take_profit_ticks', 50)

                        # ── Sub-minute three-body exit override (depth >= 6) ──
                        # Template stats produce 1000s-tick exits at sub-minute
                        # resolution (useless). When the quantum state is PURE
                        # (clean bands, z at a sigma level), compute exits
                        # directly from the physics: TP=center, SL=0.5σ out.
                        _cand_depth = getattr(best_candidate, 'depth', 5)
                        if _cand_depth >= 5:
                            _tb_sigma = getattr(best_candidate.state, 'sigma_fractal', 0.0)
                            _tb_z     = getattr(best_candidate.state, 'z_score', 0.0)
                            _tb_coh   = getattr(best_candidate.state, 'coherence', 0.0)
                            _tb_lz    = getattr(best_candidate.state, 'lagrange_zone', 'CHAOS')
                            _tb_hurst = getattr(best_candidate.state, 'hurst_exponent', 0.5)

                            _is_pure = (abs(_tb_z) >= 1.0
                                        and _tb_sigma > 0.0
                                        and _tb_coh >= 0.3
                                        and _tb_lz != 'CHAOS'
                                        and abs(_tb_hurst - 0.5) >= 0.08)

                            if _is_pure:
                                _sigma_t = _tb_sigma / self.asset.tick_size
                                _tp_ticks    = max(4, int(round(abs(_tb_z) * _sigma_t)))
                                _sl_ticks    = max(4, int(round(0.5 * _sigma_t)))
                                _trail_ticks = max(4, int(round(1.5 * _sigma_t)))
                                # Trail arms at 50% of TP — lets trade develop before protecting
                                _trail_act_ticks = max(4, int(round(0.5 * _tp_ticks)))
                            else:
                                _tp_ticks    = 20
                                _sl_ticks    = 8
                                _trail_ticks = 16
                                _trail_act_ticks = 10

                        # ── Equity risk gate ─────────────────────────────────
                        # When account_size is set, skip trades whose max loss
                        # (SL in $) would consume more than half the remaining
                        # equity. This prevents a single stop-out from wiping
                        # the account below the NinjaTrader margin floor.
                        _MAX_RISK_FRACTION = 0.50   # fraction of equity risked per trade
                        if _equity_enabled:
                            _max_loss_usd = _sl_ticks * self.asset.tick_size * self.asset.point_value
                            _max_risk_usd = running_equity * _MAX_RISK_FRACTION
                            if _max_loss_usd > _max_risk_usd:
                                skipped_ruin += 1
                                continue   # skip this trade — risk too large for current equity

                        # CST Props
                        _cst_ancestry = {
                            'timeframe': getattr(best_candidate, 'timeframe', '15s'),
                            'depth': getattr(best_candidate, 'depth', 0),
                            'parent_type': getattr(best_candidate, 'parent_type', ''),
                            'parent_chain': getattr(best_candidate, 'parent_chain', [])
                        }

                        # Template trade-awareness stats
                        _tmpl_avg_mfebar = self.pattern_library.get(best_tid, {}).get('avg_mfe_bar', 0.0)
                        _tmpl_p75_mfebar = self.pattern_library.get(best_tid, {}).get('p75_mfe_bar', 0.0)
                        _tmpl_mean_mfe   = lib_entry.get('mean_mfe_ticks', 0.0)
                        _tmpl_win_rate   = lib_entry.get('stats_win_rate', 0.5)

                        self.wave_rider.open_position(
                            entry_price=price,
                            side=side,
                            state=best_candidate.state,
                            stop_distance_ticks=_sl_ticks,
                            profit_target_ticks=_tp_ticks,
                            trailing_stop_ticks=_trail_ticks,
                            trail_activation_ticks=_trail_act_ticks,
                            template_id=best_tid,
                            cst_cell_min=getattr(best_leaf_node, 'cell_min_16d', None) if best_leaf_node else None,
                            cst_cell_max=getattr(best_leaf_node, 'cell_max_16d', None) if best_leaf_node else None,
                            cst_ancestry=_cst_ancestry,
                            tmpl_expected_mfe_ticks=_tmpl_mean_mfe,
                            tmpl_expected_hold_bars=_tmpl_avg_mfebar,
                            tmpl_win_rate=_tmpl_win_rate,
                            entry_time=ts_raw,
                        )
                        current_position_open = True
                        active_entry_price = price
                        active_entry_time = ts_raw
                        active_side = side
                        active_template_id = best_tid
                        active_entry_depth = getattr(best_candidate, 'depth', 5)
                        self._telem_entry_bar = _bar_i
                        if telemetry:
                            _oracle_dir_t = 'LONG' if _nn_marker > 0 else ('SHORT' if _nn_marker < 0 else 'NOISE')
                            _match_t = 'CORRECT' if ((_nn_marker > 0 and side == 'long') or (_nn_marker < 0 and side == 'short')) else ('NOISE' if _nn_marker == 0 else 'WRONG DIR')
                            _fk_t = f"{best_tid}_{side}"
                            _fc_t = _screening_fission.get(_fk_t, _screening_default) if _screening_fission else 'N/A'
                            print(f"\n      *** TRADE FIRED *** tid={best_tid} [{_fc_t}]")
                            print(f"          {side.upper()} @ ${price:.2f} | oracle={_ORACLE_LABEL_NAMES.get(_nn_marker,'?')} "
                                  f"({_oracle_dir_t}) → {_match_t}")
                            print(f"          MFE={best_candidate.oracle_meta.get('mfe',0):.0f}t "
                                  f"MAE={best_candidate.oracle_meta.get('mae',0):.0f}t "
                                  f"TP={_tp_ticks:.0f}t SL={_sl_ticks:.0f}t trail={_trail_ticks:.0f}t")
                            print(f"          dir_src={_dir_source} p_long={_p_long:.3f} "
                                  f"conv={_belief.conviction if _belief else 0:.3f} "
                                  f"depth={active_entry_depth}")
                        depth_traded[active_entry_depth] += 1
                        # Pass template time-scale + MFE magnitude to belief network
                        belief_network.set_active_trade_timescale(
                            _tmpl_avg_mfebar, _tmpl_p75_mfebar,
                            expected_mfe_ticks=_tmpl_mean_mfe)

                        # Start Physics Decay Tracking
                        # Pattern horizon T_k approx avg_mfe_bar (or 40 bars if unknown)
                        _horizon = int(_tmpl_avg_mfebar) if _tmpl_avg_mfebar > 0 else DEFAULT_DECAY_HORIZON
                        belief_network.start_trade_tracking(side, _bar_i, _horizon)

                        # Store oracle facts for this trade (linked at exit)
                        # Direction-gate diagnostic columns enable offline DOE sweep of
                        # bias_threshold without re-running the forward pass.
                        _live_state  = best_candidate.state
                        _dmi_at_entry = round(
                            getattr(_live_state, 'dmi_plus',  0.0)
                          - getattr(_live_state, 'dmi_minus', 0.0), 2)
                        _entry_depth = getattr(best_candidate, 'depth', 6)
                        pending_oracle = {
                            'template_id':      best_tid,
                            'direction':        'LONG' if side == 'long' else 'SHORT',
                            'entry_price':      price,
                            'entry_time':       ts,        # Unix timestamp (15s resolution)
                            'entry_depth':      _entry_depth,  # Fractal depth (1=daily,6=15s)
                            'oracle_label':     best_candidate.oracle_marker,
                            'oracle_label_name':_ORACLE_LABEL_NAMES.get(best_candidate.oracle_marker, 'UNKNOWN'),
                            'pattern_dna':      str(pattern_dna) if pattern_dna else '',
                            'dna_expectancy':   _get_dna_expectancy(best_candidate),
                            'oracle_mfe':       best_candidate.oracle_meta.get('mfe', 0.0),
                            'oracle_mae':       best_candidate.oracle_meta.get('mae', 0.0),
                            'oracle_mfe_bar':   best_candidate.oracle_meta.get('mfe_bar', -1),   # bar index where MFE peaked
                            'oracle_lookahead': best_candidate.oracle_meta.get('lookahead_bars', 0),
                            # Direction DOE diagnostics
                            'direction_source': _dir_source,
                            'long_bias':        round(long_bias,  4),
                            'short_bias':       round(short_bias, 4),
                            'dmi_diff':         _dmi_at_entry,
                            # Belief network diagnostics
                            'belief_active_levels': _belief.active_levels if _belief is not None else 0,
                            'belief_conviction':    round(_belief.conviction, 4) if _belief is not None else 0.0,
                            'wave_maturity':        round(_belief.wave_maturity, 4) if _belief is not None else 0.0,
                            'decision_wave_maturity': round(_belief.decision_wave_maturity, 4) if _belief is not None else 0.0,
                            # Per-worker snapshots at entry: each worker's dir_prob/conviction/wave_maturity/pred_mfe
                            # Stored as JSON string; parse with json.loads() for analysis.
                            # Compare entry_workers vs exit_workers to find who flipped direction.
                            'entry_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                            # Exit parameter diagnostics: actual TP/SL/trail ticks used
                            'entry_tp_ticks':    _tp_ticks,
                            'entry_sl_ticks':    _sl_ticks,
                            'entry_trail_ticks': _trail_ticks,
                            'entry_trail_act':   _trail_act_ticks if _trail_act_ticks is not None else 0,
                            # Entry geometry — for post-run bad-trade analysis
                            'entry_z_score':             round(getattr(best_candidate.state, 'z_score',                 0.0), 4),
                            'entry_lagrange_zone':        getattr(best_candidate.state, 'lagrange_zone',         'UNKNOWN'),
                            'entry_hurst':                round(getattr(best_candidate.state, 'hurst_exponent',        0.5), 4),
                            'entry_coherence':            round(getattr(best_candidate.state, 'coherence',              0.0), 4),
                            'entry_adx':                  round(getattr(best_candidate.state, 'adx_strength',           0.0), 2),
                            'entry_escape_prob':          round(getattr(best_candidate.state, 'escape_probability',     0.0), 4),
                            'entry_oscillation_coherence':round(getattr(best_candidate.state, 'oscillation_coherence',  0.0), 4),
                            'entry_momentum_strength':    round(getattr(best_candidate.state, 'momentum_strength',      0.0), 4),
                            'tmpl_avg_mfe_bar':           self.pattern_library.get(best_tid, {}).get('avg_mfe_bar', 0.0),
                            'structural_integrity':       best_candidate.oracle_meta.get('structural_integrity', []),
                            'basin_threshold':            lib_entry.get('basin_mean', 0.0) + 3.0 * lib_entry.get('basin_std', 0.0)
                        }

                        # Signal log: add 'traded' record, save index for outcome update
                        _bc_mz  = round(abs(best_candidate.z_score), 2)
                        _bc_mac = round(abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0)), 2)
                        _dm_entry = _dm_rec(
                            best_candidate, 'traded', day_date, ts,
                            _bc_mz, _bc_mac,
                            getattr(best_candidate, 'pattern_type', ''),
                            dist=best_dist,
                            conviction=round(_belief.conviction, 3) if _belief else 0.0,
                            template_id=best_tid,
                            tier=template_tier_map.get(best_tid, 3),
                            pattern_dna=str(pattern_dna) if pattern_dna else '')
                        _dm_entry['trade_direction'] = 'LONG' if side == 'long' else 'SHORT'
                        decision_matrix_records.append(_dm_entry)
                        _pending_dm_idx = len(decision_matrix_records) - 1

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
                                # Skip if PID regime covers this bar (handled by PID analyzer)
                                _s = p.state
                                _is_pid = (abs(_s.term_pid) >= 0.3
                                           and _s.oscillation_coherence >= 0.5
                                           and _s.adx_strength <= 30.0)
                                if _is_pid:
                                    continue

                                _om = _effective_oracle(p)
                                _meta = getattr(p, 'oracle_meta', {})
                                _fn_pot = (_meta.get('mfe', 0.0) if _om > 0 else _meta.get('mae', 0.0)) * self.asset.point_value

                                # Golden-path rule: if this candidate passed all gates but lost
                                # the score competition, it is NOT a missed opportunity -- we
                                # deliberately chose a better trade at this bar.
                                if id(p) in _gate_passers:
                                    score_loser_pnl += _fn_pot
                                    continue  # do NOT add to fn_oracle_records or fn_potential_pnl

                                audit_fn += 1
                                fn_potential_pnl += _fn_pot
                                fn_oracle_records.append({
                                    'timestamp':       ts,
                                    'depth':           getattr(p, 'depth', 6),
                                    'oracle_label':    _om,
                                    'oracle_label_name': _ORACLE_LABEL_NAMES.get(_om, '?'),
                                    'oracle_dir':      'LONG' if _om > 0 else 'SHORT',
                                    'fn_potential_pnl': round(_fn_pot, 2),
                                    'dna_expectancy':  _get_dna_expectancy(p),
                                    'reason':          'competed',
                                    'gate_blocked':    _candidate_gate.get(id(p), 'unknown'),
                                    'workers':         __import__('json').dumps(belief_network.get_worker_snapshot()),
                                })
                    elif _bypass_belief is not None:
                        # ── Worker-bypass trade ────────────────────────────────
                        # No cluster template matched (Gate 1) but belief conviction
                        # >= 0.65. Workers called the direction 85-100% correctly for
                        # these no-match signals -- fire using worker-derived params.
                        side         = _bypass_belief.direction   # 'long' or 'short'
                        _bp_sigma    = getattr(_bypass_candidate.state, 'sigma_fractal', 0.0)
                        _bp_sl_ticks = max(4, int(round(_bp_sigma / self.asset.tick_size * 1.5))) if _bp_sigma > 0 else 8
                        _bp_tp_ticks = (max(8, int(round(_bypass_belief.predicted_mfe)))
                                        if _bypass_belief.predicted_mfe > 2.0 else 20)
                        _bp_trail_ticks = None
                        _bp_trail_act   = None

                        # Three-body exit override for bypass depth >= 5
                        _bp_depth = getattr(_bypass_candidate, 'depth', 5)
                        if _bp_depth >= 5:
                            _bp_z     = getattr(_bypass_candidate.state, 'z_score', 0.0)
                            _bp_coh   = getattr(_bypass_candidate.state, 'coherence', 0.0)
                            _bp_lz    = getattr(_bypass_candidate.state, 'lagrange_zone', 'CHAOS')
                            _bp_hurst = getattr(_bypass_candidate.state, 'hurst_exponent', 0.5)
                            _bp_pure  = (abs(_bp_z) >= 1.0
                                         and _bp_sigma > 0.0
                                         and _bp_coh >= 0.3
                                         and _bp_lz != 'CHAOS'
                                         and abs(_bp_hurst - 0.5) >= 0.08)
                            if _bp_pure:
                                _bp_sigma_t    = _bp_sigma / self.asset.tick_size
                                _bp_tp_ticks   = max(4, int(round(abs(_bp_z) * _bp_sigma_t)))
                                _bp_sl_ticks   = max(4, int(round(0.5 * _bp_sigma_t)))
                                _bp_trail_ticks = max(4, int(round(1.5 * _bp_sigma_t)))
                                _bp_trail_act   = max(4, int(round(0.5 * _bp_tp_ticks)))
                            else:
                                _bp_tp_ticks    = 20
                                _bp_sl_ticks    = 8
                                _bp_trail_ticks = 16
                                _bp_trail_act   = 10

                        _bypass_risk_ok = True
                        if _equity_enabled:
                            _max_loss_usd = _bp_sl_ticks * self.asset.tick_size * self.asset.point_value
                            if _max_loss_usd > running_equity * 0.50:
                                skipped_ruin += 1
                                _bypass_risk_ok = False

                        if _bypass_risk_ok:
                            self.wave_rider.open_position(
                                entry_price=price,
                                side=side,
                                state=_bypass_candidate.state,
                                stop_distance_ticks=_bp_sl_ticks,
                                profit_target_ticks=_bp_tp_ticks,
                                trailing_stop_ticks=_bp_trail_ticks,
                                trail_activation_ticks=_bp_trail_act,
                                template_id=-1,
                                entry_time=ts_raw,
                            )
                            current_position_open = True
                            active_entry_price    = price
                            active_entry_time     = ts_raw
                            active_side           = side
                            active_template_id    = -1
                            active_entry_depth    = getattr(_bypass_candidate, 'depth', 5)
                            depth_traded[active_entry_depth] += 1
                            belief_network.set_active_trade_timescale(0.0, 0.0)
                            belief_network.start_trade_tracking(side, _bar_i, 40) # Default horizon for bypass
                            pending_oracle = {
                                'template_id':      -1,
                                'direction':        'LONG' if side == 'long' else 'SHORT',
                                'entry_price':      price,
                                'entry_time':       ts,
                                'entry_depth':      getattr(_bypass_candidate, 'depth', 6),
                                'oracle_label':     _effective_oracle(_bypass_candidate),
                                'oracle_label_name': 'WORKER_BYPASS',
                                'oracle_mfe':       getattr(_bypass_candidate, 'oracle_meta', {}).get('mfe', 0.0),
                                'oracle_mae':       getattr(_bypass_candidate, 'oracle_meta', {}).get('mae', 0.0),
                                'long_bias':        0.0,
                                'short_bias':       0.0,
                                'dmi_diff':         round(
                                    getattr(_bypass_candidate.state, 'dmi_plus',  0.0)
                                  - getattr(_bypass_candidate.state, 'dmi_minus', 0.0), 2),
                                'belief_active_levels': _bypass_belief.active_levels,
                                'belief_conviction':    round(_bypass_belief.conviction, 4),
                                'wave_maturity':        round(_bypass_belief.wave_maturity, 4),
                                'decision_wave_maturity': round(_bypass_belief.decision_wave_maturity, 4),
                                'entry_workers':    __import__('json').dumps(belief_network.get_worker_snapshot()),
                                # Entry geometry — for post-run bad-trade analysis
                                'entry_z_score':             round(getattr(_bypass_candidate.state, 'z_score',                 0.0), 4),
                                'entry_lagrange_zone':        getattr(_bypass_candidate.state, 'lagrange_zone',         'UNKNOWN'),
                                'entry_hurst':                round(getattr(_bypass_candidate.state, 'hurst_exponent',        0.5), 4),
                                'entry_coherence':            round(getattr(_bypass_candidate.state, 'coherence',              0.0), 4),
                                'entry_adx':                  round(getattr(_bypass_candidate.state, 'adx_strength',           0.0), 2),
                                'entry_escape_prob':          round(getattr(_bypass_candidate.state, 'escape_probability',     0.0), 4),
                                'entry_oscillation_coherence':round(getattr(_bypass_candidate.state, 'oscillation_coherence',  0.0), 4),
                                'entry_momentum_strength':    round(getattr(_bypass_candidate.state, 'momentum_strength',      0.0), 4),
                                'tmpl_avg_mfe_bar':           0.0,  # no template for bypass
                            }
                            # AUDIT: count bypass trade (Bug fix: bypasses were missing from audit totals)
                            _bp_om = pending_oracle['oracle_label']
                            _bp_dir = pending_oracle['direction']
                            if _bp_om == 0:
                                audit_fp_noise += 1
                            elif (_bp_dir == 'LONG' and _bp_om > 0) or (_bp_dir == 'SHORT' and _bp_om < 0):
                                audit_tp += 1
                            else:
                                audit_fp_wrong += 1
                            # Signal log: bypass trade record
                            _bp_mz  = round(abs(_bypass_candidate.z_score), 2)
                            _bp_mac = round(abs((getattr(_bypass_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0)), 2)
                            _dm_bypass = _dm_rec(
                                _bypass_candidate, 'traded', day_date, ts,
                                _bp_mz, _bp_mac,
                                getattr(_bypass_candidate, 'pattern_type', ''),
                                dist=_bypass_dist,
                                conviction=round(_bypass_belief.conviction, 3),
                                template_id=-1, tier='bypass')
                            _dm_bypass['trade_direction'] = 'LONG' if side == 'long' else 'SHORT'
                            decision_matrix_records.append(_dm_bypass)
                            _pending_dm_idx = len(decision_matrix_records) - 1
                    else:
                        # Audit all candidates as SKIPPED
                        for p in candidates:
                            audit_res = _audit_trade(None, p)
                            if audit_res['classification'] == 'TN':
                                audit_tn += 1
                            elif audit_res['classification'] == 'FN':
                                audit_fn += 1
                                _om = _effective_oracle(p)  # macro-to-leaf aggregated
                                _meta = getattr(p, 'oracle_meta', {})
                                _fn_pot = (_meta.get('mfe', 0.0) if _om > 0 else _meta.get('mae', 0.0)) * self.asset.point_value
                                fn_potential_pnl += _fn_pot
                                fn_oracle_records.append({
                                    'timestamp':       ts,
                                    'depth':           getattr(p, 'depth', 6),
                                    'oracle_label':    _om,
                                    'oracle_label_name': _ORACLE_LABEL_NAMES.get(_om, '?'),
                                    'oracle_dir':      'LONG' if _om > 0 else 'SHORT',
                                    'fn_potential_pnl': round(_fn_pot, 2),
                                    'dna_expectancy':  _get_dna_expectancy(p),
                                    'reason':          'no_match',  # nothing passed gates at this bar
                                    'gate_blocked':    _candidate_gate.get(id(p), 'unknown'),
                                    'workers':         __import__('json').dumps(belief_network.get_worker_snapshot()),
                                })

            # End of day cleanup -- force close any open position
            if self.wave_rider.position is not None:
                pos = self.wave_rider.position
                # Get final exit signal for logging
                _eod_sig = belief_network.get_exit_signal(pos.side)
                _eod_adj_reason = pos.last_adjustment_reason or ''  # capture before clearing

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
                _chart_day_pnl += outcome.pnl
                _chart_day_trades += 1
                _chart_day_wins += 1 if outcome.result == 'WIN' else 0
                if pending_oracle and 'entry_workers' in pending_oracle:
                    eod_review_trades.append({
                        'side': active_side,
                        'pnl':  outcome.pnl,
                        'worker_snapshots': __import__('json').loads(pending_oracle['entry_workers'])
                    })

                # Update running equity after EOD close
                if _equity_enabled:
                    running_equity += outcome.pnl
                    peak_equity   = max(peak_equity, running_equity)
                    trough_equity = min(trough_equity, running_equity)
                    if running_equity < _NINJATRADER_MNQ_MARGIN:
                        account_ruined = True
                        ruin_day = ruin_day or day_date

                # Complete oracle record for EOD-forced close
                if pending_oracle is not None:
                    o_mfe = pending_oracle['oracle_mfe']
                    o_mae = pending_oracle['oracle_mae']
                    oracle_favorable = o_mfe if pending_oracle['direction'] == 'LONG' else o_mae
                    oracle_potential = oracle_favorable * self.asset.point_value
                    capture = outcome.pnl / oracle_potential if oracle_potential > 0 else 0.0
                    _eod_exit_t = ts
                    _eod_entry_t = pending_oracle['entry_time']

                    # CST Analytics
                    _struc_list = pending_oracle.get('structural_integrity', [])
                    _thresh = pending_oracle.get('basin_threshold', 999.0)
                    if _thresh < 1e-6: _thresh = 4.5
                    _bar_death = -1
                    for i, d in enumerate(_struc_list):
                        if d > _thresh:
                            _bar_death = i
                            break
                    _bleed_bars = 0
                    _bleed_cost = 0.0
                    _hb = max(1, int((_eod_exit_t - _eod_entry_t) / 15))
                    if _bar_death >= 0 and _hb > _bar_death:
                        _bleed_bars = _hb - _bar_death
                    _exit_idx = _hb - 1
                    _si_exit = _struc_list[_exit_idx] if 0 <= _exit_idx < len(_struc_list) else -1.0

                    oracle_trade_records.append({
                        **pending_oracle,
                        'exit_price':  outcome.exit_price,
                        'exit_time':   _eod_exit_t,
                        'hold_bars':   _hb,
                        'exit_reason': 'TIME_EXIT',
                        'actual_pnl':  outcome.pnl,
                        'oracle_potential_pnl': oracle_potential,
                        'capture_rate': round(min(capture, 9.99), 4),
                        'result': outcome.result,
                        'exit_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                        'exit_conviction':    _eod_sig.get('conviction', 0.0),
                        'exit_wave_maturity': _eod_sig.get('wave_maturity', 0.0),
                        'exit_signal_reason': (_eod_adj_reason or _eod_sig.get('reason', '')),
                        # CST
                        'structural_integrity_at_exit': _si_exit,
                        'bar_of_structural_death': _bar_death,
                        'bleed_bars': _bleed_bars,
                        'bleed_cost': _bleed_cost,
                    })
                    # Update signal-log record with trade outcome
                    if _pending_dm_idx is not None:
                        decision_matrix_records[_pending_dm_idx].update({
                            'trade_result':       outcome.result,
                            'trade_pnl':          round(outcome.pnl, 2),
                            'exit_reason':        'TIME_EXIT',
                            'exit_signal_reason': (_eod_adj_reason or _eod_sig.get('reason', '')),
                            'exit_conviction':    _eod_sig.get('conviction', 0.0),
                            'exit_wave_maturity': _eod_sig.get('wave_maturity', 0.0),
                        })
                    pending_oracle = None
                    _pending_dm_idx = None
                    belief_network.clear_active_trade_timescale()
                    belief_network.clear_trade_context()
                    belief_network.stop_trade_tracking()

            # Analyze day
            if _equity_enabled and account_ruined:
                print(f"\n  !! ACCOUNT RUINED on {day_date}: equity=${running_equity:.2f} < margin=${_NINJATRADER_MNQ_MARGIN:.2f}")
                print("  !! Stopping simulation -- no remaining capital to trade.")
                break  # stop the day loop entirely

            if day_trades:
                # Regret analysis (optional, or just stats)
                d_pnl = sum(t.pnl for t in day_trades)
                d_wins = sum(1 for t in day_trades if t.result == 'WIN')
                total_pnl += d_pnl
                total_trades += len(day_trades)
                total_wins += d_wins
                print(f"Trades: {len(day_trades)}, Wins: {d_wins}, PnL: ${d_pnl:.2f} ({time.perf_counter() - t_sim_start:.1f}s)")
            else:
                d_pnl = 0.0
                d_wins = 0
                print("No trades.")

            # Flush last calendar day of this file to dashboard
            if self.dashboard_queue and _chart_cur_day is not None:
                self.dashboard_queue.put({
                    'type': 'DAY_PNL',
                    'day': _chart_cur_day,
                    'pnl': _chart_day_pnl,
                    'trades': _chart_day_trades,
                    'wins': _chart_day_wins,
                })

            # Bridge regret markers to eod_review_trades for playbook learning
            if self.wave_rider.completed_reviews:
                _regret_map = {}
                for _rm in self.wave_rider.completed_reviews:
                    _rk = (_rm.side, round(_rm.actual_pnl, 2))
                    _regret_map[_rk] = _rm
                for _t in eod_review_trades:
                    _rk = (_t['side'], round(_t['pnl'], 2))
                    _rm = _regret_map.get(_rk)
                    if _rm:
                        _t['regret_type'] = _rm.regret_type
                        _t['exit_efficiency'] = _rm.exit_efficiency
                self.wave_rider.completed_reviews.clear()

            # Worker EOD self-review: update direction accuracy + DMI reliability + playbook
            if eod_review_trades:
                belief_network.end_of_day_review(eod_review_trades)

        # Final Report
        import datetime as _datetime
        _run_ts = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"FORWARD PASS COMPLETE  (run: {_run_ts})")
        _date_range = (
            f"  Period: {start_date or daily_files_15s[0] if daily_files_15s else 'N/A'} "
            f"to {end_date or (os.path.basename(daily_files_15s[-1]).replace('.parquet','') if daily_files_15s else 'N/A')} "
            f"({len(daily_files_15s)} files)"
        ) if daily_files_15s else ""
        if _date_range:
            report_lines.append(_date_range)
        report_lines.append(f"Total Trades: {total_trades}")
        report_lines.append(f"Win Rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "Win Rate: N/A")
        report_lines.append(f"Total PnL: ${total_pnl:.2f}")

        # ── Worker state counts diagnostic ─────────────────────────────────────
        _n_files = len(daily_files_15s) if daily_files_15s else 1
        _ws_parts = []
        for _lbl in ['1h','30m','15m','5m','3m','1m','30s','15s','5s','1s']:
            _tot = _worker_total_states.get(_lbl, 0)
            _days = _worker_days_with_data.get(_lbl, 0)
            _avg  = _tot // _n_files if _n_files else 0
            _ws_parts.append(f"{_lbl}={_avg}/file({_days}/{_n_files} files ok)")
        report_lines.append("")
        report_lines.append("── WORKER STATES LOADED ──")
        # Split into two lines so it's readable
        report_lines.append("  " + "  ".join(_ws_parts[:5]))
        report_lines.append("  " + "  ".join(_ws_parts[5:]))

        # ── Account equity summary (when --account-size is set) ──────────────
        if _equity_enabled:
            _max_dd_usd  = peak_equity - trough_equity
            _max_dd_pct  = (_max_dd_usd / account_size * 100.0) if account_size > 0 else 0.0
            _final_equity = running_equity
            report_lines.append("")
            report_lines.append("── ACCOUNT EQUITY SUMMARY ──")
            report_lines.append(f"  Start equity:      ${account_size:.2f}")
            report_lines.append(f"  Final equity:      ${_final_equity:.2f}")
            report_lines.append(f"  Peak equity:       ${peak_equity:.2f}")
            report_lines.append(f"  Trough equity:     ${trough_equity:.2f}")
            report_lines.append(f"  Max drawdown:      ${_max_dd_usd:.2f}  ({_max_dd_pct:.1f}% of start)")
            report_lines.append(f"  Trades skipped (risk too large for equity): {skipped_ruin}")
            if account_ruined:
                report_lines.append(f"  !! ACCOUNT RUINED on {ruin_day} -- equity fell below margin (${_NINJATRADER_MNQ_MARGIN:.0f})")
            else:
                report_lines.append(f"  Account status:    SURVIVED ({_final_equity:.2f} remaining)")

        report_lines.append("=" * 80)

        # ── ORACLE PROFIT ATTRIBUTION ────────────────────────────────────────────
        import csv as _csv

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("ORACLE PROFIT ATTRIBUTION")
        report_lines.append("=" * 80)

        # ── 1. Opportunity landscape ─────────────────────────────────────────────
        total_real_opps = audit_tp + audit_fp_wrong + audit_fn  # oracle said real move
        total_noise_opps = audit_fp_noise + audit_tn              # oracle said noise
        tp_potential  = sum(r['oracle_potential_pnl'] for r in oracle_trade_records if r['oracle_label'] != 0 and r['oracle_label_name'] not in ('NOISE',))

        # Golden Path Ideal (sequential, with gap decomposition)
        _gp = _compute_golden_path_ideal(
            oracle_trade_records=oracle_trade_records,
            fn_oracle_records=fn_oracle_records,
            bar_seconds=15,
            point_value=self.asset.point_value,
        )
        ideal_profit    = _gp['ideal']
        gp_missed_val   = _gp['gp_missed']    # value from golden-path signals we didn't trade
        gp_traded_val   = _gp['gp_traded']     # value from golden-path signals we traded (potential)
        gp_n_missed     = _gp['gp_n_missed']
        gp_n_traded     = _gp['gp_n_traded']
        gp_traded_entry_times = _gp['gp_traded_entry_times']

        # Exact gap decomposition: tag each trade as golden-path or not
        _gp_traded_recs  = [r for r in oracle_trade_records if r.get('entry_time') in gp_traded_entry_times]
        _non_gp_recs     = [r for r in oracle_trade_records if r.get('entry_time') not in gp_traded_entry_times]
        _gp_traded_actual = sum(r['actual_pnl'] for r in _gp_traded_recs)
        _non_gp_actual    = sum(r['actual_pnl'] for r in _non_gp_recs)
        gp_shortfall      = gp_traded_val - _gp_traded_actual   # underperformance on gp trades
        non_gp_drag       = -_non_gp_actual                      # waste from non-gp trades (positive = drag)

        # Parallel upper bound (unachievable sum of all signals)
        _parallel_bound = tp_potential + fn_potential_pnl + score_loser_pnl

        report_lines.append("")
        report_lines.append(f"  TOTAL SIGNALS SEEN BY ORACLE: {total_real_opps + total_noise_opps:,}")
        report_lines.append(f"    Real moves (MEGA/SCALP):  {total_real_opps:>6,}   -- worth ${_parallel_bound:>10,.2f} (parallel bound)")
        report_lines.append(f"    Noise (no real move):     {total_noise_opps:>6,}")

        # ── 2. What we did ───────────────────────────────────────────────────────
        n_traded   = len(oracle_trade_records)
        n_skipped  = audit_fn + audit_tn
        report_lines.append("")
        report_lines.append(f"  WHAT WE DID:")
        _total_sigs = total_real_opps + total_noise_opps
        report_lines.append(f"    Traded:  {n_traded:>6,}  ({n_traded/_total_sigs*100:.1f}% of all signals)" if _total_sigs else f"    Traded:  {n_traded:>6,}")
        report_lines.append(f"    Skipped: {n_skipped:>6,}  ({n_skipped/_total_sigs*100:.1f}% of all signals)" if _total_sigs else f"    Skipped: {n_skipped:>6,}")

        # ── 2b. Skip reason breakdown ─────────────────────────────────────────────
        # n_signals_seen counts individual candidates (not unique timestamps).
        # skip_xxx are per-candidate gate rejections.
        _n_pass = n_signals_seen - skip_headroom - skip_dist - skip_brain - skip_conviction - skip_screening - skip_direction
        report_lines.append("")
        report_lines.append(f"  WHY SIGNALS WERE SKIPPED  (total candidates evaluated: {n_signals_seen:,})")
        if n_signals_seen > 0:
            _pct_s = lambda n: f"{n/n_signals_seen*100:.1f}%"
            report_lines.append(f"    Gate 0 (headroom/pattern rule): {skip_headroom:>6,}  ({_pct_s(skip_headroom)})")
            report_lines.append(f"    Gate 1 (dist > 3.0, no match):  {skip_dist:>6,}  ({_pct_s(skip_dist)})")
            report_lines.append(f"    Gate 2 (brain rejected):        {skip_brain:>6,}  ({_pct_s(skip_brain)})")
            report_lines.append(f"    Gate 3 (conviction < thresh):   {skip_conviction:>6,}  ({_pct_s(skip_conviction)})")
            report_lines.append(f"    Gate 3.5 (screening fission/temporal): {skip_screening:>6,}  ({_pct_s(skip_screening)})")
            report_lines.append(f"    Gate 5 (direction unclear):     {skip_direction:>6,}  ({_pct_s(skip_direction)})")
            report_lines.append(f"    Passed all gates -> traded:     {n_traded:>6,}  ({_pct_s(n_traded)})")

        # ── 2c. Traded signal depth distribution ─────────────────────────────────
        # depth: 1=highest TF (daily/4h), 6=lowest TF (15s)
        # Answers: "Is it only trading 15s patterns and missing 1h+?"
        _DEPTH_LABELS = {1: '1=4h+  (high TF)', 2: '2=1h   ', 3: '3=15m  ',
                         4: '4=5m   ', 5: '5=1m   ', 6: '6=15s  (leaf)'}
        if depth_traded:
            report_lines.append("")
            report_lines.append(f"  TRADED SIGNAL DEPTH (which TF level triggered the trade):")
            for d in sorted(depth_traded.keys()):
                cnt = depth_traded[d]
                bar = '#' * min(40, cnt // max(1, n_traded // 40))
                report_lines.append(f"    depth {_DEPTH_LABELS.get(d, str(d))}: {cnt:>5,} trades  {bar}")

        # ── 2d. Wave maturity at entry ────────────────────────────────────────────
        # High decision_wave_maturity at entry = we entered a wave near exhaustion.
        if oracle_trade_records and 'decision_wave_maturity' in oracle_trade_records[0]:
            _wins  = [r for r in oracle_trade_records if r['result'] == 'WIN']
            _losses= [r for r in oracle_trade_records if r['result'] != 'WIN']
            _wm_w  = np.mean([r['decision_wave_maturity'] for r in _wins])  if _wins   else 0.0
            _wm_l  = np.mean([r['decision_wave_maturity'] for r in _losses]) if _losses else 0.0
            _wm_all= np.mean([r['decision_wave_maturity'] for r in oracle_trade_records])
            report_lines.append("")
            report_lines.append(f"  DECISION-TF WAVE MATURITY AT ENTRY  (0=fresh wave, 1=exhausted)")
            report_lines.append(f"    All trades:  avg={_wm_all:.3f}")
            report_lines.append(f"    Winners:     avg={_wm_w:.3f}")
            report_lines.append(f"    Losers:      avg={_wm_l:.3f}")
            report_lines.append(f"    Insight: maturity gap={_wm_l-_wm_w:.3f} (positive = entering losers at wave exhaustion)")

        # ── 2e. Per-depth PnL breakdown ──────────────────────────────────────────
        # Shows which fractal depth levels are actually profitable.
        # These stats feed into depth_weights.json for the next run.
        if oracle_trade_records and 'entry_depth' in oracle_trade_records[0]:
            from collections import defaultdict as _ddd
            _dp_pnl  = _ddd(float)
            _dp_cnt  = _ddd(int)
            _dp_wins = _ddd(int)
            for _r in oracle_trade_records:
                _d = _r.get('entry_depth', 6)
                _dp_pnl[_d]  += _r['actual_pnl']
                _dp_cnt[_d]  += 1
                _dp_wins[_d] += 1 if _r['result'] == 'WIN' else 0
            report_lines.append("")
            report_lines.append(f"  PER-DEPTH PnL BREAKDOWN (-> depth_weights.json for next run):")
            report_lines.append(f"    {'Depth':<8} {'Trades':>7} {'Win%':>7} {'Total PnL':>12} {'Avg/trade':>10}")
            report_lines.append(f"    {'-----':<8} {'------':>7} {'----':>7} {'---------':>12} {'---------':>10}")
            for _d in sorted(_dp_cnt.keys()):
                _cnt = _dp_cnt[_d]
                _tot = _dp_pnl[_d]
                _avg = _tot / _cnt
                _wr  = _dp_wins[_d] / _cnt * 100
                _flag = "  <- FILTER NEXT RUN" if _avg < 0 and _cnt >= 5 else ("  <- TOP" if _avg > 100 else "")
                report_lines.append(
                    f"    depth {_d:<3} {_cnt:>7,} {_wr:>6.0f}% ${_tot:>10,.2f} ${_avg:>9.2f}{_flag}")

        # ── 2f. Dynamic Exit Quality ─────────────────────────────────────────────
        if oracle_trade_records and 'exit_signal_reason' in oracle_trade_records[0]:
            report_lines.append("")
            report_lines.append(f"  DYNAMIC EXIT QUALITY:")

            # Buckets
            # Belief-flip exits: urgent_flip
            # Trail-tightened: low_conviction, wave_mature
            # Trail-widened: aligned_fresh
            # Standard trail: neutral, no_belief

            # Note: exit_signal_reason is the STATE at exit.
            # If exit_reason was 'belief_flip', it maps to 'Belief-flip exits'.
            # Otherwise we group by signal reason.

            b_flip     = [r for r in oracle_trade_records if r.get('exit_signal_reason') == 'urgent_flip']
            b_tight    = [r for r in oracle_trade_records if r.get('exit_signal_reason') in ('low_conviction', 'wave_mature')]
            b_widen    = [r for r in oracle_trade_records if r.get('exit_signal_reason') == 'aligned_fresh']
            b_standard = [r for r in oracle_trade_records if r.get('exit_signal_reason') in ('neutral', 'no_belief', '')]

            def _stats(subset):
                if not subset: return "0 trades"
                n = len(subset)
                avg = sum(r['actual_pnl'] for r in subset) / n
                return f"{n:>5} trades  ->  avg PnL ${avg:>7.2f}"

            report_lines.append(f"    Belief-flip exits:  {_stats(b_flip)}")
            report_lines.append(f"    Trail-tightened:    {_stats(b_tight)}")
            report_lines.append(f"    Trail-widened:      {_stats(b_widen)}")
            report_lines.append(f"    Standard trail:     {_stats(b_standard)}")

            # Time-scale exits (requires --fresh; will be empty until templates have avg_mfe_bar)
            b_time_ex  = [r for r in oracle_trade_records if r.get('exit_signal_reason') == 'time_exhausted']
            b_time_ti  = [r for r in oracle_trade_records if r.get('exit_signal_reason') == 'time_tighten']
            if b_time_ex or b_time_ti:
                report_lines.append(f"  TIME-SCALE EXIT SUMMARY:")
                report_lines.append(f"    time_exhausted exits: {_stats(b_time_ex)}")
                report_lines.append(f"    time_tighten exits:   {_stats(b_time_ti)}")
                report_lines.append(f"    (template-time-aware exits vs wave_mature/conviction exits)")

        # ── 2g. Worker agreement analysis ────────────────────────────────────────
        # For each TF worker: what fraction of the time did it agree with the
        # trade direction at entry? Compare wins vs losses to find who is predictive.
        # Also: which workers FLIPPED direction by exit? That likely caused the loss.
        if oracle_trade_records and 'entry_workers' in oracle_trade_records[0]:
            import json as _js
            _TF_ORDER = ['1h','30m','15m','5m','3m','1m','30s','15s','5s','1s']
            _wins_r  = [r for r in oracle_trade_records if r['result'] == 'WIN']
            _loss_r  = [r for r in oracle_trade_records if r['result'] != 'WIN']

            def _agree(snap_str, direction):
                """1 if the worker snapshot agrees with the trade direction, else 0."""
                try:
                    snap = _js.loads(snap_str) if snap_str else {}
                except Exception:
                    snap = {}
                count = 0
                for v in snap.values():
                    d = v.get('d', 0.5)
                    if (direction == 'LONG' and d > 0.5) or (direction == 'SHORT' and d < 0.5):
                        count += 1
                return count, len(snap)

            def _flipped(entry_str, exit_str, direction):
                """Number of workers that flipped direction between entry and exit."""
                try:
                    e = _js.loads(entry_str) if entry_str else {}
                    x = _js.loads(exit_str)  if exit_str  else {}
                except Exception:
                    return 0
                flips = 0
                for tf in set(e) & set(x):
                    e_long = e[tf].get('d', 0.5) > 0.5
                    x_long = x[tf].get('d', 0.5) > 0.5
                    if e_long != x_long:
                        flips += 1
                return flips

            report_lines.append("")
            report_lines.append(f"  WORKER AGREEMENT AT ENTRY  (agree = worker dir matches trade direction)")
            report_lines.append(f"    {'TF':<6} {'WIN agree':>10} {'LOSS agree':>11} {'Edge':>7}  <-- positive = worker is predictive")
            for tf in _TF_ORDER:
                w_agree = [_agree(r.get('entry_workers','{}'), r['direction'])[0]
                           for r in _wins_r if tf in (r.get('entry_workers') or '')]
                l_agree = [_agree(r.get('entry_workers','{}'), r['direction'])[0]
                           for r in _loss_r if tf in (r.get('entry_workers') or '')]
                # Per-TF dir_prob agree fraction
                def _tf_agree_frac(records, tf_label):
                    vals = []
                    for r in records:
                        try:
                            snap = _js.loads(r.get('entry_workers','{}') or '{}')
                        except Exception:
                            continue
                        if tf_label in snap:
                            d = snap[tf_label].get('d', 0.5)
                            direction = r['direction']
                            vals.append(1.0 if (direction=='LONG' and d>0.5) or (direction=='SHORT' and d<0.5) else 0.0)
                    return sum(vals)/len(vals) if vals else None
                wa = _tf_agree_frac(_wins_r, tf)
                la = _tf_agree_frac(_loss_r, tf)
                if wa is None and la is None:
                    continue
                wa_s = f"{wa:.2f}" if wa is not None else "  n/a"
                la_s = f"{la:.2f}" if la is not None else "  n/a"
                edge = (wa - la) if (wa is not None and la is not None) else 0.0
                flag = "  <-- key" if edge >= 0.10 else ""
                report_lines.append(f"    {tf:<6} {wa_s:>10} {la_s:>11} {edge:>+7.2f}{flag}")

            # Flip analysis: which workers changing direction predicts a loss?
            if 'exit_workers' in oracle_trade_records[0]:
                report_lines.append(f"")
                report_lines.append(f"  DIRECTION FLIPS BETWEEN ENTRY AND EXIT:")
                report_lines.append(f"    (worker flipped = changed LONG/SHORT conviction side during the trade)")
                for tf in _TF_ORDER:
                    def _flip_rate(records, tf_label):
                        rates = []
                        for r in records:
                            try:
                                e = _js.loads(r.get('entry_workers','{}') or '{}')
                                x = _js.loads(r.get('exit_workers', '{}') or '{}')
                            except Exception:
                                continue
                            if tf_label in e and tf_label in x:
                                e_long = e[tf_label].get('d', 0.5) > 0.5
                                x_long = x[tf_label].get('d', 0.5) > 0.5
                                rates.append(1.0 if e_long != x_long else 0.0)
                        return sum(rates)/len(rates) if rates else None
                    wfr = _flip_rate(_wins_r, tf)
                    lfr = _flip_rate(_loss_r, tf)
                    if wfr is None and lfr is None:
                        continue
                    wfr_s = f"{wfr:.2f}" if wfr is not None else "  n/a"
                    lfr_s = f"{lfr:.2f}" if lfr is not None else "  n/a"
                    diff  = (lfr - wfr) if (lfr is not None and wfr is not None) else 0.0
                    flag  = "  <-- flip predicts loss" if diff >= 0.15 else ""
                    report_lines.append(f"    {tf:<6} WIN flip={wfr_s}  LOSS flip={lfr_s}  diff={diff:>+.2f}{flag}")

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
        _pct = lambda n: f"{n/n_traded*100:.1f}%" if n_traded else "N/A"
        report_lines.append(f"    Correct direction:  {len(tp_recs):>6,}  ({_pct(len(tp_recs))})  ->  actual: ${tp_pnl:>10,.2f}")
        report_lines.append(f"    Wrong direction:    {len(fp_wrong_recs):>6,}  ({_pct(len(fp_wrong_recs))})  ->  losses: ${fp_wrong_pnl:>10,.2f}")
        report_lines.append(f"    Traded noise:       {len(fp_noise_recs):>6,}  ({_pct(len(fp_noise_recs))})  ->  losses: ${fp_noise_pnl:>10,.2f}")

        # ── 4. Exit quality on correct-direction trades ──────────────────────────
        # NOTE: "Reversed" trades had the correct oracle direction but the market
        # still moved against us after entry (capture_rate <= 0).  Their actual
        # losses are a distinct leakage bucket — NOT "wrong direction at entry"
        # (which means oracle label was opposite) — but also NOT mere underperformance.
        # They are shown separately in the profit gap as "reversed after entry".
        if tp_recs:
            optimal   = [r for r in tp_recs if r['capture_rate'] >= 0.80]
            too_early = [r for r in tp_recs if 0 < r['capture_rate'] < 0.20]
            reversed_ = [r for r in tp_recs if r['capture_rate'] <= 0]

            # Partial broken into 10% bands: 20-30, 30-40, ..., 70-80
            _partial_bands = []
            for _lo in range(20, 80, 10):
                _hi = _lo + 10
                _band = [r for r in tp_recs if _lo / 100 <= r['capture_rate'] < _hi / 100]
                _partial_bands.append((_lo, _hi, _band))
            partial = [r for r in tp_recs if 0.20 <= r['capture_rate'] < 0.80]  # kept for cross-tab

            # left_on_table: only non-reversed underperformance
            non_reversed = [r for r in tp_recs if r['capture_rate'] > 0]
            left_on_table = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl'])
                                for r in non_reversed)

            def _eq_row(label, recs, indent='    ', flag=''):
                n     = len(recs)
                total = sum(r['actual_pnl'] for r in recs)
                avg   = total / n if n else 0.0
                avg_h = sum(r.get('hold_bars', 0) for r in recs) / n if n else 0.0
                avg_c = sum(r.get('capture_rate', 0) for r in recs) / n if n else 0.0
                return (f"{indent}{label:<36} {n:>5,}  ${total:>10,.0f}  "
                        f"avg${avg:>7,.0f}  {avg_h:>5.0f}bars  cap{avg_c:>+6.0%}  {flag}")

            report_lines.append("")
            report_lines.append(f"  EXIT QUALITY (correct-direction trades):")
            report_lines.append(f"    {'Bucket':<36} {'n':>5}  {'Total PnL':>11}  {'Avg PnL':>8}  {'Hold':>9}  {'Cap%':>7}")
            report_lines.append(f"    {'─'*36} {'─'*5}  {'─'*11}  {'─'*8}  {'─'*9}  {'─'*7}")
            report_lines.append(_eq_row("Optimal  (>=80% captured)",         optimal))
            # Partial bands
            for _lo, _hi, _band in _partial_bands:
                if not _band:
                    continue
                report_lines.append(_eq_row(f"  Partial  ({_lo}-{_hi}% captured)", _band, indent='    '))
            report_lines.append(_eq_row("Too early (<20% captured)",         too_early))
            report_lines.append(_eq_row("Reversed (mkt flipped after entry)",reversed_, flag="<- leakage"))
            report_lines.append(f"    Left on table (non-reversed gap):                        ${left_on_table:>10,.0f}")

            # ── Exit reason cross-breakdown ───────────────────────────────────
            report_lines.append("")
            report_lines.append(f"  EXIT REASON -> QUALITY CROSS-BREAKDOWN (correct-direction trades):")
            all_reasons = sorted({r.get('exit_reason', 'unknown') for r in tp_recs})
            buckets_def = [
                ('Optimal',   optimal),
                ('Partial',   partial),
                ('Too early', too_early),
                ('Reversed',  reversed_),
            ]
            _hdr = f"    {'Exit reason':<18}" + "".join(f"  {b[0]:>9}" for b in buckets_def) + f"  {'Total':>6}  {'Avg PnL':>8}"
            report_lines.append(_hdr)
            report_lines.append(f"    {'─'*18}" + "  ─────────" * len(buckets_def) + "  ──────  ────────")
            for reason in all_reasons:
                reason_recs = [r for r in tp_recs if r.get('exit_reason') == reason]
                if not reason_recs:
                    continue
                avg_r = sum(r['actual_pnl'] for r in reason_recs) / len(reason_recs)
                row = f"    {str(reason):<18}"
                for _, brecs in buckets_def:
                    cnt = sum(1 for r in brecs if r.get('exit_reason') == reason)
                    row += f"  {cnt:>9,}"
                row += f"  {len(reason_recs):>6,}  ${avg_r:>7,.0f}"
                report_lines.append(row)

            # ── Per-bucket capture detail ─────────────────────────────────────
            report_lines.append("")
            report_lines.append(f"  CAPTURE DETAIL (correct-direction trades):")
            report_lines.append(f"    {'Bucket':<22}  {'Avg oracle MFE':>14}  {'Avg actual PnL':>14}  {'Avg hold bars':>13}")
            _detail_buckets = (
                [('Optimal', optimal), ('Too early', too_early), ('Reversed', reversed_)]
                + [(f'Partial {lo}-{hi}%', band) for lo, hi, band in _partial_bands if band]
            )
            for label, recs in _detail_buckets:
                if not recs:
                    continue
                avg_mfe_usd = sum(r.get('oracle_mfe', 0) for r in recs) / len(recs) * self.asset.point_value
                avg_act     = sum(r['actual_pnl'] for r in recs) / len(recs)
                avg_hb      = sum(r.get('hold_bars', 0) for r in recs) / len(recs)
                report_lines.append(f"    {label:<22}  ${avg_mfe_usd:>13,.0f}  ${avg_act:>13,.0f}  {avg_hb:>13.1f}")

            # ── Per-depth exit quality (hold shown as real time, not 15s bars) ──
            _depths_seen = sorted({r.get('entry_depth', 6) for r in tp_recs})
            if len(_depths_seen) > 1:
                _DL = {1:'1h+', 2:'1h', 3:'15m', 4:'5m', 5:'1m',
                       6:'30s', 7:'15s', 8:'15s', 9:'5s', 10:'5s', 11:'1s', 12:'1s'}
                report_lines.append("")
                report_lines.append(f"  EXIT QUALITY BY DEPTH  (hold = real time from 15s bars × 15):")
                report_lines.append(f"    {'Depth':<13} {'n':>4}  {'Optimal%':>8}  {'Reversed%':>9}  "
                                    f"{'Avg PnL':>8}  {'Avg Hold':>10}  {'Left$':>10}")
                report_lines.append(f"    {'─'*13} {'─'*4}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*10}  {'─'*10}")
                for _d in _depths_seen:
                    _dr = [r for r in tp_recs if r.get('entry_depth', 6) == _d]
                    _n  = len(_dr)
                    if not _n:
                        continue
                    _opt = sum(1 for r in _dr if r['capture_rate'] >= 0.80)
                    _rev = sum(1 for r in _dr if r['capture_rate'] <= 0)
                    _avg_pnl = sum(r['actual_pnl'] for r in _dr) / _n
                    _avg_hs  = int(sum(r.get('hold_bars', 0) for r in _dr) / _n * 15)
                    _h, _m   = _avg_hs // 3600, (_avg_hs % 3600) // 60
                    _hold_str = f"{_h}h{_m:02d}m"
                    _lot = sum(max(0, r.get('oracle_potential_pnl', 0) - r['actual_pnl'])
                               for r in _dr if r['capture_rate'] > 0)
                    _tf  = _DL.get(_d, f'd{_d}')
                    report_lines.append(
                        f"    depth {_d:>2} ({_tf:<3})  {_n:>4}  "
                        f"{_opt/_n:>7.0%}  {_rev/_n:>8.0%}  "
                        f"${_avg_pnl:>7,.0f}  {_hold_str:>10}  ${_lot:>9,.0f}"
                    )
        else:
            reversed_ = []
            left_on_table = 0.0

        # ── 5. Profit gap summary ────────────────────────────────────────────────
        # Per-trade quality stats (diagnostic — NOT for gap accounting)
        reversed_loss_val  = abs(sum(r['actual_pnl'] for r in reversed_))
        non_reversed_val   = [r for r in tp_recs if r['capture_rate'] > 0] if tp_recs else []
        left_on_table_val  = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl'])
                                 for r in non_reversed_val)

        # Exact 3-term gap decomposition (mathematically guaranteed to sum)
        # gap = gp_missed + gp_shortfall + non_gp_drag
        #   gp_missed    = golden-path signals we didn't trade at all
        #   gp_shortfall = underperformance on golden-path trades (potential - actual)
        #   non_gp_drag  = losses from trades NOT on the golden path (slot waste)
        _gap = ideal_profit - total_pnl if ideal_profit else 0.0

        report_lines.append("")
        report_lines.append(f"  PROFIT GAP ANALYSIS:")
        report_lines.append(f"    Ideal (golden-path sequential, perfect exits):  ${ideal_profit:>12,.2f}")
        report_lines.append(f"    Actual profit:                                  ${total_pnl:>12,.2f}")
        report_lines.append(f"    Gap:                                            ${_gap:>12,.2f}")
        report_lines.append(f"    Capture rate:                                   {total_pnl/ideal_profit*100:>10.1f}%" if ideal_profit else "    Capture rate: N/A")
        report_lines.append(f"    -----------------------------------------------------")
        if ideal_profit:
            report_lines.append(f"    WHERE THE GAP GOES  (3 terms, guaranteed to sum):")
            report_lines.append(f"      Missed signals (gate-blocked):              ${gp_missed_val:>12,.2f}  ({gp_missed_val/_gap*100:>5.1f}% of gap)  [{gp_n_missed:,} signals]" if _gap else f"      Missed signals: ${gp_missed_val:>12,.2f}")
            report_lines.append(f"      Trade underperformance (gp trades):         ${gp_shortfall:>12,.2f}  ({gp_shortfall/_gap*100:>5.1f}% of gap)  [{gp_n_traded:,} gp trades, actual=${_gp_traded_actual:>,.0f}]" if _gap else f"      Trade underperformance: ${gp_shortfall:>12,.2f}")
            report_lines.append(f"      Non-golden-path trade drag:                 ${non_gp_drag:>12,.2f}  ({non_gp_drag/_gap*100:>5.1f}% of gap)  [{len(_non_gp_recs):,} trades, net=${_non_gp_actual:>+,.0f}]" if _gap else f"      Non-gp drag: ${non_gp_drag:>12,.2f}")
            _check_sum = gp_missed_val + gp_shortfall + non_gp_drag
            report_lines.append(f"      -------")
            report_lines.append(f"      [check] Sum = ${_check_sum:>12,.2f}  (gap = ${_gap:>12,.2f}  diff = ${_gap - _check_sum:>+.2f})")
        report_lines.append(f"    -----------------------------------------------------")
        report_lines.append(f"    [info] Parallel upper bound:   ${_parallel_bound:>12,.2f}  (not achievable)")
        report_lines.append(f"    [info] Raw FN MFE sum:         ${fn_potential_pnl:>12,.2f}  (not sequential)")
        report_lines.append(f"    [info] Score-competition pool: ${score_loser_pnl:>12,.2f}  (took better same bar)")

        # ── 5b. Per-trade quality diagnostic (separate from gap accounting) ─────
        # These categories overlap and do NOT sum to the gap above.
        # Definitions:
        #   Wrong direction  = oracle said OPPOSITE direction to our trade
        #   Reversed         = oracle said SAME direction, but market moved against us after entry
        #   Noise            = oracle saw no real move at all
        #   Left on table    = uncaptured MFE on non-reversed correct-direction trades
        report_lines.append("")
        report_lines.append(f"    TRADE QUALITY DIAGNOSTIC  (all trades, does NOT sum to gap):")
        report_lines.append(f"      Wrong direction at entry:          ${abs(fp_wrong_pnl):>10,.2f}  [{len(fp_wrong_recs):,} trades -- oracle said opposite dir]")
        report_lines.append(f"      Reversed after correct entry:      ${reversed_loss_val:>10,.2f}  [{len(reversed_):,} trades -- correct dir but mkt flipped]")
        report_lines.append(f"      Noise trades (no real move):       ${abs(fp_noise_pnl):>10,.2f}  [{len(fp_noise_recs):,} trades]")
        report_lines.append(f"      Left on table (too early exits):   ${left_on_table_val:>10,.2f}  [non-reversed correct-dir underperformance]")

        # Store for bottom-line summary at program exit
        _summary = {
            'total_trades':    total_trades,
            'total_pnl':       total_pnl,
            'win_rate':        total_wins / total_trades if total_trades else 0.0,
            'n_days':          len(daily_files_15s),
            'date_start':      start_date or (os.path.basename(daily_files_15s[0]).replace('.parquet','') if daily_files_15s else '?'),
            'date_end':        end_date   or (os.path.basename(daily_files_15s[-1]).replace('.parquet','') if daily_files_15s else '?'),
            'pct_correct':     len(tp_recs) / n_traded * 100 if n_traded else 0.0,
            'pct_wrong':       len(fp_wrong_recs) / n_traded * 100 if n_traded else 0.0,
            'pct_noise':       len(fp_noise_recs) / n_traded * 100 if n_traded else 0.0,
            'pct_skipped':     n_skipped / (total_real_opps + total_noise_opps) * 100 if (total_real_opps + total_noise_opps) else 0.0,
            'ideal_profit':    ideal_profit,
            'missed':          gp_missed_val,
            'gp_shortfall':    gp_shortfall,
            'non_gp_drag':     non_gp_drag,
            'left_on_table':   left_on_table_val,
            'wrong_dir_loss':  abs(fp_wrong_pnl),
            'reversed_loss':   reversed_loss_val,
        }
        if oos_mode:
            self._oos_summary = _summary
        else:
            self._fp_summary = _summary

        # Send to dashboard
        if self.dashboard_queue:
            self.dashboard_queue.put({
                'type': 'ORACLE_ATTRIBUTION',
                'ideal':         ideal_profit,
                'actual':        total_pnl,
                'missed':        gp_missed_val,
                'gp_shortfall':  gp_shortfall,
                'non_gp_drag':   non_gp_drag,
            })
            _final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                      'step': 'FORWARD_PASS COMPLETE', 'pct': 100,
                                      'pnl': total_pnl, 'trades': total_trades,
                                      'wr': round(_final_wr, 1)})

        # ── Data Partitioning Helper ──────────────────────────────────────────
        def _write_partitioned_csv(records: list, base_filename: str):
            """
            Write monthly-sharded CSVs to reports/{mode}/shards/.
            Combined full-run files stay in reports/{mode}/.
            """
            if not records: return
            import csv, os
            from collections import defaultdict
            from datetime import datetime

            _shards_dir = os.path.join(_rpt_dir, 'shards')
            os.makedirs(_shards_dir, exist_ok=True)

            partitions = defaultdict(list)
            for r in records:
                ts = r.get('timestamp') or r.get('entry_time') or r.get('ts')
                day_str = r.get('day', '')

                month_key = 'unknown'
                if day_str:
                    if len(day_str) >= 6 and '_' not in day_str: month_key = f"{day_str[:4]}_{day_str[4:6]}"
                    elif '_' in day_str: month_key = day_str[:7]
                elif ts:
                    try: month_key = datetime.fromtimestamp(ts).strftime('%Y_%m')
                    except (ValueError, OSError, OverflowError): pass

                partitions[month_key].append(r)

            for month, month_records in partitions.items():
                name_parts = base_filename.rsplit('.', 1)
                part_name = f"{name_parts[0]}_{month}.{name_parts[1]}"
                part_path = os.path.join(_shards_dir, part_name)

                all_keys = list(dict.fromkeys(k for mr in month_records for k in mr.keys()))

                with open(part_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=all_keys)
                    writer.writeheader()
                    writer.writerows(month_records)
            print(f"  [EXPORT] {base_filename} -> {len(partitions)} monthly shards in {_shards_dir}")

        # ── 6. Save CSV ──────────────────────────────────────────────────────────
        if oracle_trade_records:
            # Write combined trade log to reports/{mode}/
            # Union all keys across records (bypass vs template trades have different fields)
            _all_keys = dict.fromkeys(k for rec in oracle_trade_records for k in rec)
            _csv_path = os.path.join(_rpt_dir, 'oracle_trade_log.csv')
            with open(_csv_path, 'w', newline='', encoding='utf-8') as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(_all_keys), restval='')
                _w.writeheader(); _w.writerows(oracle_trade_records)
            report_lines.append(f"  Per-trade oracle log saved: {_csv_path}")
            _write_partitioned_csv(oracle_trade_records, 'oracle_trade_log.csv')

        if pid_oracle_records:
            _pid_path = os.path.join(_rpt_dir, 'pid_oracle_log.csv')
            with open(_pid_path, 'w', newline='', encoding='utf-8') as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(pid_oracle_records[0].keys()))
                _w.writeheader(); _w.writerows(pid_oracle_records)
            report_lines.append(f"  PID oracle log saved: {_pid_path} ({len(pid_oracle_records):,} signals)")
            _write_partitioned_csv(pid_oracle_records, 'pid_oracle_log.csv')

        # ── Save FN oracle log + report section ──────────────────────────────────
        # fn_oracle_records: every missed real move with worker snapshot.
        # "competed" = another candidate was traded at the same bar.
        # "no_match" = nothing passed gates at this bar.
        # Key question: when workers agreed with oracle direction on FN signals,
        # a gate was too strict (the workers were right but we still skipped).
        # ── Decision Matrix CSV ─────────────────────────────────────────────────
        # Per-candidate log: every skipped signal with gate decision + oracle context.
        # Answers: which gate blocks the most money? which patterns are mis-directed?
        if decision_matrix_records:
            _dm_path = os.path.join(_rpt_dir, 'signal_log.csv')
            with open(_dm_path, 'w', newline='', encoding='utf-8') as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(decision_matrix_records[0].keys()))
                _w.writeheader(); _w.writerows(decision_matrix_records)
            _n_traded_dm  = sum(1 for r in decision_matrix_records if r['gate'] == 'traded')
            _n_skipped_dm = len(decision_matrix_records) - _n_traded_dm
            report_lines.append(f"  Signal log saved: {_dm_path}  ({_n_traded_dm} traded  +  {_n_skipped_dm:,} skipped)")
            _write_partitioned_csv(decision_matrix_records, 'signal_log.csv')

            # ── Decision matrix summary ───────────────────────────────────────────
            from collections import defaultdict as _dd
            _dm_gate_stats = _dd(lambda: {'count':0, 'real_move':0, 'total_opnl':0.0})
            for _r in decision_matrix_records:
                _g = _r['gate']
                _dm_gate_stats[_g]['count'] += 1
                if _r['oracle_label'] != 'NOISE':
                    _dm_gate_stats[_g]['real_move'] += 1
                    _dm_gate_stats[_g]['total_opnl'] += _r['oracle_pnl']
            report_lines.append("")
            report_lines.append("  DECISION MATRIX SUMMARY  (skipped candidates -- oracle $ = profit left on table)")
            report_lines.append(f"    {'Gate':<22} {'Count':>6} {'RealMove%':>10} {'Avg Oracle$':>12} {'Total Missed$':>14}")
            for _g, _st in sorted(_dm_gate_stats.items(), key=lambda x: -x[1]['total_opnl']):
                _rm_pct = _st['real_move'] / _st['count'] * 100 if _st['count'] else 0
                _avg_o  = _st['total_opnl'] / _st['real_move'] if _st['real_move'] else 0
                report_lines.append(
                    f"    {_g:<22} {_st['count']:>6} {_rm_pct:>9.1f}% {_avg_o:>11.0f} {_st['total_opnl']:>13.0f}")

        # ── DNA Path Summary ─────────────────────────────────────────────────────
        if oracle_trade_records and 'pattern_dna' in oracle_trade_records[0]:
            from collections import defaultdict as _dd_dna
            _dna_stats = _dd_dna(lambda: {'count':0, 'wins':0, 'pnl':0.0})
            for r in oracle_trade_records:
                dna = r.get('pattern_dna', '')
                if not dna: continue
                _dna_stats[dna]['count'] += 1
                _dna_stats[dna]['pnl']   += r['actual_pnl']
                if r['result'] == 'WIN':
                    _dna_stats[dna]['wins'] += 1

            report_lines.append("")
            report_lines.append(f"  TOP 10 DNA PATHS BY WIN RATE (min 10 trades):")
            report_lines.append(f"    {'DNA':<40} {'Trades':>6} {'Win%':>6} {'Avg PnL':>10}")

            _dna_list = []
            for dna, st in _dna_stats.items():
                if st['count'] >= 10:
                    wr = st['wins'] / st['count']
                    avg = st['pnl'] / st['count']
                    _dna_list.append((dna, st['count'], wr, avg))

            _dna_list.sort(key=lambda x: -x[2]) # Sort by Win% desc

            for dna, cnt, wr, avg in _dna_list[:10]:
                report_lines.append(f"    {dna:<40} {cnt:>6} {wr*100:>5.0f}% ${avg:>9.2f}")

        if fn_oracle_records:
            _fn_path = os.path.join(_rpt_dir, 'fn_oracle_log.csv')
            with open(_fn_path, 'w', newline='', encoding='utf-8') as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(fn_oracle_records[0].keys()))
                _w.writeheader(); _w.writerows(fn_oracle_records)
            report_lines.append(f"  FN oracle log saved: {_fn_path}  ({len(fn_oracle_records):,} missed real moves)")
            _write_partitioned_csv(fn_oracle_records, 'fn_oracle_log.csv')

            # FN worker agreement analysis:
            # For each TF worker, what fraction of FN signals had the worker
            # agreeing with the oracle direction?  High agreement = gate is blocking
            # moves the workers correctly identified.
            import json as _fnjs
            _TF_ORDER_FN = ['1h','30m','15m','5m','3m','1m','30s','15s','5s','1s']
            def _fn_tf_agree(records, tf_label):
                vals = []
                for r in records:
                    try:
                        snap = _fnjs.loads(r.get('workers', '{}') or '{}')
                    except Exception:
                        continue
                    if tf_label not in snap:
                        continue
                    d   = snap[tf_label].get('d', 0.5)
                    odir = r.get('oracle_dir', 'LONG')
                    vals.append(1.0 if (odir == 'LONG' and d > 0.5) or
                                       (odir == 'SHORT' and d < 0.5) else 0.0)
                return sum(vals) / len(vals) if vals else None

            fn_competed = [r for r in fn_oracle_records if r['reason'] == 'competed']
            fn_no_match = [r for r in fn_oracle_records if r['reason'] == 'no_match']

            report_lines.append("")
            report_lines.append(f"  FN WORKER AGREEMENT (did workers agree with oracle on missed moves?)")
            report_lines.append(f"    High agree% = workers called it right but a gate still blocked the trade")
            report_lines.append(f"    FN total={len(fn_oracle_records):,}  competed={len(fn_competed):,}  no_match={len(fn_no_match):,}")
            report_lines.append(f"    {'TF':<6} {'All FN':>8} {'Competed':>9} {'No-match':>9}  <- agree% with oracle dir")
            for tf in _TF_ORDER_FN:
                a_all  = _fn_tf_agree(fn_oracle_records, tf)
                a_comp = _fn_tf_agree(fn_competed, tf)
                a_nomatch = _fn_tf_agree(fn_no_match, tf)
                if a_all is None:
                    continue
                s_all   = f"{a_all:.2f}"
                s_comp  = f"{a_comp:.2f}"  if a_comp     is not None else "  n/a"
                s_nomatch = f"{a_nomatch:.2f}" if a_nomatch is not None else "  n/a"
                flag = "  <-- workers right, gate wrong" if (a_all or 0) > 0.60 else ""
                report_lines.append(f"    {tf:<6} {s_all:>8} {s_comp:>9} {s_nomatch:>9}{flag}")

            # FN gate breakdown: which gate is responsible for blocking profitable signals?
            # Shows how many FN records (missed real moves) were blocked by each gate.
            # gate0/gate0_5 = structural rules; gate1 = no cluster match; gate2/gate3 = brain/conviction
            # 'passed' = scored fine but another candidate at same bar was chosen instead
            _GATE_LABELS = {
                'gate0':             'Gate 0  no pattern (Rule 1)',
                'gate0_noise':       'Gate 0  noise zone <0.5sigma (Rule 2)',
                'gate0_r3_snap':     'Gate 0  approach zone ROCHE_SNAP (Rule 3) -- no qualified tmpl',
                'gate0_r3_struct':   'Gate 0  approach zone STRUCTURAL_DRIVE weak trend (Rule 3)',
                'gate0_r4_nightmare':'Gate 0  extreme zone nightmare field (Rule 4)',
                'gate0_r4_struct':   'Gate 0  extreme zone STRUCTURAL_DRIVE no headroom (Rule 4)',
                'gate0_5':           'Gate 0.5 depth filter',
                'gate1':             'Gate 1  no cluster match (dist>4.5)',
                'gate2':             'Gate 2  brain rejected',
                'gate3':             'Gate 3  conviction below threshold',
                'gate4_probability': 'Gate 4  P(profitable) < 0.70',
                'gate5_direction':   'Gate 5  direction consensus unclear',
                'passed':            'Passed gates, lost to better score',
                'unknown':           'Unknown (pre-gate tracking)',
            }
            _gate_counts = {}
            for _fr in fn_oracle_records:
                _gk = _fr.get('gate_blocked', 'unknown')
                _gate_counts[_gk] = _gate_counts.get(_gk, 0) + 1
            _fn_total = len(fn_oracle_records)
            report_lines.append("")
            report_lines.append(f"  FN GATE BREAKDOWN (which gate blocked profitable signals):")
            for _gk in ['gate0', 'gate0_noise', 'gate0_r3_snap', 'gate0_r3_struct',
                        'gate0_r4_nightmare', 'gate0_r4_struct',
                        'gate0_5', 'gate1', 'gate2', 'gate3',
                        'gate4_probability', 'gate5_direction',
                        'passed', 'unknown']:
                _gc = _gate_counts.get(_gk, 0)
                if _gc == 0 and _gk == 'unknown':
                    continue
                _pct = 100.0 * _gc / _fn_total if _fn_total else 0.0
                _lbl = _GATE_LABELS.get(_gk, _gk)
                flag2 = "  <-- main bottleneck" if _pct >= 40.0 else ""
                report_lines.append(f"    {_lbl:<42} {_gc:>6,}  ({_pct:5.1f}%){flag2}")

        # ── Compute and save per-depth weights for the NEXT run ──────────────────
        # score_adj is normalised relative to the best-performing depth so it
        # stays in a [-1, +1] range that is compatible with the existing dist/tier
        # scoring.  filter_out=True means the depth had negative avg PnL with at
        # least 5 trades — it will be excluded by Gate 0.5 next run.
        if oracle_trade_records and 'entry_depth' in oracle_trade_records[0]:
            from collections import defaultdict as _ddw
            import json as _json2
            _dw_pnl = _ddw(float)
            _dw_cnt = _ddw(int)
            for _r in oracle_trade_records:
                _d = _r.get('entry_depth', 6)
                _dw_pnl[_d] += _r['actual_pnl']
                _dw_cnt[_d] += 1
            # Best depth avg_pnl used for normalisation
            _best_avg = max((_dw_pnl[d] / _dw_cnt[d] for d in _dw_cnt), default=1.0)
            _best_avg = max(_best_avg, 1.0)   # guard against all-negative
            depth_weights_out = {}
            for _d in sorted(_dw_cnt.keys()):
                _cnt = _dw_cnt[_d]
                _avg = _dw_pnl[_d] / _cnt
                # Favour profitable depths: score_adj = -avg/best_avg (range ~[-1, 0])
                # Penalise losing depths:  score_adj = positive
                _sadj = round(-_avg / _best_avg, 4)
                depth_weights_out[str(_d)] = {
                    'avg_pnl':    round(_avg, 2),
                    'n_trades':   _cnt,
                    'score_adj':  _sadj,
                    'filter_out': bool(_avg < 0 and _cnt >= 5),
                }
            if oos_mode:
                # Preserve training depth weights -- OOS results should not overwrite
                # the model's learned depth preferences from the training period.
                report_lines.append("  Depth weights: NOT updated (oos_mode preserves training weights)")
            else:
                _dw_out_path = os.path.join(self.checkpoint_dir, 'depth_weights.json')
                with open(_dw_out_path, 'w') as _dw_f2:
                    _json2.dump(depth_weights_out, _dw_f2, indent=2)
                report_lines.append(f"  Depth weights saved: {_dw_out_path}")

        for line in report_lines:
            print(line)

        # Save report to reports/{mode}/
        report_path = os.path.join(_rpt_dir, 'phase4_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f"  Report saved to {report_path}")

        # ── Trade Analytics Suite ──────────────────────────────────────────────
        try:
            from training.trade_analytics import run_trade_analytics
            _oracle_csv = os.path.join(_rpt_dir, 'oracle_trade_log.csv')
            if os.path.exists(_oracle_csv):
                print("\n  [analytics] Running trade analytics suite...")
                _analytics_txt = run_trade_analytics(
                    log_path=_oracle_csv,
                    report_path=report_path,
                )
                _analytics_path = os.path.join(_rpt_dir, 'trade_analytics.txt')
                with open(_analytics_path, 'w', encoding='utf-8') as _af:
                    _af.write(_analytics_txt)
                print(f"  [analytics] Saved to {_analytics_path}")
        except Exception as _ae:
            print(f"  [analytics] Skipped (error: {_ae})")

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
        if self.dashboard_queue:
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Improve',
                                      'step': 'STRATEGY_SELECTION', 'pct': 0})

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
        # Use oracle_trade_log.csv (Phase 4 ground truth) instead of brain.trade_history
        # which can contain ghost trades from mixed phases.
        history_by_template = defaultdict(list)
        _oracle_log_path = os.path.join(_get_reports_dir('is'), 'oracle_trade_log.csv')
        if os.path.exists(_oracle_log_path):
            import csv as _csv5
            with open(_oracle_log_path, 'r') as f:
                reader = _csv5.DictReader(f)
                for row in reader:
                    _tid = row.get('template_id')
                    if _tid:
                        try:
                            _tid_int = int(_tid)
                            # Create a lightweight object with .pnl for downstream stats
                            _pnl = float(row.get('actual_pnl', 0))
                            _trade = type('OracleTrade', (), {'pnl': _pnl, 'template_id': _tid_int})()
                            history_by_template[_tid_int].append(_trade)
                        except (ValueError, TypeError):
                            pass
            print(f"  Loaded {sum(len(v) for v in history_by_template.values())} trades from oracle_trade_log.csv")
        else:
            print("  WARNING: oracle_trade_log.csv not found, falling back to brain.trade_history")
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

        # Save full tier map -- used by forward pass for candidate weighting
        tier_map = {r['id']: r['tier'] for r in report_data}
        tiers_path = os.path.join(self.checkpoint_dir, 'template_tiers.pkl')
        with open(tiers_path, 'wb') as f:
            pickle.dump(tier_map, f)

        # Tier summary
        from collections import Counter
        tier_counts = Counter(r['tier'] for r in report_data)
        rpt.append("")
        rpt.append("TIER SUMMARY:")
        for t in sorted(tier_counts.keys()):
            label = {1: 'PRODUCTION', 2: 'PROMISING', 3: 'UNPROVEN', 4: 'TOXIC'}.get(t, '?')
            rpt.append(f"  Tier {t} ({label}): {tier_counts[t]} templates")

        # Store for bottom-line summary
        top_t1 = sorted(
            [(r['id'], r['sharpe'], r['win_rate'], r['pnl'], r['trades'])
             for r in report_data if r['tier'] == 1],
            key=lambda x: -x[1]
        )[:5]
        self._tier_summary = {
            'tier_counts':  dict(tier_counts),
            'total':        len(report_data),
            'top_t1':       top_t1,
            'tier1_pnl':    sum(r['pnl'] for r in report_data if r['tier'] == 1),
        }

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

        # ── PARETO ANALYSIS ──────────────────────────────────────────────────
        # Read oracle_trade_log.csv and find the 20% of trades driving 80% of profit.
        # Dimensions: template, direction, oracle_label, time-of-day.
        import csv as _csv
        import datetime as _dt
        oracle_csv = os.path.join(_get_reports_dir('is'), 'oracle_trade_log.csv')
        if os.path.exists(oracle_csv):
            try:
                with open(oracle_csv, newline='', encoding='utf-8') as f:
                    rows = list(_csv.DictReader(f))

                # Only winning trades with positive actual_pnl
                profit_rows = [r for r in rows if float(r.get('actual_pnl', 0)) > 0]
                total_profit = sum(float(r['actual_pnl']) for r in profit_rows)

                rpt.append("")
                rpt.append("=" * 80)
                rpt.append("PARETO ANALYSIS  (top contributors to gross profit)")
                rpt.append("=" * 80)
                rpt.append(f"  Gross profit from winning trades: ${total_profit:,.2f}  "
                           f"({len(profit_rows):,} wins of {len(rows):,} total)")

                def _pareto_table(label, key_fn, top_n=10):
                    """Aggregate by key_fn, sort desc, find 80% threshold."""
                    from collections import defaultdict
                    buckets = defaultdict(float)
                    counts  = defaultdict(int)
                    for r in profit_rows:
                        k = key_fn(r)
                        buckets[k] += float(r['actual_pnl'])
                        counts[k]  += 1
                    ranked = sorted(buckets.items(), key=lambda x: -x[1])
                    if not ranked:
                        return
                    cum = 0.0
                    threshold_idx = len(ranked)
                    for i, (k, v) in enumerate(ranked):
                        cum += v
                        if cum >= total_profit * 0.80 and threshold_idx == len(ranked):
                            threshold_idx = i + 1

                    rpt.append(f"\n  -- {label} --")
                    rpt.append(f"  {'Key':<18} {'PnL':>10} {'Trades':>7} {'Cum%':>7}")
                    cum = 0.0
                    for i, (k, v) in enumerate(ranked[:top_n]):
                        cum += v
                        marker = " <-- 80%" if i + 1 == threshold_idx else ""
                        rpt.append(f"  {str(k):<18} ${v:>9,.2f} {counts[k]:>7,}  {cum/total_profit*100:>6.1f}%{marker}")
                    pct_keys = threshold_idx / max(len(ranked), 1) * 100
                    rpt.append(f"  => {threshold_idx} of {len(ranked)} keys ({pct_keys:.0f}%) drive 80% of profit")

                # By template
                _pareto_table("BY TEMPLATE",
                              lambda r: r.get('template_id', '?'))

                # By direction
                _pareto_table("BY DIRECTION",
                              lambda r: r.get('direction', '?'), top_n=4)

                # By oracle label
                _pareto_table("BY ORACLE LABEL",
                              lambda r: r.get('oracle_label_name', '?'), top_n=6)

                # By hour-of-day (entry_price timestamp not available; use row order proxy)
                # oracle_trade_log has no timestamp col -- skip if not present
                if rows and 'entry_price' in rows[0]:
                    pass  # no timestamp in log, skip hour breakdown

            except Exception as e:
                rpt.append(f"\n  (Pareto analysis skipped: {e})")
        else:
            rpt.append("")
            rpt.append("  (Pareto analysis: oracle_trade_log.csv not found -- run forward pass first)")

        # Print to console
        for line in rpt:
            print(line)

        # Save to file
        _p5_dir = _get_reports_dir('phase5')
        report_path = os.path.join(_p5_dir, 'phase5_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
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
            # Clear report subdirs (preserve benchmarks — historical)
            import shutil as _shutil_fresh
            if os.path.isdir(REPORTS_ROOT):
                for _sub in os.listdir(REPORTS_ROOT):
                    _sub_path = os.path.join(REPORTS_ROOT, _sub)
                    if _sub == 'benchmarks':
                        continue  # never wipe benchmark history
                    try:
                        if os.path.isdir(_sub_path):
                            _shutil_fresh.rmtree(_sub_path)
                        else:
                            os.remove(_sub_path)
                    except PermissionError:
                        print(f"  [WARN] Could not remove {_sub_path} (locked by OneDrive?)")
                print(f"  [CHECKPOINT] Cleared: reports/ (benchmarks preserved)")

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

        # Launch UI — popup by default, full dashboard with --dashboard, nothing with --no-dashboard
        if not self.config.no_dashboard and DASHBOARD_AVAILABLE:
            if getattr(self.config, 'dashboard', False):
                self.launch_dashboard()
            else:
                self.launch_popup()

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

            _train_end = getattr(self.config, 'train_end', None)
            if _train_end:
                print(f"  Out-of-sample guard: training data capped at {_train_end}")

            # Count TF dirs upfront so discovery callback can emit % progress
            _disc_n_tfs = max(1, sum(
                1 for tf in TIMEFRAME_HIERARCHY
                if isinstance(data_source, str) and os.path.isdir(os.path.join(data_source, tf))
            ))

            def _discovery_cb(lvl, tf, patterns, levels):
                ckpt.save_discovery_level(patterns, levels)
                if self.dashboard_queue:
                    _pct = min(99.0, len(levels) / _disc_n_tfs * 100)
                    self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                              'step': f'DISCOVERY  lvl {len(levels)}/{_disc_n_tfs}',
                                              'pct': round(_pct, 1)})

            manifest = self._run_discovery(
                data_source,
                checkpoint_callback=_discovery_cb,
                resume_manifest=partial_manifest,
                resume_levels=partial_levels,
                train_end=_train_end
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
            print("\nPhase 2.5: Building Hypervolume Tree...")
            ckpt.update_phase('clustering', 'in_progress')

            clustering_engine = FractalClusteringEngine(n_clusters=1000, max_variance=0.5)
            self.clustering_engine = clustering_engine  # Phase 3 needs it for re-aggregation
            if self.dashboard_queue:
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                          'step': 'CLUSTERING', 'pct': 0})

            # Build Tree: I-MR on PC1 → DBSCAN on full 16D → regression per template
            hypervolume_tree = clustering_engine.fit_hypervolume_tree(manifest)
            templates = clustering_engine.templates

            print(f"  Condensed {len(manifest)} raw patterns into {len(templates)} Tight Templates (Hypervolume Leaves).")

            # Save occurrence DataFrame (matrix markers audit trail)
            df_occurrences = clustering_engine.build_occurrence_dataframe(
                hypervolume_tree.roots, manifest)
            if len(df_occurrences) > 0:
                occ_path = os.path.join(self.checkpoint_dir, 'template_occurrences.parquet')
                df_occurrences.to_parquet(occ_path, index=False)
                # Print marker distribution summary
                ts_col = df_occurrences['timestamp']
                ts_range_days = (ts_col.max() - ts_col.min()) / 86400.0 if len(ts_col) > 1 else 0.0
                n_unique_templates = df_occurrences['template_id'].nunique()
                print(f"  Occurrence markers: {len(df_occurrences)} patterns, "
                      f"{n_unique_templates} templates, {ts_range_days:.0f} day spread")
                # Warn about temporally clustered templates
                for tid, grp in df_occurrences.groupby('template_id'):
                    ts_spread = (grp['timestamp'].max() - grp['timestamp'].min()) / 86400.0
                    if ts_spread < 7 and len(grp) > 5:
                        print(f"    WARNING: Template {tid} ({len(grp)} patterns) clustered in {ts_spread:.1f} days")
                print(f"  Saved template_occurrences.parquet")

            # Save Tree
            import pickle as _pickle
            tree_path = os.path.join(self.checkpoint_dir, 'hypervolume_tree.pkl')
            with open(tree_path, 'wb') as f:
                _pickle.dump(hypervolume_tree, f)
            print(f"  Saved Hypervolume Tree to {tree_path}")

            # Save templates list for Phase 3
            ckpt.save_templates(templates)

        # ===================================================================
        # PHASE 3: Validate DOE Groups + Register Templates
        # ===================================================================
        from training.fractal_clustering import DEFAULT_PID

        t_phase3_start = time.perf_counter()
        print(f"\nPhase 3: Validating {len(templates)} templates...")

        ckpt.update_phase('optimization', 'in_progress')
        self.pattern_library = self.pattern_library or {}

        validated = 0
        flagged = 0
        dropped = 0
        total_trimmed = 0
        for i, tmpl in enumerate(templates):
            n_before = len(tmpl.patterns)

            # Stepwise refinement: trims MFE outliers until consistent
            is_valid, score, diag = _validate_template_consistency(
                tmpl, tmpl.patterns, self.asset.point_value
            )
            tmpl.consistency_score = score
            tmpl.consistency_diagnostics = diag
            n_trimmed = diag.get('trimmed', 0)
            total_trimmed += n_trimmed

            # Re-aggregate oracle stats on the refined pattern set
            if n_trimmed > 0:
                if not hasattr(self, 'clustering_engine') or self.clustering_engine is None:
                    self.clustering_engine = FractalClusteringEngine(n_clusters=1000, max_variance=0.5)
                self.clustering_engine._aggregate_oracle_intelligence(tmpl)

            # Recompute analytical exits on refined stats
            tmpl.best_params = _analytical_exits(tmpl)
            tmpl.best_params.update(DEFAULT_PID)

            if not is_valid and len(tmpl.patterns) < 30:
                # Template couldn't be refined — drop it
                dropped += 1
                print(f"    {tmpl.template_id}: DROPPED ({len(tmpl.patterns)} patterns, "
                      f"CV={diag.get('mfe_cv', 0):.1f})")
                continue

            if is_valid:
                validated += 1
                tag = f"trimmed {n_trimmed}" if n_trimmed else "clean"
            else:
                flagged += 1
                tag = f"kept (CV={diag.get('mfe_cv', 0):.1f})"

            r2 = getattr(tmpl, 'adj_r2_mfe', 0.0)
            wr = getattr(tmpl, 'stats_win_rate', 0.0)
            mfe = getattr(tmpl, 'mean_mfe_ticks', 0.0)
            print(f"    {tmpl.template_id}: {n_before:>5}→{len(tmpl.patterns):>5} │ "
                  f"R²={r2:.2f}  WR={wr:.0%}  MFE={mfe:.0f}t │ {tag}")
            self.register_template_logic(tmpl, tmpl.best_params)

            # Progress popup
            if self.dashboard_queue:
                _p3_pct = min(99.0, (i + 1) / max(1, len(templates)) * 100)
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                          'step': f'VALIDATION  tmpl {i+1}/{len(templates)}',
                                          'pct': round(_p3_pct, 1)})

        phase3_elapsed = time.perf_counter() - t_phase3_start
        ckpt.update_phase('optimization', 'complete', {
            'validated': validated,
            'flagged': flagged,
            'dropped': dropped,
            'trimmed_patterns': total_trimmed,
        })

        print(f"\n  Phase 3 Summary:")
        print(f"    {validated} validated, {flagged} flagged, {dropped} dropped")
        print(f"    {total_trimmed:,} noisy patterns trimmed across all templates")
        print(f"    Library size: {len(self.pattern_library)} entries")
        print(f"    Time: {phase3_elapsed:.1f}s")

        # Save pattern library for Phase 4
        lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
        with open(lib_path, 'wb') as f:
            pickle.dump(self.pattern_library, f)

        # Split library (Snowflake)
        lib_long = {k: v for k, v in self.pattern_library.items() if v.get('direction') == 'LONG'}
        lib_short = {k: v for k, v in self.pattern_library.items() if v.get('direction') == 'SHORT'}

        with open(os.path.join(self.checkpoint_dir, 'pattern_library_long.pkl'), 'wb') as f:
            pickle.dump(lib_long, f)
        with open(os.path.join(self.checkpoint_dir, 'pattern_library_short.pkl'), 'wb') as f:
            pickle.dump(lib_short, f)

        print(f"  Saved pattern_library.pkl ({len(self.pattern_library)} entries)")
        print(f"  Saved split libraries: LONG={len(lib_long)}, SHORT={len(lib_short)}")

        # Build DNA Tree
        if manifest:
            print(f"\n  Building Fractal DNA Tree...")
            dna_tree = FractalDNATree(n_clusters_per_level=5)
            dna_tree.fit(manifest)

            with open(os.path.join(self.checkpoint_dir, 'fractal_dna_tree.pkl'), 'wb') as f:
                pickle.dump(dna_tree, f)

            print(f"  DNA tree built: {len(dna_tree._dna_index)} unique DNA paths")

        print("\n=== Training Complete ===")
        self.print_final_summary()
        return self.day_results

    def _run_discovery(self, data_source: Any,
                       checkpoint_callback=None,
                       resume_manifest=None,
                       resume_levels=None,
                       train_end: str = None) -> List[PatternEvent]:
        """
        Run top-down fractal discovery across ATLAS timeframes.
        If data_source points to the ATLAS root (contains TF subdirectories),
        uses hierarchical top-down scanning. Otherwise falls back to flat scan.

        train_end: YYYYMMDD cutoff -- files after this date are excluded from
                   discovery so the model has no knowledge of the test period.
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
                    resume_levels=resume_levels,
                    train_end=train_end
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
            'risk_score': template.risk_score,
            'direction': getattr(template, 'direction', ''),
            # Direction bias from oracle labels (used in forward pass direction gate)
            'long_bias': getattr(template, 'long_bias', 0.0),
            'short_bias': getattr(template, 'short_bias', 0.0),
            'stats_win_rate': getattr(template, 'stats_win_rate', 0.0),
            # Oracle exit calibration -- pattern's own price-breathing stats (in ticks)
            # regression_sigma_ticks: residual std of per-cluster OLS (MFE ~ |z|) -> trail = × 1.1
            # 0.0 means _aggregate_oracle_intelligence() didn't have enough members
            'mean_mfe_ticks':          getattr(template, 'mean_mfe_ticks',          0.0),
            'mean_mae_ticks':          getattr(template, 'mean_mae_ticks',          0.0),
            'p75_mfe_ticks':           getattr(template, 'p75_mfe_ticks',           0.0),
            'p25_mae_ticks':           getattr(template, 'p25_mae_ticks',           0.0),
            'risk_variance':           getattr(template, 'risk_variance',           0.0),
            'regression_sigma_ticks':  getattr(template, 'regression_sigma_ticks',  0.0),
            # mfe_coeff @ live_scaled_features + mfe_intercept  -> predicted MFE in price pts
            # sigmoid(dir_coeff @ live_scaled_features + dir_intercept) -> P(LONG)
            'mfe_coeff':     getattr(template, 'mfe_coeff',     None),
            'mfe_intercept': getattr(template, 'mfe_intercept', 0.0),
            'dir_coeff':     getattr(template, 'dir_coeff',     None),
            'dir_intercept': getattr(template, 'dir_intercept', 0.0),
            # Time-scale: bar index where MFE historically peaks (0.0 until --fresh with mfe_bar)
            'avg_mfe_bar':   getattr(template, 'avg_mfe_bar',   0.0),
            'p75_mfe_bar':   getattr(template, 'p75_mfe_bar',   0.0),
            # CST Basin
            'basin_mean':    getattr(template, 'basin_mean',    0.0),
            'basin_std':     getattr(template, 'basin_std',     0.0),
            # DOE Phase 3 consistency validation
            'consistency_score': getattr(template, 'consistency_score', 0.0),
            # Parent DNA Matching: per-TF DNA data for 15m anchor + multi-TF verification
            'tf_depth_map':   getattr(template, 'tf_depth_map', {}),
            'dna_centroids':  {tf: c.tolist() if hasattr(c, 'tolist') else c
                               for tf, c in getattr(template, 'dna_centroids', {}).items()},
            'dna_bounds_min': {tf: b.tolist() if hasattr(b, 'tolist') else b
                               for tf, b in getattr(template, 'dna_bounds_min', {}).items()},
            'dna_bounds_max': {tf: b.tolist() if hasattr(b, 'tolist') else b
                               for tf, b in getattr(template, 'dna_bounds_max', {}).items()},
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
    def launch_popup(self):
        """Launch lightweight progress popup in background thread (default UI)."""
        self.dashboard_thread = threading.Thread(target=launch_popup, args=(self.dashboard_queue,), daemon=True)
        self.dashboard_thread.start()
        print("Progress popup launching in background...")
        time.sleep(1)

    def launch_dashboard(self):
        """Launch full dashboard in background thread (opt-in via --dashboard)."""
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

    def run_param_sweep(self):
        """
        Post-hoc DOE: sweep filter combinations on oracle_trade_log.csv.
        Reads the log from the last forward pass and ranks every combination of:
          min_tier × direction × noise_filter
        by net PnL.  Runs in seconds -- no re-simulation needed.

        Usage:
            python training/orchestrator.py --sweep-params
        """
        from itertools import product as _product

        log_path   = os.path.join(_get_reports_dir('is'), 'oracle_trade_log.csv')
        tiers_path = os.path.join(self.checkpoint_dir, 'template_tiers.pkl')

        if not os.path.exists(log_path):
            print("ERROR: reports/is/oracle_trade_log.csv not found -- run a forward pass first.")
            return

        df = pd.read_csv(log_path)
        template_tier_map = {}
        if os.path.exists(tiers_path):
            with open(tiers_path, 'rb') as f:
                template_tier_map = pickle.load(f)
        df['tier'] = df['template_id'].map(template_tier_map).fillna(4).astype(int)

        total_trades = len(df)
        total_pnl    = df['actual_pnl'].sum()

        print("\n" + "="*80)
        print("PARAMETER SWEEP -- Post-Hoc DOE on oracle_trade_log.csv")
        print(f"  Baseline  : {total_trades:,} trades  |  Net PnL: ${total_pnl:,.0f}")
        print(f"  Sweep dims: min_tier × direction × noise_filter")
        print("="*80)

        # ── Sweep grid ────────────────────────────────────────────────────────
        min_tiers   = [1, 2, 3, 4]            # 4 = all tiers (baseline)
        directions  = ['all', 'SHORT', 'LONG']
        noise_modes = ['all', 'no_noise', 'mega_only']
        # Note: noise_filter is analysis-only (we can't know oracle_label in live trading).
        #       min_tier and direction ARE live-tradeable levers.

        rows = []
        for mt, dr, nm in _product(min_tiers, directions, noise_modes):
            sub = df[df['tier'] <= mt]
            if dr != 'all':
                sub = sub[sub['direction'] == dr]
            if nm == 'no_noise':
                sub = sub[sub['oracle_label'] != 0]
            elif nm == 'mega_only':
                sub = sub[sub['oracle_label'].isin([-2, 2])]
            n = len(sub)
            if n == 0:
                continue
            pnl  = sub['actual_pnl'].sum()
            wr   = (sub['actual_pnl'] > 0).mean() * 100
            avg  = pnl / n
            rows.append({
                'min_tier': mt, 'direction': dr, 'noise_filter': nm,
                'trades': n, 'net_pnl': round(pnl, 1),
                'win_rate': round(wr, 1), 'avg_pnl': round(avg, 2),
                'live_tradeable': (nm == 'all'),   # noise_filter not usable live
            })

        rows.sort(key=lambda r: r['net_pnl'], reverse=True)

        # ── Print section 1: live-tradeable combos hitting $50K+ ─────────────
        live_hits = [r for r in rows if r['live_tradeable'] and r['net_pnl'] >= 50_000]
        print(f"\n{'':2}LIVE-TRADEABLE combos  ≥ $50K  (noise_filter='all')")
        print(f"  {'min_tier':>8}  {'direction':>9}  {'trades':>7}  {'net_pnl':>10}  {'win_rate':>9}  {'avg$/trade':>10}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*10}")
        if live_hits:
            for r in live_hits:
                flag = "  <- RECOMMENDED" if r == live_hits[0] else ""
                print(f"  {r['min_tier']:>8}  {r['direction']:>9}  {r['trades']:>7,}  "
                      f"${r['net_pnl']:>9,.0f}  {r['win_rate']:>8.1f}%  ${r['avg_pnl']:>9.2f}{flag}")
        else:
            print("  (none -- tune thresholds or collect more data)")

        # ── Print section 2: all combos top 15 ───────────────────────────────
        print(f"\n{'':2}TOP 15 COMBINATIONS  (including analysis-only noise filters)")
        print(f"  {'min_tier':>8}  {'direction':>9}  {'noise_filter':>12}  {'trades':>7}  {'net_pnl':>10}  {'win_rate':>9}  {'avg$/trade':>10}  {'live?':>5}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*5}")
        for r in rows[:15]:
            live = 'YES' if r['live_tradeable'] else 'no'
            print(f"  {r['min_tier']:>8}  {r['direction']:>9}  {r['noise_filter']:>12}  {r['trades']:>7,}  "
                  f"${r['net_pnl']:>9,.0f}  {r['win_rate']:>8.1f}%  ${r['avg_pnl']:>9.2f}  {live:>5}")

        # ── Print section 3: per-tier PnL attribution ─────────────────────────
        print(f"\n{'':2}PER-TIER ATTRIBUTION  (all directions, no filter)")
        print(f"  {'tier':>4}  {'templates':>9}  {'trades':>7}  {'net_pnl':>10}  {'avg$/trade':>10}  {'win_rate':>9}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}")
        tier_counts = {}
        if template_tier_map:
            import collections as _col
            tier_counts = dict(_col.Counter(template_tier_map.values()))
        for t in [1, 2, 3, 4]:
            sub = df[df['tier'] == t]
            n   = len(sub)
            if n == 0:
                continue
            pnl = sub['actual_pnl'].sum()
            wr  = (sub['actual_pnl'] > 0).mean() * 100
            avg = pnl / n
            tmpl_count = tier_counts.get(t, 0)
            flag = '  <- DRAG' if pnl < 0 else ('  <- STAR' if t == 1 else '')
            print(f"  {t:>4}  {tmpl_count:>9}  {n:>7,}  ${pnl:>9,.0f}  ${avg:>9.2f}  {wr:>8.1f}%{flag}")

        # ── Print section 4: expectation grounding ───────────────────────────
        print(f"\n{'':2}OUT-OF-SAMPLE EXPECTATION GUIDE")
        print(f"  These numbers are IN-SAMPLE (trained on same data).")
        print(f"  For UNKNOWN future data, apply a realistic discount:")
        print(f"  {'Scenario':30s}  {'Discount':>8}  {'Projected PnL':>14}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*14}")
        best_live_pnl = live_hits[0]['net_pnl'] if live_hits else total_pnl
        for label, disc in [("Conservative (new regime / drawdown)", 0.30),
                             ("Realistic  (normal out-of-sample)",    0.50),
                             ("Optimistic (similar market regime)",    0.70)]:
            proj = best_live_pnl * disc
            flag = '  <- meets $50K target' if proj >= 50_000 else ''
            print(f"  {label:30s}  {disc*100:>7.0f}%  ${proj:>12,.0f}{flag}")
        print()
        print(f"  NOTE: The oracle gap ($69M ideal) is NEVER achievable in live trading.")
        print(f"  It assumes perfect entries + perfect exits on every move -- physically")
        print(f"  impossible. Ground truth baseline is in-sample net PnL × discount.")
        print("="*80)

        # ── Recommendation ────────────────────────────────────────────────────
        best_overall = rows[0] if rows else None

        # ── DIRECTION GATE DOE (requires oracle_trade_log with long_bias/short_bias/dmi_diff) ──
        has_dir_cols = all(c in df.columns for c in ('long_bias', 'short_bias', 'dmi_diff'))
        if has_dir_cols:
            print(f"\n{'='*80}")
            print("DIRECTION GATE DOE -- sweep bias_threshold × dmi_threshold")
            print(f"  Simulates direction changes offline using stored long_bias/short_bias/dmi_diff.")
            print(f"  PnL estimation: direction-flipped trades swap to avg-win / avg-loss.")
            print("="*80)

            avg_win  = df[df['actual_pnl'] > 0]['actual_pnl'].mean() if (df['actual_pnl'] > 0).any() else 10.0
            avg_loss = df[df['actual_pnl'] < 0]['actual_pnl'].mean() if (df['actual_pnl'] < 0).any() else -10.0

            def _sim_dir(lb, sb, dmi, bias_thresh, soft_thresh=0.10):
                """Recompute direction as the gate would with a given bias_threshold."""
                if lb >= bias_thresh:   return 'LONG'
                if sb >= bias_thresh:   return 'SHORT'
                if lb + sb >= soft_thresh:
                    return 'LONG' if lb >= sb else 'SHORT'
                return 'LONG' if dmi > 0 else 'SHORT'

            def _estimate_pnl(actual, new_dir, old_dir):
                """Approximate PnL if direction flipped."""
                if new_dir == old_dir:
                    return actual
                # Direction changed: swap WIN↔LOSS using dataset averages
                return avg_loss if actual > 0 else avg_win

            bias_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
            dmi_thresholds  = [0.0, 2.0, 5.0]   # min |dmi_diff| needed to use DMI vs skip

            dir_results = []
            for bt, dt in _product(bias_thresholds, dmi_thresholds):
                est_pnl = 0.0
                n_flipped = 0
                for _, row in df.iterrows():
                    lb  = row['long_bias']
                    sb  = row['short_bias']
                    dmi = row['dmi_diff']
                    # Determine DMI signal with threshold guard
                    eff_dmi = dmi if abs(dmi) >= dt else 0.0
                    new_dir = _sim_dir(lb, sb, eff_dmi, bt)
                    old_dir = row['direction']
                    est_pnl += _estimate_pnl(row['actual_pnl'], new_dir, old_dir)
                    if new_dir != old_dir:
                        n_flipped += 1
                dir_results.append({
                    'bias_thresh': bt, 'dmi_thresh': dt,
                    'est_pnl': round(est_pnl, 1), 'n_flipped': n_flipped,
                })

            dir_results.sort(key=lambda r: r['est_pnl'], reverse=True)
            print(f"\n  {'bias_thresh':>11}  {'dmi_thresh':>10}  {'n_flipped':>9}  {'est_pnl':>10}")
            print(f"  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*10}")
            for r in dir_results[:12]:
                flag = '  <- BEST' if r == dir_results[0] else ''
                print(f"  {r['bias_thresh']:>11.2f}  {r['dmi_thresh']:>10.1f}  "
                      f"{r['n_flipped']:>9,}  ${r['est_pnl']:>9,.0f}{flag}")

            best_dir = dir_results[0]
            print(f"\n  NOTE: est_pnl uses avg-win/avg-loss flip approximation.")
            print(f"  Run --forward-pass with best threshold to get exact PnL.")

            if best_overall:
                print(f"\nFULL RECOMMENDATION (best tier filter + best direction gate):")
                print(f"  python training/orchestrator.py --forward-pass "
                      f"--min-tier {best_overall['min_tier']} "
                      f"--bias-threshold {best_dir['bias_thresh']:.2f} "
                      f"--dmi-threshold {best_dir['dmi_thresh']:.1f}")
            print()
        elif live_hits:
            best = live_hits[0]
            print(f"\nRECOMMENDED NEXT RUN:")
            print(f"  python training/orchestrator.py --forward-pass "
                  f"--min-tier {best['min_tier']}" +
                  (f" --direction {best['direction']}" if best['direction'] != 'all' else ''))
            print(f"  Expected in-sample PnL: ${best['net_pnl']:,.0f} "
                  f"({best['trades']:,} trades, {best['win_rate']:.1f}% win rate)")
        print()

    def print_bottom_line(self):
        """
        Consolidated bottom line printed after ALL phases complete.
        Mirrors the Oracle attribution structure the user finds most readable.
        """
        W = 80
        fp  = self._fp_summary
        ts  = self._tier_summary
        if not fp and not ts:
            return

        lines = []
        lines.append("")
        lines.append("=" * W)
        lines.append("BOTTOM LINE")
        lines.append("=" * W)

        # ── Library ──────────────────────────────────────────────────────────
        if ts:
            tc = ts['tier_counts']
            lines.append(f"\n  LIBRARY   {ts['total']} templates total")
            lines.append(
                f"    Tier 1 (PRODUCTION): {tc.get(1,0):>4}   "
                f"Tier 2 (PROMISING): {tc.get(2,0):>4}   "
                f"Tier 3 (UNPROVEN): {tc.get(3,0):>4}   "
                f"Tier 4 (TOXIC): {tc.get(4,0):>4}"
            )

        # ── Forward Pass ─────────────────────────────────────────────────────
        if fp:
            lines.append(
                f"\n  FORWARD PASS   {fp.get('date_start','?')} to {fp.get('date_end','?')}"
                f"  ({fp.get('n_days',0)} files)"
            )
            lines.append(
                f"    Trades: {fp['total_trades']:>6,}  |  "
                f"Win rate: {fp['win_rate']*100:5.1f}%  |  "
                f"Total PnL: ${fp['total_pnl']:>10,.2f}"
            )
            if fp['total_trades']:
                lines.append(
                    f"    Correct direction: {fp['pct_correct']:4.1f}%  |  "
                    f"Wrong: {fp['pct_wrong']:4.1f}%  |  "
                    f"Noise: {fp['pct_noise']:4.1f}%  |  "
                    f"Skipped: {fp['pct_skipped']:4.1f}%"
                )

        # ── Opportunity gap (exact 3-term decomposition) ─────────────────────
        if fp and fp.get('ideal_profit', 0):
            ideal = fp['ideal_profit']
            _gap = ideal - fp['total_pnl']
            captured_pct = fp['total_pnl'] / ideal * 100 if ideal else 0
            lines.append(f"\n  OPPORTUNITY GAP   (3 terms sum exactly to gap)")
            lines.append(f"    Ideal:          ${ideal:>12,.2f}")
            lines.append(f"    Actual:         ${fp['total_pnl']:>12,.2f}   ({captured_pct:.1f}% captured)")
            lines.append(f"    Gap:            ${_gap:>12,.2f}")
            _gpct = lambda v: f"({v/_gap*100:.1f}%)" if _gap else ""
            lines.append(f"    #1  missed signals:          ${fp['missed']:>12,.2f}   {_gpct(fp['missed'])}")
            lines.append(f"    #2  trade underperformance:  ${fp['gp_shortfall']:>12,.2f}   {_gpct(fp['gp_shortfall'])}")
            lines.append(f"    #3  non-gp trade drag:       ${fp['non_gp_drag']:>12,.2f}   {_gpct(fp['non_gp_drag'])}")

        # ── OOS Validation ──────────────────────────────────────────────────────
        oos = self._oos_summary
        if oos and oos.get('total_trades', 0) > 0:
            lines.append(
                f"\n  OOS VALIDATION   {oos.get('date_start','?')} to {oos.get('date_end','?')}"
                f"  ({oos.get('n_days',0)} files)"
            )
            lines.append(
                f"    Trades: {oos['total_trades']:>6,}  |  "
                f"Win rate: {oos['win_rate']*100:5.1f}%  |  "
                f"Total PnL: ${oos['total_pnl']:>10,.2f}"
            )
            if fp and fp.get('total_trades', 0) > 0:
                wr_d = oos['win_rate'] - fp['win_rate']
                pnl_per_is = fp['total_pnl'] / fp['total_trades'] if fp['total_trades'] else 0
                pnl_per_oos = oos['total_pnl'] / oos['total_trades'] if oos['total_trades'] else 0
                lines.append(
                    f"    IS->OOS delta:  WR {wr_d*100:+.1f}pp  |  "
                    f"Avg PnL/trade: ${pnl_per_is:.1f} -> ${pnl_per_oos:.1f}"
                )

        # ── Top Tier 1 ────────────────────────────────────────────────────────
        if ts and ts.get('top_t1'):
            lines.append(f"\n  TOP TIER 1   (combined PnL: ${ts['tier1_pnl']:,.2f})")
            lines.append(f"    {'ID':<10} {'Sharpe':>7} {'Win%':>6} {'PnL':>10} {'Trades':>7}")
            for tid, sharpe, wr, pnl, trades in ts['top_t1']:
                lines.append(f"    {str(tid):<10} {sharpe:>7.2f} {wr*100:>5.1f}% ${pnl:>9,.2f} {trades:>7,}")

        lines.append("")
        lines.append("=" * W)

        for line in lines:
            print(line)


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
                    print("Dependencies OK | WARNING: CUDA not available -- GPU acceleration disabled")
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
    parser.add_argument('--no-dashboard', action='store_true', help="Disable all UI (popup and dashboard)")
    parser.add_argument('--dashboard', action='store_true', help="Show full live dashboard instead of default lightweight popup")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")
    parser.add_argument('--exploration-mode', action='store_true', help="Enable unconstrained exploration mode")
    parser.add_argument('--fresh', action='store_true', help="Clear all pipeline checkpoints and start fresh")
    parser.add_argument('--forward-pass', action='store_true', help="Run Phase 4 forward pass using existing playbook")
    parser.add_argument('--train-only', action='store_true',
                        help="Run Phases 2-3 only (discovery + template optimization). "
                             "Saves library to checkpoints, skips forward pass and strategy report.")
    parser.add_argument('--oos', action='store_true',
                        help="Standalone OOS rerun (Phase 6 only): frozen templates, reports go to reports/oos/, "
                             "training depth_weights.json preserved. Uses DATA/ATLAS_OOS by default.")
    parser.add_argument('--skip-oos', action='store_true',
                        help="Skip the auto-chained OOS forward pass (Phase 6) in full pipeline or --forward-pass")
    parser.add_argument('--account-size', type=float, default=0.0, metavar='USD',
                        help="Starting account equity in USD. When set, gates trades that risk >50%% of "
                             "remaining equity (SL in dollars vs equity). Simulation ends if equity "
                             "drops below NinjaTrader MNQ intraday margin ($50). "
                             "Use 100.0 for a $100 funded account test.")
    parser.add_argument('--train-end', type=str, default=None, metavar='YYYYMMDD',
                        help="Out-of-sample guard: cap training data at this date (e.g. 20251231). "
                             "Use with --forward-start for a clean train/test split.")
    parser.add_argument('--forward-data', type=str, default=None, metavar='PATH',
                        help="Separate ATLAS root for Phase 4 forward pass (e.g. DATA/ATLAS_1DAY). "
                             "Training (Phases 2-3) still uses --data; only Phase 4 switches to this path.")
    parser.add_argument('--forward-start', type=str, default=None, metavar='YYYYMMDD',
                        help="First day to include in forward pass (inclusive, e.g. 20260101)")
    parser.add_argument('--forward-end', type=str, default=None, metavar='YYYYMMDD',
                        help="Last day to include in forward pass (inclusive, e.g. 20260209)")
    parser.add_argument('--telemetry', action='store_true',
                        help="Print per-trade decision telemetry (direction reasoning, oracle truth, gates)")
    parser.add_argument('--min-tier', type=int, default=None, choices=[1, 2, 3, 4],
                        help="Only activate templates of this tier or better (1=Tier1 only, 3=drop Tier4 losers)")
    parser.add_argument('--bias-threshold', type=float, default=None,
                        help="Oracle bias threshold for direction lock (default 0.55). Lower = more oracle-locked trades.")
    parser.add_argument('--dmi-threshold', type=float, default=None,
                        help="Min |dmi_diff| required to use DMI signal (default 0.0 = any non-zero DMI counts).")
    parser.add_argument('--r2-target', type=float, default=0.90,
                        help="Adj-R2 target for DOE convergence (default: 0.90)")
    parser.add_argument('--sweep-params', action='store_true',
                        help="Post-hoc DOE: sweep filter combinations on oracle_trade_log.csv and rank by net PnL")
    parser.add_argument('--strategy-report', action='store_true', help="Run Phase 5 strategy selection report")

    # Monte Carlo Flags (opt-in with --mc)
    parser.add_argument('--mc', action='store_true', help='Enable Monte Carlo sweep after Bayesian Phase 3')
    parser.add_argument('--mc-iters', type=int, default=2000, help='Monte Carlo iterations per (template, timeframe) combo')
    parser.add_argument('--mc-only', action='store_true', help='Skip discovery, just run Monte Carlo from existing templates')
    parser.add_argument('--anova-only', action='store_true', help='Skip MC sweep, just run ANOVA on existing results')
    parser.add_argument('--refine-only', action='store_true', help='Skip MC+ANOVA, just run Thompson refinement')

    args = parser.parse_args()

    # ── Tee stdout -> checkpoints/training_log.txt (append, one file per project) ──
    import io
    class _Tee(io.TextIOWrapper):
        def __init__(self, log_path):
            self._file = open(log_path, 'a', encoding='utf-8', buffering=1)
            self._stdout = sys.stdout
        def write(self, data):
            try:
                self._stdout.write(data)
            except UnicodeEncodeError:
                self._stdout.write(data.encode('ascii', errors='replace').decode('ascii'))
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
    _train_rpt_dir = _get_reports_dir('training')
    log_path = os.path.join(_train_rpt_dir, 'training_log.txt')
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

        if args.sweep_params:
            orchestrator.run_param_sweep()
            return 0

        if args.oos and not args.fresh:
            # Standalone OOS rerun (Phase 6 only)
            _oos_data = args.forward_data or os.path.join('DATA', 'ATLAS_OOS')
            orchestrator.run_forward_pass(_oos_data,
                                          start_date=args.forward_start,
                                          end_date=args.forward_end,
                                          min_tier=args.min_tier,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=True,
                                          account_size=args.account_size,
                                          telemetry=args.telemetry)
        elif args.forward_pass and not args.fresh:
            # Phase 4 IS → Phase 5 Strategy → Phase 6 OOS
            _fwd_data = args.forward_data or args.data
            orchestrator.run_forward_pass(_fwd_data,
                                          start_date=args.forward_start,
                                          end_date=args.forward_end,
                                          min_tier=args.min_tier,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=False,
                                          account_size=args.account_size,
                                          telemetry=args.telemetry)
            orchestrator.run_strategy_selection()

            # Auto-chain OOS if data exists and not testing custom data
            _oos_path = os.path.join('DATA', 'ATLAS_OOS')
            if (not args.skip_oos
                    and not args.forward_data
                    and os.path.isdir(_oos_path)):
                print(f"\n{'='*80}")
                print(f"  AUTO-CHAINING: Phase 6 — OOS Blind Validation")
                print(f"{'='*80}")
                orchestrator.run_forward_pass(_oos_path,
                                              start_date=args.forward_start,
                                              end_date=args.forward_end,
                                              min_tier=args.min_tier,
                                              bias_threshold=args.bias_threshold,
                                              dmi_threshold=args.dmi_threshold,
                                              oos_mode=True,
                                              account_size=args.account_size,
                                              telemetry=args.telemetry)
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

            if args.train_only:
                # --train-only: stop after Phases 2-3
                n_lib = len(orchestrator.pattern_library) if orchestrator.pattern_library else 0
                print(f"\n{'='*80}")
                print(f"TRAIN-ONLY COMPLETE: {n_lib} templates in library")
                print(f"  Run forward pass:  python training/orchestrator.py --forward-pass")
                print(f"  Quick 1-day test:  python training/orchestrator.py --forward-pass --data DATA/ATLAS_1DAY")
                print(f"{'='*80}")
            elif args.mc or args.mc_only:
                # Optional: Monte Carlo Sweep -> ANOVA -> Thompson -> Validation
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
                # Default: Bayesian path → Phase 4 IS → Phase 5 Strategy → Phase 6 OOS
                _fwd_data = args.forward_data or args.data
                orchestrator.run_forward_pass(_fwd_data,
                                          start_date=args.forward_start,
                                          end_date=args.forward_end,
                                          min_tier=args.min_tier,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=False,
                                          account_size=getattr(args, 'account_size', 0.0),
                                          telemetry=getattr(args, 'telemetry', False))
                orchestrator.run_strategy_selection()

                # Phase 6: OOS blind validation (auto-chain if data exists)
                _oos_path = os.path.join('DATA', 'ATLAS_OOS')
                if (not getattr(args, 'skip_oos', False)
                        and not args.forward_data
                        and os.path.isdir(_oos_path)):
                    print(f"\n{'='*80}")
                    print(f"  AUTO-CHAINING: Phase 6 — OOS Blind Validation")
                    print(f"{'='*80}")
                    orchestrator.run_forward_pass(_oos_path,
                                              start_date=args.forward_start,
                                              end_date=args.forward_end,
                                              min_tier=args.min_tier,
                                              bias_threshold=args.bias_threshold,
                                              dmi_threshold=args.dmi_threshold,
                                              oos_mode=True,
                                              account_size=getattr(args, 'account_size', 0.0),
                                              telemetry=getattr(args, 'telemetry', False))

        orchestrator.print_bottom_line()
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
