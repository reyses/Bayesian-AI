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
from training.timeframe_belief_network import TimeframeBeliefNetwork, BeliefState

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
from training.pid_oscillation_analyzer import PIDOscillationAnalyzer, PIDSignal

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

        # Bottom-line accumulators -- populated by run_forward_pass / run_strategy_selection
        self._fp_summary   = {}   # Phase 4 key metrics
        self._tier_summary = {}   # Phase 5 tier counts + top templates

        # PID Analyzer (Shadow Mode)
        self.pid_analyzer = PIDOscillationAnalyzer()

        # Slippage parameters
        self.BASE_SLIPPAGE = DEFAULT_BASE_SLIPPAGE
        self.VELOCITY_SLIPPAGE_FACTOR = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

    def calculate_optimal_workers(self):
        try:
            return max(1, multiprocessing.cpu_count() - 2)
        except NotImplementedError:
            return 1

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

    def run_forward_pass(self, data_source: str,
                         start_date: str = None, end_date: str = None,
                         min_tier: int = None,
                         bias_threshold: float = None,
                         dmi_threshold: float = None,
                         oos_mode: bool = False,
                         account_size: float = 0.0):
        """
        Phase 4: Forward pass -- replay full year using playbook.
        Scans fractal cascade per day, matches templates, trades via WaveRider.
        Brain learns from outcomes.

        Args:
            start_date: Inclusive lower bound YYYYMMDD (e.g. '20260101').
                        If None, no lower bound -- all days included.
            end_date:   Inclusive upper bound YYYYMMDD (e.g. '20260209').
                        If None, no upper bound -- all days included.
            oos_mode:     When True, writes separate oos_trade_log.csv / oos_report.txt
                          and preserves training depth_weights.json unchanged.
                          Use with --forward-start to run blind out-of-sample simulation.
            account_size: Starting account equity in USD (0 = disabled, no equity gate).
                          When > 0, simulates a funded account: gates new entries if
                          running equity drops below NinjaTrader MNQ intraday margin
                          ($50/contract). Report shows equity curve + max drawdown.
        """
        print("\n" + "="*80)
        if oos_mode:
            print("OOS BLIND SIMULATION (templates/scaler frozen from training)")
        else:
            print("PHASE 4: FORWARD PASS (EXECUTION MODE)")
        if start_date or end_date:
            _lo = start_date or "start"
            _hi = end_date   or "end"
            print(f"  Date slice: {_lo} -> {_hi}")
        print("="*80)
        if self.dashboard_queue:
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                      'step': 'FORWARD_PASS', 'pct': 0})

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

        # Apply min_tier filter -- removes losing tiers from the centroid index entirely
        # (oracle_trade_log shows Tier 4 = -$52K drag; min_tier=3 -> +$96K vs $44K baseline)
        if min_tier is not None and template_tier_map:
            _before = len(valid_template_ids)
            valid_template_ids = [tid for tid in valid_template_ids
                                  if template_tier_map.get(tid, 4) <= min_tier]
            print(f"  Min-tier filter (tier <= {min_tier}): {_before} -> {len(valid_template_ids)} active templates")

        centroids = np.array([self.pattern_library[tid]['centroid'] for tid in valid_template_ids])
        centroids_scaled = self.scaler.transform(centroids)
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

        # Fractal belief network: 8 TF workers (1h -> 15s), path conviction
        # Each worker: Task1 = aggregate TF bars (once/day), Task2 = cluster match + regression
        # 1h worker fires ~7x/day (light); 15s fires every bar (heavy: top-3 matching)
        belief_network = TimeframeBeliefNetwork(
            pattern_library  = self.pattern_library,
            scaler           = self.scaler,
            engine           = self.engine,
            valid_tids       = valid_template_ids,
            centroids_scaled = centroids_scaled,
        )
        print(f"  Belief network: {len(TimeframeBeliefNetwork.TIMEFRAMES_SECONDS)} TF workers "
              f"(conviction threshold: {TimeframeBeliefNetwork.MIN_CONVICTION:.2f})")

        # 2. Iterate Days
        # Use 15s daily files for position management (daily granularity).
        # Pattern entry timestamps are snapped to 1m for signal quality.
        daily_files_15s = sorted(glob.glob(os.path.join(data_source, '15s', '*.parquet')))
        if not daily_files_15s:
            print(f"  No 15s data found in {data_source}/15s/")
            return

        # ── Time-slice filter ────────────────────────────────────────────────
        # Filenames are YYYYMMDD.parquet; lexicographic comparison works fine.
        if start_date or end_date:
            _before = len(daily_files_15s)
            daily_files_15s = [
                f for f in daily_files_15s
                if (not start_date or os.path.basename(f).replace('.parquet', '') >= start_date)
                and (not end_date   or os.path.basename(f).replace('.parquet', '') <= end_date)
            ]
            print(f"  Date filter applied: {_before} -> {len(daily_files_15s)} days "
                  f"({start_date or 'start'} -> {end_date or 'end'})")

        print(f"  Found {len(daily_files_15s)} days to simulate.")

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0

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
            Chain is ordered leaf-first (index 0) → root (last), so we scan all entries
            and keep the one with the highest |oracle_marker|.
            """
            leaf_om = getattr(p, 'oracle_marker', 0)
            best = leaf_om
            for ce in (getattr(p, 'parent_chain', None) or []):
                macro_om = ce.get('oracle_marker', 0)
                if abs(macro_om) > abs(best):
                    best = macro_om
            return best

        # Per-trade oracle tracking
        _ORACLE_LABEL_NAMES = {2: 'MEGA_LONG', 1: 'SCALP_LONG', 0: 'NOISE', -1: 'SCALP_SHORT', -2: 'MEGA_SHORT'}
        oracle_trade_records = []  # completed per-trade oracle dicts
        pending_oracle = None      # oracle facts for currently open trade
        fn_potential_pnl = 0.0    # dollar potential of real moves we skipped

        # PID Shadow Log
        pid_oracle_records = []

        # Skip reason counters (per-candidate across all days)
        skip_headroom    = 0   # Gate 0: no pattern or noise zone / structural rules
        skip_dist        = 0   # No cluster match within distance 3.0
        skip_brain       = 0   # Brain gate: template probability too low
        skip_conviction  = 0   # Belief network: path conviction below MIN_CONVICTION
        n_signals_seen   = 0   # Total candidate signals evaluated (all gates combined)
        depth_traded     = defaultdict(int)  # depth -> trade count (1=high TF, 6=15s)

        # FN oracle records: per-signal log of missed real moves with worker snapshots.
        # Answers: "when we missed a profitable move, what did the workers think?"
        # If workers agreed with oracle direction on FN signals, a gate is too strict.
        fn_oracle_records = []

        n_days = len(daily_files_15s)
        for day_idx, day_file in enumerate(daily_files_15s):
            day_date = os.path.basename(day_file).replace('.parquet', '')
            print(f"\n  Day {day_idx+1}/{n_days}: {day_date} ... ", end='', flush=True)
            if self.dashboard_queue and day_idx % 5 == 0:
                pct = (day_idx / n_days) * 100
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                          'step': f'FORWARD_PASS  day {day_idx+1}/{n_days}',
                                          'pct': round(pct, 1)})

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

            # Belief network: Task 1 for all 8 TF workers
            # Aggregates 15s -> 30s/1m/3m/5m/15m/30m/1h bars + computes quantum states
            # 1h worker: ~7 states; 15s worker: ~5300 states (pre-computed, fast lookup)
            try:
                _states_15s = self.engine.batch_compute_states(df_15s, use_cuda=True)
                belief_network.prepare_day(df_15s, states_15s=_states_15s)
            except Exception as _bn_err:
                _states_15s = []
                belief_network.prepare_day(df_15s, states_15s=[])

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

            _bar_i = 0  # 15s bar index for belief network worker ticks

            for row in df_15s.itertuples():
                ts_raw = row.timestamp
                # Snap to 60s boundary to match pattern_map keys
                ts = int(ts_raw) // 60 * 60
                price = getattr(row, 'close', getattr(row, 'price', 0.0))

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
                            })
                            pending_oracle = None

                # 2. Check for entries (if no position)
                # Equity ruin check: simulation ends when equity hits 0 (no money to trade).
                if _equity_enabled and account_ruined:
                    break   # stop processing this day's bars entirely
                if not current_position_open and ts in pattern_map:
                    candidates = pattern_map[ts]
                    best_candidate = None
                    best_dist = 999.0
                    best_tid = None
                    _candidate_gate = {}  # id(p) -> gate that blocked it (for FN audit)

                    for p in candidates:
                        n_signals_seen += 1
                        # --- Gate 0: Headroom Gate (Nightmare Field Equation) ---
                        micro_z = abs(p.z_score)
                        micro_pattern = p.pattern_type

                        # Macro context
                        chain = getattr(p, 'parent_chain', [])
                        root_entry = chain[-1] if chain else None
                        macro_z = abs(root_entry['z']) if root_entry else 0.0

                        should_skip = False
                        _skip_label = 'gate0'

                        # DATA-QUALITY OVERRIDE: if this pattern maps to a high-quality
                        # template (>=10 members, WR>=55%, sigma<=10 ticks) it is admitted
                        # regardless of which structural rule would have blocked it.
                        # "Predictable structure + low residuals" beats any physics heuristic.
                        _data_override = False
                        if _exception_tids and micro_pattern:
                            _e_feat    = np.array([FractalClusteringEngine.extract_features(p)])
                            _e_scaled  = self.scaler.transform(_e_feat)
                            _e_dists   = np.linalg.norm(centroids_scaled - _e_scaled, axis=1)
                            _e_nearest = int(np.argmin(_e_dists))
                            if (_e_dists[_e_nearest] < 4.5
                                    and valid_template_ids[_e_nearest] in _exception_tids):
                                _data_override = True

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
                                headroom = macro_z < 2.0
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
                            continue

                        # Gate 0.5: depth filter -- exclude depths that had negative avg
                        # PnL/trade in the previous run (written to depth_weights.json).
                        # First run has no weights so nothing is filtered.
                        _cand_depth = getattr(p, 'depth', 6)
                        if _cand_depth in _DEPTH_FILTER_OUT:
                            skip_headroom += 1
                            _candidate_gate[id(p)] = 'gate0_5'
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

                        if dist < 4.5: # Threshold (raised from 3.0 so extreme-z profitable patterns reach their nearest cluster)
                            # Brain gate -- use low threshold for exploration
                            if self.brain.should_fire(tid, min_prob=0.05, min_conf=0.0):
                                # Score: lower is better
                                # depth: 1=4h (best) -> 6=15s (worst)
                                # Tier bonus: Tier 1 = -1.5, Tier 2 = -0.5, Tier 3 = 0, Tier 4 = +0.5
                                # Depth adj: data-driven from previous run avg PnL/trade
                                #   positive avg_pnl -> negative score_adj (favored)
                                #   negative avg_pnl -> positive score_adj (penalized)
                                p_depth   = getattr(p, 'depth', 6)
                                tier_adj  = _TIER_SCORE_ADJ.get(template_tier_map.get(tid, 3), 0.0)
                                depth_adj = _DEPTH_SCORE_ADJ.get(p_depth, 0.0)
                                score     = p_depth + dist + tier_adj + depth_adj
                                if score < best_dist:
                                    best_dist = score
                                    best_candidate = p
                                    best_tid = tid
                            else:
                                skip_brain += 1
                                _candidate_gate[id(p)] = 'gate2'
                        else:
                            skip_dist += 1
                            _candidate_gate[id(p)] = 'gate1'

                    if best_candidate:
                        # FIRE
                        params = self.pattern_library[best_tid]['params']
                        lib_entry = self.pattern_library[best_tid]

                        # ── Live feature vector (shared by direction + MFE models) ──
                        # Same 14D features used during clustering -- scaler already fitted
                        _live_feat = np.array(FractalClusteringEngine.extract_features(best_candidate))
                        _live_scaled = self.scaler.transform([_live_feat])[0]

                        # ── Direction gate ──────────────────────────────────────────
                        # Priority 0: individual oracle_marker (highest resolution signal)
                        #   oracle_marker > 0  -> LONG move followed this exact training pattern
                        #   oracle_marker < 0  -> SHORT move
                        #   oracle_marker == 0 -> NOISE -> fall through to regression tiers
                        _BIAS_THRESH = bias_threshold if bias_threshold is not None else 0.55
                        _DMI_THRESH  = dmi_threshold  if dmi_threshold  is not None else 0.0
                        long_bias  = lib_entry.get('long_bias',  0.0)
                        short_bias = lib_entry.get('short_bias', 0.0)
                        _nn_marker = _effective_oracle(best_candidate)  # macro-to-leaf aggregated

                        if _nn_marker > 0:
                            side = 'long'
                        elif _nn_marker < 0:
                            side = 'short'
                        else:
                            # NOISE pattern -- use regression model hierarchy
                            side = None

                            # Priority 1: per-cluster logistic regression P(LONG)
                            _dir_coeff = lib_entry.get('dir_coeff')
                            if _dir_coeff is not None:
                                _dir_logit = (np.dot(_live_scaled, np.array(_dir_coeff))
                                              + lib_entry.get('dir_intercept', 0.0))
                                _dir_prob  = 1.0 / (1.0 + np.exp(-_dir_logit))
                                if _dir_prob > _BIAS_THRESH:
                                    side = 'long'
                                elif _dir_prob < (1.0 - _BIAS_THRESH):
                                    side = 'short'

                            # Priority 2: template aggregate bias
                            if side is None:
                                if long_bias >= _BIAS_THRESH:
                                    side = 'long'
                                elif short_bias >= _BIAS_THRESH:
                                    side = 'short'
                                elif long_bias + short_bias >= 0.10:
                                    side = 'long' if long_bias >= short_bias else 'short'

                            # Priority 3: live DMI (trend-following)
                            if side is None:
                                _live_s   = best_candidate.state
                                _dmi_diff = (getattr(_live_s, 'dmi_plus',  0.0)
                                           - getattr(_live_s, 'dmi_minus', 0.0))
                                if abs(_dmi_diff) >= _DMI_THRESH and _dmi_diff > 0:
                                    side = 'long'
                                elif abs(_dmi_diff) >= _DMI_THRESH and _dmi_diff < 0:
                                    side = 'short'
                                else:
                                    _vel = getattr(_live_s, 'particle_velocity', 0.0)
                                    side = 'long' if _vel >= 0 else 'short'

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
                                _candidate_gate[id(p)] = 'gate3'
                                continue
                            # Path direction override for NOISE oracle_marker patterns
                            if _nn_marker == 0 and _belief.direction != side:
                                side = _belief.direction  # fractal tree overrides leaf heuristics
                            # Use network's predicted MFE if better than leaf-level estimate
                            _network_tp = max(4, int(round(_belief.predicted_mfe))) if _belief.predicted_mfe > 2.0 else None
                        else:
                            _network_tp = None

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

                        # Phase 2: trailing stop distance (tight, from HWM)
                        if _reg_sigma > 2.0:
                            _trail_ticks = max(2, int(round(_reg_sigma * 1.1)))
                        elif _mean_mae > 2.0:
                            _trail_ticks = max(2, int(round(_mean_mae * 1.1)))
                        else:
                            _trail_ticks = params.get('trailing_stop_ticks', 10)

                        # Trail activation: needs p25_mae * 0.3 profit ticks to engage
                        # (30% of the tight adverse excursion -- locks in gain quickly
                        #  once the trade is moving our way)
                        _trail_act_ticks = (max(2, int(round(_p25_mae * 0.3)))
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

                        self.wave_rider.open_position(
                            entry_price=price,
                            side=side,
                            state=best_candidate.state,
                            stop_distance_ticks=_sl_ticks,
                            profit_target_ticks=_tp_ticks,
                            trailing_stop_ticks=_trail_ticks,
                            trail_activation_ticks=_trail_act_ticks,
                            template_id=best_tid
                        )
                        current_position_open = True
                        active_entry_price = price
                        active_entry_time = ts
                        active_side = side
                        active_template_id = best_tid
                        depth_traded[getattr(best_candidate, 'depth', 6)] += 1

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
                            'oracle_mfe':       best_candidate.oracle_meta.get('mfe', 0.0),
                            'oracle_mae':       best_candidate.oracle_meta.get('mae', 0.0),
                            # Direction DOE diagnostics
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
                                # Skip if PID regime covers this bar (handled by PID analyzer)
                                # Thresholds: |term_pid| >= 0.3, osc_coh >= 0.5, adx <= 30.0
                                _s = p.state
                                _is_pid = (abs(_s.term_pid) >= 0.3
                                           and _s.oscillation_coherence >= 0.5
                                           and _s.adx_strength <= 30.0)
                                if _is_pid:
                                     # Not counted as fractal FN failure because PID analyzer handles it
                                     continue

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
                                    'reason':          'competed',  # had a trade this bar, but not this candidate
                                    'gate_blocked':    _candidate_gate.get(id(p), 'passed'),  # 'passed' = scored but lost to better candidate
                                    'workers':         __import__('json').dumps(belief_network.get_worker_snapshot()),
                                })
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
                                    'reason':          'no_match',  # nothing passed gates at this bar
                                    'gate_blocked':    _candidate_gate.get(id(p), 'unknown'),
                                    'workers':         __import__('json').dumps(belief_network.get_worker_snapshot()),
                                })

            # End of day cleanup -- force close any open position
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
                    oracle_trade_records.append({
                        **pending_oracle,
                        'exit_price':  outcome.exit_price,
                        'exit_time':   _eod_exit_t,
                        'hold_bars':   max(1, int((_eod_exit_t - _eod_entry_t) / 15)),
                        'exit_reason': 'TIME_EXIT',
                        'actual_pnl':  outcome.pnl,
                        'oracle_potential_pnl': oracle_potential,
                        'capture_rate': round(min(capture, 9.99), 4),
                        'result': outcome.result,
                        'exit_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                    })
                    pending_oracle = None

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
                print("No trades.")

        # Final Report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FORWARD PASS COMPLETE")
        _date_range = (
            f"  Period: {start_date or daily_files_15s[0] if daily_files_15s else 'N/A'} "
            f"to {end_date or (os.path.basename(daily_files_15s[-1]).replace('.parquet','') if daily_files_15s else 'N/A')} "
            f"({len(daily_files_15s)} days)"
        ) if daily_files_15s else ""
        if _date_range:
            report_lines.append(_date_range)
        report_lines.append(f"Total Trades: {total_trades}")
        report_lines.append(f"Win Rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "Win Rate: N/A")
        report_lines.append(f"Total PnL: ${total_pnl:.2f}")

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
        ideal_profit  = tp_potential + fn_potential_pnl           # perfect execution on everything

        report_lines.append("")
        report_lines.append(f"  TOTAL SIGNALS SEEN BY ORACLE: {total_real_opps + total_noise_opps:,}")
        report_lines.append(f"    Real moves (MEGA/SCALP):  {total_real_opps:>6,}   -- worth ${ideal_profit:>10,.2f} if perfectly traded")
        report_lines.append(f"    Noise (no real move):     {total_noise_opps:>6,}")

        # ── 2. What we did ───────────────────────────────────────────────────────
        n_traded   = len(oracle_trade_records)
        n_skipped  = audit_fn + audit_tn
        report_lines.append("")
        report_lines.append(f"  WHAT WE DID:")
        report_lines.append(f"    Traded:  {n_traded:>6,}  ({n_traded/(total_real_opps+total_noise_opps)*100:.1f}% of all signals)")
        report_lines.append(f"    Skipped: {n_skipped:>6,}  ({n_skipped/(total_real_opps+total_noise_opps)*100:.1f}% of all signals)")

        # ── 2b. Skip reason breakdown ─────────────────────────────────────────────
        # n_signals_seen counts individual candidates (not unique timestamps).
        # skip_xxx are per-candidate gate rejections.
        _n_pass = n_signals_seen - skip_headroom - skip_dist - skip_brain - skip_conviction
        report_lines.append("")
        report_lines.append(f"  WHY SIGNALS WERE SKIPPED  (total candidates evaluated: {n_signals_seen:,})")
        if n_signals_seen > 0:
            _pct_s = lambda n: f"{n/n_signals_seen*100:.1f}%"
            report_lines.append(f"    Gate 0 (headroom/pattern rule): {skip_headroom:>6,}  ({_pct_s(skip_headroom)})")
            report_lines.append(f"    Gate 1 (dist > 3.0, no match):  {skip_dist:>6,}  ({_pct_s(skip_dist)})")
            report_lines.append(f"    Gate 2 (brain rejected):        {skip_brain:>6,}  ({_pct_s(skip_brain)})")
            report_lines.append(f"    Gate 3 (conviction < thresh):   {skip_conviction:>6,}  ({_pct_s(skip_conviction)})")
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

        # ── 2f. Worker agreement analysis ────────────────────────────────────────
        # For each TF worker: what fraction of the time did it agree with the
        # trade direction at entry? Compare wins vs losses to find who is predictive.
        # Also: which workers FLIPPED direction by exit? That likely caused the loss.
        if oracle_trade_records and 'entry_workers' in oracle_trade_records[0]:
            import json as _js
            _TF_ORDER = ['1h','30m','15m','5m','3m','1m','30s','15s']
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
            partial   = [r for r in tp_recs if 0.20 <= r['capture_rate'] < 0.80]
            too_early = [r for r in tp_recs if 0 < r['capture_rate'] < 0.20]
            reversed_ = [r for r in tp_recs if r['capture_rate'] <= 0]

            # left_on_table: only non-reversed underperformance (how much potential
            # was uncaptured on trades that at least moved in our direction)
            non_reversed = [r for r in tp_recs if r['capture_rate'] > 0]
            left_on_table = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl'])
                                for r in non_reversed)

            report_lines.append("")
            report_lines.append(f"  EXIT QUALITY (correct-direction trades):")
            report_lines.append(f"    Optimal  (>=80% of move captured): {len(optimal):>6,}  ->  ${sum(r['actual_pnl'] for r in optimal):>10,.2f}")
            report_lines.append(f"    Partial  (20-80% captured):        {len(partial):>6,}  ->  ${sum(r['actual_pnl'] for r in partial):>10,.2f}")
            report_lines.append(f"    Too early (<20% captured):         {len(too_early):>6,}  ->  ${sum(r['actual_pnl'] for r in too_early):>10,.2f}")
            report_lines.append(f"    Reversed (correct dir, mkt flipped):{len(reversed_):>6,}  ->  ${sum(r['actual_pnl'] for r in reversed_):>10,.2f}  <- own leakage bucket")
            report_lines.append(f"    Left on table (non-reversed TP gap):             ${left_on_table:>10,.2f}")
        else:
            reversed_ = []
            left_on_table = 0.0

        # ── 5. Profit gap summary ────────────────────────────────────────────────
        # "Reversed" losses = actual dollars lost on correct-direction trades that
        # still went against us.  Distinct from:
        #   wrong_dir  = oracle label was opposite to our direction (fp_wrong_recs)
        #   noise      = oracle said no real move at all (fp_noise_recs)
        #   left_on_table = potential uncaptured on non-reversed winners
        reversed_loss_val  = abs(sum(r['actual_pnl'] for r in reversed_))
        non_reversed_val   = [r for r in tp_recs if r['capture_rate'] > 0] if tp_recs else []
        left_on_table_val  = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl'])
                                 for r in non_reversed_val)

        report_lines.append("")
        report_lines.append(f"  PROFIT GAP ANALYSIS:")
        report_lines.append(f"    Ideal (all real moves, perfect exits):  ${ideal_profit:>12,.2f}")
        report_lines.append(f"    -----------------------------------------------------")
        report_lines.append(f"    Lost -- missed opportunities (skipped):  ${fn_potential_pnl:>12,.2f}  ({fn_potential_pnl/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- wrong direction at entry:        ${abs(fp_wrong_pnl):>12,.2f}  ({abs(fp_wrong_pnl)/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- noise trades:                    ${abs(fp_noise_pnl):>12,.2f}  ({abs(fp_noise_pnl)/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- reversed after correct entry:    ${reversed_loss_val:>12,.2f}  ({reversed_loss_val/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- TP underperform (non-reversed):  ${left_on_table_val:>12,.2f}  ({left_on_table_val/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    -----------------------------------------------------")
        report_lines.append(f"    Actual profit:                          ${total_pnl:>12,.2f}  ({total_pnl/ideal_profit*100:.1f}% of ideal)" if ideal_profit else f"    Actual profit: ${total_pnl:.2f}")

        # Store for bottom-line summary at program exit
        self._fp_summary = {
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
            'left_on_table':   left_on_table_val,
            'missed':          fn_potential_pnl,
            'wrong_dir_loss':  abs(fp_wrong_pnl),
        }

        # Send to dashboard
        if self.dashboard_queue:
            self.dashboard_queue.put({
                'type': 'ORACLE_ATTRIBUTION',
                'ideal':     ideal_profit,
                'actual':    total_pnl,
                'missed':    fn_potential_pnl,
                'wrong_dir': abs(fp_wrong_pnl),
                'too_early': left_on_table_val,
                'noise':     abs(fp_noise_pnl),
            })
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                      'step': 'FORWARD_PASS COMPLETE', 'pct': 100})

        # ── 6. Save CSV ──────────────────────────────────────────────────────────
        if oracle_trade_records:
            _log_name = 'oos_trade_log.csv' if oos_mode else 'oracle_trade_log.csv'
            csv_path = os.path.join(self.checkpoint_dir, _log_name)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = _csv.DictWriter(f, fieldnames=list(oracle_trade_records[0].keys()))
                writer.writeheader()
                writer.writerows(oracle_trade_records)
            report_lines.append("")
            report_lines.append(f"  Per-trade oracle log saved: {csv_path}")

        if pid_oracle_records:
            pid_csv_path = os.path.join(self.checkpoint_dir, 'pid_oracle_log.csv')
            with open(pid_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = _csv.DictWriter(f, fieldnames=list(pid_oracle_records[0].keys()))
                writer.writeheader()
                writer.writerows(pid_oracle_records)
            report_lines.append(f"  PID oracle log saved: {pid_csv_path} ({len(pid_oracle_records)} signals)")

        # ── Save FN oracle log + report section ──────────────────────────────────
        # fn_oracle_records: every missed real move with worker snapshot.
        # "competed" = another candidate was traded at the same bar.
        # "no_match" = nothing passed gates at this bar.
        # Key question: when workers agreed with oracle direction on FN signals,
        # a gate was too strict (the workers were right but we still skipped).
        if fn_oracle_records:
            import json as _fnjs
            _fn_name = 'oos_fn_log.csv' if oos_mode else 'fn_oracle_log.csv'
            fn_csv_path = os.path.join(self.checkpoint_dir, _fn_name)
            with open(fn_csv_path, 'w', newline='', encoding='utf-8') as _fnf:
                _fn_writer = _csv.DictWriter(_fnf, fieldnames=list(fn_oracle_records[0].keys()))
                _fn_writer.writeheader()
                _fn_writer.writerows(fn_oracle_records)
            report_lines.append(f"  FN oracle log saved: {fn_csv_path}  ({len(fn_oracle_records):,} missed real moves)")

            # FN worker agreement analysis:
            # For each TF worker, what fraction of FN signals had the worker
            # agreeing with the oracle direction?  High agreement = gate is blocking
            # moves the workers correctly identified.
            _TF_ORDER_FN = ['1h','30m','15m','5m','3m','1m','30s','15s']
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
                        'gate0_5', 'gate1', 'gate2', 'gate3', 'passed', 'unknown']:
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

        # Save report
        _report_name = 'oos_report.txt' if oos_mode else 'phase4_report.txt'
        report_path = os.path.join(self.checkpoint_dir, _report_name)
        with open(report_path, 'w', encoding='utf-8') as f:
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
        oracle_csv = os.path.join(self.checkpoint_dir, 'oracle_trade_log.csv')
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
        report_path = os.path.join(self.checkpoint_dir, 'phase5_report.txt')
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

            _train_end = getattr(self.config, 'train_end', None)
            if _train_end:
                print(f"  Out-of-sample guard: training data capped at {_train_end}")
            manifest = self._run_discovery(
                data_source,
                checkpoint_callback=lambda lvl, tf, patterns, levels:
                    ckpt.save_discovery_level(patterns, levels),
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
            # Per-cluster regression model coefficients (14D scaled feature space)
            # mfe_coeff @ live_scaled_features + mfe_intercept  -> predicted MFE in price pts
            # sigmoid(dir_coeff @ live_scaled_features + dir_intercept) -> P(LONG)
            # None when cluster had < 15 members for stable fitting
            'mfe_coeff':     getattr(template, 'mfe_coeff',     None),
            'mfe_intercept': getattr(template, 'mfe_intercept', 0.0),
            'dir_coeff':     getattr(template, 'dir_coeff',     None),
            'dir_intercept': getattr(template, 'dir_intercept', 0.0),
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

        log_path   = os.path.join(self.checkpoint_dir, 'oracle_trade_log.csv')
        tiers_path = os.path.join(self.checkpoint_dir, 'template_tiers.pkl')

        if not os.path.exists(log_path):
            print("ERROR: oracle_trade_log.csv not found -- run a forward pass first.")
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

        # ── Opportunity gap ───────────────────────────────────────────────────
        if fp and fp.get('ideal_profit', 0):
            ideal = fp['ideal_profit']
            captured_pct = fp['total_pnl'] / ideal * 100 if ideal else 0
            lines.append(f"\n  OPPORTUNITY GAP   (ideal if every signal traded perfectly)")
            lines.append(f"    Ideal:          ${ideal:>12,.2f}")
            lines.append(f"    Actual:         ${fp['total_pnl']:>12,.2f}   ({captured_pct:.2f}% captured)")
            lines.append(f"    #1 leak  skipped signals:   ${fp['missed']:>12,.2f}   ({fp['missed']/ideal*100:.1f}%)")
            lines.append(f"    #2 leak  exits too early:   ${fp['left_on_table']:>12,.2f}   ({fp['left_on_table']/ideal*100:.1f}%)")
            lines.append(f"    #3 leak  wrong direction:   ${fp['wrong_dir_loss']:>12,.2f}   ({fp['wrong_dir_loss']/ideal*100:.1f}%)")

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
    parser.add_argument('--no-dashboard', action='store_true', help="Disable live dashboard")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")
    parser.add_argument('--exploration-mode', action='store_true', help="Enable unconstrained exploration mode")
    parser.add_argument('--fresh', action='store_true', help="Clear all pipeline checkpoints and start fresh")
    parser.add_argument('--forward-pass', action='store_true', help="Run Phase 4 forward pass using existing playbook")
    parser.add_argument('--oos', action='store_true',
                        help="Blind out-of-sample simulation: frozen templates, separate oos_trade_log.csv/oos_report.txt, "
                             "training depth_weights.json preserved. Implies --forward-pass. "
                             "Pair with --forward-start YYYYMMDD to slice the OOS window.")
    parser.add_argument('--account-size', type=float, default=0.0, metavar='USD',
                        help="Starting account equity in USD. When set, gates trades that risk >50%% of "
                             "remaining equity (SL in dollars vs equity). Simulation ends if equity "
                             "drops below NinjaTrader MNQ intraday margin ($50). "
                             "Use 100.0 for a $100 funded account test.")
    parser.add_argument('--train-end', type=str, default=None, metavar='YYYYMMDD',
                        help="Out-of-sample guard: cap training data at this date (e.g. 20251231). "
                             "Use with --forward-start for a clean train/test split.")
    parser.add_argument('--forward-start', type=str, default=None, metavar='YYYYMMDD',
                        help="First day to include in forward pass (inclusive, e.g. 20260101)")
    parser.add_argument('--forward-end', type=str, default=None, metavar='YYYYMMDD',
                        help="Last day to include in forward pass (inclusive, e.g. 20260209)")
    parser.add_argument('--min-tier', type=int, default=None, choices=[1, 2, 3, 4],
                        help="Only activate templates of this tier or better (1=Tier1 only, 3=drop Tier4 losers)")
    parser.add_argument('--bias-threshold', type=float, default=None,
                        help="Oracle bias threshold for direction lock (default 0.55). Lower = more oracle-locked trades.")
    parser.add_argument('--dmi-threshold', type=float, default=None,
                        help="Min |dmi_diff| required to use DMI signal (default 0.0 = any non-zero DMI counts).")
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

        if args.sweep_params:
            orchestrator.run_param_sweep()
            return 0

        if (args.forward_pass or args.oos) and not args.fresh:
            # Phase 4 only (using existing playbook) or OOS blind simulation
            orchestrator.run_forward_pass(args.data,
                                          start_date=args.forward_start,
                                          end_date=args.forward_end,
                                          min_tier=args.min_tier,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=args.oos,
                                          account_size=args.account_size)
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
                # Default: Bayesian path -> Forward Pass -> Strategy Report
                orchestrator.run_forward_pass(args.data,
                                          start_date=args.forward_start,
                                          end_date=args.forward_end,
                                          min_tier=args.min_tier,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=getattr(args, 'oos', False),
                                          account_size=getattr(args, 'account_size', 0.0))
                orchestrator.run_strategy_selection()

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
