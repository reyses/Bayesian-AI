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
import queue
import glob
import random
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

# Core components
from core.bayesian_brain import MarketBayesianBrain, TradeOutcome, record_trade
from core.statistical_field_engine import StatisticalFieldEngine
from core.exit_engine import ExitEngine, ExitAction
from core.market_state import MarketState

# Training components
from training.doe_parameter_generator import DOEParameterGenerator
from training.pattern_analyzer import PatternAnalyzer
from training.progress_reporter import ProgressReporter, DayMetrics
from training.databento_loader import DatabentoLoader
from training.fractal_discovery_agent import FractalDiscoveryAgent, PatternEvent, TIMEFRAME_SECONDS
from core.feature_extraction import extract_feature_vector
from core.fractal_clustering import FractalClusteringEngine, PatternTemplate
from training.pipeline_checkpoint import PipelineCheckpoint
from core.timeframe_belief_network import TimeframeBeliefNetwork, BeliefState
from core.execution_engine import ExecutionEngine, ActionType, TradeAction, Candidate
from core.checkpoint_loader import load_checkpoints
from core.engine_factory import create_belief_network, create_execution_engine
from core.bar_processor import BarProcessor, BarProcessorHooks, BarResult

# Execution components
from training.batch_regret_analyzer import BatchRegretAnalyzer
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
from visualization.dashboard import ProgressPopup

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


class Trainer:
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
        self.brain = MarketBayesianBrain()
        self.engine = StatisticalFieldEngine()
        self.param_generator = DOEParameterGenerator(None)
        self._position = None  # PositionState or None
        self.discovery_agent = FractalDiscoveryAgent()

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

        self.dashboard_queue = queue.Queue()

        # Pattern Library (Bayesian Priors)
        self.pattern_library = {}

        # Bottom-line accumulators -- populated by run_forward_pass / run_strategy_selection
        self._fp_summary   = {}   # IS/OOS key metrics
        self._tier_summary = {}   # Phase 6 tier counts + top templates

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
                         trade_start_date: str = None,
                         bias_threshold: float = None,
                         dmi_threshold: float = None,
                         oos_mode: bool = False,
                         account_size: float = 0.0,
                         tier_preference: bool = False,
                         live_validation_days: int = 0,
                         popup_label: str = ''):
        """
        Phase 4 (IS) / Phase 5 (OOS): Forward pass -- replay data using playbook.
        Scans fractal cascade per day, matches templates, trades via ExitEngine.
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
            popup_label:  Override for popup window label (e.g. 'oos1', 'oos2', 'oos3').
        """
        _analysis_mode_early = getattr(self, '_analysis_mode', False)
        if not _analysis_mode_early:
            # Launch popup in background thread
            _mode = popup_label or ('oos' if oos_mode else 'is')
            self._launch_popup(mode=_mode)

            print("\n" + "="*80)
            if oos_mode:
                print("OOS BLIND SIMULATION (templates/scaler frozen from training)")
                # Snapshot brain keys so we can drop new patterns at the end
                _brain_keys_before_oos = set(self.brain.table.keys())
                _brain_dir_keys_before_oos = set(self.brain.dir_table.keys())
            else:
                print("PHASE 4: IS BACKTEST" if not oos_mode else "PHASE 5: OOS VALIDATION")
            if start_date or end_date:
                _lo = start_date or "start"
                _hi = end_date   or "end"
                print(f"  Date slice: {_lo} -> {_hi}")
            print("="*80)
        if self.dashboard_queue and not _analysis_mode_early:
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                      'step': 'FORWARD_PASS', 'pct': 0})

        # ── 0. Rotate previous run files: rename current → _old ────────────────
        _analysis_mode = getattr(self, '_analysis_mode', False)
        if not _analysis_mode:
            for _old_name in ('is_report.txt', 'oracle_trade_log.csv',
                              'signal_log.csv', 'depth_weights.json',
                              'run_snapshot.json'):
                _old_path = os.path.join(self.checkpoint_dir, _old_name)
                if os.path.exists(_old_path):
                    _old_dest = os.path.join(self.checkpoint_dir,
                        _old_name.replace('.', '_old.', 1))
                    try:
                        os.replace(_old_path, _old_dest)
                    except OSError:
                        pass
            # Clean up previous quarterly shards
            for _shard_prefix in ('signal_log_', 'fn_oracle_log_',
                                  'oos_signal_log_', 'oos_fn_log_'):
                for _sf in glob.glob(os.path.join(self.checkpoint_dir, f'{_shard_prefix}*.csv')):
                    try:
                        os.remove(_sf)
                    except OSError:
                        pass

        # 1. Load Prerequisites (shared with live_engine.py)
        try:
            _bundle = load_checkpoints(self.checkpoint_dir, verbose=True)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            return

        self.pattern_library = _bundle.pattern_library
        self.scaler = _bundle.scaler
        valid_template_ids = _bundle.valid_tids
        centroids_scaled = _bundle.centroids_scaled
        template_tier_map = _bundle.template_tier_map
        _exception_tids = _bundle.exception_tids

        # Tier tiebreaker logging
        if tier_preference:
            t1 = sum(1 for v in template_tier_map.values() if v == 1)
            print(f"  Tier tiebreaker ACTIVE: {t1} Tier 1 templates")

        # Depth weight detail logging (trainer-specific verbosity)
        _depth_weights_path = os.path.join(self.checkpoint_dir, 'depth_weights.json')
        if os.path.exists(_depth_weights_path):
            import json as _json
            with open(_depth_weights_path) as _dw_f:
                _dw_data = _json.load(_dw_f)
            for _dk in sorted(_dw_data.keys(), key=int):
                _dv = _dw_data[_dk]
                print(f"    depth {_dk}: avg_pnl=${_dv.get('avg_pnl',0):.1f}/trade  "
                      f"score_adj={_dv.get('score_adj',0):+.2f}  filter={_dv.get('filter_out',False)}")

        # Belief network + Execution engine (shared factory)
        _min_hold = getattr(self, '_min_hold_bars', 0)
        _exit_eng = getattr(self, '_persisted_exit_engine', None) or ExitEngine(
            mode='training',
            tick_size=self.asset.tick_size,
            tick_value=self.asset.tick_size * self.asset.point_value,
            min_hold_bars=_min_hold,
        )
        if _min_hold > 0:
            print(f"  Min-hold active: {_min_hold} bars ({_min_hold * 15 / 60:.0f} min) — "
                  f"reversal exits only before threshold")
        belief_network = create_belief_network(_bundle, self.engine)
        belief_network._atlas_root = data_source  # for pre-built TF parquet loading
        _exec_engine = create_execution_engine(
            bundle=_bundle,
            brain=self.brain,
            belief_network=belief_network,
            exit_engine=_exit_eng,
            tick_size=self.asset.tick_size,
            point_value=self.asset.point_value,
            mode='oos' if oos_mode else 'is',
            tier_preference=tier_preference,
            bias_threshold=bias_threshold if bias_threshold is not None else 0.55,
            dmi_threshold=dmi_threshold if dmi_threshold is not None else 0.0,
            depth_only=getattr(self, '_depth_only', None),
        )

        # OOS compressed mode: widen gate1_dist to match live engine
        # Live uses: gate1_dist = 4.5 + aggression * 10.0 (default agg=0.5 → 9.5)
        if oos_mode:
            _oos_g1 = 4.5 + 0.5 * 10.0  # match live default aggression
            _exec_engine.gate1_dist = _oos_g1
            print(f"  OOS compressed mode: gate1_dist={_oos_g1:.1f}")

        # Feature extractor for IS candidates (22D when --lookback)
        _use_lb = getattr(self, '_use_lookback', False)
        _feat_extractor = FractalClusteringEngine(use_lookback=_use_lb)

        # Slippage RNG (seeded for reproducibility)
        _slip_ticks = float(getattr(self, '_slippage_ticks', 0.0))
        _slip_rng = random.Random(42) if _slip_ticks > 0 else None
        _tick_val = self.asset.tick_value  # $0.50 for MNQ
        if _slip_ticks > 0:
            print(f"  Slippage: +-{_slip_ticks:.1f} ticks/trade (+-${_slip_ticks * _tick_val:.2f})")

        # BarProcessor parity replay is created AFTER inline OOS completes
        # (with its own fresh EE to avoid shared-state mutation)

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
        if start_date or end_date:
            _before = len(daily_files_15s)
            daily_files_15s = [
                f for f in daily_files_15s
                if (not start_date or _file_sort_key(f) >= start_date)
                and (not end_date   or _file_sort_key(f) <= end_date)
            ]
            print(f"  Date filter applied: {_before} -> {len(daily_files_15s)} files "
                  f"({start_date or 'start'} -> {end_date or 'end'})")

        print(f"  Found {len(daily_files_15s)} files to simulate.")
        if trade_start_date:
            print(f"  Warmup until {trade_start_date} (context only, no trades)")

        # ── OOS warmup: load tail of IS data for regression context ────────
        # Without warmup, the first OOS file starts cold (z=0, v=0 for first
        # 21 bars of each TF). Macro TF workers (1h) lose ~4% of states.
        # Prepending IS tail gives regression full history from the start.
        # Also loads IS 4h/5s/1s for external-file TF workers.
        _oos_warmup_df = None
        _oos_warmup_n_bars = 0
        _oos_warmup_ext = {}  # {tf_label: DataFrame} for external TFs (4h, 5s, 1s)
        if oos_mode:
            # Derive IS path: DATA/ATLAS_OOS → DATA/ATLAS
            _is_root = data_source.replace('ATLAS_OOS', 'ATLAS') if 'ATLAS_OOS' in data_source else None
            _is_15s_dir = os.path.join(_is_root, '15s') if _is_root else None
            if _is_15s_dir and os.path.isdir(_is_15s_dir):
                _is_files = sorted(glob.glob(os.path.join(_is_15s_dir, '*.parquet')))
                if _is_files:
                    # Load last IS monthly file (≈21 trading days)
                    _warmup_file = _is_files[-1]
                    _warmup_stem = os.path.basename(_warmup_file).replace('.parquet', '')
                    try:
                        _oos_warmup_df = pd.read_parquet(_warmup_file)
                        if 'timestamp' in _oos_warmup_df.columns and not np.issubdtype(
                                _oos_warmup_df['timestamp'].dtype, np.number):
                            _oos_warmup_df['timestamp'] = _oos_warmup_df['timestamp'].apply(
                                lambda x: x.timestamp())
                        _oos_warmup_n_bars = len(_oos_warmup_df)
                        print(f"  OOS warmup: {_oos_warmup_n_bars:,} bars from IS 15s/{_warmup_stem}")

                        # Also load IS data for external TF workers (4h, 5s, 1s)
                        for _ext_tf in ('4h', '5s', '1s'):
                            _ext_path = os.path.join(_is_root, _ext_tf, f"{_warmup_stem}.parquet")
                            if os.path.exists(_ext_path):
                                try:
                                    _ext_df = pd.read_parquet(_ext_path)
                                    if 'timestamp' in _ext_df.columns and not np.issubdtype(
                                            _ext_df['timestamp'].dtype, np.number):
                                        _ext_df['timestamp'] = _ext_df['timestamp'].apply(
                                            lambda x: x.timestamp())
                                    _oos_warmup_ext[_ext_tf] = _ext_df
                                    print(f"  OOS warmup: {len(_ext_df):,} bars from IS {_ext_tf}/{_warmup_stem}")
                                except Exception:
                                    pass
                    except Exception as _wup_err:
                        print(f"  OOS warmup: failed to load IS tail ({_wup_err})")
                        _oos_warmup_df = None

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

        # ── Daily session tracking & drawdown survival ──────────────────────
        _daily_ledger = []          # [{date, trades, wins, pnl, equity, consec_loss_days, aggression}]
        _consec_losing_days = 0     # current streak of losing days
        _max_consec_losing_days = 0
        _dd_aggression = 1.0        # 1.0 = full, scales down during drawdown
        _daily_peak_equity = running_equity  # reset each day to track intraday DD

        # ── Intraday dip tracking (min equity calculation) ────────────────
        _day_running_pnl = 0.0      # cumulative PnL within current day, resets each day
        _day_min_pnl = 0.0          # worst intraday dip this day (most negative)
        _worst_intraday_dip = 0.0   # worst dip across ALL days (most negative)
        _worst_dip_date = ''        # which day had the worst dip
        _all_day_dips = []          # list of (date, min_pnl) for every trading day
        _cal_day_trades = []        # trades within current calendar day (per-day ledger)
        _prev_cal_date = ''         # readable date of previous calendar day
        _current_day = None         # unix day number for calendar-day boundary detection

        # ── Cumulative equity curve (never resets — carries across days) ──
        _cumul_pnl = 0.0            # running total PnL from trade 1
        _cumul_peak = 0.0           # highest point on the equity curve
        _cumul_trough = 0.0         # lowest point on the equity curve
        _cumul_trough_date = ''     # date of the lowest point
        _cumul_max_dd = 0.0         # max drawdown from peak (peak - trough)
        _cumul_dd_peak_date = ''    # date of peak before max drawdown
        _cumul_dd_trough_date = ''  # date of trough during max drawdown

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

        def _physics_fields(p):
            """Extract physics state fields from a pattern event for oracle records."""
            _st = getattr(p, 'state', None)
            _f_mom = float(getattr(_st, 'F_momentum', 0.0)) if _st else 0.0
            _f_rev = float(getattr(_st, 'mean_reversion_force', 0.0)) if _st else 0.0
            _f_rev_abs = abs(_f_rev)
            _vel = float(getattr(_st, 'velocity', 0.0)) if _st else 0.0
            _sigma = float(getattr(_st, 'regression_sigma', 0.0)) if _st else 0.0
            return {
                'F_momentum': round(_f_mom, 6),
                'F_reversion': round(_f_rev, 6),
                'mom_rev_ratio': round(abs(_f_mom) / _f_rev_abs if _f_rev_abs > 0 else 0.0, 2),
                'hurst': round(float(getattr(_st, 'hurst_exponent', 0.0)) if _st else 0.0, 3),
                'tunnel_prob': round(float(getattr(_st, 'reversion_probability', 0.0)) if _st else 0.0, 3),
                'velocity': round(_vel, 6),
                'sigma': round(_sigma, 6),
                'band_speed': round(abs(_vel) / _sigma if _sigma > 0 else 0.0, 4),
            }

        def _quantum_score(state, belief, side, template_wr, norm_dist):
            """Compute quantum P(success) score from orphaned + active fields.

            Observation only — logged but not used for decisions (yet).
            Original scoring: P_i > 0.75 AND tunnel > 0.60 AND low entropy.
            """
            if state is None:
                return {'q_score': 0.0, 'q_wave': 0.0, 'q_tunnel': 0.0,
                        'q_entropy': 0.0, 'q_coherence': 0.0}

            # Wave function probability (dominant direction)
            p_center = float(getattr(state, 'P_at_center', 0.0))
            p_upper = float(getattr(state, 'P_near_upper', 0.0))
            p_lower = float(getattr(state, 'P_near_lower', 0.0))
            p_wave = max(p_center, p_upper, p_lower)

            # Tunnel probability (reversion confidence)
            p_tunnel = float(getattr(state, 'reversion_probability', 0.0))

            # Entropy (chaos measure — lower = more decisive)
            entropy = float(getattr(state, 'entropy_normalized', 0.5))
            entropy_inv = 1.0 - entropy  # higher = better

            # Conviction (TBN consensus)
            conviction = belief.conviction if belief is not None else 0.0

            # Coherence (TF alignment)
            coherence = float(getattr(state, 'oscillation_entropy_normalized', 0.0))

            # Momentum alignment (F_momentum sign vs trade direction)
            f_mom = float(getattr(state, 'F_momentum', 0.0))
            mom_align = 1.0
            if side == 'long' and f_mom < 0:
                mom_align = 0.6
            elif side == 'short' and f_mom > 0:
                mom_align = 0.6

            # Weighted combination
            q = (0.20 * p_wave +
                 0.15 * p_tunnel +
                 0.15 * entropy_inv +
                 0.15 * conviction +
                 0.10 * template_wr +
                 0.10 * (1.0 - min(1.0, norm_dist / 3.0)) +
                 0.10 * mom_align +
                 0.05 * coherence)

            return {
                'q_score': round(q, 4),
                'q_wave': round(p_wave, 4),
                'q_tunnel': round(p_tunnel, 4),
                'q_entropy': round(entropy_inv, 4),
                'q_coherence': round(coherence, 4),
            }

        def _macro_obs(bn, trade_side):
            """Macro trend observation columns (non-actionable — research only)."""
            try:
                mt = bn.get_macro_trend()
            except Exception:
                mt = {'direction': None, 'strength': 0.0, 'macro_z': 0.0, 'macro_band_pos': 0.0, 'detail': ''}
            _dir = mt.get('direction')
            _aligned = ''
            if _dir is not None and trade_side:
                _aligned = 'WITH' if trade_side.lower() == _dir else 'COUNTER'
            return {
                'macro_direction': _dir or 'neutral',
                'macro_strength': round(mt.get('strength', 0.0), 3),
                'macro_z': round(mt.get('macro_z', 0.0), 3),
                'macro_band_pos': round(mt.get('macro_band_pos', 0.0), 3),
                'macro_alignment': _aligned or 'NEUTRAL',
                'macro_detail': mt.get('detail', ''),
            }

        def _dm_rec(p, gate, day, ts_val, micro_z_val, macro_z_val, pattern_val,
                    dist=0.0, conviction=0.0, template_id='', tier='', playbook=''):
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
                'template_id': str(template_id), 'tier': str(tier),
                'playbook': playbook,
                # Physics state (for gate threshold analysis)
                **_physics_fields(p),
                # Trade outcome (filled in later if gate='traded')
                'trade_direction': '', 'trade_result': '', 'trade_pnl': 0.0,
                'exit_reason': '', 'exit_signal_reason': '',
                'exit_conviction': 0.0, 'exit_wave_maturity': 0.0,
            }

        # Default output dirs (may be overridden later for inline OOS)
        _out_dir = self.checkpoint_dir
        _reports_out = os.path.join(os.path.dirname(self.checkpoint_dir), 'reports')
        import csv as _csv

        # Per-trade oracle tracking
        _ORACLE_LABEL_NAMES = {2: 'MEGA_LONG', 1: 'SCALP_LONG', 0: 'NOISE', -1: 'SCALP_SHORT', -2: 'MEGA_SHORT'}
        oracle_trade_records = []  # completed per-trade oracle dicts
        _trade_replays = []       # per-trade bar series for I-MR replay
        pending_oracle = None      # oracle facts for currently open trade
        _pending_dm_idx = None     # index into decision_matrix_records for open trade

        # Streaming trade log — append each trade to disk as it completes
        _stream_log_name = 'oos_trade_log.csv' if oos_mode else 'oracle_trade_log.csv'
        _stream_log_path = os.path.join(_out_dir, _stream_log_name)
        _stream_log_header_written = False
        _stream_log_file = None

        def _stream_trade(rec):
            """Append a single trade record to the streaming CSV."""
            nonlocal _stream_log_header_written, _stream_log_file
            if _stream_log_file is None:
                _stream_log_file = open(_stream_log_path, 'w', newline='', encoding='utf-8')
                _w = _csv.DictWriter(_stream_log_file, fieldnames=list(rec.keys()))
                _w.writeheader()
                _stream_log_header_written = True
                _w.writerow(rec)
                _stream_log_file.flush()
            else:
                _w = _csv.DictWriter(_stream_log_file, fieldnames=list(rec.keys()))
                _w.writerow(rec)
                # Flush every 50 trades for peekability without I/O overhead
                if len(oracle_trade_records) % 50 == 0:
                    _stream_log_file.flush()
        fn_potential_pnl    = 0.0  # dollar potential of real moves we missed (gate-blocked)
        score_loser_pnl     = 0.0  # dollar potential of real moves we correctly passed over (took better trade same bar)

        # PID Shadow Log
        pid_oracle_records = []

        # Detection funnel counters (bar-level, across all days)
        total_bars_processed  = 0   # Every 15s bar we step through
        bars_with_detection   = 0   # Bars where pattern_map had at least one candidate
        bars_slot_blocked     = 0   # Bars with detection but position was open (slot occupied)

        # Skip reason counters (per-candidate across all days)
        # Gate skip counters now tracked inside _exec_engine.gate_stats
        # Accessed via _exec_engine.get_skip_counts() in report section
        n_signals_seen   = 0   # Total candidate signals evaluated (all gates combined)
        depth_traded     = defaultdict(int)  # depth -> trade count (1=high TF, 6=15s)

        # FN oracle records: per-signal log of missed real moves with worker snapshots.
        # Answers: "when we missed a profitable move, what did the workers think?"
        # If workers agreed with oracle direction on FN signals, a gate is too strict.
        fn_oracle_records = []

        # Pre-count actual trading days for accurate progress tracking
        _total_trading_days = 0
        for _f in daily_files_15s:
            try:
                _ts_col = pd.read_parquet(_f, columns=['timestamp'])
                if not _ts_col.empty:
                    _ts = _ts_col['timestamp']
                    if np.issubdtype(_ts.dtype, np.number):
                        _total_trading_days += pd.to_datetime(_ts, unit='s').dt.date.nunique()
                    else:
                        _total_trading_days += pd.to_datetime(_ts).dt.date.nunique()
            except Exception:
                pass
        _total_trading_days = max(_total_trading_days, 1)
        _cumulative_days = 0
        print(f"  Total trading days: {_total_trading_days}")

        # ── Live Validation config ──
        # All files go through inline OOS first. After the main loop, the last N
        # trading days are replayed through a FRESH BarProcessor (own EE) for parity.
        _live_val_days = live_validation_days if (oos_mode and live_validation_days > 0) else 0
        _live_val_trades = []       # trades from live validation days
        _live_val_day_ledger = []   # per-day stats for live validation
        _live_val_gate_stats = {}   # gate funnel for live validation
        _live_val_dir_sources = {}  # direction source distribution
        if _live_val_days > 0:
            print(f"  Live validation: last {_live_val_days} trading days replayed via BarProcessor after inline OOS")

        # Pre-loop defaults (survive thin-market skips)
        _pp_enabled = getattr(self, '_ping_pong', False)
        _pp_flip_count = 0
        _pp_all_trades = []

        _pbar = tqdm(total=_total_trading_days, desc='Forward Pass', unit='day',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
                     ascii=True, dynamic_ncols=True)
        for day_idx, day_file in enumerate(daily_files_15s):
            day_date = os.path.basename(day_file).replace('.parquet', '')

            # ── Warmup gate: process TBN context but skip trading ──
            if trade_start_date:
                _day_key = day_date.replace('_', '')  # 2025_01 → 202501
                if _day_key + '01' < trade_start_date:
                    # Still run discovery to build TBN state (IS only)
                    if not oos_mode:
                        self.discovery_agent.scan_day_cascade(data_source, day_date)
                    _pbar.set_postfix_str(f"{day_date} | WARMUP (context only)", refresh=False)
                    _pbar.update(1)
                    continue

            # A. Fractal Cascade Scan (get actionable patterns with chains)
            # OOS mode: skip discovery — uses compressed per-bar features (same as live)
            # IS mode: full recursive discovery for library training
            if oos_mode:
                actionable_patterns = []
            else:
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
            _df_4h  = _load_fine('4h')

            # Pre-extract 1s numpy arrays for wick-aware inner loop
            if _df_1s is not None and not _df_1s.empty:
                _1s_ts    = _df_1s['timestamp'].values.astype(np.float64)
                _1s_highs = _df_1s['high'].values.astype(np.float64)
                _1s_lows  = _df_1s['low'].values.astype(np.float64)
                _has_1s   = True
            else:
                _has_1s = False

            # ── OOS warmup: prepend IS tail to first file for regression context ──
            # This gives macro TF workers (1h, 30m) full regression history
            # from bar 0 instead of starting cold with z=0.
            _warmup_offset = 0  # added to _bar_i for tick_all() calls
            if _oos_warmup_df is not None and day_idx == 0:
                _df_combined = pd.concat([_oos_warmup_df, df_15s], ignore_index=True)
                _warmup_offset = _oos_warmup_n_bars

                # Prepend IS data for external TF workers (4h, 5s, 1s)
                _df_5s_w = _df_5s
                _df_1s_w = _df_1s
                _df_4h_w = _df_4h
                for _ext_tf, _ext_warmup_df in _oos_warmup_ext.items():
                    if _ext_tf == '4h' and _df_4h is not None:
                        _df_4h_w = pd.concat([_ext_warmup_df, _df_4h], ignore_index=True)
                    elif _ext_tf == '5s' and _df_5s is not None:
                        _df_5s_w = pd.concat([_ext_warmup_df, _df_5s], ignore_index=True)
                    elif _ext_tf == '1s' and _df_1s is not None:
                        _df_1s_w = pd.concat([_ext_warmup_df, _df_1s], ignore_index=True)

                try:
                    _states_combined = self.engine.batch_compute_states(_df_combined, use_cuda=True)
                    # _states_map uses only OOS bars (remap bar_idx to start at 0)
                    _states_15s = [
                        {**s, 'bar_idx': s['bar_idx'] - _warmup_offset}
                        for s in _states_combined
                        if s['bar_idx'] >= _warmup_offset
                    ]
                    # TBN gets full combined data (resampled TFs have IS history)
                    # External TFs also get IS data prepended for regression context
                    belief_network.prepare_day(_df_combined, states_micro=_states_combined,
                                               df_5s=_df_5s_w, df_1s=_df_1s_w, df_4h=_df_4h_w)
                    print(f"    OOS warmup applied: {_warmup_offset:,} IS bars prepended "
                          f"({len(_states_15s):,} OOS states)")
                except Exception as _wup_err:
                    print(f"    OOS warmup failed ({_wup_err}), falling back to cold start")
                    _warmup_offset = 0
                    _states_15s = self.engine.batch_compute_states(df_15s, use_cuda=True)
                    belief_network.prepare_day(df_15s, states_micro=_states_15s,
                                               df_5s=_df_5s, df_1s=_df_1s, df_4h=_df_4h)
            else:
                # Normal path (IS mode, or OOS file 2+)
                try:
                    _states_15s = self.engine.batch_compute_states(df_15s, use_cuda=True)
                except Exception as _bn_err:
                    _states_15s = []

                # TBN: prepare workers with day data
                try:
                    belief_network.prepare_day(df_15s, states_micro=_states_15s,
                                               df_5s=_df_5s, df_1s=_df_1s, df_4h=_df_4h)
                except Exception:
                    belief_network.prepare_day(df_15s, states_micro=[],
                                               df_5s=_df_5s, df_1s=_df_1s, df_4h=_df_4h)

            # Accumulate worker state counts for report diagnostics
            for _wlbl, _wcnt in belief_network.get_worker_state_counts().items():
                _worker_total_states[_wlbl]   = _worker_total_states.get(_wlbl, 0) + _wcnt
                _worker_days_with_data[_wlbl] = _worker_days_with_data.get(_wlbl, 0) + (1 if _wcnt > 0 else 0)

            # Reset PID analyzer for the day
            _day_sigmas = [s['state'].regression_sigma for s in _states_15s if s['state'].regression_sigma > 0]
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
            # ExitEngine expects: price, state (optional for trail?), timestamp

            # Pre-compute states for the day?
            # _scan_day_cascade already computed states for the patterns.
            # For 15s bars that are NOT patterns, we might need state for exit logic?
            # ExitEngine usually needs price and timestamp. Some logic might use state.
            # Let's assume exit logic primarily uses price action (trail, stop).

            current_position_open = False
            active_entry_price = 0.0
            active_entry_time = 0.0
            active_max_hold_bars = 960  # default 4h; overwritten at entry from pattern timeframe
            active_side = 'long'
            active_template_id = None
            _belief = None  # initialized for scope safety (set at entry)

            # ExitEngine now lives inside _exec_engine (unified execution)

            # Ping-pong state (direction refinement)
            _pp_enabled = getattr(self, '_ping_pong', False)
            _pp_flip_count = 0
            _pp_last_exit_params = None  # TF-scaled params from exited trade
            _pp_all_trades = []  # cross-day PP trade accumulator

            # Output directory: route PP runs to snowflake/ subfolder
            if _pp_enabled:
                _out_dir = os.path.join(self.checkpoint_dir, 'snowflake')
                os.makedirs(_out_dir, exist_ok=True)
                _reports_out = os.path.join('reports', 'snowflake')
                os.makedirs(_reports_out, exist_ok=True)
            else:
                _out_dir = self.checkpoint_dir
                _reports_out = os.path.join(os.path.dirname(self.checkpoint_dir), 'reports')

            _bar_i = 0  # 15s bar index for belief network worker ticks

            _pp_conv_thresh = getattr(self, '_pp_conviction', 0.55)
            _pp_sl_ov = getattr(self, '_pp_sl_override', 0)
            _pp_tp_ov = getattr(self, '_pp_tp_override', 0)
            _pp_trail_ov = getattr(self, '_pp_trail_override', 0)

            def _pp_try_flip(exited_side, exited_tid, price, ts_raw, ep):
                """Check belief conviction and attempt ping-pong flip after exit.
                Returns (should_flip, flip_side) or (False, None)."""
                if not _pp_enabled or ep is None:
                    return False, None
                _belief = belief_network.get_belief()
                if _belief is None or not _belief.is_confident:
                    return False, None
                flip_side = 'short' if exited_side == 'long' else 'long'
                # Belief still agrees with old direction at high conviction -> skip
                if _belief.direction == exited_side and _belief.conviction > 0.60:
                    return False, None
                if _belief.conviction < _pp_conv_thresh:
                    return False, None
                # Check live direction bias (reject if known loser)
                bias = self.brain.get_dir_bias(exited_tid)
                if bias:
                    d = 'long' if flip_side == 'long' else 'short'
                    total = bias.get(f'{d}_w', 0) + bias.get(f'{d}_l', 0)
                    if total >= 5 and bias.get(f'{d}_w', 0) / total < 0.30:
                        return False, None
                return True, flip_side

            for row in df_15s.itertuples():
                total_bars_processed += 1
                ts_raw = row.timestamp

                # ── Per-day progress update ──────────────────────────────
                _row_day = int(ts_raw) // 86400
                if _row_day != _current_day:
                    # ── Flush previous calendar day to ledger ─────────
                    if _prev_cal_date and _cal_day_trades:
                        _cd_pnl = sum(t.pnl for t in _cal_day_trades)
                        _cd_wins = sum(1 for t in _cal_day_trades if t.result == 'WIN')
                        _cd_losing = _cd_pnl < 0
                        if _cd_losing:
                            _consec_losing_days += 1
                        else:
                            _consec_losing_days = 0
                        _max_consec_losing_days = max(_max_consec_losing_days, _consec_losing_days)
                        # Capture intraday dip
                        _all_day_dips.append((_prev_cal_date, _day_min_pnl))
                        if _day_min_pnl < _worst_intraday_dip:
                            _worst_intraday_dip = _day_min_pnl
                            _worst_dip_date = _prev_cal_date
                        _daily_ledger.append({
                            'date': _prev_cal_date,
                            'trades': len(_cal_day_trades),
                            'wins': _cd_wins,
                            'pnl': _cd_pnl,
                            'min_dip': _day_min_pnl,
                            'cumul_pnl': _cumul_pnl,
                            'cumul_dd': _cumul_peak - _cumul_pnl,
                            'equity': running_equity if _equity_enabled else total_pnl + sum(t.pnl for t in day_trades),
                            'consec_loss_days': _consec_losing_days,
                            'aggression': _dd_aggression,
                        })
                    _current_day = _row_day
                    _cumulative_days += 1
                    # Readable date from unix day
                    import datetime as _dt_mod
                    _prev_cal_date = _dt_mod.datetime.utcfromtimestamp(int(ts_raw)).strftime('%Y-%m-%d')
                    _cal_day_trades = []
                    # Reset daily peak for intraday drawdown tracking
                    if _equity_enabled:
                        _daily_peak_equity = running_equity
                    # Reset intraday PnL for min-equity dip tracking
                    _day_running_pnl = 0.0
                    _day_min_pnl = 0.0
                    _running_pnl = total_pnl + sum(t.pnl for t in day_trades)
                    _running_trades = total_trades + len(day_trades)
                    _running_wins = total_wins + sum(1 for t in day_trades if t.result == 'WIN')
                    _pbar.set_postfix_str(
                        f'{day_date} | ${_running_pnl:,.0f} PnL | {_running_trades} trades | '
                        f'{(_running_wins/_running_trades*100) if _running_trades else 0:.0f}% WR',
                        refresh=False)
                    _pbar.update(1)
                    if self.dashboard_queue:
                        _wr = (_running_wins / _running_trades * 100) if _running_trades > 0 else 0.0
                        _all_pnls = [t['actual_pnl'] for t in oracle_trade_records]
                        _gw = sum(p for p in _all_pnls if p > 0)
                        _gl = abs(sum(p for p in _all_pnls if p < 0))
                        _pf = _gw / _gl if _gl > 0 else 0.0
                        # Capture rate quartile buckets
                        _caps = [t.get('capture_rate', 0) for t in oracle_trade_records
                                 if t.get('capture_rate') is not None]
                        _c_rev = sum(1 for c in _caps if c <= 0)
                        _c_q1 = sum(1 for c in _caps if 0 < c <= 0.25)
                        _c_q2 = sum(1 for c in _caps if 0.25 < c <= 0.50)
                        _c_q3 = sum(1 for c in _caps if 0.50 < c <= 0.75)
                        _c_q4 = sum(1 for c in _caps if 0.75 < c <= 1.0)
                        _c_plus = sum(1 for c in _caps if c > 1.0)
                        self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                                  'step': f'FORWARD_PASS  day {_cumulative_days}/{_total_trading_days}',
                                                  'pct': round(_cumulative_days / _total_trading_days * 100, 1),
                                                  'pnl': _running_pnl,
                                                  'trades': _running_trades,
                                                  'wr': round(_wr, 1),
                                                  'pf': round(_pf, 2),
                                                  'gross_w': round(_gw, 0),
                                                  'gross_l': round(_gl, 0),
                                                  'cap_reversed': _c_rev,
                                                  'cap_q1': _c_q1,
                                                  'cap_q2': _c_q2,
                                                  'cap_q3': _c_q3,
                                                  'cap_q4': _c_q4,
                                                  'cap_100plus': _c_plus})

                # Snap to 60s boundary to match pattern_map keys
                ts = int(ts_raw) // 60 * 60
                price = getattr(row, 'close', getattr(row, 'price', 0.0))

                # Feed price + DMI to dashboard chart (every 20 bars = 5 min)
                if self.dashboard_queue is not None and _bar_i % 20 == 0:
                    _dmi_p, _dmi_m = 0.0, 0.0
                    _5m_w = belief_network.workers.get(300)
                    if _5m_w is not None and _5m_w._states:
                        _mi = _5m_w._last_tf_bar_idx
                        if 0 <= _mi < len(_5m_w._states):
                            _raw = _5m_w._states[_mi]
                            _ms = _raw['state'] if isinstance(_raw, dict) and 'state' in _raw else _raw
                            _dmi_p = getattr(_ms, 'dmi_plus', 0.0)
                            _dmi_m = getattr(_ms, 'dmi_minus', 0.0)
                    self.dashboard_queue.put({
                        'type': 'TICK_UPDATE', 'price': price,
                        'dmi_plus': round(_dmi_p, 2),
                        'dmi_minus': round(_dmi_m, 2)})

                # Belief network: tick all workers (event-driven by TF bar change)
                # 1h worker updates once per 240 bars; 15s worker updates every bar
                # _warmup_offset > 0 on first OOS file: IS data prepended to TBN states,
                # so bar_i=0 in OOS needs to point to states[_warmup_offset].
                belief_network.tick_all(_bar_i + _warmup_offset)

                # PID ANALYZER TICK
                _pid_state = _states_map.get(_bar_i - 1)
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

                # 0. CME maintenance cutoff: flatten before daily halt (16:45-18:00 ET)
                _in_maintenance = _exit_eng.is_maintenance_window(ts_raw)
                if _in_maintenance and _exec_engine.in_position:
                    pos = self._position
                    if pos.side == 'short':
                        _maint_pnl = (pos.entry_price - price) * self.asset.point_value
                        _trade_mfe_ticks = (pos.entry_price - pos.peak_favorable) / self.asset.tick_size
                    else:
                        _maint_pnl = (price - pos.entry_price) * self.asset.point_value
                        _trade_mfe_ticks = (pos.peak_favorable - pos.entry_price) / self.asset.tick_size
                    if _slip_rng:
                        _maint_pnl += _slip_rng.uniform(-_slip_ticks, _slip_ticks) * _tick_val
                    self._position = None
                    _exec_engine.position_closed()
                    outcome = record_trade(
                        self.brain, tid=active_template_id,
                        entry_price=active_entry_price,
                        exit_price=price,
                        pnl=_maint_pnl, side=active_side,
                        exit_reason='maintenance_flat', timestamp=ts_raw,
                        entry_time=active_entry_time, exit_time=ts_raw,
                        tick_value=self.asset.tick_value,
                        hold_bars=pos.bars_held,
                    )
                    day_trades.append(outcome)
                    _cal_day_trades.append(outcome)
                    current_position_open = False
                    _day_running_pnl += outcome.pnl
                    _day_min_pnl = min(_day_min_pnl, _day_running_pnl)
                    _cumul_pnl += outcome.pnl
                    if _cumul_pnl > _cumul_peak:
                        _cumul_peak = _cumul_pnl
                    if _cumul_pnl < _cumul_trough:
                        _cumul_trough = _cumul_pnl
                        _cumul_trough_date = _prev_cal_date
                    _dd_from_peak = _cumul_peak - _cumul_pnl
                    if _dd_from_peak > _cumul_max_dd:
                        _cumul_max_dd = _dd_from_peak
                        _cumul_dd_trough_date = _prev_cal_date
                    if _equity_enabled:
                        running_equity += outcome.pnl
                        peak_equity = max(peak_equity, running_equity)
                        trough_equity = min(trough_equity, running_equity)
                    if pending_oracle is not None:
                        o_mfe = pending_oracle['oracle_mfe']
                        o_mae = pending_oracle['oracle_mae']
                        oracle_favorable = o_mfe if pending_oracle['direction'] == 'LONG' else o_mae
                        oracle_potential = oracle_favorable * self.asset.point_value
                        capture = outcome.pnl / oracle_potential if oracle_potential > 0 else 0.0
                        _tp_potential = pending_oracle.get('tp_ticks', 0) * self.asset.tick_value
                        _target_capture = outcome.pnl / _tp_potential if _tp_potential > 0 else 0.0
                        oracle_trade_records.append({
                            **pending_oracle,
                            'exit_price': outcome.exit_price,
                            'exit_time': ts_raw,
                            'hold_bars': max(1, pos.bars_held),
                            'exit_reason': 'maintenance_flat',
                            'actual_pnl': outcome.pnl,
                            'oracle_potential_pnl': oracle_potential,
                            'capture_rate': round(min(capture, 9.99), 4),
                            'target_capture': round(min(_target_capture, 9.99), 4),
                            'result': outcome.result,
                            'exit_workers': '{}',
                            'exit_conviction': 0.0,
                            'exit_wave_maturity': 0.0,
                            'exit_signal_reason': 'maintenance_flat',
                            'exit_decay_score': 0.0,
                            'trade_mfe_ticks': round(_trade_mfe_ticks, 2),
                            'price_expected_error': 0.0,
                        })
                        _stream_trade(oracle_trade_records[-1])
                        belief_network.stop_trade_tracking()
                        pending_oracle = None
                        _pending_dm_idx = None
                    _exit_eng.record_trade_outcome(
                        trade_mfe_ticks=_trade_mfe_ticks,
                        actual_pnl_ticks=_maint_pnl / self.asset.tick_value,
                        capture_rate=0.0,
                    )
                    continue  # Skip to next bar — no entries during maintenance

                # 1. Manage existing position — via ExecutionEngine
                if _exec_engine.in_position:
                    # Update trade pace cache before exit evaluation
                    _tp = belief_network.get_trade_progress(
                        price, tick_size=self.asset.tick_size)
                    belief_network._trade_pace_cache = _tp
                    belief_network._trade_pace_blend = _tp.get('pace', 1.0) - 1.0

                    _exit_sig = belief_network.get_exit_signal(
                        _exec_engine.active_side, active_entry_price,
                        discovery_tf_seconds=self._position.discovery_tf_seconds if self._position else 300.0)
                    _band_ctx = (belief_network.get_band_confluence()
                                 if hasattr(belief_network, 'get_band_confluence') else None)
                    _bar_state = _states_map.get(_bar_i - 1)
                    _f_net = float(getattr(_bar_state, 'net_force', 0.0)) if _bar_state else 0.0
                    _bar_high = getattr(row, 'high', price)
                    _bar_low = getattr(row, 'low', price)

                    _sub_h, _sub_l = None, None
                    if _has_1s:
                        _s0 = np.searchsorted(_1s_ts, ts_raw, side='left')
                        _s1 = np.searchsorted(_1s_ts, ts_raw + 15, side='left')
                        if _s1 > _s0:
                            _sub_h = _1s_highs[_s0:_s1].tolist()
                            _sub_l = _1s_lows[_s0:_s1].tolist()

                    _noise = float(getattr(_bar_state, 'swing_noise_ticks', 0.0)) if _bar_state else 0.0
                    _exit_action = _exec_engine.on_bar(
                        price=price, bar_high=_bar_high, bar_low=_bar_low,
                        bar_index=_bar_i,
                        net_force=_f_net,
                        sub_bar_highs=_sub_h, sub_bar_lows=_sub_l,
                        band_context=_band_ctx, exit_signal=_exit_sig,
                        noise_ticks=_noise,
                    )

                    if _exit_action.type == ActionType.EXIT:
                        _ee_pnl = _exit_action.pnl_dollars
                        if _slip_rng:
                            _ee_pnl += _slip_rng.uniform(-_slip_ticks, _slip_ticks) * _tick_val
                        _ee_reason = _exit_action.exit_reason
                        _exit_result = _exit_action.exit_result
                        # Capture trade MFE before clearing position
                        _pos_st = _exec_engine.pos_state
                        if _pos_st is not None:
                            _peak = _pos_st.peak_favorable
                            if active_side == 'long':
                                _trade_mfe_ticks = (_peak - active_entry_price) / self.asset.tick_size
                            else:
                                _trade_mfe_ticks = (active_entry_price - _peak) / self.asset.tick_size
                        else:
                            _trade_mfe_ticks = 0.0
                        self._position = None
                        _exec_engine.position_closed()
                        # Trade exit marker on dashboard
                        if self.dashboard_queue is not None:
                            self.dashboard_queue.put({
                                'type': 'TRADE_MARKER', 'action': 'EXIT',
                                'side': active_side, 'price': _exit_action.price,
                                'pnl': _ee_pnl})
                        outcome = record_trade(
                            self.brain, tid=active_template_id,
                            entry_price=active_entry_price,
                            exit_price=_exit_action.price,
                            pnl=_ee_pnl, side=active_side,
                            exit_reason=_ee_reason, timestamp=ts_raw,
                            entry_time=active_entry_time, exit_time=ts_raw,
                            tick_value=self.asset.tick_value,
                            hold_bars=_exit_action.bars_held,
                        )
                        day_trades.append(outcome)
                        _cal_day_trades.append(outcome)
                        current_position_open = False

                        # Track intraday dip (min equity calc — no account needed)
                        _day_running_pnl += outcome.pnl
                        _day_min_pnl = min(_day_min_pnl, _day_running_pnl)

                        # Track cumulative equity curve (never resets)
                        _cumul_pnl += outcome.pnl
                        if _cumul_pnl > _cumul_peak:
                            _cumul_peak = _cumul_pnl
                        if _cumul_pnl < _cumul_trough:
                            _cumul_trough = _cumul_pnl
                            _cumul_trough_date = _prev_cal_date
                        _dd_from_peak = _cumul_peak - _cumul_pnl
                        if _dd_from_peak > _cumul_max_dd:
                            _cumul_max_dd = _dd_from_peak
                            _cumul_dd_trough_date = _prev_cal_date

                        if _equity_enabled:
                            running_equity += outcome.pnl
                            peak_equity = max(peak_equity, running_equity)
                            trough_equity = min(trough_equity, running_equity)
                            # Aggression scales with equity: $500→100%, $250→50%, $100→20% floor
                            _daily_peak_equity = max(_daily_peak_equity, running_equity)
                            _dd_aggression = min(1.0, max(0.2, running_equity / account_size)) if account_size > 0 else 1.0
                            if running_equity < _NINJATRADER_MNQ_MARGIN:
                                account_ruined = True
                                ruin_day = ruin_day or day_date

                        if pending_oracle is not None:
                            o_mfe = pending_oracle['oracle_mfe']
                            o_mae = pending_oracle['oracle_mae']
                            oracle_favorable = o_mfe if pending_oracle['direction'] == 'LONG' else o_mae
                            oracle_potential = oracle_favorable * self.asset.point_value
                            capture = outcome.pnl / oracle_potential if oracle_potential > 0 else 0.0
                            _tp_potential = pending_oracle.get('tp_ticks', 0) * self.asset.tick_value
                            _target_capture = outcome.pnl / _tp_potential if _tp_potential > 0 else 0.0
                            oracle_trade_records.append({
                                **pending_oracle,
                                'exit_price': outcome.exit_price,
                                'exit_time': ts_raw,
                                'hold_bars': max(1, _exit_action.bars_held),
                                'exit_reason': _ee_reason,
                                'actual_pnl': outcome.pnl,
                                'oracle_potential_pnl': oracle_potential,
                                'capture_rate': round(min(capture, 9.99), 4),
                                'target_capture': round(min(_target_capture, 9.99), 4),
                                'result': outcome.result,
                                'exit_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                                'exit_conviction': _exit_sig.get('conviction', 0.0),
                                'exit_wave_maturity': _exit_sig.get('wave_maturity', 0.0),
                                'exit_signal_reason': getattr(_exit_result, 'reason', _ee_reason),
                                'exit_decay_score': _exit_sig.get('decay_score', 0.0),
                                'trade_mfe_ticks': round(_trade_mfe_ticks, 2),
                                'price_expected_error': round(
                                    (outcome.exit_price - pending_oracle.get('price_expected', pending_oracle['entry_price']))
                                    / self.asset.tick_size, 2),
                            })
                            _stream_trade(oracle_trade_records[-1])

                            # ── Trade replay capture ──
                            # Save per-bar price path + MarketState snapshots for I-MR analysis.
                            # 10 bars before entry to 20 bars after exit (15s execution bars).
                            # Includes discovery TF state (z, sigma, DMI, Hurst, forces) per bar.
                            try:
                                _entry_bar_idx = _bar_i - max(1, _exit_action.bars_held)
                                _replay_start = max(0, _entry_bar_idx - 10)
                                _replay_end = min(len(df_15s), _bar_i + 20)
                                _replay_bars = df_15s.iloc[_replay_start:_replay_end]
                                if len(_replay_bars) > 5:
                                    # Capture MarketState at each bar (from states_map)
                                    _state_snaps = []
                                    for _ri in range(_replay_start, _replay_end):
                                        _rs = _states_map.get(_ri)
                                        if _rs is not None:
                                            _state_snaps.append({
                                                'bar_i': _ri - _replay_start,
                                                'z': round(getattr(_rs, 'z_score', 0.0), 4),
                                                'sigma': round(getattr(_rs, 'regression_sigma', 0.0), 4),
                                                'dmi_p': round(getattr(_rs, 'dmi_plus', 0.0), 2),
                                                'dmi_m': round(getattr(_rs, 'dmi_minus', 0.0), 2),
                                                'adx': round(getattr(_rs, 'adx_strength', 0.0), 2),
                                                'hurst': round(getattr(_rs, 'hurst_exponent', 0.0), 4),
                                                'f_mom': round(getattr(_rs, 'F_momentum', 0.0), 4),
                                                'f_rev': round(getattr(_rs, 'mean_reversion_force', 0.0), 4),
                                                'vel': round(getattr(_rs, 'velocity', 0.0), 4),
                                                'P_center': round(getattr(_rs, 'P_at_center', 0.0), 4),
                                                'entropy': round(getattr(_rs, 'entropy_normalized', 0.0), 4),
                                                'tunnel': round(getattr(_rs, 'reversion_probability', 0.0), 4),
                                                'coherence': round(getattr(_rs, 'oscillation_entropy_normalized', 0.0), 4),
                                            })
                                    _replay_rec = {
                                        'trade_id': len(oracle_trade_records) - 1,
                                        'template_id': active_template_id,
                                        'side': active_side,
                                        'entry_bar': _entry_bar_idx - _replay_start,
                                        'exit_bar': _bar_i - _replay_start,
                                        'entry_price': active_entry_price,
                                        'exit_price': outcome.exit_price,
                                        'actual_pnl': outcome.pnl,
                                        'exit_reason': _ee_reason,
                                        'trade_mfe_ticks': round(_trade_mfe_ticks, 2),
                                        'hold_bars': max(1, _exit_action.bars_held),
                                        'discovery_tf': lib_entry.get('discovery_tf_seconds', 15.0) if lib_entry else 15.0,
                                        'bars': _replay_bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist() if 'volume' in _replay_bars.columns else _replay_bars[['timestamp', 'open', 'high', 'low', 'close']].values.tolist(),
                                        'states': _state_snaps,
                                    }
                                    _trade_replays.append(_replay_rec)
                            except Exception:
                                pass  # don't crash forward pass for replay capture

                            if _pending_dm_idx is not None:
                                decision_matrix_records[_pending_dm_idx].update({
                                    'trade_result': outcome.result,
                                    'trade_pnl': round(outcome.pnl, 2),
                                    'exit_reason': _ee_reason,
                                    'exit_signal_reason': getattr(_exit_result, 'reason', _ee_reason),
                                    'exit_conviction': _exit_sig.get('conviction', 0.0),
                                    'exit_wave_maturity': _exit_sig.get('wave_maturity', 0.0),
                                })
                            belief_network.stop_trade_tracking()
                            pending_oracle = None
                            _pending_dm_idx = None

                        # Self-tune envelope halflife from trade outcome
                        _actual_ticks = outcome.pnl / self.asset.tick_value
                        _cap = oracle_trade_records[-1]['capture_rate'] if oracle_trade_records else 0.0
                        _exec_engine.exit_engine.record_trade_outcome(
                            _trade_mfe_ticks, _actual_ticks, _cap)

                        if isinstance(active_template_id, str) and active_template_id.startswith('PP_'):
                            _pp_all_trades.append(outcome)

                        # Ping-pong: flip after exit (skip during maintenance)
                        if _pp_enabled and not _in_maintenance:
                            _should_flip, _flip_side = _pp_try_flip(
                                active_side, active_template_id, price, ts_raw,
                                _pp_last_exit_params)
                            if _should_flip and _pp_last_exit_params:
                                ep = _pp_last_exit_params
                                _pp_tid = f'PP_{active_template_id}'
                                self._position = _exit_eng.open_position(
                                    side=_flip_side, entry_price=price,
                                    entry_bar_index=_bar_i, template_id=_pp_tid,
                                    sl_ticks=_pp_sl_ov or ep['sl'],
                                    tp_ticks=_pp_tp_ov or ep['tp'],
                                    trail_ticks=_pp_trail_ov or ep['trail'],
                                    trail_activation_ticks=ep['trail_act'])
                                _exec_engine.position_opened(
                                    side=_flip_side, price=price,
                                    bar_index=_bar_i, template_id=_pp_tid,
                                    lib_entry={'p25_mae_ticks': 0, 'mean_mae_ticks': 0,
                                               'regression_sigma_ticks': 0, 'p75_mfe_ticks': 0},
                                    sl_ticks=float(_pp_sl_ov or ep['sl']),
                                    tp_ticks=float(_pp_tp_ov or ep['tp']),
                                    max_hold_bars=ep['max_hold'],
                                )
                                current_position_open = True
                                active_entry_price = price
                                active_entry_time = ts_raw
                                active_side = _flip_side
                                active_template_id = _pp_tid
                                active_max_hold_bars = ep['max_hold']
                                _pp_flip_count += 1
                                belief_network.start_trade_tracking(
                                    side=_flip_side, entry_bar=_bar_i,
                                    pattern_horizon_bars=active_max_hold_bars)
                                pending_oracle = None

                # 2. Check for entries (if no position)
                # Equity ruin check: simulation ends when equity hits 0 (no money to trade).
                if _equity_enabled and account_ruined:
                    break   # stop processing this day's bars entirely

                # ── Candidate building: two paths ──
                # IS: full discovery (PatternEvents from scan_day_cascade)
                # OOS: compressed per-bar (15s state) + bonus multi-TF candidates
                _has_discovery_signal = ts in pattern_map
                _has_compressed_signal = False
                if oos_mode:
                    # Primary: check 15s execution state (same as original)
                    _oos_state = _states_map.get(_bar_i - 1)
                    if _oos_state:
                        _oos_pt = getattr(_oos_state, 'pattern_type', '')
                        _oos_cascade = getattr(_oos_state, 'cascade_detected', False)
                        _oos_struct = getattr(_oos_state, 'structure_confirmed', False)
                        if _oos_pt and _oos_pt != 'NONE' and (_oos_cascade or _oos_struct):
                            _has_compressed_signal = True

                # Detection funnel: count bars where signals were found
                if ts in pattern_map:
                    bars_with_detection += 1
                    if current_position_open:
                        bars_slot_blocked += 1
                elif oos_mode and _has_compressed_signal:
                    bars_with_detection += 1
                    if current_position_open:
                        bars_slot_blocked += 1

                _should_check_entry = (not current_position_open and not _in_maintenance
                                       and (_has_discovery_signal if not oos_mode
                                            else _has_compressed_signal))

                if _should_check_entry:
                    _candidate_gate = {}    # id(p) -> gate label (for FN audit)

                    if oos_mode:
                        # ── OOS: 15s primary candidate + bonus multi-TF ──
                        _oos_state = _states_map.get(_bar_i - 1)
                        _oos_z = getattr(_oos_state, 'z_score', 0.0)
                        _oos_pt = getattr(_oos_state, 'pattern_type', '')
                        _oos_tf_s = 15  # 15s anchor
                        _oos_depth = 8  # 15s depth
                        _oos_feat_list = extract_feature_vector(
                            z_score=_oos_z,
                            velocity=getattr(_oos_state, 'velocity', 0.0),
                            momentum=getattr(_oos_state, 'momentum_strength',
                                             getattr(_oos_state, 'momentum', 0.0)),
                            entropy_normalized=getattr(_oos_state, 'entropy_normalized', 0.0),
                            tf_seconds=_oos_tf_s,
                            depth=float(_oos_depth),
                            parent_is_band_reversal=0.0,
                            adx=getattr(_oos_state, 'adx_strength', 0.0) / 100.0,
                            hurst=getattr(_oos_state, 'hurst_exponent', 0.5),
                            dmi_diff=(getattr(_oos_state, 'dmi_plus', 0.0)
                                      - getattr(_oos_state, 'dmi_minus', 0.0)) / 100.0,
                            parent_z=0.0, parent_dmi_diff=0.0,
                            root_is_roche=0.0, tf_alignment=0.0,
                            pid=getattr(_oos_state, 'term_pid', 0.0),
                            osc_coherence=getattr(_oos_state, 'oscillation_entropy_normalized', 0.0),
                        )
                        # 22D mode: append 6D lookback geometry from 15s closes
                        if getattr(self, '_use_lookback', False):
                            _lb_start = max(0, _bar_i - 10)
                            if _bar_i >= 3 and 'close' in df_15s.columns:
                                from core.shape_primitives import extract_lookback_geometry
                                _lb_closes = df_15s['close'].iloc[_lb_start:_bar_i].values
                                _oos_feat_list.extend(extract_lookback_geometry(_lb_closes).tolist())
                            else:
                                _oos_feat_list.extend([0.0] * 6)
                        _oos_feat = np.array([_oos_feat_list])
                        _eng_candidates = [Candidate(
                            state=_oos_state,
                            depth=_oos_depth,
                            timeframe='15s',
                            timestamp=ts,
                            pattern_type=_oos_pt,
                            z_score=_oos_z,
                            features=_oos_feat,
                        )]
                        # Bonus: add candidates from other TF workers with signals
                        _OOS_TF_DEPTH = {
                            3600: 1, 1800: 2, 900: 3, 300: 4, 180: 5,
                            60: 6, 30: 7, 5: 9, 1: 10,  # skip 15=8 (already added)
                        }
                        _TF_LABEL = {
                            3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
                            180:'3m', 60:'1m', 30:'30s', 5:'5s', 1:'1s',
                        }
                        for _tf_sec_c in sorted(belief_network.workers.keys(),
                                                reverse=True):
                            if _tf_sec_c == 15:  # skip 15s (already primary)
                                continue
                            _w_c = belief_network.workers[_tf_sec_c]
                            if _w_c._last_tf_bar_idx < 0 or not _w_c._states:
                                continue
                            _widx_c = min(_w_c._last_tf_bar_idx,
                                          len(_w_c._states) - 1)
                            _wraw_c = _w_c._states[_widx_c]
                            _wst_c = (_wraw_c['state']
                                      if isinstance(_wraw_c, dict)
                                      and 'state' in _wraw_c else _wraw_c)
                            _wpt_c = getattr(_wst_c, 'pattern_type', '')
                            if (not _wpt_c or _wpt_c == 'NONE'):
                                continue
                            _wdepth = _OOS_TF_DEPTH.get(_tf_sec_c, 8)
                            _wz = getattr(_wst_c, 'z_score', 0.0)
                            _wfeat_list = extract_feature_vector(
                                z_score=_wz,
                                velocity=getattr(_wst_c, 'velocity', 0.0),
                                momentum=getattr(_wst_c, 'momentum_strength',
                                    getattr(_wst_c, 'momentum', 0.0)),
                                entropy_normalized=getattr(
                                    _wst_c, 'entropy_normalized', 0.0),
                                tf_seconds=_tf_sec_c,
                                depth=float(_wdepth),
                                parent_is_band_reversal=0.0,
                                adx=getattr(_wst_c, 'adx_strength',
                                            0.0) / 100.0,
                                hurst=getattr(_wst_c, 'hurst_exponent',
                                              0.5),
                                dmi_diff=(getattr(_wst_c, 'dmi_plus', 0.0)
                                          - getattr(_wst_c, 'dmi_minus',
                                                    0.0)) / 100.0,
                                parent_z=0.0, parent_dmi_diff=0.0,
                                root_is_roche=0.0, tf_alignment=0.0,
                                pid=getattr(_wst_c, 'term_pid', 0.0),
                                osc_coherence=getattr(
                                    _wst_c,
                                    'oscillation_entropy_normalized', 0.0),
                            )
                            # 22D: reuse 15s lookback geometry for bonus TF candidates
                            if getattr(self, '_use_lookback', False):
                                _lb_start_b = max(0, _bar_i - 10)
                                if _bar_i >= 3 and 'close' in df_15s.columns:
                                    from core.shape_primitives import extract_lookback_geometry
                                    _lb_c = df_15s['close'].iloc[_lb_start_b:_bar_i].values
                                    _wfeat_list.extend(extract_lookback_geometry(_lb_c).tolist())
                                else:
                                    _wfeat_list.extend([0.0] * 6)
                            _wfeat = np.array([_wfeat_list])
                            _eng_candidates.append(Candidate(
                                state=_wst_c,
                                depth=_wdepth,
                                timeframe=_TF_LABEL.get(_tf_sec_c,
                                                        f'{_tf_sec_c}s'),
                                timestamp=ts,
                                pattern_type=_wpt_c,
                                z_score=_wz,
                                features=_wfeat,
                            ))
                        raw_candidates = []  # no PatternEvents in compressed mode
                    else:
                        # ── IS: full discovery (PatternEvents) ──
                        raw_candidates = pattern_map[ts]
                        _eng_candidates = [
                            Candidate(
                                state=p.state,
                                depth=getattr(p, 'depth', 6),
                                timeframe=getattr(p, 'timeframe', '15s'),
                                timestamp=ts,
                                pattern_type=p.pattern_type,
                                z_score=p.z_score,
                                features=np.array([_feat_extractor.extract_features(p)]),
                                raw_event=p,
                            ) for p in raw_candidates
                        ]
                    n_signals_seen += len(_eng_candidates)

                    # -- PP direction override --
                    _pp_dir_ov = None
                    if _pp_enabled:
                        for _ec in _eng_candidates:
                            _raw = _ec.raw_event
                            if _raw is None:
                                continue
                            _pp_base = None
                            _ec_feat = _ec.features
                            if _ec_feat is not None:
                                _pp_2d = _ec_feat.reshape(1, -1) if _ec_feat.ndim == 1 else _ec_feat
                                _pp_exp = getattr(self.scaler, 'n_features_in_', _pp_2d.shape[-1])
                                if _pp_2d.shape[-1] < _pp_exp:
                                    _pp_2d = np.concatenate([_pp_2d, np.zeros((_pp_2d.shape[0], _pp_exp - _pp_2d.shape[-1]))], axis=-1)
                                _ec_fs = self.scaler.transform(_pp_2d)
                                _ec_ds = np.linalg.norm(centroids_scaled - _ec_fs, axis=1)
                                _ec_ni = int(np.argmin(_ec_ds))
                                if _ec_ds[_ec_ni] < 4.5:
                                    _pp_base = valid_template_ids[_ec_ni]
                            if _pp_base is not None:
                                _pp_b = self.brain.get_dir_bias(_pp_base)
                                if _pp_b:
                                    _lw, _ll = _pp_b.get('long_w', 0), _pp_b.get('long_l', 0)
                                    _sw, _sl = _pp_b.get('short_w', 0), _pp_b.get('short_l', 0)
                                    _lt, _st = _lw + _ll, _sw + _sl
                                    if _lt >= 5 and _st >= 5:
                                        _lwr = _lw / _lt
                                        _swr = _sw / _st
                                        if _lwr > 0.60 and _swr < 0.40:
                                            _pp_dir_ov = 'long'
                                        elif _swr > 0.60 and _lwr < 0.40:
                                            _pp_dir_ov = 'short'
                                if _pp_dir_ov is not None:
                                    break

                    # -- Call ExecutionEngine --
                    _entry_action = _exec_engine.on_bar(
                        price=price,
                        bar_high=getattr(row, 'high', price),
                        bar_low=getattr(row, 'low', price),
                        bar_index=_bar_i,
                        candidates=_eng_candidates,
                        pp_dir_override=_pp_dir_ov,
                    )
                    # Copy gate labels for FN audit
                    _candidate_gate = _entry_action.candidate_gates

                    # Yellow marker for evaluated-but-rejected signals
                    if (_entry_action.type != ActionType.ENTER
                            and self.dashboard_queue is not None
                            and len(_eng_candidates) > 0
                            and _bar_i % 4 == 0):  # throttle: every 4th bar max
                        self.dashboard_queue.put({
                            'type': 'TRADE_MARKER', 'action': 'SKIP',
                            'side': '', 'price': price, 'pnl': 0})

                    if _entry_action.type == ActionType.ENTER:
                        best_candidate = _entry_action.raw_event
                        best_tid = _entry_action.template_id
                        lib_entry = _entry_action.lib_entry
                        side = _entry_action.side
                        _belief = _entry_action.belief_state
                        _band = _entry_action.band_context
                        _network_tp = _entry_action.network_tp
                        _sl_ticks = int(_entry_action.sl_ticks)
                        _tp_ticks = int(_entry_action.tp_ticks)
                        _trail_ticks = int(_entry_action.trail_ticks)
                        _trail_act_ticks = int(_entry_action.trail_activation_ticks or 0)
                        _cand_depth = _entry_action.depth
                        _parent_tf = _entry_action.parent_tf
                        active_max_hold_bars = _entry_action.max_hold_bars
                        long_bias = _entry_action.long_bias
                        short_bias = _entry_action.short_bias

                        # Equity risk gate — scales with intraday drawdown aggression
                        # Full equity: risk up to 50% per trade
                        # 20% DD: risk up to 25%, 40%+ DD: risk up to 10%
                        _MAX_RISK_FRACTION = 0.50 * _dd_aggression
                        _skip_equity = False
                        if _equity_enabled:
                            _max_loss_usd = _sl_ticks * self.asset.tick_size * self.asset.point_value
                            _max_risk_usd = running_equity * _MAX_RISK_FRACTION
                            if _max_loss_usd > _max_risk_usd:
                                skipped_ruin += 1
                                _skip_equity = True

                        if not _skip_equity:
                            self._position = _exit_eng.open_position(
                                side=side, entry_price=price,
                                entry_bar_index=_bar_i, template_id=best_tid,
                                sl_ticks=_sl_ticks, tp_ticks=_tp_ticks,
                                trail_ticks=_trail_ticks,
                                trail_activation_ticks=_trail_act_ticks,
                                lib_entry=lib_entry,
                            )
                            # Trade marker on dashboard
                            if self.dashboard_queue is not None:
                                self.dashboard_queue.put({
                                    'type': 'TRADE_MARKER', 'action': 'ENTRY',
                                    'side': side, 'price': price, 'pnl': 0})
                            # Notify engine
                            _exec_engine.position_opened(
                                side=side, price=price, bar_index=_bar_i,
                                template_id=best_tid, lib_entry=lib_entry,
                                sl_ticks=_sl_ticks, tp_ticks=_tp_ticks,
                                network_tp=float(_network_tp) if _network_tp else None,
                                max_hold_bars=active_max_hold_bars,
                            )

                            current_position_open = True
                            active_entry_price = price
                            active_entry_time = ts
                            active_side = side
                            active_template_id = best_tid

                            # Store TF-scaled exit params for ping-pong
                            _pp_last_exit_params = {
                                'sl': _sl_ticks, 'tp': _tp_ticks,
                                'trail': _trail_ticks, 'trail_act': _trail_act_ticks,
                                'max_hold': active_max_hold_bars,
                                'tf': _parent_tf, 'depth': _cand_depth,
                            }

                            # Start physics decay tracking
                            _avg_mfe_bar = lib_entry.get('avg_mfe_bar', 0.0)
                            _p75_mfe_bar = lib_entry.get('p75_mfe_bar', 0.0)
                            _p75_mfe_ticks = lib_entry.get('p75_mfe_ticks', 0.0)
                            belief_network.start_trade_tracking(
                                side=side, entry_bar=_bar_i,
                                pattern_horizon_bars=active_max_hold_bars,
                                target_mfe_ticks=_p75_mfe_ticks,
                                resolve_bars=_avg_mfe_bar,
                                entry_price=price,
                            )
                            # Per-template exit timescale
                            if _avg_mfe_bar > 0:
                                belief_network.set_active_trade_timescale(_avg_mfe_bar, _p75_mfe_bar)
                            depth_traded[_cand_depth] += 1

                            # -- Oracle record assembly --
                            _live_state = best_candidate.state if best_candidate else None
                            _dmi_at_entry = round(
                                getattr(_live_state, 'dmi_plus', 0.0)
                                - getattr(_live_state, 'dmi_minus', 0.0), 2) if _live_state else 0.0
                            _nn_marker = _effective_oracle(best_candidate) if best_candidate else 0
                            _playbook = lib_entry.get('semantic_name', '') or ''
                            if (not _playbook or _playbook == 'Unknown') and lib_entry.get('centroid') is not None:
                                from core.fractal_clustering import generate_semantic_name
                                _playbook = generate_semantic_name(lib_entry['centroid'])

                            pending_oracle = {
                                'template_id':      best_tid,
                                'playbook':         _playbook,
                                'direction':        'LONG' if side == 'long' else 'SHORT',
                                'dir_source':       _entry_action.dir_source,
                                'entry_price':      price,
                                'entry_time':       ts,
                                'entry_depth':      _cand_depth,
                                'root_tf':          _parent_tf,
                                'max_hold_bars':    active_max_hold_bars,
                                'oracle_label':     _nn_marker,
                                'oracle_label_name': _ORACLE_LABEL_NAMES.get(
                                    _nn_marker, 'WORKER_BYPASS' if _entry_action.is_bypass else 'UNKNOWN'),
                                'oracle_mfe':       getattr(best_candidate, 'oracle_meta', {}).get('mfe', 0.0) if best_candidate else 0.0,
                                'oracle_mae':       getattr(best_candidate, 'oracle_meta', {}).get('mae', 0.0) if best_candidate else 0.0,
                                'long_bias':        round(long_bias, 4),
                                'short_bias':       round(short_bias, 4),
                                'dmi_diff':         _dmi_at_entry,
                                'belief_active_levels': _belief.active_levels if _belief is not None else 0,
                                'belief_conviction':    round(_belief.conviction, 4) if _belief is not None else 0.0,
                                'wave_maturity':        round(_belief.wave_maturity, 4) if _belief is not None else 0.0,
                                'decision_wave_maturity': round(_belief.decision_wave_maturity, 4) if _belief is not None else 0.0,
                                'entry_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                                'band_direction': _band['direction'] if _band else None,
                                'band_strength': round(_band['strength'], 3) if _band else 0.0,
                                'band_summary': _band.get('band_summary', '') if _band else '',
                                # ── Macro trend observation (non-actionable) ──
                                **_macro_obs(belief_network, side),
                                'tp_ticks':     _tp_ticks,
                                'sl_ticks':     _sl_ticks,
                                'target_price': round(price + (_tp_ticks if side == 'long' else -_tp_ticks) * self.asset.tick_size, 6),
                                'stop_price':   round(price - (_sl_ticks if side == 'long' else -_sl_ticks) * self.asset.tick_size, 6),
                                'expected_pnl': self.brain.get_expected_pnl(best_tid, side),
                                'anchor_mfe_ticks': round(self._position.anchor_mfe_ticks, 1) if self._position else 0.0,
                                'anchor_mfe_bars': round(self._position.anchor_mfe_bars, 1) if self._position else 0.0,
                                'predicted_mfe_ticks': round(_belief.predicted_mfe, 2) if _belief is not None else 0.0,
                                'price_expected': round(
                                    price + ((_belief.predicted_mfe if side == 'long' else -_belief.predicted_mfe)
                                             * self.asset.tick_size), 6) if _belief is not None and _belief.predicted_mfe > 0 else price,
                                **_physics_fields(best_candidate),
                                # ── Quantum score (observation only) ──
                                **_quantum_score(
                                    _live_state, _belief, side,
                                    template_wr=lib_entry.get('win_rate', 0.5),
                                    norm_dist=_entry_action.dist if hasattr(_entry_action, 'dist') else 1.0),
                            }

                            # Signal log: traded record
                            _bc_mz = round(abs(best_candidate.z_score), 2) if best_candidate else 0.0
                            _bc_mac = round(abs((getattr(best_candidate, 'parent_chain', None) or [{}])[-1].get('z', 0.0)), 2) if best_candidate else 0.0
                            _dm_entry = _dm_rec(
                                best_candidate, 'traded', day_date, ts,
                                _bc_mz, _bc_mac,
                                getattr(best_candidate, 'pattern_type', ''),
                                dist=_entry_action.dist,
                                conviction=round(_entry_action.conviction, 3),
                                template_id=best_tid,
                                tier=template_tier_map.get(best_tid, 3),
                                playbook=_playbook)
                            _dm_entry['trade_direction'] = 'LONG' if side == 'long' else 'SHORT'
                            _dm_entry.update(_macro_obs(belief_network, side))
                            decision_matrix_records.append(_dm_entry)
                            _pending_dm_idx = len(decision_matrix_records) - 1

                            # AUDIT: True Positive or False Positive
                            audit_outcome = TradeOutcome(
                                state=best_candidate.state if best_candidate else None,
                                entry_price=price, exit_price=0.0, pnl=0.0,
                                result='PENDING', timestamp=ts,
                                exit_reason='PENDING',
                                direction='LONG' if side == 'long' else 'SHORT'
                            )
                            audit_res = _audit_trade(audit_outcome, best_candidate)
                            cls = audit_res['classification']
                            if cls == 'TP': audit_tp += 1
                            elif cls == 'FP_NOISE': audit_fp_noise += 1
                            elif cls == 'FP_WRONG': audit_fp_wrong += 1

                            # Audit other candidates as SKIPPED
                            for p in raw_candidates:
                                if p == best_candidate:
                                    continue
                                audit_res = _audit_trade(None, p)
                                if audit_res['classification'] == 'TN':
                                    audit_tn += 1
                                elif audit_res['classification'] == 'FN':
                                    _s = p.state
                                    _is_pid = (abs(_s.term_pid) >= 0.3
                                               and _s.oscillation_entropy_normalized >= 0.5
                                               and _s.adx_strength <= 30.0)
                                    if _is_pid:
                                        continue
                                    _om = _effective_oracle(p)
                                    _meta = getattr(p, 'oracle_meta', {})
                                    _fn_pot = (_meta.get('mfe', 0.0) if _om > 0 else _meta.get('mae', 0.0)) * self.asset.point_value
                                    # Gate passers that lost on score are not FN
                                    if _candidate_gate.get(id(p)) == 'score_loser':
                                        score_loser_pnl += _fn_pot
                                        continue
                                    audit_fn += 1
                                    fn_potential_pnl += _fn_pot
                                    fn_oracle_records.append({
                                        'timestamp':       ts,
                                        'depth':           getattr(p, 'depth', 6),
                                        'oracle_label':    _om,
                                        'oracle_label_name': _ORACLE_LABEL_NAMES.get(_om, '?'),
                                        'oracle_dir':      'LONG' if _om > 0 else 'SHORT',
                                        'fn_potential_pnl': round(_fn_pot, 2),
                                        'reason':          'competed',
                                        'gate_blocked':    _candidate_gate.get(id(p), 'unknown'),
                                        'workers':         __import__('json').dumps(belief_network.get_worker_snapshot()),
                                        **_physics_fields(p),
                                    })
                    else:
                        # No entry -- audit all candidates as SKIPPED
                        for p in raw_candidates:
                            audit_res = _audit_trade(None, p)
                            if audit_res['classification'] == 'TN':
                                audit_tn += 1
                            elif audit_res['classification'] == 'FN':
                                audit_fn += 1
                                _om = _effective_oracle(p)
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
                                    'reason':          'no_match',
                                    'gate_blocked':    _candidate_gate.get(id(p), 'unknown'),
                                    'workers':         __import__('json').dumps(belief_network.get_worker_snapshot()),
                                    **_physics_fields(p),
                                })

            # End of day cleanup -- force close any open position
            if self._position is not None:
                pos = self._position
                # Get final exit signal for logging
                _eod_sig = belief_network.get_exit_signal(pos.side)
                _eod_adj_reason = ''

                if pos.side == 'short':
                    eod_pnl = (pos.entry_price - price) * self.asset.point_value
                    _trade_mfe_ticks = (pos.entry_price - pos.peak_favorable) / self.asset.tick_size
                else:
                    eod_pnl = (price - pos.entry_price) * self.asset.point_value
                    _trade_mfe_ticks = (pos.peak_favorable - pos.entry_price) / self.asset.tick_size
                if _slip_rng:
                    eod_pnl += _slip_rng.uniform(-_slip_ticks, _slip_ticks) * _tick_val
                self._position = None
                _exec_engine.position_closed()  # reset engine position state

                outcome = record_trade(
                    self.brain, tid=active_template_id,
                    entry_price=active_entry_price,
                    exit_price=price,
                    pnl=eod_pnl, side=active_side,
                    exit_reason='TIME_EXIT', timestamp=ts,
                    entry_time=active_entry_time, exit_time=ts,
                    tick_value=self.asset.tick_value,
                    hold_bars=pos.bars_held,
                )
                day_trades.append(outcome)
                _cal_day_trades.append(outcome)

                # Track intraday dip (min equity calc)
                _day_running_pnl += outcome.pnl
                _day_min_pnl = min(_day_min_pnl, _day_running_pnl)

                # Track cumulative equity curve (never resets)
                _cumul_pnl += outcome.pnl
                if _cumul_pnl > _cumul_peak:
                    _cumul_peak = _cumul_pnl
                if _cumul_pnl < _cumul_trough:
                    _cumul_trough = _cumul_pnl
                    _cumul_trough_date = _prev_cal_date
                _dd_from_peak = _cumul_peak - _cumul_pnl
                if _dd_from_peak > _cumul_max_dd:
                    _cumul_max_dd = _dd_from_peak
                    _cumul_dd_trough_date = _prev_cal_date

                # Update running equity after EOD close
                if _equity_enabled:
                    running_equity += outcome.pnl
                    peak_equity   = max(peak_equity, running_equity)
                    trough_equity = min(trough_equity, running_equity)
                    _daily_peak_equity = max(_daily_peak_equity, running_equity)
                    _dd_aggression = min(1.0, max(0.2, running_equity / account_size)) if account_size > 0 else 1.0
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
                    _tp_potential = pending_oracle.get('tp_ticks', 0) * self.asset.tick_value
                    _target_capture = outcome.pnl / _tp_potential if _tp_potential > 0 else 0.0
                    oracle_trade_records.append({
                        **pending_oracle,
                        'exit_price':  outcome.exit_price,
                        'exit_time':   _eod_exit_t,
                        'hold_bars':   max(1, int((_eod_exit_t - _eod_entry_t) / 15)),
                        'exit_reason': 'TIME_EXIT',
                        'actual_pnl':  outcome.pnl,
                        'oracle_potential_pnl': oracle_potential,
                        'capture_rate': round(min(capture, 9.99), 4),
                        'target_capture': round(min(_target_capture, 9.99), 4),
                        'result': outcome.result,
                        'exit_workers': __import__('json').dumps(belief_network.get_worker_snapshot()),
                        'exit_conviction':    _eod_sig.get('conviction', 0.0),
                        'exit_wave_maturity': _eod_sig.get('wave_maturity', 0.0),
                        'exit_signal_reason': (_eod_adj_reason or _eod_sig.get('reason', '')),
                        'exit_decay_score':   _eod_sig.get('decay_score', 0.0),
                        'trade_mfe_ticks':    round(_trade_mfe_ticks, 2),
                        'price_expected_error': round(
                            (outcome.exit_price - pending_oracle.get('price_expected', pending_oracle['entry_price']))
                            / self.asset.tick_size, 2),
                    })
                    _stream_trade(oracle_trade_records[-1])
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
                    belief_network.stop_trade_tracking()
                    pending_oracle = None
                    _pending_dm_idx = None

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
            else:
                d_pnl = 0.0

            # NOTE: daily ledger + dip tracking now handled at calendar-day
            # boundaries inside the inner loop (see _row_day flush above)

            # Send end-of-month update with per-month breakdown (for monthly bar chart)
            if self.dashboard_queue:
                _wr_end = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
                _gw2 = sum(t['actual_pnl'] for t in oracle_trade_records if t['actual_pnl'] > 0)
                _gl2 = abs(sum(t['actual_pnl'] for t in oracle_trade_records if t['actual_pnl'] < 0))
                _pf2 = _gw2 / _gl2 if _gl2 > 0 else 0.0
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                          'step': f'FORWARD_PASS  day {_cumulative_days}/{_total_trading_days}',
                                          'pct': round(_cumulative_days / _total_trading_days * 100, 1),
                                          'pnl': total_pnl,
                                          'trades': total_trades,
                                          'wr': round(_wr_end, 1),
                                          'pf': round(_pf2, 2),
                                          'gross_w': round(_gw2, 0),
                                          'gross_l': round(_gl2, 0),
                                          'month_pnl': d_pnl,
                                          'month_label': day_date})

        # ── Flush last calendar day to ledger ────────────────────────────
        if _prev_cal_date and _cal_day_trades:
            _cd_pnl = sum(t.pnl for t in _cal_day_trades)
            _cd_wins = sum(1 for t in _cal_day_trades if t.result == 'WIN')
            _cd_losing = _cd_pnl < 0
            if _cd_losing:
                _consec_losing_days += 1
            else:
                _consec_losing_days = 0
            _max_consec_losing_days = max(_max_consec_losing_days, _consec_losing_days)
            _all_day_dips.append((_prev_cal_date, _day_min_pnl))
            if _day_min_pnl < _worst_intraday_dip:
                _worst_intraday_dip = _day_min_pnl
                _worst_dip_date = _prev_cal_date
            _daily_ledger.append({
                'date': _prev_cal_date,
                'trades': len(_cal_day_trades),
                'wins': _cd_wins,
                'pnl': _cd_pnl,
                'min_dip': _day_min_pnl,
                'cumul_pnl': _cumul_pnl,
                'cumul_dd': _cumul_peak - _cumul_pnl,
                'equity': running_equity if _equity_enabled else total_pnl,
                'consec_loss_days': _consec_losing_days,
                'aggression': _dd_aggression,
            })

        _pbar.close()

        _dir_corrections = self._learn_oracle_directions(
            oracle_trade_records, oos_mode,
            brain_keys_before_oos=_brain_keys_before_oos if oos_mode else None,
            brain_dir_keys_before_oos=_brain_dir_keys_before_oos if oos_mode else None,
        )

        # Close streaming trade log before report generation
        if _stream_log_file is not None:
            _stream_log_file.close()
            _stream_log_file = None

        self._write_forward_pass_reports(
            total_trades=total_trades, total_wins=total_wins, total_pnl=total_pnl,
            oracle_trade_records=oracle_trade_records,
            fn_oracle_records=fn_oracle_records,
            decision_matrix_records=decision_matrix_records,
            pid_oracle_records=pid_oracle_records,
            daily_files_15s=daily_files_15s, start_date=start_date, end_date=end_date,
            _worker_total_states=_worker_total_states,
            _worker_days_with_data=_worker_days_with_data,
            _equity_enabled=_equity_enabled, account_size=account_size,
            running_equity=running_equity, peak_equity=peak_equity,
            trough_equity=trough_equity, skipped_ruin=skipped_ruin,
            _dd_aggression=_dd_aggression, account_ruined=account_ruined, ruin_day=ruin_day,
            _daily_ledger=_daily_ledger, _max_consec_losing_days=_max_consec_losing_days,
            _all_day_dips=_all_day_dips, _worst_intraday_dip=_worst_intraday_dip,
            _worst_dip_date=_worst_dip_date,
            _cumul_pnl=_cumul_pnl, _cumul_peak=_cumul_peak, _cumul_trough=_cumul_trough,
            _cumul_trough_date=_cumul_trough_date, _cumul_max_dd=_cumul_max_dd,
            _cumul_dd_trough_date=_cumul_dd_trough_date,
            audit_tp=audit_tp, audit_fp_wrong=audit_fp_wrong,
            audit_fp_noise=audit_fp_noise, audit_fn=audit_fn, audit_tn=audit_tn,
            fn_potential_pnl=fn_potential_pnl, score_loser_pnl=score_loser_pnl,
            total_bars_processed=total_bars_processed,
            bars_with_detection=bars_with_detection,
            bars_slot_blocked=bars_slot_blocked, n_signals_seen=n_signals_seen,
            depth_traded=depth_traded,
            _exec_engine=_exec_engine, _dir_corrections=_dir_corrections,
            _out_dir=_out_dir, _reports_out=_reports_out,
            oos_mode=oos_mode, _analysis_mode=_analysis_mode,
            _pp_enabled=_pp_enabled, _pp_flip_count=_pp_flip_count,
            _pp_all_trades=_pp_all_trades,
            _trade_replays=_trade_replays,
            _NINJATRADER_MNQ_MARGIN=_NINJATRADER_MNQ_MARGIN,
        )

        # Persist exit engine for multi-iteration runs
        self._persisted_exit_engine = _exec_engine.exit_engine
        self.exec_engine = _exec_engine

        # Save tuned exit params for standalone OOS3
        import json as _json_save
        _tuned_exit = {
            'envelope_half_life_bars': _exec_engine.exit_engine.envelope_half_life_bars,
            'giveback_pct': _exec_engine.exit_engine.giveback_pct,
        }
        _tuned_path = os.path.join(self.checkpoint_dir, 'exit_tuning.json')
        with open(_tuned_path, 'w') as _tf:
            _json_save.dump(_tuned_exit, _tf)
        print(f"  Exit tuning saved: hl={_tuned_exit['envelope_half_life_bars']:.1f}, "
              f"gb={_tuned_exit['giveback_pct']:.0%} → {_tuned_path}")

        # ── OOS3: Replay last N trading days through BarProcessor ──────────
        # Inline OOS ran all files. Now replay the last N days through a
        # BarProcessor with SAME belief_network (preserves 48 days of TBN
        # state) but independent execution engine — no shared mutation.
        if _live_val_days > 0 and _daily_ledger:
            # Identify last N trading days from inline OOS ledger
            _lv_target_dates = [d['date'] for d in _daily_ledger[-_live_val_days:]]
            if _lv_target_dates:
                print(f"\n  ── OOS3 Parity Replay: {len(_lv_target_dates)} days via BarProcessor ──")
                print(f"  Dates: {_lv_target_dates[0]} to {_lv_target_dates[-1]}")

                # Create FRESH EE + exit engine (no shared state with inline OOS)
                _lv_exit_eng = ExitEngine(
                    mode='training',
                    tick_size=self.asset.tick_size,
                    tick_value=self.asset.tick_size * self.asset.point_value,
                    min_hold_bars=getattr(self, '_min_hold_bars', 0),
                )
                # Copy self-tuned exit params from inline OOS (537 trades of tuning)
                _tuned = _exec_engine.exit_engine
                _lv_exit_eng.envelope_half_life_bars = _tuned.envelope_half_life_bars
                _lv_exit_eng.giveback_pct = _tuned.giveback_pct
                # FIX 1: Reuse inline OOS's warmed belief_network (48 days of state)
                # Fresh TBN has no accumulated conviction/momentum → different exits
                _lv_belief = belief_network
                _lv_exec = create_execution_engine(
                    bundle=_bundle,
                    brain=self.brain,
                    belief_network=_lv_belief,
                    exit_engine=_lv_exit_eng,
                    tick_size=self.asset.tick_size,
                    point_value=self.asset.point_value,
                    mode='oos',
                    tier_preference=tier_preference,
                    bias_threshold=bias_threshold if bias_threshold is not None else 0.55,
                    dmi_threshold=dmi_threshold if dmi_threshold is not None else 0.0,
                    depth_only=getattr(self, '_depth_only', None),
                )
                _lv_exec.gate1_dist = 4.5 + 0.5 * 10.0  # match inline OOS

                def _lv_modify_pnl_fresh(pnl_dollars):
                    if _slip_rng:
                        return pnl_dollars + _slip_rng.uniform(-_slip_ticks, _slip_ticks) * _tick_val
                    return pnl_dollars

                # 1s sub-bar wick arrays (set per-file below)
                _lv_1s_ts_arr = None
                _lv_1s_hi_arr = None
                _lv_1s_lo_arr = None
                _lv_has_1s = False
                # Current row timestamp (set per-bar in the loop)
                _lv_cur_ts = [0.0]  # mutable container for closure

                def _lv_pre_exit_eval(price, bar_index):
                    """Provide 1s sub-bar wicks + trade pace (matches inline OOS)."""
                    extra = {}
                    # Sub-bar wicks from 1s data
                    if _lv_has_1s and _lv_1s_ts_arr is not None:
                        _s0 = np.searchsorted(_lv_1s_ts_arr, _lv_cur_ts[0], side='left')
                        _s1 = np.searchsorted(_lv_1s_ts_arr, _lv_cur_ts[0] + 15, side='left')
                        if _s1 > _s0:
                            extra['sub_bar_highs'] = _lv_1s_hi_arr[_s0:_s1].tolist()
                            extra['sub_bar_lows'] = _lv_1s_lo_arr[_s0:_s1].tolist()
                    # Trade pace update (matches inline OOS lines 985-988)
                    _tp = _lv_belief.get_trade_progress(
                        price, tick_size=self.asset.tick_size)
                    _lv_belief._trade_pace_cache = _tp
                    _lv_belief._trade_pace_blend = _tp.get('pace', 1.0) - 1.0
                    return extra

                _lv_processor = BarProcessor(
                    exec_engine=_lv_exec,
                    belief_network=_lv_belief,
                    exit_engine=_lv_exit_eng,
                    brain=self.brain,
                    pattern_library=_bundle.pattern_library,
                    anchor_tf='15s',
                    anchor_depth=8,
                    tick_size=self.asset.tick_size,
                    point_value=self.asset.point_value,
                    hooks=BarProcessorHooks(
                        modify_pnl=_lv_modify_pnl_fresh,
                        pre_exit_eval=_lv_pre_exit_eval,
                    ),
                )

                # Find which files contain the target dates and replay them
                import datetime as _dt_lv
                _target_set = set(_lv_target_dates)

                for _lv_file in daily_files_15s:
                    _lv_df = pd.read_parquet(_lv_file)
                    if _lv_df.empty:
                        continue
                    _ts_col = _lv_df['timestamp']
                    if np.issubdtype(_ts_col.dtype, np.number):
                        _lv_df['_date'] = pd.to_datetime(_ts_col, unit='s').dt.strftime('%Y-%m-%d')
                    else:
                        _lv_df['_date'] = pd.to_datetime(_ts_col).dt.strftime('%Y-%m-%d')
                    _file_dates = set(_lv_df['_date'].unique())
                    if not _file_dates.intersection(_target_set):
                        continue

                    # Compute states for this file
                    try:
                        _lv_states = self.engine.batch_compute_states(_lv_df, use_cuda=True)
                    except Exception:
                        continue

                    # Load sub-TF data for TBN
                    _lv_5s_path = _lv_file.replace('/15s/', '/5s/').replace('\\15s\\', '\\5s\\')
                    _lv_1s_path = _lv_file.replace('/15s/', '/1s/').replace('\\15s\\', '\\1s\\')
                    _lv_4h_path = _lv_file.replace('/15s/', '/4h/').replace('\\15s\\', '\\4h\\')
                    _lv_df_5s = pd.read_parquet(_lv_5s_path) if os.path.exists(_lv_5s_path) else None
                    _lv_df_1s = pd.read_parquet(_lv_1s_path) if os.path.exists(_lv_1s_path) else None
                    _lv_df_4h = pd.read_parquet(_lv_4h_path) if os.path.exists(_lv_4h_path) else None

                    # Pre-extract 1s numpy arrays for sub-bar wick lookup
                    if _lv_df_1s is not None and not _lv_df_1s.empty:
                        _lv_1s_ts_arr = _lv_df_1s['timestamp'].values.astype(np.float64)
                        _lv_1s_hi_arr = _lv_df_1s['high'].values.astype(np.float64)
                        _lv_1s_lo_arr = _lv_df_1s['low'].values.astype(np.float64)
                        _lv_has_1s = True
                    else:
                        _lv_has_1s = False

                    # TBN prepare (same as inline OOS)
                    try:
                        _lv_belief.prepare_day(_lv_df, states_micro=_lv_states,
                                               df_5s=_lv_df_5s, df_1s=_lv_df_1s, df_4h=_lv_df_4h)
                    except Exception:
                        _lv_belief.prepare_day(_lv_df, states_micro=[],
                                               df_5s=_lv_df_5s, df_1s=_lv_df_1s, df_4h=_lv_df_4h)

                    # Map states by bar_idx for this file
                    _lv_states_map = {s['bar_idx']: s['state'] for s in _lv_states}

                    # First tick all bars BEFORE the target dates (TBN warmup)
                    # This matches inline OOS which ticks through all bars sequentially
                    _all_dates_sorted = sorted(_lv_df['_date'].unique())
                    _first_target = min(_target_set.intersection(_file_dates))
                    _lv_bar_counter = 0  # global bar index within file (matches inline _bar_i)
                    _lv_last_warmup_state = None  # track last warmup state for FIX 2
                    for _warmup_date in _all_dates_sorted:
                        if _warmup_date >= _first_target:
                            break
                        _warmup_indices = _lv_df.index[_lv_df['_date'] == _warmup_date].tolist()
                        for _w_idx in _warmup_indices:
                            _lv_belief.tick_all(_lv_bar_counter)
                            _lv_last_warmup_state = _lv_states_map.get(_w_idx)
                            _lv_bar_counter += 1

                    # Process bars day-by-day (only target dates)
                    for _lv_date in sorted(_target_set.intersection(_file_dates)):
                        _day_mask = _lv_df['_date'] == _lv_date
                        _day_indices = _lv_df.index[_day_mask].tolist()
                        if not _day_indices:
                            continue

                        _lv_day_trades = []
                        for _lv_i, _lv_row_idx in enumerate(_day_indices):
                            _lv_state = _lv_states_map.get(_lv_row_idx)
                            if _lv_state is None:
                                _lv_bar_counter += 1
                                continue
                            _lv_row = _lv_df.iloc[_lv_row_idx]
                            _lv_cur_ts[0] = float(_lv_row['timestamp'])
                            # Same state for entry + exit (matches inline OOS:
                            # exit uses _states_map[_bar_i-1] = current bar due to pre-increment)
                            result = _lv_processor.process_bar(
                                bar_index=_lv_bar_counter,
                                price=float(_lv_row['close']),
                                bar_high=float(_lv_row['high']),
                                bar_low=float(_lv_row['low']),
                                timestamp=float(_lv_row['timestamp']),
                                state=_lv_state,
                            )
                            _lv_bar_counter += 1
                            if result.trade_completed:
                                _lv_day_trades.append(result.trade_completed)

                        # Update last warmup state for next day's prior-bar seed
                        if _day_indices:
                            _lv_last_warmup_state = _lv_states_map.get(_day_indices[-1])

                        # Force close at EOD
                        if _lv_processor.in_position:
                            _last_row = _lv_df.iloc[_day_indices[-1]]
                            _eod = _lv_processor.force_close(
                                price=float(_last_row['close']),
                                timestamp=float(_last_row['timestamp']),
                                bar_index=_lv_bar_counter,
                            )
                            if _eod:
                                _lv_day_trades.append(_eod)

                        _lv_n = len(_lv_day_trades)
                        _lv_wins = sum(1 for t in _lv_day_trades if t['pnl'] > 0)
                        _lv_pnl = sum(t['pnl'] for t in _lv_day_trades)
                        _live_val_trades.extend(_lv_day_trades)
                        _live_val_day_ledger.append({
                            'date': _lv_date, 'trades': _lv_n,
                            'wins': _lv_wins, 'pnl': _lv_pnl,
                        })
                        print(f"    {_lv_date}: {_lv_n} trades, "
                              f"{_lv_wins/_lv_n*100:.0f}% WR, ${_lv_pnl:+.2f}" if _lv_n else
                              f"    {_lv_date}: 0 trades")

        # ── Live Validation Parity Report ──────────────────────────────────
        if _live_val_days > 0 and _live_val_trades:
            self._write_live_validation_report(
                _live_val_trades, _live_val_day_ledger,
                _daily_ledger, oracle_trade_records,
                _live_val_days, _exec_engine,
            )

        print("\n  [OK] Forward pass complete -- all files saved.", flush=True)

    @staticmethod
    def _load_tf_dir(tf_dir: str):
        """Load and concat all parquet files in a timeframe directory."""
        _files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
        if not _files:
            return None
        _dfs = [pd.read_parquet(f) for f in _files]
        return pd.concat(_dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

    def run_oos3_standalone(self, data_source: str = None,
                            n_days: int = 5,
                            bias_threshold: float = None,
                            dmi_threshold: float = None,
                            account_size: float = 0.0):
        """Standalone OOS3: BarProcessor only, no inline OOS.

        Loads checkpoints, computes states on OOS data, ticks TBN through
        warmup bars, then runs BarProcessor on the last N trading days.
        Writes parity report comparing against reports/oos_report.txt.
        """
        import re
        from collections import defaultdict

        data_source = data_source or os.path.join('DATA', 'ATLAS_OOS')
        print("\n" + "=" * 80)
        print("  OOS3 STANDALONE — BarProcessor parity test")
        print(f"  Data: {data_source}  |  Last {n_days} trading days")
        print("=" * 80)

        # ── Load checkpoints ──────────────────────────────────────────────
        from core.checkpoint_loader import load_checkpoints
        _bundle = load_checkpoints(self.checkpoint_dir, verbose=True)
        self.pattern_library = _bundle.pattern_library
        self.scaler = _bundle.scaler

        # ── Load 15s data files ───────────────────────────────────────────
        _15s_dir = os.path.join(data_source, '15s')
        if not os.path.isdir(_15s_dir):
            print(f"ERROR: no 15s directory in {data_source}")
            return
        _files = sorted(glob.glob(os.path.join(_15s_dir, '*.parquet')))
        if not _files:
            print(f"ERROR: no parquet files in {_15s_dir}")
            return
        print(f"  Found {len(_files)} 15s files")

        # ── Concat all files, identify trading days ───────────────────────
        _dfs = [pd.read_parquet(f) for f in _files]
        _df_all = pd.concat(_dfs, ignore_index=True).sort_values('timestamp')
        _ts_col = _df_all['timestamp']
        if np.issubdtype(_ts_col.dtype, np.number):
            _df_all['_date'] = pd.to_datetime(_ts_col, unit='s').dt.strftime('%Y-%m-%d')
        else:
            _df_all['_date'] = pd.to_datetime(_ts_col).dt.strftime('%Y-%m-%d')

        _all_dates = sorted(_df_all['_date'].unique())
        _target_dates = _all_dates[-n_days:]
        _warmup_dates = _all_dates[:-n_days] if len(_all_dates) > n_days else []

        print(f"  Total trading days: {len(_all_dates)}")
        print(f"  Warmup days: {len(_warmup_dates)} ({_warmup_dates[0] if _warmup_dates else 'none'} → "
              f"{_warmup_dates[-1] if _warmup_dates else 'none'})")
        print(f"  Target days: {_target_dates[0]} → {_target_dates[-1]}")

        # ── Compute states (GPU, fast) ────────────────────────────────────
        print("  Computing market states...", flush=True)
        _states = self.engine.batch_compute_states(_df_all, use_cuda=True)
        _states_map = {s['bar_idx']: s['state'] for s in _states}
        print(f"  {len(_states)} states computed")

        # ── Load sub-TF data for TBN ─────────────────────────────────────
        _5s_dir = os.path.join(data_source, '5s')
        _1s_dir = os.path.join(data_source, '1s')
        _4h_dir = os.path.join(data_source, '4h')
        _df_5s = self._load_tf_dir(_5s_dir) if os.path.isdir(_5s_dir) else None
        _df_1s = self._load_tf_dir(_1s_dir) if os.path.isdir(_1s_dir) else None
        _df_4h = self._load_tf_dir(_4h_dir) if os.path.isdir(_4h_dir) else None

        # ── Create TBN + prepare ──────────────────────────────────────────
        belief_network = create_belief_network(_bundle, self.engine)
        belief_network.prepare_day(_df_all, states_micro=_states,
                                   df_5s=_df_5s, df_1s=_df_1s, df_4h=_df_4h)

        # ── Tick TBN through warmup bars (fast, no trades) ────────────────
        _warmup_mask = _df_all['_date'].isin(set(_warmup_dates))
        _warmup_indices = _df_all.index[_warmup_mask].tolist()
        print(f"  Ticking TBN through {len(_warmup_indices)} warmup bars...", flush=True)
        for _wi, _w_idx in enumerate(_warmup_indices):
            belief_network.tick_all(_wi)
        _bar_counter = len(_warmup_indices)
        print(f"  TBN warmed ({_bar_counter} bars ticked)")

        # (exit_state removed — exit uses same state as entry, matching inline OOS)

        # ── Create EE + ExitEngine + BarProcessor ─────────────────────────
        _exit_eng = ExitEngine(
            mode='training',
            tick_size=self.asset.tick_size,
            tick_value=self.asset.tick_size * self.asset.point_value,
            min_hold_bars=getattr(self, '_min_hold_bars', 0),
        )
        # Load self-tuned exit params from last forward pass if available
        _tuned_path = os.path.join(self.checkpoint_dir, 'exit_tuning.json')
        if os.path.exists(_tuned_path):
            import json as _json
            with open(_tuned_path) as _f:
                _tuned = _json.load(_f)
            _exit_eng.envelope_half_life_bars = _tuned.get('envelope_half_life_bars', 20)
            _exit_eng.giveback_pct = _tuned.get('giveback_pct', 0.70)
            print(f"  Loaded exit tuning: hl={_exit_eng.envelope_half_life_bars}, "
                  f"gb={_exit_eng.giveback_pct:.0%}")

        _exec_engine = create_execution_engine(
            bundle=_bundle,
            brain=self.brain,
            belief_network=belief_network,
            exit_engine=_exit_eng,
            tick_size=self.asset.tick_size,
            point_value=self.asset.point_value,
            mode='oos',
            tier_preference=True,
            bias_threshold=bias_threshold if bias_threshold is not None else 0.55,
            dmi_threshold=dmi_threshold if dmi_threshold is not None else 0.0,
            depth_only=getattr(self, '_depth_only', None),
        )
        _exec_engine.gate1_dist = 4.5 + 0.5 * 10.0  # match live aggression

        # Random slippage (same as forward pass)
        _slip_rng = np.random.default_rng(42)
        _slip_ticks = 1.0
        _tick_val = self.asset.tick_size * self.asset.point_value

        def _modify_pnl(pnl_dollars):
            return pnl_dollars + _slip_rng.uniform(-_slip_ticks, _slip_ticks) * _tick_val

        # 1s sub-bar wicks
        _has_1s = _df_1s is not None and not _df_1s.empty
        _1s_ts_arr = _df_1s['timestamp'].values.astype(np.float64) if _has_1s else None
        _1s_hi_arr = _df_1s['high'].values.astype(np.float64) if _has_1s else None
        _1s_lo_arr = _df_1s['low'].values.astype(np.float64) if _has_1s else None
        _cur_ts = [0.0]

        def _pre_exit_eval(price, bar_index):
            extra = {}
            if _has_1s and _1s_ts_arr is not None:
                _s0 = np.searchsorted(_1s_ts_arr, _cur_ts[0], side='left')
                _s1 = np.searchsorted(_1s_ts_arr, _cur_ts[0] + 15, side='left')
                if _s1 > _s0:
                    extra['sub_bar_highs'] = _1s_hi_arr[_s0:_s1].tolist()
                    extra['sub_bar_lows'] = _1s_lo_arr[_s0:_s1].tolist()
            _tp = belief_network.get_trade_progress(
                price, tick_size=self.asset.tick_size)
            belief_network._trade_pace_cache = _tp
            belief_network._trade_pace_blend = _tp.get('pace', 1.0) - 1.0
            return extra

        processor = BarProcessor(
            exec_engine=_exec_engine,
            belief_network=belief_network,
            exit_engine=_exit_eng,
            brain=self.brain,
            pattern_library=_bundle.pattern_library,
            anchor_tf='15s', anchor_depth=8,
            tick_size=self.asset.tick_size,
            point_value=self.asset.point_value,
            hooks=BarProcessorHooks(
                modify_pnl=_modify_pnl,
                pre_exit_eval=_pre_exit_eval,
            ),
        )

        # ── Run BarProcessor on target days ───────────────────────────────
        _all_trades = []
        _day_ledger = []
        _target_set = set(_target_dates)

        for _date in _target_dates:
            _day_mask = _df_all['_date'] == _date
            _day_indices = _df_all.index[_day_mask].tolist()
            if not _day_indices:
                continue

            _day_trades = []
            for _idx in _day_indices:
                _state = _states_map.get(_idx)
                if _state is None:
                    _bar_counter += 1
                    continue
                _row = _df_all.iloc[_idx]
                _cur_ts[0] = float(_row['timestamp'])
                # Both inline OOS and BarProcessor now use current bar state
                # (look-ahead bug in inline OOS fixed: _bar_i-1 for all state lookups)
                result = processor.process_bar(
                    bar_index=_bar_counter,
                    price=float(_row['close']),
                    bar_high=float(_row['high']),
                    bar_low=float(_row['low']),
                    timestamp=float(_row['timestamp']),
                    state=_state,
                    # exit_state omitted → defaults to state (current bar)
                )
                _bar_counter += 1
                if result.trade_completed:
                    _day_trades.append(result.trade_completed)

            # EOD flatten
            if processor.in_position and _day_indices:
                _last_row = _df_all.iloc[_day_indices[-1]]
                _eod = processor.force_close(
                    price=float(_last_row['close']),
                    timestamp=float(_last_row['timestamp']),
                    bar_index=_bar_counter,
                )
                if _eod:
                    _day_trades.append(_eod)

            _n = len(_day_trades)
            _wins = sum(1 for t in _day_trades if t['pnl'] > 0)
            _pnl = sum(t['pnl'] for t in _day_trades)
            _all_trades.extend(_day_trades)
            _day_ledger.append({'date': _date, 'trades': _n, 'wins': _wins, 'pnl': _pnl})
            print(f"    {_date}: {_n} trades, "
                  f"{_wins/_n*100:.0f}% WR, ${_pnl:+.2f}" if _n else
                  f"    {_date}: 0 trades")

        # ── Parse OOS report for comparison ───────────────────────────────
        oos_daily = {}
        oos_report_path = os.path.join('reports', 'oos_report.txt')
        try:
            with open(oos_report_path, 'r') as f:
                in_ledger = False
                for line in f:
                    if 'DAILY SESSION LEDGER' in line:
                        in_ledger = True
                        continue
                    if in_ledger and re.match(r'\s+20\d\d-\d\d-\d\d', line):
                        parts = line.split()
                        if len(parts) >= 7:
                            _d = parts[0]
                            _nt = int(parts[1])
                            _nw = int(parts[2])
                            _pnl_str = parts[5].replace(',', '').replace('+', '')
                            try:
                                _pv = float(_pnl_str)
                            except ValueError:
                                _pv = 0.0
                            oos_daily[_d] = {'trades': _nt, 'wins': _nw, 'pnl': _pv}
                    elif in_ledger and line.strip().startswith('──') and oos_daily:
                        break
        except FileNotFoundError:
            print(f"  WARNING: {oos_report_path} not found — no OOS comparison")

        # ── Write parity report ───────────────────────────────────────────
        self._write_live_validation_report(
            _all_trades, _day_ledger,
            [{'date': d, **oos_daily.get(d, {'trades': 0, 'wins': 0, 'pnl': 0})}
             for d in _target_dates],
            [], n_days, _exec_engine,
        )

        # ── Summary ──────────────────────────────────────────────────────
        _total_trades = len(_all_trades)
        _total_pnl = sum(t['pnl'] for t in _all_trades)
        _total_wins = sum(1 for t in _all_trades if t['pnl'] > 0)
        _wr = _total_wins / _total_trades if _total_trades else 0

        # Save warmed brain
        _brain_path = os.path.join(self.checkpoint_dir, 'live_brain.pkl')
        self.brain.save(_brain_path)

        print("\n" + "=" * 80)
        print("  OOS3 STANDALONE COMPLETE")
        print(f"  Trades: {_total_trades}  |  WR: {_wr:.1%}  |  PnL: ${_total_pnl:+,.2f}")
        print(f"  Brain saved: {_brain_path}")
        print("=" * 80)

        # Store summary for external access
        self._fp_summary = {
            'total_trades': _total_trades, 'win_rate': _wr,
            'total_pnl': _total_pnl, 'trades': _total_trades,
        }

    def _write_live_validation_report(
            self, lv_trades, lv_day_ledger, oos_daily_ledger,
            oos_trade_records, n_days, exec_engine):
        """Write OOS3 parity report: OOS2 (inline) vs OOS3 (BarProcessor)."""
        import datetime as _dt
        from collections import Counter

        W = 80

        # ── OOS3 stats ──
        lv_n = len(lv_trades)
        lv_wins = sum(1 for t in lv_trades if t['pnl'] > 0)
        lv_losses = sum(1 for t in lv_trades if t['pnl'] < 0)
        lv_be = sum(1 for t in lv_trades if t['pnl'] == 0)
        lv_pnl = sum(t['pnl'] for t in lv_trades)
        lv_gross_win = sum(t['pnl'] for t in lv_trades if t['pnl'] > 0)
        lv_gross_loss = sum(t['pnl'] for t in lv_trades if t['pnl'] < 0)
        lv_wr = lv_wins / lv_n * 100 if lv_n else 0
        lv_avg = lv_pnl / lv_n if lv_n else 0
        lv_pf = lv_gross_win / abs(lv_gross_loss) if lv_gross_loss else 0

        # ── OOS2 stats (from daily ledger) ──
        lv_dates = {d['date'] for d in lv_day_ledger}
        oos_matching = [d for d in oos_daily_ledger if d['date'] in lv_dates]
        oos_n = sum(d['trades'] for d in oos_matching)
        oos_wins = sum(d['wins'] for d in oos_matching)
        oos_pnl = sum(d['pnl'] for d in oos_matching)
        oos_wr = oos_wins / oos_n * 100 if oos_n else 0
        oos_avg = oos_pnl / oos_n if oos_n else 0
        oos_days = max(1, len(oos_matching))

        # ── Exit reason breakdown (OOS3) ──
        exit_stats = {}  # reason -> {n, wins, pnl, gross_win, gross_loss}
        for t in lv_trades:
            r = t.get('exit_reason', 'unknown')
            if r not in exit_stats:
                exit_stats[r] = {'n': 0, 'wins': 0, 'pnl': 0.0,
                                 'gross_win': 0.0, 'gross_loss': 0.0}
            exit_stats[r]['n'] += 1
            exit_stats[r]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                exit_stats[r]['wins'] += 1
                exit_stats[r]['gross_win'] += t['pnl']
            elif t['pnl'] < 0:
                exit_stats[r]['gross_loss'] += t['pnl']

        # ── Direction breakdown (OOS3) ──
        dir_stats = {}
        for t in lv_trades:
            s = t.get('side', '?')
            if s not in dir_stats:
                dir_stats[s] = {'n': 0, 'wins': 0, 'pnl': 0.0}
            dir_stats[s]['n'] += 1
            dir_stats[s]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                dir_stats[s]['wins'] += 1

        # ── Direction source distribution (OOS3) ──
        dir_sources = Counter()
        for t in lv_trades:
            dir_sources[t.get('dir_source', '?')] += 1

        # ── Drawdown (OOS3) ──
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        consec_loss = 0
        max_consec = 0
        for t in lv_trades:
            cum += t['pnl']
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            if t['pnl'] < 0:
                consec_loss += 1
                max_consec = max(max_consec, consec_loss)
            else:
                consec_loss = 0

        # ── Build report ──
        L = []
        L.append("=" * W)
        L.append("  OOS3 PARITY REPORT — OOS2 (inline) vs OOS3 (BarProcessor)")
        L.append(f"  Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L.append(f"  Target: last {n_days} trading days")
        L.append("=" * W)

        # ── Key Metrics ──
        L.append("")
        L.append(f"  {'Metric':<22} {'OOS2':>14} {'OOS3':>14} {'Delta':>14}")
        L.append(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
        L.append(f"  {'Trades':<22} {oos_n:>14} {lv_n:>14} {lv_n - oos_n:>+14}")
        L.append(f"  {'Win Rate':<22} {oos_wr:>13.1f}% {lv_wr:>13.1f}% {lv_wr - oos_wr:>+13.1f}%")
        L.append(f"  {'Total PnL':<22} ${oos_pnl:>12,.2f} ${lv_pnl:>12,.2f} ${lv_pnl - oos_pnl:>+12,.2f}")
        L.append(f"  {'Avg Trade':<22} ${oos_avg:>12,.2f} ${lv_avg:>12,.2f} ${lv_avg - oos_avg:>+12,.2f}")
        L.append(f"  {'Gross Win':<22} {'':>14} ${lv_gross_win:>12,.2f}")
        L.append(f"  {'Gross Loss':<22} {'':>14} ${abs(lv_gross_loss):>12,.2f}")
        L.append(f"  {'Profit Factor':<22} {'':>14} {lv_pf:>14.2f}")
        L.append(f"  {'Max Drawdown':<22} {'':>14} ${max_dd:>12,.2f}")
        L.append(f"  {'Max Consec Losses':<22} {'':>14} {max_consec:>14}")

        # ── Exit Reason Breakdown (OOS3) ──
        L.append("")
        L.append("=" * W)
        L.append("  EXIT REASON BREAKDOWN (OOS3)")
        L.append("=" * W)
        L.append(f"  {'Reason':<18} {'Trades':>7} {'Win%':>6} {'Gross Win':>12} "
                 f"{'Gross Loss':>12} {'Net PnL':>12} {'Avg':>9}")
        L.append(f"  {'-'*18} {'-'*7} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*9}")
        for r in sorted(exit_stats.keys(), key=lambda k: -exit_stats[k]['pnl']):
            es = exit_stats[r]
            wr = es['wins'] / es['n'] * 100 if es['n'] else 0
            avg = es['pnl'] / es['n'] if es['n'] else 0
            L.append(f"  {r:<18} {es['n']:>7} {wr:>5.0f}% ${es['gross_win']:>10,.2f} "
                     f"${abs(es['gross_loss']):>10,.2f} ${es['pnl']:>+10,.2f} ${avg:>+8,.2f}")

        # ── Direction Breakdown (OOS3) ──
        L.append("")
        L.append("=" * W)
        L.append("  DIRECTION BREAKDOWN (OOS3)")
        L.append("=" * W)
        for s in sorted(dir_stats.keys()):
            ds = dir_stats[s]
            wr = ds['wins'] / ds['n'] * 100 if ds['n'] else 0
            avg = ds['pnl'] / ds['n'] if ds['n'] else 0
            L.append(f"  {s:<8} {ds['n']:>4} trades  WR={wr:.0f}%  "
                     f"PnL=${ds['pnl']:>+10,.2f}  Avg=${avg:>+8,.2f}")

        # ── Direction Source (OOS3) ──
        if dir_sources:
            L.append("")
            L.append("  DIRECTION SOURCE DISTRIBUTION")
            for ds, cnt in dir_sources.most_common():
                pct = cnt / lv_n * 100 if lv_n else 0
                L.append(f"    {ds}: {cnt} ({pct:.1f}%)")

        # ── Per-Day Comparison ──
        L.append("")
        L.append("=" * W)
        L.append("  PER-DAY COMPARISON")
        L.append("=" * W)
        L.append(f"  {'Date':<12} {'OOS2_T':>7} {'OOS2_PnL':>11} "
                 f"{'OOS3_T':>7} {'OOS3_PnL':>11} {'Delta':>11}")
        L.append(f"  {'-'*12} {'-'*7} {'-'*11} {'-'*7} {'-'*11} {'-'*11}")
        for lv_day in lv_day_ledger:
            d = lv_day['date']
            oos_day = next((o for o in oos_matching if o['date'] == d), None)
            oos_dt = oos_day['trades'] if oos_day else 0
            oos_dp = oos_day['pnl'] if oos_day else 0
            delta = lv_day['pnl'] - oos_dp
            L.append(f"  {d:<12} {oos_dt:>7} ${oos_dp:>9,.2f} "
                     f"{lv_day['trades']:>7} ${lv_day['pnl']:>9,.2f} "
                     f"${delta:>+9,.2f}")

        # ── Trade Log (OOS3) ──
        L.append("")
        L.append("=" * W)
        L.append("  TRADE LOG (OOS3)")
        L.append("=" * W)
        if lv_trades:
            L.append(f"  {'#':>3}  {'Side':<6} {'Entry':>10} {'Exit':>10} "
                     f"{'PnL':>10} {'Reason':<18} {'Bars':>5} {'DirSrc':<12}")
            L.append("  " + "-" * 75)
            cum_pnl = 0.0
            for i, t in enumerate(lv_trades, 1):
                cum_pnl += t['pnl']
                side = t.get('side', '?')
                entry = t.get('entry_price', 0)
                exit_p = t.get('exit_price', 0)
                pnl = t['pnl']
                reason = t.get('exit_reason', '?')
                bars = t.get('bars_held', 0)
                dsrc = t.get('dir_source', '?')
                L.append(f"  {i:>3}  {side:<6} {entry:>10,.2f} {exit_p:>10,.2f} "
                         f"${pnl:>+9,.2f} {reason:<18} {bars:>5} {dsrc:<12}")
            L.append("  " + "-" * 75)
            L.append(f"  {'':>3}  {'':6} {'':10} {'TOTAL':>10} "
                     f"${lv_pnl:>+9,.2f}")
        else:
            L.append("  No trades.")

        # ── Parity Verdict ──
        L.append("")
        L.append("=" * W)
        L.append("  PARITY VERDICT")
        L.append("=" * W)
        issues = []
        if oos_n > 0:
            trade_ratio = lv_n / oos_n
            if abs(trade_ratio - 1.0) > 0.3:
                issues.append(f"TRADE COUNT: OOS3={lv_n} vs OOS2={oos_n} "
                              f"({trade_ratio:.0%} ratio)")
        if abs(lv_wr - oos_wr) > 5:
            issues.append(f"WIN RATE: OOS3={lv_wr:.1f}% vs OOS2={oos_wr:.1f}%")
        if oos_avg > 0 and abs(lv_avg - oos_avg) > oos_avg * 0.5:
            issues.append(f"AVG TRADE: OOS3=${lv_avg:.2f} vs OOS2=${oos_avg:.2f}")

        sl_count = exit_stats.get('stop_loss', {}).get('n', 0)
        if lv_n > 0 and sl_count / lv_n > 0.85:
            issues.append(f"SL DOMINANT: {sl_count}/{lv_n} trades "
                          f"({sl_count/lv_n:.0%}) exit via stop_loss")

        if issues:
            L.append(f"  Status: FAILED ({len(issues)} issues)")
            for issue in issues:
                L.append(f"    - {issue}")
        else:
            L.append("  Status: PASSED")
        L.append("")
        L.append("=" * W)

        report_text = '\n'.join(L)
        print("\n" + report_text)

        # Save report
        os.makedirs(os.path.join('reports', 'live'), exist_ok=True)
        ts_str = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join('reports', 'live', f'parity_report_{ts_str}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text + '\n')
        print(f"  Parity report: {report_path}")

        # Save warmed brain for live handoff
        brain_path = os.path.join(self.checkpoint_dir, 'live_brain.pkl')
        self.brain.save(brain_path)
        print(f"  Warmed brain saved: {brain_path} ({len(self.brain.table)} states)")

    def _write_forward_pass_reports(
        self, *,
        total_trades, total_wins, total_pnl,
        oracle_trade_records, fn_oracle_records,
        decision_matrix_records, pid_oracle_records,
        daily_files_15s, start_date, end_date,
        _worker_total_states, _worker_days_with_data,
        _equity_enabled, account_size, running_equity, peak_equity, trough_equity,
        skipped_ruin, _dd_aggression, account_ruined, ruin_day,
        _daily_ledger, _max_consec_losing_days,
        _all_day_dips, _worst_intraday_dip, _worst_dip_date,
        _cumul_pnl, _cumul_peak, _cumul_trough, _cumul_trough_date,
        _cumul_max_dd, _cumul_dd_trough_date,
        audit_tp, audit_fp_wrong, audit_fp_noise, audit_fn, audit_tn,
        fn_potential_pnl, score_loser_pnl,
        total_bars_processed, bars_with_detection, bars_slot_blocked, n_signals_seen,
        depth_traded,
        _exec_engine, _dir_corrections,
        _out_dir, _reports_out,
        oos_mode, _analysis_mode,
        _pp_enabled, _pp_flip_count, _pp_all_trades,
        _NINJATRADER_MNQ_MARGIN,
        _trade_replays=None,
    ):
        """Generate forward pass reports, save CSVs, run analytics, save snapshot."""
        # Final Report
        import datetime as _datetime
        import subprocess as _rpt_sp
        _run_ts = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Git change context for report traceability ────────────────────────
        try:
            _git_hash = _rpt_sp.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=_rpt_sp.DEVNULL).decode().strip()
        except Exception:
            _git_hash = 'unknown'
        try:
            _git_msg = _rpt_sp.check_output(
                ['git', 'log', '-1', '--pretty=%s'],
                stderr=_rpt_sp.DEVNULL).decode().strip()
        except Exception:
            _git_msg = ''
        try:
            _git_diff_stat = _rpt_sp.check_output(
                ['git', 'diff', '--stat', 'HEAD~1', 'HEAD'],
                stderr=_rpt_sp.DEVNULL).decode().strip()
            # Keep only the summary line (last line: "N files changed, ...")
            _git_diff_summary = _git_diff_stat.split('\n')[-1].strip() if _git_diff_stat else ''
        except Exception:
            _git_diff_summary = ''

        report_lines = []
        _sec = {}  # section_name -> start index in report_lines (for reorder)
        _sec['header'] = len(report_lines)
        report_lines.append("=" * 80)
        _mode_tag = 'OOS' if oos_mode else 'IS'
        report_lines.append(f"{_mode_tag} FORWARD PASS COMPLETE  (run: {_run_ts})")
        report_lines.append(f"  Commit: {_git_hash} — {_git_msg}")
        if _git_diff_summary:
            report_lines.append(f"  Changes: {_git_diff_summary}")
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
            report_lines.append(f"  Trades skipped (risk/survival gate): {skipped_ruin}")
            report_lines.append(f"  Final aggression:  {_dd_aggression:.0%}  (equity/start: ${running_equity:.0f}/${account_size:.0f})")
            if account_ruined:
                report_lines.append(f"  !! ACCOUNT RUINED on {ruin_day} -- equity fell below margin (${_NINJATRADER_MNQ_MARGIN:.0f})")
            else:
                report_lines.append(f"  Account status:    SURVIVED ({_final_equity:.2f} remaining)")

        # ── Daily session ledger + survival summary ─────────────────────────────
        if _daily_ledger:
            report_lines.append("")
            report_lines.append("── DAILY SESSION LEDGER ──")
            report_lines.append(f"  {'Date':<12} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'Day PnL':>12} {'Min Dip':>10} {'Cumul PnL':>12} {'Cumul DD':>10}")
            report_lines.append("  " + "─" * 90)
            for dl in _daily_ledger:
                _dwr = (dl['wins'] / dl['trades'] * 100) if dl['trades'] > 0 else 0.0
                _dip = dl.get('min_dip', 0.0)
                _dip_str = f"${_dip:>+8,.2f}" if _dip < 0 else f"{'$0':>9}"
                _cdd = dl.get('cumul_dd', 0.0)
                _cdd_str = f"${_cdd:>8,.2f}" if _cdd > 0 else f"{'$0':>9}"
                report_lines.append(
                    f"  {dl['date']:<12} {dl['trades']:>6} {dl['wins']:>5} {_dwr:>5.1f}% "
                    f"${dl['pnl']:>+10,.2f} {_dip_str} ${dl.get('cumul_pnl', 0):>+10,.2f} {_cdd_str}"
                )
            report_lines.append("")
            report_lines.append("── DRAWDOWN SUMMARY ──")
            report_lines.append(f"  Max consecutive losing days: {_max_consec_losing_days}")
            # Count how many days aggression was reduced
            _reduced_days = sum(1 for dl in _daily_ledger if dl['aggression'] < 1.0)
            if _reduced_days:
                report_lines.append(f"  Days with reduced aggression: {_reduced_days}/{len(_daily_ledger)}")

        # ── MINIMUM EQUITY CALCULATION ────────────────────────────────────────
        # Each day starts at $0. The worst intraday dip below $0 = min equity needed.
        if _all_day_dips:
            report_lines.append("")
            report_lines.append("── MINIMUM EQUITY REQUIRED ──")
            report_lines.append(f"  Method: worst intraday dip below $0 across all trading days")
            report_lines.append(f"  (each day starts fresh at $0 — captures the deepest hole before recovery)")
            report_lines.append("")
            # Worst dip
            _min_eq = abs(_worst_intraday_dip) if _worst_intraday_dip < 0 else 0.0
            report_lines.append(f"  Worst intraday dip:  ${_worst_intraday_dip:+,.2f}  on {_worst_dip_date}")
            report_lines.append(f"  ► MINIMUM EQUITY:    ${_min_eq:,.2f}")
            report_lines.append(f"  ► With 50% buffer:   ${_min_eq * 1.5:,.2f}")
            report_lines.append("")
            # Top 5 worst days
            _sorted_dips = sorted(_all_day_dips, key=lambda x: x[1])
            _top_n = min(5, len(_sorted_dips))
            report_lines.append(f"  Top {_top_n} worst intraday dips:")
            for _d, _v in _sorted_dips[:_top_n]:
                report_lines.append(f"    {_d}:  ${_v:+,.2f}")
            # Days that dipped below zero
            _dip_days = sum(1 for _, v in _all_day_dips if v < 0)
            report_lines.append(f"  Days that dipped below $0: {_dip_days}/{len(_all_day_dips)}")

        # ── CUMULATIVE EQUITY CURVE (carries across days) ─────────────────────
        report_lines.append("")
        report_lines.append("── CUMULATIVE EQUITY CURVE ──")
        report_lines.append(f"  (starts at $0 on day 1, never resets — shows true equity path)")
        report_lines.append("")
        report_lines.append(f"  Final cumulative PnL:  ${_cumul_pnl:+,.2f}")
        report_lines.append(f"  Peak equity:           ${_cumul_peak:+,.2f}")
        report_lines.append(f"  Trough equity:         ${_cumul_trough:+,.2f}  on {_cumul_trough_date or 'N/A'}")
        report_lines.append(f"  Max drawdown from peak: ${_cumul_max_dd:,.2f}  (trough on {_cumul_dd_trough_date or 'N/A'})")
        if _cumul_trough < 0:
            report_lines.append(f"  ► MIN EQUITY (cumulative): ${abs(_cumul_trough):,.2f}  (deepest hole from $0 start)")
        else:
            report_lines.append(f"  ► Equity never went negative — $0 start survives the entire run")
        # Identify critical days: days where cumulative equity was at its lowest
        if _daily_ledger:
            report_lines.append("")
            report_lines.append("  Critical days (cumulative equity at risk):")
            _sorted_by_cumul = sorted(_daily_ledger, key=lambda d: d.get('cumul_pnl', 0))
            for _dl in _sorted_by_cumul[:5]:
                _cpnl = _dl.get('cumul_pnl', 0)
                report_lines.append(f"    {_dl['date']}:  cumul=${_cpnl:+,.2f}  day_pnl=${_dl['pnl']:+,.2f}  dip=${_dl.get('min_dip', 0):+,.2f}")

        report_lines.append("=" * 80)

        # ── ORACLE PROFIT ATTRIBUTION ────────────────────────────────────────────
        import csv as _csv

        _sec['oracle_banner'] = len(report_lines)
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("ORACLE PROFIT ATTRIBUTION")
        report_lines.append("=" * 80)

        # ── 1. Opportunity landscape ─────────────────────────────────────────────
        _oracle_available = any(r.get('oracle_label', 0) != 0
                                for r in oracle_trade_records)
        total_real_opps = audit_tp + audit_fp_wrong + audit_fn  # oracle said real move
        total_noise_opps = audit_fp_noise + audit_tn              # oracle said noise
        tp_potential  = sum(r['oracle_potential_pnl'] for r in oracle_trade_records if r['oracle_label'] != 0 and r['oracle_label_name'] not in ('NOISE',))
        ideal_profit  = tp_potential + fn_potential_pnl           # perfect execution on everything

        _sec['opportunity'] = len(report_lines)
        report_lines.append("")
        if not _oracle_available and oos_mode:
            report_lines.append(f"  ORACLE: N/A (compressed mode — no forward-looking labels)")
        else:
            report_lines.append(f"  TOTAL SIGNALS SEEN BY ORACLE: {total_real_opps + total_noise_opps:,}")
            report_lines.append(f"    Real moves (MEGA/SCALP):  {total_real_opps:>6,}   -- worth ${ideal_profit:>10,.2f} if perfectly traded")
            report_lines.append(f"    Noise (no real move):     {total_noise_opps:>6,}")

        # ── 2. What we did ───────────────────────────────────────────────────────
        n_traded   = len(oracle_trade_records)
        n_skipped  = audit_fn + audit_tn
        _sec['what_we_did'] = len(report_lines)
        report_lines.append("")
        report_lines.append(f"  WHAT WE DID:")
        _total_opps = total_real_opps + total_noise_opps
        _traded_pct = n_traded / _total_opps * 100 if _total_opps > 0 else 0
        _skipped_pct = n_skipped / _total_opps * 100 if _total_opps > 0 else 0
        report_lines.append(f"    Traded:  {n_traded:>6,}  ({_traded_pct:.1f}% of all signals)")
        report_lines.append(f"    Skipped: {n_skipped:>6,}  ({_skipped_pct:.1f}% of all signals)")

        # ── 2b. Detection funnel (bar-level) ──────────────────────────────────────
        _bars_blind = total_bars_processed - bars_with_detection
        _bars_evaluated = bars_with_detection - bars_slot_blocked
        _sec['detection_funnel'] = len(report_lines)
        report_lines.append("")
        report_lines.append(f"  DETECTION FUNNEL (bar-level)")
        if total_bars_processed > 0:
            _pct_b = lambda n: f"{n/total_bars_processed*100:.1f}%"
            report_lines.append(f"    Total 15s bars processed:  {total_bars_processed:>9,}  (100%)")
            report_lines.append(f"    Bars with detection:       {bars_with_detection:>9,}  ({_pct_b(bars_with_detection)})")
            report_lines.append(f"    Bars with NO detection:    {_bars_blind:>9,}  ({_pct_b(_bars_blind)})  <- model blind")
            report_lines.append(f"    Bars slot-blocked:         {bars_slot_blocked:>9,}  ({_pct_b(bars_slot_blocked)})  <- position open, can't trade")
            report_lines.append(f"    Bars evaluated (free slot):{_bars_evaluated:>9,}  ({_pct_b(_bars_evaluated)})")
            report_lines.append(f"    Candidates on those bars:  {n_signals_seen:>9,}  (avg {n_signals_seen/max(1,_bars_evaluated):.1f}/bar)")

        # ── 2b2. Candidate competition stats ─────────────────────────────────────
        _comp_total = _exec_engine.bars_with_competition + _exec_engine.bars_single_candidate
        if _comp_total > 0:
            report_lines.append("")
            report_lines.append(f"  CANDIDATE COMPETITION (bars where gates were passed)")
            report_lines.append(f"    Single candidate (no competition): {_exec_engine.bars_single_candidate:>7,}  "
                                f"({_exec_engine.bars_single_candidate/_comp_total*100:.1f}%)")
            report_lines.append(f"    Multiple candidates (tiebreaker):  {_exec_engine.bars_with_competition:>7,}  "
                                f"({_exec_engine.bars_with_competition/_comp_total*100:.1f}%)")
            if _exec_engine.tier_changed_winner > 0:
                report_lines.append(f"    Tier preference flipped winner:    {_exec_engine.tier_changed_winner:>7,}  "
                                    f"({_exec_engine.tier_changed_winner/_exec_engine.bars_with_competition*100:.1f}% of competitions)")

        # ── 2c. Skip reason breakdown ─────────────────────────────────────────────
        _skip = _exec_engine.get_skip_counts()
        skip_headroom = _skip['skip_headroom']
        skip_depth = _skip.get('skip_depth', 0)
        skip_dist = _skip['skip_dist']
        skip_brain = _skip['skip_brain']
        skip_conviction = _skip['skip_conviction']
        skip_mom_align = _skip.get('skip_momentum_align', 0)
        skip_physics_qg = _skip['skip_physics_qg']
        skip_competition = _skip.get('skip_competition', 0)
        skip_fdmi = _skip.get('skip_fdmi_fakeout', 0)
        _all_skipped = (skip_headroom + skip_depth + skip_dist + skip_brain
                        + skip_conviction + skip_mom_align + skip_physics_qg
                        + skip_competition + skip_fdmi)
        _unaccounted = n_signals_seen - _all_skipped - n_traded
        _sec['skip_reasons'] = len(report_lines)
        report_lines.append("")
        report_lines.append(f"  WHY SIGNALS WERE SKIPPED  (total candidates evaluated: {n_signals_seen:,})")
        if n_signals_seen > 0:
            _pct_s = lambda n: f"{n/n_signals_seen*100:.1f}%"
            report_lines.append(f"    Pattern Quality  (no match/noise/struct): {skip_headroom:>6,}  ({_pct_s(skip_headroom)})")
            report_lines.append(f"    Depth Filter     (depth<3 or blacklist):  {skip_depth:>6,}  ({_pct_s(skip_depth)})")
            report_lines.append(f"    Template Match   (dist > 3.0):            {skip_dist:>6,}  ({_pct_s(skip_dist)})")
            report_lines.append(f"    Brain Reject     (unprofitable pattern):  {skip_brain:>6,}  ({_pct_s(skip_brain)})")
            report_lines.append(f"    Score Competition (better candidate won): {skip_competition:>6,}  ({_pct_s(skip_competition)})")
            report_lines.append(f"    Low Conviction   (belief too weak):       {skip_conviction:>6,}  ({_pct_s(skip_conviction)})")
            report_lines.append(f"    Momentum Misalign (F_mom vs direction):   {skip_mom_align:>6,}  ({_pct_s(skip_mom_align)})")
            report_lines.append(f"    FDMI Fakeout     (State A block):         {skip_fdmi:>6,}  ({_pct_s(skip_fdmi)})")
            report_lines.append(f"    Physics Quality  (bypass: depth>3/z>=0):  {skip_physics_qg:>6,}  ({_pct_s(skip_physics_qg)})")
            report_lines.append(f"    Passed all gates -> traded:               {n_traded:>6,}  ({_pct_s(n_traded)})")
            if _unaccounted > 0:
                report_lines.append(f"    ⚠ Unaccounted:                    {_unaccounted:>6,}  ({_pct_s(_unaccounted)})")

        # ── 2c. Traded signal depth distribution ─────────────────────────────────
        _sec['depth_dist'] = len(report_lines)
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
        _sec['wave_maturity'] = len(report_lines)
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
        _sec['depth_pnl'] = len(report_lines)
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

        # ── 2e2. Trade Duration Bins (top 3 by trade count) ───────────────────────
        if oracle_trade_records and 'hold_bars' in oracle_trade_records[0]:
            from collections import defaultdict as _ddd2
            # Duration bins: hold_bars * 15s → minutes → bucket
            _dur_bins = _ddd2(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
            _dur_edges = [
                (0, 2, '<30s'),
                (2, 4, '30s-1m'),
                (4, 8, '1-2m'),
                (8, 20, '2-5m'),
                (20, 40, '5-10m'),
                (40, 80, '10-20m'),
                (80, 160, '20-40m'),
                (160, 480, '40m-2h'),
                (480, float('inf'), '>2h'),
            ]
            for _r in oracle_trade_records:
                _hb = _r.get('hold_bars', 0)
                for _lo, _hi, _label in _dur_edges:
                    if _lo <= _hb < _hi:
                        _dur_bins[_label]['n'] += 1
                        _dur_bins[_label]['wins'] += 1 if _r['result'] == 'WIN' else 0
                        _dur_bins[_label]['pnl'] += _r['actual_pnl']
                        break
            # Sort by trade count descending, take top 3
            _sorted_bins = sorted(
                [(lbl, d) for lbl, d in _dur_bins.items() if d['n'] > 0],
                key=lambda x: x[1]['n'], reverse=True)
            if _sorted_bins:
                report_lines.append("")
                report_lines.append(f"  TRADE DURATION BINS (top 3 by volume):")
                report_lines.append(f"    {'Duration':<10} {'Trades':>7} {'Win%':>7} {'Total PnL':>12} {'Avg/trade':>10}")
                report_lines.append(f"    {'--------':<10} {'------':>7} {'----':>7} {'---------':>12} {'---------':>10}")
                for _lbl, _d in _sorted_bins[:3]:
                    _n = _d['n']
                    _wr = _d['wins'] / _n * 100
                    _avg = _d['pnl'] / _n
                    report_lines.append(
                        f"    {_lbl:<10} {_n:>7,} {_wr:>6.0f}% ${_d['pnl']:>10,.2f} ${_avg:>9.2f}")

        # ── 2f. Dynamic Exit Quality ─────────────────────────────────────────────
        _sec['exit_quality'] = len(report_lines)
        if oracle_trade_records and 'exit_signal_reason' in oracle_trade_records[0]:
            report_lines.append("")
            report_lines.append(f"  DYNAMIC EXIT QUALITY:")

            # Buckets
            # Belief-flip exits: urgent_flip
            # Trail-tightened: low_conviction, wave_mature
            # Trail-widened: aligned_fresh
            # Standard trail: neutral, no_belief

            # exit_reason = ExitEngine's structural exit type (trail_stop, belief_flip, etc.)
            # exit_signal_reason = belief network state at exit (tighten, widen, neutral)
            # Special exits (flip, decay, watchdog) are keyed on exit_reason.
            # Trail quality (tighten/widen/standard) is keyed on exit_signal_reason.

            # Belief flip exits use ExitAction.TRAIL_STOP but reason contains "Belief flip"
            _trail_exits = [r for r in oracle_trade_records if r.get('exit_reason') == 'trail_stop']
            b_flip     = [r for r in _trail_exits if 'Belief flip' in str(r.get('exit_signal_reason', ''))]
            _non_flip_trails = [r for r in _trail_exits if r not in b_flip]
            b_decay    = [r for r in oracle_trade_records if r.get('exit_reason') == 'envelope_decay']
            b_watchdog = [r for r in oracle_trade_records if r.get('exit_reason') == 'watchdog']
            b_band     = [r for r in oracle_trade_records if r.get('exit_reason') == 'band_urgent']
            # Trail-quality buckets: only non-belief-flip trail_stop exits
            b_tight    = [r for r in _non_flip_trails if r.get('exit_signal_reason') in ('low_conviction', 'wave_mature')]
            b_widen    = [r for r in _non_flip_trails if r.get('exit_signal_reason') == 'aligned_fresh']
            b_standard = [r for r in _non_flip_trails if r.get('exit_signal_reason') in ('neutral', 'no_belief', '')]

            def _stats(subset):
                if not subset: return "0 trades"
                n = len(subset)
                avg = sum(r['actual_pnl'] for r in subset) / n
                return f"{n:>5} trades  ->  avg PnL ${avg:>7.2f}"

            report_lines.append(f"    Belief-flip exits:  {_stats(b_flip)}")
            report_lines.append(f"    Envelope decay:     {_stats(b_decay)}")
            report_lines.append(f"    Trail-tightened:    {_stats(b_tight)}")
            report_lines.append(f"    Trail-widened:      {_stats(b_widen)}")
            report_lines.append(f"    Band-urgent exits:  {_stats(b_band)}")
            report_lines.append(f"    Loss watchdog:      {_stats(b_watchdog)}")
            report_lines.append(f"    Standard trail:     {_stats(b_standard)}")

            # Decay score breakdown: WIN vs LOSS average decay at exit
            _win_decay  = [r.get('exit_decay_score', 0.0) for r in oracle_trade_records if r.get('result') == 'WIN']
            _loss_decay = [r.get('exit_decay_score', 0.0) for r in oracle_trade_records if r.get('result') == 'LOSS']
            if _win_decay and _loss_decay:
                report_lines.append(f"    Decay score at exit:  WIN avg={sum(_win_decay)/len(_win_decay):.3f}  LOSS avg={sum(_loss_decay)/len(_loss_decay):.3f}")
            # Self-tuned parameters (independent knobs)
            _ee = _exec_engine.exit_engine
            report_lines.append(f"    Envelope halflife:   {_ee.envelope_half_life_bars:.1f} bars (self-tuned from 20)")
            report_lines.append(f"    Giveback threshold:  {_ee.giveback_pct:.0%} (self-tuned from 70%)")

            # Peak giveback exits
            b_giveback = [r for r in oracle_trade_records if r.get('exit_reason') == 'peak_giveback']
            report_lines.append(f"    Peak giveback exits: {_stats(b_giveback)}")

        # ── 2g. Worker agreement analysis ────────────────────────────────────────
        _sec['workers'] = len(report_lines)
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
        _sec['trades_taken'] = len(report_lines)
        tp_recs       = [r for r in oracle_trade_records if r['oracle_label'] != 0 and
                         ((r['direction']=='LONG' and r['oracle_label']>0) or
                          (r['direction']=='SHORT' and r['oracle_label']<0))]
        fp_wrong_recs = [r for r in oracle_trade_records if r['oracle_label'] != 0 and r not in tp_recs]
        fp_noise_recs = [r for r in oracle_trade_records if r['oracle_label'] == 0]

        # Sub-classify wrong-direction: counter-trend scalps (profitable) vs genuinely wrong
        fp_counter_scalps = [r for r in fp_wrong_recs if r['actual_pnl'] > 0]
        fp_genuinely_wrong = [r for r in fp_wrong_recs if r['actual_pnl'] <= 0]

        tp_pnl       = sum(r['actual_pnl'] for r in tp_recs)
        fp_wrong_pnl = sum(r['actual_pnl'] for r in fp_wrong_recs)
        fp_noise_pnl = sum(r['actual_pnl'] for r in fp_noise_recs)
        _cs_pnl      = sum(r['actual_pnl'] for r in fp_counter_scalps)
        _gw_pnl      = sum(r['actual_pnl'] for r in fp_genuinely_wrong)

        report_lines.append("")
        _oracle_available = any(r.get('oracle_label', 0) != 0
                                for r in oracle_trade_records)
        if not _oracle_available and oos_mode:
            report_lines.append(f"  OF {n_traded:,} TRADES TAKEN:")
            report_lines.append(f"    [Oracle N/A — compressed mode has no forward-looking labels]")
            report_lines.append(f"    Total PnL: ${sum(r['actual_pnl'] for r in oracle_trade_records):>10,.2f}")
            report_lines.append(f"    Winners: {len([r for r in oracle_trade_records if r['actual_pnl'] > 0]):,}  "
                                f"Losers: {len([r for r in oracle_trade_records if r['actual_pnl'] <= 0]):,}")
        else:
            report_lines.append(f"  OF {n_traded:,} TRADES TAKEN:")
            _pct = lambda n: f"{n/n_traded*100:.1f}%" if n_traded else "N/A"
            report_lines.append(f"    Correct direction:     {len(tp_recs):>6,}  ({_pct(len(tp_recs))})  ->  actual: ${tp_pnl:>10,.2f}")
            report_lines.append(f"    Counter-trend scalps:  {len(fp_counter_scalps):>6,}  ({_pct(len(fp_counter_scalps))})  ->  profit: ${_cs_pnl:>10,.2f}  <- oracle wrong-dir but micro-peak captured")
            report_lines.append(f"    Genuinely wrong dir:   {len(fp_genuinely_wrong):>6,}  ({_pct(len(fp_genuinely_wrong))})  ->  losses: ${abs(_gw_pnl):>10,.2f}")
            report_lines.append(f"    Traded noise:          {len(fp_noise_recs):>6,}  ({_pct(len(fp_noise_recs))})  ->  losses: ${abs(fp_noise_pnl):>10,.2f}")

        # ── 3b. Counter-trend template analysis ─────────────────────────────────
        # Per-template: would blocking counter-trend trades improve profitability?
        _sec['counter_trend'] = len(report_lines)
        if fp_wrong_recs or tp_recs:
            from collections import defaultdict as _dct
            _ct_data = _dct(lambda: {'ct_n': 0, 'ct_pnl': 0.0, 'gw_n': 0, 'gw_pnl': 0.0,
                                      'tp_n': 0, 'tp_pnl': 0.0, 'noise_n': 0, 'noise_pnl': 0.0})
            for _r in oracle_trade_records:
                _tid = _r.get('template_id', '?')
                _ol = _r.get('oracle_label', 0)
                _d = _r.get('direction', '')
                _pnl = _r.get('actual_pnl', 0.0)
                if _ol == 0:
                    _ct_data[_tid]['noise_n'] += 1
                    _ct_data[_tid]['noise_pnl'] += _pnl
                elif (_d == 'LONG' and _ol > 0) or (_d == 'SHORT' and _ol < 0):
                    _ct_data[_tid]['tp_n'] += 1
                    _ct_data[_tid]['tp_pnl'] += _pnl
                elif _pnl > 0:
                    _ct_data[_tid]['ct_n'] += 1
                    _ct_data[_tid]['ct_pnl'] += _pnl
                else:
                    _ct_data[_tid]['gw_n'] += 1
                    _ct_data[_tid]['gw_pnl'] += _pnl

            # Sort by wrong-dir total (ct + gw) impact — most active first
            _ct_sorted = sorted(_ct_data.items(),
                                key=lambda x: x[1]['ct_n'] + x[1]['gw_n'], reverse=True)
            # Only show templates with >=5 wrong-dir trades
            _ct_active = [(t, d) for t, d in _ct_sorted if d['ct_n'] + d['gw_n'] >= 5]

            if _ct_active:
                # Global summary
                _total_ct_pnl = sum(d['ct_pnl'] for _, d in _ct_data.items())
                _total_gw_pnl = sum(d['gw_pnl'] for _, d in _ct_data.items())
                _total_wrong_pnl = _total_ct_pnl + _total_gw_pnl
                _n_templates_hurt = sum(1 for _, d in _ct_active if d['ct_pnl'] + d['gw_pnl'] < 0)
                _n_templates_help = sum(1 for _, d in _ct_active if d['ct_pnl'] + d['gw_pnl'] > 0)

                report_lines.append("")
                report_lines.append(f"  COUNTER-TREND TEMPLATE ANALYSIS (would blocking wrong-dir help?)")
                report_lines.append(f"    Global wrong-dir net: ${_total_wrong_pnl:>+,.2f}  "
                                    f"(scalps: ${_total_ct_pnl:>+,.2f}  genuinely wrong: ${_total_gw_pnl:>+,.2f})")
                report_lines.append(f"    Templates where wrong-dir HURTS: {_n_templates_hurt}  "
                                    f"Templates where wrong-dir HELPS: {_n_templates_help}")
                report_lines.append(f"")
                report_lines.append(f"    {'TID':>5}  {'Correct':>8} {'Correct$':>10}  "
                                    f"{'CtrScalp':>8} {'Scalp$':>10}  "
                                    f"{'Wrong':>6} {'Wrong$':>10}  "
                                    f"{'Net Wrong$':>11}  Verdict")
                report_lines.append(f"    {'-----':>5}  {'-------':>8} {'--------':>10}  "
                                    f"{'--------':>8} {'------':>10}  "
                                    f"{'-----':>6} {'------':>10}  "
                                    f"{'-----------':>11}  -------")
                for _tid, _d in _ct_active[:25]:  # top 25
                    _net_wrong = _d['ct_pnl'] + _d['gw_pnl']
                    _verdict = "BLOCK" if _net_wrong < -50 else "KEEP" if _net_wrong > 50 else "NEUTRAL"
                    report_lines.append(
                        f"    {str(_tid):>5}  {_d['tp_n']:>8,} ${_d['tp_pnl']:>9,.2f}  "
                        f"{_d['ct_n']:>8,} ${_d['ct_pnl']:>9,.2f}  "
                        f"{_d['gw_n']:>6,} ${_d['gw_pnl']:>9,.2f}  "
                        f"${_net_wrong:>10,.2f}  {_verdict}")

                # Bottom line
                _block_savings = sum(-(d['ct_pnl'] + d['gw_pnl'])
                                     for _, d in _ct_active if d['ct_pnl'] + d['gw_pnl'] < -50)
                _block_cost = sum(d['ct_pnl'] + d['gw_pnl']
                                  for _, d in _ct_active if d['ct_pnl'] + d['gw_pnl'] > 50)
                report_lines.append(f"")
                report_lines.append(f"    If BLOCK templates with wrong-dir < -$50:")
                report_lines.append(f"      Savings (avoided losses): ${_block_savings:>+,.2f}")
                report_lines.append(f"      Cost (lost scalp profit): ${_block_cost:>+,.2f}")
                report_lines.append(f"      Net impact: ${_block_savings - _block_cost:>+,.2f}")

        # ── 3c. Expected Profit Analysis ────────────────────────────────────────
        # Per-template per-direction: avg PnL = E[PnL]. Was each trade positive EV?
        _sec['expected_profit'] = len(report_lines)
        if oracle_trade_records:
            from collections import defaultdict as _dep
            _ep_data = _dep(lambda: {'long_pnl': 0.0, 'long_n': 0, 'short_pnl': 0.0, 'short_n': 0})
            for _r in oracle_trade_records:
                _tid = _r.get('template_id', '?')
                _d = _r.get('direction', 'LONG')
                _pnl = _r.get('actual_pnl', 0.0)
                _k = _d.lower()
                _ep_data[_tid][f'{_k}_pnl'] += _pnl
                _ep_data[_tid][f'{_k}_n'] += 1

            # Compute E[PnL] for each trade and classify
            _pos_ev_n = 0; _pos_ev_pnl = 0.0
            _neg_ev_n = 0; _neg_ev_pnl = 0.0
            _unknown_n = 0; _unknown_pnl = 0.0
            _neg_ev_would_save = 0.0  # if we had blocked neg-EV trades

            # Running accumulators (simulate sequential E[PnL] as trades happen)
            _run = _dep(lambda: {'long_pnl': 0.0, 'long_n': 0, 'short_pnl': 0.0, 'short_n': 0})
            for _r in oracle_trade_records:
                _tid = _r.get('template_id', '?')
                _d = _r.get('direction', 'LONG').lower()
                _pnl = _r.get('actual_pnl', 0.0)
                _n_prior = _run[_tid][f'{_d}_n']
                if _n_prior >= 3:
                    _e_pnl = _run[_tid][f'{_d}_pnl'] / _n_prior
                    if _e_pnl > 0:
                        _pos_ev_n += 1; _pos_ev_pnl += _pnl
                    else:
                        _neg_ev_n += 1; _neg_ev_pnl += _pnl
                        _neg_ev_would_save -= _pnl  # saving = not losing
                else:
                    _unknown_n += 1; _unknown_pnl += _pnl
                # Update running stats
                _run[_tid][f'{_d}_pnl'] += _pnl
                _run[_tid][f'{_d}_n'] += 1

            _total_n = max(_pos_ev_n + _neg_ev_n + _unknown_n, 1)
            report_lines.append("")
            report_lines.append(f"  EXPECTED PROFIT ANALYSIS (E[PnL] = running avg PnL per template×direction)")
            report_lines.append(f"    Positive EV trades (E[PnL]>0 at entry): {_pos_ev_n:>6,}  "
                                f"({_pos_ev_n/_total_n*100:.1f}%)  actual: ${_pos_ev_pnl:>+12,.2f}  "
                                f"avg: ${_pos_ev_pnl/_pos_ev_n:.2f}/trade" if _pos_ev_n else
                                f"    Positive EV trades: 0")
            report_lines.append(f"    Negative EV trades (E[PnL]<0 at entry): {_neg_ev_n:>6,}  "
                                f"({_neg_ev_n/_total_n*100:.1f}%)  actual: ${_neg_ev_pnl:>+12,.2f}  "
                                f"avg: ${_neg_ev_pnl/_neg_ev_n:.2f}/trade" if _neg_ev_n else
                                f"    Negative EV trades: 0")
            report_lines.append(f"    Unknown (< 3 samples):                  {_unknown_n:>6,}  "
                                f"({_unknown_n/_total_n*100:.1f}%)  actual: ${_unknown_pnl:>+12,.2f}")
            report_lines.append(f"    If blocked neg-EV trades: would save ${_neg_ev_would_save:>+,.2f}")

            # Top templates by negative EV impact
            _ep_sorted = sorted(_ep_data.items(),
                                key=lambda x: min(
                                    x[1]['long_pnl'] / max(1, x[1]['long_n']) if x[1]['long_n'] >= 3 else 999,
                                    x[1]['short_pnl'] / max(1, x[1]['short_n']) if x[1]['short_n'] >= 3 else 999))
            _neg_templates = [(t, d) for t, d in _ep_sorted
                              if (d['long_n'] >= 3 and d['long_pnl'] / d['long_n'] < 0)
                              or (d['short_n'] >= 3 and d['short_pnl'] / d['short_n'] < 0)]
            if _neg_templates:
                report_lines.append(f"")
                report_lines.append(f"    Templates with NEGATIVE E[PnL] direction (top 15):")
                report_lines.append(f"    {'TID':>5}  {'LONG n':>7} {'LONG E[PnL]':>12}  "
                                    f"{'SHORT n':>8} {'SHORT E[PnL]':>13}  Action")
                report_lines.append(f"    {'-----':>5}  {'------':>7} {'-----------':>12}  "
                                    f"{'-------':>8} {'------------':>13}  ------")
                for _tid, _d in _neg_templates[:15]:
                    _l_e = f"${_d['long_pnl']/_d['long_n']:>+8.2f}" if _d['long_n'] >= 3 else "    n/a "
                    _s_e = f"${_d['short_pnl']/_d['short_n']:>+8.2f}" if _d['short_n'] >= 3 else "     n/a "
                    _l_neg = _d['long_n'] >= 3 and _d['long_pnl'] / _d['long_n'] < 0
                    _s_neg = _d['short_n'] >= 3 and _d['short_pnl'] / _d['short_n'] < 0
                    if _l_neg and _s_neg:
                        _act = "SKIP BOTH"
                    elif _l_neg:
                        _act = "SKIP LONG"
                    else:
                        _act = "SKIP SHORT"
                    report_lines.append(f"    {str(_tid):>5}  {_d['long_n']:>7,} {_l_e:>12}  "
                                        f"{_d['short_n']:>8,} {_s_e:>13}  {_act}")

        # ── 4. Exit quality on correct-direction trades ──────────────────────────
        # NOTE: "Reversed" trades had the correct oracle direction but the market
        # still moved against us after entry (capture_rate <= 0).  Their actual
        _sec['exit_detail'] = len(report_lines)
        if tp_recs:
            optimal   = [r for r in tp_recs if r['capture_rate'] >= 0.80]
            reversed_ = [r for r in tp_recs if r['capture_rate'] <= 0]

            # Too late: reached ≥8 ticks MFE during trade but gave back ≥50%
            # (trade was right, held too long past the peak)
            _low_cap = [r for r in tp_recs if 0 < r['capture_rate'] < 0.20]
            too_late = [r for r in _low_cap
                        if r.get('trade_mfe_ticks', 0) >= 8
                        and r.get('trade_mfe_ticks', 0) > 0
                        and (1.0 - r['actual_pnl'] / (r['trade_mfe_ticks'] * self.asset.tick_value))
                            >= 0.50]
            _too_late_ids = set(id(r) for r in too_late)
            too_early = [r for r in _low_cap if id(r) not in _too_late_ids]

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
            report_lines.append(f"  EXIT QUALITY (correct-direction trades, worst → best):")
            report_lines.append(f"    {'Bucket':<36} {'n':>5}  {'Total PnL':>11}  {'Avg PnL':>8}  {'Hold':>9}  {'Cap%':>7}")
            report_lines.append(f"    {'─'*36} {'─'*5}  {'─'*11}  {'─'*8}  {'─'*9}  {'─'*7}")
            report_lines.append(_eq_row("Reversed (mkt flipped after entry)",reversed_, flag="<- leakage"))
            report_lines.append(_eq_row("Too late  (reached peak, gave back)",too_late,  flag="<- giveback"))
            # Too-late sub-bands by giveback percentage
            for _gb_lo in range(50, 100, 10):
                _gb_hi = _gb_lo + 10
                _gb_band = [r for r in too_late
                            if r.get('trade_mfe_ticks', 0) > 0
                            and _gb_lo / 100 <= (1.0 - r['actual_pnl'] / (r['trade_mfe_ticks'] * self.asset.tick_value)) < _gb_hi / 100]
                if _gb_band:
                    report_lines.append(_eq_row(f"  Gave back ({_gb_lo}-{_gb_hi}%)", _gb_band, indent='    '))
            report_lines.append(_eq_row("Too early (<20%, never reached)",   too_early))
            # Partial bands (ascending capture)
            for _lo, _hi, _band in _partial_bands:
                if not _band:
                    continue
                report_lines.append(_eq_row(f"  Partial  ({_lo}-{_hi}% captured)", _band, indent='    '))
            report_lines.append(_eq_row("Optimal  (>=80% captured)",         optimal))
            report_lines.append(f"    Left on table (non-reversed gap):                        ${left_on_table:>10,.0f}")

            # ── Exit reason cross-breakdown ───────────────────────────────────
            report_lines.append("")
            report_lines.append(f"  EXIT REASON → QUALITY CROSS-BREAKDOWN (correct-direction trades):")
            all_reasons = sorted({r.get('exit_reason', 'unknown') for r in tp_recs})
            buckets_def = [
                ('Optimal',   optimal),
                ('Partial',   partial),
                ('Too early', too_early),
                ('Too late',  too_late),
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
            report_lines.append(f"    {'Bucket':<22}  {'Avg oracle MFE':>14}  {'Avg trade MFE':>13}  {'Avg actual PnL':>14}  {'Avg hold bars':>13}")
            _detail_buckets = (
                [('Optimal', optimal), ('Too early', too_early), ('Too late', too_late), ('Reversed', reversed_)]
                + [(f'Partial {lo}-{hi}%', band) for lo, hi, band in _partial_bands if band]
            )
            for label, recs in _detail_buckets:
                if not recs:
                    continue
                avg_mfe_usd = sum(r.get('oracle_mfe', 0) for r in recs) / len(recs) * self.asset.point_value
                avg_tmfe    = sum(r.get('trade_mfe_ticks', 0) for r in recs) / len(recs)
                avg_act     = sum(r['actual_pnl'] for r in recs) / len(recs)
                avg_hb      = sum(r.get('hold_bars', 0) for r in recs) / len(recs)
                report_lines.append(f"    {label:<22}  ${avg_mfe_usd:>13,.0f}  {avg_tmfe:>10.1f}tks  ${avg_act:>13,.0f}  {avg_hb:>13.1f}")

            # ── Per-depth exit quality (hold shown as real time, not 15s bars) ──
            _depths_seen = sorted({r.get('entry_depth', 6) for r in tp_recs})
            if len(_depths_seen) > 1:
                # Approximate TF label per depth (1h tree → 1s leaf)
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
            optimal = []
            left_on_table = 0.0

        # ── 5. Profit gap summary ────────────────────────────────────────────────
        # "Reversed" losses = actual dollars lost on correct-direction trades that
        _sec['profit_gap'] = len(report_lines)
        reversed_loss_val  = abs(sum(r['actual_pnl'] for r in reversed_))
        non_reversed_val   = [r for r in tp_recs if r['capture_rate'] > 0] if tp_recs else []
        left_on_table_val  = sum(max(0, r['oracle_potential_pnl'] - r['actual_pnl'])
                                 for r in non_reversed_val)

        report_lines.append("")
        report_lines.append(f"  PROFIT GAP ANALYSIS:")
        report_lines.append(f"    Ideal (golden-path: gate-blocked + traded, perfect exits):  ${ideal_profit:>12,.2f}")
        report_lines.append(f"    -----------------------------------------------------")
        report_lines.append(f"    Lost -- missed opportunities (gate-blocked):  ${fn_potential_pnl:>12,.2f}  ({fn_potential_pnl/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- genuinely wrong direction:            ${abs(_gw_pnl):>12,.2f}  ({abs(_gw_pnl)/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Banked -- counter-trend scalps:              +${_cs_pnl:>11,.2f}  (micro-peak profit, oracle horizon mismatch)" if ideal_profit and _cs_pnl > 0 else "")
        report_lines.append(f"    Lost -- noise trades:                         ${abs(fp_noise_pnl):>12,.2f}  ({abs(fp_noise_pnl)/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- reversed after correct entry:         ${reversed_loss_val:>12,.2f}  ({reversed_loss_val/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    Lost -- TP underperform (non-reversed):       ${left_on_table_val:>12,.2f}  ({left_on_table_val/ideal_profit*100:.1f}% of ideal)" if ideal_profit else "")
        report_lines.append(f"    -----------------------------------------------------")
        report_lines.append(f"    Actual profit:                               ${total_pnl:>12,.2f}  ({total_pnl/ideal_profit*100:.1f}% of ideal)" if ideal_profit else f"    Actual profit: ${total_pnl:.2f}")
        report_lines.append(f"    [info] Score-competition pool (took better same bar): ${score_loser_pnl:>12,.2f}  (not missed -- golden path chose better candidate)")

        # ── 5a-bis. REGRET ANALYSIS (embedded) ────────────────────────────────
        try:
            from tools.regret_analysis import run_regret_analysis as _run_regret
            _regret_df = pd.DataFrame(oracle_trade_records)
            if 'trade_mfe_ticks' in _regret_df.columns and len(_regret_df) > 10:
                _regret_label = 'OOS' if oos_mode else 'IS'
                _regret_text = _run_regret(_regret_df, _regret_label)
                report_lines.append("")
                report_lines.append(_regret_text)
        except Exception as _re:
            report_lines.append(f"\n  [regret analysis skipped: {_re}]")

        # ── 5b. Detection rate by depth ──────────────────────────────────────
        # "Of all real moves oracle saw at each depth, how many did we trade?"
        _sec['detection_rate'] = len(report_lines)
        if oracle_trade_records or fn_oracle_records:
            from collections import defaultdict as _ddr
            _det_traded   = _ddr(int)    # traded real moves per depth
            _det_missed   = _ddr(int)    # missed real moves per depth
            _det_missed_pnl = _ddr(float)
            # Count traded real moves (TP + FP_WRONG, not noise)
            for _r in oracle_trade_records:
                if _r.get('oracle_label', 0) != 0:  # real move, not noise
                    _d = _r.get('entry_depth', 6)
                    _det_traded[_d] += 1
            # Count missed real moves (FN)
            for _r in fn_oracle_records:
                _d = _r.get('depth', 6)
                _det_missed[_d] += 1
                _det_missed_pnl[_d] += _r.get('fn_potential_pnl', 0.0)
            _all_depths = sorted(set(_det_traded) | set(_det_missed))
            if _all_depths:
                _total_real = sum(_det_traded.values()) + sum(_det_missed.values())
                _total_traded_real = sum(_det_traded.values())
                _total_det_rate = _total_traded_real / _total_real * 100 if _total_real else 0
                report_lines.append("")
                report_lines.append(f"  DETECTION RATE BY DEPTH (real moves: traded vs missed)")
                report_lines.append(f"    Overall: {_total_traded_real:,}/{_total_real:,} = {_total_det_rate:.1f}% detection rate")
                report_lines.append(f"    {'Depth':<16} {'Traded':>8} {'Missed':>8} {'Total':>8} {'Rate':>8} {'Missed $':>12}")
                report_lines.append(f"    {'-----':<16} {'------':>8} {'------':>8} {'-----':>8} {'----':>8} {'--------':>12}")
                for _d in _all_depths:
                    _t = _det_traded[_d]
                    _m = _det_missed[_d]
                    _tot = _t + _m
                    _rate = _t / _tot * 100 if _tot else 0
                    _mpnl = _det_missed_pnl[_d]
                    _lbl = _DEPTH_LABELS.get(_d, str(_d))
                    _flag = "  <- blind spot" if _rate < 30 and _tot >= 10 else ""
                    report_lines.append(f"    {_lbl:<16} {_t:>8,} {_m:>8,} {_tot:>8,} {_rate:>7.1f}% ${_mpnl:>10,.2f}{_flag}")

        # ── DIRECTION LEARNING SUMMARY ────────────────────────────────────
        if _dir_corrections:
            _sec['direction_learning'] = len(report_lines)
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("DIRECTION LEARNING (oracle corrections absorbed)")
            report_lines.append("=" * 80)

            _total_corrected = sum(
                1 for acc in _dir_corrections.values()
                if (acc['long_correct'] + acc['long_wrong'] +
                    acc['short_correct'] + acc['short_wrong']) >= 3
            )

            _total_smfe = sum(
                1 for acc in _dir_corrections.values()
                if len(acc['signed_mfe_samples']) >= 15
            )

            report_lines.append(f"  Templates with direction corrections: {_total_corrected}")
            report_lines.append(f"  Templates with signed MFE regression: {_total_smfe}")

            # Show biggest corrections (where the system was most wrong)
            _corrections_list = []
            for tid, acc in _dir_corrections.items():
                if tid not in self.pattern_library:
                    continue
                lib = self.pattern_library[tid]
                orig_long = lib.get('long_bias', 0.5)
                long_total = acc['long_correct'] + acc['long_wrong']
                short_total = acc['short_correct'] + acc['short_wrong']
                if long_total + short_total < 3:
                    continue

                _corrections_list.append({
                    'tid': tid,
                    'orig_long_bias': orig_long,
                    'new_long_bias': lib.get('long_bias', 0.5),
                    'long_correct': acc['long_correct'],
                    'long_wrong': acc['long_wrong'],
                    'short_correct': acc['short_correct'],
                    'short_wrong': acc['short_wrong'],
                    'long_pnl': acc['long_pnl'],
                    'short_pnl': acc['short_pnl'],
                    'shift': abs(lib.get('long_bias', 0.5) - orig_long),
                })

            _corrections_list.sort(key=lambda x: -x['shift'])

            if _corrections_list:
                report_lines.append("")
                report_lines.append(f"  TOP 15 DIRECTION CORRECTIONS (biggest bias shift):")
                report_lines.append(f"  {'TID':>8} {'Orig':>6} {'New':>6} {'Shift':>6} "
                                   f"{'L_ok':>5} {'L_bad':>6} {'S_ok':>5} {'S_bad':>6} "
                                   f"{'L_PnL':>10} {'S_PnL':>10}")
                for r in _corrections_list[:15]:
                    report_lines.append(
                        f"  {r['tid']:>8} {r['orig_long_bias']:>6.2f} "
                        f"{r['new_long_bias']:>6.2f} {r['shift']:>+5.2f} "
                        f"{r['long_correct']:>5} {r['long_wrong']:>6} "
                        f"{r['short_correct']:>5} {r['short_wrong']:>6} "
                        f"${r['long_pnl']:>9,.0f} ${r['short_pnl']:>9,.0f}")

            # Overall direction accuracy before vs after correction
            _all_long_ok = sum(a['long_correct'] for a in _dir_corrections.values())
            _all_long_bad = sum(a['long_wrong'] for a in _dir_corrections.values())
            _all_short_ok = sum(a['short_correct'] for a in _dir_corrections.values())
            _all_short_bad = sum(a['short_wrong'] for a in _dir_corrections.values())
            _all_total = _all_long_ok + _all_long_bad + _all_short_ok + _all_short_bad
            _all_correct = _all_long_ok + _all_short_ok

            if _all_total > 0:
                report_lines.append("")
                report_lines.append(f"  DIRECTION ACCURACY (this run):")
                report_lines.append(f"    Correct: {_all_correct}/{_all_total} "
                                   f"({_all_correct/_all_total*100:.1f}%)")
                report_lines.append(f"    LONG  correct: {_all_long_ok}  wrong: {_all_long_bad}")
                report_lines.append(f"    SHORT correct: {_all_short_ok}  wrong: {_all_short_bad}")
                report_lines.append(f"    NOTE: Next run will use these corrected biases as starting point")

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
            'pct_counter_scalp': len(fp_counter_scalps) / n_traded * 100 if n_traded else 0.0,
            'pct_genuinely_wrong': len(fp_genuinely_wrong) / n_traded * 100 if n_traded else 0.0,
            'pct_noise':       len(fp_noise_recs) / n_traded * 100 if n_traded else 0.0,
            'pct_skipped':     n_skipped / (total_real_opps + total_noise_opps) * 100 if (total_real_opps + total_noise_opps) else 0.0,
            'ideal_profit':    ideal_profit,
            'left_on_table':   left_on_table_val,
            'missed':          fn_potential_pnl,
            'wrong_dir_loss':  abs(_gw_pnl),
            'counter_scalp_profit': _cs_pnl,
        }

        # ── Reorder report: DETAIL sections first, SUMMARIES at end ────────────
        # User preference: see actionable per-trade data first, then high-level picture.
        _sec['end'] = len(report_lines)
        _detail_order = [
            'header',          # basic stats (always first)
            'oracle_banner',   # "ORACLE PROFIT ATTRIBUTION" banner
            'depth_pnl',       # per-depth PnL breakdown
            'exit_quality',    # dynamic exit quality (flip/decay/watchdog/trail)
            'exit_detail',     # exit quality bands + cross-breakdown + by-depth
            'workers',         # worker agreement + direction flips
            'wave_maturity',   # decision-TF wave maturity at entry
            'trades_taken',    # of N trades: correct/wrong/noise
            'opportunity',     # total signals seen by oracle
            'what_we_did',     # traded vs skipped
            'skip_reasons',    # gate breakdown
            'depth_dist',      # traded signal depth distribution
            'profit_gap',      # profit gap analysis (final summary)
            'direction_learning',  # oracle direction corrections absorbed
        ]
        # Build ordered section names (only those that exist)
        _ordered_secs = [s for s in _detail_order if s in _sec]
        # Add any sections not in the explicit order (safety)
        _ordered_secs += [s for s in _sec if s not in _ordered_secs and s != 'end']
        # Rebuild report_lines in new order
        _reordered = []
        for _sname in _ordered_secs:
            _start = _sec[_sname]
            # Find next section start (smallest index > _start among all sections)
            _next_starts = [v for v in _sec.values() if v > _start]
            _end_idx = min(_next_starts) if _next_starts else len(report_lines)
            _reordered.extend(report_lines[_start:_end_idx])
        report_lines = _reordered

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
            _final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            _gw_f = sum(t['actual_pnl'] for t in oracle_trade_records if t['actual_pnl'] > 0)
            _gl_f = abs(sum(t['actual_pnl'] for t in oracle_trade_records if t['actual_pnl'] < 0))
            _pf_f = _gw_f / _gl_f if _gl_f > 0 else 0.0
            self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Analyze',
                                      'step': 'FORWARD_PASS COMPLETE', 'pct': 100,
                                      'pnl': total_pnl, 'trades': total_trades,
                                      'wr': round(_final_wr, 1),
                                      'pf': round(_pf_f, 2),
                                      'gross_w': round(_gw_f, 0),
                                      'gross_l': round(_gl_f, 0)})

        # ── 6. Save CSV ──────────────────────────────────────────────────────────
        def _write_sharded_csv(records, base_name, date_key='day'):
            """Write records to quarterly-sharded CSVs (YYYY_Q1..Q4).
            Only shards when total records > 50k; otherwise writes single file.
            date_key: record field containing YYYYMMDD string or unix timestamp.
            """
            if not records:
                return []
            # Group by quarter
            from collections import defaultdict as _sd
            quarters = _sd(list)
            for r in records:
                dval = r.get(date_key, '')
                if isinstance(dval, (int, float)) and dval > 1e9:
                    import datetime as _sdt
                    dt = _sdt.datetime.fromtimestamp(dval)
                    month = dt.month
                    year_str = str(dt.year)
                else:
                    dstr = ''.join(c for c in str(dval) if c.isdigit())[:8]
                    year_str = dstr[:4] if len(dstr) >= 4 else '0000'
                    month = int(dstr[4:6]) if len(dstr) >= 6 else 1
                q = (month - 1) // 3 + 1
                quarters[f"{year_str}_Q{q}"].append(r)

            if len(records) <= 50_000:
                # Small enough — single file
                path = os.path.join(_out_dir, base_name)
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = _csv.DictWriter(f, fieldnames=list(records[0].keys()))
                    w.writeheader()
                    w.writerows(records)
                return [path]

            # Shard by quarter
            paths = []
            stem, ext = os.path.splitext(base_name)
            for qkey in sorted(quarters.keys()):
                chunk = quarters[qkey]
                fname = f"{stem}_{qkey}{ext}"
                path = os.path.join(_out_dir, fname)
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = _csv.DictWriter(f, fieldnames=list(chunk[0].keys()))
                    w.writeheader()
                    w.writerows(chunk)
                paths.append(path)
            return paths

        if not _analysis_mode:
            if oracle_trade_records:
                # Add trade_class column: correct_dir / counter_trend_scalp / genuinely_wrong / noise
                for _rec in oracle_trade_records:
                    _ol = _rec.get('oracle_label', 0)
                    _td = _rec.get('direction', '')
                    _pnl = _rec.get('actual_pnl', 0)
                    if _ol == 0:
                        _rec['trade_class'] = 'noise'
                    elif (_td == 'LONG' and _ol > 0) or (_td == 'SHORT' and _ol < 0):
                        _rec['trade_class'] = 'correct_dir'
                    elif _pnl > 0:
                        _rec['trade_class'] = 'counter_trend_scalp'
                    else:
                        _rec['trade_class'] = 'genuinely_wrong'

                # Final write with trade_class column (overwrites streaming version)
                _log_name = 'oos_trade_log.csv' if oos_mode else 'oracle_trade_log.csv'
                csv_path = os.path.join(_out_dir, _log_name)
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = _csv.DictWriter(f, fieldnames=list(oracle_trade_records[0].keys()))
                    writer.writeheader()
                    writer.writerows(oracle_trade_records)
                report_lines.append("")
                report_lines.append(f"  Per-trade oracle log saved: {csv_path}")

            if pid_oracle_records:
                pid_csv_path = os.path.join(_out_dir, 'pid_oracle_log.csv')
                with open(pid_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = _csv.DictWriter(f, fieldnames=list(pid_oracle_records[0].keys()))
                    writer.writeheader()
                    writer.writerows(pid_oracle_records)
                report_lines.append(f"  PID oracle log saved: {pid_csv_path} ({len(pid_oracle_records)} signals)")

        # ── Save FN oracle log + report section ──────────────────────────────────
        # Skip all CSV saves, FN analysis, and depth weights in analysis mode.
        if not _analysis_mode and decision_matrix_records:
            _dm_name = 'oos_signal_log.csv' if oos_mode else 'signal_log.csv'
            _dm_paths = _write_sharded_csv(decision_matrix_records, _dm_name, date_key='day')
            _n_traded = sum(1 for r in decision_matrix_records if r['gate'] == 'traded')
            _n_skipped = len(decision_matrix_records) - _n_traded
            if len(_dm_paths) == 1:
                report_lines.append(f"  Signal log saved: {_dm_paths[0]}  ({_n_traded} traded  +  {_n_skipped:,} skipped)")
            else:
                report_lines.append(f"  Signal log saved: {len(_dm_paths)} quarterly shards  ({_n_traded} traded  +  {_n_skipped:,} skipped)")
                for _sp in _dm_paths:
                    report_lines.append(f"    -> {os.path.basename(_sp)}")

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

        if not _analysis_mode and fn_oracle_records:
            import json as _fnjs
            _fn_name = 'oos_fn_log.csv' if oos_mode else 'fn_oracle_log.csv'
            _fn_paths = _write_sharded_csv(fn_oracle_records, _fn_name, date_key='day')
            if len(_fn_paths) == 1:
                report_lines.append(f"  FN oracle log saved: {_fn_paths[0]}  ({len(fn_oracle_records):,} missed real moves)")
            else:
                report_lines.append(f"  FN oracle log saved: {len(_fn_paths)} quarterly shards  ({len(fn_oracle_records):,} missed real moves)")
                for _sp in _fn_paths:
                    report_lines.append(f"    -> {os.path.basename(_sp)}")

            # FN worker agreement analysis:
            # For each TF worker, what fraction of FN signals had the worker
            # agreeing with the oracle direction?  High agreement = gate is blocking
            # moves the workers correctly identified.
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
                'gate0':             'Pattern Quality: no pattern match',
                'gate0_noise':       'Pattern Quality: noise zone <0.5 sigma',
                'gate0_r3_snap':     'Pattern Quality: approach zone BAND_REVERSAL no tmpl',
                'gate0_r3_struct':   'Pattern Quality: approach zone MOMENTUM_BREAK weak',
                'gate0_r4_nightmare':'Pattern Quality: extreme zone nightmare field',
                'gate0_r4_struct':   'Pattern Quality: extreme zone MOMENTUM_BREAK',
                'gate0_hurst':       'Pattern Quality: Hurst < 0.5 choppy',
                'gate0_momentum':    'Pattern Quality: momentum override breakout',
                'gate0_tunnel':      'Pattern Quality: tunnel prob < 40%',
                'gate0_5':           'Depth Filter: depth<3 or blacklist',
                'gate1':             'Template Match: no cluster match (dist>4.5)',
                'gate2':             'Brain Reject: unprofitable pattern',
                'gate3':             'Low Conviction: belief below threshold',
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
                        'gate0_hurst', 'gate0_momentum', 'gate0_tunnel',
                        'gate0_5', 'gate1', 'gate2', 'gate3', 'passed', 'unknown']:
                _gc = _gate_counts.get(_gk, 0)
                if _gc == 0 and _gk == 'unknown':
                    continue
                _pct = 100.0 * _gc / _fn_total if _fn_total else 0.0
                _lbl = _GATE_LABELS.get(_gk, _gk)
                flag2 = "  <-- main bottleneck" if _pct >= 40.0 else ""
                report_lines.append(f"    {_lbl:<42} {_gc:>6,}  ({_pct:5.1f}%){flag2}")

        # ── Compute and save per-depth weights for the NEXT run ──────────────────
        if not _analysis_mode and oracle_trade_records and 'entry_depth' in oracle_trade_records[0]:
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
                _dw_out_path = os.path.join(_out_dir, 'depth_weights.json')
                with open(_dw_out_path, 'w') as _dw_f2:
                    _json2.dump(depth_weights_out, _dw_f2, indent=2)
                report_lines.append(f"  Depth weights saved: {_dw_out_path}")

        # ── Ping-pong direction refinement summary ──────────────────────
        if _pp_enabled and _pp_flip_count > 0:
            report_lines.append("")
            report_lines.append("── PING-PONG DIRECTION REFINEMENT ──")
            report_lines.append(f"  Flips: {_pp_flip_count}")
            # PP trade outcomes (cross-day accumulator)
            _pp_trades = _pp_all_trades
            if _pp_trades:
                _pp_wins = sum(1 for t in _pp_trades if t.result == 'WIN')
                _pp_pnl  = sum(t.pnl for t in _pp_trades)
                report_lines.append(f"  PP Trades: {len(_pp_trades)}  WR: {_pp_wins/len(_pp_trades)*100:.1f}%  PnL: ${_pp_pnl:.2f}")
            if self.brain.dir_bias:
                report_lines.append("  Direction Bias Table:")
                report_lines.append(f"    {'TID':>6s}  {'L_W':>4s} {'L_L':>4s} {'L_WR':>5s}  {'S_W':>4s} {'S_L':>4s} {'S_WR':>5s}")
                for _tid, _b in sorted(self.brain.dir_bias.items(), key=lambda x: sum(x[1].values()), reverse=True)[:15]:
                    _lw, _ll = _b.get('long_w', 0), _b.get('long_l', 0)
                    _sw, _sl2 = _b.get('short_w', 0), _b.get('short_l', 0)
                    _lt = _lw + _ll
                    _st = _sw + _sl2
                    _lwr = f"{_lw/_lt*100:.0f}%" if _lt else "  -"
                    _swr = f"{_sw/_st*100:.0f}%" if _st else "  -"
                    report_lines.append(f"    {str(_tid):>6s}  {_lw:>4d} {_ll:>4d} {_lwr:>5s}  {_sw:>4d} {_sl2:>4d} {_swr:>5s}")

        if _analysis_mode:
            # Depth isolation: print one-line summary, skip full report/CSV/analytics
            _iso_d = getattr(self, '_depth_only', '?')
            _wr_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            print(f"  depth {_iso_d}:  {total_trades:>5,} trades  {_wr_pct:>5.1f}% WR  ${total_pnl:>10,.2f} PnL")
        else:
            for line in report_lines:
                print(line)

            # Save report to checkpoints (for analytics suite) + reports/ (for sharing)
            _report_name = 'oos_report.txt' if oos_mode else 'is_report.txt'
            report_path = os.path.join(_out_dir, _report_name)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines) + '\n')
            # Copy to reports/ directory for git-tracked sharing
            os.makedirs(_reports_out, exist_ok=True)
            _share_path = os.path.join(_reports_out, _report_name)
            with open(_share_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines) + '\n')
            print(f"  Report saved to {report_path}")
            print(f"  Shareable copy: {_share_path}")

            # ── 6a-bis. Save trade replays (per-bar price + state for I-MR analysis) ──
            if _trade_replays is not None and _trade_replays:
                _replay_mode = 'oos' if oos_mode else 'is'
                _replay_dir = os.path.join('reports', 'trade_replays')
                os.makedirs(_replay_dir, exist_ok=True)
                _replay_path = os.path.join(_replay_dir, f'{_replay_mode}_replays.json')
                try:
                    import json as _rj
                    with open(_replay_path, 'w') as _rf:
                        _rj.dump(_trade_replays, _rf)
                    print(f"  Trade replays: {_replay_path} ({len(_trade_replays)} trades)")
                except Exception as _re:
                    print(f"  Trade replay save failed: {_re}")

            # ── 6b. Append to run history (persistent cross-run comparison) ──────
            try:
                import datetime as _hist_dt
                import subprocess as _hist_sp
                _hist_path = os.path.join('reports', 'run_history.csv')
                _hist_exists = os.path.exists(_hist_path)
                _mode_label = 'OOS' if oos_mode else 'IS'
                try:
                    _git_hash = _hist_sp.check_output(
                        ['git', 'rev-parse', '--short', 'HEAD'],
                        stderr=_hist_sp.DEVNULL).decode().strip()
                except Exception:
                    _git_hash = 'unknown'
                _ee_h = _exec_engine.exit_engine
                _hl = getattr(_ee_h, 'envelope_half_life_bars', 0)
                _gb = getattr(_ee_h, 'giveback_pct', 0)
                _wr = round(total_wins / total_trades * 100, 1) if total_trades else 0
                _avg_pnl = round(total_pnl / total_trades, 2) if total_trades else 0
                _correct_dir_pct = round(len(tp_recs) / n_traded * 100, 1) if n_traded else 0
                _reversed_pct = round(len(reversed_) / len(tp_recs) * 100, 1) if tp_recs else 0
                _avg_capture = round(sum(r['capture_rate'] for r in tp_recs) / len(tp_recs) * 100, 1) if tp_recs else 0
                _avg_target_cap = round(sum(r.get('target_capture', 0) for r in tp_recs) / len(tp_recs) * 100, 1) if tp_recs else 0
                _worst_dip = _worst_intraday_dip if oos_mode else 0
                _max_dd = _cumul_max_dd if oos_mode else 0
                # Gross profit / gross loss
                _gross_profit = round(sum(r['actual_pnl'] for r in oracle_trade_records if r['actual_pnl'] > 0), 2)
                _gross_loss = round(abs(sum(r['actual_pnl'] for r in oracle_trade_records if r['actual_pnl'] < 0)), 2)
                # Exit quality bucket blends (% of correct-dir trades)
                _n_tp = len(tp_recs) if tp_recs else 1  # avoid div/0
                _too_early_blend = round(len(too_early) / _n_tp * 100, 1) if tp_recs else 0
                _too_late_blend = round(len(too_late) / _n_tp * 100, 1) if tp_recs else 0
                # Counter-trend scalps (% of all trades)
                _counter_trend_pct = round(len(fp_counter_scalps) / n_traded * 100, 1) if n_traded else 0
                # Average hold time in minutes (hold_bars × 15s per bar / 60)
                _all_holds = [r.get('hold_bars', 0) for r in oracle_trade_records]
                _avg_hold_min = round(sum(_all_holds) / len(_all_holds) * 15 / 60, 1) if _all_holds else 0
                # Macro → micro: profit first, then grosses, then WR, then detail
                _hist_cols = ['timestamp', 'git_hash', 'mode',
                              'total_pnl', 'profit_factor', 'gross_profit', 'gross_loss',
                              'trades', 'win_rate', 'avg_pnl',
                              'correct_dir_pct', 'counter_trend_pct', 'reversed_pct',
                              'avg_capture_pct', 'avg_target_capture_pct', 'too_early_pct', 'too_late_pct',
                              'avg_hold_min',
                              'worst_dip', 'max_dd',
                              'halflife', 'giveback_pct',
                              'wrong_dir_pnl', 'scalp_pnl', 'left_on_table']
                _hist_row = [
                    _hist_dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    _git_hash,
                    _mode_label,
                    round(total_pnl, 2),
                    round(_gross_profit / _gross_loss, 2) if _gross_loss > 0 else 999.0,
                    _gross_profit,
                    _gross_loss,
                    total_trades,
                    _wr,
                    _avg_pnl,
                    _correct_dir_pct,
                    _counter_trend_pct,
                    _reversed_pct,
                    _avg_capture,
                    _avg_target_cap,
                    _too_early_blend,
                    _too_late_blend,
                    _avg_hold_min,
                    round(_worst_dip, 2),
                    round(_max_dd, 2),
                    round(_hl, 1),
                    round(_gb * 100),
                    round(abs(_gw_pnl), 2),
                    round(_cs_pnl, 2),
                    round(left_on_table_val, 2),
                ]
                with open(_hist_path, 'a', newline='') as _hf:
                    import csv as _hist_csv
                    _hw = _hist_csv.writer(_hf)
                    if not _hist_exists:
                        _hw.writerow(_hist_cols)
                    _hw.writerow(_hist_row)
                print(f"  Run history appended: {_hist_path}")
            except Exception as _he:
                print(f"  Run history append failed: {_he}")

            # ── 6c. Run trade analytics suite (t-tests, ANOVA, OLS, logistic, capture) ──
            print("  Running trade analytics suite...", flush=True)
            _trade_log_path = os.path.join(_out_dir,
                'oos_trade_log.csv' if oos_mode else 'oracle_trade_log.csv')
            if os.path.exists(_trade_log_path):
                try:
                    from training.trade_analytics import run_trade_analytics
                    _analytics_text = run_trade_analytics(_trade_log_path, report_path)
                    # Also save standalone
                    _analytics_path = os.path.join(_out_dir,
                        'oos_analytics.txt' if oos_mode else 'trade_analytics.txt')
                    with open(_analytics_path, 'w', encoding='utf-8') as _af:
                        _af.write(_analytics_text)
                    print(f"  Trade analytics saved: {_analytics_path}")
                except Exception as _ae:
                    print(f"  Trade analytics failed: {_ae}")

        # ── 7. Save compact run snapshot (current vs _old for LLM comparison) ───
        if not oos_mode and not _analysis_mode:
            import json as _snap_json
            import datetime as _snap_dt
            _win_decay  = [r.get('exit_decay_score', 0.0) for r in oracle_trade_records if r.get('result') == 'WIN']
            _loss_decay = [r.get('exit_decay_score', 0.0) for r in oracle_trade_records if r.get('result') == 'LOSS']
            _snap = {
                'timestamp':      _snap_dt.datetime.now().isoformat(timespec='seconds'),
                'trades':         total_trades,
                'win_rate':       round(total_wins / total_trades * 100, 1) if total_trades else 0,
                'total_pnl':      round(total_pnl, 2),
                'avg_pnl_trade':  round(total_pnl / total_trades, 2) if total_trades else 0,
                'ideal_pnl':      round(ideal_profit, 2),
                'capture_pct':    round(total_pnl / ideal_profit * 100, 2) if ideal_profit else 0,
                'fn_missed_pnl':  round(fn_potential_pnl, 2),
                'wrong_dir_pnl':  round(abs(fp_wrong_pnl), 2),
                'reversed_pnl':   round(reversed_loss_val, 2),
                'left_on_table':  round(left_on_table_val, 2),
                'n_reversed':     len(reversed_) if tp_recs else 0,
                'n_correct_dir':  len(tp_recs) if tp_recs else 0,
                'total_bars':     total_bars_processed,
                'bars_detected':  bars_with_detection,
                'bars_blind':     total_bars_processed - bars_with_detection,
                'bars_slot_blocked': bars_slot_blocked,
                'gate0_skip':     _exec_engine.get_skip_counts()['skip_headroom'],
                'gate1_skip':     _exec_engine.get_skip_counts()['skip_dist'],
                'gate2_skip':     _exec_engine.get_skip_counts()['skip_brain'],
                'gate3_skip':     _exec_engine.get_skip_counts()['skip_conviction'],
                'decay_win_avg':  round(sum(_win_decay) / len(_win_decay), 3) if _win_decay else 0,
                'decay_loss_avg': round(sum(_loss_decay) / len(_loss_decay), 3) if _loss_decay else 0,
                'depth_breakdown': {
                    str(d): {'n': _dw_cnt[d], 'avg': round(_dw_pnl[d] / _dw_cnt[d], 2)}
                    for d in sorted(_dw_cnt.keys())
                } if oracle_trade_records and 'entry_depth' in oracle_trade_records[0] else {},
            }
            _snap_path = os.path.join(_out_dir, 'run_snapshot.json')
            with open(_snap_path, 'w') as _sf:
                _snap_json.dump(_snap, _sf, indent=2)
            print(f"  Run snapshot saved: {_snap_path}")

    def _learn_oracle_directions(self, oracle_trade_records, oos_mode,
                                  brain_keys_before_oos=None,
                                  brain_dir_keys_before_oos=None):
        """Oracle direction learning — update pattern library biases from forward pass.

        Returns:
            defaultdict of per-template direction correction stats.
        """
        _dir_corrections = defaultdict(lambda: {
            'long_correct': 0, 'long_wrong': 0,
            'short_correct': 0, 'short_wrong': 0,
            'long_pnl': 0.0, 'short_pnl': 0.0,
            'signed_mfe_samples': [],
        })

        if oracle_trade_records:
            print(f"\n  Learning direction corrections from oracle{' (OOS refinement)' if oos_mode else ''}...")
            for rec in oracle_trade_records:
                tid = rec.get('template_id')
                if tid is None or tid == -1:
                    continue
                direction = rec.get('direction', '')
                oracle_label = rec.get('oracle_label', 0)
                actual_pnl = rec.get('actual_pnl', 0.0)
                oracle_mfe = rec.get('oracle_mfe', 0.0)
                oracle_mae = rec.get('oracle_mae', 0.0)
                acc = _dir_corrections[tid]

                oracle_says_long = oracle_label > 0
                oracle_says_short = oracle_label < 0
                we_went_long = direction == 'LONG'
                we_went_short = direction == 'SHORT'

                if we_went_long:
                    acc['long_pnl'] += actual_pnl
                    if oracle_says_long:
                        acc['long_correct'] += 1
                    elif oracle_says_short:
                        acc['long_wrong'] += 1
                if we_went_short:
                    acc['short_pnl'] += actual_pnl
                    if oracle_says_short:
                        acc['short_correct'] += 1
                    elif oracle_says_long:
                        acc['short_wrong'] += 1

                if oracle_label != 0:
                    signed_mfe = oracle_mfe if oracle_label > 0 else -oracle_mae
                    acc['signed_mfe_samples'].append({
                        'signed_mfe': signed_mfe,
                        'entry_depth': rec.get('entry_depth', 6),
                        'dmi_diff': rec.get('dmi_diff', 0.0),
                    })

            # Update pattern library with corrected biases
            _updated_count = 0
            _regression_count = 0
            for tid, acc in _dir_corrections.items():
                if tid not in self.pattern_library:
                    continue
                lib = self.pattern_library[tid]
                long_total = acc['long_correct'] + acc['long_wrong']
                short_total = acc['short_correct'] + acc['short_wrong']
                total_dir_trades = long_total + short_total

                if total_dir_trades >= 3:
                    fp_long_correct = acc['long_correct']
                    fp_short_correct = acc['short_correct']
                    fp_total_correct = fp_long_correct + fp_short_correct
                    if fp_total_correct > 0:
                        fp_long_bias = fp_long_correct / fp_total_correct
                        fp_short_bias = fp_short_correct / fp_total_correct
                    else:
                        fp_long_bias = 0.5
                        fp_short_bias = 0.5
                    orig_long = lib.get('long_bias', 0.5)
                    orig_short = lib.get('short_bias', 0.5)
                    new_long = 0.7 * fp_long_bias + 0.3 * orig_long
                    new_short = 0.7 * fp_short_bias + 0.3 * orig_short
                    total = new_long + new_short
                    if total > 0:
                        new_long /= total
                        new_short /= total
                    lib['long_bias'] = round(new_long, 4)
                    lib['short_bias'] = round(new_short, 4)
                    lib['direction_source'] = 'oracle_corrected'
                    _updated_count += 1

                if long_total >= 2 and short_total >= 2:
                    lib['long_avg_pnl'] = round(acc['long_pnl'] / long_total, 2)
                    lib['short_avg_pnl'] = round(acc['short_pnl'] / short_total, 2)

                # Signed MFE regression
                samples = acc['signed_mfe_samples']
                if len(samples) >= 15:
                    try:
                        from sklearn.linear_model import LinearRegression
                        X = np.array([[s['entry_depth'], s['dmi_diff']] for s in samples])
                        y = np.array([s['signed_mfe'] for s in samples])
                        reg = LinearRegression().fit(X, y)
                        lib['signed_mfe_coeff'] = reg.coef_.tolist()
                        lib['signed_mfe_intercept'] = float(reg.intercept_)
                        _regression_count += 1
                    except Exception:
                        pass

            print(f"  Direction corrections: {_updated_count} templates updated")
            print(f"  Signed MFE regression: {_regression_count} templates fitted")

            # Save updated library
            _lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
            with open(_lib_path, 'wb') as _f:
                pickle.dump(self.pattern_library, _f)
            print(f"  Updated pattern_library.pkl saved")

            # Save brain with direction-specific learning for live
            _brain_path = os.path.join(self.checkpoint_dir, 'pattern_forward_brain.pkl')
            self.brain.save(_brain_path)
            print(f"  Forward pass brain saved: {_brain_path}")
            print(f"    States: {len(self.brain.table)}, "
                  f"Direction pairs: {len(self.brain.dir_table)}")

        # OOS: save brain with updated weights but drop new patterns
        if oos_mode and oracle_trade_records:
            _new_keys = set(self.brain.table.keys()) - brain_keys_before_oos
            _new_dir_keys = set(self.brain.dir_table.keys()) - brain_dir_keys_before_oos
            for k in _new_keys:
                del self.brain.table[k]
            for k in _new_dir_keys:
                del self.brain.dir_table[k]
            print(f"\n  OOS brain cleanup: dropped {len(_new_keys)} new patterns, "
                  f"{len(_new_dir_keys)} new dir entries")
            print(f"  Retained weights on {len(self.brain.table)} pre-existing patterns")
            _brain_path = os.path.join(self.checkpoint_dir, 'pattern_forward_brain.pkl')
            self.brain.save(_brain_path)
            print(f"  OOS brain saved: {_brain_path}")
            print(f"    States: {len(self.brain.table)}, "
                  f"Direction pairs: {len(self.brain.dir_table)}")

        return _dir_corrections

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
            if not oos_result.top_iterations:
                continue

            iter_res = oos_result.top_iterations[0]
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

    # ------------------------------------------------------------------
    #  DEPTH ISOLATION ANALYSIS
    # ------------------------------------------------------------------
    def run_depth_analysis(self, data_source: str,
                           start_date: str = None, end_date: str = None,
                           oos_mode: bool = False):
        """
        Run forward pass once per depth for isolated performance analysis.
        Each depth trades independently — no capital blocking between depths.
        Brain state carries across depth passes (shared learning).
        """
        _DEPTH_LABELS = {
            1: '4h+', 2: '1h', 3: '15m', 4: '5m', 5: '1m', 6: '30s',
            7: '15s', 8: '15s', 9: '5s', 10: '5s', 11: '1s', 12: '1s',
        }
        print("\n" + "=" * 80)
        print("DEPTH ISOLATION ANALYSIS")
        print("Each depth trades independently — no capital blocking from other depths.")
        print("=" * 80)

        self._analysis_mode = True
        _results = {}

        for d in range(1, 13):
            self._depth_only = d
            self._position = None  # fresh per depth
            self.run_forward_pass(
                data_source, start_date=start_date, end_date=end_date,
                oos_mode=oos_mode)
            summary = getattr(self, '_fp_summary', {})
            if summary and summary.get('total_trades', 0) > 0:
                _results[d] = {**summary}

        self._depth_only = None
        self._analysis_mode = False

        # ── Comparison table ────────────────────────────────────────────
        if not _results:
            print("\n  No depth produced any trades.")
            return

        print("\n" + "=" * 80)
        print("DEPTH ISOLATION RESULTS")
        print("=" * 80)
        print(f"  {'Depth':<12} {'TF':<5} {'Trades':>7} {'WR%':>6} "
              f"{'Total PnL':>12} {'Avg/trade':>10} {'Days':>5}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*12} {'-'*10} {'-'*5}")
        for d in sorted(_results.keys()):
            r = _results[d]
            n = r['total_trades']
            pnl = r['total_pnl']
            wr = r['win_rate'] * 100
            avg = pnl / n if n > 0 else 0
            tf = _DEPTH_LABELS.get(d, '?')
            days = r.get('n_days', 0)
            flag = "  <- TOP" if avg > 50 else ("  <- BLEED" if avg < -10 else "")
            print(f"  depth {d:<4} {tf:<5} {n:>7,} {wr:>5.1f}% "
                  f"${pnl:>10,.2f} ${avg:>9.2f} {days:>5}{flag}")

        _total_trades = sum(r['total_trades'] for r in _results.values())
        _total_pnl = sum(r['total_pnl'] for r in _results.values())
        _total_wins = sum(r['total_trades'] * r['win_rate'] for r in _results.values())
        _comb_wr = (_total_wins / _total_trades * 100) if _total_trades > 0 else 0
        _comb_avg = _total_pnl / _total_trades if _total_trades > 0 else 0
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*12} {'-'*10} {'-'*5}")
        print(f"  {'COMBINED':<12} {'':5} {_total_trades:>7,} {_comb_wr:>5.1f}% "
              f"${_total_pnl:>10,.2f} ${_comb_avg:>9.2f}")
        print(f"\n  NOTE: Combined total exceeds normal run because depths trade simultaneously.")
        print(f"  Use this to identify which depths to KEEP vs FILTER in production.\n")

    def run_strategy_selection(self):
        """
        Phase 6: Grade templates on OOS blind validation results. Assign tiers.
        """
        print("\n" + "="*80)
        print("PHASE 6: STRATEGY SELECTION (grading on OOS results)")
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

        # Read OOS trade log — this is the blind validation data
        import csv as _strat_csv
        _oos_csv_path = os.path.join(self.checkpoint_dir, 'oos_trade_log.csv')
        if not os.path.exists(_oos_csv_path):
            print("ERROR: oos_trade_log.csv not found. Run OOS forward pass first.")
            return
        with open(_oos_csv_path, newline='', encoding='utf-8') as _sf:
            _oos_trades = list(_strat_csv.DictReader(_sf))
        print(f"  Loaded {len(_oos_trades)} OOS trades for grading")

        # Group trades by template_id
        history_by_template = defaultdict(list)
        for row in _oos_trades:
            tid = row.get('template_id', '')
            # template_id may be int or str depending on CSV
            try:
                tid = int(tid)
            except (ValueError, TypeError):
                pass
            history_by_template[tid].append(float(row.get('actual_pnl', 0)))

        tier1_templates = []
        report_data = []

        print(f"\nAnalyzing {len(self.pattern_library)} strategies against OOS results...")

        for tid in self.pattern_library:
            pnls = history_by_template.get(tid, [])
            total = len(pnls)

            if not pnls:
                sharpe = 0.0
                max_dd = 0.0
                win_rate = 0.0
                risk_score = 1.0  # High risk if unseen in OOS
                avg_pnl = 0.0
            else:
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

                total_gain = sum(wins)
                dd_ratio = abs(max_dd) / (total_gain + 1e-6) if total_gain > 0 else 1.0
                loss_ratio = abs(avg_loss) / (avg_win + 1e-6)

                risk_score = (
                    0.3 * (1.0 - win_rate) +
                    0.3 * min(loss_ratio, 2.0) +
                    0.2 * min(max_consec_loss / 10.0, 1.0) +
                    0.2 * min(dd_ratio, 1.0)
                )

            # Tier assignment based on OOS performance (blind validation)
            tier = 3  # Default: UNPROVEN (not seen in OOS)
            if total >= 20 and win_rate > 0.45 and avg_pnl > 0 and sharpe > 0.3:
                tier = 1  # PRODUCTION — proven on unseen data
            elif total >= 10 and win_rate > 0.40 and avg_pnl > 0:
                tier = 2  # PROMISING
            elif total >= 10 and (win_rate < 0.35 or avg_pnl < 0):
                tier = 4  # TOXIC — failed blind validation

            _lib = self.pattern_library.get(tid, {})
            _sname = _lib.get('semantic_name', '') or ''
            if (not _sname or _sname == 'Unknown') and _lib.get('centroid') is not None:
                from core.fractal_clustering import generate_semantic_name
                _sname = generate_semantic_name(_lib['centroid'])
            _sname = _sname or 'Unknown'
            report_data.append({
                'id': tid,
                'semantic': _sname,
                'tier': tier,
                'trades': total,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'pnl': sum(pnls),
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

        # Sort report: PnL descending first, then Sharpe descending as tiebreaker
        report_data.sort(key=lambda x: (-x['pnl'], -x['sharpe']))

        # Build report
        rpt = []
        rpt.append("")
        rpt.append("STRATEGY PERFORMANCE REPORT")
        header = f"{'ID':<10} | {'Playbook':<28} | {'Tier':<4} | {'Trades':<6} | {'Win%':<5} | {'Sharpe':<6} | {'PnL':<10} | {'MaxDD':<10} | {'Risk':<5}"
        rpt.append(header)
        rpt.append("-" * 115)
        for r in report_data:
            rpt.append(f"{r['id']:<10} | {r['semantic']:<28} | {r['tier']:<4} | {r['trades']:<6} | {r['win_rate']*100:5.1f} | {r['sharpe']:6.2f} | ${r['pnl']:<9.2f} | ${r['max_dd']:<9.2f} | {r['risk']:.2f}")

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
        band_roots = 0
        struct_roots = 0
        for tid, data in tier1_templates:
            centroid = data['centroid']
            if centroid[-1] > 0.5:
                band_roots += 1
            else:
                struct_roots += 1

        rpt.append("")
        rpt.append("ANCESTRY ANALYSIS (Tier 1):")
        rpt.append(f"  Band-backed: {band_roots}")
        rpt.append(f"  Structure-backed: {struct_roots}")

        # ── PARETO ANALYSIS ──────────────────────────────────────────────────
        # Read oracle_trade_log.csv and find the 20% of trades driving 80% of profit.
        # Dimensions: template, direction, oracle_label, time-of-day.
        import csv as _csv
        import datetime as _dt
        # Pareto analysis on OOS trades (same data used for tier grading)
        oracle_csv = _oos_csv_path
        if os.path.exists(oracle_csv):
            try:
                rows = _oos_trades  # already loaded above

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
        report_path = os.path.join(self.checkpoint_dir, 'strategy_report.txt')
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

        # ===================================================================
        # PHASE 1: Discovery (with checkpoint/resume)
        # ===================================================================
        manifest = None
        templates = None

        if ckpt.has_discovery():
            cached_manifest, cached_levels = ckpt.load_discovery()
            if cached_manifest is not None:
                print(f"\n[RESUME] Phase 1 Discovery: Loaded {len(cached_manifest)} patterns "
                      f"from {len(cached_levels)} completed levels")
                manifest = cached_manifest

        # --seeds: auto-swing replaces discovery; manual seeds filter discovery
        _seed_path = getattr(self.config, 'seeds', None)
        _seed_thresholds = None
        _is_auto_swing = False
        if _seed_path:
            from training.seed_loader import is_auto_swing_format
            _is_auto_swing = is_auto_swing_format(_seed_path)
            if _is_auto_swing:
                # Auto-swing seeds REPLACE Phase 1 discovery entirely
                from training.seed_loader import load_auto_swing_as_manifest
                manifest = load_auto_swing_as_manifest(
                    _seed_path, self.config.data,
                    timeframe='15s', depth=8,
                )
                # Save to checkpoint for reuse by --forward-pass
                completed_levels = list(set(p.timeframe for p in manifest))
                ckpt.save_discovery(manifest, completed_levels)
            else:
                # Manual seeds: compute quality thresholds (applied AFTER discovery)
                from training.seed_loader import compute_seed_thresholds
                _seed_tag = getattr(self.config, 'seed_tag', None)
                _seed_thresholds = compute_seed_thresholds(_seed_path, tag_filter=_seed_tag)

        if manifest is None:
            print("\nPhase 1: Discovery — Fractal Top-Down Scan...")
            ckpt.update_phase('discovery', 'in_progress')
            if self.dashboard_queue:
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Discover',
                                          'step': 'DISCOVERY  lvl 0/13', 'pct': 0})

            # Check for partial resume (some levels done)
            partial_manifest, partial_levels = ckpt.load_discovery()

            _train_end = getattr(self.config, 'train_end', None)
            if _train_end:
                print(f"  Out-of-sample guard: training data capped at {_train_end}")

            # Progress callback wrapping both checkpoint save and dashboard update
            _n_tf_levels = 13  # total TF hierarchy depth
            def _discovery_cb(lvl, tf, patterns, levels):
                ckpt.save_discovery_level(patterns, levels)
                if self.dashboard_queue:
                    _done = len(levels) if levels else lvl + 1
                    self.dashboard_queue.put({
                        'type': 'PHASE_PROGRESS', 'phase': 'Discover',
                        'step': f'DISCOVERY  lvl {_done}/{_n_tf_levels}',
                        'pct': round(_done / _n_tf_levels * 100, 1)})

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
        roche = sum(1 for p in manifest if p.pattern_type == 'BAND_REVERSAL')
        struct = sum(1 for p in manifest if p.pattern_type == 'MOMENTUM_BREAK')
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
                r = sum(1 for p in manifest if p.timeframe == tf and p.pattern_type == 'BAND_REVERSAL')
                s = sum(1 for p in manifest if p.timeframe == tf and p.pattern_type == 'MOMENTUM_BREAK')
                print(f"    [{tf:>4s}] {count:>7,} (R:{r:,} S:{s:,})")
            print(f"  By depth: {dict(sorted(depth_counts.items()))}")

        # Apply seed-guided quality filter (--seeds)
        if _seed_thresholds and manifest:
            from training.seed_loader import filter_manifest_by_thresholds
            manifest = filter_manifest_by_thresholds(manifest, _seed_thresholds)
            # Force re-clustering since the manifest changed
            templates = None
            if ckpt.has_templates():
                os.remove(ckpt.templates_path)
                print("  [Seed Filter] Cleared cached templates (manifest changed)")

        # ===================================================================
        # PHASE 2: Clustering (with checkpoint)
        # ===================================================================
        if ckpt.has_templates():
            templates = ckpt.load_templates()
            if templates is not None:
                print(f"\n[RESUME] Phase 2 Clustering: Loaded {len(templates)} templates from checkpoint")

        if templates is None:
            print("\nPhase 2 Clustering: Generating Physically Tight Templates...")
            ckpt.update_phase('clustering', 'in_progress')
            if self.dashboard_queue:
                self.dashboard_queue.put({'type': 'PHASE_PROGRESS', 'phase': 'Cluster',
                                          'step': 'CLUSTERING', 'pct': 0})

            # Load shape primitives if --primitives flag and .pkl exists
            shape_primitives = None
            if getattr(self, '_use_primitives', False):
                _sp_path = os.path.join(self.checkpoint_dir, 'shape_primitives.pkl')
                if os.path.exists(_sp_path):
                    import pickle as _sp_pkl
                    with open(_sp_path, 'rb') as _sp_f:
                        shape_primitives = _sp_pkl.load(_sp_f)
                    print(f"  Loaded shape primitives: {len(shape_primitives.primitives)} primitives from {_sp_path}")
                else:
                    print(f"  --primitives: no shape_primitives.pkl found, using pure K-Means++")

            n_initial = max(10, len(manifest) // INITIAL_CLUSTER_DIVISOR)
            print(f"  Initial clusters: {n_initial} (from {len(manifest)} patterns / {INITIAL_CLUSTER_DIVISOR})")

            _use_lb = getattr(self, '_use_lookback', False)
            clustering_engine = FractalClusteringEngine(n_clusters=n_initial, max_variance=0.5,
                                                        use_lookback=_use_lb)
            templates = clustering_engine.create_templates(manifest, shape_primitives=shape_primitives)
            print(f"  Condensed {len(manifest)} raw patterns into {len(templates)} Tight Templates.")

            # Shape calibration: derive exit params per template from member segments
            if getattr(self, '_use_shapes', False):
                print("  Shape-aware exit calibration...")
                clustering_engine.calibrate_template_shapes(templates)

            # Save scaler for IS/OOS forward passes
            import pickle as _pickle
            scaler_path = os.path.join(self.checkpoint_dir, 'clustering_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                _pickle.dump(clustering_engine.scaler, f)
            print(f"  Saved clustering scaler to {scaler_path}")

            ckpt.save_templates(templates)

        # Template inspection (feedback loop) — when --inspect-templates or --seeds
        if getattr(self.config, 'inspect_templates', False) or getattr(self.config, 'seeds', None):
            from training.seed_loader import inspect_templates
            _inspect_path = os.path.join('reports', 'template_inspection.txt')
            inspect_templates(templates, output_path=_inspect_path)

        # Apply template feedback (KEEP/DROP + direction overrides)
        _feedback_path = getattr(self.config, 'template_feedback', None)
        if _feedback_path and os.path.isfile(_feedback_path):
            from training.seed_loader import load_template_feedback
            _fb = load_template_feedback(_feedback_path)

            _before = len(templates)
            templates = [t for t in templates if _fb.get(t.template_id, {}).get('action', 'KEEP') != 'DROP']
            _dropped = _before - len(templates)

            # Apply direction overrides to long_bias/short_bias
            _overridden = 0
            for t in templates:
                fb = _fb.get(t.template_id, {})
                override = fb.get('direction_override', 'AUTO')
                if override == 'LONG':
                    t.long_bias = 1.0
                    t.short_bias = 0.0
                    _overridden += 1
                elif override == 'SHORT':
                    t.long_bias = 0.0
                    t.short_bias = 1.0
                    _overridden += 1

            if _dropped or _overridden:
                print(f"\n  [Feedback] Applied: {_dropped} dropped, {_overridden} direction overrides")
                print(f"  [Feedback] Remaining templates: {len(templates)}")
                ckpt.save_templates(templates)  # re-save with feedback applied

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
        print(f"  Iterations per template: {INDIVIDUAL_OPTIMIZATION_ITERATIONS}")

        ckpt.update_phase('optimization', 'in_progress')

        # We need a clustering engine for fission checks (may not exist if we resumed past Phase 2 Clustering)
        try:
            clustering_engine
        except NameError:
            n_initial = max(10, len(manifest) // INITIAL_CLUSTER_DIVISOR)
            _use_lb = getattr(self, '_use_lookback', False)
            clustering_engine = FractalClusteringEngine(n_clusters=n_initial, max_variance=0.5,
                                                        use_lookback=_use_lb)

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
                _total_known = processed_count + len(current_batch) + total_in_queue
                print(f"\n  Batch {batch_number}: processing {len(current_batch)} templates ({total_in_queue} remaining in queue)...")
                if self.dashboard_queue:
                    self.dashboard_queue.put({
                        'type': 'PHASE_PROGRESS', 'phase': 'Optimize',
                        'step': f'OPTIMIZE  tmpl {processed_count}/{_total_known}',
                        'pct': round(processed_count / _total_known * 100, 1) if _total_known else 0})

                tasks = []
                for tmpl in current_batch:
                    # Pass arguments as a dictionary to _process_template_job
                    tasks.append({
                        'template': tmpl,
                        'clustering_engine': clustering_engine,
                        'iterations': INDIVIDUAL_OPTIMIZATION_ITERATIONS,
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
                        _sem = getattr(tmpl, 'semantic_name', 'Unknown')
                        print(f"    [{processed_count}] Template {tmpl_id} ({_sem}): DONE ({member_count} members) -> PnL: ${val_pnl:.2f} | {timing}")

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

        # Save pattern library for IS/OOS forward passes
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
        best_params, best_sharpe = _optimize_template_task((None, subset, INDIVIDUAL_OPTIMIZATION_ITERATIONS, self.param_generator, self.asset.point_value))
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
            'semantic_name': getattr(template, 'semantic_name', 'Unknown'),
            # Oracle exit calibration -- pattern's own price-breathing stats (in ticks)
            # regression_sigma_ticks: residual std of per-cluster OLS (MFE ~ |z|) -> trail = × 1.1
            # 0.0 means _aggregate_oracle_intelligence() didn't have enough members
            'mean_mfe_ticks':          getattr(template, 'mean_mfe_ticks',          0.0),
            'mean_mae_ticks':          getattr(template, 'mean_mae_ticks',          0.0),
            'p75_mfe_ticks':           getattr(template, 'p75_mfe_ticks',           0.0),
            'p25_mae_ticks':           getattr(template, 'p25_mae_ticks',           0.0),
            'p95_mae_ticks':           getattr(template, 'p95_mae_ticks',           0.0),
            'mae_std_ticks':           getattr(template, 'mae_std_ticks',           0.0),
            'risk_variance':           getattr(template, 'risk_variance',           0.0),
            'regression_sigma_ticks':  getattr(template, 'regression_sigma_ticks',  0.0),
            # Per-template exit timescale (bars where MFE peaks — halflife anchor)
            'avg_mfe_bar':             getattr(template, 'avg_mfe_bar',             0.0),
            'p75_mfe_bar':             getattr(template, 'p75_mfe_bar',             0.0),
            'discovery_tf_seconds':    getattr(template, 'discovery_tf_seconds',    15.0),
            # Per-cluster regression model coefficients (14D scaled feature space)
            # mfe_coeff @ live_scaled_features + mfe_intercept  -> predicted MFE in price pts
            # sigmoid(dir_coeff @ live_scaled_features + dir_intercept) -> P(LONG)
            # None when cluster had < 15 members for stable fitting
            'mfe_coeff':     getattr(template, 'mfe_coeff',     None),
            'mfe_intercept': getattr(template, 'mfe_intercept', 0.0),
            'dir_coeff':     getattr(template, 'dir_coeff',     None),
            'dir_intercept': getattr(template, 'dir_intercept', 0.0),
            # Shape-aware exit calibration (from --shapes flag)
            'shape_giveback_pct':      getattr(template, 'shape_giveback_pct',      None),
            'shape_delay_bars':        getattr(template, 'shape_delay_bars',        None),
            'shape_envelope_hl_mult':  getattr(template, 'shape_envelope_hl_mult',  None),
            'shape_peak_bar':          getattr(template, 'shape_peak_bar',          None),
            'shape_dominant':          getattr(template, 'shape_dominant',          None),
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

    def _launch_popup(self, mode='is'):
        """Launch ProgressPopup in a daemon thread (same widget as live).
        Only one popup per session — subsequent calls update the title."""
        if getattr(self, '_popup_shared', None) is not None:
            # Already running — just update the mode label
            self._popup_shared['mode'] = mode
            return
        import tkinter as tk
        shared = {'mode': mode}
        self._popup_shared = shared
        q = self.dashboard_queue

        def _run():
            root = tk.Tk()
            popup = ProgressPopup(root, q, shared_state=shared)
            root.title(f"Bayesian-AI  {mode.upper()} Training")
            root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(),))
            try:
                root.mainloop()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass
            del popup, root
            import gc; gc.collect()

        t = threading.Thread(target=_run, daemon=True, name='TrainerPopup')
        t.start()

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
            python training/trainer.py --sweep-params
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
                print(f"  python training/trainer.py --forward-pass "
                      f"--min-tier {best_overall['min_tier']} "
                      f"--bias-threshold {best_dir['bias_thresh']:.2f} "
                      f"--dmi-threshold {best_dir['dmi_thresh']:.1f}")
            print()
        elif live_hits:
            best = live_hits[0]
            print(f"\nRECOMMENDED NEXT RUN:")
            print(f"  python training/trainer.py --forward-pass "
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
            if fp['total_trades'] and 'pct_correct' in fp:
                lines.append(
                    f"    Correct direction: {fp['pct_correct']:4.1f}%  |  "
                    f"Counter-trend scalps: {fp['pct_counter_scalp']:4.1f}%  |  "
                    f"Wrong: {fp['pct_genuinely_wrong']:4.1f}%  |  "
                    f"Noise: {fp['pct_noise']:4.1f}%"
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
            lines.append(f"    #3 leak  genuinely wrong:   ${fp['wrong_dir_loss']:>12,.2f}   ({fp['wrong_dir_loss']/ideal*100:.1f}%)")
            if fp.get('counter_scalp_profit', 0) > 0:
                lines.append(f"    banked   counter-trend:    +${fp['counter_scalp_profit']:>11,.2f}   (micro-peak scalps)")

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


def _print_oos_comparison(oos1: dict, oos2: dict):
    """Print OOS1 vs OOS2 comparison showing tier preference impact."""
    print("\n" + "=" * 80)
    print("  PLAYBOOK TIEBREAKER COMPARISON: OOS1 (blind) vs OOS2 (tier preference)")
    print("=" * 80)
    _metrics = [
        ('Trades',   'total_trades', '{:>7,}'),
        ('Win Rate',  'win_rate',    '{:>6.1%}'),
        ('Total PnL', 'total_pnl',  '${:>10,.2f}'),
        ('$/trade',   None,         '${:>8,.2f}'),
    ]
    print(f"  {'Metric':<14}  {'OOS1':>12}  {'OOS2':>12}  {'Delta':>12}")
    print(f"  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}")
    for label, key, fmt in _metrics:
        if key is None:  # $/trade computed
            t1, t2 = oos1.get('total_trades', 1) or 1, oos2.get('total_trades', 1) or 1
            v1 = oos1.get('total_pnl', 0) / t1
            v2 = oos2.get('total_pnl', 0) / t2
        else:
            v1, v2 = oos1.get(key, 0), oos2.get(key, 0)
        d = v2 - v1
        f1 = fmt.format(v1)
        f2 = fmt.format(v2)
        # Delta formatting
        if isinstance(v1, float) and 'rate' in (key or ''):
            fd = f'{d:>+.1%}'
        elif '$' in fmt:
            fd = f'${d:>+,.2f}'
        else:
            fd = f'{d:>+,}'
        print(f"  {label:<14}  {f1:>12}  {f2:>12}  {fd:>12}")
    _pnl_d = oos2.get('total_pnl', 0) - oos1.get('total_pnl', 0)
    verdict = "HELPS" if _pnl_d > 0 else "HURTS" if _pnl_d < 0 else "NEUTRAL"
    print(f"\n  Verdict: Tier preference tiebreaker {verdict} (${_pnl_d:+,.2f})")
    print("=" * 80)

    # Save to file
    import datetime as _cdt
    _comp_path = os.path.join('reports', 'playbook_comparison.txt')
    with open(_comp_path, 'w', encoding='utf-8') as _cf:
        _cf.write(f"Playbook Tiebreaker Comparison ({_cdt.datetime.now():%Y-%m-%d %H:%M})\n")
        _cf.write(f"OOS1 PnL: ${oos1.get('total_pnl', 0):,.2f}  "
                  f"OOS2 PnL: ${oos2.get('total_pnl', 0):,.2f}  "
                  f"Delta: ${_pnl_d:+,.2f}  Verdict: {verdict}\n")
    print(f"  Saved: {_comp_path}")


def main():
    """Single entry point - command line interface"""
    from core.keep_awake import keep_awake
    _awake_ctx = keep_awake()
    _awake_ctx.__enter__()

    parser = argparse.ArgumentParser(
        description="Bayesian-AI Training Orchestrator (Pattern-Adaptive)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', default=os.path.join("DATA", "ATLAS"), help="Path to ATLAS root, single TF directory, or parquet file")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")
    parser.add_argument('--exploration-mode', action='store_true', help="Enable unconstrained exploration mode")
    parser.add_argument('--fresh', action='store_true', help="Wipe checkpoints + full pipeline: Train → IS → OOS → Strategy → OOS2 verify")
    parser.add_argument('--forward-pass', action='store_true', help="IS → OOS → Strategy → OOS2 verify (reuse existing checkpoints, skip training)")
    parser.add_argument('--depth-iso', action='store_true',
                        help="Run per-depth isolation analysis: forward pass once per depth (1-12), "
                             "no capital blocking between depths. Prints comparison table at end.")
    parser.add_argument('--oos', action='store_true',
                        help="Blind out-of-sample simulation: frozen templates, separate oos_trade_log.csv/oos_report.txt, "
                             "training depth_weights.json preserved. Implies --forward-pass. "
                             "Pair with --forward-start YYYYMMDD to slice the OOS window.")
    parser.add_argument('--oos-chain', action='store_true',
                        help="Run full OOS chain: OOS1 (blind) → Strategy → OOS2 (tier pref) → OOS3 (BarProcessor). "
                             "Skips IS. Uses existing checkpoints.")
    parser.add_argument('--oos3', action='store_true',
                        help="OOS3 only: bar-by-bar via BarProcessor (live mode validation). "
                             "Uses existing checkpoints + warmed brain. Saves live_brain.pkl.")
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
    parser.add_argument('--trade-start', type=str, default=None, metavar='YYYYMMDD',
                        help="First day to allow trades (earlier days = warmup/context only). "
                             "E.g. --trade-start 20250201 uses Jan as context, trades from Feb.")
    parser.add_argument('--forward-end', type=str, default=None, metavar='YYYYMMDD',
                        help="Last day to include in forward pass (inclusive, e.g. 20260209)")
    parser.add_argument('--live-prep', action='store_true', default=False,
                        help="Auto-set --forward-end to last Friday (most recent weekend cutoff). "
                             "Use before going live: trains on all data up to last week, "
                             "keeps the final week clean for live trading.")
    parser.add_argument('--bias-threshold', type=float, default=None,
                        help="Oracle bias threshold for direction lock (default 0.55). Lower = more oracle-locked trades.")
    parser.add_argument('--dmi-threshold', type=float, default=None,
                        help="Min |dmi_diff| required to use DMI signal (default 0.0 = any non-zero DMI counts).")
    parser.add_argument('--sweep-params', action='store_true',
                        help="Post-hoc DOE: sweep filter combinations on oracle_trade_log.csv and rank by net PnL")
    parser.add_argument('--strategy-report', action='store_true', help="Run Phase 6 strategy selection (requires OOS trade log)")
    parser.add_argument('--forward-data', type=str, default=None, metavar='PATH',
                        help="Custom data path for forward pass (skips auto-OOS chain)")
    parser.add_argument('--skip-oos', action='store_true',
                        help="Skip auto-chained OOS forward pass after IS + Strategy")
    parser.add_argument('--ping-pong', action='store_true',
                        help="Continuous wave-riding: flip direction after each exit "
                             "(uses belief conviction + TF-scaled params from exited trade)")
    parser.add_argument('--pp-conviction', type=float, default=0.55,
                        help="Ping-pong: min belief conviction to flip (default 0.55)")
    parser.add_argument('--pp-sl', type=int, default=0,
                        help="Ping-pong: override SL ticks (0=inherit from exited trade)")
    parser.add_argument('--pp-tp', type=int, default=0,
                        help="Ping-pong: override TP ticks (0=inherit)")
    parser.add_argument('--pp-trail', type=int, default=0,
                        help="Ping-pong: override trail ticks (0=inherit)")
    parser.add_argument('--slippage', type=float, default=0.0,
                        help="Random fill slippage per trade in ticks (e.g., 2). "
                             "Each trade PnL gets uniform(-N, +N) * tick_value noise.")
    parser.add_argument('--primitives', action='store_true',
                        help="Use shape primitives for K-Means init (requires checkpoints/*_primitives.pkl)")
    parser.add_argument('--lookback', action='store_true',
                        help="22D clustering: append 10-bar lookback geometry (6D) to 16D features")
    parser.add_argument('--shapes', action='store_true',
                        help="Shape-aware exits: calibrate giveback/envelope per template from member segment shapes")
    parser.add_argument('--seeds', nargs='?', const='AUTO', default=None, metavar='PATH',
                        help="Path to seed JSON file (manual trades). Replaces Phase 1 discovery "
                             "with seed-derived PatternEvents. Use bare --seeds to auto-detect "
                             "latest file in DATA/regime_seeds/.")
    parser.add_argument('--seed-tag', type=str, default=None, metavar='TAG',
                        help="Filter seeds by tag (e.g., 'Swing', 'Scalp'). Only used with --seeds.")
    parser.add_argument('--inspect-templates', action='store_true',
                        help="After clustering, print template inspection table for manual review (feedback loop)")
    parser.add_argument('--template-feedback', type=str, default=None, metavar='PATH',
                        help="Path to edited template feedback JSON. Applies KEEP/DROP + direction overrides "
                             "before forward pass. E.g., reports/template_inspection_feedback.json")

    # Monte Carlo Flags (opt-in with --mc)
    parser.add_argument('--mc', action='store_true', help='Enable Monte Carlo sweep after Phase 3 Optimization')
    parser.add_argument('--mc-iters', type=int, default=2000, help='Monte Carlo iterations per (template, timeframe) combo')
    parser.add_argument('--mc-only', action='store_true', help='Skip discovery, just run Monte Carlo from existing templates')
    parser.add_argument('--anova-only', action='store_true', help='Skip MC sweep, just run ANOVA on existing results')
    parser.add_argument('--refine-only', action='store_true', help='Skip MC+ANOVA, just run Thompson refinement')
    parser.add_argument('--min-hold', type=float, default=0.0, metavar='MINUTES',
                        help="Minimum hold time in minutes (e.g., 5). Suppresses non-reversal exits "
                             "before this duration. Reversal exits (belief_flip, regime_decay, band_urgent) "
                             "and stop_loss still fire.")

    args = parser.parse_args()

    # --seeds AUTO: find latest seed file (prefer auto-swing over manual)
    if args.seeds == 'AUTO':
        _seed_dir = os.path.join('DATA', 'regime_seeds')
        _found = None
        import glob as _g
        # Prefer auto-swing seeds (37K+ seeds, full coverage)
        _auto_dir = os.path.join(_seed_dir, 'auto_swing')
        if os.path.isdir(_auto_dir):
            _auto_files = sorted(_g.glob(os.path.join(_auto_dir, 'auto_seeds_all_*.json')))
            if _auto_files:
                _found = _auto_files[-1]
        # Fallback to manual seeds
        if not _found and os.path.isdir(_seed_dir):
            _manual_files = sorted(_g.glob(os.path.join(_seed_dir, 'seeds_*.json')))
            if _manual_files:
                _found = _manual_files[-1]
        if _found:
            args.seeds = _found
            print(f"  --seeds AUTO: using {args.seeds}")
        else:
            print(f"  --seeds AUTO: no seed files found in {_seed_dir}")
            args.seeds = None

    # ── Tee stdout -> checkpoints/training_log.txt (append, one file per project) ──
    import io
    import datetime as _dt_tee

    def _stamp_data(data, at_line_start):
        """Prepend [HH:MM:SS] to each new line. Skip \r-only updates (tqdm)."""
        if not data:
            return data, at_line_start
        # \r anywhere = tqdm progress bar — pass through raw
        if '\r' in data:
            return data, True  # next write after progress bar is a fresh line
        stamped = []
        for i, line in enumerate(data.split('\n')):
            if i > 0:
                stamped.append('\n')
            if line and at_line_start:
                ts = _dt_tee.datetime.now().strftime('%H:%M:%S')
                stamped.append(f'[{ts}] {line}')
            elif line:
                stamped.append(line)
            if i > 0:
                at_line_start = True
        if data and not data.endswith('\n'):
            at_line_start = False
        elif data.endswith('\n'):
            at_line_start = True
        return ''.join(stamped), at_line_start

    class _Tee:
        """Tee stdout to file + terminal with timestamps. Plain class (not
        io.TextIOWrapper — inheriting without super().__init__() deadlocks
        when tqdm or logging probe uninitialized parent attributes)."""
        def __init__(self, log_path):
            self._file = open(log_path, 'a', encoding='utf-8', buffering=1)
            self._stdout = sys.stdout
            self._at_line_start = True
            self.encoding = getattr(self._stdout, 'encoding', 'utf-8')
            self.errors = getattr(self._stdout, 'errors', 'replace')
        def write(self, data):
            if not data:
                return 0
            out, self._at_line_start = _stamp_data(data, self._at_line_start)
            try:
                self._stdout.write(out)
            except UnicodeEncodeError:
                self._stdout.write(out.encode('ascii', 'replace').decode('ascii'))
            self._file.write(out)
            return len(data)
        def flush(self):
            self._stdout.flush()
            self._file.flush()
        def isatty(self):
            return self._stdout.isatty()
        def fileno(self):
            return self._stdout.fileno()
        def writable(self):
            return True
        def readable(self):
            return False

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, 'training_log.txt')
    _tee = _Tee(log_path)
    sys.stdout = _tee

    # Also timestamp stderr (warnings, tracebacks)
    class _StderrTee:
        def __init__(self):
            self._stderr = sys.__stderr__
            self._at_line_start = True
        def write(self, data):
            if not data:
                return 0
            out, self._at_line_start = _stamp_data(data, self._at_line_start)
            self._stderr.write(out)
            return len(data)
        def flush(self):
            self._stderr.flush()
        def isatty(self):
            return self._stderr.isatty()
        def fileno(self):
            return self._stderr.fileno()
    sys.stderr = _StderrTee()

    # Configure logging module (timestamps added by _StderrTee wrapper)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('numba.cuda').setLevel(logging.ERROR)

    # Print a run separator so phases from different runs are easy to distinguish
    import datetime as _dt

    # --live-prep: auto-compute OOS cutoff to last Friday
    # Only applies to OOS pass — IS runs on full ATLAS data uncapped
    args._live_prep_cutoff = None
    if getattr(args, 'live_prep', False):
        _today = _dt.date.today()
        _days_since_fri = (_today.weekday() - 4) % 7
        if _days_since_fri == 0 and _today.weekday() == 4:
            _days_since_fri = 7
        _last_friday = _today - _dt.timedelta(days=_days_since_fri)
        args._live_prep_cutoff = _last_friday.strftime('%Y%m%d')
        print(f"  --live-prep: OOS cutoff = {args._live_prep_cutoff} (last Friday: {_last_friday})")
        print(f"  IS pass runs uncapped. OOS capped at {args._live_prep_cutoff}.")

    print(f"\n{'='*80}")
    print(f"RUN STARTED: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    if not args.skip_deps:
        check_and_install_requirements()

    orchestrator = Trainer(args)
    orchestrator._ping_pong = getattr(args, 'ping_pong', False)
    orchestrator._pp_conviction = getattr(args, 'pp_conviction', 0.55)
    orchestrator._pp_sl_override = getattr(args, 'pp_sl', 0)
    orchestrator._pp_tp_override = getattr(args, 'pp_tp', 0)
    orchestrator._pp_trail_override = getattr(args, 'pp_trail', 0)
    orchestrator._slippage_ticks = getattr(args, 'slippage', 0.0)
    orchestrator._use_primitives = getattr(args, 'primitives', False)
    orchestrator._use_lookback = True  # 22D features always on (6D lookback geometry)
    orchestrator._use_shapes = getattr(args, 'shapes', False)
    # Min-hold: convert minutes → 15s bars (execution TF)
    _min_hold_mins = getattr(args, 'min_hold', 0.0)
    orchestrator._min_hold_bars = int(_min_hold_mins * 60 / 15) if _min_hold_mins > 0 else 0

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

        if args.depth_iso and not args.fresh:
            # Per-depth isolation analysis (uses existing playbook)
            orchestrator.run_depth_analysis(args.data,
                                            start_date=args.forward_start,
                                            end_date=args.forward_end,
                                            oos_mode=args.oos)
        elif args.oos3 and not args.fresh:
            # Standalone OOS3: BarProcessor only, no inline OOS needed
            _oos_data = getattr(args, 'forward_data', None) or os.path.join('DATA', 'ATLAS_OOS')
            orchestrator.run_oos3_standalone(
                data_source=_oos_data,
                n_days=5,
                bias_threshold=args.bias_threshold,
                dmi_threshold=args.dmi_threshold,
                account_size=args.account_size,
            )

        elif getattr(args, 'oos_chain', False) and not args.fresh:
            # Full OOS chain: OOS1 → Strategy → OOS2 → OOS3 (no IS)
            _oos_path = getattr(args, 'forward_data', None) or os.path.join('DATA', 'ATLAS_OOS')
            _oos_end = getattr(args, '_live_prep_cutoff', None) or args.forward_end
            import shutil as _shutil

            # OOS1: blind, no tier preference
            print("\n" + "=" * 80)
            print("  OOS1: BLIND VALIDATION (no tier preference)")
            print("=" * 80)
            orchestrator.run_forward_pass(_oos_path,
                                          start_date=args.forward_start,
                                          end_date=_oos_end,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=True,
                                          account_size=args.account_size,
                                          popup_label='oos1')
            _oos1 = dict(orchestrator._fp_summary)
            _shutil.copy2(os.path.join(orchestrator.checkpoint_dir, 'oos_trade_log.csv'),
                          os.path.join(orchestrator.checkpoint_dir, 'oos1_trade_log.csv'))

            # Strategy grading
            orchestrator.run_strategy_selection()

            # OOS2: re-run with tier preference tiebreaker
            print("\n" + "=" * 80)
            print("  OOS2: VERIFICATION (tier preference tiebreaker active)")
            print("=" * 80)
            orchestrator.run_forward_pass(_oos_path,
                                          start_date=args.forward_start,
                                          end_date=_oos_end,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=True,
                                          account_size=args.account_size,
                                          tier_preference=True,
                                          popup_label='oos2')
            _oos2 = dict(orchestrator._fp_summary)
            _shutil.copy2(os.path.join(orchestrator.checkpoint_dir, 'oos_trade_log.csv'),
                          os.path.join(orchestrator.checkpoint_dir, 'oos2_trade_log.csv'))

            # OOS1 vs OOS2 comparison
            _print_oos_comparison(_oos1, _oos2)

            # OOS3: BarProcessor live mode validation
            print("\n" + "=" * 80)
            print("  OOS3: LIVE MODE VALIDATION (bar-by-bar via BarProcessor)")
            print("=" * 80)
            orchestrator.run_forward_pass(_oos_path,
                                          start_date=args.forward_start,
                                          end_date=_oos_end,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=True,
                                          account_size=args.account_size,
                                          tier_preference=True,
                                          live_validation_days=5,
                                          popup_label='oos3')
            _oos3 = dict(orchestrator._fp_summary)
            _shutil.copy2(os.path.join(orchestrator.checkpoint_dir, 'oos_trade_log.csv'),
                          os.path.join(orchestrator.checkpoint_dir, 'oos3_trade_log.csv'))

            # OOS2 vs OOS3 comparison
            print("\n  OOS2 vs OOS3 (inline vs BarProcessor):")
            for _k in ['trades', 'win_rate', 'total_pnl', 'avg_trade']:
                _v2 = _oos2.get(_k, 0)
                _v3 = _oos3.get(_k, 0)
                if isinstance(_v2, float):
                    print(f"    {_k:20s}  OOS2={_v2:>10.2f}  OOS3={_v3:>10.2f}  delta={_v3-_v2:>+10.2f}")
                else:
                    print(f"    {_k:20s}  OOS2={_v2:>10}  OOS3={_v3:>10}  delta={_v3-_v2:>+10}")

            # Generate OOS chain comparison chart
            _chart_path = os.path.join('reports', 'oos_chain_comparison.png')
            try:
                from tools.oos_chain_chart import generate_oos_chain_chart
                generate_oos_chain_chart(orchestrator.checkpoint_dir, _chart_path)
                print(f"\n  OOS chain chart saved: {_chart_path}")
            except Exception as _e:
                print(f"\n  OOS chain chart failed: {_e}")

        elif args.oos and not args.forward_pass and not args.fresh:
            # Standalone OOS rerun (Phase 5 only)
            _oos_data = getattr(args, 'forward_data', None) or os.path.join('DATA', 'ATLAS_OOS')
            _oos_end = getattr(args, '_live_prep_cutoff', None) or args.forward_end
            orchestrator.run_forward_pass(_oos_data,
                                          start_date=args.forward_start,
                                          end_date=_oos_end,

                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=True,
                                          account_size=args.account_size)
        elif (args.forward_pass or args.oos) and not args.fresh:
            # Phase 4 IS → Phase 5 OOS1 → Phase 6 Strategy → OOS2 verify
            _fwd_data = getattr(args, 'forward_data', None) or args.data
            orchestrator.run_forward_pass(_fwd_data,
                                          start_date=args.forward_start,
                                          end_date=args.forward_end,
                                          trade_start_date=args.trade_start,
                                          bias_threshold=args.bias_threshold,
                                          dmi_threshold=args.dmi_threshold,
                                          oos_mode=args.oos,
                                          account_size=args.account_size)

            # Auto-chain OOS1 → Strategy → OOS2 verify
            _oos_path = os.path.join('DATA', 'ATLAS_OOS')
            if (not args.skip_oos
                    and not getattr(args, 'forward_data', None)
                    and not args.oos
                    and os.path.isdir(_oos_path)):
                _oos_end = getattr(args, '_live_prep_cutoff', None) or args.forward_end

                # Phase 5: OOS1 (blind, no tier preference)
                print("\n" + "=" * 80)
                print("  PHASE 5: OOS VALIDATION (blind, no tier preference)")
                print("=" * 80)
                orchestrator.run_forward_pass(_oos_path,
                                              start_date=args.forward_start,
                                              end_date=_oos_end,
                                              bias_threshold=args.bias_threshold,
                                              dmi_threshold=args.dmi_threshold,
                                              oos_mode=True,
                                              account_size=args.account_size,
                                              popup_label='oos1')
                _oos1 = dict(orchestrator._fp_summary)

                # Phase 6: Strategy (grades on OOS trade log)
                orchestrator.run_strategy_selection()

                # OOS2 removed — tier preferences showed <$140 delta on 1,700 trades.
                # OOS1 (blind) is the real validation. OOS3 (parity) is engineering check.

                # OOS3: BarProcessor parity (standalone — only last 5 days)
                orchestrator.run_oos3_standalone(
                    data_source=_oos_path,
                    n_days=5,
                    bias_threshold=args.bias_threshold,
                    dmi_threshold=args.dmi_threshold,
                    account_size=args.account_size)

                # Save warmed brain for live handoff
                _live_brain_path = os.path.join(orchestrator.checkpoint_dir, 'live_brain.pkl')
                orchestrator.brain.save(_live_brain_path)
                print(f"\n  Saved warmed brain: {_live_brain_path}")

                # Verdict
                print("\n" + "=" * 80)
                print("  PIPELINE COMPLETE")
                print(f"  Brain saved to: {_live_brain_path}")
                print(f"\n  NEXT STEP (manual):")
                print(f"    python -m live.launcher")
                print(f"    NT8 account controls sim vs real money.")
                print("=" * 80)

        elif args.strategy_report and not args.forward_pass:
            orchestrator.run_strategy_selection()  # requires oos_trade_log.csv to exist
        else:
            # Full pipeline
            if args.mc_only:
                # MC-only: skip discovery, load existing library
                print("Skipping Phase 1/2, loading existing library...")
                lib_path = os.path.join(orchestrator.checkpoint_dir, 'pattern_library.pkl')
                if os.path.exists(lib_path):
                    with open(lib_path, 'rb') as f:
                        orchestrator.pattern_library = pickle.load(f)
                else:
                    print("ERROR: pattern_library.pkl not found for --mc-only")
                    return 1
            else:
                # Phase 1-3: only if --fresh or no checkpoints exist.
                # Default: reuse existing templates/library, fresh brain for IS.
                _lib_exists = os.path.exists(os.path.join(
                    orchestrator.checkpoint_dir, 'pattern_library.pkl'))
                _tmpl_exists = os.path.exists(os.path.join(
                    orchestrator.checkpoint_dir, 'templates.pkl'))

                if args.fresh or not (_lib_exists and _tmpl_exists):
                    # Phase 1 (Discovery) + 2 (Clustering) + 3 (Optimization)
                    orchestrator.train(args.data)
                else:
                    print("\n  [SKIP] Phase 1-3: checkpoints exist, reusing templates/library.")
                    print("         Brain will be fresh for IS. Use --fresh to rebuild everything.")
                    # Load existing checkpoints
                    _ckpt = load_checkpoints(orchestrator.checkpoint_dir)
                    if _ckpt.pattern_library:
                        orchestrator.pattern_library = _ckpt.pattern_library
                    if _ckpt.scaler:
                        orchestrator.scaler = _ckpt.scaler
                    # Templates loaded separately (not in CheckpointBundle)
                    _tmpl_path = os.path.join(orchestrator.checkpoint_dir, 'templates.pkl')
                    if os.path.exists(_tmpl_path):
                        with open(_tmpl_path, 'rb') as f:
                            orchestrator.templates = pickle.load(f)
                        print(f"  Loaded {len(orchestrator.templates)} templates")

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
                # Default: Bayesian path -> IS -> OOS1 -> Strategy -> OOS2 (verify)
                _fwd_data = getattr(args, 'forward_data', None) or args.data

                # Snapshot checkpoints before IS — restore with --forward-pass
                # to skip Phase 1-3 and re-run IS with code changes only.
                import shutil as _shutil_bak
                _pre_is_dir = os.path.join('checkpoints', 'pre_is_backup')
                if os.path.isdir(_pre_is_dir):
                    _shutil_bak.rmtree(_pre_is_dir, ignore_errors=True)
                os.makedirs(_pre_is_dir, exist_ok=True)
                for _bak_f in ['templates.pkl', 'pattern_library.pkl',
                               'discovery_manifest.pkl', 'discovery_levels.json',
                               'pipeline_state.json', 'scaler.pkl',
                               'gate_thresholds.json', 'depth_weights.json']:
                    _src = os.path.join('checkpoints', _bak_f)
                    if os.path.exists(_src):
                        _shutil_bak.copy2(_src, os.path.join(_pre_is_dir, _bak_f))
                print(f"  [BACKUP] Pre-IS checkpoint snapshot saved to {_pre_is_dir}/")

                # Phase 4: IS Backtest
                orchestrator.run_forward_pass(_fwd_data,
                                              start_date=args.forward_start,
                                              end_date=args.forward_end,
                                              bias_threshold=args.bias_threshold,
                                              dmi_threshold=args.dmi_threshold,
                                              oos_mode=getattr(args, 'oos', False),
                                              account_size=getattr(args, 'account_size', 0.0))

                if args.depth_iso:
                    orchestrator.run_depth_analysis(args.data,
                                                    start_date=args.forward_start,
                                                    end_date=args.forward_end,
                                                    oos_mode=getattr(args, 'oos', False))

                # Auto-chain OOS + Strategy + OOS2 verify
                _oos_path = os.path.join('DATA', 'ATLAS_OOS')
                if (not getattr(args, 'skip_oos', False)
                        and not getattr(args, 'forward_data', None)
                        and os.path.isdir(_oos_path)):
                    _oos_end = getattr(args, '_live_prep_cutoff', None) or args.forward_end

                    # Phase 5: OOS1 (blind, no tier preference)
                    print("\n" + "=" * 80)
                    print("  PHASE 5: OOS VALIDATION (blind, no tier preference)")
                    print("=" * 80)
                    orchestrator.run_forward_pass(_oos_path,
                                              start_date=args.forward_start,
                                              end_date=_oos_end,
                                              bias_threshold=args.bias_threshold,
                                              dmi_threshold=args.dmi_threshold,
                                              oos_mode=True,
                                              account_size=getattr(args, 'account_size', 0.0),
                                              popup_label='oos1')
                    _oos1 = dict(orchestrator._fp_summary)

                    # Phase 6: Strategy (grades on OOS trade log)
                    orchestrator.run_strategy_selection()

                    # OOS2 removed — tier preferences showed <$140 delta.

                # OOS3: BarProcessor parity (standalone — only processes last 5 days)
                orchestrator.run_oos3_standalone(
                    data_source=_oos_path,
                    n_days=5,
                    bias_threshold=args.bias_threshold,
                    dmi_threshold=args.dmi_threshold,
                    account_size=getattr(args, 'account_size', 0.0))

                # Save warmed brain for live handoff
                _live_brain_path = os.path.join(orchestrator.checkpoint_dir, 'live_brain.pkl')
                orchestrator.brain.save(_live_brain_path)
                print(f"\n  Saved warmed brain: {_live_brain_path}")

                # Verdict
                print("\n" + "=" * 80)
                print("  PIPELINE COMPLETE")
                print(f"  Brain saved to: {_live_brain_path}")
                print(f"\n  NEXT STEP (manual):")
                print(f"    python -m live.launcher")
                print(f"    NT8 account controls sim vs real money.")
                print("=" * 80)

        orchestrator.print_bottom_line()
        return 0
    except KeyboardInterrupt:
        print("\n\nWARNING: Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        # Also print to stdout so training_log.txt captures it
        import io as _io
        _tb_buf = _io.StringIO()
        traceback.print_exc(file=_tb_buf)
        print(_tb_buf.getvalue())
        return 1
    finally:
        _awake_ctx.__exit__(None, None, None)
        # Restore stdout and close log file
        if isinstance(sys.stdout, _Tee):
            sys.stdout = _tee._stdout
            _tee._file.close()


if __name__ == "__main__":
    sys.exit(main())
