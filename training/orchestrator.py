"""
BAYESIAN-AI TRAINING ORCHESTRATOR
Single entry point for all training operations

Integrates:
- Walk-forward training (day-by-day DOE)
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
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import time

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

# Execution components
from execution.integrated_statistical_system import IntegratedStatisticalEngine
from execution.batch_regret_analyzer import BatchRegretAnalyzer

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
    # Walk-Forward "Real" Results (using prior day's params)
    real_pnl: float = 0.0
    real_trades_count: int = 0
    real_win_rate: float = 0.0
    real_sharpe: float = 0.0


class BayesianTrainingOrchestrator:
    """
    UNIFIED TRAINING ORCHESTRATOR

    Runs complete walk-forward training with:
    - Day-by-day DOE parameter optimization
    - Live visualization dashboard
    - Real-time terminal progress
    - Pattern analysis and reporting
    - Batch regret analysis
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

        # Slippage parameters
        self.BASE_SLIPPAGE = DEFAULT_BASE_SLIPPAGE
        self.VELOCITY_SLIPPAGE_FACTOR = DEFAULT_VELOCITY_SLIPPAGE_FACTOR

    def train(self, data: pd.DataFrame):
        """
        Master training loop

        Args:
            data: Full dataset with timestamps
        """
        # Clear debug log for fresh run
        debug_log_path = os.path.join(PROJECT_ROOT, 'debug_outputs', PRECOMPUTE_DEBUG_LOG_FILENAME)
        if os.path.exists(debug_log_path):
            os.remove(debug_log_path)

        print("\n" + "="*80)
        print("BAYESIAN-AI TRAINING ORCHESTRATOR")
        print("="*80)

        # CUDA status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"CUDA: ENABLED  |  GPU: {gpu_name}  |  VRAM: {gpu_mem:.1f} GB")
            else:
                print("CUDA: NOT AVAILABLE  |  Running on CPU")
        except ImportError:
            print("CUDA: NOT AVAILABLE (PyTorch not installed)  |  Running on CPU")

        print(f"Asset: {self.asset.ticker}")
        print(f"Checkpoint Dir: {self.checkpoint_dir}")
        print(f"Iterations per Day: {self.config.iterations}")

        # Launch dashboard if available
        if DASHBOARD_AVAILABLE and not self.config.no_dashboard:
            self.launch_dashboard()

        # Pre-aggregate 1s → 15s once for the entire dataset (cached to disk)
        data_15s = self._get_or_create_aggregated_data(data)

        # Resample full dataset to all higher timeframes (1d, 4h, 1h, 15m, 5m, 1m)
        print("Resampling to all higher timeframes (1d, 4h, 1h, 15m, 5m, 1m)...", end='', flush=True)
        self.all_tf_data = self.mtf_context.resample_all(data)
        tf_counts = {k: len(v) for k, v in self.all_tf_data.items()}
        print(f" done | Bars: {tf_counts}")

        # Split into trading days (both 1s and 15s)
        days_1s = self.split_into_trading_days(data)
        days_15s = self.split_into_trading_days(data_15s)

        # Build lookup: date → 15s day_data
        days_15s_lookup = {date: df for date, df in days_15s}

        total_days = len(days_1s)

        if self.config.max_days:
            days_1s = days_1s[:self.config.max_days]
            total_days = len(days_1s)
            print(f"Limiting to first {total_days} days")

        print(f"\nTraining on {total_days} trading days...")
        print(f"Date range: {days_1s[0][0]} to {days_1s[-1][0]}")
        print("="*80 + "\n")

        # Initialize active params with baseline (Day 0)
        active_params = self.param_generator.generate_baseline_set(0, 1, 'CORE').parameters

        # Train day by day
        prev_day_15s = None
        for day_idx, (date, day_data) in enumerate(days_1s):
            day_number = day_idx + 1

            # Print day header
            self.progress_reporter.print_day_header(
                day_number, date, total_days, len(day_data)
            )

            # Compute higher-TF context for this day
            tf_context = self.mtf_context.get_context_for_day(
                day_idx, self.all_tf_data, date
            )

            day_data_15s = days_15s_lookup.get(date)

            # === STEP 1: PRE-COMPUTE STATES ===
            precomputed = self._precompute_day_states(
                day_data, day_data_15s=day_data_15s, tf_context=tf_context,
                prev_day_15s=prev_day_15s
            )

            # === STEP 2: WALK-FORWARD SIMULATION (Real PnL) ===
            # Trade today using yesterday's parameters (active_params)
            real_trades = self._simulate_from_precomputed(
                precomputed, day_data, active_params, pbar=None
            )
            real_pnl = sum(t.pnl for t in real_trades)
            real_win_rate = (sum(1 for t in real_trades if t.result == 'WIN') / len(real_trades)) if real_trades else 0.0
            real_sharpe = (np.mean([t.pnl for t in real_trades]) / (np.std([t.pnl for t in real_trades]) + 1e-6)) if len(real_trades) > 1 else 0.0

            print(f"  [WALK-FORWARD] Real PnL: ${real_pnl:.2f} ({len(real_trades)} trades, WR: {real_win_rate:.1%})")

            # === STEP 3: OPTIMIZE (Oracle Learning for Brain & Tomorrow's Params) ===
            day_result = self.optimize_day(
                day_number, date, day_data,
                day_data_15s=day_data_15s, tf_context=tf_context,
                prev_day_15s=prev_day_15s,
                precomputed_states=precomputed, # Reuse precomputed
                total_days=total_days
            )

            # Add Real stats to day_result for reporting
            day_result.real_pnl = real_pnl
            day_result.real_trades_count = len(real_trades)
            day_result.real_win_rate = real_win_rate
            day_result.real_sharpe = real_sharpe

            # Update params for NEXT day
            if day_result.best_params:
                active_params = day_result.best_params

            prev_day_15s = day_data_15s

            # Batch regret analysis (end of day)
            if self.todays_trades:
                regret_analysis = self.regret_analyzer.batch_analyze_day(
                    self.todays_trades,
                    day_data
                )
                self.regret_analyzer.print_analysis(regret_analysis)
            else:
                regret_analysis = None

            # Update reports - USE REAL PNL for report integrity
            day_metrics = DayMetrics(
                day_number=day_number,
                date=date,
                total_trades=day_result.real_trades_count, # REPORT REAL
                win_rate=day_result.real_win_rate,         # REPORT REAL
                sharpe=day_result.real_sharpe,             # REPORT REAL
                pnl=day_result.real_pnl,                   # REPORT REAL
                states_learned=day_result.states_learned,
                high_conf_states=day_result.high_confidence_states,
                avg_duration=day_result.avg_duration,
                execution_time=day_result.execution_time_seconds
            )

            self.progress_reporter.print_day_summary(day_metrics)

            # Get top patterns
            top_patterns = self.pattern_analyzer.get_strongest_patterns(self.brain, top_n=5)
            self.progress_reporter.print_cumulative_summary(top_patterns)

            # Accumulate best trades across days (NOT all iterations)
            self._cumulative_best_trades.extend(self._best_trades_today)

            # Write dashboard JSON (polled by LiveDashboard every 1s)
            self._write_dashboard_json(day_metrics, day_result, total_days)

            # Save checkpoint
            self.save_checkpoint(day_number, date, day_result)

            self.day_results.append(day_result)

        # Final summary
        self.print_final_summary()

        return self.day_results

    def _precompute_day_states(self, day_data: pd.DataFrame, day_data_15s: pd.DataFrame = None,
                               tf_context: Dict = None,
                               prev_day_15s: pd.DataFrame = None):
        """
        Pre-compute all states, probabilities, and confidences for a day ONCE.

        Uses vectorized batch computation (numpy arrays + optional CUDA)
        instead of per-bar loop. ~10-50x faster than original.

        Args:
            day_data: 1s OHLCV data for the day (used for trade simulation)
            day_data_15s: Pre-aggregated 15s data (if None, resamples on the fly)
            tf_context: Multi-timeframe context from MultiTimeframeContext.get_context_for_day()
            prev_day_15s: Previous day's 15s data for warming up regression windows

        Returns:
            List of dicts with bar_idx, state, price, prob, conf, structure_ok, direction
        """
        if len(day_data) < 21:
            return []

        # Ensure 'close' column exists for batch computation
        if 'price' in day_data.columns and 'close' not in day_data.columns:
            day_data = day_data.copy()
            day_data['close'] = day_data['price']

        # Pre-extract 1s arrays for fast trade simulation (kept at 1s resolution)
        prices = day_data['price'].values if 'price' in day_data.columns else day_data['close'].values
        timestamps = day_data['timestamp'].values if 'timestamp' in day_data.columns else np.zeros(len(day_data))

        # Use pre-aggregated 15s data or resample on the fly
        if day_data_15s is not None and len(day_data_15s) >= 21:
            resampled_15s = day_data_15s
            # Build index mapping from 15s timestamps to 1s indices
            idx_map_15s_to_1s = self._build_15s_to_1s_index_map(resampled_15s, day_data)
        else:
            resampled_15s, idx_map_15s_to_1s = self._resample_to_15s(day_data)

        # === WARMUP: Prepend previous day's tail to fix cold-start regression ===
        warmup_bars = 0
        if prev_day_15s is not None and not prev_day_15s.empty:
            # Need 21 bars for regression, take 50 for safety
            warmup_df = prev_day_15s.tail(50).copy()
            warmup_bars = len(warmup_df)
            resampled_15s = pd.concat([warmup_df, resampled_15s], ignore_index=True)

        if len(resampled_15s) < 21:
            self._day_prices = prices
            self._day_timestamps = timestamps
            return []

        print(f"  Pre-computing states for {len(resampled_15s) - 21:,} bars "
              f"(15s, {len(day_data):,} 1s bars)...", end='', flush=True)
        precompute_start = time.time()

        # === VECTORIZED BATCH COMPUTATION on 15s bars ===
        batch_results = self.engine.batch_compute_states(resampled_15s, use_cuda=True)

        # === FIT DYNAMIC BINNER on first day (before any hashing) ===
        if self.dynamic_binner is None and batch_results:
            z_scores = np.array([b['state'].z_score for b in batch_results], dtype=np.float64)
            momentums = np.array([b['state'].momentum_strength for b in batch_results], dtype=np.float64)
            self.dynamic_binner = DynamicBinner(min_bins=5, max_bins=30)
            self.dynamic_binner.fit({'z_score': z_scores, 'momentum': momentums})
            ThreeBodyQuantumState.set_binner(self.dynamic_binner)
            print(f"\n  {self.dynamic_binner.summary()}")

        # === INJECT MULTI-TIMEFRAME CONTEXT into each state ===
        if tf_context:
            from dataclasses import replace as dc_replace
            daily_ctx = tf_context.get('daily')
            h4_ctx = tf_context.get('h4')
            h1_ctx = tf_context.get('h1')
            context_level = tf_context.get('context_level', 'MINIMAL')

            tf_kwargs = {'context_level': context_level}
            if daily_ctx:
                tf_kwargs['daily_trend'] = daily_ctx.trend
                tf_kwargs['daily_volatility'] = daily_ctx.volatility
            if h4_ctx:
                tf_kwargs['h4_trend'] = h4_ctx.trend
                tf_kwargs['session'] = h4_ctx.session
            if h1_ctx:
                tf_kwargs['h1_trend'] = h1_ctx.trend

            for bar in batch_results:
                bar['state'] = dc_replace(bar['state'], **tf_kwargs)
                # Re-evaluate structure_ok with context (Lagrange zone check unchanged)
                bar['structure_ok'] = (
                    bar['state'].lagrange_zone in ('L2_ROCHE', 'L3_ROCHE') and
                    bool(bar['state'].structure_confirmed) and
                    bool(bar['state'].cascade_detected)
                )

        # === SLICE WARMUP: Remove the prepended bars from results ===
        if warmup_bars > 0:
            batch_results = batch_results[warmup_bars:]

        # === REMAP: 15s bar_idx → 1s bar_idx for trade simulation ===
        for bar in batch_results:
            # Adjust index to account for warmup offset
            idx_15s = bar['bar_idx'] - warmup_bars
            if idx_15s < len(idx_map_15s_to_1s):
                bar['bar_idx'] = idx_map_15s_to_1s[idx_15s]
            else:
                bar['bar_idx'] = len(prices) - 1
            # Use 1s price for consistency with trade sim
            bar['price'] = prices[bar['bar_idx']]

        # Cold start confidence modifier based on context availability
        context_level = tf_context.get('context_level', 'MINIMAL') if tf_context else 'MINIMAL'
        conf_modifier = self.mtf_context.get_confidence_modifier(context_level)

        # Add prob/conf from brain + trade direction (these are dict lookups, fast)
        #
        # CONFIDENCE-WEIGHTED PROBABILITY:
        #   Blends neutral prior (50%) with learned probability based on sample count.
        #   This prevents single observations from dominating (avoids Catch-22 where
        #   a state seen once gets locked out forever due to low confidence).
        #
        #   conf=0.0 (0 trades)  -> prob = 50% prior (explore)
        #   conf=0.33 (10 trades) -> prob = 67% prior + 33% learned
        #   conf=1.0 (30 trades) -> prob = 100% learned (exploit)
        #
        prior = 0.50
        for bar in batch_results:
            state = bar['state']
            is_unseen = state not in self.brain.table
            learned_prob = self.brain.get_probability(state)
            conf = self.brain.get_confidence(state)

            if is_unseen:
                # Never seen: use prior (allows exploration of new states)
                bar['prob'] = prior
            else:
                # Blend prior toward learned probability as confidence grows
                effective_conf = conf * conf_modifier
                bar['prob'] = prior * (1.0 - effective_conf) + learned_prob * effective_conf

            bar['conf'] = conf
            # structure_ok is purely physical: Roche zone + wave collapsed
            # No confidence gate here — let DOE threshold decide which trades fire
            # Direction: L2_ROCHE (z>+2) -> SHORT, L3_ROCHE (z<-2) -> LONG
            bar['direction'] = 'SHORT' if state.z_score > 0 else 'LONG'

        elapsed = time.time() - precompute_start
        print(f" done ({elapsed:.1f}s) | Context: {context_level} (modifier={conf_modifier:.1f})")

        # === DEBUG OUTPUT ===
        debug_dir = os.path.join(PROJECT_ROOT, 'debug_outputs')
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, PRECOMPUTE_DEBUG_LOG_FILENAME)
        with open(debug_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"PRECOMPUTE DEBUG — {len(batch_results)} bars computed in {elapsed:.1f}s\n")
            f.write(f"Context Level: {context_level} | Confidence Modifier: {conf_modifier:.2f}\n")
            if self.dynamic_binner is not None:
                f.write(f"{self.dynamic_binner.summary()}\n")
            if tf_context:
                for k in ('daily', 'h4', 'h1'):
                    ctx = tf_context.get(k)
                    if ctx:
                        f.write(f"  {k}: trend={ctx.trend}, vol={ctx.volatility}, session={getattr(ctx, 'session', 'N/A')}\n")
                    else:
                        f.write(f"  {k}: NOT AVAILABLE\n")
            f.write(f"{'='*80}\n")

            # Count filter stages
            total = len(batch_results)
            zones = {}
            n_structure = 0
            n_cascade = 0
            n_both = 0
            n_structure_ok = 0
            for bar in batch_results:
                s = bar['state']
                z = s.lagrange_zone
                zones[z] = zones.get(z, 0) + 1
                if s.structure_confirmed:
                    n_structure += 1
                if s.cascade_detected:
                    n_cascade += 1
                if s.structure_confirmed and s.cascade_detected:
                    n_both += 1
                if bar['structure_ok']:
                    n_structure_ok += 1

            f.write(f"\nLAGRANGE ZONE DISTRIBUTION:\n")
            for z, cnt in sorted(zones.items(), key=lambda x: -x[1]):
                f.write(f"  {z}: {cnt} ({cnt/total*100:.1f}%)\n")

            roche_count = zones.get('L2_ROCHE', 0) + zones.get('L3_ROCHE', 0)
            f.write(f"\nFILTER PIPELINE:\n")
            f.write(f"  Total bars:             {total}\n")
            f.write(f"  In L2/L3 ROCHE zone:    {roche_count} ({roche_count/total*100:.1f}%)\n")
            f.write(f"  structure_confirmed:     {n_structure} ({n_structure/total*100:.1f}%)\n")
            f.write(f"  cascade_detected:        {n_cascade} ({n_cascade/total*100:.1f}%)\n")
            f.write(f"  Both struct+cascade:     {n_both} ({n_both/total*100:.1f}%)\n")
            f.write(f"  Final structure_ok:      {n_structure_ok} ({n_structure_ok/total*100:.1f}%)\n")

            # Sample states
            if batch_results:
                f.write(f"\nSAMPLE STATES (first 5 with structure_ok=True):\n")
                shown = 0
                for bar in batch_results:
                    if bar['structure_ok'] and shown < 5:
                        s = bar['state']
                        f.write(f"  Bar {bar['bar_idx']}: zone={s.lagrange_zone} "
                                f"z={s.z_score:.2f} vel={s.particle_velocity:.2f} "
                                f"cascade={s.cascade_detected} struct={s.structure_confirmed} "
                                f"prob={bar['prob']:.2f} conf={bar['conf']:.2f}\n")
                        shown += 1
                if shown == 0:
                    f.write("  (NONE — all bars filtered out)\n")
                    # Show why: sample 5 bars in ROCHE zones
                    f.write(f"\n  DIAGNOSTIC — Sample ROCHE zone bars:\n")
                    shown2 = 0
                    for bar in batch_results:
                        s = bar['state']
                        if s.lagrange_zone in ('L2_ROCHE', 'L3_ROCHE') and shown2 < 5:
                            f.write(f"    Bar {bar['bar_idx']}: z={s.z_score:.2f} "
                                    f"vel={s.particle_velocity:.2f} "
                                    f"cascade={s.cascade_detected} struct={s.structure_confirmed} "
                                    f"vol_spike={s.structure_confirmed} maturity={s.pattern_maturity:.2f}\n")
                            shown2 += 1
                    if shown2 == 0:
                        f.write("    (No bars in ROCHE zones either)\n")

        # Store prices/timestamps as arrays for fast trade sim
        self._day_prices = prices
        self._day_timestamps = timestamps

        return batch_results

    def _resample_to_15s(self, day_data: pd.DataFrame):
        """
        Resample 1s OHLCV bars to 15s bars for state computation.

        Returns:
            resampled_df: DataFrame with timestamp, close, price, volume columns
            idx_map: numpy array where idx_map[i] = last 1s index in 15s bar i
        """
        df = day_data.copy()

        # Ensure timestamp is datetime for resampling
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Track original 1s indices
        df['_1s_idx'] = np.arange(len(df))

        df = df.set_index('timestamp')

        # Build aggregation dict
        agg_dict = {'_1s_idx': 'last'}

        if 'close' in df.columns:
            agg_dict['close'] = 'last'
        if 'price' in df.columns:
            agg_dict['price'] = 'last'
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        if 'open' in df.columns:
            agg_dict['open'] = 'first'
        if 'high' in df.columns:
            agg_dict['high'] = 'max'
        if 'low' in df.columns:
            agg_dict['low'] = 'min'

        # If no close but price exists, derive it
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
            agg_dict['close'] = 'last'

        resampled = df.resample('15s').agg(agg_dict).dropna(subset=['close'] if 'close' in agg_dict else ['price'])

        # Extract index mapping
        idx_map = resampled['_1s_idx'].values.astype(int)

        # Reset index: timestamp back as column
        resampled = resampled.reset_index()

        # Ensure price column exists
        if 'price' not in resampled.columns:
            resampled['price'] = resampled['close']

        # Drop helper column
        resampled = resampled.drop(columns=['_1s_idx'], errors='ignore')

        return resampled, idx_map

    def _get_or_create_aggregated_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-aggregate full 1s dataset to 15s bars once. Caches to parquet for reuse.
        """
        # Derive cache path from source data path
        source_path = getattr(self.config, 'data', None)
        if source_path:
            base = os.path.splitext(source_path)[0]
            cache_path = f"{base}_15s.parquet"
        else:
            cache_path = os.path.join(self.checkpoint_dir, "aggregated_15s.parquet")

        # Check cache
        if os.path.exists(cache_path):
            print(f"Loading cached 15s data from {cache_path}...")
            data_15s = pd.read_parquet(cache_path)
            print(f"  {len(data_15s):,} 15s bars loaded (from {len(data):,} 1s bars)")
            return data_15s

        # Resample full dataset
        print(f"Aggregating {len(data):,} 1s bars to 15s (one-time)...", end='', flush=True)
        start = time.time()

        df = data.copy()

        # Ensure timestamp is datetime for resampling
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
            df = df.set_index('timestamp')

        # Build aggregation dict
        agg_dict = {}
        if 'close' in df.columns:
            agg_dict['close'] = 'last'
        if 'price' in df.columns:
            agg_dict['price'] = 'last'
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        if 'open' in df.columns:
            agg_dict['open'] = 'first'
        if 'high' in df.columns:
            agg_dict['high'] = 'max'
        if 'low' in df.columns:
            agg_dict['low'] = 'min'

        # Derive close from price if needed
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
            agg_dict['close'] = 'last'

        dropna_col = 'close' if 'close' in agg_dict else 'price'
        resampled = df.resample('15s').agg(agg_dict).dropna(subset=[dropna_col])

        # Reset index: timestamp back as column
        resampled = resampled.reset_index()

        # Ensure price column exists
        if 'price' not in resampled.columns and 'close' in resampled.columns:
            resampled['price'] = resampled['close']

        elapsed = time.time() - start
        print(f" done ({elapsed:.1f}s) -> {len(resampled):,} bars")

        # Save cache
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            resampled.to_parquet(cache_path, index=False)
            print(f"  Cached to {cache_path}")
        except Exception as e:
            print(f"  WARNING: Could not cache: {e}")

        return resampled

    def _build_15s_to_1s_index_map(self, data_15s: pd.DataFrame, data_1s: pd.DataFrame) -> np.ndarray:
        """
        Map each 15s bar index to the corresponding last 1s bar index.
        Uses np.searchsorted on timestamps.
        """
        ts_15s = data_15s['timestamp'].values
        ts_1s = data_1s['timestamp'].values

        # Convert to int64 nanoseconds for searchsorted comparison
        if pd.api.types.is_datetime64_any_dtype(ts_15s):
            ts_15s_num = ts_15s.astype('int64')
        else:
            ts_15s_num = pd.to_numeric(pd.Series(ts_15s)).values.astype('int64')

        if pd.api.types.is_datetime64_any_dtype(ts_1s):
            ts_1s_num = ts_1s.astype('int64')
        else:
            ts_1s_num = pd.to_numeric(pd.Series(ts_1s)).values.astype('int64')

        # searchsorted 'right' gives the first 1s index AFTER the 15s timestamp
        # subtract 1 to get the last 1s bar AT or BEFORE the 15s bar
        indices = np.searchsorted(ts_1s_num, ts_15s_num, side='right') - 1

        # Clamp to valid range
        indices = np.clip(indices, 0, len(ts_1s) - 1)

        return indices.astype(int)

    def _simulate_from_precomputed(self, precomputed: list, day_data: pd.DataFrame,
                                    params: Dict[str, Any], pbar=None, best_sharpe: float = -999.0) -> List[TradeOutcome]:
        """
        Fast simulation using pre-computed states. Only applies param thresholds + trade sim.
        """
        trades = []
        # Phase-aware threshold: use the LOWER of DOE param and phase threshold
        phase_threshold = self.confidence_manager.PHASES[self.confidence_manager.phase]['prob_threshold']
        min_prob = min(params.get('confidence_threshold', 0.50), phase_threshold) if phase_threshold > 0 else 0.0
        stop_loss = params.get('stop_loss_ticks', 15) * 0.25
        take_profit = params.get('take_profit_ticks', 40) * 0.25
        max_hold = params.get('max_hold_seconds', 600)
        trading_cost = params.get('trading_cost_points', 0.50)  # Round-trip cost in points
        pv = self.asset.point_value  # Points -> dollars (MNQ: $2.0/point)

        prices = self._day_prices
        timestamps = self._day_timestamps
        # DYNAMIC LOOKAHEAD: Ensure we look far enough to cover max_hold time
        # Assuming roughly 1 bar per second, plus buffer for regret analysis (5 min)
        max_lookahead = int(max_hold + 300)
        n_bars = len(prices)

        trade_wins = 0
        trade_pnl = 0.0

        for bar in precomputed:
            # Decide if we should trade
            should_trade = False

            if self.exploration_mode and self.explorer:
                # Use Explorer logic (bypasses structure_ok and prob thresholds)
                decision = self.explorer.should_fire(bar['state'])
                should_trade = decision['should_fire']
            else:
                # Standard Logic
                should_trade = bar['structure_ok'] and bar['prob'] >= min_prob

            if not should_trade:
                continue

            # --- Inline fast trade simulation (avoid DataFrame overhead) ---
            entry_idx = bar['bar_idx']
            entry_price = bar['price']
            direction = bar.get('direction', 'LONG')
            dir_sign = -1.0 if direction == 'SHORT' else 1.0
            entry_time = timestamps[entry_idx]
            if isinstance(entry_time, pd.Timestamp):
                entry_time = entry_time.timestamp()
            elif hasattr(entry_time, 'item'):
                entry_time = pd.Timestamp(entry_time).timestamp()

            end_idx = min(n_bars, entry_idx + max_lookahead)
            outcome = None

            # Dynamic Slippage (Walk-Forward)
            velocity = bar['state'].particle_velocity
            slippage = self.BASE_SLIPPAGE + self.VELOCITY_SLIPPAGE_FACTOR * abs(velocity)
            total_slippage = slippage * 2.0

            for j in range(entry_idx + 1, end_idx):
                price = prices[j]
                pnl = (price - entry_price) * dir_sign - trading_cost - total_slippage

                exit_time = timestamps[j]
                if isinstance(exit_time, pd.Timestamp):
                    exit_time = exit_time.timestamp()
                elif hasattr(exit_time, 'item'):
                    exit_time = pd.Timestamp(exit_time).timestamp()

                duration = exit_time - entry_time

                if pnl >= take_profit:
                    outcome = TradeOutcome(
                        state=bar['state'], entry_price=entry_price, exit_price=price,
                        pnl=take_profit * pv, result='WIN', timestamp=exit_time,
                        exit_reason='TP', entry_time=entry_time, exit_time=exit_time,
                        duration=duration, direction=direction
                    )
                    break
                elif pnl <= -stop_loss:
                    outcome = TradeOutcome(
                        state=bar['state'], entry_price=entry_price, exit_price=price,
                        pnl=-stop_loss * pv, result='LOSS', timestamp=exit_time,
                        exit_reason='SL', entry_time=entry_time, exit_time=exit_time,
                        duration=duration, direction=direction
                    )
                    break
                elif duration >= max_hold:
                    outcome = TradeOutcome(
                        state=bar['state'], entry_price=entry_price, exit_price=price,
                        pnl=pnl * pv, result='WIN' if pnl > 0 else 'LOSS', timestamp=exit_time,
                        exit_reason='TIME', entry_time=entry_time, exit_time=exit_time,
                        duration=duration, direction=direction
                    )
                    break

            # End of data fallback
            if outcome is None and entry_idx + 1 < n_bars:
                last_price = prices[end_idx - 1]
                pnl = (last_price - entry_price) * dir_sign - trading_cost
                last_time = timestamps[end_idx - 1]
                if isinstance(last_time, pd.Timestamp):
                    last_time = last_time.timestamp()
                elif hasattr(last_time, 'item'):
                    last_time = pd.Timestamp(last_time).timestamp()
                outcome = TradeOutcome(
                    state=bar['state'], entry_price=entry_price, exit_price=last_price,
                    pnl=pnl * pv, result='WIN' if pnl > 0 else 'LOSS', timestamp=last_time,
                    exit_reason='EOD', entry_time=entry_time, exit_time=last_time,
                    duration=last_time - entry_time, direction=direction
                )

            if outcome:
                trades.append(outcome)
                trade_pnl += outcome.pnl
                if outcome.result == 'WIN':
                    trade_wins += 1
                # Live per-trade update
                if pbar:
                    n = len(trades)
                    pbar.set_postfix({
                        'Trades': n,
                        'WR': f'{trade_wins/n:.0%}',
                        'P&L': f'${trade_pnl:.0f}',
                        'Best': f'{best_sharpe:.2f}'
                    })

        return trades

    def optimize_day(self, day_number: int, date: str, day_data: pd.DataFrame,
                     day_data_15s: pd.DataFrame = None,
                     tf_context: Dict = None,
                     prev_day_15s: pd.DataFrame = None,
                     precomputed_states: list = None,
                     total_days: int = 1) -> DayResults:
        """
        Run DOE optimization for single day.
        Routes to GPU-parallel path when CUDA available, CPU sequential otherwise.
        """
        start_time = time.time()
        self.todays_trades = []

        # === PRE-COMPUTE ALL STATES ONCE ===
        if precomputed_states is not None:
            precomputed = precomputed_states
        else:
            precomputed = self._precompute_day_states(
                day_data, day_data_15s=day_data_15s, tf_context=tf_context,
                prev_day_15s=prev_day_15s
            )

        if not precomputed:
            return DayResults(
                day_number=day_number, date=date, total_iterations=self.config.iterations,
                best_iteration=0, best_params={}, best_sharpe=0.0, best_win_rate=0.0,
                best_pnl=0.0, total_trades=0, states_learned=len(self.brain.table),
                high_confidence_states=len(self.brain.get_all_states_above_threshold()),
                execution_time_seconds=time.time() - start_time, avg_duration=0.0
            )

        # Generate ALL param sets upfront
        all_param_sets = []
        for i in range(self.config.iterations):
            ps = self.param_generator.generate_parameter_set(iteration=i, day=day_number, context='CORE')
            all_param_sets.append(ps.parameters)

        # Route to GPU or CPU path
        try:
            import torch
            if torch.cuda.is_available():
                best_idx, all_results = self._optimize_gpu_parallel(
                    precomputed, day_data, all_param_sets, day_number
                )
            else:
                best_idx, all_results = self._optimize_cpu_sequential(
                    precomputed, day_data, all_param_sets, day_number,
                    date=date, total_days=total_days
                )
        except ImportError:
            best_idx, all_results = self._optimize_cpu_sequential(
                precomputed, day_data, all_param_sets, day_number,
                date=date, total_days=total_days
            )

        # Unpack best result
        best_sharpe = all_results[best_idx]['sharpe']
        best_params = all_param_sets[best_idx]
        best_trades = all_results[best_idx]['trades']
        self._best_trades_today = best_trades

        # Collect all trades for regret analysis
        for r in all_results:
            self.todays_trades.extend(r['trades'])

        # Update brain with best iteration's trades
        for trade in best_trades:
            self.brain.update(trade)
            self.confidence_manager.record_trade(trade)

        if best_params:
            self.param_generator.update_best_params(best_params)

        execution_time = time.time() - start_time

        if best_trades:
            pnls = [t.pnl for t in best_trades]
            durations = [t.duration for t in best_trades]
            wins = sum(1 for t in best_trades if t.result == 'WIN')
            best_win_rate = wins / len(best_trades)
            best_pnl = sum(pnls)
            avg_duration = np.mean(durations)
        else:
            best_win_rate = 0.0
            best_pnl = 0.0
            avg_duration = 0.0

        return DayResults(
            day_number=day_number, date=date,
            total_iterations=self.config.iterations,
            best_iteration=best_idx,
            best_params=best_params or {},
            best_sharpe=best_sharpe,
            best_win_rate=best_win_rate,
            best_pnl=best_pnl,
            total_trades=len(best_trades),
            states_learned=len(self.brain.table),
            high_confidence_states=len(self.brain.get_all_states_above_threshold()),
            execution_time_seconds=execution_time,
            avg_duration=avg_duration
        )

    def _optimize_gpu_parallel(self, precomputed, day_data, all_param_sets, day_number):
        """
        GPU-PARALLEL: Run ALL iterations simultaneously on CUDA.
        Each iteration = different param thresholds applied to same precomputed states.
        All trade simulations run in parallel across iterations.
        """
        import torch
        device = torch.device('cuda')
        n_iters = len(all_param_sets)

        # Phase-aware threshold: in EXPLORATION phase, force threshold=0 so all structure_ok bars fire
        phase_threshold = self.confidence_manager.PHASES[self.confidence_manager.phase]['prob_threshold']
        if phase_threshold == 0:
            # EXPLORATION: take all trades
            thresholds = torch.zeros(n_iters, device=device, dtype=torch.float64)
        else:
            thresholds = torch.tensor(
                [min(p.get('confidence_threshold', 0.50), phase_threshold) for p in all_param_sets],
                device=device, dtype=torch.float64)
        tp_points = torch.tensor([p.get('take_profit_ticks', 40) * 0.25 for p in all_param_sets],
                                 device=device, dtype=torch.float64)
        sl_points = torch.tensor([p.get('stop_loss_ticks', 15) * 0.25 for p in all_param_sets],
                                 device=device, dtype=torch.float64)
        max_holds = torch.tensor([p.get('max_hold_seconds', 600) for p in all_param_sets],
                                 device=device, dtype=torch.float64)
        trade_costs = torch.tensor([p.get('trading_cost_points', 0.50) for p in all_param_sets],
                                   device=device, dtype=torch.float64)

        # DYNAMIC LOOKAHEAD: Ensure we look far enough to cover max_hold time
        # Take the maximum hold time across all parameter sets + buffer for regret analysis
        max_lookahead = int(max_holds.max().item() + 300)

        # Price/time arrays → GPU
        prices_gpu = torch.tensor(self._day_prices, device=device, dtype=torch.float64)
        # Convert timestamps to float seconds
        ts_raw = self._day_timestamps
        if hasattr(ts_raw[0], 'timestamp'):
            ts_float = np.array([t.timestamp() for t in ts_raw], dtype=np.float64)
        elif hasattr(ts_raw[0], 'item'):
            ts_float = np.array([pd.Timestamp(t).timestamp() for t in ts_raw], dtype=np.float64)
        else:
            ts_float = ts_raw.astype(np.float64)
        times_gpu = torch.tensor(ts_float, device=device, dtype=torch.float64)
        n_bars_total = len(self._day_prices)

        # Filter bars that pass structure_ok
        candidate_bars = [b for b in precomputed if b['structure_ok']]
        n_candidates = len(candidate_bars)

        phase_name = self.confidence_manager.PHASES[self.confidence_manager.phase]['name']
        print(f"  GPU parallel: {n_iters} iterations x {n_candidates} candidates on CUDA | Phase: {phase_name} (threshold={phase_threshold:.2f})")

        if n_candidates == 0:
            empty = [{'trades': [], 'sharpe': -999.0, 'win_rate': 0.0, 'pnl': 0.0} for _ in range(n_iters)]
            return 0, empty

        # Extract candidate probs + direction → GPU [n_candidates]
        cand_probs = torch.tensor([b['prob'] for b in candidate_bars], device=device, dtype=torch.float64)
        cand_indices = [b['bar_idx'] for b in candidate_bars]
        cand_prices_entry = torch.tensor([b['price'] for b in candidate_bars], device=device, dtype=torch.float64)
        # Direction: +1.0 for LONG, -1.0 for SHORT
        cand_dir_signs = torch.tensor(
            [-1.0 if b.get('direction', 'LONG') == 'SHORT' else 1.0 for b in candidate_bars],
            device=device, dtype=torch.float64
        )

        # Dynamic Slippage (GPU Pre-calculation)
        cand_velocities = torch.tensor([b['state'].particle_velocity for b in candidate_bars], device=device, dtype=torch.float64)
        cand_slippage = self.BASE_SLIPPAGE + self.VELOCITY_SLIPPAGE_FACTOR * torch.abs(cand_velocities)
        cand_slippage_cost = cand_slippage * 2.0 # Round trip

        # === CORE GPU KERNEL: simulate all trades in parallel ===
        # For each candidate × each iteration: compute trade result
        # Result tensors: [n_candidates, n_iters]
        trade_pnl = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.float64)
        trade_won = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.bool)
        trade_fired = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.bool)
        trade_duration = torch.zeros(n_candidates, n_iters, device=device, dtype=torch.float64)

        for c_idx in range(n_candidates):
            bar_idx = cand_indices[c_idx]
            prob = cand_probs[c_idx]
            entry_price = cand_prices_entry[c_idx]
            dir_sign = cand_dir_signs[c_idx]  # +1 LONG, -1 SHORT
            entry_time = times_gpu[bar_idx]

            # Which iterations fire? (prob >= threshold)
            fires = prob >= thresholds  # [n_iters] bool
            if not fires.any():
                continue

            # Look ahead window
            end_idx = min(n_bars_total, bar_idx + max_lookahead)
            if bar_idx + 1 >= end_idx:
                continue

            future_prices = prices_gpu[bar_idx + 1:end_idx]  # [window]
            future_times = times_gpu[bar_idx + 1:end_idx]    # [window]
            window_len = len(future_prices)

            # P&L for each future bar: direction-aware + trading cost
            raw_pnl = (future_prices - entry_price) * dir_sign  # [window]
            durations = future_times - entry_time  # [window]

            # For each firing iteration, find exit bar
            # Expand for broadcasting: raw_pnl [window,1] vs tp/sl [1,n_iters]
            # Include dynamic slippage cost [n_iters] (actually scalar for this candidate, but broadcasted)
            total_cost = trade_costs.unsqueeze(0) + cand_slippage_cost[c_idx]
            pnl_expanded = raw_pnl.unsqueeze(1) - total_cost  # [window, n_iters] with cost
            dur_expanded = durations.unsqueeze(1)        # [window, 1]
            tp_expanded = tp_points.unsqueeze(0)         # [1, n_iters]
            sl_expanded = sl_points.unsqueeze(0)         # [1, n_iters]
            mh_expanded = max_holds.unsqueeze(0)         # [1, n_iters]

            # Exit conditions: [window, n_iters]
            hit_tp = pnl_expanded >= tp_expanded
            hit_sl = pnl_expanded <= -sl_expanded
            hit_time = dur_expanded >= mh_expanded
            exit_mask = hit_tp | hit_sl | hit_time  # [window, n_iters]

            # Find first exit bar for each iteration
            any_exit = exit_mask.any(dim=0)  # [n_iters]

            # For iterations that have an exit
            firing_and_exiting = fires & any_exit
            if not firing_and_exiting.any():
                # All firing iterations reach EOD
                fire_indices = fires.nonzero(as_tuple=True)[0]
                last_pnl_per_iter = pnl_expanded[-1]  # [n_iters] includes cost
                last_dur = durations[-1]
                trade_fired[c_idx, fire_indices] = True
                trade_pnl[c_idx, fire_indices] = last_pnl_per_iter[fire_indices]
                trade_won[c_idx, fire_indices] = last_pnl_per_iter[fire_indices] > 0
                trade_duration[c_idx, fire_indices] = last_dur
                continue

            # Get first exit index per iteration: [n_iters]
            exit_indices = exit_mask.float().argmax(dim=0)  # [n_iters]

            for iter_idx in firing_and_exiting.nonzero(as_tuple=True)[0]:
                ii = iter_idx.item()
                exit_bar = exit_indices[ii].item()
                pnl_val = pnl_expanded[exit_bar, ii].item()
                dur_val = durations[exit_bar].item()

                if hit_tp[exit_bar, ii]:
                    final_pnl = tp_points[ii].item()  # TP already net of structure
                    won = True
                elif hit_sl[exit_bar, ii]:
                    final_pnl = -sl_points[ii].item()
                    won = False
                else:  # time exit
                    final_pnl = pnl_val
                    won = pnl_val > 0

                trade_fired[c_idx, ii] = True
                trade_pnl[c_idx, ii] = final_pnl
                trade_won[c_idx, ii] = won
                trade_duration[c_idx, ii] = dur_val

            # Handle firing but no-exit iterations (EOD)
            fire_no_exit = fires & ~any_exit
            if fire_no_exit.any():
                fe_indices = fire_no_exit.nonzero(as_tuple=True)[0]
                last_pnl_per_iter = pnl_expanded[-1]
                last_dur = durations[-1]
                trade_fired[c_idx, fe_indices] = True
                trade_pnl[c_idx, fe_indices] = last_pnl_per_iter[fe_indices]
                trade_won[c_idx, fe_indices] = last_pnl_per_iter[fe_indices] > 0
                trade_duration[c_idx, fe_indices] = last_dur

        # === CONVERT P&L from points to dollars ===
        trade_pnl *= self.asset.point_value

        # === AGGREGATE RESULTS PER ITERATION (on GPU) ===
        # trade_fired: [n_candidates, n_iters]
        n_trades = trade_fired.sum(dim=0)               # [n_iters]
        n_wins = (trade_fired & trade_won).sum(dim=0)   # [n_iters]
        total_pnl = (trade_pnl * trade_fired).sum(dim=0)  # [n_iters]

        # Sharpe per iteration (on GPU)
        # Need mean and std of PnL per iteration
        sharpes = torch.full((n_iters,), -999.0, device=device, dtype=torch.float64)
        for ii in range(n_iters):
            mask = trade_fired[:, ii]
            count = mask.sum().item()
            if count >= 5:
                pnls_iter = trade_pnl[mask, ii]
                mean_pnl = pnls_iter.mean()
                std_pnl = pnls_iter.std() + 1e-6
                sharpes[ii] = mean_pnl / std_pnl

        # Find best iteration
        best_idx = sharpes.argmax().item()
        best_sharpe = sharpes[best_idx].item()

        # Move results to CPU
        n_trades_cpu = n_trades.cpu().numpy()
        n_wins_cpu = n_wins.cpu().numpy()
        total_pnl_cpu = total_pnl.cpu().numpy()
        sharpes_cpu = sharpes.cpu().numpy()

        # Print top-5 iterations for visibility
        valid_mask = sharpes_cpu > -999.0
        if valid_mask.any():
            top_indices = np.argsort(sharpes_cpu)[::-1][:5]
            n_long = sum(1 for b in candidate_bars if b.get('direction') == 'LONG')
            n_short = sum(1 for b in candidate_bars if b.get('direction') == 'SHORT')
            print(f"  Candidates: {n_candidates} ({n_long} LONG, {n_short} SHORT) | "
                  f"Valid iters: {valid_mask.sum()}/{n_iters}")
            for rank, idx in enumerate(top_indices):
                if sharpes_cpu[idx] <= -999.0:
                    break
                n_t = int(n_trades_cpu[idx])
                wr = int(n_wins_cpu[idx]) / n_t if n_t > 0 else 0
                marker = " <-- BEST" if idx == best_idx else ""
                print(f"    #{rank+1} Iter {idx:4d} | Sharpe: {sharpes_cpu[idx]:6.2f} | "
                      f"WR: {wr:.1%} | Trades: {n_t:3d} | P&L: ${total_pnl_cpu[idx]:8.2f}{marker}")

        # Build result dicts — reconstruct TradeOutcome objects only for best iteration
        all_results = []
        for ii in range(n_iters):
            n = int(n_trades_cpu[ii])
            wr = int(n_wins_cpu[ii]) / n if n > 0 else 0.0
            all_results.append({
                'trades': [],  # Populated only for best below
                'sharpe': float(sharpes_cpu[ii]),
                'win_rate': wr,
                'pnl': float(total_pnl_cpu[ii]),
            })

        # Reconstruct TradeOutcome objects for the BEST iteration only
        best_trades = []
        best_mask = trade_fired[:, best_idx]
        for c_idx in range(n_candidates):
            if not best_mask[c_idx].item():
                continue
            bar = candidate_bars[c_idx]
            pnl_val = trade_pnl[c_idx, best_idx].item()
            won = trade_won[c_idx, best_idx].item()
            dur = trade_duration[c_idx, best_idx].item()
            entry_time_val = ts_float[bar['bar_idx']]
            direction = bar.get('direction', 'LONG')
            dir_sign = -1.0 if direction == 'SHORT' else 1.0

            best_trades.append(TradeOutcome(
                state=bar['state'],
                entry_price=bar['price'],
                exit_price=bar['price'] + pnl_val / dir_sign if dir_sign != 0 else bar['price'],
                pnl=pnl_val,
                result='WIN' if won else 'LOSS',
                timestamp=entry_time_val + dur,
                exit_reason='GPU',
                entry_time=entry_time_val,
                exit_time=entry_time_val + dur,
                duration=dur,
                direction=direction,
            ))

        all_results[best_idx]['trades'] = best_trades

        return best_idx, all_results

    def _optimize_cpu_sequential(self, precomputed, day_data, all_param_sets, day_number, date: str = "", total_days: int = 1):
        """CPU fallback: run iterations sequentially (still uses precomputed states)."""
        n_iters = len(all_param_sets)
        best_sharpe = -999.0
        best_idx = 0
        all_results = []
        best_trades = []
        best_pnl_so_far = 0.0
        best_win_rate = 0.0
        best_avg_duration = 0.0

        pbar = tqdm(range(n_iters), desc=f"Optimizing Day {day_number} (CPU)", ncols=100)

        for iteration in pbar:
            trades = self._simulate_from_precomputed(
                precomputed, day_data, all_param_sets[iteration], pbar, best_sharpe
            )

            if trades:
                pnls = [t.pnl for t in trades]
                wins = sum(1 for t in trades if t.result == 'WIN')
                win_rate = wins / len(trades)
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6)
            else:
                win_rate = 0.0
                sharpe = 0.0

            all_results.append({
                'trades': trades,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'pnl': sum(t.pnl for t in trades),
            })

            if sharpe > best_sharpe and len(trades) >= 5:
                best_sharpe = sharpe
                best_idx = iteration
                best_pnl_so_far = sum(t.pnl for t in trades)
                best_win_rate = win_rate
                best_trades = trades
                best_avg_duration = np.mean([t.duration for t in trades])
                tqdm.write(f"  [Iter {iteration:3d}] New best! Sharpe: {best_sharpe:.2f} | WR: {win_rate:.1%} | Trades: {len(trades)} | P&L: ${best_pnl_so_far:.2f}")

            # Update dashboard immediately with current best result
            if best_trades:
                current_best_result = DayResults(
                    day_number=day_number,
                    date=date,
                    total_iterations=self.config.iterations,
                    best_iteration=best_idx,
                    best_params=all_param_sets[best_idx],
                    best_sharpe=best_sharpe,
                    best_win_rate=best_win_rate,
                    best_pnl=best_pnl_so_far,
                    total_trades=len(best_trades),
                    states_learned=len(self.brain.table),
                    high_confidence_states=len(self.brain.get_all_states_above_threshold()),
                    execution_time_seconds=0.0,
                    avg_duration=best_avg_duration,
                    real_pnl=0.0
                )
                self._write_dashboard_json(None, current_best_result, total_days, current_day_trades=best_trades)

            pbar.set_postfix({
                'Trades': len(trades),
                'WR': f'{win_rate:.1%}',
                'Sharpe': f'{sharpe:.2f}',
                'Best': f'{best_sharpe:.2f}'
            })

        pbar.close()
        return best_idx, all_results

    def simulate_trading_day(self, day_data: pd.DataFrame, params: Dict[str, Any], on_trade=None) -> List[TradeOutcome]:
        """
        Fast simulation of trading day with given parameters

        No regret analysis overhead - just execute trades

        Args:
            day_data: OHLCV data for single day
            params: Parameter set to test
            on_trade: Optional callback called with each TradeOutcome as it fires

        Returns:
            List of TradeOutcome objects
        """
        trades = []

        # Need at least 21 bars for regression
        if len(day_data) < 21:
            return trades

        # Simulate bar-by-bar
        for i in range(21, len(day_data)):
            current_row = day_data.iloc[i]

            # Get macro context
            df_macro = day_data.iloc[i-21:i+1].copy()
            if 'price' in df_macro.columns and 'close' not in df_macro.columns:
                df_macro['close'] = df_macro['price']

            # Calculate state
            current_price = current_row['price'] if 'price' in current_row else current_row['close']
            current_volume = current_row.get('volume', 0)
            if pd.isna(current_volume):
                current_volume = 0.0

            state = self.engine.calculate_three_body_state(
                df_macro=df_macro,
                df_micro=df_macro,
                current_price=current_price,
                current_volume=float(current_volume),
                tick_velocity=0.0
            )

            # Decision (using parameters) — phase-aware threshold
            phase_threshold = self.confidence_manager.PHASES[self.confidence_manager.phase]['prob_threshold']
            min_prob = min(params.get('confidence_threshold', 0.50), phase_threshold) if phase_threshold > 0 else 0.0
            min_conf = 0.0 if self.confidence_manager.phase == 1 else 0.30

            prob = self.brain.get_probability(state)
            conf = self.brain.get_confidence(state)

            # Check if should fire
            should_fire = (
                state.lagrange_zone in ['L2_ROCHE', 'L3_ROCHE'] and
                state.structure_confirmed and
                state.cascade_detected and
                prob >= min_prob and
                conf >= min_conf
            )

            if should_fire:
                # Simulate trade
                outcome = self._simulate_trade(
                    current_idx=i,
                    entry_price=current_price,
                    data=day_data,
                    state=state,
                    params=params
                )

                if outcome:
                    trades.append(outcome)
                    if on_trade:
                        on_trade(outcome)

        return trades

    def _simulate_trade(self, current_idx: int, entry_price: float,
                       data: pd.DataFrame, state: Any, params: Dict[str, Any]) -> Optional[TradeOutcome]:
        """
        Simulate single trade with lookahead — direction-aware

        Uses params for stop loss and take profit
        Direction from Lagrange zone: L2_ROCHE → SHORT, L3_ROCHE → LONG
        """
        stop_loss = params.get('stop_loss_ticks', 15) * 0.25  # Convert ticks to points
        take_profit = params.get('take_profit_ticks', 40) * 0.25
        max_hold = params.get('max_hold_seconds', 600)
        trading_cost = params.get('trading_cost_points', 0.50)
        pv = self.asset.point_value  # Points → dollars (MNQ: $2.0/point)

        # Direction from z-score: positive z → SHORT (mean reversion down), negative z → LONG
        direction = 'SHORT' if state.z_score > 0 else 'LONG'
        dir_sign = -1.0 if direction == 'SHORT' else 1.0

        # Look ahead
        max_lookahead = int(max_hold + 100)
        end_idx = min(len(data), current_idx + max_lookahead)
        future_data = data.iloc[current_idx+1:end_idx]

        if future_data.empty:
            return None

        entry_time = data.iloc[current_idx].get('timestamp', 0)
        if isinstance(entry_time, pd.Timestamp):
            entry_time = entry_time.timestamp()

        # Dynamic Slippage (Single Trade)
        velocity = state.particle_velocity
        slippage = self.BASE_SLIPPAGE + self.VELOCITY_SLIPPAGE_FACTOR * abs(velocity)
        total_slippage = slippage * 2.0

        for idx, row in future_data.iterrows():
            price = row['price'] if 'price' in row else row['close']
            pnl = (price - entry_price) * dir_sign - trading_cost - total_slippage

            exit_time = row.get('timestamp', 0)
            if isinstance(exit_time, pd.Timestamp):
                exit_time = exit_time.timestamp()

            duration = exit_time - entry_time

            # Check TP/SL
            if pnl >= take_profit:
                return TradeOutcome(
                    state=state, entry_price=entry_price, exit_price=price,
                    pnl=take_profit * pv, result='WIN', timestamp=exit_time,
                    exit_reason='TP', entry_time=entry_time,
                    duration=duration, direction=direction
                )
            elif pnl <= -stop_loss:
                return TradeOutcome(
                    state=state, entry_price=entry_price, exit_price=price,
                    pnl=-stop_loss * pv, result='LOSS', timestamp=exit_time,
                    exit_reason='SL', entry_time=entry_time,
                    duration=duration, direction=direction
                )
            elif duration >= max_hold:
                return TradeOutcome(
                    state=state, entry_price=entry_price, exit_price=price,
                    pnl=pnl * pv, result='WIN' if pnl > 0 else 'LOSS', timestamp=exit_time,
                    exit_reason='TIME', entry_time=entry_time,
                    duration=duration, direction=direction
                )

        # Reached end of data
        last_price = future_data.iloc[-1]['price'] if 'price' in future_data.iloc[-1] else future_data.iloc[-1]['close']
        pnl = (last_price - entry_price) * dir_sign - trading_cost

        last_time = future_data.iloc[-1].get('timestamp', 0)
        if isinstance(last_time, pd.Timestamp):
            last_time = last_time.timestamp()

        return TradeOutcome(
            state=state, entry_price=entry_price, exit_price=last_price,
            pnl=pnl * pv, result='WIN' if pnl > 0 else 'LOSS', timestamp=last_time,
            exit_reason='EOD', entry_time=entry_time,
            duration=last_time - entry_time, direction=direction
        )

    def split_into_trading_days(self, data: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Split data into trading days"""
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

        # Extract date
        data['date'] = data['timestamp'].dt.date

        # Group by date
        days = []
        for date, day_df in data.groupby('date'):
            date_str = str(date)
            days.append((date_str, day_df.reset_index(drop=True)))

        return days

    def launch_dashboard(self):
        """Launch dashboard in background thread"""
        def run_dashboard():
            try:
                import tkinter as tk
                root = tk.Tk()
                self.dashboard = LiveDashboard(root)
                root.mainloop()
            except Exception as e:
                print(f"WARNING: Dashboard failed to launch: {e}")

        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        print("Dashboard launching in background...")
        time.sleep(2)  # Give it time to initialize

    def _write_dashboard_json(self, day_metrics, day_result: DayResults, total_days: int, current_day_trades: List[TradeOutcome] = None):
        """Write training_progress.json for the LiveDashboard to poll."""
        import json as _json

        json_path = os.path.join(os.path.dirname(__file__), 'training_progress.json')

        # Build CUMULATIVE best-iteration trades (not all iterations!)
        cumulative_trades = self._cumulative_best_trades.copy()
        if current_day_trades:
            cumulative_trades.extend(current_day_trades)

        trades_data = []
        for t in cumulative_trades:
            trades_data.append({
                'pnl': t.pnl,
                'result': t.result,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'duration': t.duration,
                'timestamp': t.timestamp,
            })

        # Compute cumulative stats from best trades only
        all_pnls = [t.pnl for t in cumulative_trades]
        total_pnl = sum(all_pnls)
        total_trades = len(all_pnls)
        total_wins = sum(1 for t in cumulative_trades if t.result == 'WIN')
        cum_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
        cum_sharpe = (np.mean(all_pnls) / (np.std(all_pnls) + 1e-6)) if total_trades >= 2 else 0.0
        avg_duration = np.mean([t.duration for t in cumulative_trades]) if total_trades > 0 else 0.0

        # Max drawdown
        cum_pnl = np.cumsum(all_pnls) if all_pnls else np.array([0.0])
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Per-day summary
        day_summaries = []
        for dr in self.day_results:
            day_summaries.append({
                'day': dr.day_number,
                'date': dr.date,
                'trades': dr.total_trades,
                'win_rate': dr.best_win_rate,
                'pnl': dr.best_pnl,
                'sharpe': dr.best_sharpe,
            })
        # Current day (not yet appended to self.day_results)
        day_summaries.append({
            'day': day_result.day_number,
            'date': day_result.date,
            'trades': day_result.total_trades,
            'win_rate': day_result.best_win_rate,
            'pnl': day_result.best_pnl,
            'sharpe': day_result.best_sharpe,
        })

        # Best params for display
        best_params_display = {}
        if day_result.best_params:
            best_params_display = {
                'TP': f"{day_result.best_params.get('take_profit_ticks', 0) * 0.25:.1f} pts",
                'SL': f"{day_result.best_params.get('stop_loss_ticks', 0) * 0.25:.1f} pts",
                'Threshold': f"{day_result.best_params.get('confidence_threshold', 0):.2f}",
                'MaxHold': f"{day_result.best_params.get('max_hold_seconds', 0)}s",
            }

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
            'best_params': best_params_display,
            'day_summaries': day_summaries,
            # Today's best iteration stats
            'today_trades': day_result.total_trades,
            'today_pnl': day_result.best_pnl,
            'today_win_rate': day_result.best_win_rate,
            'today_sharpe': day_result.best_sharpe,
        }

        try:
            with open(json_path, 'w') as f:
                _json.dump(payload, f, default=str)
        except Exception:
            pass  # Non-critical

    def save_checkpoint(self, day_number: int, date: str, day_result: DayResults):
        """Save brain and parameters to checkpoint"""
        # Save brain
        brain_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_brain.pkl")
        self.brain.save(brain_path)

        # Save best params
        if day_result.best_params:
            params_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_params.pkl")
            with open(params_path, 'wb') as f:
                pickle.dump(day_result.best_params, f)

        # Save day results
        results_path = os.path.join(self.checkpoint_dir, f"day_{day_number:03d}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(day_result, f)

        # Save dynamic binner (once, shared across all days)
        if self.dynamic_binner is not None:
            binner_path = os.path.join(self.checkpoint_dir, "dynamic_binner.pkl")
            if not os.path.exists(binner_path):
                self.dynamic_binner.save(binner_path)

    def print_final_summary(self):
        """Print comprehensive final summary"""
        self.progress_reporter.print_final_summary()

        # Pattern analysis report (original — may show nothing if min_samples too high)
        pattern_report = self.pattern_analyzer.generate_pattern_report(
            self.brain,
            self.day_results
        )
        print(pattern_report)

        # Comprehensive pattern report (coarse-bin analysis — always shows data)
        comprehensive_report = self.pattern_analyzer.generate_comprehensive_report(
            self.brain,
            self.day_results,
            self._cumulative_best_trades
        )
        print(comprehensive_report)

        # Save comprehensive report to file
        report_path = os.path.join(PROJECT_ROOT, 'debug_outputs', 'training_pattern_report.txt')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
            print(f"\nPattern report saved to: {report_path}")
        except Exception as e:
            print(f"WARNING: Could not save pattern report: {e}")

        # Save progress log
        log_path = os.path.join(self.checkpoint_dir, "training_log.json")
        self.progress_reporter.save_progress_log(log_path)


def check_and_install_requirements():
    """Auto-install requirements.txt if missing packages detected"""
    import subprocess
    requirements_path = os.path.join(PROJECT_ROOT, 'requirements.txt')
    if not os.path.exists(requirements_path):
        return

    print("Checking dependencies...")
    try:
        # pip install --quiet skips already-installed packages fast
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_path],
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
            except ImportError:
                print("Dependencies OK | WARNING: PyTorch not installed — install manually:")
                print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except subprocess.TimeoutExpired:
        print("WARNING: pip install timed out, continuing anyway...")
    except Exception as e:
        print(f"WARNING: Could not check dependencies: {e}")


def main():
    """Single entry point - command line interface"""
    parser = argparse.ArgumentParser(
        description="Bayesian-AI Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', default=r"DATA\glbx-mdp3-20250101-20260209.ohlcv-1s.parquet", help="Path to parquet data file")
    parser.add_argument('--iterations', type=int, default=1000, help="Iterations per day (default: 1000)")
    parser.add_argument('--max-days', type=int, default=None, help="Limit number of days")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--no-dashboard', action='store_true', help="Disable live dashboard")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency check")
    parser.add_argument('--exploration-mode', action='store_true', help="Enable unconstrained exploration mode")

    args = parser.parse_args()

    # Auto-install dependencies
    if not args.skip_deps:
        check_and_install_requirements()

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        if args.data.endswith('.parquet'):
            data = pd.read_parquet(args.data)
        else:
            # Try databento loader
            data = DatabentoLoader.load_data(args.data)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return 1

    print(f"Loaded {len(data):,} rows")

    # Create orchestrator
    orchestrator = BayesianTrainingOrchestrator(args)

    # Run training
    try:
        results = orchestrator.train(data)
        print("\n=== Training Complete ===")

        # Keep dashboard open if running
        if orchestrator.dashboard_thread and orchestrator.dashboard_thread.is_alive():
            print("Dashboard is open. Close the window to exit.")
            try:
                orchestrator.dashboard_thread.join()
            except KeyboardInterrupt:
                print("Dashboard closed.")

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
