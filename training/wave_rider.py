"""
Bayesian-AI - Wave Rider Exit System with Regret Analysis
File: bayesian_ai/execution/wave_rider.py

ENHANCED: Now includes post-trade regret analysis for adaptive trail optimization
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Tuple, Literal
import numpy as np
from core.state_vector import StateVector
from core.three_body_state import ThreeBodyQuantumState
from core.quantum_field_engine import QuantumFieldEngine

# Fallback threshold if basin stats are zero
CST_FALLBACK_SIGMA_THRESHOLD = 4.5


@dataclass
class RegretMarkers:
    """Post-trade analysis markers"""
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    side: Literal['long', 'short']
    actual_pnl: float
    exit_reason: str
    peak_favorable: float
    peak_favorable_time: float
    potential_max_pnl: float
    pnl_left_on_table: float
    gave_back_pnl: float
    exit_efficiency: float
    regret_type: str  # 'closed_too_early', 'closed_too_late', 'optimal'
    bars_held: int
    bars_to_peak: int


@dataclass
class PendingReview:
    """Trade waiting for delayed regret analysis"""
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    review_end_time: float
    side: str
    exit_reason: str


class RegretAnalyzer:
    """Analyzes trade exits to learn optimal timing"""
    
    def __init__(self):
        self.regret_history: List[RegretMarkers] = []
    
    def analyze_exit(self, 
                    entry_price: float,
                    exit_price: float,
                    entry_time: float,
                    exit_time: float,
                    side: str,
                    exit_reason: str,
                    price_history: List[Tuple[float, float]],
                    tick_value: float,
                    review_end_time: Optional[float] = None) -> RegretMarkers:
        """
        Analyze trade exit quality with optional lookahead
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            side: 'long' or 'short'
            exit_reason: Why trade closed
            price_history: List of (timestamp, price) tuples
            tick_value: Dollar per tick
            review_end_time: Look ahead until this time for true peak
            
        Returns:
            RegretMarkers with analysis
        """
        # Calculate actual PnL
        if side == 'long':
            actual_pnl = (exit_price - entry_price) / 0.25 * tick_value
        else:
            actual_pnl = (entry_price - exit_price) / 0.25 * tick_value
        
        # Find peak favorable price
        peak_price = entry_price
        peak_time = entry_time
        
        # Use provided review_end_time or default to exit_time (no lookahead)
        search_end_time = review_end_time if review_end_time is not None else exit_time

        for timestamp, price in price_history:
            if timestamp < entry_time:
                continue
            if timestamp > search_end_time:
                break
            
            if side == 'long':
                if price > peak_price:
                    peak_price = price
                    peak_time = timestamp
            else:
                if price < peak_price:
                    peak_price = price
                    peak_time = timestamp
        
        # Calculate potential max PnL
        if side == 'long':
            potential_max_pnl = (peak_price - entry_price) / 0.25 * tick_value
        else:
            potential_max_pnl = (entry_price - peak_price) / 0.25 * tick_value
        
        # Regret metrics
        pnl_left = max(0, potential_max_pnl - actual_pnl)
        
        if side == 'long':
            gave_back = max(0, (peak_price - exit_price) / 0.25 * tick_value)
        else:
            gave_back = max(0, (exit_price - peak_price) / 0.25 * tick_value)
        
        # Exit efficiency
        if potential_max_pnl > 0:
            exit_efficiency = actual_pnl / potential_max_pnl
        else:
            # If no potential profit existed:
            # Loss -> 0.0 efficiency
            # Breakeven/Profit (rare if potential=0) -> 1.0 efficiency
            exit_efficiency = 1.0 if actual_pnl >= 0 else 0.0

        exit_efficiency = min(1.0, max(0.0, exit_efficiency))
        
        # Classify regret
        if actual_pnl < 0:
            regret_type = 'wrong_direction'
        elif exit_efficiency >= 0.90:
            regret_type = 'optimal'
        elif peak_time > exit_time:
            regret_type = 'closed_too_early'
        elif gave_back > potential_max_pnl * 0.20:
            regret_type = 'closed_too_late'
        else:
            regret_type = 'optimal'
        
        # Bar counts
        bars_held = len([t for t, _ in price_history if entry_time <= t <= exit_time])
        bars_to_peak = len([t for t, _ in price_history if entry_time <= t <= peak_time])
        
        markers = RegretMarkers(
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            actual_pnl=actual_pnl,
            exit_reason=exit_reason,
            peak_favorable=peak_price,
            peak_favorable_time=peak_time,
            potential_max_pnl=potential_max_pnl,
            pnl_left_on_table=pnl_left,
            gave_back_pnl=gave_back,
            exit_efficiency=exit_efficiency,
            regret_type=regret_type,
            bars_held=bars_held,
            bars_to_peak=bars_to_peak
        )
        
        self.regret_history.append(markers)
        return markers
    
    def get_exit_quality_stats(self) -> Dict:
        """Aggregate statistics on exit quality"""
        if not self.regret_history:
            return {}
        
        total = len(self.regret_history)
        closed_early = len([r for r in self.regret_history if r.regret_type == 'closed_too_early'])
        closed_late = len([r for r in self.regret_history if r.regret_type == 'closed_too_late'])
        optimal = len([r for r in self.regret_history if r.regret_type == 'optimal'])
        
        import numpy as np
        avg_efficiency = np.mean([r.exit_efficiency for r in self.regret_history])
        avg_left_on_table = np.mean([r.pnl_left_on_table for r in self.regret_history])
        avg_gave_back = np.mean([r.gave_back_pnl for r in self.regret_history])
        avg_bars_to_peak = np.mean([r.bars_to_peak for r in self.regret_history if r.potential_max_pnl > 0])
        
        return {
            'total_trades': total,
            'closed_too_early': closed_early,
            'closed_too_late': closed_late,
            'optimal_exits': optimal,
            'avg_exit_efficiency': avg_efficiency,
            'avg_pnl_left_on_table': avg_left_on_table,
            'avg_pnl_gave_back': avg_gave_back,
            'avg_bars_to_peak': avg_bars_to_peak,
            'percent_optimal': optimal / total if total > 0 else 0.0
        }
    
    def get_recommendations(self) -> Dict:
        """Suggest trail stop adjustments"""
        stats = self.get_exit_quality_stats()
        
        if not stats:
            return {'message': 'Insufficient data'}
        
        recommendations = {}
        
        # Too many early exits?
        if stats['closed_too_early'] > stats['total_trades'] * 0.4:
            recommendations['trail_adjustment'] = 'WIDEN'
            recommendations['reason'] = f"Closing too early in {stats['closed_too_early']}/{stats['total_trades']} trades"
            recommendations['suggestion'] = "Increase trail distance: 10→15, 20→25, 30→40 ticks"
        
        # Too many late exits?
        elif stats['closed_too_late'] > stats['total_trades'] * 0.4:
            recommendations['trail_adjustment'] = 'TIGHTEN'
            recommendations['reason'] = f"Closing too late in {stats['closed_too_late']}/{stats['total_trades']} trades"
            recommendations['suggestion'] = "Decrease trail distance: 30→20, 20→15, 10→8 ticks"
        
        else:
            recommendations['trail_adjustment'] = 'KEEP'
            recommendations['reason'] = f"Exit efficiency at {stats['avg_exit_efficiency']:.1%}"
            recommendations['suggestion'] = "Current trail stops are balanced"
        
        # Optimal hold time
        if 'avg_bars_to_peak' in stats:
            recommendations['typical_move_duration'] = f"{stats['avg_bars_to_peak']:.1f} bars"
        
        return recommendations


@dataclass
class Position:
    entry_price: float
    entry_time: float
    side: str  # 'long' or 'short'
    stop_loss: float          # initial wide hard stop (absolute price level)
    high_water_mark: float
    entry_layer_state: Union[StateVector, ThreeBodyQuantumState]
    template_id: Optional[int] = None
    profit_target: Optional[float] = None
    trailing_stop_ticks: Optional[int] = None
    trail_activation_ticks: Optional[int] = None  # profit ticks needed before trail engages
    original_trail_ticks: Optional[int] = None    # Store initial trail setting for reference
    is_trail_active: bool = False                 # True once profit > activation ticks
    last_adjustment_reason: Optional[str] = None  # belief reason that last tightened/widened trail
    breakeven_locked: bool = False                 # True once stop moved to entry (risk-free)
    breakeven_level: Optional[float] = None        # price level of the breakeven stop
    entry_dmi_inverse: bool = False                # True if DMI was against trade direction at entry
    bars_in_trade: int = 0                           # incremented each update_trail call

    # Trade-awareness (set at entry, updated each bar in update_trail)
    running_mfe_ticks: float = 0.0         # best favorable excursion so far
    running_mae_ticks: float = 0.0         # worst adverse excursion (positive = deeper drawdown)
    tmpl_expected_mfe_ticks: float = 0.0   # template mean_mfe_ticks (set at entry)
    tmpl_expected_hold_bars: float = 0.0   # template avg_mfe_bar (set at entry)
    tmpl_win_rate: float = 0.5             # template stats_win_rate (prior for P(profitable))

    # CST
    cst_centroid: Optional[np.ndarray] = None
    cst_basin_mean: float = 0.0
    cst_basin_std: float = 0.0
    cst_ancestry: Optional[Dict] = None
    cst_scaler_mean: Optional[np.ndarray] = None   # StandardScaler.mean_
    cst_scaler_scale: Optional[np.ndarray] = None   # StandardScaler.scale_


class WaveRider:
    """
    Adaptive position management with ML-driven exit optimization
    
    ENHANCED FEATURES:
    - Adaptive trailing stops (10/20/30 ticks based on profit)
    - Structure break detection (L7/L8 changes)
    - Post-trade regret analysis
    - Auto-calibrating trail stops every 10 trades
    - Price history tracking for learning
    """
    
    def __init__(self, asset_profile, trail_config: Optional[Dict] = None):
        """
        Initialize WaveRider
        
        Args:
            asset_profile: AssetProfile with tick_size, tick_value, point_value
            trail_config: Optional dict with 'tight', 'medium', 'wide' trail distances
                         Default: {'tight': 10, 'medium': 20, 'wide': 30}
        """
        self.asset = asset_profile
        self.position: Optional[Position] = None
        
        # Trail configuration (adaptive)
        if trail_config is None:
            self.trail_config = {
                'tight': 10,
                'medium': 20,
                'wide': 30
            }
        else:
            self.trail_config = trail_config
        
        # Regret analysis
        self.regret_analyzer = RegretAnalyzer()
        self.price_history: List[Tuple[float, float]] = []
        self.pending_reviews: List[PendingReview] = []
        self.completed_reviews: List[RegretMarkers] = []  # EOD bridge to playbooks
        self.review_wait_time = 300.0  # Default 5 minutes (can be overridden)

        # Statistics
        self.total_trades = 0

    # Dynamic Trail Constants
    # Gentle tightening (8% per bar instead of 30%) with a hard floor at 60% of original.
    # Pre-fix: 30% tighten per bar → trail collapses to 2 ticks in 4 bars → premature exit.
    MIN_TRAIL_TICKS = 4
    TIGHTEN_TRAIL_FACTOR = 0.92           # was 0.70 — gentle 8% per bar
    TRAIL_FLOOR_FACTOR   = 0.60           # trail never drops below 60% of original
    MAX_ORIGINAL_TRAIL_MULTIPLIER = 3     # allow wider extension when aligned
    WIDEN_TRAIL_FACTOR = 1.30             # was 1.20

    def open_position(self, entry_price: float, side: str,
                     state: Union[StateVector, ThreeBodyQuantumState],
                     stop_distance_ticks: int = 20,
                     profit_target_ticks: Optional[int] = None,
                     trailing_stop_ticks: Optional[int] = None,
                     trail_activation_ticks: Optional[int] = None,
                     template_id: Optional[int] = None,
                     cst_centroid: Optional[np.ndarray] = None,
                     cst_basin_mean: float = 0.0,
                     cst_basin_std: float = 0.0,
                     cst_ancestry: Optional[Dict] = None,
                     cst_scaler_mean: Optional[np.ndarray] = None,
                     cst_scaler_scale: Optional[np.ndarray] = None,
                     tmpl_expected_mfe_ticks: float = 0.0,
                     tmpl_expected_hold_bars: float = 0.0,
                     tmpl_win_rate: float = 0.5,
                     entry_time: Optional[float] = None):
        """
        Open new position

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            state: StateVector or ThreeBodyQuantumState at entry
            stop_distance_ticks: Initial WIDE hard stop -- held until trail activates
            profit_target_ticks: Optional profit target in ticks
            trailing_stop_ticks: Trail distance once the trail is active
            trail_activation_ticks: Profit ticks needed before trail engages.
                                    Until then the initial hard stop is used.
                                    None = trail active from bar 1 (legacy behaviour).
            template_id: ID of the template triggering this trade
            cst_centroid: Template centroid (SCALED space) for structural integrity checks
            cst_basin_mean: Mean distance of members to centroid (SCALED space)
            cst_basin_std: StdDev of member distances (SCALED space)
            cst_ancestry: Dictionary of static ancestry features for vector reconstruction
            cst_scaler_mean: StandardScaler.mean_ for scaling live vectors
            cst_scaler_scale: StandardScaler.scale_ for scaling live vectors
        """
        stop_dist = stop_distance_ticks * self.asset.tick_size
        stop_loss = entry_price + stop_dist if side == 'short' else entry_price - stop_dist
        
        profit_target = None
        if profit_target_ticks:
            pt_dist = profit_target_ticks * self.asset.tick_size
            profit_target = entry_price - pt_dist if side == 'short' else entry_price + pt_dist

        # A3. Lower Trail Activation Threshold
        # Proposed: Activation = max(3 ticks, TP * 0.15)
        # We override whatever was passed in trail_activation_ticks with this new logic
        if profit_target_ticks:
            calculated_activation = max(self.MIN_TRAIL_ACTIVATION_TICKS, int(profit_target_ticks * self.PROFIT_TARGET_ACTIVATION_FACTOR))
        else:
            calculated_activation = self.MIN_TRAIL_ACTIVATION_TICKS

        self.position = Position(
            entry_price=entry_price,
            entry_time=entry_time if entry_time is not None else time.time(),
            side=side,
            stop_loss=stop_loss,
            high_water_mark=entry_price,
            entry_layer_state=state,
            template_id=template_id,
            profit_target=profit_target,
            trailing_stop_ticks=trailing_stop_ticks,
            trail_activation_ticks=calculated_activation,
            original_trail_ticks=trailing_stop_ticks,
            is_trail_active=False,
            cst_centroid=cst_centroid,
            cst_basin_mean=cst_basin_mean,
            cst_basin_std=cst_basin_std,
            cst_ancestry=cst_ancestry,
            cst_scaler_mean=cst_scaler_mean,
            cst_scaler_scale=cst_scaler_scale,
            tmpl_expected_mfe_ticks=tmpl_expected_mfe_ticks,
            tmpl_expected_hold_bars=tmpl_expected_hold_bars,
            tmpl_win_rate=tmpl_win_rate,
        )
        
        # Note: Do not clear price_history here as we need it for delayed analysis
        # Just ensure it's not growing indefinitely (handled in update/process methods)

    def process_pending_reviews(self, current_time: float, current_price: float):
        """
        Process any pending reviews that have reached their wait time.
        Also updates price history buffer.
        """
        # Always track price for analysis (prevent duplicates if called multiple times per tick)
        if not self.price_history or self.price_history[-1] != (current_time, current_price):
            self.price_history.append((current_time, current_price))

        # Keep buffer manageable (e.g. last 5000 ticks ~ 1.5 hours at 1s resolution)
        if len(self.price_history) > 5000:
            self.price_history = self.price_history[-5000:]

        if not self.pending_reviews:
            return

        # Check for ripe reviews
        remaining_reviews = []
        for review in self.pending_reviews:
            if current_time >= review.review_end_time:
                # Time to analyze!
                markers = self.regret_analyzer.analyze_exit(
                    entry_price=review.entry_price,
                    exit_price=review.exit_price,
                    entry_time=review.entry_time,
                    exit_time=review.exit_time,
                    side=review.side,
                    exit_reason=review.exit_reason,
                    price_history=self.price_history,
                    tick_value=self.asset.tick_value,
                    review_end_time=review.review_end_time
                )

                # Visualize regret
                self._visualize_regret(markers)
                self.completed_reviews.append(markers)

                # Update statistics
                self.total_trades += 1
            else:
                remaining_reviews.append(review)

        self.pending_reviews = remaining_reviews

    def update_trail(self, current_price: float, 
                    current_state: Union[StateVector, ThreeBodyQuantumState],
                    timestamp: Optional[float] = None,
                    exit_signal: Optional[Dict] = None) -> Dict:
        """
        Update trailing stop and check exit conditions
        
        ENHANCED: Now uses delayed regret analysis
        
        Args:
            current_price: Current market price
            current_state: Current StateVector or ThreeBodyQuantumState
            timestamp: Optional timestamp (uses time.time() if None)
            exit_signal: Optional dict with exit recommendations
            
        Returns:
            Dict with 'should_exit', 'pnl', 'exit_reason', 'regret_markers' (if exit)
        """
        current_time = timestamp if timestamp is not None else time.time()

        if not self.position:
            # Still update price history for context if called
            self.process_pending_reviews(current_time, current_price)
            return {'should_exit': False}
        
        # Update history via process_pending_reviews
        self.process_pending_reviews(current_time, current_price)

        # Track how many bars this trade has been open
        self.position.bars_in_trade += 1

        # --- Dynamic Exit Logic ---
        urgent_exit = False
        decay_exit = False
        wave_maturity = 0.0

        if exit_signal:
            if exit_signal.get('urgent_exit'):
                urgent_exit = True
            if exit_signal.get('decay_exit'):
                decay_exit = True

            wave_maturity = exit_signal.get('wave_maturity', 0.0)

            # A4. Adaptive Trail Distance
            # Use physics to set distance based on wave maturity
            if self.position.original_trail_ticks is not None:
                base_trail = self.position.original_trail_ticks
                if wave_maturity < 0.3:
                    # Wide early — let trade develop
                    new_dist = int(base_trail * 1.5)
                    self.position.trailing_stop_ticks = new_dist
                    self.position.last_adjustment_reason = 'early_wave_widen'
                elif wave_maturity < 0.7:
                    # Standard
                    new_dist = int(base_trail * 1.0)
                    self.position.trailing_stop_ticks = new_dist
                    self.position.last_adjustment_reason = 'mid_wave_std'
                else:
                    # Tight late — protect gains
                    new_dist = int(base_trail * 0.5)
                    self.position.trailing_stop_ticks = max(self.MIN_TRAIL_TICKS, new_dist)
                    self.position.last_adjustment_reason = 'late_wave_tighten'

            # Legacy logic (tighten/widen overrides if explicitly set, though adaptive covers most)
            if exit_signal.get('tighten_trail') and self.position.trailing_stop_ticks is not None:
                # Reduce trail by TIGHTEN_TRAIL_FACTOR (gentle 8% per bar, was 30%).
                # Enforce floor: never below TRAIL_FLOOR_FACTOR × original.
                _orig = self.position.original_trail_ticks or self.position.trailing_stop_ticks
                _floor = max(self.MIN_TRAIL_TICKS, int(_orig * self.TRAIL_FLOOR_FACTOR))
                _new_trail = max(_floor, int(self.position.trailing_stop_ticks * self.TIGHTEN_TRAIL_FACTOR))
                self.position.trailing_stop_ticks = _new_trail
                self.position.last_adjustment_reason = exit_signal.get('reason', 'tighten')

            if exit_signal.get('widen_trail') and self.position.trailing_stop_ticks is not None:
                # Increase trail by WIDEN_TRAIL_FACTOR (max: original_trail × MAX_ORIGINAL_TRAIL_MULTIPLIER)
                _base = self.position.original_trail_ticks or self.position.trailing_stop_ticks
                _max_trail = _base * self.MAX_ORIGINAL_TRAIL_MULTIPLIER
                self.position.trailing_stop_ticks = min(_max_trail, int(self.position.trailing_stop_ticks * self.WIDEN_TRAIL_FACTOR))
                self.position.last_adjustment_reason = exit_signal.get('reason', 'widen')

        # Update High Water Mark
        if self.position.side == 'short':
            profit = self.position.entry_price - current_price
            self.position.high_water_mark = min(self.position.high_water_mark, current_price)
        else:
            profit = current_price - self.position.entry_price
            self.position.high_water_mark = max(self.position.high_water_mark, current_price)

        profit_usd = profit * self.asset.point_value
        profit_ticks = profit / self.asset.tick_size

        # Running MFE / MAE tracking (ticks)
        if profit_ticks > self.position.running_mfe_ticks:
            self.position.running_mfe_ticks = profit_ticks
        if profit_ticks < 0 and abs(profit_ticks) > self.position.running_mae_ticks:
            self.position.running_mae_ticks = abs(profit_ticks)

        # Breakeven lock: once profit ≥ original trail distance, move hard floor to entry+1 tick.
        # This ensures a confirmed winner can never turn into a loser.
        if (not self.position.breakeven_locked
                and self.position.original_trail_ticks is not None
                and profit_ticks >= self.position.original_trail_ticks):
            self.position.breakeven_locked = True
            _be_offset = self.asset.tick_size  # 1 tick above/below entry
            if self.position.side == 'long':
                self.position.breakeven_level = self.position.entry_price + _be_offset
            else:
                self.position.breakeven_level = self.position.entry_price - _be_offset

        # C1. Dynamic Profit Target (Runner Mode)
        # If trade reaches TP and conviction is still high: extend TP and tighten trail.
        # We check this BEFORE marking pt_hit, effectively intercepting the exit.
        if self.position.profit_target and exit_signal:
            _conv = exit_signal.get('conviction', 0.0)
            if _conv >= 0.6:
                if self.position.side == 'long':
                    if current_price >= self.position.profit_target:
                        # Extend
                        _curr_dist = abs(self.position.profit_target - self.position.entry_price)
                        self.position.profit_target = self.position.entry_price + (_curr_dist * 1.5)
                        if self.position.trailing_stop_ticks:
                            self.position.trailing_stop_ticks = max(self.MIN_TRAIL_TICKS, int(self.position.trailing_stop_ticks * 0.6))
                        self.position.last_adjustment_reason = 'runner_mode'
                else: # Short
                    if current_price <= self.position.profit_target:
                        # Extend
                        _curr_dist = abs(self.position.entry_price - self.position.profit_target)
                        self.position.profit_target = self.position.entry_price - (_curr_dist * 1.5)
                        if self.position.trailing_stop_ticks:
                            self.position.trailing_stop_ticks = max(self.MIN_TRAIL_TICKS, int(self.position.trailing_stop_ticks * 0.6))
                        self.position.last_adjustment_reason = 'runner_mode'

        # Check Profit Target
        pt_hit = False
        if self.position.profit_target:
             pt_hit = (self.position.side == 'short' and current_price <= self.position.profit_target) or \
                      (self.position.side == 'long' and current_price >= self.position.profit_target)

        # Two-phase stop logic
        # Phase 1 (initial): use wide hard stop until trail_activation_ticks profit reached
        # Phase 2 (trail):   once activated, trail from high_water_mark
        activation = self.position.trail_activation_ticks
        profit_ticks = profit / self.asset.tick_size  # points -> ticks

        if not self.position.is_trail_active:
            if activation is None or profit_ticks >= activation:
                self.position.is_trail_active = True

        if self.position.is_trail_active:
            # Trail logic: Fixed (from template) or Adaptive (from config)
            if self.position.trailing_stop_ticks:
                trail_ticks = self.position.trailing_stop_ticks
            else:
                if profit_usd < 50:
                    trail_ticks = self.trail_config['tight']
                elif profit_usd < 150:
                    trail_ticks = self.trail_config['medium']
                else:
                    trail_ticks = self.trail_config['wide']

            trail_dist = trail_ticks * self.asset.tick_size
            new_stop = (self.position.high_water_mark + trail_dist
                        if self.position.side == 'short'
                        else self.position.high_water_mark - trail_dist)
        else:
            # Phase 1: use the initial wide hard stop (absolute level set at entry)
            new_stop = self.position.stop_loss

        # Apply breakeven floor: stop can never be worse than entry+1 tick once locked
        if self.position.breakeven_locked and self.position.breakeven_level is not None:
            if self.position.side == 'long':
                new_stop = max(new_stop, self.position.breakeven_level)
            else:
                new_stop = min(new_stop, self.position.breakeven_level)

        # Loss watchdog: DMI inverse + underwater + workers agree on reversal
        # Triple confirmation prevents cutting on noise dips.
        # 8 ticks = $2.00 move = $4.00 PnL on MNQ -- filters out normal noise.
        WATCHDOG_TICKS = 8       # must be at least this far underwater
        WATCHDOG_WORKERS = 5     # at least N workers must disagree with trade side
        WATCHDOG_MIN_BARS = 5    # must hold at least N bars before watchdog can fire
        watchdog_exit = False
        if (self.position.bars_in_trade >= WATCHDOG_MIN_BARS
                and self.position.entry_dmi_inverse
                and profit_ticks <= -WATCHDOG_TICKS
                and exit_signal
                and exit_signal.get('workers_against', 0) >= WATCHDOG_WORKERS):
            watchdog_exit = True

        # Check Stop Hit, Profit Target, or Structure Break
        stop_hit = (self.position.side == 'short' and current_price >= new_stop) or \
                   (self.position.side == 'long' and current_price <= new_stop)

        structure_broken = self._check_layer_breaks(current_state)

        if stop_hit or structure_broken or pt_hit or urgent_exit or decay_exit or watchdog_exit:
            if urgent_exit:
                exit_reason = 'belief_flip'
            elif watchdog_exit:
                exit_reason = 'loss_watchdog'
            elif decay_exit:
                exit_reason = 'physics_decay'
            elif pt_hit:
                exit_reason = 'profit_target'
            elif structure_broken:
                exit_reason = 'structure_break'
            elif stop_hit:
                # A1. Separate SL from Trail Stop
                if self.position.is_trail_active:
                    exit_reason = 'trail_stop'
                else:
                    exit_reason = 'stop_loss'
            else:
                exit_reason = 'trail_stop'
            
            # QUEUE FOR REGRET ANALYSIS (Delayed)
            review = PendingReview(
                entry_price=self.position.entry_price,
                exit_price=current_price,
                entry_time=self.position.entry_time,
                exit_time=current_time,
                review_end_time=current_time + self.review_wait_time,
                side=self.position.side,
                exit_reason=exit_reason
            )
            self.pending_reviews.append(review)
            
            # Capture data for return (before clearing position)
            entry_price = self.position.entry_price
            entry_time = self.position.entry_time
            side = self.position.side
            _adj_reason = self.position.last_adjustment_reason or ''

            # Clear position (but NOT price history/pending reviews)
            self.position = None

            # Create a placeholder or partial markers object if needed immediately,
            # or just return None for markers. Orchestrator can handle it.
            # We'll return a partial object so orchestrator can log the trade.
            partial_markers = RegretMarkers(
                entry_price=entry_price,
                exit_price=current_price,
                entry_time=entry_time,
                exit_time=current_time,
                side=side,
                actual_pnl=profit_usd,
                exit_reason=exit_reason,
                peak_favorable=entry_price, # Placeholder
                peak_favorable_time=entry_time, # Placeholder
                potential_max_pnl=0.0, # Placeholder
                pnl_left_on_table=0.0,
                gave_back_pnl=0.0,
                exit_efficiency=0.0,
                regret_type='pending', # Indicate pending
                bars_held=0,
                bars_to_peak=0
            )

            return {
                'should_exit': True,
                'exit_price': current_price,
                'exit_reason': exit_reason,
                'adjustment_reason': _adj_reason,  # belief reason that last adjusted the trail
                'pnl': profit_usd,
                'regret_markers': partial_markers
            }

        self.position.stop_loss = new_stop
        return {'should_exit': False, 'current_stop': new_stop}

    def check_stops_hilo(self, high: float, low: float, timestamp: float) -> dict:
        """
        Lightweight intra-bar stop check using 1s high/low.
        Does NOT modify trail distances, does NOT increment bars_in_trade.
        Called from the 1s inner loop when a position is open.
        """
        if not self.position:
            return {'should_exit': False}

        pos = self.position

        # Update high water mark from 1s extremes
        if pos.side == 'long':
            pos.high_water_mark = max(pos.high_water_mark, high)
        else:
            pos.high_water_mark = min(pos.high_water_mark, low)

        fill_price = 0.0
        exit_reason = ''

        # Profit target check FIRST (if both PT and stop hit in same bar, PT wins)
        if pos.profit_target:
            if pos.side == 'long' and high >= pos.profit_target:
                fill_price = pos.profit_target
                exit_reason = 'profit_target'
            elif pos.side == 'short' and low <= pos.profit_target:
                fill_price = pos.profit_target
                exit_reason = 'profit_target'

        # Stop check (worst-case fill: gap-through uses actual extreme)
        if not exit_reason:
            if pos.side == 'long' and low <= pos.stop_loss:
                fill_price = min(pos.stop_loss, low)
                if pos.is_trail_active:
                    exit_reason = 'trail_stop'
                else:
                    exit_reason = 'stop_loss'
            elif pos.side == 'short' and high >= pos.stop_loss:
                fill_price = max(pos.stop_loss, high)
                if pos.is_trail_active:
                    exit_reason = 'trail_stop'
                else:
                    exit_reason = 'stop_loss'

        # Breakeven floor check
        if not exit_reason and pos.breakeven_locked and pos.breakeven_level is not None:
            if pos.side == 'long' and low <= pos.breakeven_level:
                fill_price = pos.breakeven_level
                exit_reason = 'trail_stop'
            elif pos.side == 'short' and high >= pos.breakeven_level:
                fill_price = pos.breakeven_level
                exit_reason = 'trail_stop'

        if exit_reason:
            if pos.side == 'long':
                pnl = (fill_price - pos.entry_price) * self.asset.point_value
            else:
                pnl = (pos.entry_price - fill_price) * self.asset.point_value

            review = PendingReview(
                entry_price=pos.entry_price, exit_price=fill_price,
                entry_time=pos.entry_time, exit_time=timestamp,
                review_end_time=timestamp + self.review_wait_time,
                side=pos.side, exit_reason=exit_reason
            )
            self.pending_reviews.append(review)
            _adj_reason = pos.last_adjustment_reason or ''
            self.position = None

            return {
                'should_exit': True, 'exit_price': fill_price,
                'exit_reason': exit_reason, 'exit_time': timestamp,
                'adjustment_reason': _adj_reason, 'pnl': pnl
            }

        return {'should_exit': False}

    def _check_layer_breaks(self, current: Union[StateVector, ThreeBodyQuantumState]) -> bool:
        """Check if market structure broke"""
        if isinstance(current, ThreeBodyQuantumState):
            return False  # Phase 0: No structure checks, rely on trail stop

        entry = self.position.entry_layer_state
        if isinstance(entry, StateVector):
            if entry.L7_pattern != current.L7_pattern:
                return True
            if not current.L8_confirm:
                return True
        
        return False
    
    def _visualize_regret(self, markers: RegretMarkers):
        """Print 1-line regret summary"""
        print(f"  Regret: {markers.regret_type.upper():>16s} | "
              f"Eff: {markers.exit_efficiency:>4.0%} | "
              f"PnL: ${markers.actual_pnl:>8.2f} | "
              f"Potential: ${markers.potential_max_pnl:>8.2f} | "
              f"Bars: {markers.bars_held}")
    
    def _calibrate_trail_stops(self):
        """Calibrate trail stops based on regret analysis"""
        recs = self.regret_analyzer.get_recommendations()
        
        print(f"\n{'='*60}")
        print(f"TRAIL STOP CALIBRATION - After {self.total_trades} Trades")
        print(f"{'='*60}")
        
        stats = self.regret_analyzer.get_exit_quality_stats()
        print(f"Exit Efficiency: {stats['avg_exit_efficiency']:.1%}")
        print(f"Closed Too Early: {stats['closed_too_early']}/{stats['total_trades']}")
        print(f"Closed Too Late: {stats['closed_too_late']}/{stats['total_trades']}")
        print(f"Optimal Exits: {stats['optimal_exits']}/{stats['total_trades']}")
        
        adjustment = recs.get('trail_adjustment', 'KEEP')
        
        if adjustment == 'WIDEN':
            old_config = self.trail_config.copy()
            self.trail_config['tight'] = min(15, self.trail_config['tight'] + 5)
            self.trail_config['medium'] = min(30, self.trail_config['medium'] + 5)
            self.trail_config['wide'] = min(50, self.trail_config['wide'] + 5)
            
            print(f"\n[WARN] WIDENING TRAIL STOPS")
            print(f"Old: Tight={old_config['tight']}, Medium={old_config['medium']}, Wide={old_config['wide']}")
            print(f"New: Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Closing too early')}")
        
        elif adjustment == 'TIGHTEN':
            old_config = self.trail_config.copy()
            self.trail_config['tight'] = max(5, self.trail_config['tight'] - 5)
            self.trail_config['medium'] = max(10, self.trail_config['medium'] - 5)
            self.trail_config['wide'] = max(15, self.trail_config['wide'] - 5)
            
            print(f"\n[WARN] TIGHTENING TRAIL STOPS")
            print(f"Old: Tight={old_config['tight']}, Medium={old_config['medium']}, Wide={old_config['wide']}")
            print(f"New: Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Holding too long')}")
        
        else:
            print(f"\n[OK] KEEPING CURRENT TRAIL STOPS")
            print(f"Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Exits are optimal')}")
        
        print(f"{'='*60}\n")
    
    def check_structural_integrity(self, current_state: ThreeBodyQuantumState,
                                    profit_ticks: float = 0.0,
                                    net_pressure: float = 0.0) -> bool:
        """
        Returns True if structural integrity is maintained (distance <= basin radius),
        False if structure is broken (tether snapped).

        All comparisons are done in SCALED space to match how basin_mean/basin_std
        were computed during clustering.

        profit_ticks: current unrealized P&L in ticks.
        net_pressure: continuous hold/exit pressure from belief network.
            > 0.3 = CST disabled (profitable + early + aligned)
            0 to 0.3 = wider sigma (4.0)
            -0.3 to 0 = standard sigma (3.0)
            < -0.3 = tight sigma (2.0, actively squeezing)
        """
        if not self.position or self.position.cst_centroid is None:
            return True

        # Pressure-controlled CST: strong hold pressure = CST disabled
        if net_pressure > 0.3:
            return True

        # Grace period: template-based minimum hold before CST can fire
        _grace = max(10, int(self.position.tmpl_expected_hold_bars / 3))
        if self.position.bars_in_trade < _grace:
            return True

        try:
            current_vec = np.array(QuantumFieldEngine.build_16d_vector(current_state, self.position.cst_ancestry))

            # Scale live vector to match centroid space
            if self.position.cst_scaler_mean is not None and self.position.cst_scaler_scale is not None:
                current_vec = (current_vec - self.position.cst_scaler_mean) / self.position.cst_scaler_scale

            dist = np.linalg.norm(current_vec - self.position.cst_centroid)

            # Pressure-modulated sigma
            if net_pressure > 0:
                _sigma = 4.0   # some hold pressure — wider
            elif net_pressure > -0.3:
                _sigma = 3.0   # neutral — standard
            else:
                _sigma = 2.0   # exit pressure — tighter

            threshold = self.position.cst_basin_mean + _sigma * self.position.cst_basin_std

            # Fallback for single-point basins
            if threshold < 1e-6:
                threshold = CST_FALLBACK_SIGMA_THRESHOLD

            return dist <= threshold
        except Exception as e:
            print(f"CST check failed: {e}")
            return True # Fail safe

    def get_statistics(self) -> Dict:
        """Get position management statistics."""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'current_trail_config': self.trail_config
            }
        
        stats = self.regret_analyzer.get_exit_quality_stats()
        
        return {
            'total_trades': self.total_trades,
            'current_trail_config': self.trail_config,
            'avg_exit_efficiency': stats.get('avg_exit_efficiency', 0.0),
            'closed_too_early_pct': stats.get('closed_too_early', 0) / stats.get('total_trades', 1) * 100,
            'closed_too_late_pct': stats.get('closed_too_late', 0) / stats.get('total_trades', 1) * 100,
            'optimal_exits_pct': stats.get('percent_optimal', 0.0) * 100,
            'avg_pnl_left': stats.get('avg_pnl_left_on_table', 0.0),
            'avg_pnl_gave_back': stats.get('avg_pnl_gave_back', 0.0),
            'typical_move_duration': stats.get('avg_bars_to_peak', 0.0)
        }
