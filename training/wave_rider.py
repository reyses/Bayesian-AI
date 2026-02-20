"""
Bayesian-AI - Wave Rider Exit System with Regret Analysis
File: bayesian_ai/execution/wave_rider.py

ENHANCED: Now includes post-trade regret analysis for adaptive trail optimization
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Tuple, Literal
from core.state_vector import StateVector
from core.three_body_state import ThreeBodyQuantumState


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
        self.review_wait_time = 300.0  # Default 5 minutes (can be overridden)
        
        # Statistics
        self.total_trades = 0
        self.trades_since_calibration = 0
        self.calibration_interval = 10

    def open_position(self, entry_price: float, side: str,
                     state: Union[StateVector, ThreeBodyQuantumState],
                     stop_distance_ticks: int = 20,
                     profit_target_ticks: Optional[int] = None,
                     trailing_stop_ticks: Optional[int] = None,
                     trail_activation_ticks: Optional[int] = None,
                     template_id: Optional[int] = None):
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
        """
        stop_dist = stop_distance_ticks * self.asset.tick_size
        stop_loss = entry_price + stop_dist if side == 'short' else entry_price - stop_dist
        
        profit_target = None
        if profit_target_ticks:
            pt_dist = profit_target_ticks * self.asset.tick_size
            profit_target = entry_price - pt_dist if side == 'short' else entry_price + pt_dist

        self.position = Position(
            entry_price=entry_price,
            entry_time=time.time(),
            side=side,
            stop_loss=stop_loss,
            high_water_mark=entry_price,
            entry_layer_state=state,
            template_id=template_id,
            profit_target=profit_target,
            trailing_stop_ticks=trailing_stop_ticks,
            trail_activation_ticks=trail_activation_ticks,
            original_trail_ticks=trailing_stop_ticks,
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

                # Update statistics
                self.total_trades += 1
                self.trades_since_calibration += 1

                # Calibrate trail stops periodically
                if self.trades_since_calibration >= self.calibration_interval:
                    self._calibrate_trail_stops()
                    self.trades_since_calibration = 0
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

        # --- Dynamic Exit Logic ---
        urgent_exit = False
        if exit_signal:
            if exit_signal.get('urgent_exit'):
                urgent_exit = True

            if exit_signal.get('tighten_trail') and self.position.trailing_stop_ticks is not None:
                # Reduce trail by 30% (min: 2 ticks)
                _new_trail = max(2, int(self.position.trailing_stop_ticks * 0.70))
                self.position.trailing_stop_ticks = _new_trail

            if exit_signal.get('widen_trail') and self.position.trailing_stop_ticks is not None:
                # Increase trail by 20% (max: original_trail * 2.0)
                _base = self.position.original_trail_ticks or self.position.trailing_stop_ticks
                _max_trail = _base * 2
                self.position.trailing_stop_ticks = min(_max_trail, int(self.position.trailing_stop_ticks * 1.20))

        # Update High Water Mark
        if self.position.side == 'short':
            profit = self.position.entry_price - current_price
            self.position.high_water_mark = min(self.position.high_water_mark, current_price)
        else:
            profit = current_price - self.position.entry_price
            self.position.high_water_mark = max(self.position.high_water_mark, current_price)

        profit_usd = profit * self.asset.point_value

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

        trail_active = (activation is None) or (profit_ticks >= activation)

        if trail_active:
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

        # Check Stop Hit, Profit Target, or Structure Break
        stop_hit = (self.position.side == 'short' and current_price >= new_stop) or \
                   (self.position.side == 'long' and current_price <= new_stop)

        structure_broken = self._check_layer_breaks(current_state)

        if stop_hit or structure_broken or pt_hit or urgent_exit:
            if urgent_exit:
                exit_reason = 'belief_flip'
            elif pt_hit:
                exit_reason = 'profit_target'
            elif structure_broken:
                exit_reason = 'structure_break'
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
            
            # Capture data for return
            entry_price = self.position.entry_price
            entry_time = self.position.entry_time
            side = self.position.side
            
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
                'pnl': profit_usd,
                'regret_markers': partial_markers
            }

        self.position.stop_loss = new_stop
        return {'should_exit': False, 'current_stop': new_stop}

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
        """Print regret analysis"""
        print(f"\n{'='*60}")
        print(f"POST-TRADE REGRET ANALYSIS")
        print(f"{'='*60}")
        print(f"Side: {markers.side.upper()}")
        print(f"Entry: {markers.entry_price:.2f}")
        print(f"Exit:  {markers.exit_price:.2f} ({markers.exit_reason})")
        print(f"Peak:  {markers.peak_favorable:.2f}")
        print(f"-" * 60)
        print(f"Actual PnL:    ${markers.actual_pnl:>8.2f}")
        print(f"Potential Max: ${markers.potential_max_pnl:>8.2f}")
        print(f"Left on Table: ${markers.pnl_left_on_table:>8.2f}  {'⚠️  CLOSED TOO EARLY' if markers.pnl_left_on_table > 20 else ''}")
        print(f"Gave Back:     ${markers.gave_back_pnl:>8.2f}  {'⚠️  HELD TOO LONG' if markers.gave_back_pnl > 20 else ''}")
        print(f"-" * 60)
        print(f"Exit Efficiency: {markers.exit_efficiency:.1%}")
        print(f"Regret Type: {markers.regret_type.upper()}")
        print(f"Bars Held: {markers.bars_held} | Bars to Peak: {markers.bars_to_peak}")
        print(f"{'='*60}\n")
    
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
            
            print(f"\n⚠️ WIDENING TRAIL STOPS")
            print(f"Old: Tight={old_config['tight']}, Medium={old_config['medium']}, Wide={old_config['wide']}")
            print(f"New: Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Closing too early')}")
        
        elif adjustment == 'TIGHTEN':
            old_config = self.trail_config.copy()
            self.trail_config['tight'] = max(5, self.trail_config['tight'] - 5)
            self.trail_config['medium'] = max(10, self.trail_config['medium'] - 5)
            self.trail_config['wide'] = max(15, self.trail_config['wide'] - 5)
            
            print(f"\n⚠️ TIGHTENING TRAIL STOPS")
            print(f"Old: Tight={old_config['tight']}, Medium={old_config['medium']}, Wide={old_config['wide']}")
            print(f"New: Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Holding too long')}")
        
        else:
            print(f"\n[OK] KEEPING CURRENT TRAIL STOPS")
            print(f"Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Exits are optimal')}")
        
        print(f"{'='*60}\n")
    
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
