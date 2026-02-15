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
    regret_type: str  # 'closed_too_early', 'closed_too_late', 'optimal', 'wrong_direction'
    bars_held: int
    bars_to_peak: int


@dataclass
class PendingReview:
    """Pending trade for delayed analysis (5-bar lookahead)"""
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    side: str
    exit_reason: str
    tick_value: float
    review_due_time: float  # When to perform analysis
    price_history: List[Tuple[float, float]]  # [(timestamp, price), ...]


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
                    lookahead_end_time: Optional[float] = None) -> RegretMarkers:
        """
        Analyze trade exit quality
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            side: 'long' or 'short'
            exit_reason: Why trade closed
            price_history: List of (timestamp, price) tuples
            tick_value: Dollar per tick
            lookahead_end_time: If provided, find peak up to this time (delayed analysis)
            
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
        
        # Determine search window
        search_end = lookahead_end_time if lookahead_end_time is not None else exit_time

        for timestamp, price in price_history:
            if timestamp < entry_time:
                continue
            if timestamp > search_end:
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
            # If no potential profit, efficiency is 0.0 for losses, 1.0 for break-even/profit
            exit_efficiency = 1.0 if actual_pnl >= 0 else 0.0

        exit_efficiency = min(1.0, max(0.0, exit_efficiency))
        
        # Classify regret
        if actual_pnl < 0:
            regret_type = 'wrong_direction'
        elif exit_efficiency >= 0.90:
            regret_type = 'optimal'
        else:
            # If delayed analysis (lookahead) is available:
            # Check if True Peak occurred AFTER Exit
            if lookahead_end_time is not None and peak_time > exit_time:
                regret_type = 'closed_too_early'
            else:
                # If peak was before or at exit, we held too long
                regret_type = 'closed_too_late'
        
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
    stop_loss: float
    high_water_mark: float
    entry_layer_state: Union[StateVector, ThreeBodyQuantumState]


class WaveRider:
    """
    Adaptive position management with ML-driven exit optimization
    
    ENHANCED FEATURES:
    - Adaptive trailing stops (10/20/30 ticks based on profit)
    - Structure break detection (L7/L8 changes)
    - Post-trade regret analysis (Delayed 5-bar review)
    - Auto-calibrating trail stops every 10 trades
    - Price history tracking for learning
    """
    
    TIMEFRAME_HIERARCHY = ['5s', '15s', '60s', '5min', '15min', '1h']

    def __init__(self, asset_profile, trail_config: Optional[Dict] = None, timeframe: str = '15s'):
        """
        Initialize WaveRider
        
        Args:
            asset_profile: AssetProfile with tick_size, tick_value, point_value
            trail_config: Optional dict with 'tight', 'medium', 'wide' trail distances
                         Default: {'tight': 10, 'medium': 20, 'wide': 30}
            timeframe: Trading timeframe (e.g., '15s') for lookahead calculation
        """
        self.asset = asset_profile
        self.position: Optional[Position] = None
        self.trading_timeframe = timeframe
        
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
        
        # Statistics
        self.total_trades = 0
        self.trades_since_calibration = 0
        self.calibration_interval = 10

    def open_position(self, entry_price: float, side: str, 
                     state: Union[StateVector, ThreeBodyQuantumState],
                     stop_distance_ticks: int = 20):
        """
        Open new position
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            state: StateVector or ThreeBodyQuantumState at entry
            stop_distance_ticks: Initial stop distance (default 20)
        """
        stop_dist = stop_distance_ticks * self.asset.tick_size
        stop_loss = entry_price + stop_dist if side == 'short' else entry_price - stop_dist
        
        self.position = Position(
            entry_price=entry_price,
            entry_time=time.time(),
            side=side,
            stop_loss=stop_loss,
            high_water_mark=entry_price,
            entry_layer_state=state
        )
        
        # Reset price history
        self.price_history = [(time.time(), entry_price)]

    def update_trail(self, current_price: float, 
                    current_state: Union[StateVector, ThreeBodyQuantumState]) -> Dict:
        """
        Update trailing stop and check exit conditions
        
        ENHANCED: Now tracks price history for regret analysis
        
        Args:
            current_price: Current market price
            current_state: Current StateVector or ThreeBodyQuantumState
            
        Returns:
            Dict with 'should_exit', 'pnl', 'exit_reason', 'regret_markers' (if exit)
        """
        if not self.position:
            return {'should_exit': False}
        
        # Track price for regret analysis
        self.price_history.append((time.time(), current_price))
        
        # Keep last 200 ticks
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]

        # Update High Water Mark
        if self.position.side == 'short':
            profit = self.position.entry_price - current_price
            self.position.high_water_mark = min(self.position.high_water_mark, current_price)
        else:
            profit = current_price - self.position.entry_price
            self.position.high_water_mark = max(self.position.high_water_mark, current_price)

        profit_usd = profit * self.asset.point_value

        # Adaptive Trail logic (using calibrated config)
        if profit_usd < 50:
            trail_ticks = self.trail_config['tight']
        elif profit_usd < 150:
            trail_ticks = self.trail_config['medium']
        else:
            trail_ticks = self.trail_config['wide']

        trail_dist = trail_ticks * self.asset.tick_size
        new_stop = self.position.high_water_mark + trail_dist if self.position.side == 'short' else self.position.high_water_mark - trail_dist

        # Check Stop Hit or Structure Break
        stop_hit = (self.position.side == 'short' and current_price >= new_stop) or \
                   (self.position.side == 'long' and current_price <= new_stop)

        structure_broken = self._check_layer_breaks(current_state)

        if stop_hit or structure_broken:
            exit_reason = 'structure_break' if structure_broken else 'trail_stop'
            exit_time = time.time()
            
            # 1. Immediate "Preliminary" Analysis (no lookahead)
            # Used for trade logging and basic stats
            markers = self.regret_analyzer.analyze_exit(
                entry_price=self.position.entry_price,
                exit_price=current_price,
                entry_time=self.position.entry_time,
                exit_time=exit_time,
                side=self.position.side,
                exit_reason=exit_reason,
                price_history=self.price_history,
                tick_value=self.asset.tick_value
            )
            
            # 2. Schedule "Deep" Analysis (with lookahead)
            # Used for parameter calibration
            wait_seconds = self._get_wait_duration()
            review = PendingReview(
                entry_price=self.position.entry_price,
                exit_price=current_price,
                entry_time=self.position.entry_time,
                exit_time=exit_time,
                side=self.position.side,
                exit_reason=exit_reason,
                tick_value=self.asset.tick_value,
                review_due_time=exit_time + wait_seconds,
                price_history=list(self.price_history) # Copy
            )
            self.pending_reviews.append(review)

            # Visualize preliminary regret
            self._visualize_regret(markers, title="PRELIMINARY REGRET ANALYSIS")
            
            # Update statistics (count trade immediately)
            self.total_trades += 1
            
            # Note: Calibration now happens in process_pending_reviews after delayed analysis
            
            # Clear position and price history
            self.position = None
            self.price_history = []

            return {
                'should_exit': True,
                'exit_price': current_price,
                'exit_reason': exit_reason,
                'pnl': profit_usd,
                'regret_markers': markers
            }

        self.position.stop_loss = new_stop
        return {'should_exit': False, 'current_stop': new_stop}

    def process_pending_reviews(self, current_time: float, current_price: float):
        """
        Process pending trade reviews.
        Accumulate price data and trigger analysis when review window closes.
        """
        completed_indices = []

        for i, review in enumerate(self.pending_reviews):
            # Accumulate price
            review.price_history.append((current_time, current_price))

            # Check if review is due
            if current_time >= review.review_due_time:
                # Perform Delayed Analysis
                markers = self.regret_analyzer.analyze_exit(
                    entry_price=review.entry_price,
                    exit_price=review.exit_price,
                    entry_time=review.entry_time,
                    exit_time=review.exit_time,
                    side=review.side,
                    exit_reason=review.exit_reason,
                    price_history=review.price_history,
                    tick_value=review.tick_value,
                    lookahead_end_time=review.review_due_time
                )

                # Visualize Delayed Regret
                self._visualize_regret(markers, title=f"DELAYED REVIEW (Wait: {review.review_due_time - review.exit_time:.0f}s)")

                # Trigger Calibration (based on deep analysis)
                self.trades_since_calibration += 1
                if self.trades_since_calibration >= self.calibration_interval:
                    self._calibrate_trail_stops()
                    self.trades_since_calibration = 0

                completed_indices.append(i)

        # Remove completed
        for i in reversed(completed_indices):
            self.pending_reviews.pop(i)

    def _get_wait_duration(self) -> float:
        """Calculate wait duration (5 bars of higher timeframe)"""
        current_tf = self.trading_timeframe

        # Normalize
        if current_tf == '5m': current_tf = '5min'
        if current_tf == '15m': current_tf = '15min'
        if current_tf == '1min': current_tf = '60s'
        if current_tf == '1h': current_tf = '60m'

        try:
            # Find next higher TF
            if current_tf in self.TIMEFRAME_HIERARCHY:
                idx = self.TIMEFRAME_HIERARCHY.index(current_tf)
                higher_tf = self.TIMEFRAME_HIERARCHY[min(idx + 1, len(self.TIMEFRAME_HIERARCHY) - 1)]
            else:
                # Fallback logic
                higher_tf = '60s'
        except ValueError:
            higher_tf = '60s'

        # Parse duration
        seconds = 60
        if higher_tf.endswith('s'):
            seconds = int(higher_tf[:-1])
        elif higher_tf.endswith('min'):
            seconds = int(higher_tf[:-3]) * 60
        elif higher_tf.endswith('h'):
            seconds = int(higher_tf[:-1]) * 3600

        return seconds * 5.0 # 5 bars

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
    
    def _visualize_regret(self, markers: RegretMarkers, title: str = "POST-TRADE REGRET ANALYSIS"):
        """Print regret analysis"""
        print(f"\n{'='*60}")
        print(title)
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
            print(f"\n✓ KEEPING CURRENT TRAIL STOPS")
            print(f"Tight={self.trail_config['tight']}, Medium={self.trail_config['medium']}, Wide={self.trail_config['wide']}")
            print(f"Reason: {recs.get('reason', 'Exits are optimal')}")
        
        print(f"{'='*60}\n")
    
    def get_statistics(self) -> Dict:
        """Get position management statistics"""
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
