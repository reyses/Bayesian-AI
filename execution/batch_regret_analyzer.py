"""
Batch Regret Analyzer - End-of-day multi-timeframe analysis
Analyzes all trades with extended price history to identify exit inefficiencies

Philosophy:
- Run AFTER trading day completes (no overhead during simulation)
- Use 2-minute resampled data for broader context
- Find patterns: "70% of exits were too early during trends"
- Generate parameter adjustment recommendations
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RegretMarkers:
    """Regret metrics for a single trade"""
    trade_id: int
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    side: str
    pnl: float
    result: str

    # Regret metrics
    peak_favorable: float  # Best price achieved
    potential_max_pnl: float  # What we could have made
    pnl_left_on_table: float  # Missed opportunity
    gave_back_pnl: float  # Profit given back from peak
    exit_efficiency: float  # actual_pnl / potential_pnl
    regret_type: str  # 'optimal', 'closed_too_early', 'closed_too_late'

    # Context
    state_hash: int
    context: str


class BatchRegretAnalyzer:
    """End-of-day regret analysis with multi-timeframe context"""

    def __init__(self):
        self.analysis_history = []

    def batch_analyze_day(self, all_trades: List, full_day_data: pd.DataFrame,
                          resample_to: str = '2min') -> Dict:
        """
        Analyze all trades from day with extended price history

        Args:
            all_trades: List of TradeOutcome objects
            full_day_data: Complete OHLCV data for the day
            resample_to: Timeframe to resample for broader context

        Returns:
            Dictionary with regret analysis results
        """
        if not all_trades:
            return self._empty_analysis()

        # Resample data to broader timeframe for context
        if 'timestamp' in full_day_data.columns:
            full_day_data = full_day_data.set_index('timestamp')

        resampled_data = self._resample_data(full_day_data, resample_to)

        # Analyze each trade
        regret_markers = []
        for idx, trade in enumerate(all_trades):
            markers = self._analyze_single_trade(
                trade,
                resampled_data,
                trade_id=idx
            )
            if markers:
                regret_markers.append(markers)

        # Aggregate analysis
        analysis = self._aggregate_analysis(regret_markers)

        # Find patterns
        patterns = self._find_exit_patterns(regret_markers)

        # Generate recommendations
        recommendations = self._generate_recommendations(patterns)

        return {
            'total_trades': len(all_trades),
            'analyzed_trades': len(regret_markers),
            'avg_exit_efficiency': analysis['avg_efficiency'],
            'exits_too_early': analysis['too_early'],
            'exits_too_late': analysis['too_late'],
            'exits_optimal': analysis['optimal'],
            'patterns': patterns,
            'recommendations': recommendations,
            'regret_markers': regret_markers
        }

    def _analyze_single_trade(self, trade, price_data: pd.DataFrame,
                              trade_id: int) -> RegretMarkers:
        """
        Analyze single trade with extended price history

        Args:
            trade: TradeOutcome object
            price_data: Resampled price data
            trade_id: Unique trade identifier

        Returns:
            RegretMarkers with analysis
        """
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        entry_time = trade.entry_time
        exit_time = trade.exit_time

        # Get price history after entry
        try:
            # Find data between entry and 5 minutes after exit
            mask = (price_data.index >= entry_time) & (price_data.index <= exit_time + 300)
            price_history = price_data[mask]

            if price_history.empty:
                return None

            # Determine side (assume long for now)
            side = 'long'  # TODO: Get from trade if available

            # Find peak favorable price
            if side == 'long':
                peak_favorable = price_history['high'].max()
                potential_max_pnl = peak_favorable - entry_price
            else:
                peak_favorable = price_history['low'].min()
                potential_max_pnl = entry_price - peak_favorable

            # Actual P&L
            actual_pnl = exit_price - entry_price if side == 'long' else entry_price - exit_price

            # Calculate regret metrics
            pnl_left_on_table = max(0, potential_max_pnl - actual_pnl)
            gave_back_pnl = max(0, (peak_favorable - exit_price) if side == 'long' else (exit_price - peak_favorable))

            # Exit efficiency (what % of potential did we capture?)
            exit_efficiency = actual_pnl / potential_max_pnl if potential_max_pnl > 0 else 0.0

            # Classify regret type
            if exit_efficiency >= 0.90:
                regret_type = 'optimal'
            elif pnl_left_on_table > gave_back_pnl:
                regret_type = 'closed_too_early'
            else:
                regret_type = 'closed_too_late'

            return RegretMarkers(
                trade_id=trade_id,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                side=side,
                pnl=actual_pnl,
                result=trade.result,
                peak_favorable=peak_favorable,
                potential_max_pnl=potential_max_pnl,
                pnl_left_on_table=pnl_left_on_table,
                gave_back_pnl=gave_back_pnl,
                exit_efficiency=exit_efficiency,
                regret_type=regret_type,
                state_hash=hash(trade.state),
                context=trade.exit_reason
            )

        except Exception as e:
            # If analysis fails, return None
            return None

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to broader timeframe"""
        if 'price' in data.columns:
            # If we only have price, use it for OHLC
            resampled = data['price'].resample(timeframe).agg(['first', 'max', 'min', 'last'])
            resampled.columns = ['open', 'high', 'low', 'close']
        else:
            # Standard OHLCV resampling
            resampled = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

        return resampled.dropna()

    def _aggregate_analysis(self, regret_markers: List[RegretMarkers]) -> Dict:
        """Aggregate regret analysis across all trades"""
        if not regret_markers:
            return self._empty_analysis()['aggregate']

        efficiencies = [r.exit_efficiency for r in regret_markers]
        too_early = sum(1 for r in regret_markers if r.regret_type == 'closed_too_early')
        too_late = sum(1 for r in regret_markers if r.regret_type == 'closed_too_late')
        optimal = sum(1 for r in regret_markers if r.regret_type == 'optimal')

        return {
            'avg_efficiency': np.mean(efficiencies),
            'median_efficiency': np.median(efficiencies),
            'too_early': too_early,
            'too_late': too_late,
            'optimal': optimal,
            'total_left_on_table': sum(r.pnl_left_on_table for r in regret_markers),
            'total_gave_back': sum(r.gave_back_pnl for r in regret_markers)
        }

    def _find_exit_patterns(self, regret_markers: List[RegretMarkers]) -> Dict[str, Any]:
        """Identify patterns in exit inefficiencies"""
        if not regret_markers:
            return {}

        patterns = {}

        # Pattern 1: Exit efficiency by context
        by_context = defaultdict(list)
        for marker in regret_markers:
            by_context[marker.context].append(marker.exit_efficiency)

        patterns['by_context'] = {
            context: {
                'avg_efficiency': np.mean(efficiencies),
                'count': len(efficiencies)
            }
            for context, efficiencies in by_context.items()
        }

        # Pattern 2: Regret type distribution
        regret_dist = defaultdict(int)
        for marker in regret_markers:
            regret_dist[marker.regret_type] += 1

        patterns['regret_distribution'] = dict(regret_dist)

        # Pattern 3: Winners vs Losers efficiency
        winners = [r.exit_efficiency for r in regret_markers if r.result == 'WIN']
        losers = [r.exit_efficiency for r in regret_markers if r.result == 'LOSS']

        patterns['by_outcome'] = {
            'winners_avg_efficiency': np.mean(winners) if winners else 0.0,
            'losers_avg_efficiency': np.mean(losers) if losers else 0.0
        }

        return patterns

    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate actionable parameter adjustment recommendations"""
        recommendations = []

        if not patterns:
            return recommendations

        # Check overall efficiency
        if 'by_outcome' in patterns:
            winner_eff = patterns['by_outcome']['winners_avg_efficiency']
            loser_eff = patterns['by_outcome']['losers_avg_efficiency']

            if winner_eff < 0.70:
                recommendations.append(
                    f"Winners: {winner_eff:.1%} efficiency - Consider widening trail stops to capture more profit"
                )

            if loser_eff > 0.50:
                recommendations.append(
                    f"Losers: {loser_eff:.1%} efficiency - Consider tightening stop losses to exit faster"
                )

        # Check regret distribution
        if 'regret_distribution' in patterns:
            dist = patterns['regret_distribution']
            total = sum(dist.values())

            too_early_pct = dist.get('closed_too_early', 0) / total if total > 0 else 0
            too_late_pct = dist.get('closed_too_late', 0) / total if total > 0 else 0

            if too_early_pct > 0.60:
                recommendations.append(
                    f"{too_early_pct:.0%} of exits too early - Increase trail_activation_profit threshold"
                )

            if too_late_pct > 0.60:
                recommendations.append(
                    f"{too_late_pct:.0%} of exits too late - Tighten trail_distance parameters"
                )

        # Check context-specific patterns
        if 'by_context' in patterns:
            for context, metrics in patterns['by_context'].items():
                if metrics['avg_efficiency'] < 0.50 and metrics['count'] >= 5:
                    recommendations.append(
                        f"{context} exits: {metrics['avg_efficiency']:.1%} efficiency - Review exit strategy for this context"
                    )

        return recommendations

    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'total_trades': 0,
            'analyzed_trades': 0,
            'avg_exit_efficiency': 0.0,
            'exits_too_early': 0,
            'exits_too_late': 0,
            'exits_optimal': 0,
            'patterns': {},
            'recommendations': [],
            'regret_markers': [],
            'aggregate': {
                'avg_efficiency': 0.0,
                'median_efficiency': 0.0,
                'too_early': 0,
                'too_late': 0,
                'optimal': 0,
                'total_left_on_table': 0.0,
                'total_gave_back': 0.0
            }
        }

    def print_analysis(self, analysis: Dict):
        """Print formatted regret analysis report"""
        print(f"\n{'='*80}")
        print(f"BATCH REGRET ANALYSIS")
        print(f"{'='*80}")

        if analysis['analyzed_trades'] == 0:
            print("No trades to analyze.")
            return

        print(f"\nEXIT EFFICIENCY:")
        print(f"  Average Efficiency: {analysis['avg_exit_efficiency']:>6.1%}")
        print(f"  Optimal Exits: {analysis['exits_optimal']:>5}/{analysis['analyzed_trades']}")
        print(f"  Too Early: {analysis['exits_too_early']:>5}/{analysis['analyzed_trades']}")
        print(f"  Too Late: {analysis['exits_too_late']:>5}/{analysis['analyzed_trades']}")

        if analysis['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")


# Example usage
if __name__ == "__main__":
    analyzer = BatchRegretAnalyzer()

    # Would normally have real trade data
    print("Batch Regret Analyzer initialized")
    print("Run batch_analyze_day() at end of each trading day")
