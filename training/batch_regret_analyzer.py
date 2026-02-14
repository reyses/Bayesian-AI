"""
Batch Regret Analyzer - End-of-day multi-timeframe analysis
Analyzes all trades with extended price history to identify exit inefficiencies

Multi-timeframe peak detection:
- 15s peaks: noisy, spike-sensitive
- 1m peaks: moderate noise filtering
- 2m peaks: "true" sustained target level
- 5m peaks: macro confirmation

Direction-aware: uses trade.direction for correct peak/PnL computation
Context-aware: links exit efficiency to 15m trend for targeted recommendations
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
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
    peak_favorable: float  # Best price achieved (on 2m "true" target)
    potential_max_pnl: float  # What we could have made
    pnl_left_on_table: float  # Missed opportunity
    gave_back_pnl: float  # Profit given back from peak
    exit_efficiency: float  # actual_pnl / potential_pnl
    regret_type: str  # 'optimal', 'closed_too_early_spike', 'closed_too_early_trend', 'closed_too_late'

    # Multi-TF peaks
    peak_15s: float = 0.0
    peak_1m: float = 0.0
    peak_2m: float = 0.0
    peak_5m: float = 0.0

    # Context
    state_hash: int = 0
    context: str = ''
    trend_15m: str = 'UNKNOWN'


class BatchRegretAnalyzer:
    """End-of-day regret analysis with multi-timeframe context"""

    def __init__(self):
        self.analysis_history = []

    def batch_analyze_day(self, all_trades: List, full_day_data: pd.DataFrame,
                          resample_to: str = '2min') -> Dict:
        """
        Analyze all trades from day with multi-timeframe peak detection.

        Resamples to 15s, 1m, 2m, 5m and finds peaks at each level.
        Uses 2m peak as "true" target (sustained level, not spike).

        Args:
            all_trades: List of TradeOutcome objects
            full_day_data: Complete OHLCV data for the day
            resample_to: Primary timeframe (kept for backwards compat)

        Returns:
            Dictionary with regret analysis results
        """
        if not all_trades:
            return self._empty_analysis()

        # Prepare data with datetime index
        day_data = full_day_data.copy()
        if 'timestamp' in day_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(day_data['timestamp']):
                day_data['timestamp'] = pd.to_datetime(pd.to_numeric(day_data['timestamp']), unit='s')
            day_data = day_data.set_index('timestamp')

        # Resample to multiple timeframes for peak detection
        data_15s = self._resample_data(day_data, '15s')
        data_1m = self._resample_data(day_data, '1min')
        data_2m = self._resample_data(day_data, '2min')
        data_5m = self._resample_data(day_data, '5min')

        # Analyze each trade
        regret_markers = []
        for idx, trade in enumerate(all_trades):
            markers = self._analyze_single_trade_mtf(
                trade, data_15s, data_1m, data_2m, data_5m, trade_id=idx
            )
            if markers:
                regret_markers.append(markers)

        # Aggregate analysis
        analysis = self._aggregate_analysis(regret_markers)

        # Find patterns (including context-aware patterns)
        patterns = self._find_exit_patterns(regret_markers)

        # Generate recommendations (context-aware)
        recommendations = self._generate_recommendations(patterns)

        return {
            'total_trades': len(all_trades),
            'analyzed_trades': len(regret_markers),
            'avg_exit_efficiency': analysis['avg_efficiency'],
            'exits_too_early': analysis['too_early'],
            'exits_too_late': analysis['too_late'],
            'exits_optimal': analysis['optimal'],
            'early_exits_pct': analysis['too_early'] / max(len(regret_markers), 1) * 100,
            'late_exits_pct': analysis['too_late'] / max(len(regret_markers), 1) * 100,
            'patterns': patterns,
            'recommendations': recommendations,
            'regret_markers': regret_markers
        }

    def _analyze_single_trade_mtf(self, trade, data_15s, data_1m, data_2m, data_5m,
                                    trade_id: int) -> Optional[RegretMarkers]:
        """
        Analyze single trade with multi-timeframe peak detection.
        Direction-aware: uses trade.direction for correct peak computation.
        """
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        entry_time = trade.entry_time
        exit_time = trade.exit_time

        try:
            # Convert float timestamps to pd.Timestamp
            if isinstance(entry_time, (int, float)):
                entry_ts = pd.Timestamp(entry_time, unit='s')
                exit_ts = pd.Timestamp(exit_time, unit='s') + pd.Timedelta(minutes=5)
            else:
                entry_ts = entry_time
                exit_ts = exit_time + pd.Timedelta(minutes=5)

            # Direction from trade (fixed: no longer always LONG)
            side = getattr(trade, 'direction', 'LONG').lower()
            if side not in ('long', 'short'):
                side = 'long'

            # Find peaks on each timeframe
            peak_15s = self._find_peak(data_15s, entry_ts, exit_ts, side)
            peak_1m = self._find_peak(data_1m, entry_ts, exit_ts, side)
            peak_2m = self._find_peak(data_2m, entry_ts, exit_ts, side)
            peak_5m = self._find_peak(data_5m, entry_ts, exit_ts, side)

            if peak_2m is None:
                # Fallback to any available peak
                true_peak = peak_1m or peak_15s
                if true_peak is None:
                    return None
            else:
                true_peak = peak_2m  # 2m is the "true" sustained target

            # Compute potential and actual PnL
            if side == 'long':
                potential_max_pnl = true_peak - entry_price
                actual_pnl = exit_price - entry_price
                gave_back = max(0, true_peak - exit_price)
            else:
                potential_max_pnl = entry_price - true_peak
                actual_pnl = entry_price - exit_price
                gave_back = max(0, exit_price - true_peak)

            potential_max_pnl = max(potential_max_pnl, 0.001)  # Prevent division by zero
            pnl_left_on_table = max(0, potential_max_pnl - actual_pnl)
            exit_efficiency = actual_pnl / potential_max_pnl if potential_max_pnl > 0 else 0.0

            # Classify exit type with multi-timeframe context
            if exit_efficiency >= 0.90:
                regret_type = 'optimal'
            elif pnl_left_on_table > gave_back:
                # Exited too early — but was it a spike or a trend?
                if peak_1m is not None and peak_2m is not None:
                    # If 1m peak much > 2m peak, it was a spike (noise)
                    if side == 'long':
                        spike_ratio = (peak_1m - entry_price) / max(peak_2m - entry_price, 0.001)
                    else:
                        spike_ratio = (entry_price - peak_1m) / max(entry_price - peak_2m, 0.001)

                    if spike_ratio > 1.05:
                        regret_type = 'closed_too_early_spike'
                    else:
                        regret_type = 'closed_too_early_trend'
                else:
                    regret_type = 'closed_too_early_trend'
            else:
                regret_type = 'closed_too_late'

            # Get 15m trend from trade state if available
            trend_15m = 'UNKNOWN'
            if hasattr(trade, 'state') and hasattr(trade.state, 'trend_direction_15m'):
                trend_15m = trade.state.trend_direction_15m

            return RegretMarkers(
                trade_id=trade_id,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                side=side,
                pnl=actual_pnl,
                result=trade.result,
                peak_favorable=true_peak,
                potential_max_pnl=potential_max_pnl,
                pnl_left_on_table=pnl_left_on_table,
                gave_back_pnl=gave_back,
                exit_efficiency=exit_efficiency,
                regret_type=regret_type,
                peak_15s=peak_15s or 0.0,
                peak_1m=peak_1m or 0.0,
                peak_2m=peak_2m or 0.0,
                peak_5m=peak_5m or 0.0,
                state_hash=hash(trade.state) if hasattr(trade, 'state') else 0,
                context=trade.exit_reason,
                trend_15m=trend_15m,
            )

        except Exception:
            return None

    def _find_peak(self, data: pd.DataFrame, entry_ts, exit_ts, side: str) -> Optional[float]:
        """Find peak favorable price between entry and extended exit on a given timeframe."""
        if data is None or data.empty:
            return None

        mask = (data.index >= entry_ts) & (data.index <= exit_ts)
        window = data[mask]

        if window.empty:
            return None

        if side == 'long':
            return float(window['high'].max()) if 'high' in window.columns else float(window['close'].max())
        else:
            return float(window['low'].min()) if 'low' in window.columns else float(window['close'].min())

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to broader timeframe"""
        if 'price' in data.columns:
            resampled = data['price'].resample(timeframe).agg(['first', 'max', 'min', 'last'])
            resampled.columns = ['open', 'high', 'low', 'close']
        elif all(c in data.columns for c in ['open', 'high', 'low', 'close']):
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            if 'volume' in data.columns:
                agg['volume'] = 'sum'
            resampled = data.resample(timeframe).agg(agg)
        else:
            col = 'close' if 'close' in data.columns else data.columns[0]
            resampled = data[col].resample(timeframe).agg(['first', 'max', 'min', 'last'])
            resampled.columns = ['open', 'high', 'low', 'close']

        return resampled.dropna()

    def _aggregate_analysis(self, regret_markers: List[RegretMarkers]) -> Dict:
        """Aggregate regret analysis across all trades"""
        if not regret_markers:
            return self._empty_analysis()['aggregate']

        efficiencies = [r.exit_efficiency for r in regret_markers]
        too_early = sum(1 for r in regret_markers if 'closed_too_early' in r.regret_type)
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
        """Identify patterns in exit inefficiencies — includes context-aware analysis"""
        if not regret_markers:
            return {}

        patterns = {}

        # Pattern 1: Exit efficiency by exit reason
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

        # Pattern 2: Regret type distribution (with sub-categories)
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

        # Pattern 4: Direction breakdown
        by_side = defaultdict(list)
        for marker in regret_markers:
            by_side[marker.side].append(marker.exit_efficiency)

        patterns['by_direction'] = {
            side: {
                'avg_efficiency': np.mean(effs),
                'count': len(effs)
            }
            for side, effs in by_side.items()
        }

        # Pattern 5: Context-aware — exit efficiency by 15m trend direction
        by_trend = defaultdict(list)
        for marker in regret_markers:
            by_trend[marker.trend_15m].append(marker)

        trend_analysis = {}
        for trend, markers in by_trend.items():
            effs = [m.exit_efficiency for m in markers]
            early = sum(1 for m in markers if 'closed_too_early' in m.regret_type)
            late = sum(1 for m in markers if m.regret_type == 'closed_too_late')
            trend_analysis[trend] = {
                'avg_efficiency': np.mean(effs),
                'count': len(markers),
                'early_pct': early / max(len(markers), 1),
                'late_pct': late / max(len(markers), 1),
            }

        patterns['by_15m_trend'] = trend_analysis

        # Pattern 6: Multi-TF peak deviation analysis
        if any(m.peak_2m > 0 for m in regret_markers):
            peaks_15s = [m.peak_15s for m in regret_markers if m.peak_15s > 0 and m.peak_2m > 0]
            peaks_1m = [m.peak_1m for m in regret_markers if m.peak_1m > 0 and m.peak_2m > 0]
            peaks_2m = [m.peak_2m for m in regret_markers if m.peak_2m > 0]
            peaks_5m = [m.peak_5m for m in regret_markers if m.peak_5m > 0 and m.peak_2m > 0]

            if peaks_2m:
                patterns['peak_deviation'] = {
                    '15s_vs_2m': np.mean([abs(a - b) / max(abs(b), 0.001)
                                          for a, b in zip(peaks_15s, peaks_2m)]) if peaks_15s else 0,
                    '1m_vs_2m': np.mean([abs(a - b) / max(abs(b), 0.001)
                                         for a, b in zip(peaks_1m, peaks_2m)]) if peaks_1m else 0,
                    '5m_vs_2m': np.mean([abs(a - b) / max(abs(b), 0.001)
                                         for a, b in zip(peaks_5m, peaks_2m)]) if peaks_5m else 0,
                }

        return patterns

    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate actionable parameter adjustment recommendations — context-aware"""
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

            early_spike = dist.get('closed_too_early_spike', 0) / total if total > 0 else 0
            early_trend = dist.get('closed_too_early_trend', 0) / total if total > 0 else 0
            too_late_pct = dist.get('closed_too_late', 0) / total if total > 0 else 0

            if early_trend > 0.30:
                recommendations.append(
                    f"{early_trend:.0%} early exits in trends - WIDEN trail stops by 5-10 ticks during sustained moves"
                )

            if early_spike > 0.20:
                recommendations.append(
                    f"{early_spike:.0%} early exits at spikes - Acceptable (noise, not trend)"
                )

            if too_late_pct > 0.40:
                recommendations.append(
                    f"{too_late_pct:.0%} late exits - Consider tightening max hold time or trail distance"
                )

        # Context-aware: 15m trend recommendations
        if 'by_15m_trend' in patterns:
            for trend, stats in patterns['by_15m_trend'].items():
                if stats['count'] < 5:
                    continue

                if trend in ('UP', 'DOWN') and stats['early_pct'] > 0.40:
                    recommendations.append(
                        f"In 15m {trend}: {stats['avg_efficiency']:.1%} eff, {stats['early_pct']:.0%} early exits "
                        f"- Widen stops when 15m trending"
                    )

                if trend == 'RANGE' and stats['late_pct'] > 0.40:
                    recommendations.append(
                        f"In 15m RANGE: {stats['avg_efficiency']:.1%} eff, {stats['late_pct']:.0%} late exits "
                        f"- Tighten stops in range conditions"
                    )

        # Direction-specific recommendations
        if 'by_direction' in patterns:
            for direction, stats in patterns['by_direction'].items():
                if stats['count'] >= 5 and stats['avg_efficiency'] < 0.50:
                    recommendations.append(
                        f"{direction.upper()} trades: {stats['avg_efficiency']:.1%} efficiency ({stats['count']} trades) "
                        f"- Review {direction} entry quality"
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
            'early_exits_pct': 0,
            'late_exits_pct': 0,
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
        """Print formatted regret analysis report with multi-TF context"""
        print(f"\n{'='*80}")
        print(f"BATCH REGRET ANALYSIS (Multi-Timeframe)")
        print(f"{'='*80}")

        if analysis['analyzed_trades'] == 0:
            print("No trades to analyze.")
            return

        total = analysis['analyzed_trades']

        print(f"\nEXIT EFFICIENCY: {analysis['avg_exit_efficiency']:.1%}")

        # Regret type breakdown
        patterns = analysis.get('patterns', {})
        dist = patterns.get('regret_distribution', {})
        optimal = dist.get('optimal', 0)
        early_spike = dist.get('closed_too_early_spike', 0)
        early_trend = dist.get('closed_too_early_trend', 0)
        late = dist.get('closed_too_late', 0)

        print(f"\n  EXIT TYPE BREAKDOWN:")
        print(f"    Optimal (>90% eff):     {optimal:3d}/{total} ({optimal/total:.0%})")
        print(f"    Early (spike):          {early_spike:3d}/{total} ({early_spike/total:.0%})")
        print(f"    Early (trend):          {early_trend:3d}/{total} ({early_trend/total:.0%})")
        print(f"    Late (gave back):       {late:3d}/{total} ({late/total:.0%})")

        # Direction breakdown
        by_dir = patterns.get('by_direction', {})
        if by_dir:
            print(f"\n  DIRECTION BREAKDOWN:")
            for direction, stats in by_dir.items():
                print(f"    {direction.upper():5s}: {stats['avg_efficiency']:.1%} eff | {stats['count']} trades")

        # 15m trend context
        by_trend = patterns.get('by_15m_trend', {})
        if by_trend:
            print(f"\n  CONTEXT (15m Trend):")
            for trend, stats in by_trend.items():
                if stats['count'] >= 3:
                    print(f"    {trend:8s}: {stats['avg_efficiency']:.1%} eff | "
                          f"{stats['count']:3d} trades | "
                          f"Early: {stats['early_pct']:.0%} | Late: {stats['late_pct']:.0%}")

        # Peak deviation
        peak_dev = patterns.get('peak_deviation', {})
        if peak_dev:
            print(f"\n  PEAK ANALYSIS:")
            print(f"    15s vs 2m deviation: {peak_dev.get('15s_vs_2m', 0):.1%} (noise)")
            print(f"    1m  vs 2m deviation: {peak_dev.get('1m_vs_2m', 0):.1%}")
            print(f"    5m  vs 2m deviation: {peak_dev.get('5m_vs_2m', 0):.1%} (macro)")

        if analysis['recommendations']:
            print(f"\n  RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"    - {rec}")


# Example usage
if __name__ == "__main__":
    analyzer = BatchRegretAnalyzer()
    print("Batch Regret Analyzer initialized (Multi-Timeframe)")
    print("Run batch_analyze_day() at end of each trading day")
