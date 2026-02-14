"""
Batch Regret Analyzer - End-of-day multi-timeframe analysis
Analyzes all trades with extended price history to identify exit inefficiencies

Fractal Analysis:
- Reviews trade outcomes in 2 timeframes above the execution timeframe.
- Lookahead window = 5 bars of the higher timeframe.
- Example: 15s trade -> Check 60s (5 bars) and 5m (5 bars).

Direction-aware: uses trade.direction for correct peak/PnL computation
Context-aware: links exit efficiency to trend for targeted recommendations
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
    peak_favorable: float  # Best price achieved (on TF+1 sustained target)
    potential_max_pnl: float  # What we could have made
    pnl_left_on_table: float  # Missed opportunity
    gave_back_pnl: float  # Profit given back from peak
    exit_efficiency: float  # actual_pnl / potential_pnl
    regret_type: str  # 'optimal', 'closed_too_early_spike', 'closed_too_early_trend', 'closed_too_late'

    # Fractal peaks
    peak_current_tf: float = 0.0
    peak_tf1: float = 0.0
    peak_tf2: float = 0.0

    # Metadata
    timeframe_used: str = '15s'
    tf1_interval: str = '60s'
    tf2_interval: str = '5min'

    # Context
    state_hash: int = 0
    context: str = ''
    trend_15m: str = 'UNKNOWN'


class BatchRegretAnalyzer:
    """End-of-day regret analysis with fractal multi-timeframe context"""

    # Timeframe hierarchy for fractal analysis
    # Note: '60s' is preferred over '1m' to avoid pandas 'm' (month) vs 'min' (minute) ambiguity
    TIMEFRAME_HIERARCHY = ['5s', '15s', '60s', '5min', '15min', '1h']

    def __init__(self):
        self.analysis_history = []

    def _get_higher_timeframes(self, current_tf: str) -> Tuple[str, str]:
        """
        Get the next two higher timeframes from the hierarchy.
        If current_tf is at the top, returns the highest available.
        """
        try:
            # Normalize current_tf to match hierarchy (handle '5m' vs '5min')
            if current_tf == '5m': current_tf = '5min'
            if current_tf == '15m': current_tf = '15min'
            if current_tf == '1min': current_tf = '60s'

            idx = self.TIMEFRAME_HIERARCHY.index(current_tf)
        except ValueError:
            # Default fallback if unknown
            return '60s', '5min'

        tf1 = self.TIMEFRAME_HIERARCHY[min(idx + 1, len(self.TIMEFRAME_HIERARCHY) - 1)]
        tf2 = self.TIMEFRAME_HIERARCHY[min(idx + 2, len(self.TIMEFRAME_HIERARCHY) - 1)]

        # Ensure distinct if possible (handle edge case at top of hierarchy)
        if tf1 == tf2 and idx < len(self.TIMEFRAME_HIERARCHY) - 1:
             pass # Should not happen with logic above unless at very end

        return tf1, tf2

    def batch_analyze_day(self, all_trades: List, full_day_data: pd.DataFrame,
                          current_timeframe: str = '15s') -> Dict:
        """
        Analyze all trades from day with fractal multi-timeframe peak detection.

        Args:
            all_trades: List of TradeOutcome objects
            full_day_data: Complete OHLCV data for the day
            current_timeframe: The timeframe used for execution (e.g., '15s')

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

        # Determine fractal timeframes
        tf1, tf2 = self._get_higher_timeframes(current_timeframe)

        print(f"  Regret Analysis: Base={current_timeframe} -> TF+1={tf1}, TF+2={tf2}")

        # Resample to needed timeframes
        data_base = self._resample_data(day_data, current_timeframe)
        data_tf1 = self._resample_data(day_data, tf1)
        data_tf2 = self._resample_data(day_data, tf2)

        # Analyze each trade
        regret_markers = []
        for idx, trade in enumerate(all_trades):
            markers = self._analyze_single_trade_fractal(
                trade,
                data_base, data_tf1, data_tf2,
                current_timeframe, tf1, tf2,
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
            'early_exits_pct': analysis['too_early'] / max(len(regret_markers), 1) * 100,
            'late_exits_pct': analysis['too_late'] / max(len(regret_markers), 1) * 100,
            'patterns': patterns,
            'recommendations': recommendations,
            'regret_markers': regret_markers,
            'fractal_timeframes': {'base': current_timeframe, 'tf1': tf1, 'tf2': tf2}
        }

    def _analyze_single_trade_fractal(self, trade,
                                      data_base, data_tf1, data_tf2,
                                      base_tf_str, tf1_str, tf2_str,
                                      trade_id: int) -> Optional[RegretMarkers]:
        """
        Analyze single trade with fractal peak detection.
        Lookahead = 5 bars of each timeframe.
        """
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        entry_time = trade.entry_time
        exit_time = trade.exit_time

        try:
            # Convert float timestamps to pd.Timestamp
            if isinstance(entry_time, (int, float)):
                entry_ts = pd.Timestamp(entry_time, unit='s')
                exit_ts = pd.Timestamp(exit_time, unit='s')
            else:
                entry_ts = entry_time
                exit_ts = exit_time

            # Direction
            side = getattr(trade, 'direction', 'LONG').lower()
            if side not in ('long', 'short'):
                side = 'long'

            # Define lookahead windows (5 bars)
            # Helper to parse interval string to Timedelta
            def parse_interval(s):
                if s.endswith('s'): return pd.Timedelta(seconds=int(s[:-1]))
                # Handle 'min' vs 'm' ambiguity. Pandas 2.2+ dislikes 'm' for minutes.
                if s.endswith('min'): return pd.Timedelta(minutes=int(s[:-3]))
                if s.endswith('m') and not s.endswith('min'): return pd.Timedelta(minutes=int(s[:-1]))
                if s.endswith('h'): return pd.Timedelta(hours=int(s[:-1]))
                return pd.Timedelta(minutes=1)

            delta_base = parse_interval(base_tf_str) * 5
            delta_tf1 = parse_interval(tf1_str) * 5
            delta_tf2 = parse_interval(tf2_str) * 5

            # Find peaks on each timeframe with respective 5-bar lookahead
            peak_base = self._find_peak(data_base, entry_ts, exit_ts + delta_base, side)
            peak_tf1 = self._find_peak(data_tf1, entry_ts, exit_ts + delta_tf1, side)
            peak_tf2 = self._find_peak(data_tf2, entry_ts, exit_ts + delta_tf2, side)

            # Use TF+1 as the "True" target for efficiency calculation (sustained move)
            # Fallback to base if TF+1 unavailable
            true_peak = peak_tf1 if peak_tf1 is not None else (peak_base if peak_base is not None else entry_price)

            # Compute potential and actual PnL
            if side == 'long':
                potential_max_pnl = true_peak - entry_price
                actual_pnl = exit_price - entry_price
                gave_back = max(0, true_peak - exit_price)
            else:
                potential_max_pnl = entry_price - true_peak
                actual_pnl = entry_price - exit_price
                gave_back = max(0, exit_price - true_peak)

            potential_max_pnl = max(potential_max_pnl, 0.001)
            pnl_left_on_table = max(0, potential_max_pnl - actual_pnl)
            exit_efficiency = actual_pnl / potential_max_pnl if potential_max_pnl > 0 else 0.0

            # Classify exit type
            if exit_efficiency >= 0.90:
                regret_type = 'optimal'
            elif pnl_left_on_table > gave_back:
                # Exited too early.
                regret_type = 'closed_too_early'
            elif gave_back > pnl_left_on_table:
                # Held too long (gave back profit)
                regret_type = 'closed_too_late'
            else:
                # Equal (e.g. Winner where Peak > Exit > Entry, so Left == GaveBack)
                # Differentiate by checking if peak was AFTER exit (left on table) or BEFORE (gave back)
                # Since we don't have peak timestamp easily accessible here without re-finding,
                # we assume if it's a Fractal Peak (TF1) significantly higher than exit, it's Early Exit.
                if true_peak > (exit_price * 1.0005 if side == 'long' else exit_price * 0.9995):
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
                peak_favorable=true_peak,
                potential_max_pnl=potential_max_pnl,
                pnl_left_on_table=pnl_left_on_table,
                gave_back_pnl=gave_back,
                exit_efficiency=exit_efficiency,
                regret_type=regret_type,
                peak_current_tf=peak_base or 0.0,
                peak_tf1=peak_tf1 or 0.0,
                peak_tf2=peak_tf2 or 0.0,
                timeframe_used=base_tf_str,
                tf1_interval=tf1_str,
                tf2_interval=tf2_str,
                state_hash=hash(trade.state) if hasattr(trade, 'state') else 0,
                context=trade.exit_reason,
            )

        except Exception as e:
            # print(f"Error analyzing trade {trade_id}: {e}")
            return None

    def _find_peak(self, data: pd.DataFrame, entry_ts, end_ts, side: str) -> Optional[float]:
        """Find peak favorable price between entry and end_ts."""
        if data is None or data.empty:
            return None

        # Look in window [entry, end_ts]
        mask = (data.index >= entry_ts) & (data.index <= end_ts)
        window = data[mask]

        if window.empty:
            # Fallback: check closest bar after entry if window empty (e.g. sparse data)
            after = data[data.index >= entry_ts]
            if not after.empty:
                # Take first bar
                window = after.iloc[:1]
            else:
                return None

        if side == 'long':
            return float(window['high'].max()) if 'high' in window.columns else float(window['close'].max())
        else:
            return float(window['low'].min()) if 'low' in window.columns else float(window['close'].min())

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to broader timeframe"""
        # Convert custom '60s' -> '1min' for pandas if needed, though '60s' works
        # Standardize for pandas resampling

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
        """Identify patterns in exit inefficiencies"""
        if not regret_markers:
            return {}

        patterns = {}

        # Fractal Peak Analysis
        # Compare peaks across timeframes to see if moves continued
        fractal_stats = []
        for m in regret_markers:
            # Ratio of TF1 peak to Current TF peak
            # For LONG: Peak(TF1) / Peak(Base). If > 1.0, move continued.
            if m.side == 'long' and m.peak_current_tf > 0:
                ratio = m.peak_tf1 / m.peak_current_tf
            elif m.side == 'short' and m.peak_current_tf > 0:
                ratio = m.peak_current_tf / m.peak_tf1 # Inverse for short (lower is better)
            else:
                ratio = 1.0

            fractal_stats.append(ratio)

        patterns['fractal_continuation'] = {
            'avg_continuation_ratio': np.mean(fractal_stats) if fractal_stats else 1.0,
            'continuation_frequency': sum(1 for x in fractal_stats if x > 1.05) / len(fractal_stats) if fractal_stats else 0
        }

        # Regret distribution
        regret_dist = defaultdict(int)
        for marker in regret_markers:
            regret_dist[marker.regret_type] += 1
        patterns['regret_distribution'] = dict(regret_dist)

        return patterns

    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate actionable parameter adjustment recommendations"""
        recommendations = []

        if not patterns:
            return recommendations

        # Fractal recommendations
        fractal = patterns.get('fractal_continuation', {})
        continuation_freq = fractal.get('continuation_frequency', 0)

        if continuation_freq > 0.40:
            recommendations.append(
                f"Fractal Continuation High ({continuation_freq:.0%}): Moves often continue on higher timeframe. "
                "Consider increasing profit targets or trail distance."
            )

        # General regret
        dist = patterns.get('regret_distribution', {})
        total = sum(dist.values())
        if total > 0:
            early_pct = dist.get('closed_too_early', 0) / total
            if early_pct > 0.40:
                recommendations.append(
                    f"Early Exits ({early_pct:.0%}): Consistently leaving money on table. Relax trail stop."
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
            'fractal_timeframes': {}
        }

    def print_analysis(self, analysis: Dict):
        """Print formatted regret analysis report"""
        print(f"\n{'='*80}")
        print(f"BATCH REGRET ANALYSIS (Fractal)")
        print(f"{'='*80}")

        if analysis['analyzed_trades'] == 0:
            print("No trades to analyze.")
            return

        total = analysis['analyzed_trades']
        tfs = analysis.get('fractal_timeframes', {})
        print(f"Timeframes: Base={tfs.get('base')} -> TF1={tfs.get('tf1')} -> TF2={tfs.get('tf2')}")
        print(f"EXIT EFFICIENCY: {analysis['avg_exit_efficiency']:.1%}")

        # Regret type breakdown
        patterns = analysis.get('patterns', {})
        dist = patterns.get('regret_distribution', {})
        optimal = dist.get('optimal', 0)
        early = dist.get('closed_too_early', 0)
        late = dist.get('closed_too_late', 0)

        print(f"\n  EXIT TYPE BREAKDOWN:")
        print(f"    Optimal (>90% eff):     {optimal:3d}/{total} ({optimal/total:.0%})")
        print(f"    Early (left profit):    {early:3d}/{total} ({early/total:.0%})")
        print(f"    Late (gave back):       {late:3d}/{total} ({late/total:.0%})")

        if analysis['recommendations']:
            print(f"\n  RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"    - {rec}")


# Example usage
if __name__ == "__main__":
    analyzer = BatchRegretAnalyzer()
    print("Batch Regret Analyzer initialized (Fractal)")
