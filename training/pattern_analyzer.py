"""
Pattern Analyzer - Identify strongest learned patterns
Analyzes BayesianBrain state table to find high-performing configurations
"""
from typing import List, Dict, Any, Tuple
import pandas as pd
from collections import defaultdict


class PatternAnalyzer:
    """Analyzes learned states and identifies strongest patterns"""

    def __init__(self):
        self.analysis_history = []

    def get_strongest_patterns(self, brain, top_n: int = 20, min_samples: int = 10) -> List[Dict]:
        """
        Find top N states by win rate with sufficient sample size

        Args:
            brain: QuantumBayesianBrain instance
            top_n: Number of top patterns to return
            min_samples: Minimum trades required per state

        Returns:
            List of pattern dictionaries sorted by win rate
        """
        patterns = []

        for state, record in brain.table.items():
            total = record['total']
            if total < min_samples:
                continue

            wins = record['wins']
            losses = record['losses']
            win_rate = wins / total if total > 0 else 0.0

            # Calculate average P&L if we have trade history
            state_trades = [t for t in brain.trade_history if t.state == state]
            avg_pnl = sum(t.pnl for t in state_trades) / len(state_trades) if state_trades else 0.0
            avg_duration = sum(t.duration for t in state_trades) / len(state_trades) if state_trades else 0.0

            # Get state details
            state_str = self._format_state(state)
            lagrange_zone = getattr(state, 'lagrange_zone', 'unknown')

            patterns.append({
                'state': state,
                'state_str': state_str,
                'lagrange_zone': lagrange_zone,
                'wins': wins,
                'losses': losses,
                'total': total,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'avg_duration': avg_duration,
                'confidence': brain.get_confidence(state),
                'probability': brain.get_probability(state)
            })

        # Sort by win rate, then by total trades
        patterns.sort(key=lambda x: (x['win_rate'], x['total']), reverse=True)

        return patterns[:top_n]

    def analyze_by_context(self, brain) -> Dict[str, Dict]:
        """
        Breakdown performance by context (killzone, pattern type, etc.)

        Returns:
            Dictionary of context breakdowns
        """
        contexts = defaultdict(lambda: {'wins': 0, 'losses': 0, 'trades': []})

        for state, record in brain.table.items():
            # Categorize by Lagrange zone
            lagrange_zone = getattr(state, 'lagrange_zone', 'unknown')
            contexts[f'lagrange_{lagrange_zone}']['wins'] += record['wins']
            contexts[f'lagrange_{lagrange_zone}']['losses'] += record['losses']

            # Categorize by structure confirmation
            if hasattr(state, 'structure_confirmed'):
                key = 'structure_confirmed' if state.structure_confirmed else 'structure_unconfirmed'
                contexts[key]['wins'] += record['wins']
                contexts[key]['losses'] += record['losses']

            # Categorize by cascade detection
            if hasattr(state, 'cascade_detected'):
                key = 'cascade_yes' if state.cascade_detected else 'cascade_no'
                contexts[key]['wins'] += record['wins']
                contexts[key]['losses'] += record['losses']

        # Calculate win rates
        for context, data in contexts.items():
            total = data['wins'] + data['losses']
            data['total'] = total
            data['win_rate'] = data['wins'] / total if total > 0 else 0.0

        return dict(contexts)

    def analyze_by_lagrange_zone(self, brain) -> Dict[str, Dict]:
        """
        Breakdown performance by Lagrange zones (L1, L2, L3, etc.)

        Returns:
            Dictionary mapping zone → performance metrics
        """
        zones = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnls': []})

        for state, record in brain.table.items():
            if hasattr(state, 'lagrange_zone'):
                zone = state.lagrange_zone
                zones[zone]['wins'] += record['wins']
                zones[zone]['losses'] += record['losses']

                # Get PNLs for this zone
                state_trades = [t for t in brain.trade_history if t.state == state]
                zones[zone]['pnls'].extend([t.pnl for t in state_trades])

        # Calculate metrics
        result = {}
        for zone, data in zones.items():
            total = data['wins'] + data['losses']
            win_rate = data['wins'] / total if total > 0 else 0.0
            avg_pnl = sum(data['pnls']) / len(data['pnls']) if data['pnls'] else 0.0

            result[zone] = {
                'wins': data['wins'],
                'losses': data['losses'],
                'total': total,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': sum(data['pnls'])
            }

        return result

    def generate_pattern_report(self, brain, day_results: List[Dict]) -> str:
        """
        Generate comprehensive pattern analysis report

        Args:
            brain: QuantumBayesianBrain instance
            day_results: List of daily results

        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*80)
        report.append("PATTERN ANALYSIS REPORT")
        report.append("="*80)

        # Section 1: Top 20 Strongest Patterns
        report.append("\n### TOP 20 STRONGEST PATTERNS")
        report.append("(Min 10 samples, sorted by win rate)\n")

        strongest = self.get_strongest_patterns(brain, top_n=20, min_samples=10)

        if strongest:
            report.append(f"{'#':<3} {'Win Rate':<10} {'Trades':<8} {'Avg P&L':<10} {'Zone':<15} {'Pattern'}")
            report.append("-" * 80)

            for idx, pattern in enumerate(strongest, 1):
                report.append(
                    f"{idx:<3} "
                    f"{pattern['win_rate']:>7.1%}   "
                    f"{pattern['total']:>6}   "
                    f"${pattern['avg_pnl']:>7.2f}   "
                    f"{pattern['lagrange_zone']:<15} "
                    f"{pattern['state_str'][:40]}"
                )
        else:
            report.append("No patterns with sufficient samples yet.")

        # Section 2: Performance by Lagrange Zone
        report.append("\n\n### PERFORMANCE BY LAGRANGE ZONE")
        zone_analysis = self.analyze_by_lagrange_zone(brain)

        if zone_analysis:
            report.append(f"\n{'Zone':<15} {'Win Rate':<10} {'Trades':<8} {'Avg P&L':<10} {'Total P&L'}")
            report.append("-" * 70)

            for zone, metrics in sorted(zone_analysis.items(), key=lambda x: x[1]['win_rate'], reverse=True):
                report.append(
                    f"{zone:<15} "
                    f"{metrics['win_rate']:>7.1%}   "
                    f"{metrics['total']:>6}   "
                    f"${metrics['avg_pnl']:>7.2f}   "
                    f"${metrics['total_pnl']:>9.2f}"
                )

        # Section 3: Context Breakdown
        report.append("\n\n### PERFORMANCE BY CONTEXT")
        context_analysis = self.analyze_by_context(brain)

        if context_analysis:
            report.append(f"\n{'Context':<25} {'Win Rate':<10} {'Trades':<8}")
            report.append("-" * 50)

            for context, metrics in sorted(context_analysis.items(), key=lambda x: x[1]['win_rate'], reverse=True):
                report.append(
                    f"{context:<25} "
                    f"{metrics['win_rate']:>7.1%}   "
                    f"{metrics['total']:>6}"
                )

        # Section 4: Learning Progression
        if day_results and len(day_results) > 1:
            report.append("\n\n### LEARNING PROGRESSION")
            report.append(f"\n{'Day':<5} {'Win Rate':<10} {'Sharpe':<8} {'Trades':<8} {'States':<8} {'High-Conf'}")
            report.append("-" * 60)

            for day in day_results[-10:]:  # Last 10 days
                report.append(
                    f"{day.get('day_number', '?'):<5} "
                    f"{day.get('best_win_rate', 0):>7.1%}   "
                    f"{day.get('best_sharpe', 0):>6.2f}   "
                    f"{day.get('total_trades', 0):>6}   "
                    f"{day.get('states_learned', 0):>6}   "
                    f"{day.get('high_confidence_states', 0):>6}"
                )

        report.append("\n" + "="*80 + "\n")

        return "\n".join(report)

    def _format_state(self, state) -> str:
        """Format state for display"""
        if hasattr(state, 'lagrange_zone'):
            # ThreeBodyQuantumState
            parts = [
                f"Zone={state.lagrange_zone}",
                f"Struct={'Y' if state.structure_confirmed else 'N'}",
                f"Casc={'Y' if state.cascade_detected else 'N'}"
            ]
            return ", ".join(parts)
        elif hasattr(state, 'L1_bias'):
            # StateVector
            parts = [
                f"L1={state.L1_bias}",
                f"L2={state.L2_regime}",
                f"L4={state.L4_zone}"
            ]
            return ", ".join(parts)
        else:
            return str(state)

    def get_summary_stats(self, brain) -> Dict:
        """Get quick summary statistics"""
        total_states = len(brain.table)
        total_trades = sum(r['total'] for r in brain.table.values())

        # High confidence states
        high_conf = len([s for s in brain.table if brain.get_confidence(s) >= 0.80])

        # High win rate states (>70% with >10 samples)
        high_wr = len([
            s for s, r in brain.table.items()
            if r['total'] >= 10 and (r['wins'] / r['total']) >= 0.70
        ])

        return {
            'total_states': total_states,
            'total_trades': total_trades,
            'high_confidence_states': high_conf,
            'high_win_rate_states': high_wr,
            'avg_trades_per_state': total_trades / total_states if total_states > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    from core.bayesian_brain import QuantumBayesianBrain

    # Demo
    brain = QuantumBayesianBrain()
    analyzer = PatternAnalyzer()

    # Would normally have actual data
    strongest = analyzer.get_strongest_patterns(brain, top_n=5)
    print(f"Found {len(strongest)} strong patterns")

    stats = analyzer.get_summary_stats(brain)
    print(f"Summary: {stats}")
