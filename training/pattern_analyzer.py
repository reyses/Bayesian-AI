"""
Pattern Analyzer - Identify strongest learned patterns
Analyzes BayesianBrain state table to find high-performing configurations
"""
import numpy as np
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
                dn = getattr(day, 'day_number', day.get('day_number', '?')) if isinstance(day, dict) else day.day_number
                wr = getattr(day, 'best_win_rate', 0) if not isinstance(day, dict) else day.get('best_win_rate', 0)
                sh = getattr(day, 'best_sharpe', 0) if not isinstance(day, dict) else day.get('best_sharpe', 0)
                tr = getattr(day, 'total_trades', 0) if not isinstance(day, dict) else day.get('total_trades', 0)
                sl = getattr(day, 'states_learned', 0) if not isinstance(day, dict) else day.get('states_learned', 0)
                hc = getattr(day, 'high_confidence_states', 0) if not isinstance(day, dict) else day.get('high_confidence_states', 0)
                report.append(
                    f"{dn:<5} "
                    f"{wr:>7.1%}   "
                    f"{sh:>6.2f}   "
                    f"{tr:>6}   "
                    f"{sl:>6}   "
                    f"{hc:>6}"
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

    def generate_comprehensive_report(self, brain, day_results, cumulative_trades) -> str:
        """
        Generate a comprehensive pattern report that groups by coarse categories
        so patterns are visible even when fine-grained state space is fragmented.

        Args:
            brain: QuantumBayesianBrain instance
            day_results: List of DayResults objects
            cumulative_trades: List of TradeOutcome objects (best iteration only)

        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("COMPREHENSIVE PATTERN ANALYSIS")
        report.append("=" * 80)

        # --- Section 1: State Space Diagnostics ---
        report.append("\n--- 1. STATE SPACE DIAGNOSTICS ---\n")
        total_states = len(brain.table)
        total_brain_trades = sum(r['total'] for r in brain.table.values())

        # Distribution of trades per state bucket
        trades_per_state = [r['total'] for r in brain.table.values()]
        if trades_per_state:
            hist = defaultdict(int)
            for t in trades_per_state:
                if t == 1:
                    hist['1 trade'] += 1
                elif t <= 3:
                    hist['2-3 trades'] += 1
                elif t <= 10:
                    hist['4-10 trades'] += 1
                elif t <= 30:
                    hist['11-30 trades'] += 1
                else:
                    hist['30+ trades'] += 1

            report.append(f"  Total unique states: {total_states:,}")
            report.append(f"  Total brain trades:  {total_brain_trades:,}")
            report.append(f"  Avg trades/state:    {total_brain_trades / total_states:.2f}" if total_states > 0 else "  Avg trades/state:    0")
            report.append(f"\n  State bucket distribution:")
            for bucket in ['1 trade', '2-3 trades', '4-10 trades', '11-30 trades', '30+ trades']:
                count = hist.get(bucket, 0)
                pct = count / total_states * 100 if total_states > 0 else 0
                bar = '#' * int(pct / 2)
                report.append(f"    {bucket:>13}: {count:>5} ({pct:>5.1f}%) {bar}")
        else:
            report.append("  No states in brain.")

        # --- Section 2: Performance by Zone + Z-Score Range ---
        report.append("\n--- 2. PERFORMANCE BY ZONE + Z-SCORE RANGE ---\n")

        z_groups = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnls': [], 'durations': []})
        for trade in cumulative_trades:
            state = trade.state
            zone = getattr(state, 'lagrange_zone', 'UNKNOWN')
            z = getattr(state, 'z_score', 0.0)
            if np.isnan(z):
                z = 0.0

            # Coarse z-score range
            abs_z = abs(z)
            if abs_z < 2.5:
                z_range = '2.0-2.5'
            elif abs_z < 3.0:
                z_range = '2.5-3.0'
            elif abs_z < 4.0:
                z_range = '3.0-4.0'
            elif abs_z < 6.0:
                z_range = '4.0-6.0'
            else:
                z_range = '6.0+'

            key = f"{zone} | z={z_range}"
            z_groups[key]['wins'] += 1 if trade.result == 'WIN' else 0
            z_groups[key]['losses'] += 1 if trade.result == 'LOSS' else 0
            z_groups[key]['pnls'].append(trade.pnl)
            z_groups[key]['durations'].append(trade.duration)

        if z_groups:
            report.append(f"  {'Zone + Z-Range':<25} {'Trades':>7} {'WR':>7} {'Avg P&L':>9} {'Total P&L':>11} {'Avg Dur':>8}")
            report.append("  " + "-" * 70)
            for key in sorted(z_groups.keys()):
                g = z_groups[key]
                total = g['wins'] + g['losses']
                wr = g['wins'] / total if total > 0 else 0
                avg_pnl = np.mean(g['pnls']) if g['pnls'] else 0
                total_pnl = sum(g['pnls'])
                avg_dur = np.mean(g['durations']) if g['durations'] else 0
                report.append(
                    f"  {key:<25} {total:>7} {wr:>6.1%} ${avg_pnl:>8.2f} ${total_pnl:>10.2f} {avg_dur:>7.0f}s"
                )

        # --- Section 3: Performance by Exit Reason ---
        report.append("\n--- 3. PERFORMANCE BY EXIT REASON ---\n")

        exit_groups = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnls': []})
        for trade in cumulative_trades:
            reason = getattr(trade, 'exit_reason', 'UNKNOWN')
            exit_groups[reason]['wins'] += 1 if trade.result == 'WIN' else 0
            exit_groups[reason]['losses'] += 1 if trade.result == 'LOSS' else 0
            exit_groups[reason]['pnls'].append(trade.pnl)

        if exit_groups:
            report.append(f"  {'Exit Reason':<12} {'Trades':>7} {'WR':>7} {'Avg P&L':>9} {'Total P&L':>11}")
            report.append("  " + "-" * 50)
            for reason in sorted(exit_groups.keys()):
                g = exit_groups[reason]
                total = g['wins'] + g['losses']
                wr = g['wins'] / total if total > 0 else 0
                avg_pnl = np.mean(g['pnls']) if g['pnls'] else 0
                total_pnl = sum(g['pnls'])
                report.append(
                    f"  {reason:<12} {total:>7} {wr:>6.1%} ${avg_pnl:>8.2f} ${total_pnl:>10.2f}"
                )

        # --- Section 4: Performance by Spin Inversion ---
        report.append("\n--- 4. PERFORMANCE BY SPIN INVERSION (Candle Reversal) ---\n")

        spin_groups = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnls': []})
        for trade in cumulative_trades:
            state = trade.state
            zone = getattr(state, 'lagrange_zone', 'UNKNOWN')
            spin = getattr(state, 'spin_inverted', False)
            key = f"{zone} | spin={'YES' if spin else 'NO'}"
            spin_groups[key]['wins'] += 1 if trade.result == 'WIN' else 0
            spin_groups[key]['losses'] += 1 if trade.result == 'LOSS' else 0
            spin_groups[key]['pnls'].append(trade.pnl)

        if spin_groups:
            report.append(f"  {'Zone + Spin':<30} {'Trades':>7} {'WR':>7} {'Avg P&L':>9} {'Total P&L':>11}")
            report.append("  " + "-" * 68)
            for key in sorted(spin_groups.keys()):
                g = spin_groups[key]
                total = g['wins'] + g['losses']
                wr = g['wins'] / total if total > 0 else 0
                avg_pnl = np.mean(g['pnls']) if g['pnls'] else 0
                total_pnl = sum(g['pnls'])
                report.append(
                    f"  {key:<30} {total:>7} {wr:>6.1%} ${avg_pnl:>8.2f} ${total_pnl:>10.2f}"
                )

        # --- Section 5: Momentum Strength Ranges ---
        report.append("\n--- 5. PERFORMANCE BY MOMENTUM STRENGTH ---\n")

        mom_groups = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnls': []})
        for trade in cumulative_trades:
            state = trade.state
            mom = getattr(state, 'momentum_strength', 0.0)
            if np.isnan(mom):
                mom = 0.0
            if mom < 0.5:
                mom_range = 'LOW (<0.5)'
            elif mom < 2.0:
                mom_range = 'MED (0.5-2.0)'
            elif mom < 10.0:
                mom_range = 'HIGH (2.0-10)'
            else:
                mom_range = 'EXTREME (10+)'
            mom_groups[mom_range]['wins'] += 1 if trade.result == 'WIN' else 0
            mom_groups[mom_range]['losses'] += 1 if trade.result == 'LOSS' else 0
            mom_groups[mom_range]['pnls'].append(trade.pnl)

        if mom_groups:
            report.append(f"  {'Momentum Range':<20} {'Trades':>7} {'WR':>7} {'Avg P&L':>9} {'Total P&L':>11}")
            report.append("  " + "-" * 58)
            for key in ['LOW (<0.5)', 'MED (0.5-2.0)', 'HIGH (2.0-10)', 'EXTREME (10+)']:
                if key in mom_groups:
                    g = mom_groups[key]
                    total = g['wins'] + g['losses']
                    wr = g['wins'] / total if total > 0 else 0
                    avg_pnl = np.mean(g['pnls']) if g['pnls'] else 0
                    total_pnl = sum(g['pnls'])
                    report.append(
                        f"  {key:<20} {total:>7} {wr:>6.1%} ${avg_pnl:>8.2f} ${total_pnl:>10.2f}"
                    )

        # --- Section 6: Learning Progression ---
        if day_results and len(day_results) > 0:
            report.append("\n--- 6. LEARNING PROGRESSION (All Days) ---\n")
            report.append(f"  {'Day':<5} {'Date':<12} {'Trades':>7} {'WR':>7} {'P&L':>10} {'Sharpe':>7} {'States':>7} {'HiConf':>7}")
            report.append("  " + "-" * 70)

            for day in day_results:
                dn = day.day_number
                dt = day.date
                tr = day.total_trades
                wr = day.best_win_rate
                pnl = day.best_pnl
                sh = day.best_sharpe
                sl = day.states_learned
                hc = day.high_confidence_states
                report.append(
                    f"  {dn:<5} {dt:<12} {tr:>7} {wr:>6.1%} ${pnl:>9.2f} {sh:>7.2f} {sl:>7} {hc:>7}"
                )

        # --- Section 7: Top Patterns with Low Min Samples ---
        report.append("\n--- 7. TOP PATTERNS (min 2 samples) ---\n")

        strongest = self.get_strongest_patterns(brain, top_n=20, min_samples=2)
        if strongest:
            report.append(f"  {'#':<3} {'WR':>6} {'W/L':>7} {'Avg P&L':>9} {'Zone':<12} {'Z-Bin':>6} {'Mom-Bin':>8} {'Spin':<5}")
            report.append("  " + "-" * 70)
            for idx, p in enumerate(strongest, 1):
                state = p['state']
                z_bin = ''
                mom_bin = ''
                spin = ''
                if hasattr(state, '_get_hash_bins'):
                    z_b, m_b = state._get_hash_bins()
                    z_bin = f"{z_b:.1f}"
                    mom_bin = f"{m_b:.2f}" if isinstance(m_b, (int, float)) else f"{m_b}"
                if hasattr(state, 'spin_inverted'):
                    spin = 'Y' if state.spin_inverted else 'N'
                report.append(
                    f"  {idx:<3} {p['win_rate']:>5.0%} {p['wins']:>3}/{p['total']:<3} "
                    f"${p['avg_pnl']:>8.2f} {p['lagrange_zone']:<12} {z_bin:>6} {mom_bin:>8} {spin:<5}"
                )
        else:
            report.append("  No patterns with 2+ samples.")

        # --- Section 8: TP/SL Math Check ---
        if cumulative_trades:
            report.append("\n--- 8. P&L DISTRIBUTION ---\n")

            pnls = [t.pnl for t in cumulative_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            report.append(f"  Total trades:   {len(pnls)}")
            report.append(f"  Wins:           {len(wins)} ({len(wins)/len(pnls)*100:.1f}%)")
            report.append(f"  Losses:         {len(losses)} ({len(losses)/len(pnls)*100:.1f}%)")
            report.append(f"  Total P&L:      ${sum(pnls):,.2f}")
            if wins:
                report.append(f"  Avg win:        ${np.mean(wins):.2f}")
                report.append(f"  Max win:        ${max(wins):.2f}")
            if losses:
                report.append(f"  Avg loss:       ${np.mean(losses):.2f}")
                report.append(f"  Max loss:       ${min(losses):.2f}")

            if wins and losses:
                avg_w = np.mean(wins)
                avg_l = abs(np.mean(losses))
                breakeven = avg_l / (avg_w + avg_l) * 100
                report.append(f"\n  TP:SL ratio:    {avg_w:.2f} : {avg_l:.2f} = {avg_w/avg_l:.1f}:1")
                report.append(f"  Breakeven WR:   {breakeven:.1f}%")
                report.append(f"  Actual WR:      {len(wins)/len(pnls)*100:.1f}%")
                edge = len(wins)/len(pnls)*100 - breakeven
                report.append(f"  Edge over BE:   {edge:+.1f}%")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

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
