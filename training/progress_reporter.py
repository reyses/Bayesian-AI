"""
Progress Reporter - Real-time terminal output for training
Provides comprehensive metrics tracking without timeouts
"""
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DayMetrics:
    """Metrics for a single training day"""
    day_number: int
    date: str
    total_trades: int
    win_rate: float
    sharpe: float
    pnl: float
    states_learned: int
    high_conf_states: int
    avg_duration: float
    execution_time: float


class ProgressReporter:
    """Terminal-friendly progress reporting"""

    def __init__(self):
        self.day_metrics: List[DayMetrics] = []
        self.start_time = time.time()

    def print_day_header(self, day_number: int, date: str, total_days: int, bars: int):
        """Print header for new day"""
        print("\n" + "="*80)
        print(f"DAY {day_number}/{total_days}: {date} ({bars:,} bars)")
        print("="*80)

    def print_day_summary(self, metrics: DayMetrics):
        """Print comprehensive day summary"""
        print(f"\n{'='*80}")
        print(f"✅ DAY {metrics.day_number} COMPLETE: {metrics.date}")
        print(f"{'='*80}")
        print(f"\nTRADING METRICS:")
        print(f"  Total Trades: {metrics.total_trades:>6}        Win Rate: {metrics.win_rate:>6.1%}        P&L: ${metrics.pnl:>8,.2f}")
        print(f"  Sharpe Ratio: {metrics.sharpe:>6.2f}        Avg Duration: {metrics.avg_duration:>6.1f}s")

        print(f"\nLEARNING METRICS:")
        print(f"  States Learned: {metrics.states_learned:>5}      High-Conf States: {metrics.high_conf_states:>5}")

        print(f"\nEXECUTION:")
        print(f"  Day Execution Time: {metrics.execution_time:>6.1f}s")

        self.day_metrics.append(metrics)

    def print_cumulative_summary(self, top_patterns: List[Dict] = None):
        """Print cumulative statistics across all days"""
        if not self.day_metrics:
            return

        total_days = len(self.day_metrics)
        total_trades = sum(d.total_trades for d in self.day_metrics)
        total_pnl = sum(d.pnl for d in self.day_metrics)
        avg_wr = sum(d.win_rate * d.total_trades for d in self.day_metrics) / total_trades if total_trades > 0 else 0.0
        avg_sharpe = sum(d.sharpe for d in self.day_metrics) / total_days if total_days > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"CUMULATIVE SUMMARY (Days 1-{total_days})")
        print(f"{'='*80}")
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Trades: {total_trades:>6}        Overall Win Rate: {avg_wr:>6.1%}")
        print(f"  Total P&L: ${total_pnl:>10,.2f}      Average Sharpe: {avg_sharpe:>6.2f}")

        # Latest day metrics
        latest = self.day_metrics[-1]
        print(f"\nLATEST DAY ({latest.date}):")
        print(f"  States Learned: {latest.states_learned:>5}      High-Conf States: {latest.high_conf_states:>5}")

        # Top patterns if provided
        if top_patterns:
            print(f"\nTOP 5 PATTERNS:")
            for idx, pattern in enumerate(top_patterns[:5], 1):
                wins = pattern['wins']
                total = pattern['total']
                wr = pattern['win_rate']
                avg_pnl = pattern['avg_pnl']
                zone = pattern.get('lagrange_zone', 'unknown')

                print(f"  {idx}. [{zone}] → {wins}/{total} wins ({wr:.1%}) | Avg: ${avg_pnl:.2f}/trade")

    def print_trade_detail(self, trade_num: int, trade, state_str: str):
        """Print individual trade result (for debugging)"""
        result_symbol = "✓" if trade.result == 'WIN' else "✗"
        print(f"  {result_symbol} Trade #{trade_num}: "
              f"${trade.pnl:>7.2f} | "
              f"{trade.duration:>5.1f}s | "
              f"{trade.exit_reason:<8} | "
              f"{state_str[:40]}")

    def print_iteration_progress(self, iteration: int, total: int, trades: int, wr: float, sharpe: float):
        """Print quick iteration update (called periodically)"""
        pct = (iteration / total) * 100
        print(f"  Iteration {iteration:>4}/{total} ({pct:>5.1f}%) | "
              f"Trades: {trades:>4} | WR: {wr:>5.1%} | Sharpe: {sharpe:>5.2f}",
              end='\r')

    def print_final_summary(self):
        """Print final summary when all training complete"""
        if not self.day_metrics:
            print("\nNo training data to summarize.")
            return

        total_time = time.time() - self.start_time
        total_days = len(self.day_metrics)
        total_trades = sum(d.total_trades for d in self.day_metrics)
        total_pnl = sum(d.pnl for d in self.day_metrics)

        # Calculate weighted averages
        avg_wr = sum(d.win_rate * d.total_trades for d in self.day_metrics) / total_trades if total_trades > 0 else 0.0
        avg_sharpe = sum(d.sharpe for d in self.day_metrics) / total_days if total_days > 0 else 0.0

        # Latest state
        latest = self.day_metrics[-1]

        print("\n\n" + "="*80)
        print("TRAINING COMPLETE - FINAL SUMMARY")
        print("="*80)

        print(f"\nTRAINING DURATION:")
        print(f"  Total Time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        print(f"  Days Trained: {total_days}")
        print(f"  Avg Time per Day: {total_time/total_days:.1f}s")

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Trades: {total_trades:>6}")
        print(f"  Overall Win Rate: {avg_wr:>6.1%}")
        print(f"  Average Sharpe: {avg_sharpe:>6.2f}")
        print(f"  Total P&L: ${total_pnl:>10,.2f}")

        print(f"\nLEARNING METRICS:")
        print(f"  Final States Learned: {latest.states_learned:>5}")
        print(f"  High-Confidence States: {latest.high_conf_states:>5}")
        print(f"  Approval Rate: {(latest.high_conf_states/latest.states_learned*100):>5.1f}%")

        # Calculate learning curve
        if total_days >= 5:
            first_5_wr = sum(d.win_rate for d in self.day_metrics[:5]) / 5
            last_5_wr = sum(d.win_rate for d in self.day_metrics[-5:]) / 5
            improvement = last_5_wr - first_5_wr

            print(f"\nLEARNING CURVE:")
            print(f"  First 5 Days WR: {first_5_wr:>6.1%}")
            print(f"  Last 5 Days WR: {last_5_wr:>6.1%}")
            print(f"  Improvement: {improvement:>+6.1%}")

        print("\n" + "="*80)
        print("System trained and ready for live trading analysis.")
        print("="*80 + "\n")

    def print_parameter_evolution(self, param_history: List[Dict]):
        """Print how parameters evolved over time"""
        if not param_history or len(param_history) < 2:
            return

        print(f"\n{'='*80}")
        print(f"PARAMETER EVOLUTION")
        print(f"{'='*80}")

        # Track key parameters
        key_params = ['stop_loss_ticks', 'take_profit_ticks', 'confidence_threshold', 'trail_distance_tight']

        print(f"\n{'Day':<6}", end='')
        for param in key_params:
            print(f"{param:<20}", end='')
        print()
        print("-" * 80)

        # Show first, middle, and last days
        indices = [0, len(param_history)//2, -1]
        for idx in indices:
            params = param_history[idx]
            day = idx + 1 if idx >= 0 else len(param_history)
            print(f"{day:<6}", end='')
            for param in key_params:
                val = params.get(param, 'N/A')
                print(f"{str(val):<20}", end='')
            print()

    def print_regret_summary(self, regret_analysis: Dict):
        """Print regret analysis summary"""
        if not regret_analysis:
            return

        print(f"\n{'='*80}")
        print(f"REGRET ANALYSIS")
        print(f"{'='*80}")

        avg_efficiency = regret_analysis.get('avg_exit_efficiency', 0.0)
        exits_too_early = regret_analysis.get('exits_too_early', 0)
        exits_too_late = regret_analysis.get('exits_too_late', 0)
        exits_optimal = regret_analysis.get('exits_optimal', 0)
        total = exits_too_early + exits_too_late + exits_optimal

        print(f"\nEXIT QUALITY:")
        print(f"  Average Efficiency: {avg_efficiency:>6.1%}")
        print(f"  Optimal Exits: {exits_optimal:>5}/{total} ({exits_optimal/total*100:>5.1f}%)")
        print(f"  Too Early: {exits_too_early:>5}/{total} ({exits_too_early/total*100:>5.1f}%)")
        print(f"  Too Late: {exits_too_late:>5}/{total} ({exits_too_late/total*100:>5.1f}%)")

        # Recommendations if available
        recommendations = regret_analysis.get('recommendations', [])
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")

    def save_progress_log(self, filepath: str):
        """Save progress to file for later analysis"""
        import json

        data = {
            'start_time': self.start_time,
            'total_duration': time.time() - self.start_time,
            'days': [
                {
                    'day': m.day_number,
                    'date': m.date,
                    'trades': m.total_trades,
                    'win_rate': m.win_rate,
                    'sharpe': m.sharpe,
                    'pnl': m.pnl,
                    'states': m.states_learned,
                    'high_conf': m.high_conf_states,
                    'duration': m.avg_duration,
                    'exec_time': m.execution_time
                }
                for m in self.day_metrics
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Progress saved to: {filepath}")


# Example usage
if __name__ == "__main__":
    reporter = ProgressReporter()

    # Demo
    reporter.print_day_header(1, "2025-01-01", 30, 35000)

    metrics = DayMetrics(
        day_number=1,
        date="2025-01-01",
        total_trades=24,
        win_rate=0.625,
        sharpe=1.42,
        pnl=1240.50,
        states_learned=12,
        high_conf_states=3,
        avg_duration=185.3,
        execution_time=2534.1
    )

    reporter.print_day_summary(metrics)
    reporter.print_cumulative_summary()
    reporter.print_final_summary()
