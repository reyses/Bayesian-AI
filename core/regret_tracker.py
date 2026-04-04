"""
Regret Tracker — forensic analysis of what-if alternatives.

At every trade, records what ALL other options would have done.
Purely diagnostic — never touches trade logic.

Answers:
  - Direction regret: how often was opposite direction better?
  - Duration regret: how often was a different hold time better?
  - Skip regret: how often would no-trade have been better?
  - Magnitude: how much was left on the table?

Usage:
    regret = RegretTracker()

    # At trade entry:
    regret.record_entry(bar_idx, direction, duration, price, closes_array)

    # At end of day:
    print(regret.report())
    brain_adjustments = regret.get_brain_feedback()
"""
import numpy as np
from typing import List, Dict

TICK = 0.25
TV = 0.50
DURATIONS = [1, 3, 5, 10, 15, 20, 30]


class RegretTracker:

    def __init__(self):
        self.entries = []  # list of regret records

    def record_entry(self, bar_idx: int, chosen_dir: str, chosen_dur: int,
                     entry_price: float, closes: np.ndarray,
                     nn_p_profit: float = 0.0, nn_expected_pnl: float = 0.0):
        """Record a trade entry and compute what-if for all alternatives.

        Args:
            bar_idx: index into today's close array
            chosen_dir: 'long' or 'short' — what the NN chose
            chosen_dur: bars to hold — what the NN chose
            entry_price: entry price
            closes: FULL day's close array (for forward lookup — forensics only)
            nn_p_profit: NN's predicted P(profit)
            nn_expected_pnl: NN's predicted PnL
        """
        n = len(closes)
        record = {
            'bar_idx': bar_idx,
            'chosen_dir': chosen_dir,
            'chosen_dur': chosen_dur,
            'entry_price': entry_price,
            'nn_p_profit': nn_p_profit,
            'nn_expected_pnl': nn_expected_pnl,
            'alternatives': {},
        }

        # Compute PnL for every direction x duration combination
        for dur in DURATIONS:
            end_idx = bar_idx + dur
            if end_idx >= n:
                continue

            exit_price = closes[end_idx]
            long_pnl = (exit_price - entry_price) / TICK * TV
            short_pnl = (entry_price - exit_price) / TICK * TV

            record['alternatives'][('long', dur)] = long_pnl
            record['alternatives'][('short', dur)] = short_pnl
            record['alternatives'][('skip', dur)] = 0.0

        # What the NN actually got (at chosen duration)
        chosen_key = (chosen_dir, chosen_dur)
        record['chosen_pnl'] = record['alternatives'].get(chosen_key, 0.0)

        # Best alternative
        if record['alternatives']:
            best_key = max(record['alternatives'], key=record['alternatives'].get)
            record['best_key'] = best_key
            record['best_pnl'] = record['alternatives'][best_key]
            record['regret'] = record['best_pnl'] - record['chosen_pnl']

            # Direction regret: was opposite direction better at same duration?
            opp_dir = 'short' if chosen_dir == 'long' else 'long'
            opp_key = (opp_dir, chosen_dur)
            record['opp_pnl'] = record['alternatives'].get(opp_key, 0.0)
            record['dir_regret'] = record['opp_pnl'] - record['chosen_pnl']

            # Duration regret: was a different duration better at same direction?
            same_dir_pnls = {d: record['alternatives'].get((chosen_dir, d), 0.0)
                            for d in DURATIONS if (chosen_dir, d) in record['alternatives']}
            if same_dir_pnls:
                best_dur = max(same_dir_pnls, key=same_dir_pnls.get)
                record['best_dur_same_dir'] = best_dur
                record['dur_regret'] = same_dir_pnls[best_dur] - record['chosen_pnl']
            else:
                record['dur_regret'] = 0.0

            # Skip regret: would doing nothing have been better?
            record['skip_regret'] = 0.0 - record['chosen_pnl'] if record['chosen_pnl'] < 0 else 0.0
        else:
            record['regret'] = 0.0
            record['dir_regret'] = 0.0
            record['dur_regret'] = 0.0
            record['skip_regret'] = 0.0
            record['best_pnl'] = 0.0

        self.entries.append(record)

    def report(self) -> str:
        """End-of-day regret summary."""
        if not self.entries:
            return 'Regret Tracker: no trades recorded'

        n = len(self.entries)
        chosen_pnls = [e['chosen_pnl'] for e in self.entries]
        best_pnls = [e['best_pnl'] for e in self.entries]
        regrets = [e['regret'] for e in self.entries]
        dir_regrets = [e['dir_regret'] for e in self.entries]
        dur_regrets = [e['dur_regret'] for e in self.entries]
        skip_regrets = [e['skip_regret'] for e in self.entries]

        # How often was NN wrong?
        dir_wrong = sum(1 for e in self.entries if e['dir_regret'] > 0)
        dur_wrong = sum(1 for e in self.entries if e['dur_regret'] > 0)
        should_skip = sum(1 for e in self.entries if e['chosen_pnl'] < 0)
        chose_best = sum(1 for e in self.entries if e['regret'] < 0.5)

        lines = [
            f'Regret Analysis ({n} trades):',
            f'  Chosen total PnL:  ${sum(chosen_pnls):>8.1f}  (avg ${np.mean(chosen_pnls):>6.1f})',
            f'  Best-case PnL:     ${sum(best_pnls):>8.1f}  (avg ${np.mean(best_pnls):>6.1f})',
            f'  Total regret:      ${sum(regrets):>8.1f}  (avg ${np.mean(regrets):>6.1f})',
            f'',
            f'  Direction regret:  {dir_wrong}/{n} trades ({dir_wrong/n*100:.0f}%) opposite was better',
            f'    Avg dir regret:  ${np.mean(dir_regrets):>6.1f} (positive = opposite was better)',
            f'',
            f'  Duration regret:   {dur_wrong}/{n} trades ({dur_wrong/n*100:.0f}%) different hold was better',
            f'    Avg dur regret:  ${np.mean(dur_regrets):>6.1f}',
            f'',
            f'  Skip regret:       {should_skip}/{n} trades ({should_skip/n*100:.0f}%) should have skipped',
            f'    Cost of overtrading: ${sum(skip_regrets):>6.1f}',
            f'',
            f'  Chose best option: {chose_best}/{n} ({chose_best/n*100:.0f}%)',
            f'',
        ]

        # NN confidence vs accuracy
        confident = [e for e in self.entries if e['nn_p_profit'] > 0.75]
        uncertain = [e for e in self.entries if e['nn_p_profit'] <= 0.60]
        if confident:
            conf_wr = sum(1 for e in confident if e['chosen_pnl'] > 0) / len(confident) * 100
            conf_regret = np.mean([e['regret'] for e in confident])
            lines.append(f'  High confidence (p>{0.75}): {len(confident)} trades, WR={conf_wr:.0f}%, avg regret=${conf_regret:.1f}')
        if uncertain:
            unc_wr = sum(1 for e in uncertain if e['chosen_pnl'] > 0) / len(uncertain) * 100
            unc_regret = np.mean([e['regret'] for e in uncertain])
            lines.append(f'  Low confidence (p<={0.60}): {len(uncertain)} trades, WR={unc_wr:.0f}%, avg regret=${unc_regret:.1f}')

        # Best missed opportunities (top 5 by regret)
        sorted_by_regret = sorted(self.entries, key=lambda e: -e['regret'])
        if sorted_by_regret[:5]:
            lines.append(f'')
            lines.append(f'  Top 5 regrets:')
            for e in sorted_by_regret[:5]:
                best_d, best_dur = e.get('best_key', ('?', 0))
                lines.append(f'    bar {e["bar_idx"]}: chose {e["chosen_dir"]}_{e["chosen_dur"]} '
                           f'=${e["chosen_pnl"]:.1f}, best was {best_d}_{best_dur} '
                           f'=${e["best_pnl"]:.1f} (regret=${e["regret"]:.1f})')

        return '\n'.join(lines)

    def get_brain_feedback(self) -> Dict:
        """Summarize regret into actionable feedback for the brain.

        Returns dict with:
          direction_accuracy: % of times NN picked the right direction
          duration_accuracy: % of times NN picked the right duration
          overtrade_rate: % of trades that should have been skipped
          avg_regret_by_strategy: {strategy_id: avg_regret}
        """
        if not self.entries:
            return {}

        n = len(self.entries)
        return {
            'direction_accuracy': sum(1 for e in self.entries if e['dir_regret'] <= 0) / n,
            'duration_accuracy': sum(1 for e in self.entries if e['dur_regret'] <= 0) / n,
            'overtrade_rate': sum(1 for e in self.entries if e['chosen_pnl'] < 0) / n,
            'avg_regret': np.mean([e['regret'] for e in self.entries]),
            'total_left_on_table': sum(e['regret'] for e in self.entries),
        }
