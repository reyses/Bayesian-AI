"""Counterfactual Engine (Monkey Brain) -- phantom trades for every decision.

Every skip and every trade spawns lightweight phantom trades with alternative
parameter settings. The optimization surface emerges from real-time what-if
evaluation. No offline optimization needed.

Usage:
  - Created by AdvanceEngine when --monkey flag is active
  - on_skip(): peak was blocked -> spawn phantom that enters anyway
  - on_entry(): trade entered -> spawn phantoms with alt exit thresholds
  - on_bar(): update all active phantoms with current bar data
  - report(): summary of what parameter settings would have been optimal

Evolution: Cat (regime) -> Crow (seeds) -> Monkey (counterfactual) -> Goat (CNN)
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import numpy as np


@dataclass
class PhantomTrade:
    """Simulated trade for counterfactual evaluation."""
    spawn_bar: int
    spawn_reason: str           # 'skip_sensor', 'skip_cat', 'alt_giveback', etc.
    parameter_name: str         # which parameter is being tested
    parameter_value: float      # the alternative value

    direction: str              # LONG or SHORT
    entry_price: float
    entry_bar: int

    # Tracked state
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0
    bars_held: int = 0
    peak_bar: int = 0

    # Exit state
    exited: bool = False
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ''
    exit_pnl_ticks: float = 0.0


class CounterfactualEngine:
    """Spawns and manages phantom trades for what-if analysis.

    Parameters
    ----------
    max_phantoms : int
        Maximum active phantoms at once. Oldest expire when exceeded.
    max_hold_bars : int
        Phantom trades expire after this many bars (20 min at 15s = 80).
    """

    def __init__(self, max_phantoms: int = 500, max_hold_bars: int = 80,
                 tick_size: float = 0.25):
        self._max_phantoms = max_phantoms
        self._max_hold = max_hold_bars
        self._tick_size = tick_size

        self.active: list = []
        self.completed: deque = deque(maxlen=5000)

        # Parameter optimization surface
        # Key: (parameter_name, parameter_value)
        # Value: counts and totals
        self.surface: dict = {}

        # Counters
        self.n_spawned = 0
        self.n_completed = 0
        self.n_expired = 0

    def on_skip(self, bar_index: int, price: float, direction: str,
                skip_reason: str):
        """Peak was skipped -- spawn phantom that enters anyway."""
        self._spawn(
            bar_index=bar_index,
            price=price,
            direction=direction,
            spawn_reason=f'skip_{skip_reason}',
            parameter_name='entry_gate',
            parameter_value=0,  # no gate
        )

    def on_entry(self, bar_index: int, price: float, direction: str,
                 current_giveback: float = 0.50):
        """Trade entered -- spawn phantoms with alternative exit thresholds."""
        # Test different giveback thresholds
        for gb in [0.15, 0.25, 0.35, 0.50, 0.65]:
            if abs(gb - current_giveback) < 0.01:
                continue  # skip current setting
            self._spawn(
                bar_index=bar_index,
                price=price,
                direction=direction,
                spawn_reason=f'alt_giveback_{gb:.0%}',
                parameter_name='giveback_pct',
                parameter_value=gb,
            )

    def _spawn(self, *, bar_index, price, direction, spawn_reason,
               parameter_name, parameter_value):
        """Create a phantom trade."""
        # Cap active phantoms
        while len(self.active) >= self._max_phantoms:
            oldest = self.active.pop(0)
            oldest.exited = True
            oldest.exit_reason = 'phantom_cap'
            self.completed.append(oldest)
            self.n_expired += 1

        phantom = PhantomTrade(
            spawn_bar=bar_index,
            spawn_reason=spawn_reason,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
            direction=direction,
            entry_price=price,
            entry_bar=bar_index,
        )
        self.active.append(phantom)
        self.n_spawned += 1

    def on_bar(self, bar_index: int, price: float, bar_high: float,
               bar_low: float):
        """Update all active phantoms with current bar."""
        _to_remove = []

        for i, p in enumerate(self.active):
            p.bars_held += 1

            # MFE / MAE
            if p.direction == 'LONG':
                fav = (bar_high - p.entry_price) / self._tick_size
                adv = (p.entry_price - bar_low) / self._tick_size
                current = (price - p.entry_price) / self._tick_size
            else:
                fav = (p.entry_price - bar_low) / self._tick_size
                adv = (bar_high - p.entry_price) / self._tick_size
                current = (p.entry_price - price) / self._tick_size

            if fav > p.mfe_ticks:
                p.mfe_ticks = fav
                p.peak_bar = bar_index
            p.mae_ticks = max(p.mae_ticks, adv)

            # Check exit conditions based on parameter being tested
            _exit = False

            if p.parameter_name == 'giveback_pct':
                # Exit when price gives back X% of MFE
                if p.mfe_ticks > 4.0:  # minimum peak
                    gave_back = p.mfe_ticks - current
                    if gave_back > 0 and gave_back / p.mfe_ticks >= p.parameter_value:
                        _exit = True
                        p.exit_reason = f'giveback_{p.parameter_value:.0%}'
                        p.exit_pnl_ticks = current

            elif p.parameter_name == 'entry_gate':
                # Skip phantom: track what would have happened with default exit
                # Use simple giveback at 50% as exit
                if p.mfe_ticks > 4.0:
                    gave_back = p.mfe_ticks - current
                    if gave_back > 0 and gave_back / p.mfe_ticks >= 0.50:
                        _exit = True
                        p.exit_reason = 'default_exit'
                        p.exit_pnl_ticks = current

            # Hard stop loss at -40 ticks
            if current < -40:
                _exit = True
                p.exit_reason = 'phantom_sl'
                p.exit_pnl_ticks = current

            # Max hold expiry
            if p.bars_held >= self._max_hold:
                _exit = True
                p.exit_reason = 'phantom_expired'
                p.exit_pnl_ticks = current

            if _exit:
                p.exited = True
                p.exit_bar = bar_index
                p.exit_price = price
                _to_remove.append(i)

        # Remove completed (reverse order to preserve indices)
        for i in reversed(_to_remove):
            phantom = self.active.pop(i)
            self.completed.append(phantom)
            self._update_surface(phantom)
            self.n_completed += 1

    def _update_surface(self, phantom: PhantomTrade):
        """Update parameter optimization surface."""
        key = (phantom.parameter_name, phantom.parameter_value)
        if key not in self.surface:
            self.surface[key] = {
                'n': 0, 'total_pnl': 0.0, 'wins': 0,
                'total_mfe': 0.0, 'total_mae': 0.0,
            }
        s = self.surface[key]
        s['n'] += 1
        s['total_pnl'] += phantom.exit_pnl_ticks
        s['total_mfe'] += phantom.mfe_ticks
        s['total_mae'] += phantom.mae_ticks
        if phantom.exit_pnl_ticks > 0:
            s['wins'] += 1

    def get_optimal(self, parameter_name: str) -> Optional[float]:
        """Get optimal value for a parameter from phantom outcomes."""
        best_val = None
        best_avg = -999
        for (pname, pval), stats in self.surface.items():
            if pname != parameter_name or stats['n'] < 20:
                continue
            avg = stats['total_pnl'] / stats['n']
            if avg > best_avg:
                best_avg = avg
                best_val = pval
        return best_val

    def report(self) -> str:
        """Generate counterfactual summary."""
        lines = []
        lines.append('COUNTERFACTUAL ANALYSIS (Monkey Brain)')
        lines.append('=' * 60)
        lines.append(f'  Spawned: {self.n_spawned:,}  Completed: {self.n_completed:,}  '
                     f'Active: {len(self.active):,}  Expired: {self.n_expired:,}')
        lines.append('')

        if not self.surface:
            lines.append('  No completed phantoms yet.')
            return '\n'.join(lines)

        # Group by parameter name
        params = sorted(set(pname for pname, _ in self.surface.keys()))
        for pname in params:
            lines.append(f'  PARAMETER: {pname}')
            lines.append(f'  {"Value":>8} {"N":>6} {"Avg PnL":>9} {"WR":>6} '
                         f'{"Avg MFE":>9} {"Avg MAE":>9} {"PF":>6}')

            entries = []
            for (p, v), s in self.surface.items():
                if p != pname or s['n'] < 5:
                    continue
                avg_pnl = s['total_pnl'] / s['n']
                wr = s['wins'] / s['n'] * 100
                avg_mfe = s['total_mfe'] / s['n']
                avg_mae = s['total_mae'] / s['n']
                gw = s['total_pnl'] if s['total_pnl'] > 0 else 0
                # Approximate PF from win/loss counts
                avg_win = s['total_pnl'] / s['wins'] if s['wins'] > 0 else 0
                n_loss = s['n'] - s['wins']
                avg_loss = abs(s['total_pnl'] - avg_win * s['wins']) / n_loss if n_loss > 0 else 0
                pf = (avg_win * s['wins']) / (avg_loss * n_loss) if n_loss > 0 and avg_loss > 0 else 0
                entries.append((v, s['n'], avg_pnl, wr, avg_mfe, avg_mae, pf))

            entries.sort(key=lambda x: -x[2])  # sort by avg PnL
            best_val = entries[0][0] if entries else None
            for v, n, avg, wr, mfe, mae, pf in entries:
                flag = ' <-- BEST' if v == best_val else ''
                lines.append(f'  {v:>8.2f} {n:>6} {avg:>+8.1f}t {wr:>5.1f}% '
                             f'{mfe:>8.1f}t {mae:>8.1f}t {pf:>5.2f}{flag}')
            lines.append('')

        # Recommendations
        lines.append('  RECOMMENDATIONS:')
        for pname in params:
            opt = self.get_optimal(pname)
            if opt is not None:
                lines.append(f'    {pname}: use {opt:.2f}')

        lines.append('=' * 60)
        return '\n'.join(lines)
