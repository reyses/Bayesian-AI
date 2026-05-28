"""ZIGZAG — streaming pivot-reversal strategy on 5s closes, ATR-sized R.

Design (per user 2026-05-17 / 2026-05-24):
  - Trigger TF        : 5s closes (per-bar streaming detection)
  - R-trigger sizing  : ATR(14) of 1m bars × ATR_MULT (default 4)
  - Min bars between  : 36 in 5s units = 3 minutes (matches the offline
                        dataset builder `tools/zigzag/build_zigzag_pivot_dataset.py`)

The streaming pivot detector mirrors `tools/zigzag/live_zigzag_baseline.py`:
emit per-bar direction state (0/+1/-1) at the time the bar is processed; no
future info. When direction flips, this strategy returns an EntrySignal in
the new leg direction.

Caller responsibility: pass `min_reversal_ticks` (= ATR_pts/TICK_SIZE * atr_mult)
precomputed once per day. The strategy does not compute ATR itself.

Offline baseline (build IS+OOS pivot datasets to canonical paths):
  python -m training.strategies.zigzag                 # full IS+OOS rebuild
  python -m training.strategies.zigzag --is-days 2025_06_16 --oos-days 2026_05_15 \\
      --out-is c:/tmp/is.parquet --out-oos c:/tmp/oos.parquet     # smoke
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy


TICK_SIZE = 0.25

# ── Canonical baseline parameters ───────────────────────────────────
ATR_MULT_DEFAULT = 4.0
MIN_BARS_5S_DEFAULT = 36

# ── Canonical dataset paths (consumed by B-model trainers) ──────────
CANONICAL_IS_PATH = 'reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet'
CANONICAL_OOS_PATH = 'reports/findings/regret_oracle/zigzag_pivot_dataset_OOS_atr4.parquet'

# Atlas roots
ATLAS_IS_ROOT = 'DATA/ATLAS'
ATLAS_OOS_ROOT = 'DATA/ATLAS_NT8'


class ZigzagStrategy(Strategy):
    """Streaming zigzag pivot-reversal entry on 5s closes.

    Direction state machine (0 / +1 / -1):
        0  = undecided (no extreme far enough from first close)
        +1 = current leg is LONG (last confirmed pivot was a LOW)
        -1 = current leg is SHORT (last confirmed pivot was a HIGH)

    Flips when 5s close crosses (extreme ± R) AND min_bars elapsed since
    the extreme. Emits an EntrySignal on the flip bar only.
    """

    name = 'ZIGZAG'

    def __init__(self,
                 min_reversal_ticks: int = 0,
                 min_bars_5s: int = MIN_BARS_5S_DEFAULT):
        if min_reversal_ticks < 0:
            raise ValueError(f'min_reversal_ticks must be >= 0, got {min_reversal_ticks}')
        self.min_reversal_ticks = int(min_reversal_ticks)
        self.min_bars_5s = int(min_bars_5s)
        self.reset()

    def reset(self) -> None:
        self._direction = 0
        self._extreme_ticks: Optional[int] = None
        self._extreme_bar = -1
        self._bar_count = 0
        self._first_close_ticks: Optional[int] = None

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if self.min_reversal_ticks <= 0:
            return None

        close_5s = state.ohlcv_5s.get('close', state.price) if state.ohlcv_5s else state.price
        price_ticks = close_5s / TICK_SIZE
        i = self._bar_count
        self._bar_count += 1

        if self._first_close_ticks is None:
            self._first_close_ticks = price_ticks
            self._extreme_ticks = price_ticks
            self._extreme_bar = i
            return None

        flipped_to: Optional[str] = None

        if self._direction == 0:
            if price_ticks > self._extreme_ticks:
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
            if (price_ticks < self._first_close_ticks
                    and (self._first_close_ticks - price_ticks) >= self.min_reversal_ticks):
                self._direction = -1
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
                flipped_to = 'short'
            elif (price_ticks > self._first_close_ticks
                    and (price_ticks - self._first_close_ticks) >= self.min_reversal_ticks):
                self._direction = 1
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
                flipped_to = 'long'

        elif self._direction == 1:
            if price_ticks >= self._extreme_ticks:
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
            elif ((self._extreme_ticks - price_ticks) >= self.min_reversal_ticks
                    and (i - self._extreme_bar) >= self.min_bars_5s):
                self._direction = -1
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
                flipped_to = 'short'

        elif self._direction == -1:
            if price_ticks <= self._extreme_ticks:
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
            elif ((price_ticks - self._extreme_ticks) >= self.min_reversal_ticks
                    and (i - self._extreme_bar) >= self.min_bars_5s):
                self._direction = 1
                self._extreme_ticks = price_ticks
                self._extreme_bar = i
                flipped_to = 'long'

        if flipped_to is None:
            return None
        return EntrySignal(direction=flipped_to, tier=self.name,
                           extras={'min_reversal_ticks': self.min_reversal_ticks,
                                   'extreme_bar': self._extreme_bar})


# ── Execution logic: baseline dataset build (IS + OOS → two parquets) ──

def build_datasets(atr_mult: float = ATR_MULT_DEFAULT,
                   is_days: Optional[list] = None,
                   oos_days: Optional[list] = None,
                   out_is: str = CANONICAL_IS_PATH,
                   out_oos: str = CANONICAL_OOS_PATH) -> None:
    """Build IS + OOS pivot datasets at canonical paths.

    Wraps the canonical CLI `tools/zigzag/build_zigzag_pivot_dataset.py` so
    the dataset build is a single execution-logic call inside this strategy
    module. IS uses Databento (DATA/ATLAS); OOS uses NT8 (DATA/ATLAS_NT8).
    """
    script = 'tools/zigzag/build_zigzag_pivot_dataset.py'

    def _invoke(target: str, root: str, out: str, days: Optional[list]) -> None:
        cmd = [sys.executable, script,
               '--target', target,
               '--atr-mult', str(atr_mult),
               '--root', root,
               '--out', out]
        if days:
            cmd += ['--days'] + list(days)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)

    _invoke('is', ATLAS_IS_ROOT, out_is, is_days)
    _invoke('oos', ATLAS_OOS_ROOT, out_oos, oos_days)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Build canonical zigzag pivot datasets (IS + OOS → two parquets).')
    ap.add_argument('--atr-mult', type=float, default=ATR_MULT_DEFAULT)
    ap.add_argument('--is-days', nargs='*', default=None,
                    help='Override IS day list (smoke testing).')
    ap.add_argument('--oos-days', nargs='*', default=None,
                    help='Override OOS day list (smoke testing).')
    ap.add_argument('--out-is', default=CANONICAL_IS_PATH)
    ap.add_argument('--out-oos', default=CANONICAL_OOS_PATH)
    args = ap.parse_args()
    build_datasets(atr_mult=args.atr_mult,
                   is_days=args.is_days, oos_days=args.oos_days,
                   out_is=args.out_is, out_oos=args.out_oos)
