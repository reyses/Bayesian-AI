"""Per-trade diagnostic CSV logger.

Captures every bar's state while a position is open, writes one CSV per trade
to reports/live/trades/ for post-session inspection.
"""

import csv
import time
from pathlib import Path


class TradeLogger:
    """Lightweight per-trade state capture -> CSV on disk."""

    def __init__(self, output_dir: str = 'reports/live/trades'):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        self._trade_num = 0
        self._side = ''
        self._entry_price = 0.0
        self._active = False

    def start_trade(self, trade_num: int, side: str, entry_price: float,
                    timestamp: float):
        self._rows = []
        self._trade_num = trade_num
        self._side = side.upper()
        self._entry_price = entry_price
        self._start_ts = timestamp
        self._active = True

    def log_bar(self, data: dict):
        if not self._active:
            return
        self._rows.append(data)

    def finish_trade(self, exit_reason: str, exit_price: float):
        if not self._active or not self._rows:
            self._active = False
            return
        # Tag last row with exit info
        self._rows[-1]['exit_reason'] = exit_reason
        self._rows[-1]['exit_price'] = exit_price

        # Write CSV
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(self._start_ts))
        fname = f'trade_{self._trade_num:03d}_{self._side}_{ts_str}.csv'
        path = self._dir / fname
        fieldnames = list(self._rows[0].keys())
        # Ensure exit columns appear even if only on last row
        for col in ('exit_reason', 'exit_price'):
            if col not in fieldnames:
                fieldnames.append(col)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(self._rows)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active
