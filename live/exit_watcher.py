"""Post-exit counterfactual tracking  -- measures exit quality in hindsight."""

import logging
import time

logger = logging.getLogger(__name__)


class ExitWatcher:
    """Tracks price after exit to measure exit quality (regret analysis).

    After each exit, watches price for 60 bars (~15 min). Delivers a
    verdict: EXIT OK, EXIT REGRET (left money), EXIT EARLY (recovered),
    or EXIT CORRECT (loss trade, didn't recover).
    """

    def __init__(self, tick_size: float, point_value: float):
        self._tick = tick_size
        self._pv = point_value
        self._watchers: list[dict] = []

    def add(self, tid, side: str, entry_px: float, exit_px: float,
            exit_pnl: float, reason: str):
        """Start watching price movement after this exit."""
        self._watchers.append({
            'tid': tid, 'side': side,
            'entry_px': entry_px, 'exit_px': exit_px,
            'exit_pnl': exit_pnl, 'exit_time': time.time(),
            'peak_favorable': exit_px, 'peak_adverse': exit_px,
            'bars_watched': 0, 'reason': reason,
        })

    def tick(self, price: float):
        """Called every 15s bar. Delivers verdicts after 60 bars."""
        if not self._watchers:
            return
        done = []
        for w in self._watchers:
            w['bars_watched'] += 1
            # Track peak favorable/adverse since exit
            if w['side'] in ('LONG', 'long'):
                w['peak_favorable'] = max(w['peak_favorable'], price)
                w['peak_adverse'] = min(w['peak_adverse'], price)
            else:
                w['peak_favorable'] = min(w['peak_favorable'], price)
                w['peak_adverse'] = max(w['peak_adverse'], price)

            # After 60 bars (~15 min), deliver verdict
            if w['bars_watched'] >= 60:
                exit_px = w['exit_px']
                if w['side'] in ('LONG', 'long'):
                    _could_have = (w['peak_favorable'] - w['entry_px']) * self._pv
                    _peak_extra = (w['peak_favorable'] - exit_px) * self._pv
                else:
                    _could_have = (w['entry_px'] - w['peak_favorable']) * self._pv
                    _peak_extra = (exit_px - w['peak_favorable']) * self._pv

                _left = _could_have - w['exit_pnl']
                if _left > 10:  # left more than $10 on the table
                    logger.info(
                        f"EXIT REGRET: tid={w['tid']} {w['side']} exited ${w['exit_pnl']:+.0f} "
                        f"({w['reason']}) but peak was ${_could_have:+.0f} "
                        f"-> left ${_left:.0f} on table")
                elif w['exit_pnl'] > 0:
                    logger.info(
                        f"EXIT OK: tid={w['tid']} {w['side']} exited ${w['exit_pnl']:+.0f} "
                        f"({w['reason']}) peak was ${_could_have:+.0f} -> good exit")
                else:
                    # Loss trade  -- did it get worse or recover?
                    if w['side'] in ('LONG', 'long'):
                        _recovery = (price - exit_px) * self._pv
                    else:
                        _recovery = (exit_px - price) * self._pv
                    if _recovery > 20:
                        logger.info(
                            f"EXIT EARLY: tid={w['tid']} {w['side']} exited ${w['exit_pnl']:+.0f} "
                            f"({w['reason']}) but price recovered ${_recovery:+.0f} "
                            f"-> should have held")
                    else:
                        logger.info(
                            f"EXIT CORRECT: tid={w['tid']} {w['side']} exited ${w['exit_pnl']:+.0f} "
                            f"({w['reason']})  -- price didn't recover")
                done.append(w)

        for w in done:
            self._watchers.remove(w)
