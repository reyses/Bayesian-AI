"""Ping-pong flip logic  -- direction, sizing, deferred flip management."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FlipDecision:
    """Result of a flip direction determination."""
    side: str              # 'long' or 'short'
    p_long: float          # directional probability
    dir_source: str        # which cascade level decided
    sl_ticks: int
    tp_ticks: int
    trail_ticks: int
    trail_act: int
    same_side: bool        # True if continuing same direction


class PingPongManager:
    """Manages continuous wave-riding with direction refinement.

    Owns:
    - Flip direction determination (delegates to ExecutionEngine cascade)
    - ATR-based sizing computation
    - Deferred flip state management
    - Flip count tracking

    Does NOT own: position creation, order submission, brain learning.
    LiveEngine calls determine_flip(), then executes the result.
    """

    def __init__(self, config, tuning: dict):
        self._cfg = config
        self._tuning = tuning

        # Overrides from CLI
        self._sl_override = config.pp_sl_override
        self._tp_override = config.pp_tp_override
        self._trail_override = config.pp_trail_override

        self.flip_count = 0
        self.pending_flip: Optional[dict] = None  # {'exited_side', 'price', 'ts'}
        self.last_exit_side = ''

    @property
    def enabled(self) -> bool:
        return self._cfg.ping_pong

    def compute_sizing(self, atr_ticks: float) -> Tuple[int, int, int, int]:
        """Compute SL/TP/trail/trail_act from ATR and tuning overrides.

        Returns (sl_ticks, tp_ticks, trail_ticks, trail_act).
        """
        atr = atr_ticks if atr_ticks > 0 else 8.0
        _floor = max(4, self._tuning.get('min_tick_floor', 4))
        sl = (self._sl_override
              or self._tuning.get('pp_sl', 0)
              or max(_floor, int(round(atr * self._tuning.get('exit_sl_mult', 3.0)))))
        tp = (self._tp_override
              or self._tuning.get('pp_tp', 0)
              or max(_floor, int(round(atr * self._tuning.get('exit_tp_mult', 5.0)))))
        trail = (self._trail_override
                 or self._tuning.get('pp_trail', 0)
                 or max(_floor, int(round(atr * self._tuning.get('exit_trail_mult', 2.5)))))
        trail_act = max(_floor, int(round(
            atr * self._tuning.get('exit_trail_act_mult', 0.6))))
        return sl, tp, trail, trail_act

    def determine_flip(self, exited_side: str, state, exec_engine,
                       anchor_depth: int, anchor_tf, ts: float,
                       active_tid, side_lock: str = None,
                       atr_ticks: float = 8.0) -> Optional[FlipDecision]:
        """Decide direction and sizing for a ping-pong flip.

        Returns FlipDecision, or None if no states available.
        """
        from core.execution_engine import Candidate

        tid = active_tid or 'MANUAL'
        base_tid = (tid[3:] if isinstance(tid, str)
                    and tid.startswith('PP_') else tid)

        _pp_dir = side_lock if side_lock else None
        cand = Candidate(state=state, depth=anchor_depth,
                         timeframe=anchor_tf, timestamp=ts,
                         pattern_type='PP_FLIP',
                         z_score=state.z_score)
        side, p_long, dir_src = exec_engine._direction_cascade(
            cand, base_tid, lib_entry={}, pp_dir_override=_pp_dir)

        sl, tp, trail, trail_act = self.compute_sizing(atr_ticks)

        self.flip_count += 1
        same_side = (side.lower() == exited_side.lower())

        return FlipDecision(
            side=side, p_long=p_long, dir_source=dir_src,
            sl_ticks=sl, tp_ticks=tp, trail_ticks=trail,
            trail_act=trail_act, same_side=same_side,
        )

    def schedule_flip(self, exited_side: str, price: float, ts: float):
        """Deferred flip  -- fires when NT8 confirms flat."""
        logger.info(f"PING-PONG: scheduling flip after {exited_side} exhaustion")
        self.pending_flip = {
            'exited_side': exited_side, 'price': price, 'ts': ts,
        }

    def consume_pending(self) -> Optional[dict]:
        """Pop and return pending flip (None if nothing pending)."""
        flip = self.pending_flip
        self.pending_flip = None
        return flip
