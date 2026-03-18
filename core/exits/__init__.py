"""Exit modules  -- each implements a single exit check."""
from core.exits.stop_loss import StopLossCheck
from core.exits.take_profit import TakeProfitCheck
from core.exits.breakeven import BreakevenLock
from core.exits.envelope import EnvelopeDecay
from core.exits.giveback import PeakGiveback
from core.exits.band_exit import BandUrgentExit
from core.exits.watchdog import WatchdogCheck
from core.exits.belief_flip import BeliefFlipExit

__all__ = [
    'StopLossCheck', 'TakeProfitCheck', 'BreakevenLock',
    'EnvelopeDecay', 'PeakGiveback', 'BandUrgentExit',
    'WatchdogCheck', 'BeliefFlipExit',
]
