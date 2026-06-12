from training.strategies.base import EntrySignal, Strategy
from training.strategies._nmp_base import NMPBaseStrategy, NMPSeed, evaluate_nmp_seed

# Diagnostic baselines — no velocity/wick filter, used to test if filters help
from training.strategies.nmp_baseline import NMPFadeRaw, NMPRideRaw

# Filtered NMP
from training.strategies.filtered_nmp import FilteredRegimeAwareReversion

# 2026-05-24 streaming zigzag (5s trigger, 1m-ATR-sized R, ATR×4)
from training.strategies.zigzag import ZigzagStrategy

__all__ = [
    'EntrySignal', 'Strategy',
    # Diagnostic baselines (filter ablation tests)
    'NMPFadeRaw', 'NMPRideRaw',
    # Filtered NMP
    'FilteredRegimeAwareReversion',
    # 2026-05-24 streaming zigzag (5s trigger, ATR(1m)×4 R)
    'ZigzagStrategy',
    # Helpers
    'NMPBaseStrategy', 'NMPSeed', 'evaluate_nmp_seed',
]
