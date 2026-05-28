from training.strategies.base import EntrySignal, Strategy
from training.strategies._nmp_base import NMPBaseStrategy, NMPSeed, evaluate_nmp_seed

# Diagnostic baselines — no velocity/wick filter, used to test if filters help
from training.strategies.nmp_baseline import NMPFadeRaw, NMPRideRaw

# Band-fade strategies — overextension trigger at higher-TF band, target lower-TF mean
from training.strategies.fade_at_band import FadeAtBand

# V2-native ports of the legacy 9 ExNMP tiers
from training.strategies.fade_calm import FadeCalm
from training.strategies.fade_momentum import FadeMomentum
from training.strategies.ride_calm import RideCalm
from training.strategies.ride_momentum import RideMomentum
from training.strategies.fade_against import FadeAgainst
from training.strategies.ride_against import RideAgainst
from training.strategies.kill_shot import KillShot
from training.strategies.cascade import Cascade
from training.strategies.freight_train import FreightTrain

# Strategies inherited from training_v2 (still useful for iso comparison)
from training.strategies.ma_align import MAAlignTrendFollow
from training.strategies.reversion import ReversionFromExtreme
from training.strategies.velocity_body import VelocityBodyChord
from training.strategies.regime_aware import RegimeAwareReversion
from training.strategies.filtered_nmp import FilteredRegimeAwareReversion

# 2026-05-10 PRIORITY SHIFT — trend-following / harvest tiers
from training.strategies.compression_bounce import CompressionBounceLong
from training.strategies.cat_harvest import CatHarvestRide

# 2026-05-10 CRM cusp — fade at confirmed |z| local-max (validation: cusp_research/)
from training.strategies.crm_cusp import CrmCuspFade

# 2026-05-16 direction classifier — LR on V2 entry features, AUC 0.864 IS
from training.strategies.direction_classifier import DirectionClassifierStrategy

# 2026-05-16 entry-timing + direction combined (gold-moment finder)
from training.strategies.golden_combined import GoldenCombinedStrategy

# 2026-05-17 trend-3 (3-class direction classifier: LONG/SHORT/NEUTRAL)
from training.strategies.trend3 import Trend3Strategy

# 2026-05-17 DMI-smoothed trend3 (regime confirmation from EMA + state machine)
from training.strategies.trend3_smoothed import Trend3SmoothedStrategy

# 2026-05-24 streaming zigzag pivot-reversal (5s trigger, 1m-ATR-sized R, ATR×4)
from training.strategies.zigzag import ZigzagStrategy

__all__ = [
    'EntrySignal', 'Strategy',
    # 9 legacy tiers ported V2-native
    'FadeCalm', 'FadeMomentum', 'RideCalm', 'RideMomentum',
    'FadeAgainst', 'RideAgainst', 'KillShot', 'Cascade', 'FreightTrain',
    # Diagnostic baselines (filter ablation tests)
    'NMPFadeRaw', 'NMPRideRaw',
    # Band-fade strategies
    'FadeAtBand',
    # Other V2-native strategies
    'MAAlignTrendFollow', 'ReversionFromExtreme', 'VelocityBodyChord',
    'RegimeAwareReversion', 'FilteredRegimeAwareReversion',
    # 2026-05-10 priority shift — trend tiers
    'CompressionBounceLong', 'CatHarvestRide',
    # 2026-05-10 CRM cusp — confirmed |z| local-max fade
    'CrmCuspFade',
    # 2026-05-16 direction classifier (LR on V2 entry features)
    'DirectionClassifierStrategy',
    'GoldenCombinedStrategy',
    'Trend3Strategy',
    'Trend3SmoothedStrategy',
    # 2026-05-24 streaming zigzag (5s trigger, ATR(1m)×4 R)
    'ZigzagStrategy',
    # Helpers
    'NMPBaseStrategy', 'NMPSeed', 'evaluate_nmp_seed',
]
