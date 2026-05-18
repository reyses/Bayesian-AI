from training_iso_v2.strategies.base import EntrySignal, Strategy
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy, NMPSeed, evaluate_nmp_seed

# Diagnostic baselines — no velocity/wick filter, used to test if filters help
from training_iso_v2.strategies.nmp_baseline import NMPFadeRaw, NMPRideRaw

# Band-fade strategies — overextension trigger at higher-TF band, target lower-TF mean
from training_iso_v2.strategies.fade_at_band import FadeAtBand

# V2-native ports of the legacy 9 ExNMP tiers
from training_iso_v2.strategies.fade_calm import FadeCalm
from training_iso_v2.strategies.fade_momentum import FadeMomentum
from training_iso_v2.strategies.ride_calm import RideCalm
from training_iso_v2.strategies.ride_momentum import RideMomentum
from training_iso_v2.strategies.fade_against import FadeAgainst
from training_iso_v2.strategies.ride_against import RideAgainst
from training_iso_v2.strategies.kill_shot import KillShot
from training_iso_v2.strategies.cascade import Cascade
from training_iso_v2.strategies.freight_train import FreightTrain

# Strategies inherited from training_v2 (still useful for iso comparison)
from training_iso_v2.strategies.ma_align import MAAlignTrendFollow
from training_iso_v2.strategies.reversion import ReversionFromExtreme
from training_iso_v2.strategies.velocity_body import VelocityBodyChord
from training_iso_v2.strategies.regime_aware import RegimeAwareReversion
from training_iso_v2.strategies.filtered_nmp import FilteredRegimeAwareReversion

# 2026-05-10 PRIORITY SHIFT — trend-following / harvest tiers
from training_iso_v2.strategies.compression_bounce import CompressionBounceLong
from training_iso_v2.strategies.cat_harvest import CatHarvestRide

# 2026-05-10 CRM cusp — fade at confirmed |z| local-max (validation: cusp_research/)
from training_iso_v2.strategies.crm_cusp import CrmCuspFade

# 2026-05-16 direction classifier — LR on V2 entry features, AUC 0.864 IS
from training_iso_v2.strategies.direction_classifier import DirectionClassifierStrategy

# 2026-05-16 entry-timing + direction combined (gold-moment finder)
from training_iso_v2.strategies.golden_combined import GoldenCombinedStrategy

# 2026-05-17 trend-3 (3-class direction classifier: LONG/SHORT/NEUTRAL)
from training_iso_v2.strategies.trend3 import Trend3Strategy

# 2026-05-17 DMI-smoothed trend3 (regime confirmation from EMA + state machine)
from training_iso_v2.strategies.trend3_smoothed import Trend3SmoothedStrategy

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
    # Helpers
    'NMPBaseStrategy', 'NMPSeed', 'evaluate_nmp_seed',
]
