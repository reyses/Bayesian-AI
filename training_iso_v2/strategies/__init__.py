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
    # Helpers
    'NMPBaseStrategy', 'NMPSeed', 'evaluate_nmp_seed',
]
