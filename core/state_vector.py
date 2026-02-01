"""
ProjectX v2.0 - State Vector Module
9-Layer market state representation for Bayesian probability mapping
"""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class StateVector:
    """
    Immutable 9-layer state snapshot
    Used as HashMap key for probability lookups
    """
    # STATIC LAYERS (Session-level)
    L1_bias: str        # '90d': 'bull', 'bear', 'range'
    L2_regime: str      # '30d': 'trending', 'chopping'
    L3_swing: str       # '1wk': 'higher_highs', 'lower_lows', 'sideways'
    L4_zone: str        # 'daily': 'at_support', 'at_resistance', 'mid_range', 'at_killzone'
    
    # FLUID LAYERS (Intraday)
    L5_trend: str       # '4hr': 'up', 'down', 'flat'
    L6_structure: str   # '1hr': 'bullish', 'bearish', 'neutral'
    L7_pattern: str     # '15m': 'flag', 'wedge', 'compression', 'breakdown', 'none'
    L8_confirm: bool    # '5m': Setup ready? (True/False)
    L9_cascade: bool    # '1s': Velocity trigger? (True/False)
    
    # METADATA (not part of hash, but useful for analysis)
    timestamp: Optional[float] = None
    price: Optional[float] = None
    
    def __hash__(self):
        """Hash only the state attributes (not metadata)"""
        return hash((
            self.L1_bias, self.L2_regime, self.L3_swing, self.L4_zone,
            self.L5_trend, self.L6_structure, self.L7_pattern,
            self.L8_confirm, self.L9_cascade
        ))
    
    def __eq__(self, other):
        """Equality check for HashMap lookups"""
        if not isinstance(other, StateVector):
            return False
        return hash(self) == hash(other)
    
    def to_dict(self):
        """Export for logging/analysis"""
        return {
            'L1_bias': self.L1_bias,
            'L2_regime': self.L2_regime,
            'L3_swing': self.L3_swing,
            'L4_zone': self.L4_zone,
            'L5_trend': self.L5_trend,
            'L6_structure': self.L6_structure,
            'L7_pattern': self.L7_pattern,
            'L8_confirm': self.L8_confirm,
            'L9_cascade': self.L9_cascade,
            'timestamp': self.timestamp,
            'price': self.price
        }
    
    @classmethod
    def null_state(cls):
        """Default state when no data available"""
        return cls(
            L1_bias='range',
            L2_regime='chopping',
            L3_swing='sideways',
            L4_zone='mid_range',
            L5_trend='flat',
            L6_structure='neutral',
            L7_pattern='none',
            L8_confirm=False,
            L9_cascade=False,
            timestamp=0.0,
            price=0.0
        )
