"""Forward Pass System — canonical V2 bar-by-bar forward pass.

The only supported forward pass for V2. Any code running a bar-by-bar
simulation must use ForwardPassSystem or MultiDayForwardPassSystem from
this package. No hardcoded data paths — callers supply atlas_root,
features_root, and labels_csv explicitly.
"""
from .forward_pass_system import ForwardPassSystem, MultiDayForwardPassSystem
from .state import BarState

__all__ = ['ForwardPassSystem', 'MultiDayForwardPassSystem', 'BarState']
