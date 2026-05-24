"""
Base plugin interface for the VizEngine.
"""
from typing import Optional, List, Dict, Any
import matplotlib.axes

class VizPlugin:
    """Base class for all VizEngine plugins."""
    
    def __init__(self):
        self.engine = None  # Reference to the VizEngine

    def setup(self, engine, **kwargs):
        """Called once when the plugin is loaded into the engine.
        
        Args:
            engine: The VizEngine instance.
            kwargs: Command-line arguments or other initialization data.
        """
        self.engine = engine

    def draw(self, ax: matplotlib.axes.Axes, ax_ind: Optional[matplotlib.axes.Axes], time_range: tuple, patches_list: list):
        """Called whenever the engine redraws the scene.
        
        Plugins should draw their specific overlays (markers, zones, text) here
        and append the created patches to patches_list so they can be cleared.
        
        Args:
            ax: The main matplotlib axes.
            ax_ind: The indicator axes (if requires_indicator_panel is True).
            time_range: A tuple of (start_timestamp, end_timestamp).
            patches_list: List to append matplotlib artists to.
        """
        pass

    def on_key(self, event) -> bool:
        """Called on keyboard events."""
        return False

    def on_click(self, event) -> bool:
        """Called on mouse click events."""
        return False

    def get_title_stats(self) -> str:
        """Returns a string to append to the engine's main title."""
        return ""

    def get_buttons(self) -> list:
        """Returns a list of dicts for on-screen buttons.
        Example: [{'label': 'Grade A', 'action': self.grade_a}]
        """
        return []

    def get_sliders(self) -> list:
        """Returns a list of dicts for on-screen Tkinter scales/sliders.
        Example: [{'label': 'Threshold', 'min': 0, 'max': 100, 'step': 1, 'valinit': 50, 'action': self.set_threshold}]
        """
        return []
