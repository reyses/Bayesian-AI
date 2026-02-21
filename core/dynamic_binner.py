"""
Dynamic Histogram Binner (Freedman-Diaconis / Sturges)

Computes optimal bin edges from observed data, then maps any value
to its bin center.  All bin values remain continuous floats — no
categorical conversion.

Approach mirrors Minitab's automatic histogram binning:
  1. Collect sample of observed values
  2. Choose bin count via Freedman-Diaconis rule (robust to outliers)
     Fallback: Sturges' rule when IQR == 0
  3. Build equal-width bins across [min, max]
  4. Map each new value → nearest bin center (float)

Usage:
    binner = DynamicBinner()
    binner.fit({'z_score': z_values, 'momentum': mom_values})
    z_bin = binner.transform('z_score', 2.37)   # → e.g. 2.25
    m_bin = binner.transform('momentum', 0.81)  # → e.g. 0.75
"""
import numpy as np
import pickle
import bisect
from typing import Dict, Optional, List


class VariableBins:
    """Bin specification for a single continuous variable."""

    __slots__ = ('edges', 'centers', 'n_bins', 'edge_list', 'center_list')

    def __init__(self, edges: np.ndarray):
        self.edges = edges                               # (n_bins + 1,)
        self.centers = (edges[:-1] + edges[1:]) / 2.0   # (n_bins,)
        self.n_bins = len(self.centers)
        self.edge_list: List[float] = edges.tolist()     # Python list for faster bisect
        self.center_list: List[float] = self.centers.tolist() # Python list for faster access

    def transform(self, value: float) -> float:
        """Map a single value to its bin center."""
        # bisect: faster than np.searchsorted for scalar lookups
        # using python list avoids numpy array access overhead
        idx = bisect.bisect_right(self.edge_list, value) - 1
        idx = max(0, min(idx, self.n_bins - 1))
        return self.center_list[idx]

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        """Vectorized: map array of values to bin centers."""
        idx = np.searchsorted(self.edges, values, side='right') - 1
        idx = np.clip(idx, 0, self.n_bins - 1)
        return self.centers[idx]


class DynamicBinner:
    """
    Fits histogram bin edges per variable, then maps values to bin centers.

    Thread-safe for reads after fit() — no mutation during transform.
    Serializable via pickle for checkpoint persistence.
    """

    def __init__(self, min_bins: int = 5, max_bins: int = 30):
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.variables: Dict[str, VariableBins] = {}
        self._fitted = False

    def fit(self, data: Dict[str, np.ndarray]):
        """
        Compute optimal bin edges for each variable from observed data.

        Args:
            data: {variable_name: 1-D array of observed values}
        """
        self.variables = {}

        for name, values in data.items():
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]

            if len(values) < 2:
                # Degenerate: single-bin covering [-inf, inf]
                self.variables[name] = VariableBins(np.array([-1e10, 1e10]))
                continue

            n = len(values)
            v_min, v_max = float(np.min(values)), float(np.max(values))

            # ── Freedman-Diaconis bin width ──
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25

            if iqr > 0:
                fd_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
                n_bins = int(np.ceil((v_max - v_min) / fd_width))
            else:
                # Fallback: Sturges' rule
                n_bins = int(np.ceil(1.0 + np.log2(n)))

            n_bins = max(self.min_bins, min(n_bins, self.max_bins))

            # ── Equal-width edges ──
            edges = np.linspace(v_min, v_max, n_bins + 1)

            self.variables[name] = VariableBins(edges)

        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def transform(self, name: str, value: float) -> float:
        """Map a single value to its bin center for the named variable."""
        if name not in self.variables:
            return value  # passthrough if variable unknown
        return self.variables[name].transform(value)

    def get_info(self, name: str) -> Optional[Dict]:
        """Return bin metadata for a variable (for logging/debugging)."""
        if name not in self.variables:
            return None
        vb = self.variables[name]
        return {
            'n_bins': vb.n_bins,
            'min': float(vb.edges[0]),
            'max': float(vb.edges[-1]),
            'width': float(vb.edges[1] - vb.edges[0]),
            'centers': vb.centers.tolist(),
        }

    def save(self, filepath: str):
        """Persist binner to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'min_bins': self.min_bins,
                'max_bins': self.max_bins,
                'variables': {
                    name: vb.edges for name, vb in self.variables.items()
                },
                'fitted': self._fitted,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'DynamicBinner':
        """Load binner from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        binner = cls(min_bins=data['min_bins'], max_bins=data['max_bins'])
        for name, edges in data['variables'].items():
            binner.variables[name] = VariableBins(np.asarray(edges))
        binner._fitted = data.get('fitted', True)
        return binner

    def summary(self) -> str:
        """Human-readable summary of all fitted variables."""
        lines = ["DynamicBinner:"]
        for name, vb in self.variables.items():
            width = vb.edges[1] - vb.edges[0] if vb.n_bins > 0 else 0
            lines.append(
                f"  {name}: {vb.n_bins} bins, "
                f"range [{vb.edges[0]:.3f}, {vb.edges[-1]:.3f}], "
                f"width {width:.3f}"
            )
        return "\n".join(lines)
