"""
Calibration — Binomial logistic regression per horizon per TF.

Converts raw model P(D) outputs to calibrated probabilities that match
actual outcomes. Also computes the chop zone boundary per horizon.

Usage:
  cal = TrajectoryCalibrator.fit(model, val_features, val_labels, horizons)
  calibrated_p = cal.calibrate(raw_p)  # (K,) raw -> (K,) calibrated
  chop_zone = cal.chop_zones  # per-horizon [low, high] boundaries
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import json
import os


class HorizonCalibrator:
    """Calibrates one horizon's P(D) via logistic regression."""

    def __init__(self):
        self.lr = None
        self.a = 0.0  # logistic slope
        self.b = 0.0  # logistic intercept
        self.chop_low = 0.45   # default
        self.chop_high = 0.55  # default

    def fit(self, raw_p, actual_dir):
        """Fit logistic regression: actual_direction ~ raw_P(long).

        raw_p: model's P(long) predictions (n,)
        actual_dir: binary direction labels (n,) — 1=long, 0=short
        """
        X = raw_p.reshape(-1, 1)
        y = actual_dir.astype(int)

        self.lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.lr.fit(X, y)

        self.a = float(self.lr.coef_[0][0])
        self.b = float(self.lr.intercept_[0])

        # Chop zone: where 95% CI of P(actual=long) includes 50%
        # Approximate via Wilson CI on binned predictions
        n_bins = 50
        bin_edges = np.linspace(0, 1, n_bins + 1)
        z = 1.96

        chop_low = 0.0
        chop_high = 1.0

        for i in range(n_bins):
            mask = (raw_p >= bin_edges[i]) & (raw_p < bin_edges[i + 1])
            n = mask.sum()
            if n < 10:
                continue
            p_hat = actual_dir[mask].mean()
            # Wilson CI
            denom = 1 + z ** 2 / n
            center = (p_hat + z ** 2 / (2 * n)) / denom
            spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
            ci_low = center - spread
            ci_high = center + spread

            # If CI includes 50%, this bin is in the chop zone
            if ci_low <= 0.5 <= ci_high:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                chop_low = min(chop_low, bin_center) if chop_low > 0 else bin_center
                chop_high = max(chop_high, bin_center) if chop_high < 1 else bin_center

        self.chop_low = chop_low
        self.chop_high = chop_high

    def calibrate(self, raw_p):
        """Convert raw P(long) to calibrated probability."""
        return expit(self.a * raw_p + self.b)

    def to_dict(self):
        return {
            'a': self.a, 'b': self.b,
            'chop_low': self.chop_low, 'chop_high': self.chop_high,
        }

    @classmethod
    def from_dict(cls, d):
        cal = cls()
        cal.a = d['a']
        cal.b = d['b']
        cal.chop_low = d['chop_low']
        cal.chop_high = d['chop_high']
        return cal


class TrajectoryCalibrator:
    """Calibrates all horizons for one TF's TrajectoryPredictor."""

    def __init__(self, horizons):
        self.horizons = horizons
        self.calibrators = [HorizonCalibrator() for _ in horizons]

    def fit(self, raw_p_all, actual_dir_all):
        """Fit calibration for each horizon.

        raw_p_all: (n, K) model P(long) per horizon
        actual_dir_all: (n, K) actual direction per horizon
        """
        for hi in range(len(self.horizons)):
            self.calibrators[hi].fit(raw_p_all[:, hi], actual_dir_all[:, hi])

        print(f"  Calibration fitted ({len(self.horizons)} horizons):")
        for hi, h in enumerate(self.horizons):
            cal = self.calibrators[hi]
            print(f"    n+{h}: a={cal.a:.3f} b={cal.b:.3f} "
                  f"chop=[{cal.chop_low:.2f}, {cal.chop_high:.2f}]")

    def calibrate(self, raw_p):
        """Calibrate a trajectory curve.

        raw_p: (K,) raw P(long) per horizon
        Returns: (K,) calibrated P(long)
        """
        calibrated = np.zeros_like(raw_p)
        for hi in range(len(self.horizons)):
            calibrated[hi] = self.calibrators[hi].calibrate(raw_p[hi])
        return calibrated

    def calibrate_batch(self, raw_p_all):
        """Calibrate a batch of trajectories.

        raw_p_all: (n, K) raw P(long) per horizon
        Returns: (n, K) calibrated
        """
        calibrated = np.zeros_like(raw_p_all)
        for hi in range(len(self.horizons)):
            calibrated[:, hi] = self.calibrators[hi].calibrate(raw_p_all[:, hi])
        return calibrated

    @property
    def chop_zones(self):
        """Per-horizon chop zone boundaries: list of (low, high)."""
        return [(c.chop_low, c.chop_high) for c in self.calibrators]

    def save(self, path):
        """Save calibration to JSON."""
        data = {
            'horizons': self.horizons,
            'calibrators': [c.to_dict() for c in self.calibrators],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load calibration from JSON."""
        with open(path) as f:
            data = json.load(f)
        cal = cls(data['horizons'])
        cal.calibrators = [HorizonCalibrator.from_dict(d) for d in data['calibrators']]
        return cal
