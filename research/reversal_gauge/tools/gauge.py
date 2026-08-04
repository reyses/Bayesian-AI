"""Reversal gauge scorer — pure inference over a fitted logistic combiner.

Loads research/reversal_gauge/coeffs.json (produced by
pipeline/fit_combiner.py) and scores a feature dict into
p(resume) plus per-feature signed contributions. No I/O besides
reading coeffs.json.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

# coeffs.json lives at the research project root, one level above tools/.
COEFFS_PATH = Path(__file__).resolve().parent.parent / "coeffs.json"

PCT_SCALE = 100.0
DEFAULT_N_TOP = 3

# Short display names for the one-line gauge string.
_SHORT_NAMES = {
    "giveback_frac": "giveback",
    "pace_pts_s": "pace",
    "spike_score": "spike",
    "repoke": "repoke",
    "worn_touches": "worn",
    "is_flushV": "flushV",
    "clock_sin": "clock_sin",
    "clock_cos": "clock_cos",
}


@lru_cache(maxsize=1)
def load_coeffs():
    """Load and cache the fitted combiner parameters."""
    with open(COEFFS_PATH) as f:
        return json.load(f)


def p_resume(feat):
    """Score a feature dict.

    Returns (p, drivers) where p is the resume probability and drivers is
    a list of (feature_name, signed_contribution) in coeffs feature order.
    Missing features fall back to the stored training mean, i.e. z = 0
    and contribution exactly 0.
    """
    c = load_coeffs()
    logit = c["intercept"]
    drivers = []
    for name, mean, std, coef in zip(
        c["features"], c["means"], c["stds"], c["coef"]
    ):
        x = feat.get(name)
        # Missing feature or degenerate (zero-variance) column -> exact 0
        # contribution (avoids -0.0 from a negative coef times 0.0).
        if x is None or std == 0.0:
            contribution = 0.0
        else:
            contribution = coef * (float(x) - mean) / std
        logit += contribution
        drivers.append((name, contribution))
    p = 1.0 / (1.0 + math.exp(-logit))
    return p, drivers


def format_gauge(p, drivers, n_top=DEFAULT_N_TOP):
    """One-line gauge string, e.g. 'p(resume) 62% | +pace +flushV -giveback'."""
    ranked = sorted(drivers, key=lambda d: abs(d[1]), reverse=True)
    parts = []
    for name, contribution in ranked[:n_top]:
        if contribution == 0.0:
            continue  # absent/degenerate features carry no signal
        sign = "+" if contribution > 0 else "-"
        parts.append(sign + _SHORT_NAMES.get(name, name))
    return f"p(resume) {round(p * PCT_SCALE)}% | " + " ".join(parts)


if __name__ == "__main__":
    # Smoke test: synthetic freeze event, one feature deliberately missing
    # (worn_touches) to exercise the mean-fallback path.
    synthetic = {
        "giveback_frac": 0.32,
        "pace_pts_s": 0.09,
        "spike_score": 0.55,
        "repoke": 1,
        "is_flushV": 1,
        "clock_sin": 0.5,
        "clock_cos": -0.87,
    }
    p, drivers = p_resume(synthetic)
    print(format_gauge(p, drivers))
    for name, contribution in sorted(
        drivers, key=lambda d: abs(d[1]), reverse=True
    ):
        print(f"  {name:<14} {contribution:+.4f}")
