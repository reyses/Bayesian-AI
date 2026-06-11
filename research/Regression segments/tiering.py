import numpy as np

def max_consecutive(arr):
    if not np.any(arr): return 0
    padded = np.pad(arr, (1, 1), mode='constant')
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return np.max(ends - starts)

def classify_tier(residuals: np.ndarray, E: float, max_tier: int = 8) -> int:
    """Tier t passes if max_res <= (1.0 + 0.5*t)*E and
    max_consecutive(residuals > (0.5 + 0.5*t)*E) < 3. Returns max_tier+1 if none pass."""
    for t in range(1, max_tier + 1):
        hi = (1.0 + 0.5 * t) * E
        lo = (0.5 + 0.5 * t) * E
        if residuals.max() <= hi and max_consecutive(residuals > lo) < 3:
            return t
    return max_tier + 1
