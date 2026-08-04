#!/usr/bin/env python3
"""Causal cubic-regression endpoint — the 'curve regression' (owner 2026-07-27):
NOT z_se (that's the standardized OLS *residual*). This is the actual curve:
fit y = a·t³ + b·t² + c·t + d over a rolling 7.5-min window (t in MINUTES) and
read the ENDPOINT value, slope (1st deriv = velocity/direction) and curvature
(2nd deriv). Replicates docs/nt8/2-CubicRegressionEndpoint_v1.0-RC.cs (the SFE
'orange line'); fixed grid → precomputed weights → O(1) per bar via dot product.
"""
import numpy as np


class CubicEndpoint:
    def __init__(self, window, bar_seconds):
        """window = bars in the trailing fit; bar_seconds = each bar's seconds."""
        self.n = int(window)
        x = np.arange(self.n) * (bar_seconds / 60.0)      # minutes, chronological
        xe = x[-1]
        V = np.vstack([x**3, x**2, x, np.ones_like(x)]).T  # (n,4) design
        Vp = np.linalg.pinv(V)                             # (4,n): coef = Vp @ y
        self.wv = np.array([xe**3, xe**2, xe, 1.0]) @ Vp   # endpoint value weights
        self.ws = np.array([3*xe**2, 2*xe, 1.0, 0.0]) @ Vp  # endpoint slope (pts/min)
        self.wc = np.array([6*xe, 2.0, 0.0, 0.0]) @ Vp     # endpoint curvature (pts/min^2)

    def eval(self, closes_window):
        """closes_window = last n closes (chronological). Returns (value,slope,curv)."""
        y = np.asarray(closes_window, float)
        if len(y) != self.n or not np.isfinite(y).all():
            return np.nan, np.nan, np.nan
        return float(self.wv @ y), float(self.ws @ y), float(self.wc @ y)


def rolling(closes, window, bar_seconds):
    """Vectorized endpoint value/slope/curv at every bar (NaN until warm)."""
    ce = CubicEndpoint(window, bar_seconds)
    c = np.asarray(closes, float)
    n = len(c)
    val = np.full(n, np.nan); slp = np.full(n, np.nan); cur = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = c[i - window + 1:i + 1]
        if np.isfinite(w).all():
            val[i] = ce.wv @ w; slp[i] = ce.ws @ w; cur[i] = ce.wc @ w
    return val, slp, cur


if __name__ == '__main__':
    # self-test: a pure quadratic ramp -> slope>0, curvature ~ const, value≈last
    import numpy as _np
    t = _np.arange(90) * (5 / 60.0)
    y = 100 + 2 * t + 0.5 * t**2
    ce = CubicEndpoint(90, 5)
    v, s, cu = ce.eval(y)
    print(f'value={v:.3f} (last={y[-1]:.3f}) slope={s:.3f} pts/min curv={cu:.3f}')
    print('slope>0:', s > 0, '| value≈last:', abs(v - y[-1]) < 0.5)
