import numpy as np
from scipy.ndimage import correlate1d

def get_cubic_weights(N: int):
    """
    Computes exact OLS projection weights for a centered cubic polynomial.
    Returns weights for price (0th deriv), slope (1st deriv), and curvature (2nd deriv) at the center.
    """
    x = np.arange(N) - N // 2
    # Design matrix: [x^3, x^2, x, 1]
    X = np.vstack([x**3, x**2, x, np.ones(N)]).T
    # (X^T X)^-1 X^T
    try:
        P = np.linalg.inv(X.T @ X) @ X.T
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(X)
    
    # smoothed price = coeff of 1 (index 3)
    w_price = P[3, :]
    # slope = coeff of x (index 2)
    w_slope = P[2, :]
    # curvature = 2 * coeff of x^2 (index 1)
    w_curv = 2 * P[1, :]
    
    return w_price, w_slope, w_curv

def find_raw_turns(close_prices: np.ndarray, N: int):
    """
    Apply centered cubic regression to find candidate pivots (turns).
    Returns (turns, price_smooth, slope, curv).
    turns is a list of dicts: {'index': i, 'type': 'top'|'bottom'}
    """
    w_price, w_slope, w_curv = get_cubic_weights(N)
    
    # Sliding dot product
    price_smooth = correlate1d(close_prices, w_price, mode='constant', cval=np.nan)
    slope = correlate1d(close_prices, w_slope, mode='constant', cval=np.nan)
    curv = correlate1d(close_prices, w_curv, mode='constant', cval=np.nan)
    
    # Explicitly nullify edges
    trim = N // 2
    price_smooth[:trim] = np.nan
    price_smooth[-trim:] = np.nan
    slope[:trim] = np.nan
    slope[-trim:] = np.nan
    curv[:trim] = np.nan
    curv[-trim:] = np.nan
    
    sign_slope = np.sign(slope)
    
    turns = []
    # Find zero crossings in slope
    for i in range(1, len(sign_slope)):
        if np.isnan(sign_slope[i]) or np.isnan(sign_slope[i-1]):
            continue
            
        if sign_slope[i] != sign_slope[i-1] and sign_slope[i] != 0:
            if curv[i] < 0:
                turns.append({'index': i, 'type': 'top'})
            elif curv[i] > 0:
                turns.append({'index': i, 'type': 'bottom'})
                
    return turns, price_smooth, slope, curv
