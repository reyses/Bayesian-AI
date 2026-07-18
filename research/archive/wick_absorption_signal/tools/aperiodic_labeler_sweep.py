import pandas as pd
import numpy as np
from scipy.ndimage import correlate1d
from pathlib import Path

RAW_DATA_PATH = "DATA/ATLAS/order_flow_delta_5s.parquet"
OUTPUT_DIR = Path("research/wick_absorption_signal/reports/labeler_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_cubic_weights(N):
    """
    Computes exact OLS projection weights for a centered cubic polynomial.
    Returns weights for slope (1st deriv) and curvature (2nd deriv) at the center.
    """
    x = np.arange(N) - N//2
    # Design matrix: [x^3, x^2, x, 1]
    X = np.vstack([x**3, x**2, x, np.ones(N)]).T
    # (X^T X)^-1 X^T
    P = np.linalg.inv(X.T @ X) @ X.T
    
    # slope = coeff of x (index 2)
    w_slope = P[2, :]
    # curvature = 2 * coeff of x^2 (index 1)
    w_curv = 2 * P[1, :]
    
    return w_slope, w_curv

def find_raw_turns(close_prices, N):
    w_slope, w_curv = get_cubic_weights(N)
    
    # Use correlate1d (sliding dot product, no kernel flipping)
    slope = correlate1d(close_prices, w_slope, mode='constant', cval=np.nan)
    curv = correlate1d(close_prices, w_curv, mode='constant', cval=np.nan)
    
    # Explicitly nullify the N/2 edges (centered window lookahead/lookback)
    trim = N // 2
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
            # We have a turn at i
            # Top = curvature < 0, Bottom = curvature > 0
            if curv[i] < 0:
                turns.append({'index': i, 'type': 'top'})
            elif curv[i] > 0:
                turns.append({'index': i, 'type': 'bottom'})
                
    return turns

def apply_magnitude_filter(turns, prices, sigmas, k):
    if len(turns) == 0:
        return []
        
    valid_turns = []
    last_turn = turns[0]
    valid_turns.append(last_turn)
    
    for current_turn in turns[1:]:
        if current_turn['type'] == last_turn['type']:
            # Replace if more extreme (ZigZag accumulation)
            p_curr = prices[current_turn['index']]
            p_last = prices[last_turn['index']]
            if current_turn['type'] == 'top' and p_curr > p_last:
                valid_turns[-1] = current_turn
                last_turn = current_turn
            elif current_turn['type'] == 'bottom' and p_curr < p_last:
                valid_turns[-1] = current_turn
                last_turn = current_turn
        else:
            # Different type, evaluate significance
            p_curr = prices[current_turn['index']]
            p_last = prices[last_turn['index']]
            pct_diff = abs(p_curr - p_last) / p_last
            
            # Trailing sigma AT the current turn (causal)
            sigma_pct = sigmas[current_turn['index']]
            
            if pct_diff >= k * sigma_pct:
                valid_turns.append(current_turn)
                last_turn = current_turn
                
    return valid_turns

def df_to_md_table(df):
    headers = " | ".join(df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(x) for x in row.values))
    return f"| {headers} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows)

def run():
    print("Loading 1m data for sweep...")
    df_raw = pd.read_parquet(RAW_DATA_PATH).sort_index()
    df = df_raw.resample('1min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    prices = df['close'].values
    
    # Trailing rolling volatility (causal)
    # 50-bar standard deviation of log returns
    log_rets = np.log(prices[1:] / prices[:-1])
    log_rets = np.insert(log_rets, 0, 0)
    sigmas = pd.Series(log_rets).rolling(50, min_periods=20).std().bfill().values
    
    # ---------------------------------------------------------
    # SWEEP 1: N-Stability (Fix k=3.0, vary N)
    # ---------------------------------------------------------
    print("Running N-Stability Sweep...")
    n_sweep = [12, 20, 40]
    n_results = []
    
    k_fixed = 3.0
    for N in n_sweep:
        raw_turns = find_raw_turns(prices, N)
        filtered_turns = apply_magnitude_filter(raw_turns, prices, sigmas, k_fixed)
        
        # Calculate stats
        turn_indices = [t['index'] for t in filtered_turns]
        spacings = np.diff(turn_indices)
        
        n_results.append({
            'Resolution (N)': N,
            'Raw Turns': len(raw_turns),
            'Filtered Turns (k=3.0)': len(filtered_turns),
            'Median Spacing (Mins)': f"{np.median(spacings):.1f}" if len(spacings) > 0 else "NaN",
            'Mean Spacing (Mins)': f"{np.mean(spacings):.1f}" if len(spacings) > 0 else "NaN"
        })
        
    df_n = pd.DataFrame(n_results)
    
    # ---------------------------------------------------------
    # SWEEP 2: k-Scale (Fix N=20, vary k)
    # ---------------------------------------------------------
    print("Running k-Scale Sweep...")
    k_sweep = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    k_results = []
    
    N_fixed = 20
    raw_turns_fixed = find_raw_turns(prices, N_fixed)
    
    for k in k_sweep:
        filtered_turns = apply_magnitude_filter(raw_turns_fixed, prices, sigmas, k)
        
        turn_indices = [t['index'] for t in filtered_turns]
        spacings = np.diff(turn_indices)
        
        # Calculate swing magnitudes
        magnitudes_pct = []
        for i in range(1, len(filtered_turns)):
            p_curr = prices[filtered_turns[i]['index']]
            p_last = prices[filtered_turns[i-1]['index']]
            magnitudes_pct.append(abs(p_curr - p_last) / p_last * 100.0) # In %
            
        k_results.append({
            'Magnitude Threshold (k)': k,
            'Filtered Turns': len(filtered_turns),
            'Median Swing Mag (%)': f"{np.median(magnitudes_pct):.3f}%" if len(magnitudes_pct) > 0 else "NaN",
            'Mean Swing Mag (%)': f"{np.mean(magnitudes_pct):.3f}%" if len(magnitudes_pct) > 0 else "NaN",
            'Median Spacing (Mins)': f"{np.median(spacings):.1f}" if len(spacings) > 0 else "NaN"
        })
        
    df_k = pd.DataFrame(k_results)
    
    # Generate Output
    summary = f"""# Aperiodic Turn-Labeler Diagnostics

**Timeframe:** 1-minute
**Volatility Filter:** 50-bar trailing standard deviation of log-returns (causal).
**Turn Logic:** Plain centered cubic regression (slope=0, curv sign) + ZigZag accumulation filter.

## Table 1: Resolution Stability (N-Sweep)
We fix the magnitude threshold $k=3.0$ and vary the centered cubic resolution $N$. 
If the located turn set is stable, $N$ is just a resolution slider, and we can safely lock $N=20$.

{df_to_md_table(df_n)}

## Table 2: Economic Scale (k-Sweep)
We fix the resolution $N=20$ and vary the magnitude threshold $k$. 
Choose $k$ based on the **Median Swing Magnitude** that corresponds to the minimum move size the strategy intends to capture.

{df_to_md_table(df_k)}
"""

    with open(OUTPUT_DIR / "labeler_diagnostics.md", "w") as f:
        f.write(summary)
        
    print("Diagnostics complete. Saved to reports/labeler_sweep/labeler_diagnostics.md")

if __name__ == "__main__":
    run()
