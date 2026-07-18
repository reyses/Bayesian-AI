import numpy as np
import pandas as pd
import os
import time

def get_poly_slope_weights(window_size, degree):
    x = np.arange(window_size)
    X = np.vander(x, degree + 1)
    P = np.linalg.pinv(X)
    
    if degree == 2:
        weights = 2 * x[-1] * P[0, :] + P[1, :]
    elif degree == 3:
        weights = 3 * (x[-1]**2) * P[0, :] + 2 * x[-1] * P[1, :] + P[2, :]
    return weights

def run_backtest(dates):
    # Setup projection weights for instantaneous slope calculations
    blue_w = get_poly_slope_weights(1440, 2)  # 24m Quadratic
    orange_w = get_poly_slope_weights(450, 3) # 7.5m Cubic
    
    all_trades = []
    
    for day in dates:
        parquet_path = f'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s/{day}.parquet'
        if not os.path.exists(parquet_path):
            continue
            
        print(f"Processing {day}...")
        df = pd.read_parquet(parquet_path).sort_values('timestamp').reset_index(drop=True)
        
        # Pink line kinematics (15m span)
        span_15 = 900
        v_raw = df['close'].diff()
        v_ema_15 = v_raw.ewm(span=span_15, adjust=False).mean()
        
        prices = df['close'].values
        timestamps = df['timestamp'].values
        pink_v = v_ema_15.values
        
        current_position = 0 # 0=flat, 1=LONG, -1=SHORT
        entry_price = 0
        entry_ts = 0
        
        for i in range(1440, len(prices)):
            # The weights are dotted against the window of prices up to i-1
            w_blue = prices[i-1440:i]
            blue_slope = np.dot(blue_w, w_blue)
            
            w_orange = prices[i-450:i]
            orange_slope = np.dot(orange_w, w_orange)
            
            p_v = pink_v[i-1]
            
            if current_position == 0:
                if orange_slope > 0 and blue_slope > 0 and p_v > 0:
                    current_position = 1
                    entry_price = prices[i] # Execute on current 1s tick
                    entry_ts = timestamps[i]
                elif orange_slope < 0 and blue_slope < 0 and p_v < 0:
                    current_position = -1
                    entry_price = prices[i]
                    entry_ts = timestamps[i]
            elif current_position == 1:
                pnl_open = (prices[i] - entry_price) * 20
                
                # Exit conditions: Geometric OR Hard Stop
                if orange_slope < 0 or p_v < 0 or pnl_open <= -100.0:
                    exit_price = prices[i]
                    exit_ts = timestamps[i]
                    
                    pnl = exit_price - entry_price
                    # Commission and slippage
                    gross = pnl * 20
                    net = gross - 2.5 # standard NQ round trip friction
                    
                    all_trades.append({
                        'entry_ts': pd.to_datetime(entry_ts),
                        'exit_ts': pd.to_datetime(exit_ts),
                        'gross_usd': gross,
                        'net_usd': net,
                        'type': 'LONG'
                    })
                    current_position = 0
            elif current_position == -1:
                pnl_open = (entry_price - prices[i]) * 20
                
                if orange_slope > 0 or p_v > 0 or pnl_open <= -100.0:
                    exit_price = prices[i]
                    exit_ts = timestamps[i]
                    
                    pnl = entry_price - exit_price
                    gross = pnl * 20
                    net = gross - 2.5
                    
                    all_trades.append({
                        'entry_ts': pd.to_datetime(entry_ts),
                        'exit_ts': pd.to_datetime(exit_ts),
                        'gross_usd': gross,
                        'net_usd': net,
                        'type': 'SHORT'
                    })
                    current_position = 0
                    
    df_trades = pd.DataFrame(all_trades)
    return df_trades

def main():
    dates = [f'2024_01_{str(d).zfill(2)}' for d in range(2, 10)]
    start_time = time.time()
    df_t = run_backtest(dates)
    end_time = time.time()
    
    if len(df_t) == 0:
        print("No trades executed.")
        return

    win_rate = (df_t['net_usd'] > 0).mean() * 100
    total_pnl = df_t['net_usd'].sum()
    avg_trade = df_t['net_usd'].mean()
    max_dd = df_t['net_usd'].min()
    max_win = df_t['net_usd'].max()
    
    print("\n--- TRIPLE TIMELINE (Trajectory Option 2) BACKTEST ---")
    print(f"Dates: 1 Week (Early Jan 2024)")
    print(f"Total Trades: {len(df_t)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Net PnL: ${total_pnl:.2f}")
    print(f"Average Trade: ${avg_trade:.2f}")
    print(f"Worst Trade: ${max_dd:.2f}")
    print(f"Best Trade: ${max_win:.2f}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    
    df_t.to_csv('C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/reports/findings/nmp_triple_timeline_option2.csv', index=False)

if __name__ == "__main__":
    main()
