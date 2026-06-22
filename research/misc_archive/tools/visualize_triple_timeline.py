import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    day = '2024_01_03'
    parquet_path = f'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s/{day}.parquet'
    
    print("Loading data...")
    df = pd.read_parquet(parquet_path).sort_values('timestamp').reset_index(drop=True)

    # 1. Kinematics for Pink Line (15m span = 900 points)
    span_15 = 900 
    df['v_raw'] = df['close'].diff()
    df['v_ema_15'] = df['v_raw'].ewm(span=span_15, adjust=False).mean()
    df['a_raw_15'] = df['v_ema_15'].diff()
    df['a_ema_15'] = df['a_raw_15'].ewm(span=span_15, adjust=False).mean()

    # Target the specific V-Top trade window
    trades_csv = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/reports/findings/nmp_pure_1s_10m_rcurve.csv'
    tr = pd.read_csv(trades_csv)
    worst_trade = tr.loc[tr['net_usd'].idxmin()]
    ets = int(worst_trade['entry_ts'])
    xts = int(worst_trade['exit_ts'])

    window_size = 1800 # 30 mins
    start_ts = ets - window_size
    end_ts = xts + 900 # 15 mins
    
    # Pre-slice data for regression calculations
    blue_sample_size = 1440 # 24 mins
    orange_sample_size = 450 # 7.5 mins (5m + 50%)
    
    calc_start_ts = start_ts - blue_sample_size
    mask_full = (df['timestamp'] >= calc_start_ts) & (df['timestamp'] <= end_ts)
    df_calc = df[mask_full].reset_index(drop=True)
    
    prices_calc = df_calc['close'].values
    times_calc = df_calc['timestamp'].values
    
    # 2. Compute 24m Blue Line (Quadratic Regression)
    blue_t = []
    blue_y = []
    x_mins_blue = np.arange(0, 24, 1/60.0)
    if len(x_mins_blue) > blue_sample_size:
        x_mins_blue = x_mins_blue[:blue_sample_size]
    elif len(x_mins_blue) < blue_sample_size:
        x_mins_blue = np.linspace(0, 24, blue_sample_size)
    
    print("Pre-calculating 1s TF / 24m Blue Regression (Quadratic)...")
    for i in range(blue_sample_size, len(df_calc)):
        curr_ts = times_calc[i-1]
        if curr_ts < start_ts:
            continue
            
        w_p = prices_calc[i-blue_sample_size:i]
        c = np.polyfit(x_mins_blue, w_p, 2)
        y_curr = c[0]*(x_mins_blue[-1]**2) + c[1]*x_mins_blue[-1] + c[2]
        blue_y.append(y_curr)
        blue_t.append(curr_ts)

    blue_t = np.array(blue_t)
    blue_y = np.array(blue_y)

    # 3. Compute 7.5m Orange Line (CUBIC Regression)
    orange_t = []
    orange_y = []
    x_mins_orange = np.arange(0, 7.5, 1/60.0)
    if len(x_mins_orange) > orange_sample_size:
        x_mins_orange = x_mins_orange[:orange_sample_size]
    elif len(x_mins_orange) < orange_sample_size:
        x_mins_orange = np.linspace(0, 7.5, orange_sample_size)
    
    print("Pre-calculating 1s TF / 7.5m Orange Regression (Cubic)...")
    for i in range(orange_sample_size, len(df_calc)):
        curr_ts = times_calc[i-1]
        if curr_ts < start_ts:
            continue
            
        w_p = prices_calc[i-orange_sample_size:i]
        c = np.polyfit(x_mins_orange, w_p, 3) # CUBIC
        x_val = x_mins_orange[-1]
        y_curr = c[0]*(x_val**3) + c[1]*(x_val**2) + c[2]*x_val + c[3]
        orange_y.append(y_curr)
        orange_t.append(curr_ts)

    orange_t = np.array(orange_t)
    orange_y = np.array(orange_y)

    # Prepare zooming mask
    mask_zoom = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
    df_zoom = df[mask_zoom].reset_index(drop=True)

    timestamps_1s = df_zoom['timestamp'].values
    prices_1s = df_zoom['close'].values
    v_emas_15 = df_zoom['v_ema_15'].values
    a_emas_15 = df_zoom['a_ema_15'].values

    fig, ax = plt.subplots(figsize=(12, 7))
    
    rel_time_1s_full = (timestamps_1s - start_ts) / 60.0
    t_forward_15 = np.arange(0, 900, 5) 

    frames = np.arange(0, len(df_zoom), 10)

    def update(frame):
        ax.clear()
        idx_1s = frame
        current_ts = timestamps_1s[idx_1s]
        current_p = prices_1s[idx_1s]
        
        ax.plot(rel_time_1s_full, prices_1s, color='gray', alpha=0.3, label='Static Price Context')
        
        past_times = timestamps_1s[:idx_1s+1]
        past_prices = prices_1s[:idx_1s+1]
        rel_past = (past_times - start_ts) / 60.0
        ax.plot(rel_past, past_prices, color='black', linewidth=1, label='Unfolding Price')
        
        # 1. Macro Blue Line (24m Quadratic Regression)
        mask_blue = (blue_t >= start_ts) & (blue_t <= current_ts)
        rel_blue_t = (blue_t[mask_blue] - start_ts) / 60.0
        if len(rel_blue_t) > 0:
            ax.plot(rel_blue_t, blue_y[mask_blue], color='blue', linewidth=4, zorder=3, label='Macro Anchor (24m Quad)')
            ax.scatter([rel_blue_t[-1]], [blue_y[mask_blue][-1]], color='blue', s=80, zorder=5)

        # 2. Micro Orange Line (7.5m Cubic Regression)
        mask_orange = (orange_t >= start_ts) & (orange_t <= current_ts)
        rel_orange_t = (orange_t[mask_orange] - start_ts) / 60.0
        if len(rel_orange_t) > 0:
            ax.plot(rel_orange_t, orange_y[mask_orange], color='darkorange', linewidth=3, linestyle='-', zorder=4, label='Micro Anchor (7.5m Cubic)')
            ax.scatter([rel_orange_t[-1]], [orange_y[mask_orange][-1]], color='darkorange', s=60, zorder=6)

        # 3. Trigger Pink Line (15m Kinematics)
        v_15 = v_emas_15[idx_1s]
        a_15 = a_emas_15[idx_1s]
        proj_p_15 = current_p + (v_15 * t_forward_15) + (0.5 * a_15 * (t_forward_15**2))
        proj_times_15 = current_ts + t_forward_15
        rel_proj_15 = (proj_times_15 - start_ts) / 60.0
        ax.plot(rel_proj_15, proj_p_15, color='magenta', linewidth=3, zorder=4, label='Trigger (15m Kinematics)')
        
        # Center Dot
        ax.scatter([rel_past[-1]], [current_p], color='magenta', s=40, zorder=7)
        
        ax.set_title(f"Dual Regressions | Blue(Quad): 24m | Orange(Cubic): 7.5m | Pink Trigger: {v_15*60:+.2f}", fontweight='bold')
        ax.set_ylabel("Price")
        ax.set_xlabel("Minutes from Window Start")
        ax.set_ylim(prices_1s.min() - 5, prices_1s.max() + 5)
        ax.set_xlim(0, max(rel_time_1s_full))
        ax.legend(loc='lower left')

    print(f"Generating GIF with {len(frames)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    
    gif_path = "C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/triple_timeline_24m_blue.gif"
    ani.save(gif_path, writer='pillow', fps=10)
    print(f"Saved GIF to {gif_path}")

if __name__ == '__main__':
    main()
