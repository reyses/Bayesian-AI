import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def kalman_ca(p, dt=1.0, q=1.81e-09, r=12.5533):
    n = len(p)
    pos = np.empty(n); vel = np.empty(n); acc = np.empty(n)
    x0, x1, x2 = p[0], 0.0, 0.0
    P00, P01, P02 = 1e3, 0.0, 0.0
    P10, P11, P12 = 0.0, 1e3, 0.0
    P20, P21, P22 = 0.0, 0.0, 1e3
    
    dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt; dt5 = dt4 * dt
    Q00 = q * dt5 / 20.0; Q01 = q * dt4 / 8.0; Q02 = q * dt3 / 6.0
    Q10 = Q01; Q11 = q * dt3 / 3.0; Q12 = q * dt2 / 2.0
    Q20 = Q02; Q21 = Q12; Q22 = q * dt
    
    for i in range(n):
        z = p[i]
        xp0 = x0 + dt * x1 + 0.5 * dt2 * x2
        xp1 = x1 + dt * x2
        xp2 = x2
        
        FP00 = P00 + dt * P10 + 0.5 * dt2 * P20
        FP01 = P01 + dt * P11 + 0.5 * dt2 * P21
        FP02 = P02 + dt * P12 + 0.5 * dt2 * P22
        FP10 = P10 + dt * P20; FP11 = P11 + dt * P21; FP12 = P12 + dt * P22
        FP20 = P20; FP21 = P21; FP22 = P22
        
        Pp00 = FP00 + FP01 * dt + FP02 * 0.5 * dt2 + Q00
        Pp01 = FP01 + FP02 * dt + Q01
        Pp02 = FP02 + Q02
        Pp10 = FP10 + FP11 * dt + FP12 * 0.5 * dt2 + Q10
        Pp11 = FP11 + FP12 * dt + Q11
        Pp12 = FP12 + Q12
        Pp20 = FP20 + FP21 * dt + FP22 * 0.5 * dt2 + Q20
        Pp21 = FP21 + FP22 * dt + Q21
        Pp22 = FP22 + Q22
        
        y = z - xp0
        S = Pp00 + r
        K0 = Pp00 / S; K1 = Pp10 / S; K2 = Pp20 / S
        
        x0 = xp0 + K0 * y
        x1 = xp1 + K1 * y
        x2 = xp2 + K2 * y
        
        pos[i] = x0; vel[i] = x1; acc[i] = x2
        
        P00 = (1.0 - K0) * Pp00; P01 = (1.0 - K0) * Pp01; P02 = (1.0 - K0) * Pp02
        P10 = -K1 * Pp00 + Pp10; P11 = -K1 * Pp01 + Pp11; P12 = -K1 * Pp02 + Pp12
        P20 = -K2 * Pp00 + Pp20; P21 = -K2 * Pp01 + Pp21; P22 = -K2 * Pp02 + Pp22
    return pos, vel, acc

def main():
    trades_csv = 'reports/findings/kalman_full_trades.csv'
    tr = pd.read_csv(trades_csv)
    
    # Pick a big winning trade to visualize the tracking
    trade = tr.loc[tr['net_usd'].idxmax()]
    day = trade['day']
    ets = int(trade['entry_ts'])
    xts = int(trade['exit_ts'])
    
    parquet_path = f'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s/{day}.parquet'
    
    print(f"Loading data for {day}...")
    df = pd.read_parquet(parquet_path).sort_values('timestamp').reset_index(drop=True)
    
    start_ts = ets - 1800 # 30 mins before entry
    end_ts = xts + 600    # 10 mins after exit
    
    mask_full = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
    df_zoom = df[mask_full].reset_index(drop=True)
    
    ts = df_zoom['timestamp'].values
    px = df_zoom['close'].values
    
    # Calculate Kalman CA for the entire window
    print("Calculating tuned Kalman CA...")
    pos, vel, acc = kalman_ca(px)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rel_time = (ts - start_ts) / 60.0
    
    # Generate animation frames (step by 15s to keep GIF small)
    frames = np.arange(0, len(px), 15)
    
    def update(frame):
        ax.clear()
        idx = frame
        current_ts = ts[idx]
        
        # Static background
        ax.plot(rel_time, px, color='gray', alpha=0.3, label='Static Price Context')
        
        # Unfolding price
        ax.plot(rel_time[:idx+1], px[:idx+1], color='black', linewidth=1, label='Unfolding Price')
        
        # Kalman Position
        ax.plot(rel_time[:idx+1], pos[:idx+1], color='#e8730c', linewidth=3, label=f'Kalman Smoothed (q=1.8e-9, r=12.5)')
        ax.scatter([rel_time[idx]], [pos[idx]], color='#e8730c', s=60, zorder=6)
        
        # Entry/Exit markers
        entry_rel = (ets - start_ts) / 60.0
        exit_rel = (xts - start_ts) / 60.0
        
        if current_ts >= ets:
            ax.axvline(x=entry_rel, color='green', linestyle='--', label=f"ENTRY ({trade['dir']})")
            if ts[0] <= ets <= current_ts:
                ex_idx = np.where(ts == ets)[0][0]
                ax.scatter([entry_rel], [px[ex_idx]], color='green', s=100, zorder=10)
                
        if current_ts >= xts:
            ax.axvline(x=exit_rel, color='red', linestyle='--', label='EXIT (79pt Trailing)')
            if ts[0] <= xts <= current_ts:
                xx_idx = np.where(ts == xts)[0][0]
                ax.scatter([exit_rel], [px[xx_idx]], color='red', s=100, zorder=10)
                
        ax.set_title(f"Kalman CA Tuning | Trade PnL: ${trade['net_usd']:.2f} | Vel: {vel[idx]*60:+.2f} pts/min", fontweight='bold')
        ax.set_ylabel("Price")
        ax.set_xlabel("Minutes from Start")
        ax.set_ylim(px.min() - 5, px.max() + 5)
        ax.set_xlim(0, max(rel_time))
        ax.legend(loc='lower left')

    print(f"Generating GIF with {len(frames)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    
    os.makedirs('reports/findings', exist_ok=True)
    gif_path = "reports/findings/kalman_tuned_trade.gif"
    ani.save(gif_path, writer='pillow', fps=10)
    print(f"Saved GIF to {gif_path}")

if __name__ == '__main__':
    main()
