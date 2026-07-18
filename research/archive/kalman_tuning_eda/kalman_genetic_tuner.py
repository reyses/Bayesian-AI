import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import numba

@numba.njit
def create_quadratic_weights(window_size):
    w = np.empty(window_size)
    for i in range(1, window_size + 1):
        w[i-1] = (i / window_size)**2
    
    mean_w = np.mean(w)
    for i in range(window_size):
        w[i] -= mean_w
        
    sum_w = np.sum(np.abs(w))
    for i in range(window_size):
        w[i] /= sum_w
    return w

@numba.njit
def kalman_ca_numba(p, dt, q, r):
    n = len(p)
    pos = np.empty(n)
    vel = np.empty(n)
    acc = np.empty(n)
    
    x0, x1, x2 = p[0], 0.0, 0.0
    
    P00, P01, P02 = 1e3, 0.0, 0.0
    P10, P11, P12 = 0.0, 1e3, 0.0
    P20, P21, P22 = 0.0, 0.0, 1e3
    
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    
    Q00 = q * dt5 / 20.0
    Q01 = q * dt4 / 8.0
    Q02 = q * dt3 / 6.0
    
    Q10 = Q01
    Q11 = q * dt3 / 3.0
    Q12 = q * dt2 / 2.0
    
    Q20 = Q02
    Q21 = Q12
    Q22 = q * dt
    
    for i in range(n):
        z = p[i]
        
        xp0 = x0 + dt * x1 + 0.5 * dt2 * x2
        xp1 = x1 + dt * x2
        xp2 = x2
        
        FP00 = P00 + dt * P10 + 0.5 * dt2 * P20
        FP01 = P01 + dt * P11 + 0.5 * dt2 * P21
        FP02 = P02 + dt * P12 + 0.5 * dt2 * P22
        
        FP10 = P10 + dt * P20
        FP11 = P11 + dt * P21
        FP12 = P12 + dt * P22
        
        FP20 = P20
        FP21 = P21
        FP22 = P22
        
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
        
        K0 = Pp00 / S
        K1 = Pp10 / S
        K2 = Pp20 / S
        
        x0 = xp0 + K0 * y
        x1 = xp1 + K1 * y
        x2 = xp2 + K2 * y
        
        pos[i] = x0
        vel[i] = x1
        acc[i] = x2
        
        P00 = (1.0 - K0) * Pp00
        P01 = (1.0 - K0) * Pp01
        P02 = (1.0 - K0) * Pp02
        
        P10 = -K1 * Pp00 + Pp10
        P11 = -K1 * Pp01 + Pp11
        P12 = -K1 * Pp02 + Pp12
        
        P20 = -K2 * Pp00 + Pp20
        P21 = -K2 * Pp01 + Pp21
        P22 = -K2 * Pp02 + Pp22
        
    return pos, vel, acc

@numba.njit
def fast_backtest_day(px, b_slope, dt, q, r, entry_vel_thresh, exit_type, trailing_pts):
    n = len(px)
    pos, vel, acc = kalman_ca_numba(px, dt, q, r)
    
    in_pos = 0
    eprice = 0.0
    mfe_px = 0.0
    
    net_usd = 0.0
    trades = 0
    
    for i in range(1440, n):
        p = px[i]
        v = vel[i]
        a = acc[i]
        b = b_slope[i]
        
        if i == n - 1 and in_pos != 0:
            pnl = (p - eprice) * 2.0 if in_pos == 1 else (eprice - p) * 2.0
            net_usd += (pnl - 2.5)
            trades += 1
            in_pos = 0
            continue
            
        if in_pos != 0:
            if in_pos == 1:
                mfe_px = max(mfe_px, p)
            else:
                mfe_px = min(mfe_px, p)
                
            pnl_pts = (p - eprice) if in_pos == 1 else (eprice - p)
            pnl_usd = pnl_pts * 2.0
            
            # $100 stop loss (MNQ = $2/pt = 50 pts)
            hit_stop = pnl_usd <= -100.0
            
            exit_signal = False
            if exit_type == 0:
                # Vel Flip
                exit_signal = (in_pos == 1 and v < 0) or (in_pos == -1 and v > 0)
            elif exit_type == 1:
                # Accel Flip
                exit_signal = (in_pos == 1 and a < 0) or (in_pos == -1 and a > 0)
            elif exit_type == 2:
                # Blue Flip
                exit_signal = (in_pos == 1 and b < 0) or (in_pos == -1 and b > 0)
            elif exit_type == 3:
                # Trailing Giveback
                if in_pos == 1:
                    exit_signal = p <= mfe_px - trailing_pts
                else:
                    exit_signal = p >= mfe_px + trailing_pts
                    
            if hit_stop or exit_signal:
                net_usd += (pnl_usd - 2.5)
                trades += 1
                in_pos = 0
                
        if in_pos == 0:
            if b > 0 and v > entry_vel_thresh and vel[i-1] <= entry_vel_thresh:
                in_pos = 1
                eprice = p
                mfe_px = p
            elif b < 0 and v < -entry_vel_thresh and vel[i-1] >= -entry_vel_thresh:
                in_pos = -1
                eprice = p
                mfe_px = p
                
    return net_usd, trades

def main():
    ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
    
    print("Loading H1 2024 (IS) Data into memory...")
    # Get all days in H1 (01 to 06)
    files = []
    for m in ['01', '02', '03', '04', '05', '06']:
        files.extend(glob.glob(f'{ATLAS}/2024_{m}_*.parquet'))
    files = sorted(files)
    
    # Pre-calculate 24m blue slopes to save massive time
    # 24m = 1440 seconds
    blue_w = create_quadratic_weights(1440)
    
    days_data = []
    
    for f in files:
        df = pd.read_parquet(f, columns=['close'])
        px = df['close'].to_numpy(np.float64)
        if len(px) < 1440:
            continue
        
        # Convolve in python/numpy is fast
        b_slope = np.convolve(px, blue_w[::-1], mode='full')[:len(px)]
        
        days_data.append((px, b_slope))
        
    print(f"Loaded {len(days_data)} days for H1 2024 IS Tuning.")

    # Objective function
    def objective(params):
        log10_q, r, entry_vel_thresh, exit_type_float, trailing_pts = params
        q = 10 ** log10_q
        exit_type = int(exit_type_float)
        
        total_net = 0.0
        total_trades = 0
        
        for px, b_slope in days_data:
            net, trds = fast_backtest_day(px, b_slope, 1.0, q, r, entry_vel_thresh, exit_type, trailing_pts)
            total_net += net
            total_trades += trds
            
        # We want to maximize total_net. We minimize -total_net.
        # Add penalty for extreme trade counts (too many or too few)
        # We want at least 1 trade per day (~120 trades minimum)
        if total_trades < 60:
            return 999999.0
            
        return -total_net

    bounds = [
        (-10.0, -4.0),     # log10_q
        (0.1, 20.0),       # r
        (0.0001, 0.1),     # entry_vel_thresh (pts/sec)
        (0.0, 3.99),       # exit_type (0=Vel, 1=Accel, 2=Blue, 3=Trailing)
        (5.0, 100.0)       # trailing_pts (used only if exit_type=3)
    ]
    
    print("Starting Differential Evolution Optimization...")
    t0 = time.time()
    
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=20,     # keep it short for quick turnaround
        popsize=10, 
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        seed=42,
        disp=True,
        workers=1 # single thread to avoid numba multiprocessing issues
    )
    
    print(f"\nOptimization finished in {time.time()-t0:.2f} seconds.")
    print("\n=== BEST IS PARAMETERS ===")
    
    best_log10_q, best_r, best_entry, best_exit_f, best_trail = result.x
    best_exit = int(best_exit_f)
    best_q = 10 ** best_log10_q
    
    exit_names = {0: "Velocity Flip", 1: "Acceleration Flip", 2: "Blue Flip", 3: f"Trailing {best_trail:.1f} pts"}
    
    print(f"Q_JERK:          {best_q:.2e}")
    print(f"R_MEAS:          {best_r:.4f}")
    print(f"Entry Vel:       {best_entry:.5f} pts/sec")
    print(f"Exit Type:       {best_exit} ({exit_names[best_exit]})")
    print(f"Total IS PnL:    ${-result.fun:,.2f}")
    
    # Save parameters for OOS pass
    with open('reports/findings/kalman_ga_best_params.txt', 'w') as f:
        f.write(f"{best_q}\n{best_r}\n{best_entry}\n{best_exit}\n{best_trail}")

if __name__ == '__main__':
    main()
