import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import numba

@numba.njit
def create_quadratic_weights(window_size):
    w = np.empty(window_size)
    for i in range(1, window_size + 1):
        w[i-1] = (i / window_size)**2
    mean_w = np.mean(w)
    for i in range(window_size): w[i] -= mean_w
    sum_w = np.sum(np.abs(w))
    for i in range(window_size): w[i] /= sum_w
    return w

@numba.njit
def kalman_ca_numba(p, dt, q, r):
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

@numba.njit
def fast_backtest_trades(ts, px, b_slope, dt, q, r, entry_vel_thresh, exit_type, trailing_pts):
    n = len(px)
    pos, vel, acc = kalman_ca_numba(px, dt, q, r)
    
    in_pos = 0
    eprice = 0.0
    ets = 0
    mfe_px = 0.0
    mae_px = 0.0
    
    # We will pre-allocate arrays to return the trades
    # Assuming max 1000 trades per day
    MAX_TRADES = 1000
    out_ets = np.zeros(MAX_TRADES, dtype=np.int64)
    out_xts = np.zeros(MAX_TRADES, dtype=np.int64)
    out_dir = np.zeros(MAX_TRADES, dtype=np.int32)
    out_epx = np.zeros(MAX_TRADES, dtype=np.float64)
    out_xpx = np.zeros(MAX_TRADES, dtype=np.float64)
    out_mfe = np.zeros(MAX_TRADES, dtype=np.float64)
    out_mae = np.zeros(MAX_TRADES, dtype=np.float64)
    
    t_idx = 0
    
    for i in range(1440, n):
        p = px[i]
        t = ts[i]
        v = vel[i]
        a = acc[i]
        b = b_slope[i]
        
        if i == n - 1 and in_pos != 0:
            if t_idx < MAX_TRADES:
                out_ets[t_idx] = ets
                out_xts[t_idx] = t
                out_dir[t_idx] = in_pos
                out_epx[t_idx] = eprice
                out_xpx[t_idx] = p
                out_mfe[t_idx] = mfe_px
                out_mae[t_idx] = mae_px
                t_idx += 1
            in_pos = 0
            continue
            
        if in_pos != 0:
            if in_pos == 1:
                mfe_px = max(mfe_px, p)
                mae_px = min(mae_px, p)
            else:
                mfe_px = min(mfe_px, p)
                mae_px = max(mae_px, p)
                
            pnl_pts = (p - eprice) if in_pos == 1 else (eprice - p)
            pnl_usd = pnl_pts * 2.0
            
            # $100 stop loss (MNQ = $2/pt = 50 pts)
            hit_stop = pnl_usd <= -100.0
            
            exit_signal = False
            if exit_type == 0:
                exit_signal = (in_pos == 1 and v < 0) or (in_pos == -1 and v > 0)
            elif exit_type == 1:
                exit_signal = (in_pos == 1 and a < 0) or (in_pos == -1 and a > 0)
            elif exit_type == 2:
                exit_signal = (in_pos == 1 and b < 0) or (in_pos == -1 and b > 0)
            elif exit_type == 3:
                if in_pos == 1:
                    exit_signal = p <= mfe_px - trailing_pts
                else:
                    exit_signal = p >= mfe_px + trailing_pts
                    
            if hit_stop or exit_signal:
                if t_idx < MAX_TRADES:
                    out_ets[t_idx] = ets
                    out_xts[t_idx] = t
                    out_dir[t_idx] = in_pos
                    out_epx[t_idx] = eprice
                    out_xpx[t_idx] = p
                    out_mfe[t_idx] = mfe_px
                    out_mae[t_idx] = mae_px
                    t_idx += 1
                in_pos = 0
                
        if in_pos == 0:
            if b > 0 and v > entry_vel_thresh and vel[i-1] <= entry_vel_thresh:
                in_pos = 1
                eprice = p
                ets = t
                mfe_px = p
                mae_px = p
            elif b < 0 and v < -entry_vel_thresh and vel[i-1] >= -entry_vel_thresh:
                in_pos = -1
                eprice = p
                ets = t
                mfe_px = p
                mae_px = p
                
    return out_ets[:t_idx], out_xts[:t_idx], out_dir[:t_idx], out_epx[:t_idx], out_xpx[:t_idx], out_mfe[:t_idx], out_mae[:t_idx]

def main():
    ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
    files = sorted(glob.glob(f'{ATLAS}/*.parquet'))
    
    # Genetic Tuned Params
    q = 1.81e-09
    r = 12.5533
    entry_vel = 0.06626
    exit_type = 3
    trailing_pts = 79.4
    
    print(f"Running Full Dataset Evaluation (N={len(files)} days)...")
    blue_w = create_quadratic_weights(1440)
    
    all_trades = []
    
    t0 = time.time()
    for f in files:
        day = os.path.basename(f).replace('.parquet', '')
        # Determine Dataset Split
        if day.startswith('2024_01') or day.startswith('2024_02') or day.startswith('2024_03') or day.startswith('2024_04') or day.startswith('2024_05') or day.startswith('2024_06'):
            split = 'IS'
        elif day.startswith('2024'):
            split = 'OOS_H2_24'
        else:
            split = 'OOS_25_26'
            
        df = pd.read_parquet(f, columns=['timestamp', 'close'])
        if len(df) < 1440:
            continue
            
        px = df['close'].to_numpy(np.float64)
        ts = df['timestamp'].to_numpy(np.int64)
        
        b_slope = np.convolve(px, blue_w[::-1], mode='full')[:len(px)]
        
        ets, xts, ddir, epx, xpx, mfe, mae = fast_backtest_trades(ts, px, b_slope, 1.0, q, r, entry_vel, exit_type, trailing_pts)
        
        for i in range(len(ets)):
            gross_pts = (xpx[i] - epx[i]) if ddir[i] == 1 else (epx[i] - xpx[i])
            mfe_pts = (mfe[i] - epx[i]) if ddir[i] == 1 else (epx[i] - mfe[i])
            mae_pts = (mae[i] - epx[i]) if ddir[i] == 1 else (epx[i] - mae[i])
            
            pnl_usd = gross_pts * 2.0
            net_usd = pnl_usd - 2.5
            
            all_trades.append({
                'day': day,
                'split': split,
                'entry_ts': ets[i],
                'exit_ts': xts[i],
                'dir': 'LONG' if ddir[i] == 1 else 'SHORT',
                'entry_price': epx[i],
                'exit_price': xpx[i],
                'gross_usd': pnl_usd,
                'net_usd': net_usd,
                'mfe_pts': mfe_pts,
                'mae_pts': mae_pts
            })
            
    print(f"Finished in {time.time()-t0:.2f} seconds.")
    
    if len(all_trades) == 0:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    
    os.makedirs('reports/findings', exist_ok=True)
    df_trades.to_csv('reports/findings/kalman_full_trades.csv', index=False)
    
    # Calculate Metrics per Split
    print("\n" + "="*50)
    print("KALMAN STRATEGY PERFORMANCE")
    print("="*50)
    for split in ['IS', 'OOS_H2_24', 'OOS_25_26']:
        s_df = df_trades[df_trades['split'] == split]
        if len(s_df) == 0:
            continue
        net = s_df['net_usd'].sum()
        gross_prof = s_df[s_df['gross_usd'] > 0]['gross_usd'].sum()
        gross_loss = abs(s_df[s_df['gross_usd'] < 0]['gross_usd'].sum())
        pf = gross_prof / gross_loss if gross_loss > 0 else np.inf
        wr = pf - 1
        
        print(f"\n[{split}]")
        print(f"Trades:    {len(s_df)}")
        print(f"Net PnL:   ${net:,.2f}")
        print(f"WR (PF-1): {wr:.3f}")
        print(f"Avg MFE:   {s_df['mfe_pts'].mean():.2f} pts")

if __name__ == '__main__':
    main()
