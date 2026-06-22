import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# The CA Kalman from numba_kalman
def kalman_ca(p, dt=1.0, q=1e-4, r=1.0):
    n = len(p)
    pos = np.empty(n)
    vel = np.empty(n)
    acc = np.empty(n)
    
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
        
        FP10 = P10 + dt * P20
        FP11 = P11 + dt * P21
        FP12 = P12 + dt * P22
        
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

def eval_sweep(path, q_val):
    p = np.array(path, dtype=np.float64)
    n = len(p)
    if n < 60:
        return {'Base': p[-1], 'AccelFlip': p[-1], 'VelDecay': p[-1]}
        
    mfe = np.max(p)
    mae = np.min(p)
    
    pos, vel, acc = kalman_ca(p, q=q_val)
    
    # Candidate 1: Accel Flip (accel drops < 0)
    accel_flip_idx = n - 1
    for i in range(60, n):
        if acc[i] < 0:
            accel_flip_idx = i
            break
            
    # Candidate 2: Vel Decay (velocity drops < 50% of max so far)
    vel_decay_idx = n - 1
    max_v = 0
    for i in range(60, n):
        if vel[i] > max_v:
            max_v = vel[i]
        if max_v > 0.01 and vel[i] < 0.5 * max_v:
            vel_decay_idx = i
            break

    return {
        'Base': p[-1],
        'AccelFlip': p[accel_flip_idx],
        'VelDecay': p[vel_decay_idx],
        'MFE': mfe,
        'MAE': mae
    }

def main():
    print("Loading trade paths...")
    df = pd.read_parquet('C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/reports/findings/trade_paths.parquet')
    
    q_vals = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    metrics = []
    
    for q in q_vals:
        print(f"Sweeping q={q}...")
        res = []
        for i, row in df.iterrows():
            out = eval_sweep(row['path'], q)
            res.append(out)
            
        res_df = pd.DataFrame(res)
        
        for c in ['AccelFlip', 'VelDecay']:
            pnl_pts = res_df[c].sum()
            pnl_usd = pnl_pts * 2.0 - (len(res_df) * 1.56)
            wr = (res_df[c] > 0).mean() * 100
            avg_pts = res_df[c].mean()
            
            winners = res_df[res_df['MFE'] > 20]
            capture = (winners[c] / winners['MFE']).mean() * 100
            
            metrics.append({'Q': q, 'Exit': c, 'PnL_Pts': pnl_pts, 'PnL_USD': pnl_usd, 'WR': wr, 'Avg_Pts': avg_pts, 'MFE_Capture_Pct': capture})
            
    # Also add base
    base_pts = res_df['Base'].sum()
    base_usd = base_pts * 2.0 - (len(res_df) * 1.56)
    base_wr = (res_df['Base'] > 0).mean() * 100
    base_avg = res_df['Base'].mean()
    winners = res_df[res_df['MFE'] > 20]
    base_cap = (winners['Base'] / winners['MFE']).mean() * 100
    metrics.append({'Q': 'N/A', 'Exit': 'Base', 'PnL_Pts': base_pts, 'PnL_USD': base_usd, 'WR': base_wr, 'Avg_Pts': base_avg, 'MFE_Capture_Pct': base_cap})

    with open('reports/findings/geometric_exit_results.md', 'w') as f:
        f.write("# Geometric Exit Candidates - Q Parameter Sweep\n\n")
        f.write("| Q Value | Exit | PnL (Pts) | PnL (USD) | Win Rate | Avg Pts | MFE Capture |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m in metrics:
            f.write(f"| {m['Q']} | {m['Exit']} | {m['PnL_Pts']:.1f} | ${m['PnL_USD']:.2f} | {m['WR']:.1f}% | {m['Avg_Pts']:.2f} | {m['MFE_Capture_Pct']:.1f}% |\n")
    print("Saved results to reports/findings/geometric_exit_results.md")
    
    # Animate the best trade for the best exit
    # Let's find a trade with a massive MFE where AccelFlip did well
    best_idx_series = res_df.loc[(res_df['MFE'] > 100) & (res_df['AccelFlip'] > 80), 'idx']
    if len(best_idx_series) > 0:
        best_idx = int(best_idx_series.values[0])
    else:
        best_idx = 0
        
    best_path = df.iloc[best_idx]['path']
    
    print(f"\nAnimating trade {best_idx} (MFE: {np.max(best_path):.1f})")
    pos, vel, acc = kalman_ca(best_path, q=1e-4)
    
    ex_base = int(res_df.iloc[best_idx]['idx_Base'])
    ex_accel = int(res_df.iloc[best_idx]['idx_AccelFlip'])
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Draw static lines
    ax[0].plot(best_path, color='#bdbdbd', label='Price (Normalized)')
    ax[0].plot(pos, color='#1565c0', label='Kalman Pos')
    ax[0].axvline(ex_accel, color='red', linestyle='--', label='Accel Flip Exit')
    ax[0].legend()
    ax[0].set_title(f"Trade {best_idx} - Accel Flip captures {best_path[ex_accel]:.1f} pts (MFE: {np.max(best_path):.1f})")
    
    ax[1].plot(vel, color='#e8730c', label='Velocity')
    ax[1].axhline(0, color='black', lw=0.5)
    ax[1].axvline(ex_accel, color='red', linestyle='--')
    ax[1].legend()
    
    ax[2].plot(acc, color='#6a1b9a', label='Acceleration')
    ax[2].axhline(0, color='black', lw=0.5)
    ax[2].axvline(ex_accel, color='red', linestyle='--')
    ax[2].legend()
    
    os.makedirs('reports/findings', exist_ok=True)
    fig.savefig('reports/findings/geometric_exit_demo.png')
    print("Saved geometric_exit_demo.png")

if __name__ == '__main__':
    main()
