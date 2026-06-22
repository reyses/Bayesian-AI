"""ONE-OFF CANDIDATE: Kalman constant-acceleration (CA) filter vs the 7.5m cubic.

Goal: smooth the orange line WITHOUT clipping the top. The cubic read at its endpoint
is high-variance (~378 turns/day). A Kalman CA filter gives an OPTIMAL CAUSAL estimate
of position/velocity/acceleration in one model, adapts its gain (lags LESS through real
moves), and unifies the orange curve (position) + pink kinematics (velocity/accel).

CAUSAL: forward filter only (NOT the RTS smoother, which peeks at the future).
The q/r ratio is the smooth-vs-lag dial — TUNE IT (see handover).

Run: python research/orange_kalman_candidate.py [YYYY_MM_DD]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS'
ONE_S = f'{ATLAS}/1s'

# ── Kalman CA tuning (THE smooth/lag dial — tune these) ──
Q_JERK = 1e-4      # process noise (jerk PSD): higher = more responsive, less smooth
R_MEAS = 1.0       # measurement noise variance (price jitter, pts^2)
DT = 1.0           # 1-second steps

# ── cubic (for the comparison overlay) ──
CW, CDEG = 450, 3
CX = np.arange(CW) / 60.0
CXE = CX[-1]


def cubic_lines(p):
    P = np.linalg.pinv(np.vander(CX, CDEG + 1))
    val_w = CXE**3 * P[0] + CXE**2 * P[1] + CXE * P[2] + P[3]
    slp_w = 3 * CXE**2 * P[0] + 2 * CXE * P[1] + P[2]
    out_v = np.full(len(p), np.nan); out_s = np.full(len(p), np.nan)
    if len(p) > CW:
        out_v[CW:] = np.convolve(p, val_w[::-1], 'valid')[:-1]
        out_s[CW:] = np.convolve(p, slp_w[::-1], 'valid')[:-1]
    return out_v, out_s


def kalman_ca(p, dt=DT, q=Q_JERK, r=R_MEAS):
    """Constant-acceleration Kalman filter. Returns pos, vel, acc arrays (causal)."""
    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    H = np.array([[1.0, 0, 0]])
    # continuous white-noise-jerk Q
    Q = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                      [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
                      [dt**3 / 6,  dt**2 / 2, dt]])
    x = np.array([p[0], 0.0, 0.0]); P = np.eye(3) * 1e3
    pos = np.empty(len(p)); vel = np.empty(len(p)); acc = np.empty(len(p))
    for i, z in enumerate(p):
        x = F @ x; P = F @ P @ F.T + Q                 # predict
        y = z - (H @ x)[0]; S = (H @ P @ H.T)[0, 0] + r
        K = (P @ H.T / S).ravel()                      # gain
        x = x + K * y; P = (np.eye(3) - np.outer(K, H)) @ P   # update
        pos[i], vel[i], acc[i] = x
    return pos, vel, acc


def turns(x):
    s = np.sign(np.nan_to_num(x))
    return int(((s[:-1] != 0) & (s[1:] != 0) & (s[:-1] != s[1:])).sum())


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else '2024_03_18'
    f = f'{ONE_S}/{day}.parquet'
    df = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
    p = df['close'].to_numpy(np.float64)
    t = (df['timestamp'].to_numpy(np.int64) - df['timestamp'].iloc[0]) / 3600.0

    kp, kv, ka = kalman_ca(p)
    cv, cs = cubic_lines(p)

    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1.2, 1.2]})
    ax[0].plot(t, p, color='#bdbdbd', lw=0.4, label='price (1s)')
    ax[0].plot(t, cv, color='#e8730c', lw=1.2, alpha=0.8, label='cubic value (orange)')
    ax[0].plot(t, kp, color='#1565c0', lw=1.6, label='Kalman position')
    ax[0].set_ylabel('price'); ax[0].legend(loc='upper left', fontsize=8)
    ax[0].set_title(f"Kalman CA vs cubic — {day}  |  velocity turns: cubic {turns(cs)} vs Kalman {turns(kv)} "
                    f"(q={Q_JERK}, r={R_MEAS})", fontsize=11, loc='left')
    ax[1].plot(t, cs, color='#e8730c', lw=0.8, alpha=0.7, label='cubic slope')
    ax[1].plot(t, kv, color='#1565c0', lw=1.0, label='Kalman velocity'); ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_ylabel('velocity'); ax[1].legend(fontsize=7)
    ax[2].plot(t, ka, color='#6a1b9a', lw=1.0, label='Kalman acceleration'); ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set_ylabel('acceleration'); ax[2].set_xlabel('hours from session start'); ax[2].legend(fontsize=7)

    out = f'reports/findings/orange_kalman_vs_cubic_{day}.png'
    os.makedirs('reports/findings', exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f"cubic velocity turns: {turns(cs)}  |  Kalman velocity turns: {turns(kv)}")
    print(f"saved {out}")


if __name__ == '__main__':
    main()
