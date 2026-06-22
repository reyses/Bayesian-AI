"""FIX the garbage-in: re-run the GA-Kalman backtest with a GAP/SESSION GUARD so trades
can't span price discontinuities (the $100 stop is defeated by gaps → fake -$200..-$454 losses).

Guard: if the next bar is a gap (time-gap > GAP_SEC, i.e. a halt/roll seam, OR a single-bar
price jump > GAP_PT, i.e. a thin-liquidity/roll discontinuity), FORCE-FLAT at the last clean
bar before it, and do not enter on gap bars. Everything else = the exact GA config.

Reuses Gemini's Numba Kalman (GA params q=1.81e-9, r=12.55, entry 0.06626, exit trail 79.4, $100 stop).
Output: reports/findings/kalman_clean_trades.csv + reports/findings/kalman_gapguard_compare.md
Run: python research/kalman_backtest_gapguarded.py
"""
import os, glob
import numpy as np
import pandas as pd
import numba

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
Q, R, DT = 1.81e-09, 12.5533, 1.0
ENTRY_VEL, TRAIL, STOP_USD = 0.06626, 79.4, 100.0
GAP_PT, GAP_SEC = 15.0, 300
BLUE_N = 1440; USD = 2.0; COST = 2.5
RNG = np.random.RandomState(42)


@numba.njit
def quad_weights(n):
    w = np.empty(n)
    for i in range(1, n + 1): w[i-1] = (i / n) ** 2
    m = np.mean(w)
    for i in range(n): w[i] -= m
    s = np.sum(np.abs(w))
    for i in range(n): w[i] /= s
    return w


@numba.njit
def kalman_ca(p, dt, q, r):
    n = len(p); vel = np.empty(n)
    x0, x1, x2 = p[0], 0.0, 0.0
    P00,P01,P02 = 1e3,0.,0.; P10,P11,P12 = 0.,1e3,0.; P20,P21,P22 = 0.,0.,1e3
    dt2=dt*dt; dt3=dt2*dt; dt4=dt3*dt; dt5=dt4*dt
    Q00=q*dt5/20; Q01=q*dt4/8; Q02=q*dt3/6; Q11=q*dt3/3; Q12=q*dt2/2; Q22=q*dt
    for i in range(n):
        z=p[i]; xp0=x0+dt*x1+0.5*dt2*x2; xp1=x1+dt*x2; xp2=x2
        FP00=P00+dt*P10+0.5*dt2*P20; FP01=P01+dt*P11+0.5*dt2*P21; FP02=P02+dt*P12+0.5*dt2*P22
        FP10=P10+dt*P20; FP11=P11+dt*P21; FP12=P12+dt*P22; FP20=P20; FP21=P21; FP22=P22
        Pp00=FP00+FP01*dt+FP02*0.5*dt2+Q00; Pp01=FP01+FP02*dt+Q01; Pp02=FP02+Q02
        Pp10=FP10+FP11*dt+FP12*0.5*dt2+Q01; Pp11=FP11+FP12*dt+Q11; Pp12=FP12+Q12
        Pp20=FP20+FP21*dt+FP22*0.5*dt2+Q02; Pp21=FP21+FP22*dt+Q12; Pp22=FP22+Q22
        y=z-xp0; S=Pp00+r; K0=Pp00/S; K1=Pp10/S; K2=Pp20/S
        x0=xp0+K0*y; x1=xp1+K1*y; x2=xp2+K2*y; vel[i]=x1
        P00=(1-K0)*Pp00; P01=(1-K0)*Pp01; P02=(1-K0)*Pp02
        P10=-K1*Pp00+Pp10; P11=-K1*Pp01+Pp11; P12=-K1*Pp02+Pp12
        P20=-K2*Pp00+Pp20; P21=-K2*Pp01+Pp21; P22=-K2*Pp02+Pp22
    return vel


@numba.njit
def backtest(ts, px, vel, b, thr, trail, gap_pt, gap_sec):
    n = len(px); ip = 0; ep = 0.0; ets = 0; mfe = 0.0; mae = 0.0
    MAX = 2000
    o_e = np.zeros(MAX, np.int64); o_x = np.zeros(MAX, np.int64); o_d = np.zeros(MAX, np.int32)
    o_ep = np.zeros(MAX); o_xp = np.zeros(MAX); o_mfe = np.zeros(MAX); o_mae = np.zeros(MAX)
    o_gap = np.zeros(MAX, np.int32)
    k = 0; ngap_close = 0
    for i in range(BLUE_N, n):
        p = px[i]; v = vel[i]
        is_gap = (ts[i] - ts[i-1] > gap_sec) or (abs(px[i] - px[i-1]) > gap_pt)
        if is_gap:
            if ip != 0 and k < MAX:   # force-flat at LAST CLEAN bar before the gap
                pc = px[i-1]
                o_e[k]=ets; o_x[k]=ts[i-1]; o_d[k]=ip; o_ep[k]=ep; o_xp[k]=pc
                o_mfe[k]=mfe; o_mae[k]=mae; o_gap[k]=1; k+=1; ngap_close+=1; ip=0
            continue              # never enter or evaluate on a gap bar
        if ip != 0:
            if ip == 1: mfe=max(mfe,p); mae=min(mae,p)
            else: mfe=min(mfe,p); mae=max(mae,p)
            pp = (p-ep) if ip==1 else (ep-p)
            stop = pp*USD <= -STOP_USD
            trail_hit = (p <= mfe-trail) if ip==1 else (p >= mfe+trail)
            eod = (i == n-1)
            if stop or trail_hit or eod:
                if k < MAX:
                    o_e[k]=ets; o_x[k]=ts[i]; o_d[k]=ip; o_ep[k]=ep; o_xp[k]=p
                    o_mfe[k]=mfe; o_mae[k]=mae; o_gap[k]=0; k+=1
                ip=0
        if ip == 0 and i < n-1:
            if b[i] > 0 and v > thr and vel[i-1] <= thr:
                ip=1; ep=p; ets=ts[i]; mfe=p; mae=p
            elif b[i] < 0 and v < -thr and vel[i-1] >= -thr:
                ip=-1; ep=p; ets=ts[i]; mfe=p; mae=p
    return (o_e[:k], o_x[:k], o_d[:k], o_ep[:k], o_xp[:k], o_mfe[:k], o_mae[:k], o_gap[:k], ngap_close)


def split_of(d):
    if d[:7] in ('2024_01','2024_02','2024_03','2024_04','2024_05','2024_06'): return 'IS'
    if d.startswith('2024'): return 'OOS_H2_24'
    return 'OOS_25_26'


def ci(x):
    if len(x) < 3: return (np.nan, np.nan)
    b = [x[RNG.randint(0,len(x),len(x))].mean() for _ in range(4000)]
    return tuple(np.percentile(b,[2.5,97.5]))


def main():
    files = sorted(glob.glob(f'{ATLAS}/*.parquet'))
    bw = quad_weights(BLUE_N)
    rows = []; tot_gap = 0
    for f in files:
        day = os.path.basename(f)[:-8]
        d = pd.read_parquet(f, columns=['timestamp','close'])
        if len(d) < BLUE_N + 60: continue
        px = d['close'].to_numpy(np.float64); ts = d['timestamp'].to_numpy(np.int64)
        vel = kalman_ca(px, DT, Q, R)
        b = np.convolve(px, bw[::-1], 'full')[:len(px)]
        e,x,dd,ep,xp,mfe,mae,gap,ng = backtest(ts, px, vel, b, ENTRY_VEL, TRAIL, GAP_PT, GAP_SEC)
        tot_gap += ng
        for j in range(len(e)):
            sgn = 1 if dd[j]==1 else -1
            gp = (xp[j]-ep[j])*sgn
            rows.append(dict(day=day, split=split_of(day), entry_ts=int(e[j]), exit_ts=int(x[j]),
                             dir='LONG' if dd[j]==1 else 'SHORT', entry_price=ep[j], exit_price=xp[j],
                             gross_usd=gp*USD, net_usd=gp*USD-COST,
                             mfe_pts=(mfe[j]-ep[j])*sgn, mae_pts=(mae[j]-ep[j])*sgn,
                             gap_close=int(gap[j])))
    tr = pd.DataFrame(rows)
    os.makedirs('reports/findings', exist_ok=True)
    tr.to_csv('reports/findings/kalman_clean_trades.csv', index=False)

    L = ["# Gap-guarded GA-Kalman re-run — clean trades (vs contaminated)\n",
         f"Guard: force-flat before any time-gap>{GAP_SEC}s or price-jump>{GAP_PT}pt; no entry on gap bars.",
         f"Total trades: {len(tr)} | gap-forced closes: {tot_gap}",
         f"Worst single loss now: ${tr['net_usd'].min():.0f} (was -$454 contaminated)\n",
         "| split | trades | net $/tr | PF | net $/day | 95% day-block CI | sig | (contaminated $/day) |",
         "|---|---|---|---|---|---|---|---|"]
    contam = {'IS': '+57.9', 'OOS_H2_24': '+32.0', 'OOS_25_26': '+1.4'}
    for s in ['IS','OOS_H2_24','OOS_25_26']:
        d = tr[tr['split']==s]
        if not len(d): continue
        net = d['net_usd'].to_numpy()
        pf = net[net>0].sum()/abs(net[net<0].sum()) if (net<0).any() else float('inf')
        pday = d.groupby('day')['net_usd'].sum()
        m = pday.mean(); lo,hi = ci(pday.to_numpy())
        sig = 'EXCL 0' if (lo>0 or hi<0) else 'incl 0'
        L.append(f"| {s} | {len(d)} | {net.mean():+.2f} | {pf:.2f} | {m:+.1f} | [{lo:+.1f},{hi:+.1f}] | {sig} | {contam[s]} |")
    rep = "\n".join(L)
    open('reports/findings/kalman_gapguard_compare.md','w',encoding='utf-8').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
