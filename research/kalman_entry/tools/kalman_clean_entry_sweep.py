"""#2 (clean): entry-timing × exit sweep on GAP-GUARDED trades, reported at TRADE LEVEL
(no day stats — per the decision to distrust day aggregation). Does earlier entry + a good
exit produce a positive net $/trade on CLEAN data?

Gap guard: force-flat before time-gap>300s or price-jump>15pt; no entry on gap bars.
Metrics per (entry_thr, exit, split): trades, net $/tr mean+median, PF-based WR, trade
bootstrap CI on mean net $/tr. CAVEAT: trade-level CI overstates significance (within-day
correlation) — read conservatively.
Output: reports/findings/kalman_clean_entry_sweep.md
"""
import os, glob
import numpy as np
import pandas as pd
import numba

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
Q, R, DT = 1.81e-09, 12.5533, 1.0
GAP_PT, GAP_SEC, BLUE_N, USD, COST = 15.0, 300, 1440, 2.0, 2.5
ENTRY_SWEEP = [0.02, 0.033, 0.06626, 0.10]
EXITS = [('vel_flip', 0, 0.0), ('accel_flip', 1, 0.0), ('trail_30', 3, 30.0), ('trail_79_GA', 3, 79.4)]
RNG = np.random.RandomState(42)


@numba.njit
def quad_weights(n):
    w = np.empty(n)
    for i in range(1, n+1): w[i-1] = (i/n)**2
    m = np.mean(w)
    for i in range(n): w[i] -= m
    s = np.sum(np.abs(w))
    for i in range(n): w[i] /= s
    return w


@numba.njit
def kalman_ca(p, dt, q, r):
    n = len(p); vel = np.empty(n); acc = np.empty(n)
    x0,x1,x2 = p[0],0.,0.; P00,P01,P02=1e3,0.,0.; P10,P11,P12=0.,1e3,0.; P20,P21,P22=0.,0.,1e3
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
        x0=xp0+K0*y; x1=xp1+K1*y; x2=xp2+K2*y; vel[i]=x1; acc[i]=x2
        P00=(1-K0)*Pp00; P01=(1-K0)*Pp01; P02=(1-K0)*Pp02
        P10=-K1*Pp00+Pp10; P11=-K1*Pp01+Pp11; P12=-K1*Pp02+Pp12
        P20=-K2*Pp00+Pp20; P21=-K2*Pp01+Pp21; P22=-K2*Pp02+Pp22
    return vel, acc


@numba.njit
def bt(ts, px, vel, acc, b, thr, etype, trail, gap_pt, gap_sec):
    n=len(px); ip=0; ep=0.; mfe=0.; out=np.zeros(3000); k=0
    for i in range(BLUE_N, n):
        p=px[i]; v=vel[i]; a=acc[i]
        is_gap = (ts[i]-ts[i-1] > gap_sec) or (abs(px[i]-px[i-1]) > gap_pt)
        if is_gap:
            if ip!=0 and k<3000:
                out[k] = ((px[i-1]-ep) if ip==1 else (ep-px[i-1])); k+=1; ip=0
            continue
        if ip!=0:
            if ip==1: mfe=max(mfe,p)
            else: mfe=min(mfe,p)
            pp=(p-ep) if ip==1 else (ep-p)
            stop = pp*USD <= -100.0
            ex=False
            if etype==0: ex=(ip==1 and v<0) or (ip==-1 and v>0)
            elif etype==1: ex=(ip==1 and a<0) or (ip==-1 and a>0)
            elif etype==3: ex=(p<=mfe-trail) if ip==1 else (p>=mfe+trail)
            if stop or ex or i==n-1:
                if k<3000: out[k]=pp; k+=1
                ip=0
        if ip==0 and i<n-1:
            if b[i]>0 and v>thr and vel[i-1]<=thr: ip=1; ep=p; mfe=p
            elif b[i]<0 and v<-thr and vel[i-1]>=-thr: ip=-1; ep=p; mfe=p
    return out[:k]


def split_of(d):
    if d[:7] in ('2024_01','2024_02','2024_03','2024_04','2024_05','2024_06'): return 'IS'
    if d.startswith('2024'): return 'OOS_H2_24'
    return 'OOS_25_26'


def ci(x):
    if len(x) < 10: return (np.nan, np.nan)
    b = [x[RNG.randint(0,len(x),len(x))].mean() for _ in range(4000)]
    return tuple(np.percentile(b,[2.5,97.5]))


def main():
    files = sorted(glob.glob(f'{ATLAS}/*.parquet'))
    bw = quad_weights(BLUE_N)
    cache = []
    for f in files:
        day = os.path.basename(f)[:-8]
        d = pd.read_parquet(f, columns=['timestamp','close'])
        if len(d) < BLUE_N+60: continue
        px=d['close'].to_numpy(np.float64); ts=d['timestamp'].to_numpy(np.int64)
        vel,acc = kalman_ca(px,DT,Q,R); b=np.convolve(px,bw[::-1],'full')[:len(px)]
        cache.append((split_of(day), ts, px, vel, acc, b))
    print(f"cached {len(cache)} days")

    L = ["# Clean (gap-guarded) entry×exit sweep — TRADE LEVEL (no day stats)\n",
         "net $/tr = pnl_pts*$2 - $2.5 cost. CAVEAT: trade-level CI overstates significance (within-day corr).\n"]
    for sp in ['OOS_25_26','OOS_H2_24','IS']:
        L += [f"## {sp}", "| entry | exit | trades | net $/tr mean | median | PF | trade-CI on mean |",
              "|---|---|---|---|---|---|---|"]
        res = []
        for thr in ENTRY_SWEEP:
            for en, et, tr in EXITS:
                pts = []
                for (s, ts, px, vel, acc, b) in cache:
                    if s != sp: continue
                    pts.append(bt(ts, px, vel, acc, b, thr, et, tr, GAP_PT, GAP_SEC))
                p = np.concatenate(pts) if pts else np.array([])
                net = p*USD - COST
                if len(net)==0: continue
                pf = net[net>0].sum()/abs(net[net<0].sum()) if (net<0).any() else float('inf')
                lo,hi = ci(net)
                res.append((net.mean(), thr, en, len(net), net.mean(), np.median(net), pf, lo, hi))
        for _,thr,en,n,mn,md,pf,lo,hi in sorted(res, reverse=True):
            sig = 'excl0' if (lo>0 or hi<0) else 'incl0'
            L.append(f"| {thr} | {en} | {n} | {mn:+.2f} | {md:+.2f} | {pf:.2f} | [{lo:+.2f},{hi:+.2f}] {sig} |")
        L.append("")
    rep="\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/kalman_clean_entry_sweep.md','w',encoding='utf-8').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
