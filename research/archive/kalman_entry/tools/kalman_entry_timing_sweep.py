"""Leading-entry experiment: does entering EARLIER (lower velocity threshold) + a GOOD
exit beat the GA's confirmatory entry (0.066 pts/sec) + 79pt dumb trail, OOS?

Regret v2 showed the entry is RIGHT-direction but LATE (~46pt of move already gone at
entry). This sweeps the entry threshold (earlier = lower) across several causal exits, and
measures net OOS $/day with a day-block CI. False starts are penalized naturally (earlier
entry → more chop trades → more cost/reversals → it shows in net $/day).

# FUNDING-STAGE NOTE (comment, not a rule): at this stage, entering half-way through a move
# is acceptable IF the exit is clean — so we evaluate the ENTRY+EXIT COMBO's net $/day, not
# entry-earliness in isolation, and we test several exits (not the broken 79pt trail).

Reuses Gemini's Numba Kalman (research/kalman_tuning_eda/nmp_kalman_oos_validation.py).
Output: reports/findings/kalman_entry_timing_sweep.md
"""
import os, glob
import numpy as np
import pandas as pd
import numba

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
Q, R, DT = 1.81e-09, 12.5533, 1.0      # GA-tuned Kalman params (kept fixed)
BLUE_W_N = 1440
USD_PER_PT = 2.0
COST = 2.5                              # $/trade RT (Gemini's assumption, kept for comparability)
ENTRY_SWEEP = [0.02, 0.033, 0.05, 0.06626, 0.10]   # lower = EARLIER/leading; 0.06626 = GA
EXITS = [('vel_flip', 0, 0.0), ('accel_flip', 1, 0.0), ('trail_30', 3, 30.0), ('trail_79_GA', 3, 79.4)]
RNG = np.random.RandomState(42)


@numba.njit
def quad_weights(n):
    w = np.empty(n)
    for i in range(1, n + 1):
        w[i - 1] = (i / n) ** 2
    m = np.mean(w)
    for i in range(n):
        w[i] -= m
    s = np.sum(np.abs(w))
    for i in range(n):
        w[i] /= s
    return w


@numba.njit
def kalman_ca(p, dt, q, r):  # Gemini's filter (copied for self-containment)
    n = len(p); vel = np.empty(n); acc = np.empty(n)
    x0, x1, x2 = p[0], 0.0, 0.0
    P00, P01, P02 = 1e3, 0.0, 0.0; P10, P11, P12 = 0.0, 1e3, 0.0; P20, P21, P22 = 0.0, 0.0, 1e3
    dt2 = dt*dt; dt3 = dt2*dt; dt4 = dt3*dt; dt5 = dt4*dt
    Q00=q*dt5/20; Q01=q*dt4/8; Q02=q*dt3/6; Q11=q*dt3/3; Q12=q*dt2/2; Q22=q*dt
    for i in range(n):
        z = p[i]
        xp0 = x0+dt*x1+0.5*dt2*x2; xp1 = x1+dt*x2; xp2 = x2
        FP00=P00+dt*P10+0.5*dt2*P20; FP01=P01+dt*P11+0.5*dt2*P21; FP02=P02+dt*P12+0.5*dt2*P22
        FP10=P10+dt*P20; FP11=P11+dt*P21; FP12=P12+dt*P22; FP20=P20; FP21=P21; FP22=P22
        Pp00=FP00+FP01*dt+FP02*0.5*dt2+Q00; Pp01=FP01+FP02*dt+Q01; Pp02=FP02+Q02
        Pp10=FP10+FP11*dt+FP12*0.5*dt2+Q01; Pp11=FP11+FP12*dt+Q11; Pp12=FP12+Q12
        Pp20=FP20+FP21*dt+FP22*0.5*dt2+Q02; Pp21=FP21+FP22*dt+Q12; Pp22=FP22+Q22
        y = z-xp0; S = Pp00+r; K0=Pp00/S; K1=Pp10/S; K2=Pp20/S
        x0=xp0+K0*y; x1=xp1+K1*y; x2=xp2+K2*y
        vel[i]=x1; acc[i]=x2
        P00=(1-K0)*Pp00; P01=(1-K0)*Pp01; P02=(1-K0)*Pp02
        P10=-K1*Pp00+Pp10; P11=-K1*Pp01+Pp11; P12=-K1*Pp02+Pp12
        P20=-K2*Pp00+Pp20; P21=-K2*Pp01+Pp21; P22=-K2*Pp02+Pp22
    return vel, acc


@numba.njit
def run(ts, px, vel, acc, b, thr, exit_type, trail):
    n = len(px); ip = 0; ep = 0.0; mfe = 0.0
    pnl = 0.0  # day net in points
    for i in range(BLUE_W_N, n):
        p = px[i]; v = vel[i]; a = acc[i]
        if ip != 0:
            if ip == 1: mfe = max(mfe, p)
            else: mfe = min(mfe, p)
            pp = (p-ep) if ip == 1 else (ep-p)
            hit_stop = pp*USD_PER_PT <= -100.0
            ex = False
            if exit_type == 0: ex = (ip==1 and v<0) or (ip==-1 and v>0)
            elif exit_type == 1: ex = (ip==1 and a<0) or (ip==-1 and a>0)
            elif exit_type == 3: ex = (p <= mfe-trail) if ip==1 else (p >= mfe+trail)
            eod = (i == n-1)
            if hit_stop or ex or eod:
                pnl += pp; ip = 0
        if ip == 0 and i < n-1:
            if b[i] > 0 and v > thr and vel[i-1] <= thr:
                ip = 1; ep = p; mfe = p
            elif b[i] < 0 and v < -thr and vel[i-1] >= -thr:
                ip = -1; ep = p; mfe = p
    return pnl   # day net, points  (count via separate? keep simple: points only)


@numba.njit
def run_count(ts, px, vel, acc, b, thr, exit_type, trail):
    n = len(px); ip = 0; ep = 0.0; mfe = 0.0; pnl = 0.0; ntr = 0
    for i in range(BLUE_W_N, n):
        p = px[i]; v = vel[i]; a = acc[i]
        if ip != 0:
            if ip == 1: mfe = max(mfe, p)
            else: mfe = min(mfe, p)
            pp = (p-ep) if ip == 1 else (ep-p)
            hit_stop = pp*USD_PER_PT <= -100.0
            ex = False
            if exit_type == 0: ex = (ip==1 and v<0) or (ip==-1 and v>0)
            elif exit_type == 1: ex = (ip==1 and a<0) or (ip==-1 and a>0)
            elif exit_type == 3: ex = (p <= mfe-trail) if ip==1 else (p >= mfe+trail)
            if hit_stop or ex or (i == n-1):
                pnl += pp; ntr += 1; ip = 0
        if ip == 0 and i < n-1:
            if b[i] > 0 and v > thr and vel[i-1] <= thr:
                ip = 1; ep = p; mfe = p
            elif b[i] < 0 and v < -thr and vel[i-1] >= -thr:
                ip = -1; ep = p; mfe = p
    return pnl, ntr


def split_of(day):
    if day[:7] in ('2024_01','2024_02','2024_03','2024_04','2024_05','2024_06'): return 'IS'
    if day.startswith('2024'): return 'OOS_H2_24'
    return 'OOS_25_26'


def ci(x):
    if len(x) < 3: return (np.nan, np.nan)
    b = [x[RNG.randint(0,len(x),len(x))].mean() for _ in range(4000)]
    return tuple(np.percentile(b,[2.5,97.5]))


def main():
    files = sorted(glob.glob(f'{ATLAS}/*.parquet'))
    bw = quad_weights(BLUE_W_N)
    # per-day cache of (vel, acc, b, ts, px, split)
    days = []
    for f in files:
        day = os.path.basename(f)[:-8]
        d = pd.read_parquet(f, columns=['timestamp','close'])
        if len(d) < BLUE_W_N + 60: continue
        px = d['close'].to_numpy(np.float64); ts = d['timestamp'].to_numpy(np.int64)
        vel, acc = kalman_ca(px, DT, Q, R)
        b = np.convolve(px, bw[::-1], 'full')[:len(px)]
        days.append((day, split_of(day), ts, px, vel, acc, b))
    print(f"cached {len(days)} days")

    rows = []
    for thr in ENTRY_SWEEP:
        for ename, etype, trail in EXITS:
            per_day = {}; ntot = 0
            for (day, sp, ts, px, vel, acc, b) in days:
                pnl, ntr = run_count(ts, px, vel, acc, b, thr, etype, trail)
                per_day[(day, sp)] = pnl * USD_PER_PT - ntr * COST
                ntot += ntr
            for sp in ['IS','OOS_H2_24','OOS_25_26']:
                vals = np.array([v for (d_,s_),v in per_day.items() if s_==sp])
                nd = len(vals)
                if nd == 0: continue
                m = vals.mean(); lo,hi = ci(vals)
                rows.append(dict(thr=thr, exit=ename, split=sp, ndays=nd, mday=m, lo=lo, hi=hi,
                                 wd=int((vals>0).sum())))
    df = pd.DataFrame(rows)
    L = ["# Leading-entry sweep — earlier entry × good exit, net $/day OOS (MNQ $2/pt)\n",
         "Entry threshold (pts/sec): lower = EARLIER. 0.06626 = GA baseline. Exits: vel/accel flip, trail.",
         "FUNDING-STAGE NOTE: half-way entry is fine if the exit is clean — we judge the COMBO's net $/day.\n"]
    for sp in ['OOS_25_26','OOS_H2_24','IS']:
        L += [f"## {sp}", "| entry thr | exit | $/day | 95% CI | sig | win-days |", "|---|---|---|---|---|---|"]
        s = df[df['split']==sp].sort_values('mday', ascending=False)
        for _,r in s.iterrows():
            sig = 'EXCL 0' if (r['lo']>0 or r['hi']<0) else 'incl 0'
            L.append(f"| {r['thr']} | {r['exit']} | {r['mday']:+.1f} | [{r['lo']:+.1f},{r['hi']:+.1f}] | {sig} | {r['wd']}/{r['ndays']} |")
        L.append("")
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/kalman_entry_timing_sweep.md','w',encoding='utf-8').write(rep)
    print(rep.encode('ascii','replace').decode())


if __name__ == '__main__':
    main()
