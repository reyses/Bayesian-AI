"""Genuine 1s features vs TF-mix, same day/grid. Thesis: TF-mix = collinear half-measurements."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('.'))
from core_v2.statistical_field_engine import StatisticalFieldEngine
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

DAY='2025_02_20'; WINS=[60,300,1800]; FWD=300
b=pd.read_parquet(f'DATA/ATLAS/1s/{DAY}.parquet').sort_values('timestamp').reset_index(drop=True)
ts=b['timestamp'].values
sfe=StatisticalFieldEngine()
def ix(df):
    df=df.reset_index(drop=True).copy(); df.index=ts[:len(df)]; return df

# Arm A: genuine 1s features
partsA=[ix(sfe.compute_L1(b, tf='1s'))]
for N in WINS:
    partsA.append(ix(sfe.compute_L2(b, tf='1s', N=N)))
    partsA.append(ix(sfe.compute_L3(b, tf='1s', N=N)))
A=pd.concat(partsA,axis=1); A=A[[c for c in A.columns if 'vwap' not in c]]
os.makedirs('DATA/ATLAS/FEATURES_1s_TRUE',exist_ok=True)
A.reset_index(names='timestamp').to_parquet(f'DATA/ATLAS/FEATURES_1s_TRUE/{DAY}.parquet')
print(f"Arm A (genuine 1s) saved: {A.shape[1]} feats, {A.shape[0]} rows -> DATA/ATLAS/FEATURES_1s_TRUE/{DAY}.parquet")

# Arm B: TF-mix (FEATURES_1s_v2 L1/L2/L3 x 8 TFs)
base='DATA/ATLAS/FEATURES_1s_v2'
partsB=[]
for fam in [d for d in os.listdir(base) if d[:2] in ('L1','L2','L3')]:
    f=f'{base}/{fam}/{DAY}.parquet'
    if os.path.exists(f): partsB.append(pd.read_parquet(f).set_index('timestamp'))
B=pd.concat(partsB,axis=1); B=B[[c for c in B.columns if 'vwap' not in c]]
print(f"Arm B (TF-mix): {B.shape[1]} feats")

px=b.set_index('timestamp')['close']; y=(px.shift(-FWD)-px)
def ev(X,name):
    df=X.join(y.rename('y')).replace([np.inf,-np.inf],np.nan).dropna()
    Xs=StandardScaler().fit_transform(df.drop(columns='y').values); yv=df['y'].values
    cond=np.linalg.cond(Xs)
    r2=cross_val_score(RidgeCV(alphas=[.1,1,10,100]),Xs,yv,cv=5,scoring='r2').mean()
    h=len(Xs)//2
    b1=LinearRegression().fit(Xs[:h],yv[:h]).coef_; b2=LinearRegression().fit(Xs[h:],yv[h:]).coef_
    bs=np.corrcoef(b1,b2)[0,1]; sf=(np.sign(b1)!=np.sign(b2)).mean()*100
    print(f"\n[{name}] feats={Xs.shape[1]} n={len(Xs)}")
    print(f"  collinearity cond#: {cond:,.0f}")
    print(f"  fit R2 (Ridge 5cv): {r2:+.4f}")
    print(f"  beta stab (halves corr): {bs:+.3f}  sign-flips: {sf:.0f}%")
    return cond,r2,bs,sf
a=ev(A,'A genuine-1s'); c=ev(B,'B TF-mix')
print("\n=== VERDICT (genuine-1s vs TF-mix) ===")
print(f"cond#:     {a[0]:,.0f} vs {c[0]:,.0f}  ({'A cleaner' if a[0]<c[0] else 'B cleaner'})")
print(f"R2:        {a[1]:+.4f} vs {c[1]:+.4f}")
print(f"beta stab: {a[2]:+.3f} vs {c[2]:+.3f}  | sign-flips {a[3]:.0f}% vs {c[3]:.0f}%")
