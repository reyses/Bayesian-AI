import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.abspath(os.path.join(HERE, '..', '..', 'nt8_catalog', 'reports'))
OUT = os.path.abspath(os.path.join(HERE, '..', 'reports'))

BASE = ['pivot_age_min', 'sig_with_leg', 'tod', 'inter']
CONSENSUS_S = 180

def load_pool():
    frames = []
    for f in sorted(glob.glob(os.path.join(REP, 'signal_rows_*.parquet'))):
        det = os.path.basename(f)[12:-8]
        df = pd.read_parquet(f)
        df['det'] = det
        frames.append(df)
    P = pd.concat(frames, ignore_index=True).sort_values('ts').reset_index(drop=True)
    
    ts = P['ts'].values.astype(np.int64)
    lng = P['is_long'].values.astype(bool)
    lo = np.searchsorted(ts, ts - CONSENSUS_S, 'left')
    hi = np.searchsorted(ts, ts + CONSENSUS_S, 'right')
    def wcount(flags):
        c = np.concatenate([[0], np.cumsum(flags)])
        return c[hi] - c[lo]
    same_dir = np.where(lng, wcount(lng), wcount(~lng))
    own = np.zeros(len(P), dtype=np.int64)
    for (d, is_l), g in P.groupby(['det', 'is_long'], sort=False):
        flags = np.zeros(len(P), dtype=np.int64)
        flags[g.index.values] = 1
        own[g.index.values] = wcount(flags)[g.index.values]
    P['consensus'] = (same_dir - own).astype(np.int16)
    return P

def main():
    print("Loading pool...")
    P = load_pool()
    P = P.dropna(subset=['y']).copy()
    P['year'] = P['day'].str[:4]
    dets = sorted(P['det'].unique())
    for d in dets: P[f'is_{d}'] = (P['det'] == d).astype(int)
    
    cols = BASE + ['consensus'] + [f'is_{d}' for d in dets]
    
    # 2024-sealed model
    trm = P['year'] == '2024'
    Xtr = P.loc[trm, cols].values.astype(float)
    ytr = P.loc[trm, 'y'].astype(int).values
    
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    print("Fitting logistic regression...")
    clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    
    coefs = clf.coef_[0]
    
    # Separate base features vs streams
    base_feats = cols[:len(BASE)+1] # BASE + consensus
    base_coefs = coefs[:len(BASE)+1]
    
    stream_names = cols[len(BASE)+1:]
    stream_coefs = coefs[len(BASE)+1:]
    
    # Sort streams by absolute coefficient
    abs_stream_coefs = np.abs(stream_coefs)
    total_abs = np.sum(abs_stream_coefs)
    
    idx_sorted = np.argsort(abs_stream_coefs)[::-1]
    
    cum_sum = 0
    top_k_streams = []
    
    for i in idx_sorted:
        stream = stream_names[i]
        c = stream_coefs[i]
        c_abs = abs_stream_coefs[i]
        cum_sum += c_abs
        top_k_streams.append((stream, c))
        if cum_sum / total_abs >= 0.8:
            break
            
    print(f"Total streams: {len(stream_names)}")
    print(f"Top K streams to reach 80% absolute sum: {len(top_k_streams)}")
    
    with open(os.path.join(OUT, 'top_k_streams.txt'), 'w') as f:
        f.write("Base features & Coefs:\n")
        for b, c in zip(base_feats, base_coefs):
            f.write(f"{b}: {c:.4f}\n")
            
        f.write("\nTop K Streams:\n")
        for stream, c in top_k_streams:
            f.write(f"{stream}: {c:.4f}\n")
            
        f.write(f"\nNormalization (mu, sd):\n")
        for b, m, s in zip(cols, mu, sd):
            f.write(f"{b} - mu: {m:.4f}, sd: {s:.4f}\n")
            
    print("Saved to reports/top_k_streams.txt")

if __name__ == '__main__':
    main()
