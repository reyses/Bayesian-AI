"""
ADX -> P(direction right) binary logistic (stage-0 combiner, first signal).

Pipeline (now the STANDARD per dossier, incl. zigzag — Moises 2026-07-15):
  1) signals vs AI labels (overlap)  2) transition profile  3) feature rows
  4) binary logistic -> calibrated P(right).

Feature row per ADX signal (doc-074 setting, continuous windows):
  - pivot_age_min : CAUSAL streaming-zigzag pivot age (v1: 1m ATR(14)x4 reversal
    confirmation off running extremes — canonical zigzag spec, causal, no oracle).
  - sig_with_leg  : ADX direction == current causal leg direction (0/1)
  - adx_val       : ADX value at fire
  - tod_frac      : fraction of RTH elapsed
  - interaction   : sig_with_leg x pivot_age_min   (the doc-078 inversion carrier)
Target: signal direction agreed with the ACTIVE AI label (ground truth).
Discipline: train 2024, held-out 2025+2026, day-block CIs on tercile lifts.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adx_label_overlap import D5, LBL, RTH0, RTH1, N_ADX, N_CROSS, THR, TAIL  # noqa
from sklearn.linear_model import LogisticRegression

ATR_N, ATR_MULT, BAR_1M = 14, 4.0, 12


def stream_day(full, start, day):
    """Compute per-bar: adx, cross fires, causal zigzag pivot age + leg dir."""
    h, l, c = full['high'], full['low'], full['close']
    up, dn = h.diff(), -l.diff()
    dm_p = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=full.index)
    dm_m = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=full.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr_s = tr.rolling(N_ADX, min_periods=N_ADX).mean().replace(0, np.nan)
    di_p = 100 * dm_p.rolling(N_ADX, min_periods=N_ADX).mean() / tr_s
    di_m = 100 * dm_m.rolling(N_ADX, min_periods=N_ADX).mean() / tr_s
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx = dx.rolling(N_ADX, min_periods=N_ADX).mean()
    ma = c.rolling(N_CROSS, min_periods=N_CROSS).mean()
    x_up = ((c.shift(1) <= ma.shift(1)) & (c > ma)).values
    x_dn = ((c.shift(1) >= ma.shift(1)) & (c < ma)).values
    # 1m ATR(14) for zigzag threshold (causal, trailing)
    c1 = c.groupby(np.arange(len(c)) // BAR_1M).last()
    h1 = h.groupby(np.arange(len(h)) // BAR_1M).max()
    l1 = l.groupby(np.arange(len(l)) // BAR_1M).min()
    pc1 = c1.shift(1)
    tr1 = pd.concat([h1 - l1, (h1 - pc1).abs(), (l1 - pc1).abs()], axis=1).max(axis=1)
    atr1 = tr1.rolling(ATR_N, min_periods=ATR_N).mean()
    thr_by_bar = (atr1.reindex(np.arange(len(c)) // BAR_1M).values * ATR_MULT)
    # streaming zigzag: track extremes; reversal >= thr confirms pivot
    cv = c.values
    piv_i, leg = np.zeros(len(cv), dtype=np.int64), np.zeros(len(cv), dtype=np.int8)
    hi_i = lo_i = 0; hi_v = lo_v = cv[0]; d = 0; last_piv = 0
    for i in range(1, len(cv)):
        x = cv[i]; t = thr_by_bar[i] if np.isfinite(thr_by_bar[i]) else np.inf
        if x > hi_v: hi_v, hi_i = x, i
        if x < lo_v: lo_v, lo_i = x, i
        if d >= 0 and hi_v - x >= t:      # down-reversal: hi was a pivot
            last_piv, d = hi_i, -1; lo_v, lo_i = x, i
        elif d <= 0 and x - lo_v >= t:    # up-reversal: lo was a pivot
            last_piv, d = lo_i, 1; hi_v, hi_i = x, i
        piv_i[i], leg[i] = last_piv, d
    dt = pd.to_datetime(full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    rth = ((dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)).values
    rows = []
    fire = (adx.values > THR) & (x_up | x_dn) & rth
    tod0 = dt.dt.normalize() + pd.Timedelta(hours=8, minutes=30)
    tod = ((dt - tod0.dt.tz_localize(None).dt.tz_localize('America/Chicago')).dt.total_seconds() / (6.75 * 3600)).values
    for i in np.flatnonzero(fire):
        if i < start: continue
        is_long = bool(x_up[i])
        rows.append(dict(ts=int(full['timestamp'].iloc[i]), is_long=is_long,
                         adx=float(adx.values[i]),
                         pivot_age_min=(i - piv_i[i]) * 5 / 60.0,
                         sig_with_leg=int((leg[i] > 0) == is_long) if leg[i] != 0 else 0,
                         tod=float(np.clip(tod[i], 0, 1)), day=day))
    return rows


def main():
    lblf = {os.path.basename(f)[9:19]: f for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    files = [f for f in sorted(glob.glob(os.path.join(D5, '*.parquet')))
             if os.path.basename(f)[:10].replace('_', '-') in lblf]
    tail, feats = None, []
    for p in files:
        df = pd.read_parquet(p, columns=['timestamp', 'high', 'low', 'close']).sort_values('timestamp').reset_index(drop=True)
        full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
        feats += stream_day(full, len(tail) if tail is not None else 0,
                            os.path.basename(p)[:10])
        tail = df.tail(max(TAIL, ATR_N * BAR_1M + BAR_1M))
    F = pd.DataFrame(feats)
    # target: agreement with active label
    tgt = []
    for day, g in F.groupby('day'):
        iso = day.replace('_', '-')
        tr = json.load(open(lblf[iso])).get('trades', [])
        labs = [(t['entry_ts'], t['exit_ts'], t.get('direction') == 'LONG') for t in tr if t.get('exit_ts')]
        for _, r in g.iterrows():
            hit = [lg for a, b, lg in labs if a <= r['ts'] <= b]
            tgt.append(int(hit[0] == r['is_long']) if hit else np.nan)
    F['y'] = tgt
    F = F.dropna(subset=['y'])
    F['year'] = F['day'].str[:4]
    F['inter'] = F['sig_with_leg'] * F['pivot_age_min']
    cols = ['pivot_age_min', 'sig_with_leg', 'adx', 'tod', 'inter']
    tr_m, te_m = F['year'] == '2024', F['year'] != '2024'
    Xtr, ytr = F.loc[tr_m, cols].values, F.loc[tr_m, 'y'].astype(int).values
    Xte, yte = F.loc[te_m, cols].values, F.loc[te_m, 'y'].astype(int).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    pte = clf.predict_proba((Xte - mu) / sd)[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f'train 2024 N={len(ytr)} (base {ytr.mean():.2f}) | test 2025+26 N={len(yte)} (base {yte.mean():.2f})')
    print('coefs:', dict(zip(cols, np.round(clf.coef_[0], 3))))
    print(f'OOS AUC = {roc_auc_score(yte, pte):.3f}')
    q = pd.qcut(pte, 3, labels=['low', 'mid', 'high'])
    days_te = F.loc[te_m, 'day'].values
    for b in ['low', 'mid', 'high']:
        m = np.asarray(q == b)
        uq = np.unique(days_te[m]); boots = []
        for _ in range(2000):
            s = np.random.choice(uq, len(uq), True)
            v = np.concatenate([yte[m][days_te[m] == d] for d in s])
            if len(v): boots.append(v.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f'  P-tercile {b:4}: N={m.sum():4} observed agreement {yte[m].mean():.2f} CI[{lo:.2f},{hi:.2f}]  meanP {pte[m].mean():.2f}')
    F.to_parquet(os.path.join(os.path.dirname(__file__), '..', 'reports', 'adx_signal_features.parquet'))
    print('rows -> reports/adx_signal_features.parquet')


if __name__ == '__main__':
    main()
