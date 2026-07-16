"""
ADX -> AI-LABEL overlap (the north-star metric: does the signal detect the labels?).

Setting under test: the doc-074 candidate — SMA smoothing, N_adx=84 (7 min), thr=25,
cross-MA=240 (20 min) — computed CONTINUOUSLY (no cold start; 240-bar tail carried
across file boundaries), triggers RTH-only.

Metrics (labels = DATA/ai_cusp_picks, the golden ground truth):
  0. Label time-coverage of RTH (sanity: the labeler CHAINS trades, so coverage is
     expected near-total -> "inside a label" is trivial; DIRECTION is the real test).
  1. Signal-side: % of signals whose direction AGREES with the label active at that
     moment (baseline 50% — labels are 50/50 long/short). Day-block bootstrap CI.
  2. Phase-in-label at signal time: (ts - entry)/(exit - entry). Early phase = the
     signal fires while most of the label's move is still ahead = actionable.
  3. Label-side: % of labels containing >=1 direction-agreeing signal (vs ~47/day oracle).
Split by year. No stops, no P&L — detection quality only (Moises 2026-07-15).
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
D5 = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
LBL = os.path.join(ROOT, 'DATA', 'ai_cusp_picks')
RTH0, RTH1 = pd.Timestamp('08:30').time(), pd.Timestamp('15:15').time()
N_ADX, N_CROSS, THR = 84, 240, 25.0
TAIL = 400          # bars carried across file boundaries (> max window)


def signals_for_files(files):
    """Yield (day_str, ts, is_long) triggers; continuous windows via tail carry."""
    tail = None
    for p in files:
        df = pd.read_parquet(p, columns=['timestamp', 'high', 'low', 'close'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
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
        x_up = (c.shift(1) <= ma.shift(1)) & (c > ma)
        x_dn = (c.shift(1) >= ma.shift(1)) & (c < ma)
        dt = pd.to_datetime(full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        rth = (dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)
        fire = (adx > THR) & (x_up | x_dn) & rth
        start = len(tail) if tail is not None else 0
        day = os.path.basename(p).replace('.parquet', '')
        for i in np.flatnonzero(fire.values):
            if i < start:
                continue
            yield day, int(full['timestamp'].iloc[i]), bool(x_up.iloc[i])
        tail = df.tail(TAIL)


def main():
    lbl_files = {os.path.basename(f)[9:19]: f
                 for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    d5_files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    d5_files = [f for f in d5_files
                if os.path.basename(f)[:10].replace('_', '-') in lbl_files]
    print(f'{len(d5_files)} days with both 5s data and labels')

    rows, cover = [], []
    lab_hit = {}          # (day, label_idx) -> agreeing-signal found
    lab_tot = 0
    cur_labels, cur_day = None, None
    for day, ts, is_long in signals_for_files(d5_files):
        iso = day.replace('_', '-')
        if iso != cur_day:
            cur_day = iso
            trades = json.load(open(lbl_files[iso])).get('trades', [])
            cur_labels = [(t['entry_ts'], t['exit_ts'], t.get('direction') == 'LONG', k)
                          for k, t in enumerate(trades) if t.get('exit_ts')]
            lab_tot += len(cur_labels)
            span = sum(b - a for a, b, _, _ in cur_labels)
            cover.append(min(1.0, span / (6.75 * 3600)))
        hit = [(a, b, lg, k) for a, b, lg, k in cur_labels if a <= ts <= b]
        if not hit:
            rows.append((iso, day[:4], 0, np.nan, np.nan))
            continue
        a, b, lab_long, k = hit[0]
        agree = int(lab_long == is_long)
        phase = (ts - a) / max(1, b - a)
        rows.append((iso, day[:4], 1, agree, phase))
        if agree:
            lab_hit[(iso, k)] = True

    df = pd.DataFrame(rows, columns=['day', 'year', 'in_label', 'agree', 'phase'])
    print(f'\nlabel RTH coverage (mean of per-day union approx): {np.mean(cover):.2f}')
    print(f'signals total: {len(df)}  inside-a-label: {df.in_label.mean():.2f}')
    for yr, g in df[df.in_label == 1].groupby('year'):
        agr = g['agree'].dropna()
        # day-block bootstrap
        days = g['day'].values
        uq = np.unique(days)
        boots = []
        for _ in range(2000):
            samp = np.random.choice(uq, len(uq), True)
            vals = np.concatenate([agr.values[days == d] for d in samp])
            if len(vals):
                boots.append(vals.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ph = g['phase'].dropna()
        buckets = pd.cut(ph, [0, .1, .25, .5, .75, 1.0]).value_counts().sort_index()
        print(f'\n{yr}: N={len(g)} signals | direction agreement {agr.mean():.2f} '
              f'CI[{lo:.2f},{hi:.2f}] (baseline 0.50)')
        print(f'  phase-in-label: ' + ' | '.join(f'{iv}: {n}' for iv, n in buckets.items()))
    hit_by_year = {}
    for (iso, k) in lab_hit:
        hit_by_year[iso[:4]] = hit_by_year.get(iso[:4], 0) + 1
    print(f'\nlabels with >=1 agreeing signal: {sum(hit_by_year.values())}/{lab_tot} '
          f'({100*sum(hit_by_year.values())/max(1,lab_tot):.1f}%) by year: {hit_by_year}')


if __name__ == '__main__':
    main()
