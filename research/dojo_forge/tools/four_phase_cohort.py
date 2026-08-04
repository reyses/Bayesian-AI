"""Four-phase cohort (owner: 'down then up then a small stall then down —
and whatever happens next?'). Phases: flush >=60pt by 09:50; V-recovery
>=60% by 10:20; STALL = 10min after the peak holding above peak-0.30*flush;
DOWN = price reaches the shelf zone (dump_low+0.45*flush) by 12:30.
From t_arrival: first +-15pt move, 30/60min net, later low-break /
peak-reclaim, RTH close in V-range. Live day excluded."""
import glob, json, os
import numpy as np, pandas as pd

FLUSH_MIN, REC_FRAC = 60.0, 0.60
STALL_MIN, STALL_GIVE = 10, 0.30
SHELF_FRAC, MOVE_PT = 0.45, 15.0

st = json.load(open('research/dojo_forge/gate_state/pocket_dojo_state.json'))
LIVE = st.get('day')
rows = []
for path in sorted(glob.glob('DATA/ATLAS/1m/*.parquet')):
    day = os.path.basename(path).replace('.parquet', '')
    if day == LIVE:
        continue
    d = pd.read_parquet(path)
    et = (pd.to_datetime(d['timestamp'], unit='s', utc=True)
          .dt.tz_convert('America/New_York'))
    d = d.assign(hm=et.dt.strftime('%H:%M')).reset_index(drop=True)
    rth = d[d['hm'] >= '09:30']
    if not len(rth):
        continue
    open_px = float(rth['open'].iloc[0])
    dump = d[(d['hm'] >= '09:30') & (d['hm'] <= '09:50')]
    if not len(dump):
        continue
    dump_low = float(dump['low'].min())
    flush = open_px - dump_low
    if flush < FLUSH_MIN:
        continue
    t_low = dump.loc[dump['low'].idxmin(), 'hm']
    vwin = d[(d['hm'] > t_low) & (d['hm'] <= '10:20')]
    if not len(vwin):
        continue
    v_peak = float(vwin['high'].max())
    if (v_peak - dump_low) / flush < REC_FRAC:
        continue
    ipk = int(vwin['high'].idxmax())
    stall = d.iloc[ipk + 1: ipk + 1 + STALL_MIN]
    if len(stall) < STALL_MIN or float(stall['low'].min()) < v_peak - STALL_GIVE * flush:
        continue                                   # no small stall -> out
    shelf_zone = dump_low + SHELF_FRAC * flush
    post = d[(d.index > ipk + STALL_MIN) & (d['hm'] <= '12:30')]
    arr = post.index[post['low'] <= shelf_zone]
    if not len(arr):
        continue                                   # never made the 2nd down
    ia = int(arr[0])
    a_px = float(d['close'].iloc[ia])
    fwd = d[(d.index > ia) & (d['hm'] <= '16:00')].reset_index(drop=True)
    if len(fwd) < 30:
        continue
    up_i = fwd.index[fwd['high'] >= a_px + MOVE_PT]
    dn_i = fwd.index[fwd['low'] <= a_px - MOVE_PT]
    ui = int(up_i[0]) if len(up_i) else None
    di = int(dn_i[0]) if len(dn_i) else None
    first = ('UP' if ui is not None and (di is None or ui < di)
             else 'DOWN' if di is not None else 'NEITHER')
    n30 = float(fwd['close'].iloc[min(29, len(fwd) - 1)]) - a_px
    n60 = float(fwd['close'].iloc[min(59, len(fwd) - 1)]) - a_px
    lb = bool((fwd['low'] < dump_low).any())
    pr = bool((fwd['high'] > v_peak).any())
    close_frac = (float(fwd['close'].iloc[-1]) - dump_low) / (v_peak - dump_low)
    rows.append(dict(day=day, arr_hm=d['hm'].iloc[ia], first=first,
                     n30=round(n30, 1), n60=round(n60, 1), low_break=lb,
                     peak_reclaim=pr, close_frac=round(close_frac, 2)))

f = pd.DataFrame(rows)
n = len(f)
print(f'FOUR-PHASE cohort (stall required): N = {n}')
if n:
    fc = f['first'].value_counts()
    up = int(fc.get('UP', 0)); dn = int(fc.get('DOWN', 0))
    m = up + dn
    if m:
        p = up / m
        se = 1.96 * np.sqrt(p * (1 - p) / m)
        print(f'first +-15pt from arrival: UP {up} ({p:.0%} '
              f'[{max(0,p-se):.0%},{min(1,p+se):.0%}]) vs DOWN {dn}'
              + (f' | NEITHER {int(fc.get("NEITHER",0))}' if fc.get('NEITHER') else ''))
    print(f'net 30min from arrival: median {f["n30"].median():+.1f} '
          f'[q25 {f["n30"].quantile(.25):+.1f}, q75 {f["n30"].quantile(.75):+.1f}]')
    print(f'net 60min: median {f["n60"].median():+.1f} '
          f'[q25 {f["n60"].quantile(.25):+.1f}, q75 {f["n60"].quantile(.75):+.1f}]')
    print(f'later: V-low broken {f["low_break"].mean():.0%} | '
          f'V-peak reclaimed {f["peak_reclaim"].mean():.0%}')
    print(f'RTH close in V-range: median {f["close_frac"].median():.2f} '
          f'[{f["close_frac"].quantile(.25):.2f}, {f["close_frac"].quantile(.75):.2f}]')
    print(); print(f.to_string(index=False))
