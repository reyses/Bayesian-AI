"""V-shape retest cohort: p(continuation) at the first defended poke of the
V-floor shelf, conditioned on today's day-shape (owner 2026-08-03, mid-dojo
on 2024_09_16: "describe in words and delta points from overnight to current
bar, then search for that specific cohort, and then run probability").

Signature being matched (2024_09_16 measured: flush -173.5pt by 09:35,
V-recovery 79% by 09:56, stair-down retest of the 19600 V-floor at 10:26):
  1. OPEN FLUSH:  open(09:30) - min_low(09:30-09:50) >= FLUSH_MIN pts
  2. V-RECOVERY:  (max_high(low_t..10:20) - dump_low) / flush >= REC_FRAC
  3. SHELF:       modal 1m close (2pt bins) in [dump_low, dump_low+0.45*flush]
                  during 09:30-10:05 (the flush-consolidation dwell)
  4. RETEST:      first bar after the recovery peak, 10:00-12:30, with
                  low <= shelf + RETEST_PT
  5. DEFENDED POKE: poke_extreme = min_low(first 3 bars of retest); defended
                  if high within 5 bars of the poke >= poke_extreme + DEF_PT
  6. OUTCOME:     within 90 min of the defense —
                  CRACK if low <= poke_extreme - CRACK_PT happens BEFORE
                  high >= poke_extreme + HOLD_PT; HOLD on the reverse;
                  UNRESOLVED if neither in window.

All times ET. Per-day ATLAS outrights — no cross-day levels, so the contract
roll seam does not bite. Causal by construction: every quantity is computed
from bars at or before the decision moment except the OUTCOME, which is the
thing being measured.

Run from repo root:
  python research/dojo_forge/tools/vshape_retest_cohort.py
Writes research/dojo_forge/reports/vshape_retest_cohort.md
"""
import glob
import os

import numpy as np
import pandas as pd

FLUSH_MIN = 60.0     # pts; today 173.5 — cohort floor for "giant open flush"
REC_FRAC = 0.60      # today 0.79
RETEST_PT = 5.0
DEF_PT = 5.0         # today's defense bounce: 6.75
CRACK_PT = 5.0
HOLD_PT = 15.0
OUT_MIN = 90         # outcome window, minutes

DAYS = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
REPORT = 'research/dojo_forge/reports/vshape_retest_cohort.md'

# LIVE-DAY GUARD (added after the 2026-08-03 leak): if the pocket-dojo sim
# is mid-day, truncate that day's bars at the sim clock so the scan can
# never print the frozen instant's future into a live decision. The first
# run of this tool leaked today's poke depth + outcome mid-freeze — the
# owner's decision had to be corpus-flagged BLIND-INTEGRITY.
try:
    import json
    _st = json.load(open('research/dojo_forge/gate_state/'
                         'pocket_dojo_state.json'))
    LIVE_DAY, LIVE_TS = _st.get('day'), int(_st.get('halt_ts5') or 0)
except Exception:
    LIVE_DAY, LIVE_TS = None, 0


def one_day(path):
    day = os.path.basename(path).replace('.parquet', '')
    if day == LIVE_DAY and LIVE_TS:
        # EXCLUDE the live day outright. Bar-level truncation is not enough:
        # 1m bars are labeled by minute START, so the bar containing the
        # frozen instant still carries up to 59s of its future.
        return None
    d = pd.read_parquet(path)
    et = (pd.to_datetime(d['timestamp'], unit='s', utc=True)
          .dt.tz_convert('America/New_York'))
    d = d.assign(hm=et.dt.strftime('%H:%M'))
    rth = d[d['hm'] >= '09:30']
    if not len(rth):
        return None
    open_px = float(rth['open'].iloc[0])
    dump = d[(d['hm'] >= '09:30') & (d['hm'] <= '09:50')]
    if not len(dump):
        return None
    dump_low = float(dump['low'].min())
    flush = open_px - dump_low
    if flush < FLUSH_MIN:
        return None
    t_low = dump.loc[dump['low'].idxmin(), 'hm']
    vwin = d[(d['hm'] > t_low) & (d['hm'] <= '10:20')]
    if not len(vwin):
        return None
    v_peak = float(vwin['high'].max())
    rec = (v_peak - dump_low) / flush
    if rec < REC_FRAC:
        return None
    t_peak = vwin.loc[vwin['high'].idxmax(), 'hm']
    # shelf: modal close in the lower 45% of the flush range, 09:30-10:05
    cwin = d[(d['hm'] >= '09:30') & (d['hm'] <= '10:05')]
    zone = cwin[(cwin['close'] >= dump_low)
                & (cwin['close'] <= dump_low + 0.45 * flush)]
    if len(zone) < 5:
        return None
    bins = np.arange(zone['close'].min(), zone['close'].max() + 2, 2.0)
    if len(bins) < 2:
        return None
    hist, edges = np.histogram(zone['close'], bins=bins)
    shelf = float(edges[np.argmax(hist)] + 1.0)
    # retest after the recovery peak
    post = d[(d['hm'] > max(t_peak, '10:00')) & (d['hm'] <= '12:30')]
    post = post.reset_index(drop=True)
    hit = post.index[post['low'] <= shelf + RETEST_PT]
    if not len(hit):
        return dict(day=day, flush=flush, rec=rec, shelf=shelf,
                    outcome='NO_RETEST')
    i0 = int(hit[0])
    poke = float(post['low'].iloc[i0:i0 + 3].min())
    after = post.iloc[i0 + 1:i0 + 6]
    if not len(after) or float(after['high'].max()) < poke + DEF_PT:
        return dict(day=day, flush=flush, rec=rec, shelf=shelf,
                    outcome='NO_DEFENSE')
    i_def = i0 + 1 + int(after['high'].to_numpy().argmax())
    ow = post.iloc[i_def:i_def + OUT_MIN]
    crack_i = ow.index[ow['low'] <= poke - CRACK_PT]
    hold_i = ow.index[ow['high'] >= poke + HOLD_PT]
    ci = int(crack_i[0]) if len(crack_i) else None
    hi = int(hold_i[0]) if len(hold_i) else None
    if ci is not None and (hi is None or ci < hi):
        out = 'CRACK'
    elif hi is not None:
        out = 'HOLD'
    else:
        out = 'UNRESOLVED'
    return dict(day=day, flush=round(flush, 1), rec=round(rec, 2),
                shelf=shelf, poke=poke, outcome=out,
                t_retest=post['hm'].iloc[i0])


rows = [r for r in (one_day(p) for p in DAYS) if r]
res = pd.DataFrame(rows)
lines = ['# V-shape retest cohort — p(continuation) at the first defended '
         'poke', '',
         f'Scanned {len(DAYS)} ATLAS days. Signature days (flush >= '
         f'{FLUSH_MIN:.0f}pt by 09:50, V-recovery >= {REC_FRAC:.0%} by '
         f'10:20): **{len(res)}**', '']
if len(res):
    lines.append(res.to_string(index=False))
    lines.append('')
    dec = res[res['outcome'].isin(['CRACK', 'HOLD'])]
    n, k = len(dec), int((dec['outcome'] == 'CRACK').sum())
    if n:
        p = k / n
        z = 1.96
        den = 1 + z * z / n
        ctr = (p + z * z / (2 * n)) / den
        hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lines.append(f'**Decided pokes: {n} — CRACK {k} ({p:.0%}), Wilson '
                     f'95% CI [{ctr - hw:.0%}, {ctr + hw:.0%}]**')
        lines.append(f'(others: {list(res["outcome"].value_counts().items())})')
os.makedirs(os.path.dirname(REPORT), exist_ok=True)
open(REPORT, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
