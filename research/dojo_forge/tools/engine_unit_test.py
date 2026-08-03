"""Unit tests for _engine_run branches unreachable on the 2024_09_16 tape:
(1) freeze-release second WITH prot_hard armed (old code: None*0.7 TypeError)
(2) gap-through stop -> fill at OPEN not at the level
Monkeypatches the data loaders and persistence; no state file touched.
"""
import sys
sys.path.insert(0, 'research/dojo_forge/tools')
import pandas as pd
import pocket_dojo as pdj

BASE = 1726496400  # any aligned minute


def bars_1s(rows):
    return pd.DataFrame([dict(timestamp=BASE + i, open=o, high=h, low=l,
                              close=c) for i, (o, h, l, c) in rows])


def frame_1m():
    return pd.DataFrame([dict(timestamp=BASE - 60, open=100, high=101,
                              low=99, close=100),
                         dict(timestamp=BASE, open=100, high=101, low=99,
                              close=100)])


saved = []
pdj._save = lambda s: saved.append(dict(s))
pdj._log = lambda *a, **k: None
pdj._bars = lambda day: frame_1m()

# ---- T-A: release second with hard armed --------------------------------
tape = bars_1s([
    (1, (91.0, 91.5, 90.5, 91.0)),    # cur 9.0 > floor 8.4, no new peak
    (2, (91.0, 91.0, 87.0, 88.0)),    # new MFE 13 -> RELEASE (hard armed!)
    (3, (88.0, 89.7, 88.0, 89.5)),    # warn = 100-0.8*13 = 89.6 wicked -> FREEZE 13
])
pdj._bars_tele = lambda day, res: tape if res == '1s' else None
s = dict(day='x', cur=1, peek_offset=0, halt_ts5=BASE,
         pos=dict(dir='short', entry=100.0, stop=120.0, peak=12.0,
                  frozen=12.0, prot_armed=True, prot_hard=True),
         protect=dict(on=True, warn=0.80, hard=0.70, min_mfe=10.0,
                      arm='always', region=None, prox_pt=3.0),
         owner_lines=[], pnl_pts=0.0, trades=0)
ev, halted = pdj._engine_run(s, frame_1m(), 10)
print('T-A events:', ev)
p = s['pos']
assert p is not None, 'position must survive the release second'
assert p['frozen'] == 13.0 and p['peak'] == 13.0, (p['frozen'], p['peak'])
assert p['prot_hard'] is True, 'hard stays armed through the ratchet'
print('T-A PASS: release-with-hard survived (old code: TypeError), '
      're-froze at 13.0, hard still armed')

# ---- T-B: gap-through stop fills at OPEN --------------------------------
tape = bars_1s([
    (1, (99.0, 99.5, 98.5, 99.0)),
    (2, (93.0, 94.0, 92.5, 93.5)),    # gaps THROUGH the 95 stop: open 93
])
pdj._bars_tele = lambda day, res: tape if res == '1s' else None
s = dict(day='x', cur=1, peek_offset=0, halt_ts5=BASE,
         pos=dict(dir='long', entry=100.0, stop=95.0, peak=0.0),
         protect=dict(on=False), owner_lines=[], pnl_pts=0.0, trades=0)
ev, halted = pdj._engine_run(s, frame_1m(), 10)
print('T-B events:', ev)
exp = (93.0 - 100.0) - pdj.FRICTION_PT
assert s['pos'] is None and abs(s['pnl_pts'] - round(exp, 2)) < 1e-9, \
    (s['pos'], s['pnl_pts'], exp)
assert 'gapped' in ev[0], ev[0]
print(f'T-B PASS: gapped stop filled at open 93.00 ({exp:+.2f}pt), '
      f'not at the 95 level')
print('ALL UNIT TESTS PASS')
