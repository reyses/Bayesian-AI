"""Autonomous dojo run with the GBM onset model arming and the owner's
protection stack doing the exits (owner 2026-08-04: "pick a new day run the
dojo and use the GBM and report back").

PRE-REGISTERED before the first bar (this matters — everything measured says
DIRECTION is not predictable, so this run is a demonstration of the MACHINE,
not a claim of edge):

  ENTRY  : onset model flags fakeout_poke or leg_descent >= 0.70 AND the
           last 60s net move is >= 2pt; enter WITH that move.
           Stop -10. One position at a time. Max 12 entries.
  EXIT   : the owner's stack, unchanged —
           ladder locks 50% of peak once MFE >= 5,
           entry-touch warning halts,
           75% ratchet from MFE >= 10,
           a close-through of 75% retention exits.
  EXPECT : ~zero EV before friction. The tables showed these events resolve
           by barrier geometry; a better detector buys latency, not edge.
           Reporting N and the honest number either way.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = '/media/moi/WindowsCode/Bayesian-AI'
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import pocket_dojo as pdj                                    # noqa: E402

PD = ['/home/moi/miniforge3/envs/bayesian/bin/python',
      os.path.join(REPO, 'research', 'dojo_forge', 'tools', 'pocket_dojo.py')]
ST = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state',
                  'pocket_dojo_state.json')
ONSET_MIN, MOVE_MIN, MAX_TRADES, STOP_PT = 0.70, 2.0, 12, 10.0
RETENTION_EXIT = 0.75


def state():
    return json.load(open(ST))


def run(args):
    return subprocess.run(PD + args, capture_output=True, text=True).stdout.strip()


def onset(s, t):
    line = pdj._gbm_line(s, t)
    out = {}
    for part in line.replace('ONSET(10s) ', '').split('|'):
        bits = part.strip().split()
        if len(bits) == 2 and bits[1].endswith('%'):
            out[bits[0]] = int(bits[1][:-1]) / 100
    return out


def net_move(s, t, secs=60):
    d1 = pdj._bars_tele(s['day'], '1s')
    w = d1[(d1['timestamp'] > t - secs) & (d1['timestamp'] <= t)]
    if len(w) < 10:
        return 0.0
    return float(w['close'].iloc[-1] - w['close'].iloc[0])


def main():
    log = []
    s = state()
    df = pdj._bars(s['day'])
    if not s.get('halt_ts5'):
        s['halt_ts5'] = int(df['timestamp'].iloc[s['cur']]) + 59
        json.dump(s, open(ST, 'w'), indent=1)
    for cycle in range(400):
        s = state()
        t = int(s['halt_ts5'])
        p = s.get('pos')
        if s.get('trades', 0) >= MAX_TRADES and not p:
            log.append('max trades reached')
            break
        if pdj._ets(t) >= '15:30:00':
            log.append('session end')
            break
        if p is None:
            o = onset(s, t)
            hit = max([o.get('fakeout_poke', 0), o.get('leg_descent', 0)])
            mv = net_move(s, t)
            if hit >= ONSET_MIN and abs(mv) >= MOVE_MIN:
                px = pdj._halt_px(s)
                d = 'long' if mv > 0 else 'short'
                stop = px - STOP_PT if d == 'long' else px + STOP_PT
                run(['call', d, '--stop', f'{stop:.2f}'])
                log.append(f'{pdj._ets(t)} ENTER {d} {px:.2f} '
                           f'(onset {hit:.0%}, 60s move {mv:+.2f})')
                continue
            out = run(['run', '60'])
            if 'no tape' in out:
                break
            continue
        out = run(['run', '300'])
        s = state()
        p = s.get('pos')
        if p is None:
            log.append(f'  BOOKED: {[l for l in out.splitlines() if "->" in l]}')
            continue
        if 'FROZEN' in out or 'ENTRY TOUCHED' in out or 'WARNING' in out:
            px = pdj._halt_px(s)
            pk = p.get('peak', 0.0)
            d = 1 if p['dir'] == 'long' else -1
            ret = ((px - p['entry']) * d) / pk if pk else 0
            if ret < RETENTION_EXIT:
                run(['exit'])
                s2 = state()
                log.append(f'  {pdj._ets(int(s2["halt_ts5"]))} EXIT ret '
                           f'{ret:.0%} -> day {s2["pnl_pts"]:+.2f}')
            else:
                run(['protect', 'rearm'])
        elif 'no tape' in out:
            break
    s = state()
    print('\n'.join(log))
    print(f'\nFINAL: day {s["day"]} {s["pnl_pts"]:+.2f}pt over '
          f'{s["trades"]} trades, clock {pdj._ets(int(s["halt_ts5"]))}')


if __name__ == '__main__':
    main()
