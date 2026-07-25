#!/usr/bin/env python3
"""Replay-equivalence test: the STREAMING LegHealthGauge must reproduce the
batch composite-gauge states bar-for-bar on historical episodes — the proof
that the per-trade dynamic indicator is the same instrument the studies
validated. Any mismatch is listed loudly; zero mismatches = PASS."""
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'pipeline'))
from leg_health_gauge import LegHealthGauge, SICK_DETECTORS, Z_SICK, Z_FADE, LAG, MIN_BASE  # noqa: E402

REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})


def parse(text):
    feats, px, leg = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
                if k in NEED:
                    feats[k] = float(v)
        elif s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    return feats, px, leg


def batch_states(rows):
    """The composite_gauge.py walk, verbatim semantics."""
    out = []
    events, fade_at = {}, None
    conv_ok_rows = []
    hist_rows = []
    prev_leg_start = None
    for i, (feats, px, leg_age) in enumerate(rows):
        if px is None or leg_age is None:
            out.append(None)
            hist_rows.append(feats)
            conv_ok_rows.append(None)
            continue
        leg_start = int(i - leg_age)
        if prev_leg_start is None or abs(leg_start - prev_leg_start) > 1:
            events, fade_at = {}, None
        prev_leg_start = leg_start
        for fname, tail in SICK_DETECTORS:
            base = [hist_rows[j].get(fname) for j in range(max(0, leg_start), i)]
            base = [b for b in base if b is not None]
            v = feats.get(fname)
            if v is not None and len(base) >= MIN_BASE:
                sd = st.pstdev(base)
                if sd:
                    z = (v - st.mean(base)) / sd
                    fired = (z >= Z_SICK) if tail == 'hi' else (z <= -Z_SICK)
                    if fired and (fname, tail) not in events:
                        events[(fname, tail)] = i
        conv = None
        if feats.get('bar_range') and 'body' in feats:
            conv = feats['body'] / feats['bar_range']
        cbase = [conv_ok_rows[j] for j in range(max(0, leg_start), i)
                 if conv_ok_rows[j] is not None]
        if conv is not None and len(cbase) >= MIN_BASE:
            sd = st.pstdev(cbase)
            if sd and (conv - st.mean(cbase)) / sd <= -Z_FADE and fade_at is None:
                fade_at = i
        sick = sum(1 for t0 in events.values() if (i - t0) >= LAG)
        faded = fade_at is not None and (i - fade_at) >= LAG
        out.append((('FADED' if faded else 'ALIVE'), sick))
        hist_rows.append(feats)
        conv_ok_rows.append(conv)
    return out


def main():
    mismatches = checked = 0
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json')))[:40]:
        pkt = json.load(open(pkt_path))
        rows = [parse(fr['text']) for fr in pkt['frames']]
        want = batch_states(rows)
        g = LegHealthGauge()
        for i, (feats, px, leg_age) in enumerate(rows):
            if px is None or leg_age is None:
                continue
            got = g.update(leg_age=leg_age, feats=feats)
            checked += 1
            if want[i] is not None and (got['vigor'], got['sick']) != want[i]:
                mismatches += 1
                if mismatches <= 5:
                    print(f"MISMATCH {os.path.basename(pkt_path)} bar {i}: "
                          f"stream={got['vigor'],got['sick']} batch={want[i]}")
    print(f"checked {checked} bars across 40 episodes: "
          f"{mismatches} mismatches -> {'PASS' if mismatches == 0 else 'FAIL'}")
    sys.exit(0 if mismatches == 0 else 1)


if __name__ == '__main__':
    main()
