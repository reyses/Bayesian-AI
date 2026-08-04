#!/usr/bin/env python3
"""HUMAN DOJO recorder (owner 2026-07-28, TG: "maybe I should run the dojo and
explain what I do?"). Two days of mechanical features all hit the ~coin ceiling,
but the owner's MANUAL picks work (doc 2026-05-10). So capture the HUMAN process
on the BLIND gate: serve the exact episode frames one at a time (no chart, no
future), the owner types their READ + DECISION each frame; log everything; reveal
the outcome only AFTER commit. Then score the human vs coin + mine the reasoning
for the signal the features miss.

Run:  python research/dojo_forge/tools/human_dojo.py --list
      python research/dojo_forge/tools/human_dojo.py --eid <id> [--who moises]
Per frame you type free-text reasoning; end a line with a decision token to log it:
  HOLD / EXIT   (exit drill)   or   LONG / SHORT   (direction)   or   TURN (call a pivot)
Commands: /next reveal next frame · /commit lock your final call + reveal outcome · /quit
Log: research/dojo_forge/reports/human_dojo/<eid>.<who>.jsonl
"""
import argparse
import glob
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.abspath(os.path.join(HERE, '..'))
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
LOGDIR = os.path.join(DOJO, 'reports', 'human_dojo')
_PX = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*pts')
_DEC = re.compile(r'\b(HOLD|EXIT|LONG|SHORT|TURN)\b', re.I)


def load(eid):
    return json.load(open(os.path.join(PACKETS, f'{eid}.json'), encoding='utf-8'))


def reveal(fav, decided_at):
    vals = [v for v in fav if v is not None]
    if not vals:
        print('[no fav-pts to reveal]'); return
    peak = max(vals); pk = fav.index(peak)
    here = fav[decided_at] if (decided_at is not None and decided_at < len(fav)) else None
    print('\x1b[33m──── OUTCOME (was blind until now) ────\x1b[0m')
    print('fav pts/frame: ' + ' '.join(f'{v:+.0f}' if v is not None else ' . ' for v in fav))
    if here is not None:
        print(f'you committed at frame {decided_at} = {here:+.0f} pts')
    print(f'peak {peak:+.0f} @frame {pk} · final {vals[-1]:+.0f} pts\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eid', default=None)
    ap.add_argument('--who', default='owner')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    if args.list or not args.eid:
        ids = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(PACKETS, '*.json')))]
        print(f'{len(ids)} episodes. Sample:')
        for i in ids[::max(1, len(ids) // 20)][:20]:
            print(' ', i)
        return
    os.makedirs(LOGDIR, exist_ok=True)
    pkt = load(args.eid); frames = pkt['frames']; meta = pkt.get('meta', {})
    fav = [float(m.group(1)) if (m := _PX.search(f.get('text', ''))) else None for f in frames]
    logp = os.path.join(LOGDIR, f'{args.eid}.{args.who}.jsonl')
    log = open(logp, 'a', encoding='utf-8')

    def rec(ev, **kw):
        log.write(json.dumps(dict(ts=time.time(), eid=args.eid, who=args.who, event=ev, **kw)) + '\n'); log.flush()

    print(f'\x1b[36m═══ HUMAN DOJO · {args.eid} · dir={meta.get("direction","?")} · '
          f'{len(frames)} frames · BLIND ═══\x1b[0m')
    print('Type your read each frame; include HOLD/EXIT or LONG/SHORT or TURN to log a call.')
    print('/next = reveal next frame · /commit = final call + outcome · /quit\n')
    rec('start', direction=meta.get('direction'), n_frames=len(frames))
    revealed = 0

    def show(i):
        print('\x1b[36m' + f'──── FRAME {i}/{len(frames)-1} (you see only this) ────' + '\x1b[0m')
        print(frames[i]['text'].rstrip() + '\n')
    show(0); rec('frame_shown', frame=0); revealed = 1
    while True:
        try:
            u = input(f'f{revealed-1}> ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not u:
            continue
        if u == '/quit':
            break
        if u == '/next':
            if revealed >= len(frames):
                print('[no more frames — /commit to reveal outcome]'); continue
            show(revealed); rec('frame_shown', frame=revealed); revealed += 1
            continue
        if u == '/commit':
            dec = _DEC.search(u)
            rec('commit', frame=revealed - 1)
            reveal(fav, revealed - 1)
            print(f'[logged -> {logp}]')
            continue
        d = _DEC.search(u)
        rec('note', frame=revealed - 1, text=u, decision=(d.group(1).upper() if d else None))
        if d:
            print(f'  \x1b[32m[logged call: {d.group(1).upper()} @frame {revealed-1}]\x1b[0m')
    rec('end')
    log.close()
    print(f'session saved -> {logp}')


if __name__ == '__main__':
    main()
