#!/usr/bin/env python3
"""Morning-bundle stats: distiller bank + day-carry episode behavior."""
import json
import os
import re
import sqlite3

DOJO = os.path.join(os.path.dirname(__file__), '..')
con = sqlite3.connect(
    f"file:{os.path.join(DOJO, 'gate_state', 'teacher_memory_v2.db')}?mode=ro",
    uri=True)
texts = [t for (t,) in con.execute(
    "SELECT text FROM memos WHERE created_run='distiller'")]
rule = re.compile(r'G\d+(?:\.\d+)?')
mag = re.compile(r'\d+\.\d+|\d{2,}')
info = sum(1 for t in texts if mag.search(rule.sub('', t)))
bec = sum(1 for t in texts if 'BECAUSE' in t)
days = con.execute("SELECT day, COUNT(*) FROM memos WHERE "
                   "created_run='distiller' GROUP BY day ORDER BY day").fetchall()
print(f"bank[distiller]: {len(texts)} memos | data-bearing {info} | "
      f"BECAUSE {bec} | per-day {days}")

eps = [json.loads(l) for l in
       open(os.path.join(DOJO, 'gate_state', 'memo_run_distiller.jsonl'))
       if l.strip()]
print(f"episodes: {len(eps)}/16 done")
for e in eps:
    confs = [float(f['conf']) for f in e['frames']
             if f.get('conf') not in (None, '?')]
    carry = '' if e['day'] == '2025_04_08' else ' [DAY-CARRY]'
    print(f"  {e['episode_id']}{carry}: {e['n_frames']}f "
          f"exit={e['exit_frame']} retr={e.get('retrievals_used', 0)} "
          f"conf {min(confs) if confs else '?'}-{max(confs) if confs else '?'} "
          f"admitted={e['memos_written']}")
for e in eps:
    if e['day'] != '2025_04_08':
        for f in e['frames']:
            if f.get('retrieved_ids'):
                print("sample memory-informed reason:", f['reason'][:160])
                break
        break
