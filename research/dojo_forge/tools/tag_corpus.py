"""Day-tag the Telegram corpus by joining it to the dojo event log on
wall-clock (owner 2026-08-04: "we added markers for which day and session
didn't we?" — we did, but only on the DOJO log; the conversation corpus has
no day/slice, which is why his directional theses could not be scored).

Every dojo event carries (wall, day, slice, bar). Every message carries wall.
A message belongs to whatever sim day/slice was current when it was sent, so
each message inherits the tags of the most recent PRECEDING dojo event.

Messages before any dojo event, or more than STALE_H hours after the last one,
are tagged day=None — an honest "not during a session" rather than a guess.

  python research/dojo_forge/tools/tag_corpus.py
Writes research/dojo_forge/reports/corpus_tagged.parquet + a summary.
"""
import json
import os
import sqlite3

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
TRANS = os.path.join(REPO, 'tools', 'telegram_bridge', 'state',
                     'transcript.jsonl')
DB = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state',
                  'pocket_dojo.db')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                   'corpus_tagged.parquet')
STALE_H = 3.0        # a message this long after the last dojo event is not
                     # part of that session


def main():
    msgs = [json.loads(l) for l in open(TRANS) if l.strip()]
    m = pd.DataFrame(msgs)
    # Transcript walls are tz-AWARE local; dojo walls are NAIVE local.
    # Parsing the dojo side as UTC shifted every event 7h and threw the join
    # onto the wrong sessions (it tagged our 2024_09_16 work as 2025_12_19).
    # Normalise BOTH to naive local wall-clock.
    # Transcript walls are MIXED: some carry a -07:00 offset, some are
    # naive local. Parsing with utc=True shifted the naive ones by 7h and
    # scattered the join across the wrong sessions. Both logs are written in
    # local wall-clock, so take the first 19 chars and compare like for like.
    m['wall_dt'] = pd.to_datetime(m['wall'].str.slice(0, 19),
                                  format='%Y-%m-%dT%H:%M:%S')
    m = m.sort_values('wall_dt').reset_index(drop=True)

    con = sqlite3.connect(DB)
    ev = pd.read_sql('select day, slice, bar, event, wall from events '
                     'order by wall', con)
    ev['wall_dt'] = pd.to_datetime(ev['wall'].str.slice(0, 19),
                                   format='%Y-%m-%dT%H:%M:%S')
    ev = ev.dropna(subset=['wall_dt']).sort_values('wall_dt')

    tagged = pd.merge_asof(m, ev[['wall_dt', 'day', 'slice', 'bar', 'event']],
                           on='wall_dt', direction='backward',
                           suffixes=('', '_ev'))
    gap = (tagged['wall_dt'] - tagged['wall_dt'].where(
        tagged['day'].notna())).dt.total_seconds()
    # blank the tag when the nearest preceding event is stale
    last_ev = pd.merge_asof(m[['wall_dt']], ev[['wall_dt']].assign(e=ev['wall_dt']),
                            on='wall_dt', direction='backward')['e']
    stale = (tagged['wall_dt'] - last_ev).dt.total_seconds() > STALE_H * 3600
    tagged.loc[stale, ['day', 'slice', 'bar']] = None
    tagged['in_session'] = tagged['day'].notna()
    tagged[['wall', 'direction', 'kind', 'text', 'day', 'slice', 'bar',
            'in_session']].to_parquet(OUT, index=False)

    own = tagged[tagged['direction'] == 'in']
    print(f'{len(tagged)} messages | {len(own)} from the owner')
    print(f'tagged to a sim day: {int(tagged["in_session"].sum())} '
          f'({tagged["in_session"].mean():.0%})')
    print('\nowner messages per sim day:')
    print(own[own['in_session']].groupby('day').size().to_string())
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
