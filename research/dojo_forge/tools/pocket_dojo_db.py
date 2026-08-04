"""POCKET DOJO SQL CORPUS — SQLite mirror of pocket_dojo.py's JSONL logs
(owner 2026-07-29: "are we capturing all of this in a journal and SQL
database?"). The JSONL files (reports/human_dojo/pocket_<day>.jsonl) remain
the append-only source of truth; this is a queryable index over them.

Schema: ONE ROW PER EVENT (call/fill/close/note/owner_line/retcon/...),
day-partitioned, with a generic `payload` JSON column PLUS pulled-out common
fields (dir, price, pts, reason, text) so simple SQL doesn't need json_extract
for the 90% case.

Run standalone to (re)ingest all JSONL -> DB (idempotent — dedupes on
(day, wall, event, seq) so re-running after a live session just adds new rows):
  python research/dojo_forge/tools/pocket_dojo_db.py ingest
  python research/dojo_forge/tools/pocket_dojo_db.py query "select * from events where reason='close' limit 5"

pocket_dojo.py also calls `write_event()` directly on every _log() so the DB
stays current live, without needing a separate ingest step each session.
"""
import argparse
import glob
import json
import os
import sqlite3

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOGDIR = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'human_dojo')
DB = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state', 'pocket_dojo.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY,
    day     TEXT,
    wall    TEXT,        -- wall-clock timestamp of the action (ISO)
    bar     INTEGER,      -- 1m bar index at the time of the event
    event   TEXT,         -- new/step/call/fill/close/exit_req/note/owner_line/retcon_entry/retcon_close/...
    dir     TEXT,         -- long/short, when applicable
    price   REAL,
    pts     REAL,         -- realized points, for close events
    reason  TEXT,         -- target/stop/manual/scratch/theme/eod/reverse/...
    text    TEXT,         -- note text / retcon note / why
    payload TEXT,         -- full original JSON record (fallback for anything not pulled out)
    who     TEXT,         -- 'owner' or 'claude' (attribution — 60-owner-leg corpus target)
    slice   INTEGER,      -- global decision-point number ("S38") for backtracking
    UNIQUE(day, wall, event, bar, price) ON CONFLICT IGNORE
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE INDEX IF NOT EXISTS idx_events_event ON events(event);
"""


def _conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def write_event(rec: dict, con=None):
    """Insert one already-built log record (the same dict pocket_dojo._log
    writes to JSONL). Safe to call per-event for live write-through."""
    owns = con is None
    con = con or _conn()
    con.execute(
        "INSERT OR IGNORE INTO events (day, wall, bar, event, dir, price, pts, reason, text, payload, who, slice) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (rec.get('day'), rec.get('wall'), rec.get('bar'), rec.get('event'),
         rec.get('dir'), rec.get('price'), rec.get('pts'), rec.get('reason'),
         rec.get('text') or rec.get('note') or rec.get('why'),
         json.dumps(rec), rec.get('who', 'owner'), rec.get('slice')))
    con.commit()
    if owns:
        con.close()


def ingest_all():
    con = _conn()
    n = 0
    for path in sorted(glob.glob(os.path.join(LOGDIR, 'pocket_*.jsonl'))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                write_event(rec, con)
                n += 1
    con.close()
    print(f'ingested {n} lines -> {DB}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ingest', 'query'])
    ap.add_argument('sql', nargs='?')
    a = ap.parse_args()
    if a.cmd == 'ingest':
        ingest_all()
    else:
        con = _conn()
        con.row_factory = sqlite3.Row
        for row in con.execute(a.sql):
            print(dict(row))


if __name__ == '__main__':
    main()
