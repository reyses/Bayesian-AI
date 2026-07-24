#!/usr/bin/env python3
"""TEACHER MEMORY BANK — the cross-episode journal, SQL form factor (docs 149 + 151).

The teacher's long-term disk memory: entries are the SOURCE OF TRUTH in a SQLite
table, mirrored into an FTS5 external-content index for retrieval. Per-frame the
runner builds a DETERMINISTIC query from the NOW frame's state (giveback bucket,
1m price-velocity sign, leg-age bucket), pulls the top-k memos into a RELEVANT
MEMORY block, and the teacher decides with that context injected. Context stays
flat regardless of how much the teacher has learned.

GUARD v2 — FULLY MECHANICAL, three layers, zero judgment (doc 151 §"GUARD v2"):
  (a) ADMISSION  : writes are accepted ONLY for days in an explicit allowlist
                   passed to this bank. A day not on the list NEVER INGESTS —
                   protection by construction; there is nothing to leak (doc 151
                   layer 1 "store admission", generalized to any non-allowlisted
                   day, which is how lockbox/burned days are kept out).
  (b) LOOKBACK   : retrieval returns ONLY memos whose day is STRICTLY BEFORE the
                   querying episode's day AND within N=LOOKBACK_DAYS calendar days
                   behind it (doc 151 layer 2 "lookback cap", default N=10). No
                   same-day cross-episode reads — v1 semantics are prior-day only
                   (doc 149 MEMO protocol: cross-episode journal, walk-forward).

Every retrieval AND every write is appended to memory_ledger.jsonl — the audit
trail and the leakage proof for the gate record (doc 151 "Request/grant ledger").

snapshot()/restore() copy the DB file for the kill-switch / OOS copy-on-write
branch (doc 149 "snapshot isolation").

stdlib only (sqlite3 ships with FTS5 in the repo's build — verified 3.46.1).
"""
import datetime
import json
import os
import re
import shutil
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
GATE_STATE_DIR = os.path.join(DOJO_ROOT, 'gate_state')

DEFAULT_DB = os.path.join(GATE_STATE_DIR, 'teacher_memory.db')
DEFAULT_LEDGER = os.path.join(GATE_STATE_DIR, 'memory_ledger.jsonl')

# --- retrieval constants (doc 151 GUARD v2; owner-ratified defaults) ----------
LOOKBACK_DAYS = 10   # retrieval window: memos up to this many calendar days behind
                     # the querying episode's day (doc 151 layer 2, default N=10).
TOP_K = 3            # memos pulled per retrieval (doc 149 "fixed k"; mirrors the
                     # top-3 RELEVANT MEMORY block the runner injects).

# --- deterministic state-bucket thresholds (the query vocabulary) -------------
# The NOW frame is bucketed into a SMALL fixed token vocabulary. A memo is STORED
# tagged with the buckets of the frame it was formed on (prefixing its text), so
# FTS5 retrieval-by-state-similarity is exact and reproducible. Tokens are single
# alphanumeric words (no separators) so the unicode61 tokenizer keeps them intact.
GIVEBACK_LO_PCT = 20   # < this -> 'gblow'  (shallow retrace)
GIVEBACK_HI_PCT = 50   # >= this -> 'gbhigh' (deep giveback); between -> 'gbmid'
LEG_YOUNG_MAX_MIN = 3  # leg age <= this -> 'legyoung'
LEG_OLD_MIN_MIN = 8    # leg age >  this -> 'legold'; between -> 'legmid'

_RE_GIVEBACK = re.compile(r'giveback\s+(\d+)%')
_RE_LEGAGE = re.compile(r'leg age\s+(\d+)m')
# first [1m] line carrying an instantaneous price velocity (sign is the bucket)
_RE_PV1M = re.compile(r'\[1m\][^\n]*price_velocity_1b=([+-][\d.]+)')

# state-tag prefix marker in stored memo text (kept human-readable + FTS-matchable)
_STATE_PREFIX = "state:"


def state_tags(now_frame_text):
    """Extract the fixed deterministic bucket tokens from a NOW-frame's text.

    Returns a list like ['gbmid', 'pvdn', 'legmid'] — the SAME function tags a
    memo at write time and builds the OR-query at read time, so a memo formed in
    state S is exactly the memo an identical state S retrieves. Missing fields are
    simply omitted (never guessed); the local: line always yields giveback + leg.
    """
    tags = []
    m = _RE_GIVEBACK.search(now_frame_text)
    if m:
        gb = int(m.group(1))
        tags.append('gblow' if gb < GIVEBACK_LO_PCT
                    else 'gbhigh' if gb >= GIVEBACK_HI_PCT else 'gbmid')
    m = _RE_PV1M.search(now_frame_text)
    if m:
        pv = float(m.group(1))
        tags.append('pvup' if pv > 0 else 'pvdn' if pv < 0 else 'pvflat')
    m = _RE_LEGAGE.search(now_frame_text)
    if m:
        age = int(m.group(1))
        tags.append('legyoung' if age <= LEG_YOUNG_MAX_MIN
                    else 'legold' if age > LEG_OLD_MIN_MIN else 'legmid')
    return tags


def build_query(now_frame_text):
    """Deterministic FTS OR-query from a NOW frame's fixed state buckets.

    Returns (query_string, tags). E.g. tags ['gbmid','pvdn','legmid'] ->
    '"gbmid" OR "pvdn" OR "legmid"'. Quoting each token makes it a phrase so the
    FTS parser never treats a token as an operator. bm25 then ranks memos by how
    many (and how rare) the shared state buckets are; rowid ASC breaks ties.
    """
    tags = state_tags(now_frame_text)
    query = " OR ".join(f'"{t}"' for t in tags) if tags else '"nostate"'
    return query, tags


def _day_str_to_date(day):
    """'2025_04_08' -> datetime.date(2025, 4, 8)."""
    y, mth, d = day.split('_')
    return datetime.date(int(y), int(mth), int(d))


def _tag_memo_text(memo_text, tags):
    """Store a memo prefixed with its state tags so FTS matches it by state.

    Kept human-readable AND injected verbatim into RELEVANT MEMORY — the state
    context ('state: gbmid pvdn legmid') is itself useful signal to the teacher.
    """
    return f"({_STATE_PREFIX} {' '.join(tags)}) {memo_text.strip()}" if tags \
        else memo_text.strip()


class TeacherMemory:
    """SQLite-backed teacher journal with the mechanical GUARD v2 enforced in code."""

    def __init__(self, db_path=DEFAULT_DB, ledger_path=DEFAULT_LEDGER,
                 write_allowlist=None, run_tag='', lookback_days=LOOKBACK_DAYS,
                 top_k=TOP_K):
        self.db_path = db_path
        self.ledger_path = ledger_path
        # ADMISSION allowlist: the ONLY days whose memos may be written (guard a).
        self.write_allowlist = set(write_allowlist or [])
        self.run_tag = run_tag
        self.lookback_days = lookback_days
        self.top_k = top_k
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Rollback-journal (default) mode: a single .db file is a consistent unit
        # to copy for snapshot()/restore() when no transaction is open.
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # -- schema ---------------------------------------------------------------
    def _ensure_schema(self):
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS memos(
                id          INTEGER PRIMARY KEY,
                episode_id  TEXT,
                day         TEXT,
                minute      INTEGER,
                text        TEXT,
                created_run TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memos_fts
                USING fts5(text, content='memos', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memos_ai AFTER INSERT ON memos BEGIN
                INSERT INTO memos_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memos_ad AFTER DELETE ON memos BEGIN
                INSERT INTO memos_fts(memos_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memos_au AFTER UPDATE ON memos BEGIN
                INSERT INTO memos_fts(memos_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
                INSERT INTO memos_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        c.commit()

    # -- ledger ---------------------------------------------------------------
    def _ledger(self, event):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        event = dict(event)
        event.setdefault('ts', time.time())
        event.setdefault('run', self.run_tag)
        with open(self.ledger_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
            f.flush()
            os.fsync(f.fileno())

    # -- writes (guard a: ADMISSION) ------------------------------------------
    def write_memo(self, episode_id, day, minute, memo_text, now_frame_text=''):
        """Insert a memo IFF its day is on the admission allowlist.

        Returns dict(admitted: bool, reason, id). A rejected write ingests NOTHING
        (protection by construction) and is ledgered. The stored text is prefixed
        with the frame's state tags (see _tag_memo_text) for retrieval-by-state.
        """
        if day not in self.write_allowlist:
            self._ledger(dict(event='write_rejected', episode_id=episode_id,
                              day=day, minute=minute, reason='not_in_allowlist'))
            return dict(admitted=False, reason='not_in_allowlist', id=None)
        tags = state_tags(now_frame_text)
        stored = _tag_memo_text(memo_text, tags)
        cur = self.conn.execute(
            "INSERT INTO memos(episode_id, day, minute, text, created_run) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, day, minute, stored, self.run_tag))
        self.conn.commit()
        mid = cur.lastrowid
        self._ledger(dict(event='write_admitted', episode_id=episode_id, day=day,
                          minute=minute, memo_id=mid, tags=tags))
        return dict(admitted=True, reason='admitted', id=mid)

    # -- reads (guards b: LOOKBACK + strict prior-day) ------------------------
    def _allowed_days(self, episode_day):
        """Days eligible for retrieval: strictly BEFORE episode_day and within
        lookback_days calendar days behind it (guard b). Computed from the days
        actually present in the DB so the SQL IN-list stays small + exact."""
        ep = _day_str_to_date(episode_day)
        rows = self.conn.execute("SELECT DISTINCT day FROM memos").fetchall()
        allowed = []
        for r in rows:
            try:
                d = _day_str_to_date(r['day'])
            except Exception:  # noqa: BLE001 - malformed day never becomes eligible
                continue
            delta = (ep - d).days
            if 1 <= delta <= self.lookback_days:   # strictly before AND within N
                allowed.append(r['day'])
        return allowed

    def retrieve(self, now_frame_text, episode_id, episode_day, minute):
        """Top-k memos for the NOW frame under the mechanical guards.

        Deterministic: fixed query template + fixed k + bm25 rank with rowid ASC
        tiebreak. Same query on the same DB always returns the same ids. Every
        retrieval (query + granted ids) is appended to the ledger.
        """
        query, tags = build_query(now_frame_text)
        allowed = self._allowed_days(episode_day)
        granted = []
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            sql = (
                "SELECT m.id AS id, m.episode_id AS episode_id, m.day AS day, "
                "       m.minute AS minute, m.text AS text, bm25(memos_fts) AS score "
                "FROM memos_fts JOIN memos m ON m.id = memos_fts.rowid "
                f"WHERE memos_fts MATCH ? AND m.day IN ({placeholders}) "
                "ORDER BY score ASC, m.id ASC LIMIT ?"
            )
            rows = self.conn.execute(sql, [query, *allowed, self.top_k]).fetchall()
            granted = [dict(id=r['id'], episode_id=r['episode_id'], day=r['day'],
                            minute=r['minute'], text=r['text']) for r in rows]
        self._ledger(dict(event='retrieve', episode_id=episode_id,
                          episode_day=episode_day, minute=minute, query=query,
                          tags=tags, allowed_days=allowed,
                          granted_ids=[g['id'] for g in granted]))
        return granted

    # -- snapshot / restore (kill-switch, OOS branch) -------------------------
    def snapshot(self, dest_path):
        """Copy the DB file to dest_path (copy-on-write branch / kill-switch)."""
        self.conn.commit()
        shutil.copy2(self.db_path, dest_path)
        return dest_path

    def restore(self, src_path):
        """Replace the live DB with a snapshot and reopen the connection."""
        self.conn.close()
        shutil.copy2(src_path, self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self):
        self.conn.close()


def _selftest():
    """Standalone mechanical selftest of the guards (no LLM). Runs on a temp DB."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix='teachermem_')
    db = os.path.join(tmp, 'tm.db')
    ledger = os.path.join(tmp, 'ledger.jsonl')
    # episode day D; allowlist admits D and D-1 (seed the prior day too).
    D, Dm1, Dm5 = '2025_04_08', '2025_04_07', '2025_04_03'
    now = ("local: px +0.00pts | leg age 4m amp 87.2pts giveback 31% | "
           "[1m] closed-bar price_velocity_1b=-50.500")
    mem = TeacherMemory(db_path=db, ledger_path=ledger,
                        write_allowlist={D, Dm1}, run_tag='selftest')
    ok = True
    # (1) ADMISSION: allowlisted day admitted; non-allowlisted day REJECTED.
    a1 = mem.write_memo('epP', D, 3, 'same-day memo', now)
    a2 = mem.write_memo('epQ', Dm1, 5, 'prior-day memo', now)
    a3 = mem.write_memo('epR', Dm5, 2, 'out-of-allowlist memo', now)
    ok &= a1['admitted'] and a2['admitted'] and (not a3['admitted'])
    print(f"[selftest] admission: D admitted={a1['admitted']} "
          f"Dm1 admitted={a2['admitted']} Dm5 REJECTED={not a3['admitted']}")
    # (2) LOOKBACK: retrieving for an episode on D returns the prior-day memo,
    #     NOT the same-day memo (strictly-before rule).
    got = mem.retrieve(now, 'epNow', D, 4)
    ids = [g['id'] for g in got]
    days = {g['day'] for g in got}
    lookback_ok = (Dm1 in days) and (D not in days)
    ok &= lookback_ok
    print(f"[selftest] lookback: granted days={sorted(days)} "
          f"(prior-day in, same-day out) -> {lookback_ok}")
    # (3) DETERMINISM: same query twice -> identical ids.
    got2 = mem.retrieve(now, 'epNow', D, 4)
    det_ok = ids == [g['id'] for g in got2]
    ok &= det_ok
    print(f"[selftest] determinism: {ids} == {[g['id'] for g in got2]} -> {det_ok}")
    # (4) LEDGER: writes + retrievals appended.
    with open(ledger) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    events = [l['event'] for l in lines]
    ledger_ok = ('write_admitted' in events and 'write_rejected' in events
                 and events.count('retrieve') == 2)
    ok &= ledger_ok
    print(f"[selftest] ledger: {len(lines)} events {events} -> {ledger_ok}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'} (temp db {db})")
    mem.close()
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
