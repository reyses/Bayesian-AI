"""TG SEND + TRANSCRIPT — shared send helper that logs BOTH directions.

Root cause fixed (owner 2026-07-29, "we need a transcript of the conversation"):
outbound (assistant) messages were being sent via ad-hoc inline `requests.post`
calls scattered across sessions, with no log anywhere. Telegram's Bot API has
NO method to fetch a bot's own send history, so anything sent before this file
existed is NOT recoverable verbatim — only reconstructable from journal
summaries. Going forward, ALWAYS send through this module (send_text /
send_photo) so nothing is lost again.

Storage:
  - transcript.jsonl (this dir)      — append-only raw log, both directions
  - gate_state/session_transcript.db — SQLite mirror (table `transcript`),
    same DB the pocket-dojo corpus lives in style-wise but a separate table
    since this is conversation-level, not trade-event-level.

Usage:
  from tools.telegram_bridge.tg_send import send_text, send_photo, backfill_inbox
  send_text("hello")
  send_photo("/path/to.png", caption="...")
"""
import json
import os
import sqlite3
import time

import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
load_dotenv(os.path.join(HERE, '.env'))
load_dotenv(os.path.join(REPO, '.env'))
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

TRANSCRIPT_JSONL = os.path.join(HERE, 'state', 'transcript.jsonl')
DB = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state', 'session_transcript.db')
INBOX = os.path.join(HERE, 'state', 'inbox.jsonl')

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript (
    id        INTEGER PRIMARY KEY,
    ts        INTEGER,      -- unix epoch (owner-side: TG update ts; assistant-side: send time)
    wall      TEXT,         -- ISO wall-clock
    direction TEXT,         -- 'in' (owner->assistant) or 'out' (assistant->owner)
    kind      TEXT,         -- 'text' or 'photo'
    text      TEXT,         -- message text / photo caption
    meta      TEXT,         -- JSON: extra fields (update_id, files, caption path, etc.)
    UNIQUE(ts, direction, kind, text) ON CONFLICT IGNORE
);
"""


def _db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def _log(direction, kind, text, ts=None, meta=None):
    ts = ts if ts is not None else int(time.time())
    rec = dict(ts=ts, wall=time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(ts)),
               direction=direction, kind=kind, text=text, meta=meta or {})
    os.makedirs(os.path.dirname(TRANSCRIPT_JSONL), exist_ok=True)
    with open(TRANSCRIPT_JSONL, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    con = _db()
    con.execute(
        "INSERT OR IGNORE INTO transcript (ts, wall, direction, kind, text, meta) VALUES (?,?,?,?,?,?)",
        (rec['ts'], rec['wall'], direction, kind, text, json.dumps(rec['meta'])))
    con.commit()
    con.close()


def send_text(text, chat_id=None, retries=5, timeout=60):
    chat_id = chat_id or CHAT_ID
    for i in range(retries):
        try:
            r = requests.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',
                              data={'chat_id': chat_id, 'text': text}, timeout=timeout)
            if r.ok:
                # record the TG message_id: without it a sent message can
                # never be redacted (deleteMessage needs it, and bots have
                # no history API). Learned 2026-08-03 when a hindsight-
                # contaminated message could not be deleted before the
                # owner might read it.
                try:
                    mid = r.json()['result']['message_id']
                except Exception:
                    mid = None
                _log('out', 'text', text, meta={'message_id': mid})
                return True
        except Exception:
            if i == retries - 1:
                raise
        time.sleep(min(2 * (i + 1), 8))    # backoff -- rapid-fire retries hit
                                           # the same transient blip (observed bug)
    return False


def send_photo(path, caption='', chat_id=None, retries=5, timeout=90):
    chat_id = chat_id or CHAT_ID
    for i in range(retries):
        try:
            r = requests.post(f'https://api.telegram.org/bot{TOKEN}/sendPhoto',
                              data={'chat_id': chat_id, 'caption': caption[:1000]},
                              files={'photo': open(path, 'rb')}, timeout=timeout)
            if r.ok:
                _log('out', 'photo', caption, meta={'path': path})
                return True
        except Exception:
            if i == retries - 1:
                raise
        time.sleep(min(2 * (i + 1), 8))
    return False


def backfill_inbox():
    """One-time: pull owner's historical inbound messages from inbox.jsonl into
    the transcript (recoverable — outbound history before this file is NOT)."""
    n = 0
    if not os.path.exists(INBOX):
        print('no inbox.jsonl found'); return
    with open(INBOX) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            _log('in', 'text', rec.get('text', ''), ts=rec.get('ts'),
                 meta={'update_id': rec.get('update_id'), 'files': rec.get('files')})
            n += 1
    print(f'backfilled {n} inbound messages -> {TRANSCRIPT_JSONL} + {DB}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'backfill':
        backfill_inbox()
    else:
        print(__doc__)
