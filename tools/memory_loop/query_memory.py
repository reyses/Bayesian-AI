#!/usr/bin/env python3.11
"""Query the DERIVED memory FTS mirror. Read-only. stdlib sqlite3 only.

  python3.11 tools/memory_loop/query_memory.py "<fts terms>" [--limit N] [--tier T] [--db PATH]

Prints one hit per line:  date | tier | tag | text
Uses FTS5 MATCH ranked by relevance. Multi-word queries are implicit-AND
(FTS5 default). Use quotes for phrases, OR / NOT for boolean, prefix* for stems.

The DB is strictly derived from the markdown files (see build_memory_db.py);
if it is missing or stale, rebuild it first.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

try:  # ensure UTF-8 stdout on Windows (cp1252 default chokes on em-dash/BOM)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DB = os.path.join(REPO, "docs", "memory", "memory.db")
DISPLAY_CHARS = 240  # truncate long rows so output stays one-per-line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FTS query over derived memory DB.")
    ap.add_argument("query", help="FTS5 MATCH expression, e.g. \"worker background\"")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--tier", choices=["stable", "context", "volatile"], default=None)
    ap.add_argument("--tag", default=None,
                    help="restrict to tags with this prefix, e.g. comms: / code: / report:")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--full", action="store_true", help="do not truncate text")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"[query_memory] DB not found: {args.db}", file=sys.stderr)
        print("[query_memory] rebuild it: python3.11 tools/memory_loop/build_memory_db.py",
              file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    sql = (
        "SELECT l.date, l.tier, l.tag, l.text "
        "FROM learnings_fts f JOIN learnings l ON l.id = f.rowid "
        "WHERE learnings_fts MATCH ? "
    )
    params: list = [args.query]
    if args.tier:
        sql += "AND l.tier = ? "
        params.append(args.tier)
    if args.tag:
        sql += "AND l.tag LIKE ? "
        params.append(args.tag + "%")
    sql += "ORDER BY rank LIMIT ?"
    params.append(args.limit)

    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as exc:
        print(f"[query_memory] bad FTS query: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print(f"(no hits for: {args.query})")
        return 0

    for d, tier, tag, text in rows:
        if not args.full and len(text) > DISPLAY_CHARS:
            text = text[:DISPLAY_CHARS] + " …"
        print(f"{d} | {tier:8s} | {tag} | {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
