#!/usr/bin/env python3.11
"""Build the DERIVED SQLite + FTS5 mirror of the markdown memory/journal corpus.

    STRICTLY DERIVED. This database is rebuilt idempotently from the markdown
    files on every run (DROP + CREATE + INSERT). It is NEVER a source of truth
    and NEVER a write-target for content. The markdown files remain the only
    authoritative store; nothing here is ever edited back into them. The
    dual-copy MEMORY.md sync hook must never see a DB-born edit, because there
    are none — this script only reads the .md/.txt files and writes the .db.

Sources indexed (read-only):
  - docs/memory/*.md            (MEMORY.md split by ## section; every other
                                 detail file = one row; frontmatter `type` drives tier)
  - docs/memory/archive/*.md    (one row per archived file, tier=context)
  - docs/daily/INDEX.md         (one row per dated table line, tier=volatile)
  - docs/reference/RESEARCH_JOURNAL.txt (one row per dated entry, tier=volatile)

Schema:
  learnings(id, date, tier, tag, source_file, text)
  learnings_fts  -- FTS5 external-content mirror over `text` (content='learnings')

Usage:
  python3.11 tools/memory_loop/build_memory_db.py [--db PATH] [--quiet]

Run cadence: session end, or on demand before querying memory.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys
from datetime import date

# --- repo-root-relative paths (run from anywhere) ---------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))               # tools/memory_loop -> repo
MEM_DIR = os.path.join(REPO, "docs", "memory")
ARCHIVE_DIR = os.path.join(MEM_DIR, "archive")
INDEX_MD = os.path.join(REPO, "docs", "daily", "INDEX.md")
JOURNAL_TXT = os.path.join(REPO, "docs", "reference", "RESEARCH_JOURNAL.txt")
DEFAULT_DB = os.path.join(MEM_DIR, "memory.db")

DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")
# a dated entry in the free-text journal starts a line with a date, optionally [bracketed]
JOURNAL_ENTRY_RE = re.compile(r"^\[?(20\d\d-\d\d-\d\d)")
# an INDEX table row: | 2026-07-18 (...) | tags | text |   (also tolerate a leading "- 2026..")
INDEX_ROW_RE = re.compile(r"^\s*[|\-]\s*(20\d\d-\d\d-\d\d)")

# memory files that are "ways of working / identity" but carry no `type: feedback`
STABLE_BY_NAME = {
    "USER_PERSONA_AND_PROTOCOL.md",
    "AGENT_FEEDBACK_RULES.md",
    "ce_methodology.md",
}
# MEMORY.md ## sections that read as stable identity/convention (vs learned context)
STABLE_SECTION_KEYS = ("HARD RULES", "METRIC DEFINITION", "USER PROFILE")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def collapse(text: str) -> str:
    text = text.replace("﻿", "").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def frontmatter_type(text: str) -> str | None:
    """Return the `type:` value from YAML-ish frontmatter, if present."""
    if not text.startswith("---"):
        # some files carry frontmatter fields without a leading fence
        head = text[:600]
    else:
        end = text.find("\n---", 3)
        head = text[: end if end != -1 else 600]
    m = re.search(r"^\s*type:\s*(\w+)", head, re.MULTILINE)
    return m.group(1).lower() if m else None


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            return text[nl + 1 :] if nl != -1 else ""
    return text


def first_date(text: str, fallback_path: str | None = None) -> str:
    m = DATE_RE.search(text)
    if m:
        return m.group(1)
    if fallback_path and os.path.exists(fallback_path):
        ts = os.path.getmtime(fallback_path)
        return date.fromtimestamp(ts).isoformat()
    return "0000-00-00"


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:48]


def tier_for_memory_file(name: str, ftype: str | None) -> str:
    if ftype == "feedback":
        return "stable"
    if name in STABLE_BY_NAME:
        return "stable"
    return "context"


def build_rows() -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    counts: dict[str, int] = {}

    def add(date_, tier, tag, source_file, text):
        text = collapse(text)
        if not text:
            return
        rows.append(
            dict(date=date_, tier=tier, tag=tag, source_file=source_file, text=text)
        )

    # --- docs/memory/*.md (top-level, non-archive) --------------------------
    n = 0
    for path in sorted(glob.glob(os.path.join(MEM_DIR, "*.md"))):
        name = os.path.basename(path)
        rel = os.path.relpath(path, REPO).replace("\\", "/")
        raw = read_text(path)
        ftype = frontmatter_type(raw)

        if name == "MEMORY.md":
            # split by level-2 ## sections -> one row each (graveyard/§ blocks)
            body = strip_frontmatter(raw)
            parts = re.split(r"(?m)^(##\s+.*)$", body)
            # parts = [pre, header1, body1, header2, body2, ...]
            for i in range(1, len(parts), 2):
                header = parts[i].lstrip("# ").strip()
                sec_body = parts[i + 1] if i + 1 < len(parts) else ""
                section_text = parts[i] + "\n" + sec_body
                stier = (
                    "stable"
                    if any(k in header.upper() for k in STABLE_SECTION_KEYS)
                    else "context"
                )
                add(
                    first_date(sec_body, path),
                    stier,
                    "MEMORY#" + slug(header),
                    rel,
                    section_text,
                )
                n += 1
            continue

        body = strip_frontmatter(raw)
        add(
            first_date(body, path),
            tier_for_memory_file(name, ftype),
            os.path.splitext(name)[0],
            rel,
            body,
        )
        n += 1
    counts["docs/memory/*.md"] = n

    # --- docs/memory/archive/*.md ------------------------------------------
    n = 0
    for path in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.md"))):
        name = os.path.basename(path)
        rel = os.path.relpath(path, REPO).replace("\\", "/")
        body = strip_frontmatter(read_text(path))
        add(first_date(body, path), "context", "archive/" + os.path.splitext(name)[0], rel, body)
        n += 1
    counts["docs/memory/archive/*.md"] = n

    # --- docs/daily/INDEX.md (one row per dated line) -----------------------
    n = 0
    rel = os.path.relpath(INDEX_MD, REPO).replace("\\", "/")
    for line in read_text(INDEX_MD).splitlines():
        m = INDEX_ROW_RE.match(line)
        if not m:
            continue
        add(m.group(1), "volatile", m.group(1), rel, line)
        n += 1
    counts["docs/daily/INDEX.md"] = n

    # --- docs/reference/RESEARCH_JOURNAL.txt (one row per dated entry) -------
    n = 0
    rel = os.path.relpath(JOURNAL_TXT, REPO).replace("\\", "/")
    cur_date: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal n
        if cur_date and buf:
            add(cur_date, "volatile", cur_date, rel, "\n".join(buf))
            n += 1

    for line in read_text(JOURNAL_TXT).splitlines():
        m = JOURNAL_ENTRY_RE.match(line)
        if m:
            flush()
            cur_date = m.group(1)
            buf = [line]
        elif cur_date is not None:
            buf.append(line)
    flush()
    counts["docs/reference/RESEARCH_JOURNAL.txt"] = n

    return rows, counts


def write_db(rows: list[dict], db_path: str) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)  # strictly derived -> rebuild from scratch
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE learnings (
            id          INTEGER PRIMARY KEY,
            date        TEXT NOT NULL,
            tier        TEXT NOT NULL,
            tag         TEXT,
            source_file TEXT NOT NULL,
            text        TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE learnings_fts
            USING fts5(text, content='learnings', content_rowid='id');
        """
    )
    cur.executemany(
        "INSERT INTO learnings(date,tier,tag,source_file,text) VALUES(?,?,?,?,?)",
        [(r["date"], r["tier"], r["tag"], r["source_file"], r["text"]) for r in rows],
    )
    # populate the external-content FTS index from the base table
    cur.execute("INSERT INTO learnings_fts(learnings_fts) VALUES('rebuild')")
    con.commit()
    con.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build derived memory FTS mirror.")
    ap.add_argument("--db", default=DEFAULT_DB, help="output .db path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rows, counts = build_rows()
    write_db(rows, args.db)

    if not args.quiet:
        print(f"DERIVED memory DB rebuilt: {args.db}")
        print(f"total rows: {len(rows)}")
        for src, c in counts.items():
            print(f"  {src:42s} {c:5d} rows")
        by_tier: dict[str, int] = {}
        for r in rows:
            by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        print("by tier: " + ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
