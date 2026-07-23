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
  --- full-corpus layer (doc 123, 2026-07-18) --------------------------------
  - research/nt8_catalog/comms/*.md          (per ## section; tag=comms:NNN)
  - research/**/reports/**/*.md              (per ## section; tag=report:<topic>)
  - docs/daily/*.md  (full journals)         (per ## section; tag=journal:<date>)
  - docs/{northstar,nt8,Active}/*.md + singletons (per ## section; tag=doc:<name>)
  - research/dojo_forge/*_SPEC.md            (per ## section; tag=spec:<name>)
  --- code layer (scope-ext, 2026-07-18) -------------------------------------
  - **/*.py across core_v2/ training/ live/ tools/ DATA/pipeline/
    research/*/{pipeline,tools,builders}/ research/dojo_forge/
    (ONE row per module: AST-extracted docstring + signatures + IO paths;
     tag=code:<repo-rel-path>; pure mechanical extraction, no model text)

Schema:
  learnings(id, date, tier, tag, source_file, text)
  learnings_fts  -- FTS5 external-content mirror over `text` (content='learnings')

Usage:
  python3.11 tools/memory_loop/build_memory_db.py [--db PATH] [--quiet]
                                                  [--sources a,b,c]

Run cadence: session end, or on demand before querying memory.
"""
from __future__ import annotations

import argparse
import ast
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

# code-layer limits
CODE_MAX_BYTES = 300 * 1024          # skip generated/huge modules (>300KB)
CODE_DOC_MAX_LINES = 40              # module docstring: first N lines only
CODE_MAX_IO_PATHS = 20              # deduped path-like literals per module

DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")
# a dated entry in the free-text journal starts a line with a date, optionally [bracketed]
JOURNAL_ENTRY_RE = re.compile(r"^\[?(20\d\d-\d\d-\d\d)")
# an INDEX table row: | 2026-07-18 (...) | tags | text |   (also tolerate a leading "- 2026..")
INDEX_ROW_RE = re.compile(r"^\s*[|\-]\s*(20\d\d-\d\d-\d\d)")
# leading NNN in a comms filename
COMMS_NUM_RE = re.compile(r"^(\d{3,})")
# a string literal that looks like a filesystem path / IO target
_IO_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,6}(?:$|[/\\'\"])")
_IO_KEY_RE = re.compile(r"DATA|checkpoints|reports")

# memory files that are "ways of working / identity" but carry no `type: feedback`
STABLE_BY_NAME = {
    "USER_PERSONA_AND_PROTOCOL.md",
    "AGENT_FEEDBACK_RULES.md",
    "ce_methodology.md",
}
# MEMORY.md ## sections that read as stable identity/convention (vs learned context)
STABLE_SECTION_KEYS = ("HARD RULES", "METRIC DEFINITION", "USER PROFILE")

# canonical source-class keys (for --sources partial rebuilds + coverage table)
ALL_SOURCES = [
    "memory", "archive", "index", "journal_txt",   # original layer
    "comms", "reports", "daily", "docs", "specs",   # full-corpus layer (doc 123)
    "code",                                          # code layer (scope-ext)
]


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


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (title, section_text) on level-2 ## headers.

    Preamble text before the first ## becomes its own section (titled from the
    leading # H1, else its first words). A doc with no ## headers falls back to
    a single whole-doc section. Mirrors the MEMORY.md handler's ^## split.
    """
    parts = re.split(r"(?m)^(##\s+.*)$", body)
    sections: list[tuple[str, str]] = []
    pre = parts[0]
    if collapse(pre):
        m = re.search(r"(?m)^#\s+(.*)$", pre)
        title = m.group(1).strip() if m else (collapse(pre)[:48] or "preamble")
        sections.append((title, pre))
    for i in range(1, len(parts), 2):
        header = parts[i].lstrip("# ").strip()
        sec_body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, parts[i] + "\n" + sec_body))
    if not sections:
        sections.append(("doc", body))
    return sections


# --- code-layer AST extraction ---------------------------------------------
def _signature(node: ast.AST) -> str:
    """Verbatim signature line for a top-level class/function (AST-derived)."""
    if isinstance(node, ast.ClassDef):
        try:
            bases = [ast.unparse(b) for b in node.bases]
        except Exception:
            bases = []
        return "class " + node.name + (("(" + ", ".join(bases) + ")") if bases else "")
    kind = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    try:
        argstr = ast.unparse(node.args)
    except Exception:
        argstr = "..."
    ret = ""
    if getattr(node, "returns", None) is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:
            ret = ""
    return f"{kind}{node.name}({argstr}){ret}"


def extract_code_surface(path: str) -> str | None:
    """AST + regex extraction of a module's knowledge surface. VERBATIM only.

    Returns assembled text, or None if the file fails to parse (caller counts
    it as a parse-failure). No model-generated text anywhere.
    """
    src = read_text(path)
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None

    lines: list[str] = []

    # 1. module docstring (first CODE_DOC_MAX_LINES lines)
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        lines.append("\n".join(mod_doc.splitlines()[:CODE_DOC_MAX_LINES]))

    # 2. every top-level class / function signature + first docstring line
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sig = _signature(node)
            doc = ast.get_docstring(node)
            first = doc.splitlines()[0].strip() if doc else ""
            lines.append(sig + (("  # " + first) if first else ""))

    # 3. path-like string literals (the module's IO surface), deduped, capped
    io_paths: list[str] = []
    seen_io: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if not s or len(s) > 200:
                continue
            has_sep = "/" in s or "\\" in s
            if not has_sep:
                continue
            if _IO_EXT_RE.search(s) or _IO_KEY_RE.search(s):
                if s not in seen_io:
                    seen_io.add(s)
                    io_paths.append(s)
    if io_paths:
        lines.append("IO: " + " ; ".join(io_paths[:CODE_MAX_IO_PATHS]))

    return "\n".join(lines).strip()


def iter_code_files() -> list[str]:
    """Collect module paths for the code layer (deduped, filtered)."""
    globs: list[str] = []
    for r in ("core_v2", "training", "live", "tools", os.path.join("DATA", "pipeline")):
        globs.append(os.path.join(REPO, r, "**", "*.py"))
    for sub in ("pipeline", "tools", "builders"):
        globs.append(os.path.join(REPO, "research", "*", sub, "**", "*.py"))
    globs.append(os.path.join(REPO, "research", "dojo_forge", "**", "*.py"))

    found: dict[str, str] = {}   # abspath -> original path (dedupe across globs)
    for g in globs:
        for p in glob.glob(g, recursive=True):
            norm = p.replace("\\", "/")
            if "/research/archive/" in norm:
                continue
            if any(x in norm for x in ("__pycache__", "/venv/", "/.venv/", "node_modules")):
                continue
            ap = os.path.abspath(p)
            if ap in found:
                continue
            try:
                if os.path.getsize(p) > CODE_MAX_BYTES:
                    continue
            except OSError:
                continue
            found[ap] = p
    return sorted(found.values())


def report_topic(rel: str) -> str:
    """Extract <topic> from a research report path (handles research/archive/)."""
    tail = rel.split("/research/", 1)[-1] if "/research/" in rel else rel
    tail = tail.split("research/", 1)[-1]
    if tail.startswith("archive/"):
        tail = tail[len("archive/"):]
    return tail.split("/", 1)[0] if "/" in tail else tail


def build_rows(sources: set[str] | None = None) -> tuple[list[dict], dict]:
    """Assemble rows. `sources` limits which source classes run (partial rebuild).

    Returns (rows, stats) where stats[key] = {"rows": n, "bytes": b} plus a
    special stats["_code_parse_failures"] counter.
    """
    def want(key: str) -> bool:
        return sources is None or key in sources

    rows: list[dict] = []
    stats: dict = {}
    seen_sections: set[tuple[str, str]] = set()   # (source_file, section-slug) dedupe

    def add(date_, tier, tag, source_file, text) -> int:
        """Append one row; returns collapsed char-length (0 if empty/skipped)."""
        text = collapse(text)
        if not text:
            return 0
        rows.append(
            dict(date=date_, tier=tier, tag=tag, source_file=source_file, text=text)
        )
        return len(text)

    def bump(key: str, added: int) -> None:
        st = stats.setdefault(key, {"rows": 0, "bytes": 0})
        if added:
            st["rows"] += 1
            st["bytes"] += added

    def ingest_sectioned(path, rel, tier, tag, key, date_hint=None):
        """Split a markdown doc into ## sections; one row each, deduped."""
        body = strip_frontmatter(read_text(path))
        for idx, (title, sec_text) in enumerate(split_sections(body)):
            sslug = slug(title) or f"s{idx}"
            dkey = (rel, sslug)
            if dkey in seen_sections:
                sslug = f"{sslug}-{idx}"
                dkey = (rel, sslug)
                if dkey in seen_sections:
                    continue
            d = date_hint or first_date(sec_text, path)
            added = add(d, tier, tag, rel, sec_text)
            if added:
                seen_sections.add(dkey)
                bump(key, added)

    # --- docs/memory/*.md (top-level, non-archive) --------------------------
    # + docs/memory/agents/<agent>/*.md — per-agent SEGMENTS (2026-07-18:
    #   full segmentation, not mirror: each agent owns its segment, everyone
    #   reads all; tag carries the segment owner for scoped retrieval).
    if want("memory"):
        seg_paths = sorted(glob.glob(os.path.join(MEM_DIR, "agents", "*", "*.md")))
        # research distillation cards (doc 119 swarm): one card per topic flows
        # into shared recall (tier=context, tag=distilled:<topic>)
        dist_paths = sorted(
            glob.glob(os.path.join(REPO, "research", "*", "DISTILLED.md"))
            + glob.glob(os.path.join(REPO, "research", "archive", "*", "DISTILLED.md")))
        for path in sorted(glob.glob(os.path.join(MEM_DIR, "*.md"))) + seg_paths + dist_paths:
            name = os.path.basename(path)
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            raw = read_text(path)
            ftype = frontmatter_type(raw)

            if name == "MEMORY.md" and "/agents/" not in path.replace("\\", "/"):
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
                    added = add(
                        first_date(sec_body, path),
                        stier,
                        "MEMORY#" + slug(header),
                        rel,
                        section_text,
                    )
                    bump("memory", added)
                continue

            body = strip_frontmatter(raw)
            # segment-owned file? prefix the tag with the owning agent (ag:, claude:)
            tag = os.path.splitext(name)[0]
            norm = path.replace("\\", "/")
            if "/agents/" in norm:
                owner = norm.split("/agents/")[1].split("/")[0]
                tag = f"{owner}:{tag}"
            elif norm.endswith("/DISTILLED.md") and "/research/" in norm:
                topic = norm.split("/research/")[1].split("/")[0]
                tag = f"distilled:{topic}"
            added = add(
                first_date(body, path),
                tier_for_memory_file(name, ftype),
                tag,
                rel,
                body,
            )
            bump("memory", added)

    # --- docs/memory/archive/*.md ------------------------------------------
    if want("archive"):
        for path in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.md"))):
            name = os.path.basename(path)
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            body = strip_frontmatter(read_text(path))
            added = add(first_date(body, path), "context",
                        "archive/" + os.path.splitext(name)[0], rel, body)
            bump("archive", added)

    # --- docs/daily/INDEX.md (one row per dated line) -----------------------
    if want("index"):
        rel = os.path.relpath(INDEX_MD, REPO).replace("\\", "/")
        for line in read_text(INDEX_MD).splitlines():
            m = INDEX_ROW_RE.match(line)
            if not m:
                continue
            added = add(m.group(1), "volatile", m.group(1), rel, line)
            bump("index", added)

    # --- docs/reference/RESEARCH_JOURNAL.txt (one row per dated entry) -------
    if want("journal_txt"):
        rel = os.path.relpath(JOURNAL_TXT, REPO).replace("\\", "/")
        cur_date: str | None = None
        buf: list[str] = []

        def flush():
            if cur_date and buf:
                added = add(cur_date, "volatile", cur_date, rel, "\n".join(buf))
                bump("journal_txt", added)

        for line in read_text(JOURNAL_TXT).splitlines():
            m = JOURNAL_ENTRY_RE.match(line)
            if m:
                flush()
                cur_date = m.group(1)
                buf = [line]
            elif cur_date is not None:
                buf.append(line)
        flush()

    # === full-corpus layer (doc 123) ========================================
    # 1. comms — the program's decision spine (per ## section; tag=comms:NNN)
    if want("comms"):
        for path in sorted(glob.glob(os.path.join(
                REPO, "research", "nt8_catalog", "comms", "*.md"))):
            name = os.path.basename(path)
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            m = COMMS_NUM_RE.match(name)
            tag = f"comms:{m.group(1)}" if m else f"comms:{os.path.splitext(name)[0]}"
            ingest_sectioned(path, rel, "context", tag, "comms")

    # 2. research reports (SKIP raw_articles* + assets/) ; tag=report:<topic>
    if want("reports"):
        report_paths = (
            glob.glob(os.path.join(REPO, "research", "*", "reports", "**", "*.md"),
                      recursive=True)
            + glob.glob(os.path.join(REPO, "research", "archive", "*", "reports",
                                     "**", "*.md"), recursive=True)
        )
        for path in sorted(set(report_paths)):
            norm = path.replace("\\", "/")
            if "raw_articles" in norm or "/assets/" in norm:
                continue
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            tag = f"report:{report_topic(rel)}"
            ingest_sectioned(path, rel, "context", tag, "reports")

    # 3. docs/daily/*.md full journals (per ## section; tag=journal:<date>)
    if want("daily"):
        for path in sorted(glob.glob(os.path.join(REPO, "docs", "daily", "*.md"))):
            name = os.path.basename(path)
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            m = DATE_RE.search(name)
            if not m:
                continue   # skip INDEX.md / README.md / TIMELINE.md (non-dated)
            dstr = m.group(1)
            ingest_sectioned(path, rel, "volatile", f"journal:{dstr}", "daily",
                             date_hint=dstr)

    # 4. governing docs / roadmaps / whitepaper / index ; tag=doc:<name>
    if want("docs"):
        doc_paths: list[str] = []
        for sub in ("northstar", "nt8", "Active"):
            doc_paths += glob.glob(os.path.join(REPO, "docs", sub, "*.md"))
        doc_paths += glob.glob(os.path.join(REPO, "research", "*", "README.md"))
        for singleton in (
            os.path.join(REPO, "docs", "ONBOARDING.md"),
            os.path.join(REPO, "docs", "WOW_TEMPLATE.md"),
            os.path.join(REPO, "ROADMAP_LAMBDA_COMPLETION.md"),
            os.path.join(REPO, "rl_whitepaper.md"),
            # rl_whitepaper.md moved to archive (2026-06); fall back so the
            # RL architecture doc is still indexed (deviation, see report).
            os.path.join(REPO, "archive", "root_2026_06", "rl_whitepaper.md"),
            os.path.join(REPO, "AGENTS.ini"),
        ):
            if os.path.exists(singleton):
                doc_paths.append(singleton)
        seen_docs: set[str] = set()
        for path in sorted(doc_paths):
            ap = os.path.abspath(path)
            if ap in seen_docs:
                continue
            seen_docs.add(ap)
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]
            # README.md collides across projects -> qualify by parent dir
            if base.lower() == "readme.md":
                stem = os.path.basename(os.path.dirname(path))
            ingest_sectioned(path, rel, "context", f"doc:{stem}", "docs")

    # 5. governing specs (dojo_forge) ; tier=stable, tag=spec:<name>
    if want("specs"):
        for path in sorted(
            glob.glob(os.path.join(REPO, "research", "dojo_forge", "RIDE_EDGE_GATE_SPEC.md"))
            + glob.glob(os.path.join(REPO, "research", "mamba_zigzag_baseline", "PRODUCTION_RUN_SPEC.md"))
        ):
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            stem = os.path.splitext(os.path.basename(path))[0]
            ingest_sectioned(path, rel, "stable", f"spec:{stem}", "specs")

    # === code layer (scope-ext) — one row per module, AST-extracted =========
    if want("code"):
        parse_failures = 0
        for path in iter_code_files():
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            surface = extract_code_surface(path)
            if surface is None:
                parse_failures += 1
                continue
            if not surface:
                surface = f"# {rel} (no docstrings / signatures)"
            added = add(first_date(surface, path), "context", f"code:{rel}", rel, surface)
            bump("code", added)
        stats["_code_parse_failures"] = parse_failures

    return rows, stats


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
    ap.add_argument(
        "--sources",
        default=None,
        help="comma-separated source classes for a PARTIAL rebuild "
             "(default: all). keys: " + ",".join(ALL_SOURCES),
    )
    args = ap.parse_args(argv)

    sources = None
    if args.sources:
        sources = {s.strip() for s in args.sources.split(",") if s.strip()}
        unknown = sources - set(ALL_SOURCES)
        if unknown:
            print(f"[build] unknown --sources: {sorted(unknown)}; "
                  f"valid: {ALL_SOURCES}", file=sys.stderr)
            return 2

    rows, stats = build_rows(sources)
    write_db(rows, args.db)

    if not args.quiet:
        code_fail = stats.pop("_code_parse_failures", 0)
        total_bytes = sum(st["bytes"] for st in stats.values())
        print(f"DERIVED memory DB rebuilt: {args.db}")
        print(f"total rows: {len(rows)}   indexed text: {total_bytes/1e6:.2f} MB")
        print(f"{'source class':24s} {'rows':>6s} {'MB':>8s}")
        for key in ALL_SOURCES:
            st = stats.get(key)
            if not st:
                continue
            print(f"  {key:22s} {st['rows']:6d} {st['bytes']/1e6:8.3f}")
        if sources is None or "code" in sources:
            print(f"  code parse-failures: {code_fail}")
        by_tier: dict[str, int] = {}
        for r in rows:
            by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        print("by tier: " + ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
