#!/usr/bin/env python3.11
"""Analytics over the DERIVED memory FTS mirror. Read-only. stdlib only.

Two reports:
  1. Recurring-correction candidates — near-duplicate rows grouped across dates
     via normalized-token Jaccard clustering. Surfaces the same lesson learned
     (or re-learned) more than once, so it can be promoted / consolidated.
  2. Stale-entry report — memory rows that name a file path which no longer
     exists on disk. FLAG ONLY — never deleted, never edited.

Output: stdout AND tools/memory_loop/last_stats.md (overwritten each run,
derived artifact).

  python3.11 tools/memory_loop/memory_stats.py [--db PATH] [--jaccard 0.5]
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys

try:  # ensure UTF-8 stdout on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DB = os.path.join(REPO, "docs", "memory", "memory.db")
OUT_MD = os.path.join(HERE, "last_stats.md")

STOPWORDS = set(
    """the a an and or of to in on for with without via per is are was were be been
    it its this that these those at by from as into out not no than then so but our
    we you your not new now all any each one two via vs vs. -> = + & etc etc.
    day days run runs report reports docs doc""".split()
)
TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{2,}")
# path-ish token: MUST contain a directory separator + a known code/data
# extension. Bare filenames in prose (e.g. "run_all.py") are excluded — they
# cannot be resolved from repo root and produce false stale flags.
PATH_RE = re.compile(
    r"(?<![\w./-])([\w.-]+(?:/[\w.-]+)+\.(?:py|md|txt|json|csv|ini|cs|h5|pth|pt|onnx|parquet|yml|yaml|sh))"
)


def tokens(text: str) -> frozenset[str]:
    toks = {
        t
        for t in TOKEN_RE.findall(text.lower())
        if t not in STOPWORDS and not t.isdigit()
    }
    return frozenset(toks)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def recurring_corrections(rows, thresh):
    """Union-find cluster rows by token-set Jaccard >= thresh; keep clusters
    that span >= 2 distinct dates. Correction bias: prioritise volatile
    (journal/index) + stable (feedback) rows, which carry the corrections."""
    cand = [
        r
        for r in rows
        if r["tier"] in ("volatile", "stable") and len(r["_tok"]) >= 5
    ]
    parent = list(range(len(cand)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(len(cand)):
        ti = cand[i]["_tok"]
        for j in range(i + 1, len(cand)):
            if jaccard(ti, cand[j]["_tok"]) >= thresh:
                union(i, j)

    clusters: dict[int, list] = {}
    for i, r in enumerate(cand):
        clusters.setdefault(find(i), []).append(r)

    out = []
    for members in clusters.values():
        dates = {m["date"] for m in members}
        if len(members) >= 2 and len(dates) >= 2:
            out.append(sorted(members, key=lambda m: m["date"]))
    # rank: most distinct dates, then most members
    out.sort(key=lambda ms: (len({m["date"] for m in ms}), len(ms)), reverse=True)
    return out


def stale_entries(rows):
    """Flag file paths named in ACTIVE memory that no longer exist.

    Scope = stable/context tier, excluding archive/*. The volatile journal &
    INDEX and the pre-condense archive dumps are historical logs: they name
    hundreds of deleted/moved research files by design, so checking them just
    reproduces known project churn. Active memory pointing at a missing file
    is the actionable signal."""
    flagged = []
    for r in rows:
        if r["tier"] == "volatile" or r["source_file"].startswith("docs/memory/archive/"):
            continue
        for path in set(PATH_RE.findall(r["text"])):
            full = os.path.join(REPO, path)
            if not os.path.exists(full):
                flagged.append((r, path))
    return flagged


def short(text, n=140):
    return text if len(text) <= n else text[:n] + " …"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analytics over derived memory DB.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--jaccard", type=float, default=0.5)
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"[memory_stats] DB not found: {args.db}", file=sys.stderr)
        print("[memory_stats] rebuild: python3.11 tools/memory_loop/build_memory_db.py",
              file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id,date,tier,tag,source_file,text FROM learnings")]
    for r in rows:
        r["_tok"] = tokens(r["text"])

    clusters = recurring_corrections(rows, args.jaccard)
    stale = stale_entries(rows)

    lines = []
    lines.append("# Derived memory analytics — last_stats.md")
    lines.append("")
    lines.append(f"Rows analyzed: {len(rows)}  ·  jaccard>={args.jaccard}  "
                 f"·  DB: {os.path.relpath(args.db, REPO).replace(os.sep,'/')}")
    lines.append("(DERIVED report; regenerate with memory_stats.py. Flags only — nothing deleted.)")
    lines.append("")

    lines.append("## Recurring-correction candidates (near-duplicate across dates)")
    if not clusters:
        lines.append("_none above threshold_")
    else:
        for k, ms in enumerate(clusters[:15], 1):
            dates = sorted({m["date"] for m in ms})
            lines.append(f"### #{k}  ({len(ms)} rows over {len(dates)} dates: "
                         f"{', '.join(dates)})")
            for m in ms:
                lines.append(f"- `{m['date']}` [{m['tier']}] {m['tag']}: {short(m['text'])}")
            lines.append("")

    lines.append("## Stale-entry report (named file not found on disk — FLAG ONLY)")
    lines.append("_Scope: active memory (stable/context, excl. archive). Volatile journal/"
                 "INDEX + archive dumps are excluded — they log deleted files by design._")
    if not stale:
        lines.append("_no missing file references detected_")
    else:
        by_src: dict[str, int] = {}
        for r, _ in stale:
            by_src[r["source_file"]] = by_src.get(r["source_file"], 0) + 1
        lines.append("")
        lines.append(f"**{len(stale)} missing-file references, by source "
                     "(long historical-arc files expected to dominate):**")
        for src, c in sorted(by_src.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {c:4d}  {src}")
        lines.append("")
        lines.append("<details>")
        for r, path in stale:
            lines.append(f"- MISSING `{path}` — in {r['source_file']} "
                         f"[{r['tier']}] {r['tag']} (`{r['date']}`)")
        lines.append("</details>")
    lines.append("")

    report = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(report)
    print(f"\n[memory_stats] written: {os.path.relpath(OUT_MD, REPO).replace(os.sep,'/')}")

    # machine-readable top candidate line for the caller
    if clusters:
        top = clusters[0]
        tdates = sorted({m["date"] for m in top})
        print(f"[TOP RECURRING-CORRECTION CANDIDATE] {len(top)} rows over "
              f"{len(tdates)} dates ({tdates[0]}..{tdates[-1]}); "
              f"sample: {short(top[0]['text'], 120)}")
    else:
        print("[TOP RECURRING-CORRECTION CANDIDATE] none above threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
