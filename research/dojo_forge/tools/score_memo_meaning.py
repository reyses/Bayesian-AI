#!/usr/bin/env python3
"""Mechanical meaning-check on a memo bank (the distillation-race gate,
owner 2026-07-24: "what we are looking for is that the LLM can produce
meaningful memories"). Pre-registered PASS bar (stated BEFORE the v2seed data
existed — not tuned on it):

  PASS iff  info_rate >= 0.20   (>=20% of memos carry a concrete magnitude:
                                 a decimal number or >=2-digit integer that is
                                 NOT a rule id like G1.3)
       and  selectivity <= 0.80 (memo emitted on <=80% of decision frames —
                                 v1's failure mode was a memo EVERY frame)

v1 baseline for reference: info_rate 1/151 = 0.7%, selectivity ~100%.
Exit code 0 = PASS, 1 = FAIL (orchestrator branches on this). Prints + writes
reports/memo_meaning_<tag>.md.
usage: score_memo_meaning.py --db <bank.db> --ckpt <memo_run_ckpt.jsonl> --tag v2seed
"""
import argparse
import json
import os
import re
import sqlite3
import sys

DOJO = os.path.join(os.path.dirname(__file__), '..')
RULE_ID = re.compile(r'G\d+(?:\.\d+)?')
MAGNITUDE = re.compile(r'\d+\.\d+|\d{2,}')

INFO_RATE_MIN = 0.20
SELECTIVITY_MAX = 0.80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='v2seed')
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    memos = [r[0] for r in con.execute("SELECT text FROM memos")]
    n_info = sum(1 for t in memos if MAGNITUDE.search(RULE_ID.sub('', t)))
    info_rate = n_info / len(memos) if memos else 0.0

    frames = memo_frames = 0
    with open(args.ckpt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            for fr in ep.get('frames', []):
                frames += 1
                memo_frames += bool(fr.get('memo_present') or fr.get('memo'))
    selectivity = memo_frames / frames if frames else 1.0

    ok = info_rate >= INFO_RATE_MIN and selectivity <= SELECTIVITY_MAX
    verdict = 'PASS' if ok else 'FAIL'
    report = (
        f"# memo meaning check — {args.tag}\n"
        f"memos: {len(memos)} | data-bearing: {n_info} "
        f"(info_rate {info_rate:.0%}, bar >= {INFO_RATE_MIN:.0%})\n"
        f"frames: {frames} | memo-emitting: {memo_frames} "
        f"(selectivity {selectivity:.0%}, bar <= {SELECTIVITY_MAX:.0%})\n"
        f"v1 baseline: info_rate 0.7%, selectivity ~100%\n\n"
        f"**VERDICT: {verdict}** — "
        + ("meaningful memories: BLEND and scale (distillation race on)."
           if ok else
           "distillate not meaningful yet; fall back to grounding census, "
           "diagnose the memo channel before scaling.")
        + "\n")
    out = os.path.join(DOJO, 'reports', f'memo_meaning_{args.tag}.md')
    with open(out, 'w') as f:
        f.write(report)
    print(report)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
