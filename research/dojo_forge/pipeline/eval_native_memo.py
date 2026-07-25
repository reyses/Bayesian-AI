#!/usr/bin/env python3
"""MEMO-mode teacher runner — generation harness with the memory loop (docs 149+151).

Derived from tools/exam_day.py's chat/generation pattern (NOT the exam: no probes,
no debrief). Per frame the teacher decides HOLD/EXIT and MAY emit its own
`MEMO: <=30 words` salience compression (doc 149 MEMO protocol). Memos land in the
teacher_memory bank (subject to the mechanical GUARD v2 admission), and later
episodes retrieve prior-day memos into a RELEVANT MEMORY block — the cross-episode
"trader journal", learning WITHOUT weight updates.

Per-frame context (HIST_MIN=10, the memo trade-off: doc 149 "the mechanical 1m/5m
history window shrinks 20->10 min in exchange" for the memo channel):
  [ANCHOR]  frame-0 wide field (pinned)
  [HISTORY] last 10 min of [1m]/[5m] closed-bar lines + the decision trail
  [RELEVANT MEMORY]  top-3 retrieved prior-day memos   (ONLY when --use-memory on)
  [NOW]     the current minute's full tape
  format demand: DECISION / CONFIDENCE / REASON  (+ optional MEMO)

Arms (the day-carry natural experiment, doc 149):
  --days D            comma-separated day list (e.g. 2025_04_08 or 2025_04_08,2025_04_09)
  --use-memory on|off inject retrieved prior-day memos
  --write-memos on|off admit this run's memos into the bank (admission = --days set)
  --arm-tag TAG       names the artifacts:
        gate_state/memo_run_<TAG>.jsonl  (ckpt: per-episode decisions/memos/retrievals)
        reports/memo_run_<TAG>.csv       (eid,frame_idx,decision,conf,memo_present)

Copies exam v3 lessons: MAX_GEN=2600, temp 0, seed 42, TRUNCATED loud (a truncated
/ unparseable answer is NEVER silently defaulted to HOLD). Resume-safe: episodes
already in the ckpt are skipped.

Verify-then-stop: --selftest runs 2 frames of one episode with a FAKE (canned) llm,
exercising admission, retrieval determinism, lookback exclusion, ledger append.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(DOJO, 'tools'))
import eval_native_ckpt as base                      # shared machinery (loader, genome)
from eval_native_tiered import filter_hist           # 1m/5m closed-bar line filter
from exam_day import chat, visible, parse_answer     # generation chat + answer parse
from teacher_memory import TeacherMemory             # the memory bank (GUARD v2)

PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
GATE_STATE = os.path.join(DOJO, 'gate_state')
REPORTS = os.path.join(DOJO, 'reports')

NUM_CTX = 13312         # matches exam_day (headroom for the shorter memo window)
MAX_GEN = 2600          # exam v3: 1400 still truncated <think> on this content; 2600 + brevity
HIST_MIN = 10           # doc 149 MEMO trade-off: 1m/5m window shrinks 20->10 for the memo channel
TOP_K_MEMO = 3          # RELEVANT MEMORY block = top-3 retrieved memos (mirrors TeacherMemory.top_k)
MEMO_MAX_WORDS = 30     # doc 149 "MEMO: <=30 words" — the model's own salience compression cap

# System = current GENOME rules + brief memo instructions (doc 149 MEMO protocol).
MEMO_SYSTEM = (
    "You are trading one episode minute by minute, deciding HOLD (stay in) or EXIT "
    "(close now); your FIRST EXIT is binding and ends the episode. Every price number "
    "is FAVORABLE-SIGNED points from entry (entry=0.00): positive good, negative bad, "
    "regardless of LONG/SHORT.\n"
    "At each decision respond EXACTLY:\n"
    "DECISION: HOLD|EXIT\nCONFIDENCE: <0.00-1.00>\nREASON: <=60 words citing the "
    "specific Genome rule(s) and market features you used.\n"
    "You MAY then add ONE line:\n"
    f"MEMO: <=30 words — a durable, day-agnostic lesson for your future self (a "
    "recurring signature and what it meant), NO date/day references. Omit MEMO if "
    "nothing is worth remembering.\n"
    "When a RELEVANT MEMORY block is present it holds your OWN prior-day notes for "
    "similar states — use them as priors, not commands.\n"
    "Keep your <think> block UNDER 250 words; an unfinished answer scores zero.\n\n"
    "RULES (Genome):\n"
)

# v2 (owner 2026-07-24, "seed memo #9 as the expected"): pilot arm A showed the
# v1 wording ("durable, day-agnostic lesson") over-generalizes into genome-echo
# mottos — 17/18 memos were rule restatements; the ONE useful memo carried a
# concrete magnitude (reversion_prob split). v2 seeds that format with the
# teacher's OWN best memo as exemplar (no human alpha enters the bank), makes
# no-memo the default, and bans rule restatement. Day-agnostic ban unchanged —
# favorable-signed magnitudes are entry-relative and do not identify days.
MEMO_SYSTEM_V2 = MEMO_SYSTEM.replace(
    "You MAY then add ONE line:\n"
    "MEMO: <=30 words — a durable, day-agnostic lesson for your future self (a "
    "recurring signature and what it meant), NO date/day references. Omit MEMO if "
    "nothing is worth remembering.\n",
    "MOST frames deserve NO memo. Only when you observe a NEW, reusable market "
    "signature — not on any frame where you would merely restate a rule — add "
    "ONE line:\n"
    "MEMO: <=30 words with AT LEAST ONE concrete magnitude (a feature value, "
    "signed-points level, or duration) and what it resolved into. NEVER restate "
    "a Genome rule (you already know them); NO date/day references.\n"
    "Format exemplar (from your own notes): "
    "\"MEMO: reversion_prob_30 split 0.80(1m)/0.97(5m) during gb>40% resolved "
    "as continuation - multi-TF split favored holding.\"\n")

_RE_MEMO = re.compile(r'^\s*MEMO:\s*(.+?)\s*$', re.M)


def day_of(eid):
    """'2025_04_08_1744119065_S' -> '2025_04_08' (the day = unit of independence)."""
    return "_".join(eid.split('_')[:3])


def extract_memo(text):
    """Pull the MEMO line (if any), capped at MEMO_MAX_WORDS words. None if absent."""
    m = _RE_MEMO.search(text)
    if not m:
        return None
    words = m.group(1).split()
    return " ".join(words[:MEMO_MAX_WORDS]) if words else None


def build_memo_content(frames, i, trail, memory_block):
    """Per-frame tiered context (HIST_MIN=10) + optional RELEVANT MEMORY block.

    Mirrors exam_day.build_exam_content but with the shorter memo-mode window and
    the injected memory block between history and NOW.
    """
    anchor = frames[0]['text']
    lo = max(1, i - HIST_MIN)
    hist = []
    for j in range(lo, i):
        t_lab = frames[j]['text'].splitlines()[0].split(']')[0].lstrip('[')
        d = trail[j] if j < len(trail) else None
        dec = (f" (you said: {d['decision']} conf {d['conf']}: {d['reason'][:60]})"
               if d else "")
        hist.append(f"[{t_lab}]{dec}\n{filter_hist(frames[j]['text'])}")
    parts = [anchor,
             "== 1m/5m HISTORY + your decision trail ==\n"
             + ("\n".join(hist) if hist else "(none)")]
    if memory_block:
        parts.append("== RELEVANT MEMORY (your prior-day notes) ==\n" + memory_block)
    parts.append("== NOW (full tape) ==\n" + frames[i]['text'])
    return "\n\n".join(parts)


def render_memory_block(granted):
    """Render retrieved memos into the injected RELEVANT MEMORY text."""
    if not granted:
        return ""
    lines = []
    for g in granted:
        lines.append(f"- (from a prior day, minute {g['minute']}) {g['text']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Checkpoint CSV (per doc convention: eid,frame_idx,decision,conf,memo_present) #
# --------------------------------------------------------------------------- #
def rebuild_csv(csv_path, records):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("eid,frame_idx,decision,conf,memo_present\n")
        for rec in records:
            eid = rec['episode_id']
            for fr in rec['frames']:
                f.write(f"{eid},{fr['frame_idx']},{fr['decision']},"
                        f"{fr['conf']},{1 if fr['memo'] else 0}\n")


def eval_episode_memo(llm, eid, packet, system, mem, use_memory, write_memos):
    """Run one episode frame-by-frame in generation mode with the memory loop."""
    frames = packet['frames']
    eday = day_of(eid)
    trail = []
    frame_recs = []
    memos_written = 0
    retrievals_used = 0
    exit_frame = None
    t0 = time.time()

    for i in range(len(frames)):
        granted = []
        if use_memory:
            granted = mem.retrieve(frames[i]['text'], eid, eday, i)
            if granted:
                retrievals_used += 1
        content = build_memo_content(frames, i, trail, render_memory_block(granted))
        turns = [("user", content
                  + "\n\nYour decision for THIS minute (DECISION/CONFIDENCE/REASON"
                    " + optional MEMO):")]
        ans = visible(chat(llm, system, turns, max_tokens=MAX_GEN))
        a = parse_answer(ans)                          # TRUNCATED stays loud (exam v2 lesson)
        memo = None if a['decision'] == 'TRUNCATED' else extract_memo(ans)
        trail.append(a)

        memo_written = False
        if write_memos and memo:
            res = mem.write_memo(eid, eday, i, memo, frames[i]['text'])
            memo_written = res['admitted']
            if memo_written:
                memos_written += 1

        frame_recs.append(dict(frame_idx=i, decision=a['decision'], conf=a['conf'],
                               reason=a['reason'], memo=memo,
                               memo_written=memo_written,
                               retrieved_ids=[g['id'] for g in granted]))
        if a['decision'] == 'EXIT' and exit_frame is None:
            exit_frame = i
        print(f"[{eid} m{i:02d}] {a['decision']} conf={a['conf']} "
              f"mem={len(granted)} memo={'Y' if memo else '-'} :: {a['reason'][:60]}",
              flush=True)

    return dict(episode_id=eid, day=eday, use_memory=use_memory,
                write_memos=write_memos, exit_frame=exit_frame,
                n_frames=len(frame_recs), memos_written=memos_written,
                retrievals_used=retrievals_used,
                elapsed_s=round(time.time() - t0, 1), ts=time.time(),
                frames=frame_recs)


# --------------------------------------------------------------------------- #
#  Fake/canned llm for --dry-run / --selftest (monkeypatch-friendly)            #
# --------------------------------------------------------------------------- #
_CANNED_ANSWER = (
    "<think>\nbrief\n</think>\n"
    "DECISION: HOLD\nCONFIDENCE: 0.70\n"
    "REASON: giveback within tolerance and trend intact per G0.3; 1m velocity down "
    "but leg still developing.\n"
    "MEMO: mid giveback on an intact trend with velocity down early in the leg tends "
    "to resolve back — hold through it."
)


def make_canned_llm(text=_CANNED_ANSWER):
    """Return an llm-shaped callable: llm(prompt, **kw) -> {'choices':[{'text':...}]}."""
    def _llm(prompt, **kw):
        return {'choices': [{'text': text}]}
    return _llm


def _selftest():
    """2 frames of one real episode with a canned llm; verify the mechanical guards."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix='memorun_')
    db = os.path.join(tmp, 'tm.db')
    ledger = os.path.join(tmp, 'ledger.jsonl')

    pkt_path = sorted(glob.glob(os.path.join(PACKETS, '2025_04_08*.json')))[0]
    packet = json.load(open(pkt_path))
    packet = dict(packet, frames=packet['frames'][:2])   # 2 frames only
    eid = packet['episode_id']
    eday = day_of(eid)                                   # 2025_04_08
    prior_day = '2025_04_07'                             # 1 day before -> within lookback

    llm = make_canned_llm()
    system = MEMO_SYSTEM + base.load_genome()
    ok = True

    # Bank admits the episode's OWN day only (write allowlist = the run's --days set).
    mem = TeacherMemory(db_path=db, ledger_path=ledger,
                        write_allowlist={eday}, run_tag='selftest')
    # Seed a prior-day memo so a later same-state retrieval has something causal to
    # grant. Seeded via a throwaway bank whose allowlist admits the prior day.
    seed = TeacherMemory(db_path=db, ledger_path=ledger,
                         write_allowlist={prior_day}, run_tag='selftest_seed')
    seed.write_memo('epPRIOR', prior_day, 3, 'prior-day lesson for this state',
                    packet['frames'][0]['text'])
    seed.close()

    # Run 2 frames WITH memory + writes on.
    rec = eval_episode_memo(llm, eid, packet, system, mem,
                            use_memory=True, write_memos=True)
    print(f"[selftest] ran {rec['n_frames']} frames; decisions="
          f"{[f['decision'] for f in rec['frames']]} memos_written={rec['memos_written']} "
          f"retrievals_used={rec['retrievals_used']}")

    # (1) ADMISSION: same-day writes admitted; a non-allowlisted day is REJECTED.
    admit_ok = rec['memos_written'] == 2      # both canned frames carried a MEMO
    rej = mem.write_memo('epX', '2025_04_01', 0, 'should be rejected',
                         packet['frames'][0]['text'])
    admit_ok &= (not rej['admitted'])
    ok &= admit_ok
    print(f"[selftest] admission: same-day writes={rec['memos_written']}/2 "
          f"non-allowlisted REJECTED={not rej['admitted']} -> {admit_ok}")

    # (2) DETERMINISM: same NOW frame retrieved twice -> identical ids.
    g1 = mem.retrieve(packet['frames'][0]['text'], eid, eday, 0)
    g2 = mem.retrieve(packet['frames'][0]['text'], eid, eday, 0)
    det_ok = [g['id'] for g in g1] == [g['id'] for g in g2]
    ok &= det_ok
    print(f"[selftest] determinism: {[g['id'] for g in g1]} == "
          f"{[g['id'] for g in g2]} -> {det_ok}")

    # (3) LOOKBACK: retrieval grants the prior-day memo, NEVER a same-day memo.
    days = {g['day'] for g in g1}
    lookback_ok = (prior_day in days) and (eday not in days)
    ok &= lookback_ok
    print(f"[selftest] lookback: granted days={sorted(days)} "
          f"(prior-day in, same-day out) -> {lookback_ok}")

    # (4) LEDGER: writes + retrievals appended.
    with open(ledger) as f:
        events = [json.loads(x)['event'] for x in f if x.strip()]
    ledger_ok = ('write_admitted' in events and 'write_rejected' in events
                 and events.count('retrieve') >= 2)
    ok &= ledger_ok
    print(f"[selftest] ledger: {len(events)} events, "
          f"admitted={events.count('write_admitted')} "
          f"rejected={events.count('write_rejected')} "
          f"retrieve={events.count('retrieve')} -> {ledger_ok}")

    mem.close()
    print(f"[selftest] {'PASS' if ok else 'FAIL'} (temp db {db})")
    return ok


def main():
    ap = argparse.ArgumentParser(description="MEMO-mode teacher runner (docs 149+151)")
    ap.add_argument('--days', help="comma-separated day list, e.g. 2025_04_08,2025_04_09")
    ap.add_argument('--use-memory', choices=['on', 'off'], default='off')
    ap.add_argument('--write-memos', choices=['on', 'off'], default='off')
    ap.add_argument('--limit', type=int, default=None, help="max NEW episodes this pass")
    ap.add_argument('--arm-tag', default='untagged', help="names the artifacts")
    ap.add_argument('--knowledge', choices=['off', 'v1'], default='off',
                    help="v1 = insert frozen KNOWLEDGE_PACK_v1 (education) "
                         "before the Genome rules; hash logged per record")
    ap.add_argument('--memo-style', choices=['v1', 'v2seed'], default='v1',
                    help="v2seed = #9-exemplar format (concrete magnitude, "
                         "no-memo default, no rule restatement)")
    ap.add_argument('--packets-dir', default=PACKETS)
    ap.add_argument('--db', default=None, help="memory DB path (default gate_state/teacher_memory.db)")
    ap.add_argument('--ledger', default=None, help="ledger path (default gate_state/memory_ledger.jsonl)")
    ap.add_argument('--num-ctx', type=int, default=NUM_CTX)
    ap.add_argument('--n-gpu-layers', type=int, default=None)
    ap.add_argument('--model-blob', default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help="use a canned llm (no llama_cpp) — exercises the loop offline")
    ap.add_argument('--selftest', action='store_true',
                    help="verify-then-stop: 2 frames + guard checks with a canned llm")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.days:
        ap.error("--days is required (unless --selftest)")

    use_memory = args.use_memory == 'on'
    write_memos = args.write_memos == 'on'
    days = [d.strip() for d in args.days.split(',') if d.strip()]

    ckpt = os.path.join(GATE_STATE, f'memo_run_{args.arm_tag}.jsonl')
    csv_path = os.path.join(REPORTS, f'memo_run_{args.arm_tag}.csv')
    db = args.db or os.path.join(GATE_STATE, 'teacher_memory.db')
    ledger = args.ledger or os.path.join(GATE_STATE, 'memory_ledger.jsonl')

    # Select episodes for the requested day(s).
    todo = []
    for path in sorted(glob.glob(os.path.join(args.packets_dir, '*.json'))):
        eid = os.path.basename(path).replace('.json', '')
        if day_of(eid) in days:
            todo.append((eid, path))
    if not todo:
        print(f"No packets for days {days} in {args.packets_dir}", file=sys.stderr)
        sys.exit(1)

    completed = base.load_completed(ckpt)              # resume-safe (episode_id keyed)
    todo = [(eid, p) for eid, p in todo if eid not in completed]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[plan] arm={args.arm_tag} days={days} use_memory={use_memory} "
          f"write_memos={write_memos} | {len(todo)} episodes this pass "
          f"({len(completed)} already done)", flush=True)

    # ADMISSION allowlist = the run's --days set (writes only for days being run).
    mem = TeacherMemory(db_path=db, ledger_path=ledger,
                        write_allowlist=set(days), run_tag=args.arm_tag,
                        top_k=TOP_K_MEMO)
    chosen = MEMO_SYSTEM_V2 if args.memo_style == 'v2seed' else MEMO_SYSTEM
    knowledge_hash = None
    if args.knowledge == 'v1':
        import hashlib as _h
        kp = os.path.join(DOJO, 'genome', 'KNOWLEDGE_PACK_v1.md')
        raw = open(kp, encoding='utf-8').read()
        knowledge_hash = _h.sha256(raw.encode()).hexdigest()[:16]
        education = '== YOUR EDUCATION' + raw.split('== YOUR EDUCATION')[1]
        chosen = chosen.replace("RULES (Genome):\n",
                                education + "\n\nRULES (Genome):\n")
    system = chosen + base.load_genome()

    if args.dry_run:
        llm = make_canned_llm()
    else:
        model_blob = args.model_blob or base.DEFAULT_BLOB_LINUX
        if not os.path.exists(model_blob):
            model_blob = base.DEFAULT_BLOB_WSL
        n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else -1
        n_gpu_layers = base.preflight_vram(n_gpu_layers, args.num_ctx)
        from llama_cpp import Llama
        print(f"Loading model n_ctx={args.num_ctx} n_gpu_layers={n_gpu_layers} "
              f"n_batch={base.N_BATCH} ...", flush=True)
        llm = Llama(model_path=model_blob, n_gpu_layers=n_gpu_layers,
                    n_ctx=args.num_ctx, n_batch=base.N_BATCH, seed=42,
                    temperature=0.0, logits_all=False, flash_attn=True, verbose=False)

    for k, (eid, path) in enumerate(todo, 1):
        packet = json.load(open(path))
        rec = eval_episode_memo(llm, eid, packet, system, mem, use_memory, write_memos)
        base.append_checkpoint(ckpt, rec)
        completed[eid] = rec
        rebuild_csv(csv_path, list(completed.values()))
        print(f"[{k}/{len(todo)}] {eid}: {rec['n_frames']} frames {rec['elapsed_s']}s "
              f"exit_frame={rec['exit_frame']} memos={rec['memos_written']} "
              f"retr={rec['retrievals_used']}", flush=True)

    mem.close()
    print(f"[done] arm={args.arm_tag}: ran {len(todo)} episodes -> {ckpt} / {csv_path}",
          flush=True)


if __name__ == '__main__':
    main()
