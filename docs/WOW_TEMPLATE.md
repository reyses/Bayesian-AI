# WoW Template — a generalized Way-of-Working for AI-orchestrated research projects
> Extracted from the Bayesian-AI program (2026). Domain-agnostic: replace
> <PLACEHOLDERS> and delete sections that don't apply. Drop this into a new
> repo as the seed for CLAUDE.md / AGENTS.md and build the folder skeleton in §3.

---

## 1. Roles — the delegation ladder
One REVIEWER orchestrates; WORKERS execute. Capability flows down, evidence flows up.

| role | model tier | does | never does |
|---|---|---|---|
| **Reviewer** | strongest available | writes specs, gates launches, verifies artifacts, issues verdicts, commits | long inline builds (delegate them) |
| **Builder drone** | mid/heavy | self-contained builds + runs from a written spec | commit, launch fleets, self-certify |
| **Runner drone** | cheap | mechanical, verifiable runs (apply patch X, run Y, tabulate Z) | design decisions |
| **Watcher** | cheapest / plain script | poll a folder/process, notify on completion | anything else |
| **Fleet agent** | cheap, many | one bounded episode/task each, least-privilege | see ground truth; exceed its allowlist |

Standing rules:
- **Spec on disk before dispatch** (numbered comms doc, §4) — drones survive
  session loss; anyone can resume from the file.
- **Drones RUN SYNCHRONOUSLY** — the recurring defect is backgrounding a long
  run and stopping. Say it in every spec; shepherd if it happens anyway.
- **Drones commit NOTHING.** The reviewer reproduces at least one number from
  the drone's own artifacts before accepting/committing (pick a DIFFERENT
  sample than the drone verified).
- **Claim–evidence coupling**: every claim in a report must point at an
  artifact (file, table, log line). Skip-rather-than-fabricate.
- Workers asking for a spec deviation is a PASS behavior — ratify or reject
  explicitly in the verdict doc; never let a deviation pass silently.

## 2. The lifecycle of a unit of work
```
idea (owner) → PROPOSE design + discuss (never build in the same turn)
  → SPEC doc N (inputs, method, deliverables, kill rule, verify-then-stop)
  → dispatch drone → drone builds + verifies on toy/dummy first → STOPS
  → reviewer gate: verify artifacts, spot-check causality/correctness
  → full run (fleet or heavy run) → score
  → VERDICT doc N+1: findings + explicit kill/pass + what changes downstream
  → journal + commit + push
```
- **Verify-then-stop**: every build proves itself on 1-2 scripted dummy cases
  + exactly one real case, then stops for the reviewer gate before scale.
- **One change at a time** from any baseline; if worse, revert immediately.
- **Pre-registered kill rules**: state the failure criterion in the SPEC,
  before results exist. Letter vs spirit both count — a result can pass the
  letter and fail the spirit; the spirit governs (say so in the verdict).

## 3. Repository structure
```
<repo>/
  CLAUDE.md / AGENTS.md      # this WoW, project-specialized
  research/<topic>/          # ONE folder per research effort — never flat
    pipeline/  builders/  tools/   # code by function
    reports/                 # findings .md + assets (THIS topic only)
    README.md                # what each script is, how to run, where data lives
  <production code dirs>     # kept separate from research
  tools/                     # cross-cutting reusable tools (indexed!)
  docs/
    daily/YYYY-MM-DD.md      # session journals
    daily/INDEX.md           # one line per day, newest first — the recall surface
    reference/RESEARCH_JOURNAL.txt   # condensed long-arc log
    Active/                  # current roadmaps ONLY (prune aggressively)
  <comms dir>/NNN_DATE_TITLE.md      # numbered specs/verdicts (append-only)
  checkpoints|artifacts/     # heavy outputs, gitignored, regenerable
```
Rules: research reports live IN their topic folder; top-level report dirs are
for cross-cutting only. Big/regenerable data stays out of git with a
documented regeneration recipe. Set structure up at the START — retrofitting
is a path-refactor. Keep a TOOLS index; extend, don't rebuild.

## 4. Comms docs — the append-only paper trail
Numbered `NNN_DATE_TYPE_TITLE.md`. Types: TASK (spec), VERDICT/FINAL,
DIRECTIVE, REPORT. Norms:
- Append-only: never rewrite history; corrections are new docs citing the old.
- A VERDICT records: what was claimed, what the reviewer verified (with the
  reproduced numbers), ratified deviations, the ruling, downstream effects.
- Own your errors in-doc ("the fires/day cap was a REVIEWER SPEC ERROR") —
  the record is for the program, not for looking good.

## 5. Epistemic discipline (the anti-self-deception kit)
1. **Sealed evaluation**: tune on TRAIN only, freeze, then touch HOLDOUT once.
   Anything tuned after seeing holdout is dead on arrival.
2. **Dumb-baseline bar**: every clever method must beat the dumbest comparable
   rule (constant answer, fixed threshold, naive stop) on the SAME data, as a
   swept ROC/frontier — not a single cherry point. "Above the dumb frontier
   AND beats its best point" or it's the dumb rule with extra steps.
3. **Report distributions, mode-first**: histogram/mode before mean; mean only
   WITH a bootstrap CI; block the bootstrap on the natural correlation unit
   (day/user/batch — pseudo-replication inflates N).
4. **No effect claim without**: (a) CI on the delta, (b) explicit significance
   statement ("CI includes 0 → not significant"), (c) the N it rests on.
5. **Effect-size floor**: agree a magnitude below which a "signal" is noise
   regardless of p-value (<FLOOR>); label conditional effects as conditional.
6. **Leakage firewall**: generation-time lookahead is the cardinal sin. Fix
   the convention once (e.g. "an interval labeled B closes at B+period; a
   consumer at time t may use it only if B+period ≤ t"), centralize it in ONE
   function, and ASSERT it in every builder (violation = build failure, not
   warning). When an eval can peek (single-prompt LLM play, shared context),
   either make blindness structural (a serving gate + audit chain) or stamp
   every number HYPOTHESIS-ONLY.
7. **Graduation firewall**: exploratory numbers are never results. A rule
   graduates only by being re-implemented causally and passing the sealed
   harness. Keep the two populations of numbers visually distinct.
8. **The graveyard**: a standing never-retry list with the MEASURED cost of
   each dead idea. Prevents relitigating; new members require a verdict doc.
9. **Contrast method (Red X)**: to find what separates good from bad cases,
   contrast extremes, isolate the dominant variable, cut a boundary on it,
   re-test sealed. Iterate one variable at a time.
10. **Robustness beats optimality**: a structure that only works at one magic
    threshold is overfit. Prefer quantile-matching thresholds across regime/
    estimator drift over free re-optimization; "verdicts stable under
    threshold shift" is itself a key positive result.
11. **Honest nulls are deliverables**: "X adds nothing over the trivial
    baseline" ends a lane and is journaled with the same care as a win.
12. **Anti-doom/anti-hype compounding**: when projecting outcomes, vary the
    load-bearing assumptions on a grid (0/30/60/100% of each gap) instead of
    stacking worst cases (or best cases) into a single number.

## 6. Blind evaluation sandboxes (for LLM-as-subject experiments)
When agents must be evaluated on sequential decisions without future access:
- A GATE process serves one step at a time; the next step is served only
  after a committed decision carrying the serve-time NONCE; first terminal
  decision is binding; ground truth lives server-side, never served.
- The nonce chain is the audit: scoring requires a verified chain per episode.
- Fleet agents run with a least-privilege allowlist scoped to the gate
  command ONLY (never permission-bypass flags) — the "don't peek" rule then
  holds by construction, not by instruction.
- Verify the sandbox with scripted dummy agents (one per behavior class) and
  exactly one real agent before funding the fleet. Balance the case mix so
  the trivial constant policy can't score (e.g. 50/50 with the hard subclass
  tagged), and score against the dumb-baseline frontier (§5.2).

## 7. The documentation & memory system (the load-bearing core)
Premise: **context loss is routine, not exceptional.** Sessions compact, apps
restart, orchestrators get replaced mid-task. The project's continuity lives
on disk, in layers, each with a distinct WRITE trigger and READ moment. If a
future session can't reconstruct intent from files alone, the documentation
failed — "lost journals = lost days."

### 7.1 The layers (fast-decay → permanent)
| layer | granularity | written | read |
|---|---|---|---|
| `docs/daily/YYYY-MM-DD.md` | one file per day, sectioned per effort | START and END of EVERY session + after any change | when drilling into a specific day |
| `docs/daily/INDEX.md` | ONE line per day, newest first | same moment as the daily file | EVERY session start (read INDEX, not full journals) |
| `RESEARCH_JOURNAL.txt` (condensed long log) | 2-4 lines per day | end of session | reconstructing an arc across weeks |
| comms `NNN_*.md` (specs/verdicts) | one per task/ruling | at dispatch / at verdict | when resuming or auditing that task |
| research `reports/` | one per finding | when a result exists | whenever the topic resurfaces |
| MEMORY index (always auto-loaded) | ONE line per fact | when a durable fact/preference/pattern emerges | every session, automatically |
| memory detail files | one file per fact | with the index line | on-demand recall |
| `PROJECT_HISTORY` | era summaries | at condensation time | archaeology |

### 7.2 Journal write-discipline
- **Change report after ANY code change**: (1) what changed, (2) which files,
  (3) what to look for in the next run, (4) expected impact. This lets the
  next session read the report instead of reconstructing intent from diffs.
- **Findings carry receipts**: numbers + the artifact paths that produced
  them; quote the owner VERBATIM when their words are the spec ("<quote>") —
  paraphrase drifts, quotes don't.
- **Sequence matters**: journal the honest arc including dead ends and
  reversals (the $-story of a failure is often the most-cited entry later).
- **INDEX line = recall bait**: 2-3 lines max, packed with the terms future-
  you will grep for (names, numbers, verdicts). It is the ONLY surface most
  sessions will ever read — write it like the abstract of the day.
- **Tool outputs to FILE, always** — stdout dies with the terminal; scripts
  write their results into `reports/` so sessions can read them directly.
- **Backfill audits**: periodically ask "are journals up to date?" and check
  ALL layers — the common failure is one layer silently stalling (a daily
  file skipped on a busy day; the condensed log weeks behind).

### 7.3 Memory write-discipline (the always-loaded brain)
- **Two-tier**: an INDEX (one line per fact: hook + pointer) that is loaded
  every session, and DETAIL files (one fact each, with frontmatter: name /
  description / type). Never put content in the index; never let it grow —
  condense narrative back to one status line per topic.
- **What goes in**: durable facts a future session can't derive from the repo
  — owner preferences and protocol ("mode-first, short messages"), hard-won
  process rules with their WHY ("workers background-and-stop: shepherd
  them"), architecture invariants ("bar labeled B closes at B+period"),
  graveyard entries with measured costs, data locations, standing metaphors.
- **What stays OUT**: anything the repo already records (code structure, past
  fixes, git history) — memory points at those, never duplicates them.
- **Append with dates, never silently delete**: wrong memories get corrected
  by a dated superseding entry (the correction history is itself information).
- **Feedback memories carry Why + How-to-apply** so the rule transfers to new
  situations instead of being a dead quote.
- **Recalled memories are point-in-time**: verify a file/flag still exists
  before acting on an old memory (staleness is the default, not the
  exception).
- **Condensation ritual**: when the index bloats, collapse dated narrative
  into one-line statuses, move originals to an `archive/` with a date suffix,
  and keep a PROJECT_HISTORY for the long arc. Never let the always-loaded
  surface exceed what a session can afford to read.
- If memory lives in two copies (agent-private + repo-synced), EVERY edit
  goes to BOTH or a sync hook reverts it — document the sync mechanism.

### 7.4 The recall path (how a cold session boots)
1. Auto-loaded: WoW file + MEMORY index → the rules and the map.
2. Read `docs/daily/INDEX.md` top lines → what's in flight.
3. Read the ACTIVE roadmap + the newest comms docs for the current task.
4. Only then open code. (Total boot cost: a few hundred lines, by design.)

## 7b. Other operational rules
- **Baselines**: new best result → tag + safety branch + persist the exact
  artifacts/recipe to reproduce the number.
- **Versioned deliverables**: released = bare version; candidates = -RC;
  rejected = -RC.REJECTED kept as artifact. Each revision = its own file, so
  revisions can be A/B'd. A DEPLOY GATE: nothing is copied to the production/
  live target without explicit per-revision owner approval.
- **No magic numbers**: every constant is named, placed in config, and
  carries a comment with its origin.
- **Unit discipline**: ONE canonical internal unit per quantity, asserted at
  boundaries; human-facing conversions only at the reporting edge. (Unit
  dead-wires are silent killers — a reward/metric can be structurally zero
  or flat for weeks without erroring.)
- **Heavy runs are owner-launched**: the orchestrator preps and says "ready
  to run"; the owner presses the button (cost + accountability).
- **Restart drill**: background processes DIE with the host app. After any
  restart: relaunch fleets (make every fleet resume-safe: skip completed
  work), dedupe singleton daemons (two pollers on one token = message
  races), resume drones from their saved transcripts.
- **Watchers over polling**: completion notifications (or a cheap watcher
  script pinging the owner's phone) instead of the orchestrator polling; the
  owner chooses ping cadence (default: one ETA + one completion message).
- **Autonomous mode**: an explicit ordered night-queue from the owner; act
  only on established intent; decisions that are the owner's (budgets,
  deploys, architecture forks) get parked in a MORNING-DECISIONS list, never
  guessed. Leave a safety resume note/timer in case of context loss.

## 8. Collaboration protocol with the owner
- Critical collaborator, not a yes-man: challenge before agreeing, name the
  failure mode, commit to disagreements, propose alternatives, say no.
  Data beats intuition — including the reviewer's.
- Propose designs BEFORE building when the owner floats an idea; one question
  at a time; short messages (lead with the key point).
- Escalation override (<SUDO-word>): owner takes the wheel — state risks +
  one alternative FIRST, then execute their call.
- Report distributions the way the owner thinks (mode-first here); translate
  jargon into the owner's vocabulary in chat while code stays precise.
- Own errors loudly and in the record; never soften a null.

## 9. Skeletons (copy-paste)
**Task spec (comms/NNN):** Owner intent (verbatim quote) · Background/receipts
· Method (numbered) · What to HOLD constant · Deliverables (exact paths) ·
Pre-registered kill rule · Verify-then-STOP instruction · Ladder discipline
footer (sync, commit nothing, claim=evidence).

**Verdict (comms/NNN):** Claimed vs verified (reproduced numbers) · Deviations
ratified/rejected · RULING (pass/kill/narrow) · What changes downstream ·
Bookkeeping (files, commits).

**Daily journal:** Headline · per-effort sections (what/files/next/expected) ·
ops notes. INDEX line: `| DATE (headline) | tags | 3-line summary |`.

**Kill rule:** "X is retained only if [metric] beats [dumb baseline] on
[sealed holdout] with [CI excluding 0 / above effect floor]; otherwise the
lane is CLOSED and the graveyard gains an entry with the measured cost."
