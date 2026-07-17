# Exit Dojo

A "testing grounds" where an LLM agent replays a real historical trade entry
frame-by-frame (1-minute steps) and commits `HOLD`/`EXIT` decisions with reasons.
This is **hypothesis GENERATION for exit rules** — it is NOT a live decider, and a
pilot run through this dojo is NOT a sealed-test result. Any exit rule an agent
appears to discover here must be codified in code and pass the repo's sealed
2024 (train) / 2025-26 (test) harness before anyone believes it. See the
**Leakage caveat** below for why pilot scores specifically cannot be trusted as
performance numbers.

## What's here

```
research/exit_dojo/
  README.md                    this file
  builders/episode_builder.py  builds episodes/ + episodes/truth/ + reports/selection_table.md
  tools/score_decisions.py     scores an agent transcript against a truth sidecar
  episodes/ep_NN.md            the 10 pilot episode packets (agent-facing)
  episodes/truth/ep_NN.json    ground truth sidecars (scorer-only, NEVER agent-facing)
  reports/selection_table.md   how/why each of the 10 episodes was picked
  reports/decisions/ep_NN.txt  an agent's raw HOLD/EXIT transcript for episode NN
  reports/pilot_scorecard.md   scorer output across whatever's in reports/decisions/
```

## How to run

```
python research/exit_dojo/builders/episode_builder.py [--seed N]
```
(Re)builds the 10 pilot episodes + truth sidecars + selection table. Deterministic
given a seed (default baked into the script). Takes a few seconds (scans the whole
2025-26 test-split decile-9 fire pool, then builds frame text for the 10 selected).

To play an episode: give an LLM agent the full contents of `episodes/ep_NN.md` in
one message (see "Leakage caveat" — this pilot is intentionally single-prompt) and
save its raw output to `reports/decisions/ep_NN.txt`.

```
python research/exit_dojo/tools/score_decisions.py
```
Scores every `reports/decisions/ep_*.txt` present against its `episodes/truth/ep_*.json`
sidecar and writes `reports/pilot_scorecard.md`.

## Where the data lives (all pre-existing, read-only inputs)

- `research/nt8_catalog/reports/econ_drift_rows.parquet` — candidate ENTRY fires
  (`ts, day, det, is_long, P, decile, split, drift_1m..60m, trunc_1m..60m`). We use
  `split=='test' & decile==9` rows from 2025/2026 days as the entry pool.
- `research/nt8_catalog/reports/signal_rows_{EXITKMDR,TURNCLIMAX,TURNHA,PROPTURNP}.parquet`
  — the 4 AUX fire streams behind each frame's `fires last 3m` field.
- `DATA/ATLAS/5s/<day>.parquet` — the only price source used to build frames
  (`timestamp, open, high, low, close, volume`).
- `DATA/ai_cusp_picks/ai_picks_YYYY-MM-DD_multi.json` — hindsight "oracle" trade
  labels (`entry_ts, exit_ts, direction, ...`). Used **ONLY** for (a) stratifying
  the pilot selection into winner/mid-flip/instant-fail buckets, and (b) the
  ground-truth sidecar's `label_end_minute`/`oracle_capture`. **Never** copied
  into a packet — packets only ever contain relative price deltas + ratios, never
  an absolute date, price, or label.

All four generator scripts (`dossier_signal_pipeline.py`, `propturn_p_tune.py`) that
produce these parquets live in `research/nt8_catalog/tools/` and are cited by line
number in `builders/episode_builder.py`'s docstrings wherever their logic is reused
or mirrored.

## Design choices worth knowing about

**Sign convention.** Verified empirically against `DATA/ATLAS/5s` (not assumed):
`econ_drift_rows.drift_Xm = sign * (close_future - close_now)`, `sign = +1` if
`is_long` else `-1` — i.e. FAVORABLE-SIGNED, positive always means "good for the
position taken." Every point-delta emitted into a frame (`px` and the 1m-bar
O/H/L/C) follows this same convention for internal consistency, and it is stated
explicitly in each episode's header so the playing agent knows it up front.

**Leg geometry** (`leg: age Xm, amp Ypts, giveback Z%`) mirrors the running-pivot /
running-extreme / amplitude / giveback state machine in
`research/nt8_catalog/tools/dossier_signal_pipeline.py::_propturn_core` (lines
1389-1434), reusing its frozen STATIC constants (`PROPTURN_R=0.05`,
`PROPTURN_S_MIN=3.0min`, `PROPTURN_A_MIN=15.0pts`). `track_leg_state()` in the
builder is a plain-Python re-implementation (the original is numba-jitted and only
returns emitted fires; the dojo needs a per-bar snapshot for descriptive display,
not fires). **Giveback can exceed 100%** for legs whose amplitude is below
`PROPTURN_A_MIN` — that's the original tracker's "escape clause" territory (a full
15pt countermove, not a fraction of a small amplitude, is what re-designates a
sub-minimal leg), faithfully carried over, not a display bug. `ep_09`/`ep_10`
(chop episodes) show this clearly — small legs, giveback readings well over 100%.

**ER10** = Kaufman efficiency ratio on 1-minute closes, N=10, and **vol(5m)** = std
of the last 60 5s closes (ddof=1) — both copied verbatim from
`dossier_signal_pipeline.py::gen_ctx_er` / `_pp_arrays` (lines 912/1532/1538).

**Declared deviations from the literal task spec** (flagging for reviewer visibility):
1. **One episode per real day.** Not in the literal spec, but necessary: an
   unconstrained scan let a single volatile day supply 8 of the first 10 matches
   (all bucket types), which defeats the point of a "2025-26" stratified pilot.
   `select_episodes()` now caps at one selected episode per distinct real day, so
   all 10 episodes come from 10 different days (see `reports/selection_table.md`).
2. **Chop tolerance widened from +-3pts to +-4pts.** An empirical sweep (see the
   comment on `CHOP_TOL_PTS` in the builder) found only **1 of 282** decile-9
   test-split days has *any* 15-minute stretch within +-3pts; +-4pts unlocks 9
   days, +-5pts unlocks 26. This is a real property of the entry pool — decile-9
   (highest-confidence) fires essentially never stay this flat — not a search bug.
   +-4pts is the minimal widening that still allows picking 2 diverse chop days
   rather than being forced to reuse the same single day twice.
3. **Chop is a sliding window, not anchored at t=0.** "Drift stayed within the
   tolerance for 15+ min" is tested over *any* 15-consecutive-minute stretch in the
   episode, not required to start at entry — a decile-9 entry that moves initially
   then stalls is just as legitimate a "chop" exit-drilling scenario as one that's
   flat from minute 1, and anchoring strictly at t=0 made the bucket unfindable.
4. **"Oracle (label-end) capture"** = the favorable-signed drift at
   `label_end_minute` (first minute the `ai_cusp_picks` hindsight-optimal direction
   flips to the OPPOSITE of the entry direction), or at the window's last frame if
   the label never flips. This is the literal "vs oracle (label-end) capture" the
   scorer spec asks for — a *different*, deliberately simpler notion of "oracle"
   than "best possible exit price in the window," which the ground-truth sidecar
   does not separately track.

## Leakage caveat (read before trusting any pilot score)

Pilot episodes are played **single-prompt** — the agent receives all frames in one
message with a sequential-commitment contract ("process frames in order, commit
before looking further"). Attention mechanically **can** see future frames even
though the contract asks the agent not to reason about them, so pilot scores are
**optimistic** and are used **only** for hypothesis generation. Any rule an agent
appears to discover must be codified in code and pass the sealed 2024 (train) /
2025-26 (test) harness before it's believed. A true stepwise-blind runner (the
agent physically cannot see frame `t+1` until it has committed at frame `t`) is a
later build, only worth building if measured LLM exit performance is ever wanted
for its own sake.

## Full run -- CAUSAL telescope sandbox (stepwise-blind by construction)

The pilot above is single-prompt (leakage-caveated). The full run is a true stepwise-blind
sandbox: frames are served ONE AT A TIME by a gate that will not reveal frame k+1 until a
decision on frame k has been committed with the serve-time random nonce. An agent
mechanically cannot look ahead.

```
builders/telescope_packet_builder.py   samples 200 episodes + builds per-episode telescope packets
tools/dojo_gate.py                     the sandbox: next / commit / finish (nonce-gated, append-only)
tools/dojo_fleet.py                    drives real Sonnet agents (claude -p) through the gate, N-parallel
tools/dojo_dummy_agent.py              scripted python agent (verification only)
tools/score_full_run.py                verifies the nonce chain + scores captured-vs-refs + wrong-side speed
reports/full_run/
  selection.json / selection_table.md  the 200-episode manifest + why each was picked
  packets/<eid>.json                   agent-facing telescope frames (gate serves these)
  truth/<eid>.json                     ground-truth sidecar (scorer-only, NEVER served)
  gate_state/<eid>.{state.json,transcript.jsonl,agent_stdout.txt}   per-episode play record
  scorecard.md                         scored results (audit-gated)
```

Run order:
```
python3.11 research/exit_dojo/builders/telescope_packet_builder.py            # select 200 + build all packets
python3.11 research/exit_dojo/tools/dojo_fleet.py --episodes 200 --parallel 4 # play them (reviewer-gated)
python3.11 research/exit_dojo/tools/score_full_run.py                         # audit + score
```

**Telescope layout** (memory `telescope-nested-cadence`): frame 0 is the full wide field
(every TF's last-closed V2 feature block, grouped by TF, + per-TF last-closed OHLC in
points-from-entry + the pilot local-state block); frames k>=1 re-emit a TF's block ONLY
when that TF's bar has closed since the previous frame. **Causality law** (asserted per-TF,
per-frame, hard build failure on violation): every row/bar in a frame at wall time
`frame_ts` satisfies `row_ts + period <= frame_ts` at EVERY TF including the 5s base layer
(the `-period` shift -- the phold-anchor bug must not recur). Verified against raw parquet:
54/54 independent per-TF checks pass, packet OHLC matches raw recompute exactly.

**Sampling**: 200 episodes, 60/60/40/40 winner/midflip/instantfail/chop over the phold
engagement population (P>=p90(train), 60s/day/dir de-dup), one distinct 2025-26 day each,
excluding the 10 pilot days. **Declared deviation**: chop is graduated to +-6pt tolerance
(the minimal ladder rung reaching 40 distinct chop days; +-4pt yields only 9), extending
the pilot's own +-3->+-4 widening -- high-confidence entries essentially never stay flat.

**The audit**: the nonce chain proves sequential blind play for the served path (every
commit carries, in order, the exact serve-time nonce; gapless 0..k prefix; one terminal
EXIT). Residual risk (an agent reading raw ATLAS itself) is forbidden by the agent
instructions and, with `dojo_fleet.py --trace`, captured in the stream-json tool trace.
The graduation firewall stands regardless: no dojo number is ever itself a result -- any
rule must be codified and pass the sealed 2024/2025-26 harness before belief.

## Git / size note

Unlike the `research/nt8_catalog/reports/*.parquet` catalog outputs (gitignored
via the repo's blanket `*.parquet` rule), `episodes/*.md`, `episodes/truth/*.json`,
and `reports/*.md`/`reports/decisions/*.txt` are **not** covered by any existing
`.gitignore` pattern and would be tracked as-is. No new ignore rules were added for
them (per the task spec) — they're kept intentionally small instead: 10 episodes
of ~41 frames each is on the order of 100KB total, and truth sidecars are a few KB
each.
