# The Self-Healing Control Plane — system doctrine (2026-07-24/25)

Owner: "the daemon self heal — that's the mindset we should follow."
Ratified across two domains in one day: the Telegram bridge (drilled live) and
the trading stack (designed, partially built). This doc is the durable form.

## The pattern
Never one smart monolith. Layered:
1. **Cheap fast sensors** — heartbeat files; the logit p-stream; the PnL
   excursion path.
2. **Mechanical detection** — strike counters, control-chart limits. Dumb,
   deterministic, auditable. (tg-verify stale-strikes; p-stream ambiguity/
   jitter limits; excursion strike levels.)
3. **Escalation to intelligence ONLY on anomaly** — repair Sonnet on
   restart-failure; reasoning-mode teacher on out-of-control frames; Opus
   interrogation when fixed probes are ambiguous. Escalation rate is capped
   and REPORTED (an escalation storm is itself a finding).
4. **Audit everything** — health.log, memory ledger, sprint history. A
   decision that isn't ledgered didn't happen. (Paid off same-day: sprint-4's
   18 silent-looking rejections were a one-query diagnosis.)
5. **Drill every link before trusting it** — plant real faults; watch the
   machinery fix them; a chain is drilled only when every link has fired for
   real (spawn syntax AND spawn survival were separately broken links).
   Drills end with a RESTORE CHECKLIST walked and verified.

## Trading-side instantiation (built/designed 2026-07-24)
- **p-stream control chart** (built): gen-0 exited on noise — 85% of its 734
  exits fired on out-of-control frames (ambig/jitter/flip); GENOME v1
  stabilized the process (escalation 34%→3%). Hybrid economics: consult the
  reasoning layer on ~3% of frames; every trigger-pull reviewed.
- **Excursion strike ladder** (designed): −25pts = wake the analyzer
  (oscillation vs anchors; verdict LOGGED — riding unanalyzed is negligence
  even when riding is right); −50 = re-analyze + notify the human while
  salvageable (backtest: 50pt-salvage → +$5k/20d — needs CI receipt);
  mechanical floor (catastrophic stop) BELOW the measured salvage zone; only
  the floor acts, strikes escalate attention. ATR-scale the strikes.
- **Mind-vs-world cross-check**: p-stream watches the mind, excursion watches
  the world; disagreement between them is itself a strike.
- **Conflict rule**: the mechanical floor always wins; intelligence may add
  caution or propose deviation within the leash (policy v1.2 asymmetry).

## The memory loop as the same pattern (built 2026-07-25, owner-designed)
Trade with knowledge → write memos AS HYPOTHESES (writer knows the curator is
coming) → episode-end REFLECTION looks back on the full tape and curates WITH
the same knowledge (educated curator) → keep only causally-warranted claims
("| BECAUSE: <tape evidence>") → mechanical guards demoted to safety+backstop
(day-agnostic, allowlist, cap, dedup) → every verdict ledgered.
Reflection IS the guard; knowledge IS part of the curation mechanism;
assumptions become proven or disproven — the scientific method in the loop.

## Standing infra lessons (memory: feedback-background-watchers)
consume-only-after-verified-delivery · never bare-`wait` behind a backgrounded
server · pattern strings live in FILES never ad-hoc shell calls · systemd-run
for children of oneshot units · one loop at a time, field-check before
relaunch · drills end with restore checklists.
