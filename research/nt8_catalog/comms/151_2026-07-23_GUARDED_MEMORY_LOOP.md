# SPEC — The Guarded Memory Loop (teacher memory + Sonnet gatekeeper)
**Doc:** 151 · **Date:** 2026-07-23 · **Author:** Claude (Fable) · **Design:** Moises (via Telegram) · **Status:** PROPOSED (builds after exam day + handbook gate)

## The idea (owner, assembled from tonight's thread)
The teacher explores PRIOR EVERYTHING — prior days' market data AND its own prior
trades — through the memory scaffolding, BUT it does not get free read access:
it must ASK a **Sonnet gatekeeper** ("the adjudicator") for information, and the
gatekeeper guards what is released. "We can use adjacent-ish days, but we need
the guard so we can guard it."

## Architecture
```
qwen (trader)
   │  REQUEST: "what did I trade on similar setups?" / "yesterday's range?"
   ▼
Sonnet GATEKEEPER (librarian + firewall)
   │  1. MECHANICAL rules (code, not judgment): time-causality (nothing with
   │     timestamp >= episode start), lockbox exclusion, day-burn state check
   │  2. JUDGMENT (Sonnet): ambiguous-request adjudication, relevance filtering,
   │     COMPRESSION of granted material to a token-budgeted brief
   ▼
teacher_memory.db (SQL FTS form factor, doc 149) + prior-day market store
   │
   ▼  GRANT: vetted, compressed, causal-only brief → injected as RELEVANT MEMORY
```
Every request AND grant/denial is LOGGED to an audit ledger — the teacher's
information-seeking behavior is itself a new data channel (what does a good
trader ask for?), and the ledger is the leakage proof for the gate record.

## Guard rules (hard, mechanical — code enforces, Sonnet cannot override)
1. **Time-causality:** nothing with timestamp ≥ the requesting episode's start.
2. **Lockbox:** held-out days are invisible — requests touching them return
   DENIED(lockbox) and are ledgered.
3. **Day-burn state:** memory branches follow doc-149's lifecycle (snapshot →
   branch on OOS → day-agnostic harvest → whole-day burn → graduate).
4. **Budget:** grants are compressed to ≤N tokens (default 400) per request,
   ≤K requests per episode (default 3) — minimal-sufficient-context applies to
   memory too.

## Sonnet's judgment scope (the soft layer)
- Interpret vague asks ("anything like this before?") into concrete queries.
- Rank/filter retrieved memos + prior-day slices for decision relevance.
- Compress grants into the token budget without editorializing the decision.
- REFUSE-and-log asks that smell like fishing for outcome data ("how did this
  day end?" mid-episode) even when technically causal-ambiguous.

## Cost + sequencing
- Sonnet cost bounded: ≤3 requests/episode × ~1-2k tokens ≈ cents/episode.
- Prior-day exploration uses adjacent-ish TRAINING-side days first (owner:
  conservation rule; no new lockbox burns). The 4 in-curriculum adjacent pairs
  (doc 149) are the free pilot.
- BUILD ORDER: (1) exam day completes under current protocol → (2) handbook
  genome gate run → (3) MEMO v1 + SQL store → (4) gatekeeper (this doc) →
  (5) day-carry natural experiment (WITH vs COLD on the 4 pairs).

## GUARD v2 — mechanical-first (owner simplification, 2026-07-23)
Owner: "mechanically only allow lookback up to a certain amount of days so they
are always out of reach." Adopted, with one correction: a lookback cap ALONE
does not protect an INTERLEAVED lockbox (a held-out day falling inside the
window would be visible). Fully-mechanical guard, three layers, zero judgment:
1. **Store admission**: the prior-day store NEVER INGESTS lockbox/burned days —
   protection by construction; there is nothing to leak.
2. **Lookback cap**: retrieval limited to the last N trading days (default
   N=10) behind the episode's day.
3. **Timestamp wall**: nothing with timestamp ≥ episode start (intraday guard).
Sonnet's role SHRINKS to optional librarian work (relevance ranking, grant
compression) — a quality layer, not a security layer; v1 can ship without it
(plain FTS top-k), Sonnet added later if grants prove noisy. Request/grant
ledger retained (audit + the information-seeking data channel).

## Open questions (owner)
1. Request format: free-text asks (Sonnet interprets) vs a fixed menu of query
   types (cheaper, more auditable)?
2. Should DENIED requests be visible to qwen (it learns the boundaries) or
   silent (it can't probe the lockbox's shape)?
3. ≤3 requests/episode default — right number?
