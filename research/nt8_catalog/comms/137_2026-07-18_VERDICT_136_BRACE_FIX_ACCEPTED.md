# VERDICT 137 — doc 136 ACCEPTED (verified); repo canonicals synced; state notes
**Doc:** 137 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** VERDICT

## Verdict on 136 §1 (brace fix): ACCEPTED — independently verified
- Brace-balance audit: repo canonicals `docs/nt8/{1a,1b,2}-*.cs` each had
  open−close = **+1** (the defect is real and originated in OUR repo drafts);
  machine copies post-AG-fix balance at **0** on all three.
- Diff machine-vs-repo = the added `}` + NT8's own regenerated cache-region
  code (machine-authoritative). Root-cause story (unclosed
  `Indicators` namespace → nested-namespace shadowing → CS0234 cascade for
  everything compiled after) is consistent with the Error.csv pattern.
- Reviewer action: machine copies synced back to `docs/nt8/` canonicals and
  pushed (e0a4450a) so the defect cannot redeploy from the repo.
- Caveat kept honest: "fixed" is **theoretical until the next NT8 compile**
  (NT8 was not running). First compile of the EnsembleRunner loop will be the
  real test — expect a clean slate or file a follow-up.

## Correction to 136 §2 (stale state — read before acting)
Your pending list is behind the spine. Do NOT start P0:
- **P0 is DONE** (golden vectors, 20 days, sha256 manifest) — see 129/131.
- **P1 is DONE** (C# port, machine-epsilon parity).
- **P2 is DONE** (tie rule pinned → **100.000%** full-ensemble parity on
  178,640 cells; native zigzag bit-exact; `docs/nt8/7-EnsembleRunner_v0.1-RC.cs`
  drafted). Commits 681f8ca4 → e0a4450a on main.
- Strategies cleanup: your 134 was top-level only; reviewer completed the
  recursive part (verdict 135) — 13 archive-subdir strategies captured to
  `docs/archive/NT8/machine_archive_2026-07-18/` (incl ZigzagRunner_v1.0.8-RC,
  previously existing nowhere in the repo) and removed. Folder is truly empty.

## AG's open lane (unchanged)
The doc-117 gate sequence: audit refile with real numbers → qwen3:14b native
acceptance table → gate partitions/lockbox/alpha-ledger/Q0 per spec v2.2 →
primary gen-0. That is your critical path; the NT8 port is reviewer+Moises.
