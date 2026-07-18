# VERDICT 135 — doc 134 PARTIAL; reviewer completed the cleanup
**Doc:** 135 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** VERDICT + completion of record

## Verdict on 134: PARTIAL (good evidence discipline, incomplete scope)
What AG did right: process check, SHA256 source/dest tables pasted, archive
committed on main (5e6b973e), top level verifiably empty. The evidence format
is exactly what the protocol asks for — credit where due.

What it missed: **NT8 compiles the entire `bin\Custom\Strategies` tree
recursively.** The pre-existing `archive\` subdir (2026-06-19) still held
**13 strategy .cs files in the compile path** — the ZigzagRunner v1.x lineage,
BaseNmpRunner_v1.0-RC, VWAP_MTF_Rev12, MyCustomStrategy{,1}. "0 File(s)" at
the top level ≠ "folder clean" for the stated goal (only the new strategy
lives). The report's final claim ("directory is now empty and ready") was
therefore wrong in effect.

## Material find during completion
- `ZigzagRunner_v1.0.8-RC.cs` (66,107 bytes, 2026-04-28) existed **nowhere in
  the repo** — docs/nt8/archive has three files *named* "superseded_by_v1.0.8"
  but v1.0.8 itself was never captured. The machine copy was the only copy.
- All other 12 machine copies differ byte-wise from their repo namesakes
  (the machine bytes are what actually compiled/ran) — only
  ZigzagRunner_v1.2.cs was byte-exact in the repo.

## Completion of record (reviewer)
1. Copied all 13 files → `docs/archive/NT8/machine_archive_2026-07-18/`.
2. SHA256 verified: 13/13 OK, 0 mismatches (pasted in session log).
3. Committed on main, then removed `Strategies\archive\` from the machine.
4. Final state: `bin\Custom\Strategies` contains nothing — ready for
   EnsembleRunner_v0.1-RC deploy (which still requires Moises' per-revision
   approval before any copy; the gate is untouched).

## Protocol note for AG
When a cleanup task says "so only X lives", the acceptance test is the
*recursive* content of the target, not the top-level listing. Same class of
error as the top-level-only inventory in earlier cycles: the claim must be
tested at the boundary the system actually reads (here: NT8's recursive
compile glob).
