# examples/ â€” Reference Docs

Standalone reference documents for the teacher-student northstar. Each doc stands
on its own; this README is only a wrapper/index. Read in the order below.

## Documents

1. **[projectteacherstudentnorthstar.md](../memory/project-teacher-student-northstar.md)** *(source of truth, repo root)*
   The program north star â€” the pipeline, the five Acts, the gated sequencing.
   Everything else here reviews or hardens this.

2. **[northstar_review.md](./northstar_review.md)**
   Architecture critique + novelty assessment. What's strong, what to push back on,
   and how novel the approach actually is (component vs. composition vs. governance).
   Points to the gate spec as its companion.

3. **[ride_edge_gate_spec.md](./ride_edge_gate_spec.md)** *(v2)*
   Statistical validity spec for the load-bearing ride-edge gate â€” power-checked,
   peek-proof, lockboxed, parity-aware. The whole distillation tower passes through here.
   Freeze its git commit hash before scoring: the hash IS the pre-registration.

4. **[hermes_memory_loop.md](./hermes_memory_loop.md)**
   Adaptation spec for a Hermes-style learning/memory loop (stable / context / journal /
   loop tiers + optional SQL+FTS5 store). Read-and-merge blueprint, not a fresh install.

## Reading paths

- **Evaluating the plan:** 1 â†’ 2 â†’ 3.
- **Implementing the gate:** 3 (standalone; 2 for context on why it matters).
- **Wiring project memory:** 4 (independent of 1-3).

## Convention

Docs are snake_case to match the repo. Cross-links between docs use relative paths,
so they resolve whether opened in VS Code, GitHub, or a plain markdown viewer.



## Operational note (2026-07-18)
The FROZEN pre-registration copy of the gate spec lives at
`research/dojo_forge/RIDE_EDGE_GATE_SPEC.md` (v2 + amendments v2.1/v2.2;
pre-reg tip = the latest commit touching that file). The copy here is the
readable reference; the frozen copy governs.
