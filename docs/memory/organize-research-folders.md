---
name: organize-research-folders
description: "Keep each research effort in ONE organized research/<topic>/ folder (code in pipeline/ builders/ tools/ subfolders + reports/ + a README); never flat, never mixed into the shared top-level reports/ — the user navigates by folder and gets lost otherwise. Set it up from the START."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a1bfcbce-37ff-4c61-931f-a5324a849c31
---

User (2026-06-22), after the F-space research sprawled across a flat `research/fspace_experiment/` + the shared
`reports/findings/`: *"you have a mess… have a folder in research where you keep everything related to this
research, and inside, folders for tools, the main research scripts, reports etc. organized — I open it and
there's a lot of stuff I get lost in."* (Then: "incorporate into instructions" + "propagate skill to Antigravity.")

**Why:** the user navigates the repo by FOLDER. A flat pile of scripts, or reports dumped into a shared dir, means
he can't find anything. Organization is a first-class requirement, not cosmetic.

**How to apply — every research effort, from the START (not as a later cleanup):**
- ONE dedicated `research/<topic>/` folder. Code in SUBFOLDERS: `pipeline/` (core engine), `builders/`
  (data/feature builders), `tools/` (analysis + orchestration). Plus `reports/` (findings `.md` + `assets/`)
  and a `README.md` index — what each script is, how to run, where the data lives.
- NEVER dump scripts flat in one folder; NEVER mix a project's reports into the shared top-level `reports/`
  (that dir holds *other* research — leave it).
- Large/gitignored data (parquets, `artifacts/`) stays at the repo root; reference it from the README; scripts
  stay repo-root-relative (run from root).
- Reorganizing later is a path-fixing refactor (sys.path depth, cross-imports, output paths, run-script paths) —
  cheap to do right up front, annoying to retrofit.

Canonical example: `research/fspace_cadence/` (pipeline/ builders/ tools/ reports/ + README). Codified in project
CLAUDE.md (Conventions) and the Gemini `research_discipline` skill v2 + `comms/CONTEXT_FOR_GEMINI.md`.
See [[report-distributions-and-mode]] (same spirit: make outputs legible to the user).

**REPORTS ROUTING (effective 2026-06-22):** Reports routing: research -> research/<topic>/reports/; training/baseline -> training/reports/; top-level reports/ = ONLY standalone reports tied to neither a research project nor a training run.
