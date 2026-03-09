# Agent Instructions

## For Jules (VS Code)
- Read `docs/ROADMAP.md` and `docs/CHANGELOG.md` for project context
- Check `docs/Active/` for current task specs
- Update `docs/daily/YYYY-MM-DD.md` at session end
- Accumulate learnings in `.Jules/bolt.md` and `.Jules/palette.md`

## For Claude Code
- Reads `CLAUDE.md` automatically (project root)
- Check `docs/Active/` for current task specs
- Update `docs/daily/YYYY-MM-DD.md` at session end

## For Claude (chat — claude.ai)
- Architecture review, spec writing, research direction
- Does not modify code directly
- Outputs go to `docs/` or research specs

## Shared Rules
- No physics metaphors in new code (statistical/regression language only)
- CUDA-only (no CPU fallback)
- Surgical updates to MEMORY.md (append history, update navigation)
- Progress bars (tqdm) mandatory for loops > 100 iterations
- Never run training — tell user to run manually
