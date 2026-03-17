# Prompt for Jules: Audit Improvements - 2026-03-17

Please execute the following improvements based on the latest project audit (`AUDIT/AUDIT_2026_03_17.md`):

1. **Fix Duplicate 'Active' Directory**: Merge `docs/active/` into `docs/Active/` (or vice-versa, depending on desired casing) and ensure all active specs are placed there.
2. **Move Top-Level Markdown Files**: Move `DEEP_RESEARCH_ENTRY_IMPROVEMENTS.md` and `DMI_ADX_IMPROVEMENT_SPEC.md` from the root directory into `docs/specs/` or `docs/Active/` as appropriate.
3. **Organize Docs**: Move the loose `JULES_*.md` files from `docs/` into `docs/specs/` or `docs/Active/`.
4. **Evaluate 'examples'**: Check if the `.png` files in `examples/` should be ignored by git or if their contents are essential, and handle them accordingly.
5. **Clean Tools Directory**: Create a `tools/research/` or `scripts/research/` directory and move all `tools/research_*.py` and standalone analysis scripts there to clean up the `tools/` root directory.
6. **Gitignore Tracking**: Verify that `.gitignore` properly excludes `reports/*.txt`, `reports/*.csv`, `reports/*.png`, and `tools/plots/`, and run `git rm --cached` on any tracked generated artifacts if necessary, as requested in previous audits.

Ensure all file structure changes and cleanups are reflected in documentation updates where appropriate.
