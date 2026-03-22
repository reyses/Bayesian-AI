---
name: Backup critical files before destructive operations
description: Always copy untracked pkl/checkpoint files before --fresh or any operation that overwrites them
type: feedback
---

Before running `--fresh` or any operation that rebuilds/overwrites checkpoint files, ALWAYS copy critical untracked files to a safe location.

**Why:** Lost the oracle brain's matching pattern_library.pkl when `--fresh` overwrote it. Had to restore from OneDrive version history. The pre_is_backup only captured the NEW files, not the old ones.

**How to apply:**
- Before `--fresh`: copy `checkpoints/pattern_library.pkl`, `clustering_scaler.pkl`, `template_tiers.pkl`, `is_brain_checkpoint.pkl`, `live_brain.pkl` to `checkpoints/backup_YYYYMMDD/`
- Before any pkl-overwriting operation: same
- Name backups with date so they're identifiable
- These files are gitignored (*.pkl, checkpoints/) so git can't recover them
