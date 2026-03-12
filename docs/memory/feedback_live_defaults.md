---
name: Live launcher defaults
description: Default live launcher mode should be real trading (NT8 account controls sim/real). --dry-run is opt-in for observation only.
type: feedback
---

## Live Launcher Defaults

- Default mode = send orders to NT8. NT8's account setting (sim vs real) controls risk.
- `--dry-run` = observation only (no orders sent at all). This is opt-in, not the default.
- Don't suggest --dry-run when user wants to trade with sim money — that's NT8's job.
- The replay validation should not block sim accounts either — only matters for real money.
