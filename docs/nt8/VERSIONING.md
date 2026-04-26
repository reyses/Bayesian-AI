# NT8 Strategy Versioning Policy

**Effective: 2026-04-25** (refined 2026-04-25 evening).

## Rule

A version is **RELEASED** only when the user explicitly approves it for live
deployment. Until then, every version is a **Release Candidate (RC)**.

The `-RC` suffix is mandatory in code and docs for any non-released version.
Promotion to release = drop the suffix and bump the live deployment.

## File-naming convention (2026-04-25 evening, path updated 2026-04-26)

**Each version gets a version-suffixed filename** so multiple versions can
compile and run in parallel in NT8 (A/B testing). All NT8 source files
live under `docs/nt8/` (reorg 2026-04-26 — previously sat at the top of
`docs/` with an `NT8_` prefix that the folder name now conveys).

```
docs/nt8/ZigzagRunner_v1.0.cs           ← released, currently live
docs/nt8/ZigzagRunner_v1.2.cs           ← released, currently live
docs/nt8/ZigzagRunner_v1.5-RC.cs        ← future RC (not yet built)
```

Inside the .cs file:
- `public class ZigzagRunner_v1X[Y]` — class name carries the version so NT8
  treats it as a distinct strategy (otherwise compilation collides).
- `Name = "ZigzagRunner_vX.Y[-RC]"` — strategy display name in the picker
  also carries the version.
- `private const string VERSION = "X.Y[-RC]"` — version constant.

Result: in NT8 strategies dropdown, user can apply both `ZigzagRunner_v1.0`
and `ZigzagRunner_v1.2-RC` to different charts simultaneously, run A/B
parallel sessions, compare CSV logs head-to-head.

## Suffix grammar

| Suffix | Meaning |
|---|---|
| (none)                 | Released. Currently live. Only v1.0 qualifies as of 2026-04-25. |
| `-RC`                  | Release candidate. Built, tested in Python and/or NT8 sim, awaiting promotion decision. |
| `-RC.REJECTED`         | Candidate evaluated and rejected. File kept as research artifact only. Do NOT deploy. |
| `-RC.<n>`              | Optional micro-revision within the same RC family (e.g. `1.3.0-RC.2` after a syntax fix to `1.3.0-RC`). |

## Files that must carry the suffix

- The `private const string VERSION = "..."` constant in any `docs/nt8/*.cs`.
- The header docstring banner of the file.
- All CHANGELOG section labels.
- Daily journal entries when referring to the version.
- Findings docs when comparing version outcomes.

## Current status (2026-04-25, late session — version labels consolidated)

User instruction (2026-04-25): the granular v1.1/v1.2/v1.3 labels I had been
using overstated the deploy-candidate count. Consolidated:
v1.2-RC IS the next deploy candidate, combining trail + SL.

| Version | Status | File | Notes |
|---|---|---|---|
| **v1.0** | **RELEASED** | `Documents/NinjaTrader 8/.../ZigzagRunner.cs` | Live on Sim101 since 2026-04-24, Day 1 +$455. ~$833/day across 3 April days per user-measured NT8 PnL. Continues running in parallel with v1.2. |
| v1.0.1 | RC (safety patch — superseded by v1.2 release) | `docs/nt8/ZigzagRunner_v1.0.cs` | v1.0 + position-size hardening. Folded into v1.2's lineage. |
| **v1.1** | RC (CSV ledger — superseded by v1.2 release) | `docs/nt8/ZigzagRunner_v1.1.cs` (class `ZigzagRunner_v11`) | v1.0.1 + per-trade CSV ledger via `OnExecutionUpdate`. Folded into v1.2's lineage. |
| **v1.2** | **RELEASED** (2026-04-25, refactored 2026-04-26 → v1.2.6) | `docs/nt8/ZigzagRunner_v1.2.cs` (class `ZigzagRunner_v12`) | v1.1.1 + trailing stop + hard SL = −$50 (25 pts MNQ) + StagnationMonitor. Two-phase trail: arms at 10pt unrealized profit, 5pt floor, 10% of HWM beyond crossover. v1.2.6 (2026-04-26) extracted DynamicRiskManager + StagnationMonitor classes; CSV schema now 17 cols (added `max_neg_bars`). **Open issue**: 18.9% of "Stop loss" exits in 2026-04-26 Playback exceeded the 25pt cap (worst: −739pt) — caused by OnInitialFill timing moved out of OnExecutionUpdate + isSimulatedStop=true. See `docs/daily/2026-04-26.md` for fix plan. |
| v1.2-RC.REJECTED | DISCARDED | `docs/nt8/ZigzagRunner_v1.2-RC.REJECTED.cs` | Earlier v1.2 attempt that ALSO had hard SL=10pt. SL fired on bar-level noise; regression vs v1.0 by ~$68/day in current regime. Renamed to .REJECTED so it doesn't pollute the dropdown if accidentally compiled. |
| v1.4-RC.REJECTED | REJECTED | `docs/nt8/ZigzagRunnerHybrid.cs` | Hybrid 1m+5s timing. Disproved by Phase 2 backtest. Do NOT deploy. |
| v1.5-RC | DESIGN — postponed | (not yet written) | Was planned as v1.2-RC + filter. Now blocked: v1.2-RC itself is a regression. Filter on top of v1.0 directly is a different design — needs separate research. |

## Promotion workflow

1. RC built and self-tested (Python backtest, .cs compile, sim parity).
2. User reviews findings doc + risk assessment.
3. User says "promote v1.X-RC to release".
4. Drop `-RC` from VERSION constant + headers.
5. Increment a release-tag in git: `vX.Y.Z`.
6. Deploy to NT8 production folder.
7. Live VOE begins.
8. Previous live version archived to `docs/nt8/archive/<version>.cs`.

## Why this matters

Until 2026-04-25 the codebase had v1.0/v1.1/v1.2/v1.3/v1.4 listed as if all
were released versions. None were live except v1.0. The naming overstated
the completeness of the work. The `-RC` suffix corrects that signal: every
non-suffixed version means a user has approved live deployment.
