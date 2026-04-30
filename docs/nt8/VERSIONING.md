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
| (none)                 | Released. Currently live. As of 2026-04-26: v1.0 and v1.2 qualify. |
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
| **v1.0** | **RELEASED** (latest = v1.0.4) | `Documents/NinjaTrader 8/.../ZigzagRunner.cs` | Live on Sim101 since 2026-04-24, Day 1 +$455. ~$833/day across 3 April days per user-measured NT8 PnL. Iterated in place: v1.0.1 (safety hardening) → v1.0.2 (RideWithTrend) → v1.0.3 (PivotAction enums) → v1.0.4 (decouple exit from entry, structural bugfix). Continues running. |
| v1.0.1 | RC (safety patch — superseded by v1.2 release) | `docs/nt8/ZigzagRunner_v1.0.cs` | v1.0 + position-size hardening. Folded into v1.2's lineage. |
| v1.0.5-RC | RC — superseded by v1.0.5.1-RC | `docs/nt8/archive/` | Initial v1.0.5-RC release. Bug: AddDataSeries(BarsPeriodType.Second, 60) produced 2× pivots vs v1.0.4 (Minute × 1). Empirical: 1062 trades vs 506 in same Playback window. Source archived under `docs/nt8/archive/ZigzagRunner_v1.0.5-RC_pre_minute_fix_2026-04-27.cs`. |
| v1.0.5.1-RC | RC — superseded by v1.0.6-RC | `docs/nt8/ZigzagRunner_v1.0.5.1-RC.cs` (class `ZigzagRunner_v1051`) | v1.0.4 + hard stop + chart-TF independence + bar-type fix + Costs group. Playback test 2026-04-27 showed 1056 trades (2× v1.0.4) — bar-type fix did not reduce trade count. Trade-management improvements moved to v1.0.6-RC. Source kept for reference. |
| **v1.0.6-RC** | RC (trade management — SL + MFE-cut + 25% trail) | `docs/nt8/ZigzagRunner_v1.0.6-RC.cs` (class `ZigzagRunner_v106`) | v1.0.5.1-RC + three orthogonal exit rules from v1.0.4 EDA: **Rule 1** SL lowered 75→30pt (=−$60 MNQ, tighter risk). **Rule 2** `MfeCutBarsAfterEntry`=5 + `MfeCutThresholdUsd`=$5: at bar 5 if MFE so far ≤$5, exit (cuts ~10% of trades, mostly losers). **Rule 4** `TrailActivatePoints`=10 + `TrailGivebackPct`=0.25: once MFE≥$20, exit when current PnL ≤ MFE × 0.75. Each rule independently toggleable (set activator to 0 to disable). New CSV exit reasons: `TrailExitLong/Short`, `MfeCutLong/Short`. Genetic-optimized 32-day defaults (R=45, SL=90, MfeCut=17/$2, Trail=21/0.05) won the 32-day window (+$74/day) but produced -$75/day on full 14-month sweep — window-fit confirmed. |
| **v1.0.8-RC** | RC (LinReg slope filter + daily regime gate) | `docs/nt8/ZigzagRunner_v1.0.8-RC.cs` (class `ZigzagRunner_v108`) | v1.0.7-RC + two new SKIP-only filters validated by overnight EDA 2026-04-29: (1) **Entry slope filter**: skip a pivot entry if abs(LinReg slope) > threshold AND slope opposes the trade direction (validated +$33k IS / +$9.7k holdout). (2) **Daily regime gate**: at each daily bar close, compute 1D LinReg slope and decide allow/block via `DailyRegimeModeProp` enum (BlockBelowThreshold default = block any DOWN day per the v1.0.4 finding that DOWN regimes account for 101% of v1.0.4 losses). New BIP_DAILY=3 secondary added only when gate enabled. Both filters default OFF — opt-in via NT8 UI. Backwards compatible: with both filters off, behaves identical to v1.0.7-RC. Awaits NT8 SA validation before promote. |
| **BaseNmpRunner_v1.0-RC** | RC (NEW STRATEGY — DOWN/trending regime specialist) | `docs/nt8/BaseNmpRunner_v1.0-RC.cs` (class `BaseNmpRunner_v10`) | Native NT8 port of the BASE_NMP tier from `training/nightmare.py`. Entry: `\|z_se\| > 2.0 AND vr < 1.0` → fade the deviation. Exit: `\|z_se\| < 0.5` (mean reached) OR `vr > 1.0` (regime flip) OR hard SL. Computes z_se via OLS LinReg + StdDev of residuals; computes vr via Lo-MacKinlay `var(N-bar return) / (N × var(1-bar return))`. **Edge basis**: per overnight EDA 2026-04-29, BASE_NMP fired exclusively in late-2025/Q1-2026 trending regimes when other tiers (FADE_CALM, RIDE_AGAINST) stopped firing. Tier produced 1,195 trades, $19,997, $16.7/trade Python sim across Jan-Mar 2026. Optional LinReg slope filter (validated +$4,162 IS, +$809 70/30 holdout). Awaits NT8 Strategy Analyzer validation before promote. |
| **v1.0.7-RC** | RC (dynamic R via bar-rolling ATR) | `docs/nt8/ZigzagRunner_v1.0.7-RC.cs` (class `ZigzagRunner_v107`) | v1.0.6-RC + Option B dynamic R. Adds 5 [NinjaScriptProperty] under "Dynamic R" group: `UseDynamicR` (master switch, default false = v1.0.6-RC behavior), `AtrLookbackBars` (default 60 = 1h on 1m pivot), `AtrMultiplier` (default 5.0 placeholder, GA sweeps [0.5, 20.0]), `MinRPoints` (default 5 = floor on calm sessions), `MaxRPoints` (default 200 = ceiling on vol spikes). On each pivot bar close: `currentEffectiveR = clamp(ATR(N) * Multiplier, [MinR, MaxR])`. Zigzag state machine + plot R-trigger line both consume `currentEffectiveR`. Warmup falls back to static `RPoints`. Goal: scale R to intraday volatility — calm days narrow R catches more pivots, volatile days wider R filters noise. **Caveat**: GA on 12-D space (vs v1.0.6-RC's 7-D) increases overfit risk; 14-month walk-forward validation required before promote. Awaiting GA + Playback validation. |
| **v1.1** | RC (CSV ledger — superseded by v1.2 release) | `docs/nt8/ZigzagRunner_v1.1.cs` (class `ZigzagRunner_v11`) | v1.0.1 + per-trade CSV ledger via `OnExecutionUpdate`. Folded into v1.2's lineage. |
| **v1.2** | **RELEASED** (2026-04-25, refactored 2026-04-26 → v1.2.6) | `docs/nt8/ZigzagRunner_v1.2.cs` (class `ZigzagRunner_v12`) | v1.1.1 + trailing stop + hard SL = −$50 (25 pts MNQ) + StagnationMonitor. Two-phase trail: arms at 10pt unrealized profit, 5pt floor, 10% of HWM beyond crossover. v1.2.6 (2026-04-26) extracted DynamicRiskManager + StagnationMonitor classes; CSV schema now 17 cols (added `max_neg_bars`). **Open issue**: 18.9% of "Stop loss" exits in 2026-04-26 Playback exceeded the 25pt cap (worst: −739pt) — caused by OnInitialFill timing moved out of OnExecutionUpdate + isSimulatedStop=true. See `docs/daily/2026-04-26.md` for fix plan. |
| v1.2-RC.REJECTED | DISCARDED | `docs/nt8/ZigzagRunner_v1.2-RC.REJECTED.cs` | Earlier v1.2 attempt that ALSO had hard SL=10pt. SL fired on bar-level noise; regression vs v1.0 by ~$68/day in current regime. Renamed to .REJECTED so it doesn't pollute the dropdown if accidentally compiled. |
| **v1.3-RC** | RC (multi-TF risk machinery) | `docs/nt8/ZigzagRunner_v1.3-RC.cs` (class `ZigzagRunner_v13`) | v1.2 + 3 configurable secondary TFs added via AddDataSeries. Pivot TF (default 60s) drives entries/EOD/stagnation; Hard SL TF (default 5s) drives Initial+Tier1 stop evaluation; Trail TP TF (default 1s) drives Tier2 fast-ratchet. Tier2ActivatePoints now an explicit NinjaScriptProperty. DRM split into UpdateMaxPnlAndState + RouteStopForState(eligible). OnInitialFill back in OnExecutionUpdate. **Bugfix 2026-04-26 evening**: DRM now uses strategy-passed entryPrice + entryDir instead of Position.AveragePrice (which was stale during Strategy Analyzer multi-TF backtests, producing entry±5pt phantom stops). Awaiting Playback validation. |
| **v1.5-RC** | RC (chop-specialist 2-feature bleed-score filter) | `docs/nt8/ZigzagRunner_v1.5-RC.cs` (class `ZigzagRunner_v15`) | v1.4-RC + replaces single-feature `MaxMeanRange5dPts` filter with combined-z bleed-score classifier. `bleed_score = z(prior_range) + z(range_compression)`; if score > `BleedThresholdZ` (default -0.34, OOS-validated MVP), skip session. IS-calibrated on 1/2-3/1/2026 (N=48 days). 95-day backtest: converts -$552 unfiltered to +$5,021 filtered (z=-0.5) or +$5,214 (z=+0.75). 3-fold walk-forward: net +$1,272/test-period (positive but variable). Alt classifiers (decision tree, LR, AND-gates, hour-filter): all overfit IS, simple linear MVP wins. Caveat: filter HURTS in pure-chop regimes (T1 fold). Spec: `docs/JULES_v15_chop_specialist.md`. Findings: `reports/findings/2026-04-27_overnight_eda.md`. NOT yet deployed; awaiting validation backtest. |
| **v1.4-RC** | RC (regime filter) | `docs/nt8/ZigzagRunner_v1.4-RC.cs` (class `ZigzagRunner_v14`) | v1.3-RC + daily regime filter. Adds `AddDataSeries(BarsPeriodType.Day, 1)` to load prior daily OHLC. At the start of each new session, computes mean(high-low) across the last 5 daily bars. If `mean_range_5d > MaxMeanRange5dPts` (default 350pt), `tradeAllowedToday=false` and no new entries fire that session (existing positions still managed). Empirical basis: 100-day Spearman ρ(mean_range_5d, $/day) = -0.30; bottom-quartile of feature flips strategy from -$162/day to +$20/day at 41% Day WR. Logs each session's decision to `nt8_zigzag_v1.4_regime_log.csv`. Goal: NT8 backtest demonstrates filter skips bad days and improves overall PnL. |
| v1.4-RC.REJECTED | REJECTED | `docs/nt8/ZigzagRunnerHybrid.cs` | Hybrid 1m+5s timing. Disproved by Phase 2 backtest. Do NOT deploy. (Same version label, different file — kept under -RC.REJECTED suffix.) |
| v1.5-RC | DESIGN — postponed | (not yet written) | Was planned as v1.2-RC + filter. v1.4-RC now incorporates the filter idea. Re-scope this slot if/when needed. |

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
