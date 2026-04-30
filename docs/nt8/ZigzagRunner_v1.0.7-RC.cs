// =============================================================================
// ZigzagRunner 1.0.7-RC -- 2026-04-28  (Dynamic R via bar-rolling ATR)
// =============================================================================
//
// CHANGELOG 1.0.7-RC (2026-04-28 - Option B: dynamic R for genetic tuning):
//
//   USER REQUEST: "make R dynamic depending on the day". Static R=45 won the
//   genetic optimization on a 32-day Playback window but produces -$75/day on
//   the full 14-month dataset (window-fit). Hypothesis: optimal R should
//   scale with intraday volatility - calm sessions need smaller R to catch
//   pivots, volatile sessions need larger R to filter noise.
//
//   IMPLEMENTATION: bar-rolling ATR on the pivot series multiplied by a
//   tunable scalar, clamped to a floor/ceiling. All five parameters exposed
//   as [NinjaScriptProperty] so the genetic optimizer can search the dynamic
//   R surface directly. Backwards-compatible: UseDynamicR=false reproduces
//   v1.0.6-RC exactly.
//
//   NEW PROPERTIES (group "Dynamic R"):
//     - UseDynamicR (bool, default false): master switch. false = static
//       RPoints (v1.0.6-RC behavior). true = ATR-driven R per pivot bar.
//     - AtrLookbackBars (int, default 60): rolling window for ATR on pivot
//       series. 60 = last 60 pivot bars (1h on 1m pivot TF).
//     - AtrMultiplier (double, default 5.0): R = ATR x this scalar before
//       clamping. 5.0 is a placeholder; GA should sweep [0.5, 20.0].
//     - MinRPoints (double, default 5.0): floor on dynamic R. Prevents
//       degenerate behavior on low-vol sessions where R would otherwise
//       collapse and generate excess pivots.
//     - MaxRPoints (double, default 200.0): ceiling on dynamic R. Caps
//       runaway R during vol spikes.
//
//   COMPUTATION:
//     On each pivot bar close, currentEffectiveR = clamp(ATR(N)*k, [Lo, Hi]).
//     The zigzag state machine and chart-plot R-trigger line both consume
//     currentEffectiveR instead of the static RPoints field. During the
//     warmup period (CurrentBars[BIP_PIVOT] < AtrLookbackBars), falls back
//     to static RPoints.
//
//   CAVEATS:
//     - ATR is lagging: a fresh vol spike shows up only after AtrLookbackBars
//       fill. Use shorter lookback for faster adaptation, longer for stability.
//     - If UseDynamicR=true and AtrLookbackBars is too short relative to your
//       session length, the early session uses static fallback (v1.0.6-RC R).
//     - GA may still overfit a 32-day window; validate full 14-month sweep
//       before promote.
//
//   No change to entry logic, exit rules, EOD, entry-cutoff, plot wiring,
//   or pivot-detection state machine. Only the R threshold becomes dynamic.
//
// CHANGELOG 1.0.6-RC (REVISED 2026-04-28 — genetic-optimization defaults):
//
//   Strategy Analyzer Optimizer (genetic) on 32-day Playback window found
//   combo #1: $+3,491 gross / 589 trades / 71.6% WR / -$1,256 max DD =
//   +$74/day after $1.90/trade commission. BEATS v1.0.4 baseline (+$50/day).
//   These defaults reflect that combo:
//     RPoints=45 (was 30 — wider pivot reduces trade count to ~590)
//     MaxUnrealizedLossPoints=90 (= $180 SL)
//     MfeCutBarsAfterEntry=17, MfeCutThresholdUsd=$2 (late, strict)
//     TrailActivatePoints=21 (= $42 MFE), TrailGivebackPct=0.05 (5% — tight ratchet)
//   SlippagePoints=0.25 and CommissionPerRoundtripUsd=1.90 KEPT as realistic
//   defaults (optimizer used 0/0 to maximize gross — would mislead live).
//   CAVEAT: optimized on a single 32-day window. Validate OOS before live.
//
// CHANGELOG 1.0.6-RC (initial 2026-04-27, superseded above):
//
//   First Playback test (2026-04-27 with prior defaults: SL=30pt, Trail
//   activate=10pt) returned -$60/day on 32-day window — a regression vs
//   v1.0.4. Root cause: the prior defaults were aggressive vs what the EDA
//   sweep actually showed optimal. Specifically, SL=30pt was in the
//   catastrophic zone (kills 27% of winners) and Trail activate=10pt
//   (= $20 MFE) armed within first bars and strangled winners.
//
//   This revision aligns defaults with the EDA-derived optima:
//
// CHANGELOG 1.0.6-RC (2026-04-27 16:00 — trade management on v1.0.5.1-RC):
//
//   Three independent exit rules from v1.0.4 EDA findings:
//
//   * RULE 1 — Hard stop. Default MaxUnrealizedLossPoints = 50 (revised
//     2026-04-28 from 30). Caps single-trade loss at -$100 on MNQ.
//     EDA hard-stop sweep: 50pt is near the optimum (best 55pt @ +$155);
//     30pt was -$3,057 (catastrophic, kills 27% of winners); 75pt = -$671.
//
//   * RULE 2 — MFE-conditional cut. UNCHANGED, EDA-validated. Properties
//     MfeCutBarsAfterEntry=5, MfeCutThresholdUsd=$5. At N pivot-bars after
//     entry, if running MFE ≤ X dollars, exit. EDA: cuts 36L/12W, +$433/32d.
//     Set MfeCutBarsAfterEntry = 0 to disable.
//
//   * RULE 4 — 25% trail stop. REVISED 2026-04-28: TrailActivatePoints
//     bumped 10 → 50 (i.e., $20 → $100 MFE activation). Prior $20 trigger
//     armed within first bars on every winner-bound trade and exited at
//     $15 PnL — capture rate dropped to 0.4%. Winners' median MFE is $224,
//     so $100 activation lets trades develop. TrailGivebackPct unchanged
//     (0.25 = 25% giveback). Set TrailActivatePoints = 0 to disable.
//
//   Each rule independently toggleable. New CSV exit reasons: HardStopLong/Short,
//   MfeCutLong/Short, TrailExitLong/Short.
//
//   No change to entry logic, EOD, entry-cutoff, or pivot detection.
//
// CHANGELOG 1.0.5.1-RC (2026-04-27 14:00 — bar-type fix kept):
//   AddPivotOrSlSeries prefers Minute bars when cadence is multiple of 60s.
//
// =============================================================================
// Original v1.0.5.1-RC header below
// =============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// CHANGELOG 1.0.5.1-RC (2026-04-27 14:00 — bar-type bug fix on top of v1.0.5-RC):
//
//   * BUGFIX: empirically observed 2x trade count vs v1.0.4 in Playback at
//     default PivotTfSeconds=60. Root cause: NT8 treats
//     `BarsPeriodType.Second` x 60 as a DIFFERENT bar series than
//     `BarsPeriodType.Minute` x 1 even though both are 60-second cadence.
//     The Second-typed series produces extra pivots, possibly from
//     session-restart stub bars or different gap handling. Fix: in
//     State.Configure, when a TF cadence is a multiple of 60 seconds,
//     use Minute bars; otherwise Second bars. Same applies to HardSlTfSeconds.
//     Result: PivotTfSeconds=60 now produces v1.0.4-equivalent pivot
//     timing. Sub-minute cadences (e.g. 1s, 5s, 30s) still use Second.
//     Verified by Playback ledger: v1.0.5-RC produced 1062 trades vs
//     v1.0.4's 506 in same window before the fix.
//
//   * Cost-group additions kept (Slippage + Commission exposed as
//     dashboard properties for SA/Playback parity).
//
//   * NEW: Chart-TF INDEPENDENCE. Pivot timing is now driven by an explicit
//     `PivotTfSeconds` property (default 60 = 1 minute) via AddDataSeries.
//     The chart's primary TF is IGNORED — apply this strategy to any
//     chart (1s, 1m, 1h, daily) and the pivot logic always runs at the
//     configured `PivotTfSeconds` cadence. Default 60s reproduces v1.0.4
//     behavior exactly when the chart is 1m.
//
//   * NEW property `MaxUnrealizedLossPoints` (default 75 = -$150 on MNQ).
//     Hard cap on unrealized loss. When breached, position is flattened
//     immediately. 0 disables the stop entirely (= v1.0.4 behaviour).
//
//   * NEW property `HardSlTfSeconds` (default 1) — secondary data series
//     in seconds for stop-loss evaluation. Default 1s = check the cap
//     every second. Set to PivotTfSeconds to evaluate at pivot cadence
//     (slower; matches v1.0.4-style "next bar close" behavior).
//
//   * NEW: COSTS group exposed as strategy properties (motivation: NT8
//     Strategy Analyzer and Playback do NOT honor commission templates
//     identically — Playback often returns $0 commission on the trades
//     CSV. Baking these into the strategy makes results reproducible
//     across SA and Playback.):
//
//     - `SlippagePoints` (default 0.25 = 1 MNQ tick): driven into NT8's
//       built-in `Slippage` property in State.Configure. Applied to all
//       market-order fills by NT8's simulation engine. Same value used
//       in SA and Playback.
//
//     - `CommissionPerRoundtripUsd` (default 1.90 = $0.95/side × 2):
//       tracked internally by the strategy. Subtracted from gross
//       unrealized PnL when computing the effective hard-stop trigger
//       AND accumulated for an Output-panel session summary. NT8's
//       Trade Performance display still shows broker-template PnL
//       (independent of this property), but our internal accounting
//       reflects the configured commission.
//
//     The hard stop check is GROSS by design: if you want the SL to
//     fire at -$150 NET on MNQ, set MaxUnrealizedLossPoints = 76 (76pt
//     × $2 = $152 gross - $1.90 commission ≈ $150 net).
//
//   * Data series layout (BarsInProgress index → role):
//       0 = primary chart series — IGNORED. Strategy returns early on this.
//       1 = pivot series (Second × PivotTfSeconds) — pivot/EOD/entry-cutoff/exits.
//       2 = SL series    (Second × HardSlTfSeconds) — unrealized-PnL cap check.
//
//   * Why 75 points / -$150 default: per `docs/memory/tier_building_playbook.md`
//     §8c (Winner MAE), 97% of winners DIP NEGATIVE FIRST before becoming
//     winners. Hard stops at -$50 (25 pts) kill ~25% of winners; -$150
//     (75 pts) kill ~0.2%. Sweet spot is the knee of the kill-rate curve.
//
//   * NEW exit reasons in CSV: HardStopLong / HardStopShort.
//
//   * Backward-compat: with PivotTfSeconds=60, MaxUnrealizedLossPoints=0,
//     HardSlTfSeconds=60 the strategy is logically equivalent to v1.0.4
//     (the SL check returns immediately; pivot cadence matches 1m chart).
//
//   * RC GATE: this file is RC. Class name is ZigzagRunner_v105 (not
//     ZigzagRunner) so it appears as a SEPARATE strategy in NT8's picker
//     and does NOT replace the live v1.0.4. Promote by:
//       1. Validate via NT8 Strategy Analyzer over a clean window.
//       2. User explicit approval to drop -RC.
//       3. Replace bin/Custom/Strategies/ZigzagRunner.cs with this file
//          (renamed back to ZigzagRunner.cs + class ZigzagRunner).
//       4. Bump VERSIONING.md status row from RC to RELEASED.
//
// CHANGELOG 1.0.4 (2026-04-27) -- STRUCTURAL BUGFIX (kept in v1.0.5-RC):
//   * Always exit current position BEFORE any entry decision. Closed a
//     v1.0.0-1.0.3 bug where exits were an accidental side effect of
//     opposite-direction entries. See archive for full v1.0.4 changelog.
//
// CHANGELOG 1.0.3 (2026-04-27): Per-pivot direction control via
//   OnHighPivot / OnLowPivot enums.
//
// CHANGELOG 1.0.2 (2026-04-26): Direction RideWithTrend bool.
//   Superseded by v1.0.3.
//
// CHANGELOG 1.0.1 (2026-04-25): Position-size hardening (panic-close,
//   idempotent entries, explicit exit-before-entry).
//
// LOGIC (1.0.5-RC):
//   1. Strategy runs on ANY chart TF — primary series is ignored.
//   2. Pivot detection runs at PivotTfSeconds cadence on a secondary series.
//      Zigzag with R-point retracement threshold drives pivot confirmations.
//   3. When a pivot is CONFIRMED:
//      - HIGH pivot -> action defined by OnHighPivot (Long, Short, or Skip)
//      - LOW  pivot -> action defined by OnLowPivot
//   4. Past Entry Cutoff UTC: no new entries. Existing position held until
//      EOD or hard stop.
//   5. EOD force-close at configurable UTC time (pivot series cadence).
//   6. Every HardSlTfSeconds seconds, check unrealized PnL. If unrealized
//      loss reaches MaxUnrealizedLossPoints, flatten now.
//
// INSTALLATION (1.0.5-RC, RC GATE — does NOT auto-deploy live):
//   1. File lives in docs/nt8/ZigzagRunner_v1.0.5-RC.cs (this file).
//   2. To run in NT8: copy to bin/Custom/Strategies/.
//   3. F5 in NT8 NinjaScript Editor.
//   4. Apply ZigzagRunner_v1.0.5-RC to a chart (any TF works).
//   5. Defaults match v1.0.4 behavior at 1m + adds the 75-pt hard stop.
//   6. To DISABLE the hard stop: MaxUnrealizedLossPoints=0.
//
// =============================================================================

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;            // Brushes (for AddPlot colors)
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;   // ATR (v1.0.7-RC)
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ZigzagRunner_v107 : Strategy
    {
        // ── Settings ─────────────────────────────────────────────────────

        [NinjaScriptProperty]
        [Display(Name = "R (points)", Description = "Zigzag retracement threshold in price points (30 = MNQ $60)", Order = 1, GroupName = "Zigzag")]
        public double RPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contracts", Description = "Contracts per trade", Order = 2, GroupName = "Zigzag")]
        public int Contracts { get; set; }

        // ── Pivot timeframe (v1.0.5-RC) ──────────────────────────────────
        // Decouples pivot timing from chart TF. Strategy works on any chart.

        [NinjaScriptProperty]
        [Range(1, 86400)]
        [Display(Name = "Pivot TF (seconds)",
                 Description = "Cadence (in seconds) for the pivot-detection series. " +
                               "Default 60 = 1 minute (matches v1.0.4 behavior on a 1m chart). " +
                               "Strategy is independent of chart TF — it always uses this value.",
                 Order = 3, GroupName = "Zigzag")]
        public int PivotTfSeconds { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Hour UTC", Description = "Force close hour (UTC, 0-23). 20 = before NYSE close", Order = 1, GroupName = "Schedule")]
        public int EodHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Minute UTC", Description = "Force close minute (0-59)", Order = 2, GroupName = "Schedule")]
        public int EodMinuteUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Hour UTC", Description = "No new entries after this time (UTC)", Order = 3, GroupName = "Schedule")]
        public int EntryCutoffHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Minute UTC", Description = "No new entries after this minute (0-59)", Order = 4, GroupName = "Schedule")]
        public int EntryCutoffMinuteUtc { get; set; }

        // ── Per-pivot direction control (v1.0.3) ─────────────────────────

        [NinjaScriptProperty]
        [Display(Name = "On High Pivot", Description = "Action when a HIGH pivot is confirmed. Default Short = v1.0 counter-trend.", Order = 1, GroupName = "Direction")]
        public PivotAction_v107 OnHighPivot { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "On Low Pivot", Description = "Action when a LOW pivot is confirmed. Default Long = v1.0 counter-trend.", Order = 2, GroupName = "Direction")]
        public PivotAction_v107 OnLowPivot { get; set; }

        // ── Risk (v1.0.5-RC) ─────────────────────────────────────────────

        [NinjaScriptProperty]
        [Range(0, 500)]
        [Display(Name = "Max Unrealized Loss (points)",
                 Description = "Rule 1: Hard stop. If unrealized PnL drops to -X points, flatten. " +
                               "0 = disabled. Default 90 = -$180 on MNQ (genetic-optimized 2026-04-28).",
                 Order = 1, GroupName = "Risk")]
        public double MaxUnrealizedLossPoints { get; set; }

        [NinjaScriptProperty]
        [Range(1, 86400)]
        [Display(Name = "Hard SL TF (seconds)",
                 Description = "Secondary series cadence (in seconds) for unrealized-loss check. " +
                               "Default 1 = check every second. Set to PivotTfSeconds to match pivot cadence.",
                 Order = 2, GroupName = "Risk")]
        public int HardSlTfSeconds { get; set; }

        // ── Rule 2: MFE-conditional cut (v1.0.6-RC) ──────────────────────

        [NinjaScriptProperty]
        [Range(0, 60)]
        [Display(Name = "MFE Cut Bars After Entry",
                 Description = "Rule 2: at N pivot-bars after entry, if running MFE has not exceeded " +
                               "MfeCutThresholdUsd, exit immediately. 0 = disabled. " +
                               "Default 17 (genetic-optimized).",
                 Order = 3, GroupName = "Risk")]
        public int MfeCutBarsAfterEntry { get; set; }

        [NinjaScriptProperty]
        [Range(0, 200)]
        [Display(Name = "MFE Cut Threshold (USD)",
                 Description = "Rule 2: if running MFE ≤ this USD value at MfeCutBarsAfterEntry, exit. " +
                               "Default $2 (genetic-optimized — strict signal of upside).",
                 Order = 4, GroupName = "Risk")]
        public double MfeCutThresholdUsd { get; set; }

        // ── Rule 4: Trail stop (v1.0.6-RC) ───────────────────────────────

        [NinjaScriptProperty]
        [Range(0, 200)]
        [Display(Name = "Trail Activate (points)",
                 Description = "Rule 4: when running MFE reaches this many points (USD = pts × 2), " +
                               "trail arms. 0 = disabled. Default 21 (= $42 USD, genetic-optimized).",
                 Order = 5, GroupName = "Risk")]
        public double TrailActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 1.0)]
        [Display(Name = "Trail Giveback Pct",
                 Description = "Rule 4: once trail armed, exit when current PnL ≤ MFE × (1 - this). " +
                               "Default 0.05 = 5% giveback (genetic-optimized — very tight ratchet).",
                 Order = 6, GroupName = "Risk")]
        public double TrailGivebackPct { get; set; }

        // ── Costs (v1.0.5-RC) — exposed for SA/Playback parity ───────────

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "Slippage (points)",
                 Description = "Per-fill slippage in points applied by NT8's simulation engine. " +
                               "MNQ tick = 0.25 pts ($0.50). Default 0.25 = 1 tick worse than mid. " +
                               "Drives base.Slippage in State.Configure so SA and Playback agree.",
                 Order = 1, GroupName = "Costs")]
        public double SlippagePoints { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 20.0)]
        [Display(Name = "Commission per round-trip (USD)",
                 Description = "Per round-trip commission in USD per contract. " +
                               "Default 1.90 = $0.95/side × 2 (NinjaTrader Brokerage Free template). " +
                               "Tracked internally; printed in session summary. " +
                               "Does NOT affect order placement or NT8's Trade Performance display.",
                 Order = 2, GroupName = "Costs")]
        public double CommissionPerRoundtripUsd { get; set; }

        // ── Dynamic R (v1.0.7-RC) ────────────────────────────────────────
        // Master switch + 4 tunables for ATR-driven R. UseDynamicR=false
        // reproduces v1.0.6-RC behavior exactly (R = static RPoints).
        // All five are [NinjaScriptProperty] so the GA can search them.

        [NinjaScriptProperty]
        [Display(Name = "Use Dynamic R",
                 Description = "Master switch. false = static RPoints (v1.0.6-RC behavior). " +
                               "true = R is recomputed each pivot bar as ATR(N) * Multiplier, " +
                               "clamped to [MinRPoints, MaxRPoints].",
                 Order = 1, GroupName = "Dynamic R")]
        public bool UseDynamicR { get; set; }

        [NinjaScriptProperty]
        [Range(2, 1440)]
        [Display(Name = "ATR Lookback (bars)",
                 Description = "Rolling window for ATR on the pivot series. 60 = last 60 pivot " +
                               "bars (1h on 1m pivot TF). Shorter = faster adaptation, more noise. " +
                               "During warmup (CurrentBars < this) falls back to static RPoints.",
                 Order = 2, GroupName = "Dynamic R")]
        public int AtrLookbackBars { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 20.0)]
        [Display(Name = "ATR Multiplier",
                 Description = "R = ATR * this scalar before clamping. 5.0 placeholder; GA should " +
                               "sweep [0.5, 20.0]. On 1m pivot bars typical ATR(60) is 3-10 pts so " +
                               "multiplier 5 lands R in the 15-50 range.",
                 Order = 3, GroupName = "Dynamic R")]
        public double AtrMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(0.25, 100.0)]
        [Display(Name = "Min R (points)",
                 Description = "Floor on dynamic R. Prevents collapse on low-vol sessions. " +
                               "Default 5.0 (= $10 on MNQ). Set close to 1 tick for aggressive " +
                               "behavior, higher to require minimum move size.",
                 Order = 4, GroupName = "Dynamic R")]
        public double MinRPoints { get; set; }

        [NinjaScriptProperty]
        [Range(10.0, 1000.0)]
        [Display(Name = "Max R (points)",
                 Description = "Ceiling on dynamic R. Caps runaway R during vol spikes. " +
                               "Default 200.0 (= $400 on MNQ). Lower to force more pivots " +
                               "even on volatile days.",
                 Order = 5, GroupName = "Dynamic R")]
        public double MaxRPoints { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0.7-RC";

        // ── BarsInProgress index constants (readability) ─────────────────
        private const int BIP_PRIMARY = 0;  // chart's own series — IGNORED
        private const int BIP_PIVOT   = 1;  // PivotTfSeconds series
        private const int BIP_SL      = 2;  // HardSlTfSeconds series

        // ── Zigzag state (driven by pivot series) ────────────────────────
        // direction: 0 = undefined (before first pivot), +1 = up leg, -1 = down leg
        private int direction;
        private double extremePrice;
        private int extremeBarIdx;
        private int lastPivotDir;      // +1 = high pivot (just formed), -1 = low pivot; 0 = none yet
        private double lastPivotPrice;

        // ── Dynamic R state (v1.0.7-RC) ───────────────────────────────────
        // ATR is initialized in State.DataLoaded (after secondary series exist).
        // currentEffectiveR is recomputed each pivot bar close in OnBarUpdate
        // and consumed by the zigzag state machine + plot updates.
        private ATR atr;
        private double currentEffectiveR;

        // ── Cost tracking (v1.0.5-RC) ────────────────────────────────────
        private int    closedRoundtripsCount;
        private double estimatedCommissionTotalUsd;

        // ── Plot accessors (NT8 convention) ──────────────────────────────
        [System.Xml.Serialization.XmlIgnore]
        [Browsable(false)]
        public Series<double> Extreme  { get { return Values[0]; } }
        [System.Xml.Serialization.XmlIgnore]
        [Browsable(false)]
        public Series<double> RTrigger { get { return Values[1]; } }

        // ── Per-trade tracking (v1.0.6-RC) ───────────────────────────────
        // Reset on each new entry (detected via Position transition Flat→non-Flat).
        // Updated on every SL-series tick.
        private int    currentTradePivotBars;     // pivot-bars elapsed since entry
        private int    currentTradeSlTicks;       // SL-series ticks elapsed since entry
        private double currentTradeMfeUsd;        // running max unrealized PnL (USD)
        private bool   trailArmed;                // true once MFE >= TrailActivate
        private MarketPosition lastSeenPosition;  // for entry-detection edge

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "ZigzagRunner_v1.0.7-RC";
                Description                      = "Zigzag pivot-retracement w/ dynamic R + hard stop + MFE-cut + trail. Chart-TF independent. v" + VERSION;
                Calculate                        = Calculate.OnBarClose;
                EntriesPerDirection              = 1;
                EntryHandling                    = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy     = true;
                ExitOnSessionCloseSeconds        = 30;
                IsFillLimitOnTouch               = false;
                MaximumBarsLookBack              = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution              = OrderFillResolution.Standard;
                // base.Slippage is driven from SlippagePoints in State.Configure
                Slippage                         = 0;
                StartBehavior                    = StartBehavior.WaitUntilFlat;
                TimeInForce                      = TimeInForce.Gtc;
                TraceOrders                      = false;
                RealtimeErrorHandling            = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling               = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade              = 2;

                // Defaults — REVISED 2026-04-28 from genetic optimization (combo #1):
                // Top result: $+3,491 gross, 589 trades, 71.6% WR, +$74/day after $1.90 comm
                // (vs v1.0.4 baseline +$50/day). Optimizer searched the full risk surface
                // including R, SL, MFE-cut, and Trail params on 32-day Playback window.
                RPoints                          = 45.0;   // Was 30. Wider pivots = fewer-but-higher-quality
                                                            // setups. Reduces trade count from ~1100 to ~590.
                Contracts                        = 1;
                PivotTfSeconds                   = 60;     // 1 minute = v1.0.4 baseline
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;
                OnHighPivot                      = PivotAction_v107.Short;  // counter-trend
                OnLowPivot                       = PivotAction_v107.Long;   // counter-trend

                // v1.0.6-RC risk defaults — genetic-optimization output (2026-04-28)
                MaxUnrealizedLossPoints          = 90.0;   // Rule 1: -$180 SL.
                HardSlTfSeconds                  = 1;      // check every second

                // Rule 2 (MFE-conditional cut) — late check, strict threshold
                MfeCutBarsAfterEntry             = 17;     // bar 17 (17 min on 1m pivot TF)
                MfeCutThresholdUsd               = 2.0;    // need only >$2 MFE by bar 17

                // Rule 4 (trail stop) — early activation, very tight giveback
                TrailActivatePoints              = 21.0;   // arm trail when MFE >= $42 USD
                TrailGivebackPct                 = 0.05;   // exit when PnL <= MFE * 0.95 (5% giveback)

                // v1.0.5-RC cost defaults — exposed for SA/Playback parity
                SlippagePoints                   = 0.25;   // 1 MNQ tick
                CommissionPerRoundtripUsd        = 1.90;   // $0.95/side x 2 (NinjaTrader Brokerage Free)

                // v1.0.7-RC dynamic R defaults — UseDynamicR=false reproduces
                // v1.0.6-RC behavior exactly (R = static RPoints). Flip to true
                // and let the GA tune AtrMultiplier in [0.5, 20.0].
                UseDynamicR                      = false;  // OFF by default (backwards-compat)
                AtrLookbackBars                  = 60;     // 60 pivot bars = 1h on 1m
                AtrMultiplier                    = 5.0;    // R ~= ATR(60) * 5 (placeholder, GA tunes)
                MinRPoints                       = 5.0;    // floor 5pt = $10 MNQ
                MaxRPoints                       = 200.0;  // ceiling 200pt = $400 MNQ

                // ── Chart plots (v1.0.6-RC) ─────────────────────────────────
                // Plot 0: current zigzag extreme price (orange).
                // Plot 1: R trigger level (cyan) — price at which the next
                //         pivot would confirm if reached this bar.
                // Using the simple AddPlot(Brush, name) overload to avoid
                // Stroke/DashStyleHelper assembly-reference issues.
                AddPlot(Brushes.Orange, "Extreme");
                AddPlot(Brushes.Cyan,   "RTrigger");
            }
            else if (State == State.Configure)
            {
                // Drive NT8's built-in Slippage from our exposed property.
                // This is what NT8 actually uses for fill price adjustment
                // in both SA and Playback — settable per-run/per-optimization.
                Slippage = SlippagePoints;

                // Add the two secondary series in fixed order to keep
                // BarsInProgress indices stable: 1 = pivot, 2 = SL.
                //
                // BUGFIX 2026-04-27: NT8 treats `Second x 60` differently
                // from `Minute x 1` (the Second-typed series produced 2x
                // trade count vs v1.0.4 in Playback). When a cadence is
                // a multiple of 60s, prefer Minute bars to match v1.0.4
                // semantics exactly. Sub-minute cadences must use Second.
                AddPivotOrSlSeries(PivotTfSeconds);   // BIP 1 = pivot
                AddPivotOrSlSeries(HardSlTfSeconds);  // BIP 2 = SL

                direction       = 0;
                extremePrice    = double.NaN;
                extremeBarIdx   = -1;
                lastPivotDir    = 0;
                lastPivotPrice  = double.NaN;

                // Cost-tracking init
                closedRoundtripsCount       = 0;
                estimatedCommissionTotalUsd = 0.0;

                // Per-trade tracking init (v1.0.6-RC)
                currentTradePivotBars  = 0;
                currentTradeSlTicks    = 0;
                currentTradeMfeUsd     = 0.0;
                trailArmed             = false;
                lastSeenPosition       = MarketPosition.Flat;

                // v1.0.7-RC: seed effective R with static RPoints. Will be
                // overwritten on each pivot-bar close if UseDynamicR=true.
                currentEffectiveR      = RPoints;
            }
            else if (State == State.DataLoaded)
            {
                // v1.0.7-RC: instantiate ATR on the pivot series. Must be in
                // DataLoaded (NOT Configure) because BarsArray[BIP_PIVOT] is
                // only populated AFTER the secondary series are attached.
                // ATR returns true range in price points — exactly the unit
                // our R threshold uses, so no conversion needed.
                if (UseDynamicR)
                {
                    atr = ATR(BarsArray[BIP_PIVOT], AtrLookbackBars);
                }
            }
            else if (State == State.Terminated)
            {
                // Print session-end cost summary so user sees the impact
                // of the configured Costs settings regardless of NT8's
                // template-based Trade Performance display. Pull native
                // trade count from NT8 if available so we count completed
                // round-trips authoritatively.
                int rtCount = closedRoundtripsCount;
                if (SystemPerformance != null && SystemPerformance.AllTrades != null)
                {
                    rtCount = SystemPerformance.AllTrades.Count;
                }
                double estCommission = rtCount * CommissionPerRoundtripUsd;

                Print(string.Format(
                    "[v1.0.7-RC COST SUMMARY] roundtrips={0}  est_commission=${1:F2}  " +
                    "slippage_pts={2:F4}  per_roundtrip=${3:F2}  " +
                    "use_dynamic_r={4}  atr_lookback={5}  atr_mult={6:F2}  min_r={7:F1}  max_r={8:F1}",
                    rtCount, estCommission, SlippagePoints, CommissionPerRoundtripUsd,
                    UseDynamicR, AtrLookbackBars, AtrMultiplier, MinRPoints, MaxRPoints));
            }
        }

        protected override void OnBarUpdate()
        {
            // ─── 1.0.1 SAFETY GUARD (runs on every series update) ───────────
            // If position size somehow exceeded Contracts (stale order, manual
            // intervention, NT8 framework bug), panic-close immediately.
            if (Math.Abs(Position.Quantity) > Contracts)
            {
                Print("ZigzagRunner_v107 SAFETY: Position.Quantity=" + Position.Quantity +
                      " exceeds Contracts=" + Contracts + ", panic-closing.");
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "SafetyPanicLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "SafetyPanicShort", "");
                return;
            }

            int bip = BarsInProgress;

            // ─── Primary chart series: PLOT UPDATE + IGNORE for trading ─────
            // Strategy logic is chart-TF independent (only secondaries used for
            // pivots/SL). On primary bars we only update chart plots.
            if (bip == BIP_PRIMARY)
            {
                if (!double.IsNaN(extremePrice))
                {
                    Values[0][0] = extremePrice;   // current zigzag extreme
                    if (direction != 0)
                    {
                        // R trigger = price at which a pivot would CONFIRM:
                        //   direction +1 (up leg, tracking HIGH)  -> trigger = extreme - R
                        //   direction -1 (down leg, tracking LOW) -> trigger = extreme + R
                        // v1.0.7-RC: uses currentEffectiveR (dynamic if enabled).
                        Values[1][0] = direction > 0
                            ? extremePrice - currentEffectiveR
                            : extremePrice + currentEffectiveR;
                    }
                    else
                    {
                        Values[1][0] = extremePrice;  // no direction yet
                    }
                }
                return;
            }

            // ─── 1.0.6-RC: TRADE MANAGEMENT on SL series ────────────────────
            // Three rules evaluated each SL-series tick (default 1s cadence).
            // Position-state-machine: detect new entries via Flat→non-Flat
            // transition, reset per-trade tracking on entry.
            if (bip == BIP_SL)
            {
                if (CurrentBars[BIP_SL] < 1) return;
                MarketPosition pos = Position.MarketPosition;

                // Reset on flat
                if (pos == MarketPosition.Flat)
                {
                    if (lastSeenPosition != MarketPosition.Flat)
                    {
                        // Just exited a position; reset tracking
                        currentTradePivotBars = 0;
                        currentTradeSlTicks   = 0;
                        currentTradeMfeUsd    = 0.0;
                        trailArmed            = false;
                    }
                    lastSeenPosition = MarketPosition.Flat;
                    return;
                }

                // New entry detected (Flat -> Long/Short)
                if (lastSeenPosition == MarketPosition.Flat)
                {
                    currentTradePivotBars = 0;
                    currentTradeSlTicks   = 0;
                    currentTradeMfeUsd    = 0.0;
                    trailArmed            = false;
                }
                lastSeenPosition = pos;

                currentTradeSlTicks++;
                double refPrice = Closes[BIP_SL][0];
                double unrealizedUsd = Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Currency, refPrice);
                double unrealizedPts = Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Points, refPrice);

                // Update running MFE (max unrealized PnL seen during this trade)
                if (unrealizedUsd > currentTradeMfeUsd)
                    currentTradeMfeUsd = unrealizedUsd;

                // ── RULE 4: 25% trail stop (highest priority — captures peak) ──
                if (TrailActivatePoints > 0)
                {
                    double activateUsd = TrailActivatePoints * 2.0; // MNQ $2/pt × 1 contract assumption
                    if (!trailArmed && currentTradeMfeUsd >= activateUsd)
                    {
                        trailArmed = true;
                    }
                    if (trailArmed)
                    {
                        double trailExitThresholdUsd = currentTradeMfeUsd * (1.0 - TrailGivebackPct);
                        if (unrealizedUsd <= trailExitThresholdUsd)
                        {
                            if (pos == MarketPosition.Long)
                                ExitLong(Position.Quantity, "TrailExitLong", "");
                            else
                                ExitShort(Position.Quantity, "TrailExitShort", "");
                            return;
                        }
                    }
                }

                // ── RULE 1: Hard stop on unrealized loss ──
                if (MaxUnrealizedLossPoints > 0 && unrealizedPts <= -MaxUnrealizedLossPoints)
                {
                    if (pos == MarketPosition.Long)
                        ExitLong(Position.Quantity, "HardStopLong", "");
                    else
                        ExitShort(Position.Quantity, "HardStopShort", "");
                    return;
                }

                // ── RULE 2: MFE-conditional cut at bar N ──
                // Fires once when SL-tick count crosses the equivalent pivot-bar threshold.
                // Convert pivot-bar count to SL-tick count: bar N = N × PivotTfSeconds ticks (assuming 1-tick = 1 SL-second).
                if (MfeCutBarsAfterEntry > 0)
                {
                    int triggerSlTicks = MfeCutBarsAfterEntry * PivotTfSeconds / Math.Max(HardSlTfSeconds, 1);
                    if (currentTradeSlTicks == triggerSlTicks)
                    {
                        if (currentTradeMfeUsd <= MfeCutThresholdUsd)
                        {
                            if (pos == MarketPosition.Long)
                                ExitLong(Position.Quantity, "MfeCutLong", "");
                            else
                                ExitShort(Position.Quantity, "MfeCutShort", "");
                            return;
                        }
                    }
                }

                return;
            }

            // ─── PIVOT series: ALL pivot/EOD/entry-cutoff logic ─────────────
            if (bip != BIP_PIVOT) return;
            if (CurrentBars[BIP_PIVOT] < BarsRequiredToTrade) return;

            // v1.0.7-RC: refresh effective R for this pivot bar BEFORE the
            // zigzag state machine reads it. During ATR warmup or when
            // UseDynamicR=false, ComputeEffectiveR returns static RPoints.
            currentEffectiveR = ComputeEffectiveR();

            // Increment pivot-bar counter for in-trade diagnostics
            if (Position.MarketPosition != MarketPosition.Flat)
                currentTradePivotBars++;

            double c = Closes[BIP_PIVOT][0];
            DateTime barUtc = Times[BIP_PIVOT][0].ToUniversalTime();
            int minsOfDay      = barUtc.Hour * 60 + barUtc.Minute;
            int eodMins        = EodHourUtc * 60 + EodMinuteUtc;
            int entryCutMins   = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

            // ─── EOD force-close ───────────────────────────────────────
            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "EodExitLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "EodExitShort", "");
                return;
            }

            // ─── Initialize extreme on first pivot bar ─────────────────
            if (double.IsNaN(extremePrice))
            {
                extremePrice  = c;
                extremeBarIdx = CurrentBars[BIP_PIVOT];
                return;
            }

            // ─── Zigzag state machine ──────────────────────────────────
            bool pivotConfirmed = false;
            int newPivotDir     = 0;   // +1 = high pivot, -1 = low pivot

            // v1.0.7-RC: use currentEffectiveR (dynamic if enabled) for all
            // R-threshold comparisons. Frozen for the duration of this bar.
            double r = currentEffectiveR;

            if (direction == 0)
            {
                // No direction yet — first R-retracement defines the first leg
                if (c - extremePrice >= r)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // extreme was a LOW pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
                else if (extremePrice - c >= r)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // extreme was a HIGH pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
            }
            else if (direction == +1)
            {
                // In an UP leg — watch for new highs, otherwise R-retracement confirms HIGH pivot
                if (c > extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBars[BIP_PIVOT];
                }
                else if (extremePrice - c >= r)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // high pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
            }
            else  // direction == -1 (down leg)
            {
                if (c < extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBars[BIP_PIVOT];
                }
                else if (c - extremePrice >= r)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // low pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
            }

            if (!pivotConfirmed) return;
            lastPivotDir = newPivotDir;

            // ─── Entry cutoff (v1.0 behavior: skip entry, no exit) ─────
            if (minsOfDay >= entryCutMins) return;

            // ─── 1.0.4: STRUCTURAL FIX — decouple exit from entry ────────
            // Every pivot ALWAYS exits the current position FIRST.
            PivotAction_v107 action;
            string pivotLabel;
            if (newPivotDir == +1)
            {
                action     = OnHighPivot;
                pivotLabel = "HighPivot";
            }
            else
            {
                action     = OnLowPivot;
                pivotLabel = "LowPivot";
            }

            // ALWAYS exit current position first (whatever direction it is).
            if (Position.MarketPosition == MarketPosition.Long)
                ExitLong(Position.Quantity, "PivotExitLong", "");
            else if (Position.MarketPosition == MarketPosition.Short)
                ExitShort(Position.Quantity, "PivotExitShort", "");

            // Skip = stay flat after the exit above. No new entry fires.
            if (action == PivotAction_v107.Skip)
                return;

            if (action == PivotAction_v107.Short)
                EnterShort(Contracts, "ShortAt" + pivotLabel);
            else  // PivotAction_v107.Long
                EnterLong(Contracts, "LongAt" + pivotLabel);
        }

        // ─── Helpers ────────────────────────────────────────────────────────

        /// <summary>
        /// Add a secondary data series for pivot or SL evaluation, choosing
        /// the bar period type that best mirrors v1.0.4's primary 1m series.
        ///
        /// NT8 distinguishes BarsPeriodType.Second vs BarsPeriodType.Minute
        /// even when the cadence is identical (60s == 1min). Empirically the
        /// Second-typed series produces extra pivots vs Minute (likely from
        /// session-restart stub bar handling). To match v1.0.4 trade counts:
        ///   - Cadences that are exact multiples of 60s and ≤ 1440 minutes
        ///     are added as `Minute × N`.
        ///   - Sub-minute cadences (1s, 5s, 30s, etc.) MUST use `Second × N`.
        /// </summary>
        private void AddPivotOrSlSeries(int seconds)
        {
            if (seconds >= 60 && (seconds % 60) == 0 && (seconds / 60) <= 1440)
                AddDataSeries(BarsPeriodType.Minute, seconds / 60);
            else
                AddDataSeries(BarsPeriodType.Second, seconds);
        }

        /// <summary>
        /// v1.0.7-RC: compute the effective R for the current pivot bar.
        /// Returns static RPoints when UseDynamicR=false OR during ATR warmup.
        /// Otherwise: ATR[0] * AtrMultiplier, clamped to [MinRPoints, MaxRPoints].
        /// Called at the top of each BIP_PIVOT bar in OnBarUpdate.
        /// </summary>
        private double ComputeEffectiveR()
        {
            if (!UseDynamicR) return RPoints;
            if (atr == null) return RPoints;
            if (CurrentBars[BIP_PIVOT] < AtrLookbackBars) return RPoints;
            double atrPts = atr[0];
            if (double.IsNaN(atrPts) || atrPts <= 0) return RPoints;
            double dynR = atrPts * AtrMultiplier;
            if (dynR < MinRPoints) dynR = MinRPoints;
            if (dynR > MaxRPoints) dynR = MaxRPoints;
            return dynR;
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // PivotAction_v107 (v1.0.5.1-RC, 2026-04-27)
    // ─────────────────────────────────────────────────────────────────────
    // Per-pivot-type direction action. Public + version-suffixed name so
    // it can coexist with v1.0.4's PivotAction_v10 in the same compiled
    // assembly (NT8 compiles all Strategies/*.cs into one DLL).
    // ═════════════════════════════════════════════════════════════════════
    public enum PivotAction_v107
    {
        Long,    // EnterLong on this pivot type
        Short,   // EnterShort on this pivot type
        Skip,    // No entry on this pivot type
    }
}
