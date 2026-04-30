// =============================================================================
// ZigzagRunner 1.0.5-RC -- 2026-04-27  (ADD: hard stop + chart-TF independence)
// =============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// CHANGELOG 1.0.5-RC (2026-04-27):
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
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ZigzagRunner_v105 : Strategy
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
        public PivotAction_v105 OnHighPivot { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "On Low Pivot", Description = "Action when a LOW pivot is confirmed. Default Long = v1.0 counter-trend.", Order = 2, GroupName = "Direction")]
        public PivotAction_v105 OnLowPivot { get; set; }

        // ── Risk (v1.0.5-RC) ─────────────────────────────────────────────

        [NinjaScriptProperty]
        [Range(0, 500)]
        [Display(Name = "Max Unrealized Loss (points)",
                 Description = "Hard stop: if unrealized PnL drops to -X points, flatten. " +
                               "0 = disabled. Default 75 = -$150 on MNQ. " +
                               "Per playbook §8c: 75pt kills 0.2% of winners; 25pt kills 25%. " +
                               "Optimizer-friendly range: 0..500.",
                 Order = 1, GroupName = "Risk")]
        public double MaxUnrealizedLossPoints { get; set; }

        [NinjaScriptProperty]
        [Range(1, 86400)]
        [Display(Name = "Hard SL TF (seconds)",
                 Description = "Secondary series cadence (in seconds) for unrealized-loss check. " +
                               "Default 1 = check every second. Set to PivotTfSeconds to match pivot cadence.",
                 Order = 2, GroupName = "Risk")]
        public int HardSlTfSeconds { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0.5-RC";

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

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "ZigzagRunner_v1.0.5-RC";
                Description                      = "Zigzag pivot-retracement w/ hard stop. Chart-TF independent. v" + VERSION;
                Calculate                        = Calculate.OnBarClose;
                EntriesPerDirection              = 1;
                EntryHandling                    = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy     = true;
                ExitOnSessionCloseSeconds        = 30;
                IsFillLimitOnTouch               = false;
                MaximumBarsLookBack              = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution              = OrderFillResolution.Standard;
                Slippage                         = 0;
                StartBehavior                    = StartBehavior.WaitUntilFlat;
                TimeInForce                      = TimeInForce.Gtc;
                TraceOrders                      = false;
                RealtimeErrorHandling            = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling               = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade              = 2;

                // Defaults
                RPoints                          = 30.0;
                Contracts                        = 1;
                PivotTfSeconds                   = 60;     // 1 minute = v1.0.4 baseline
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;
                OnHighPivot                      = PivotAction_v105.Short;  // v1.0 baseline (counter-trend)
                OnLowPivot                       = PivotAction_v105.Long;   // v1.0 baseline (counter-trend)

                // v1.0.5-RC risk defaults
                MaxUnrealizedLossPoints          = 75.0;   // -$150 on MNQ; ~0.2% winner-kill rate
                HardSlTfSeconds                  = 1;      // check every second
            }
            else if (State == State.Configure)
            {
                // Add the two secondary series in fixed order to keep
                // BarsInProgress indices stable: 1 = pivot, 2 = SL.
                AddDataSeries(BarsPeriodType.Second, PivotTfSeconds);   // BIP 1
                AddDataSeries(BarsPeriodType.Second, HardSlTfSeconds);  // BIP 2

                direction       = 0;
                extremePrice    = double.NaN;
                extremeBarIdx   = -1;
                lastPivotDir    = 0;
                lastPivotPrice  = double.NaN;
            }
        }

        protected override void OnBarUpdate()
        {
            // ─── 1.0.1 SAFETY GUARD (runs on every series update) ───────────
            // If position size somehow exceeded Contracts (stale order, manual
            // intervention, NT8 framework bug), panic-close immediately.
            if (Math.Abs(Position.Quantity) > Contracts)
            {
                Print("ZigzagRunner_v105 SAFETY: Position.Quantity=" + Position.Quantity +
                      " exceeds Contracts=" + Contracts + ", panic-closing.");
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "SafetyPanicLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "SafetyPanicShort", "");
                return;
            }

            int bip = BarsInProgress;

            // ─── Primary chart series: IGNORED ──────────────────────────────
            // Strategy is chart-TF independent. We only consume the pivot and
            // SL secondaries we explicitly added.
            if (bip == BIP_PRIMARY) return;

            // ─── 1.0.5-RC: HARD STOP CHECK on SL series ─────────────────────
            // Runs on every HardSlTfSeconds-second bar close. Independent of
            // pivot series so the cap can fire MID-pivot-bar instead of
            // waiting for the next pivot close. MaxUnrealizedLossPoints == 0
            // disables the cap entirely.
            if (bip == BIP_SL)
            {
                if (MaxUnrealizedLossPoints <= 0) return;
                if (Position.MarketPosition == MarketPosition.Flat) return;
                if (CurrentBars[BIP_SL] < 1) return;

                double refPrice = Closes[BIP_SL][0];
                double unrealizedPts = Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Points, refPrice);

                if (unrealizedPts <= -MaxUnrealizedLossPoints)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(Position.Quantity, "HardStopLong", "");
                    else
                        ExitShort(Position.Quantity, "HardStopShort", "");
                }
                return;
            }

            // ─── PIVOT series: ALL pivot/EOD/entry-cutoff logic ─────────────
            if (bip != BIP_PIVOT) return;
            if (CurrentBars[BIP_PIVOT] < BarsRequiredToTrade) return;

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

            if (direction == 0)
            {
                // No direction yet — first R-retracement defines the first leg
                if (c - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // extreme was a LOW pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
                else if (extremePrice - c >= RPoints)
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
                else if (extremePrice - c >= RPoints)
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
                else if (c - extremePrice >= RPoints)
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
            PivotAction_v105 action;
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
            if (action == PivotAction_v105.Skip)
                return;

            if (action == PivotAction_v105.Short)
                EnterShort(Contracts, "ShortAt" + pivotLabel);
            else  // PivotAction_v105.Long
                EnterLong(Contracts, "LongAt" + pivotLabel);
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // PivotAction_v105 (v1.0.5-RC, 2026-04-27)
    // ─────────────────────────────────────────────────────────────────────
    // Per-pivot-type direction action. Public + version-suffixed name so
    // it can coexist with v1.0.4's PivotAction_v10 in the same compiled
    // assembly (NT8 compiles all Strategies/*.cs into one DLL).
    // ═════════════════════════════════════════════════════════════════════
    public enum PivotAction_v105
    {
        Long,    // EnterLong on this pivot type
        Short,   // EnterShort on this pivot type
        Skip,    // No entry on this pivot type
    }
}
