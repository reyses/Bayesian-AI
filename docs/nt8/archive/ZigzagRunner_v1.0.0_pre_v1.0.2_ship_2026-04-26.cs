// =============================================================================
// ZigzagRunner 1.0.0 -- 2026-04-23  (BASELINE - no trail, no CSV, no post-cutoff exit)
// =============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// This is the BASELINE version deliberately pinned here so recompiling
// in NT8 will NOT silently switch behavior. For newer versions (trail,
// CSV logging, post-cutoff exit) see docs/NT8_ZigzagRunner.cs in the
// Bayesian-AI repo.
//
// LOGIC:
//   1. Zigzag on bar closes with threshold R (points).
//   2. When a pivot is CONFIRMED (price retraces R from current extreme):
//      - High pivot  -> reverse to SHORT
//      - Low  pivot  -> reverse to LONG
//   3. Past Entry Cutoff UTC: no new entries. Existing position is held
//      until EOD. (v1.1+ adds a reverse-exit that is NOT in this version.)
//   4. EOD force-close at configurable UTC time.
//
// INSTALLATION:
//   1. This file IS the live copy in NT8 Strategies/ folder.
//   2. In NT8: Tools > NinjaScript Editor > Compile (F5).
//   3. Apply to MNQ 1-minute chart, Sim101 account, Slippage = 1 tick.
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
    public class ZigzagRunner : Strategy
    {
        // ── Settings ─────────────────────────────────────────────────────

        [NinjaScriptProperty]
        [Display(Name = "R (points)", Description = "Zigzag retracement threshold in price points (30 = MNQ $60)", Order = 1, GroupName = "Zigzag")]
        public double RPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contracts", Description = "Contracts per trade", Order = 2, GroupName = "Zigzag")]
        public int Contracts { get; set; }

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

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0.0";

        // ── Zigzag state ─────────────────────────────────────────────────
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
                Name                             = "ZigzagRunner";
                Description                      = "Zigzag pivot-retracement strategy (pure rule, no ML). v" + VERSION;
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

                // Defaults matching Python backtest sweet spot
                RPoints                          = 30.0;
                Contracts                        = 1;
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;
            }
            else if (State == State.Configure)
            {
                direction       = 0;
                extremePrice    = double.NaN;
                extremeBarIdx   = -1;
                lastPivotDir    = 0;
                lastPivotPrice  = double.NaN;
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;

            double c = Close[0];
            DateTime barUtc = Time[0].ToUniversalTime();
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

            // ─── Initialize extreme on first bar ───────────────────────
            if (double.IsNaN(extremePrice))
            {
                extremePrice  = c;
                extremeBarIdx = CurrentBar;
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
                    extremeBarIdx  = CurrentBar;
                }
                else if (extremePrice - c >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // extreme was a HIGH pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
            }
            else if (direction == +1)
            {
                // In an UP leg — watch for new highs, otherwise R-retracement confirms HIGH pivot
                if (c > extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBar;
                }
                else if (extremePrice - c >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // high pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
            }
            else  // direction == -1 (down leg)
            {
                if (c < extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBar;
                }
                else if (c - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // low pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBar;
                }
            }

            if (!pivotConfirmed) return;
            lastPivotDir = newPivotDir;

            // ─── Entry cutoff (v1.0 behavior: skip entry, no exit) ─────
            // Note: v1.1+ adds a reverse-exit here. v1.0 just holds.
            if (minsOfDay >= entryCutMins) return;

            // ─── Place orders (NT8 auto-reverses opposite positions) ───
            if (newPivotDir == +1)
            {
                // High pivot just confirmed → new DOWN leg → SHORT
                EnterShort(Contracts, "ShortAtHighPivot");
            }
            else
            {
                // Low pivot just confirmed → new UP leg → LONG
                EnterLong(Contracts, "LongAtLowPivot");
            }
        }
    }
}
