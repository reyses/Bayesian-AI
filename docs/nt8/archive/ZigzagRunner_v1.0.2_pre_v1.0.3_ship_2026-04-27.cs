// =============================================================================
// ZigzagRunner 1.0.2 -- 2026-04-26  (BASELINE + safety patch + RideWithTrend)
// =============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// CHANGELOG 1.0.2 (2026-04-26):
//   * NEW: RideWithTrend bool (default false). Default behavior is
//     IDENTICAL to v1.0.1 (mean-reversion: HIGH pivot -> SHORT, LOW pivot
//     -> LONG = fade the breakout). When true, FLIPS direction at every
//     pivot: HIGH pivot -> LONG (continuation), LOW pivot -> SHORT
//     (continuation) = ride with the prior leg's trend resumption.
//
//     Implementation: single multiplier on the entry-direction decision.
//        int dirMod = RideWithTrend ? -1 : +1;
//        int effectiveDir = newPivotDir * dirMod;
//     Signal names also adapt so the audit trail (NT8 Order Tab,
//     trade-export CSV) shows which mode was active.
//
//     Empirical basis (2026-04-26 100-day Python sweep on the same data):
//        Counter-trend (default false): -$162/day mean, 31% Day WR
//        With-trend    (RideWithTrend): +$162/day mean, 69% Day WR
//        Spearman rho(mean_range_5d, daily PnL): -0.30 vs +0.30 (flips).
//     High-vol regimes ARE trend regimes; counter-trend bleeds on them.
//     The flip converts the strategy from "fade the pivot" to
//     "buy the dip / sell the rip".
//
//   * No change to entry/exit timing, EOD, R, position-size hardening,
//     or any other tunable. The flip is purely a direction multiplier.
//
// CHANGELOG 1.0.1 (2026-04-25):
//   * Position-size hardening (no behavior change in normal flow):
//     1. Defensive guard: panic-close if |Position.Quantity| > Contracts
//        at start of any bar update (catches stale/duplicate orders).
//     2. Idempotent entries: skip EnterShort if already short at Contracts;
//        skip EnterLong if already long at Contracts.
//     3. Explicit exit-before-entry: close opposite position with its own
//        ExitLong/ExitShort order BEFORE submitting the new EnterShort/
//        EnterLong. This replaces NT8's combined-flip order (which shows
//        as a 2-contract trade in the order log even though net position
//        is bounded) with two clean 1-contract orders.
//   No change to entry/exit logic, EOD, R, or any tunable parameter.
//
// LOGIC (mode-dependent on RideWithTrend, otherwise unchanged from 1.0.1):
//   1. Zigzag on bar closes with threshold R (points).
//   2. When a pivot is CONFIRMED (price retraces R from current extreme):
//      - HIGH pivot -> SHORT (counter mode, default) or LONG (with mode)
//      - LOW  pivot -> LONG  (counter mode, default) or SHORT (with mode)
//   3. Past Entry Cutoff UTC: no new entries. Existing position is held
//      until EOD.
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

        [NinjaScriptProperty]
        [Display(Name = "Ride With Trend", Description = "If true, FLIP direction at every pivot: HIGH pivot -> LONG (continuation), LOW pivot -> SHORT. Default false = original v1.0 mean-reversion (fade the pivot). Toggle this for A/B without redeploying.", Order = 1, GroupName = "Direction")]
        public bool RideWithTrend { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0.2";

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
                RideWithTrend                    = false;   // false = baseline v1.0 logic; true = flipped
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

            // ─── 1.0.1 SAFETY GUARD ──────────────────────────────────────
            // If position size somehow exceeded Contracts (stale order, manual
            // intervention, NT8 framework bug), panic-close immediately.
            // This is a "should never happen" backstop.
            if (Math.Abs(Position.Quantity) > Contracts)
            {
                Print("ZigzagRunner SAFETY: Position.Quantity=" + Position.Quantity +
                      " exceeds Contracts=" + Contracts + ", panic-closing.");
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "SafetyPanicLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "SafetyPanicShort", "");
                return;
            }

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

            // ─── 1.0.1: explicit exit-before-entry (no NT8 auto-flip) ────
            // Pattern: if currently in opposite direction, close that position
            // with its own ExitLong/ExitShort order, then place a clean
            // EnterShort/EnterLong of exactly `Contracts` units. Each order
            // logs as exactly Contracts units, never a combined flip.
            //
            // Idempotent guard: skip the new entry if already at target
            // direction & size.
            //
            // ─── 1.0.2: RideWithTrend modifier ───────────────────────────
            // dirMod = +1 (counter, default) preserves original v1.0 logic.
            // dirMod = -1 (with-trend) flips: HIGH pivot -> LONG, LOW -> SHORT.
            int dirMod = RideWithTrend ? -1 : +1;
            int effectiveDir = newPivotDir * dirMod;

            if (effectiveDir == +1)
            {
                // Want SHORT
                if (Position.MarketPosition == MarketPosition.Short &&
                    Position.Quantity >= Contracts)
                    return; // already short at target
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "FlipExitLong", "");
                string sig = RideWithTrend ? "ShortAtLowPivot" : "ShortAtHighPivot";
                EnterShort(Contracts, sig);
            }
            else
            {
                // Want LONG
                if (Position.MarketPosition == MarketPosition.Long &&
                    Position.Quantity >= Contracts)
                    return; // already long at target
                if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "FlipExitShort", "");
                string sig = RideWithTrend ? "LongAtHighPivot" : "LongAtLowPivot";
                EnterLong(Contracts, sig);
            }
        }
    }
}
