// =============================================================================
// ZigzagRunner 1.1.1 -- 2026-04-25  (vanilla v1.0 + safety + CSV ledger)
// =============================================================================
// CHANGELOG 1.1.1 (2026-04-25):
//   * Bugfix: CSV qty was reporting 0 on flip-fills because
//     Math.Abs(Position.Quantity) reads 0 in NT8's mid-flip transient state.
//     Now uses `Contracts` (canonical position size) for qty.
//     Also fixes entry-reason on flips: was "FlipExitLong"/"FlipExitShort"
//     (the prior exit tag); now correctly maps to "LongAtLowPivot" /
//     "ShortAtHighPivot" based on resulting direction.
//
//=============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// CHANGELOG 1.1.0 (2026-04-25):
//   * Per-trade CSV ledger via OnExecutionUpdate (CsvPath parameter).
//     Appends one row per closed trade: close timestamp UTC, day,
//     entry/exit time + price, direction, qty, P&L (points + USD),
//     held-minutes, entry-reason tag, exit-reason tag.
//     Captures both flat-close and reversal-close paths.
//     CSV path is configurable; default points to repo reports/findings.
//   * No change to entry/exit logic, EOD, R, or position-size handling.
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
//
// LOGIC (unchanged from 1.0.0):
//   1. Zigzag on bar closes with threshold R (points).
//   2. When a pivot is CONFIRMED (price retraces R from current extreme):
//      - High pivot  -> reverse to SHORT
//      - Low  pivot  -> reverse to LONG
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
using System.Globalization;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ZigzagRunner_v11 : Strategy
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
        [Display(Name = "Trade Log CSV Path", Description = "If non-empty, append one row per closed trade. Use full path. Empty = disable.", Order = 1, GroupName = "Logging")]
        public string CsvPath { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.1.1";
        private const string CSV_HEADER =
            "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason";

        // ── Zigzag state ─────────────────────────────────────────────────
        // direction: 0 = undefined (before first pivot), +1 = up leg, -1 = down leg
        private int direction;
        private double extremePrice;
        private int extremeBarIdx;
        private int lastPivotDir;      // +1 = high pivot (just formed), -1 = low pivot; 0 = none yet
        private double lastPivotPrice;

        // ── Trade tracking for CSV logging ────────────────────────────────
        private double currentEntryPrice;
        private DateTime currentEntryTime;
        private int currentEntryDir;       // +1 long, -1 short, 0 flat
        private int currentEntryQty;
        private string currentEntryReason;
        private readonly object csvLock = new object();

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "ZigzagRunner_v1.1";
                Description                      = "Zigzag pivot-retracement strategy (pure rule, no ML, with CSV ledger). v" + VERSION;
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
                CsvPath                          = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_v1.1_trades.csv";
            }
            else if (State == State.Configure)
            {
                direction          = 0;
                extremePrice       = double.NaN;
                extremeBarIdx      = -1;
                lastPivotDir       = 0;
                lastPivotPrice     = double.NaN;
                currentEntryPrice  = 0.0;
                currentEntryTime   = DateTime.MinValue;
                currentEntryDir    = 0;
                currentEntryQty    = 0;
                currentEntryReason = string.Empty;
                EnsureCsvHeader();
            }
        }

        // ─── CSV ledger (1.1.0) ───────────────────────────────────────────
        private void EnsureCsvHeader()
        {
            if (string.IsNullOrWhiteSpace(CsvPath)) return;
            try
            {
                string dir = Path.GetDirectoryName(CsvPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);
                if (!File.Exists(CsvPath))
                {
                    lock (csvLock)
                    {
                        File.WriteAllText(CsvPath, CSV_HEADER + Environment.NewLine);
                    }
                }
            }
            catch (Exception ex)
            {
                Print("ZigzagRunner v1.1 CSV init error: " + ex.Message);
            }
        }

        private static string CsvEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.Contains(",") || s.Contains("\""))
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }

        private void AppendTradeCsv(DateTime exitTime, string exitReason, double exitPrice, int qty)
        {
            if (string.IsNullOrWhiteSpace(CsvPath)) return;
            if (currentEntryDir == 0) return;
            try
            {
                double pnlPts = currentEntryDir * (exitPrice - currentEntryPrice);
                double pnlUsd = pnlPts * 2.0 * qty;   // MNQ: $2 per point per contract
                double heldMin = (exitTime - currentEntryTime).TotalMinutes;
                string dir = currentEntryDir > 0 ? "long" : "short";
                string day = currentEntryTime.ToUniversalTime().ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
                DateTime exitUtc = exitTime.ToUniversalTime();
                DateTime entryUtc = currentEntryTime.ToUniversalTime();

                string row = string.Join(",", new string[] {
                    DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                    day,
                    entryUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    exitUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    dir,
                    currentEntryPrice.ToString("F4", CultureInfo.InvariantCulture),
                    exitPrice.ToString("F4", CultureInfo.InvariantCulture),
                    qty.ToString(CultureInfo.InvariantCulture),
                    pnlPts.ToString("F4", CultureInfo.InvariantCulture),
                    pnlUsd.ToString("F2", CultureInfo.InvariantCulture),
                    heldMin.ToString("F2", CultureInfo.InvariantCulture),
                    CsvEscape(currentEntryReason),
                    CsvEscape(exitReason),
                });

                lock (csvLock)
                {
                    File.AppendAllText(CsvPath, row + Environment.NewLine);
                }
            }
            catch (Exception ex)
            {
                Print("ZigzagRunner v1.1 CSV append error: " + ex.Message);
            }
        }

        // OnExecutionUpdate fires on every fill. We use it to track entry state
        // and append a row when a trade closes (flat OR reverses to opposite).
        protected override void OnExecutionUpdate(Execution execution, string executionId,
                                                  double price, int quantity,
                                                  MarketPosition marketPosition, string orderId,
                                                  DateTime time)
        {
            if (execution == null || execution.Order == null) return;

            string orderName = execution.Order.Name ?? execution.Order.OrderAction.ToString();

            if (currentEntryDir != 0)
            {
                bool flatAfter   = marketPosition == MarketPosition.Flat;
                bool flippedSign = (currentEntryDir > 0 && marketPosition == MarketPosition.Short) ||
                                   (currentEntryDir < 0 && marketPosition == MarketPosition.Long);

                if (flatAfter || flippedSign)
                {
                    // Close the prior trade record at this fill price.
                    // Use the captured currentEntryQty (= Contracts at entry time)
                    // rather than Position.Quantity which may be in flux during fill.
                    AppendTradeCsv(time, orderName, price, currentEntryQty);

                    if (flatAfter)
                    {
                        currentEntryDir    = 0;
                        currentEntryQty    = 0;
                        currentEntryPrice  = 0.0;
                        currentEntryReason = string.Empty;
                    }
                    else
                    {
                        // Reversed in same fill (NT8 collapsed a flip).
                        // Use Contracts (= 1) for qty since Position.Quantity may
                        // read 0 mid-flip. Map entry reason from the new direction
                        // because orderName at this point is the prior exit's tag.
                        currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                        currentEntryPrice  = price;
                        currentEntryTime   = time;
                        currentEntryQty    = Contracts;
                        currentEntryReason = (marketPosition == MarketPosition.Long)
                            ? "LongAtLowPivot" : "ShortAtHighPivot";
                    }
                    return;
                }
                return;
            }

            // We were flat; this is a fresh entry fill.
            if (marketPosition != MarketPosition.Flat)
            {
                currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                currentEntryPrice  = price;
                currentEntryTime   = time;
                currentEntryQty    = Contracts;   // canonical; quantity may report 0 mid-fill
                currentEntryReason = orderName;   // fresh entry has clean orderName
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
            if (newPivotDir == +1)
            {
                // Want SHORT
                if (Position.MarketPosition == MarketPosition.Short &&
                    Position.Quantity >= Contracts)
                    return; // already short at target
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "FlipExitLong", "");
                EnterShort(Contracts, "ShortAtHighPivot");
            }
            else
            {
                // Want LONG
                if (Position.MarketPosition == MarketPosition.Long &&
                    Position.Quantity >= Contracts)
                    return; // already long at target
                if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "FlipExitShort", "");
                EnterLong(Contracts, "LongAtLowPivot");
            }
        }
    }
}
