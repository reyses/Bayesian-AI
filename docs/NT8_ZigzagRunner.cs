// =============================================================================
// ZigzagRunner 1.3.0 -- 2026-04-24
// =============================================================================
// CHANGELOG 1.3.0:
//   - Trailing stop now uses max(TrailDistancePoints, TrailPercent * HWM_profit).
//     Below the crossover (~$100 profit on MNQ at defaults), the fixed
//     distance floor dominates (locks $10 minimum once armed).
//     Above crossover, the percentage rule kicks in and always protects
//     (100% - TrailPercent * 100) of HWM profit -> 90% at default 10%.
//   - New param: TrailPercent (default 0.10 = 10%). Set to 0 to keep
//     strict fixed-distance trail (v1.2 behavior).
//
// CHANGELOG 1.2.0:
//   - Trailing stop: arms after TrailActivatePoints of unrealized profit,
//     trails TrailDistancePoints behind the high-water mark. Defaults 10/5
//     points ($20/$10 on MNQ). Set TrailActivatePoints <= 0 to disable.
//     Trail is checked on bar close (OnBarClose calc). Exits as market via
//     ExitLong/ExitShort -> fills next bar open. Exit reason tag:
//     "TrailStopLong" / "TrailStopShort".
//
// CHANGELOG 1.1.0:
//   - Per-trade CSV logging via OnExecutionUpdate (CsvPath parameter).
//     Appends one row per closed trade: entry/exit time + price, direction,
//     qty, PnL (points + USD), held duration, entry/exit reason tags.
//     Handles both flat-close and reversal-close paths.
//   - Patch: exit open position on reverse pivot after entry cutoff
//     (instead of silently holding until EOD).
// =============================================================================
//
// Pure zigzag pivot-retracement strategy for NinjaTrader 8.
// No CNN, no ML, no Python bridge. Self-contained NinjaScript Strategy.
//
// LOGIC (matches Python backtest in tools/zigzag_backtest.py):
//   1. Zigzag on bar closes with threshold R (points).
//   2. When a pivot is CONFIRMED (price retraces R from current extreme):
//      - High pivot   -> reverse to SHORT (enter short or close long+short)
//      - Low  pivot   -> reverse to LONG
//   3. EOD force-close at configurable UTC time.
//   4. Entry cutoff before EOD so we don't open new positions near close.
//
// EXPECTED ECONOMICS (Python backtest, R=30 points on MNQ 1m):
//   IS  2025:  +$986/day  over 277 active days, 80% winning days
//   OOS 2026:  +$1,059/day over 26 active days,  92% winning days
//   Realistic haircut for slippage + discrete-bar timing: ~30% -> +$690-740/day
//   Clear of MNQ $400/day ship threshold with margin.
//
// INSTALLATION:
//   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Strategies\
//   2. In NT8: Tools > NinjaScript Editor > right-click > Compile
//   3. Strategy Analyzer OR live/sim chart:
//        - Instrument: MNQ 06-26 (or front-month)
//        - Bar type: Minute, 1-minute (matches Python backtest)
//        - Data: Last/trade series
//        - Strategy: ZigzagRunner, RPoints=30, Contracts=1
//   4. Backtest in Strategy Analyzer OR enable on Sim account.
//
// IMPORTANT:
//   - Backtest uses Calculate.OnBarClose = entries fill at next bar open.
//     This adds real slippage vs the idealized Python backtest.
//   - Run on SIM first. Verify reversal orders behave as expected.
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
        [Display(Name = "Trail Activate (points)", Description = "Arm trailing stop once unrealized profit reaches this many points. 0 = disabled. MNQ: 10 pts = $20", Order = 1, GroupName = "TrailStop")]
        public double TrailActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Distance (points)", Description = "FLOOR trail distance in points (trail never tighter than this). MNQ: 5 pts = $10", Order = 2, GroupName = "TrailStop")]
        public double TrailDistancePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Percent", Description = "Trail this fraction of HWM profit. Whichever is greater: Distance or Percent * HWM profit. 0.10 = 10% => protects 90% of max profit once past crossover. Set 0 to disable and use fixed distance only.", Order = 3, GroupName = "TrailStop")]
        public double TrailPercent { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Log CSV Path", Description = "If non-empty, append one row per closed trade (full path)", Order = 1, GroupName = "Logging")]
        public string CsvPath { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.3.0";
        private const string CSV_HEADER = "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason";

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

        // ── Trailing stop state ──────────────────────────────────────────
        private bool trailArmed;           // flips true once TrailActivatePoints hit
        private double trailWaterMark;     // highest close for long / lowest for short
        private bool trailExitSubmitted;   // guard against double-firing exit call

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
                TrailActivatePoints              = 10.0;   // MNQ $20 (arms trail)
                TrailDistancePoints              = 5.0;    // MNQ $10 (floor)
                TrailPercent                     = 0.10;   // 10% of HWM profit (protects 90% past crossover)
                CsvPath                          = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_trades.csv";
            }
            else if (State == State.Configure)
            {
                direction            = 0;
                extremePrice         = double.NaN;
                extremeBarIdx        = -1;
                lastPivotDir         = 0;
                lastPivotPrice       = double.NaN;
                currentEntryPrice    = 0.0;
                currentEntryTime     = DateTime.MinValue;
                currentEntryDir      = 0;
                currentEntryQty      = 0;
                currentEntryReason   = string.Empty;
                trailArmed           = false;
                trailWaterMark       = 0.0;
                trailExitSubmitted   = false;
                EnsureCsvHeader();
            }
        }

        // ─── CSV logging ──────────────────────────────────────────────────
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
                Print("ZigzagRunner CSV init error: " + ex.Message);
            }
        }

        private void AppendTradeCsv(DateTime exitTime, string exitReason, double exitPrice, int qty)
        {
            if (string.IsNullOrWhiteSpace(CsvPath)) return;
            if (currentEntryDir == 0) return;  // no open trade to log

            try
            {
                double pnlPts = currentEntryDir * (exitPrice - currentEntryPrice);
                double pnlUsd = pnlPts * 2.0 * qty;   // MNQ: $2/pt
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
                Print("ZigzagRunner CSV append error: " + ex.Message);
            }
        }

        private static string CsvEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.Contains(",") || s.Contains("\""))
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }

        // Called by NT8 after every fill. We use this to detect entry/exit events
        // and append a CSV row when a position closes (flat OR reversal).
        protected override void OnExecutionUpdate(Execution execution, string executionId,
                                                  double price, int quantity,
                                                  MarketPosition marketPosition, string orderId,
                                                  DateTime time)
        {
            if (execution == null || execution.Order == null) return;

            // `marketPosition` param is the RESULTING net position AFTER this fill.
            // `execution.Price` is the fill price. `quantity` is this fill's qty.
            string orderName = execution.Order.Name ?? execution.Order.OrderAction.ToString();

            // Case A: we had an open position and this fill either flattens it or flips sign.
            if (currentEntryDir != 0)
            {
                bool flatAfter    = marketPosition == MarketPosition.Flat;
                bool flippedSign  = (currentEntryDir > 0 && marketPosition == MarketPosition.Short)
                                 || (currentEntryDir < 0 && marketPosition == MarketPosition.Long);

                if (flatAfter || flippedSign)
                {
                    // Close out the old trade and log it.
                    AppendTradeCsv(time, orderName, price, currentEntryQty);

                    if (flatAfter)
                    {
                        currentEntryDir    = 0;
                        currentEntryQty    = 0;
                        currentEntryPrice  = 0.0;
                        currentEntryReason = string.Empty;
                        // Reset trail state — next entry starts fresh
                        trailArmed         = false;
                        trailWaterMark     = 0.0;
                        trailExitSubmitted = false;
                    }
                    else
                    {
                        // Reversal: this single fill both closed old and opened new.
                        currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                        currentEntryPrice  = price;
                        currentEntryTime   = time;
                        currentEntryQty    = Math.Abs(Position.Quantity);
                        currentEntryReason = orderName;
                        // Reset trail for the new entry
                        trailArmed         = false;
                        trailWaterMark     = price;   // init at fill price
                        trailExitSubmitted = false;
                    }
                    return;
                }
                // Same-sign additional fill (scale-in) — rare for this strategy; skip update
                return;
            }

            // Case B: we were flat and this fill opens a new position.
            if (marketPosition != MarketPosition.Flat)
            {
                currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                currentEntryPrice  = price;
                currentEntryTime   = time;
                currentEntryQty    = quantity;
                currentEntryReason = orderName;
                // Arm trail state for this entry
                trailArmed         = false;
                trailWaterMark     = price;
                trailExitSubmitted = false;
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

            // ─── Trailing stop check ───────────────────────────────────
            // Only meaningful when we have a confirmed in-position entry (currentEntryDir != 0).
            // Trail arms once unrealized profit >= TrailActivatePoints from entry price.
            // Exits if price breaches trailWaterMark +- TrailDistancePoints.
            if (currentEntryDir != 0 && TrailActivatePoints > 0.0 && !trailExitSubmitted)
            {
                double unrealizedPts = currentEntryDir * (c - currentEntryPrice);

                // Update water-mark even before arming (so once armed, we already have the best price)
                if (currentEntryDir > 0)
                    trailWaterMark = Math.Max(trailWaterMark, c);
                else
                    trailWaterMark = Math.Min(trailWaterMark, c);

                if (!trailArmed && unrealizedPts >= TrailActivatePoints)
                    trailArmed = true;

                if (trailArmed)
                {
                    // Adaptive trail distance:
                    //   effDist = max(TrailDistancePoints, TrailPercent * HWM_profit_in_points)
                    // Below crossover, fixed floor dominates (locks $10 min).
                    // Above crossover, percentage dominates (protects 1-TrailPercent of max profit).
                    // Stop price derived from HWM only tightens (never widens) because HWM is monotone.
                    double hwmProfitPts = currentEntryDir * (trailWaterMark - currentEntryPrice);
                    if (hwmProfitPts < 0) hwmProfitPts = 0;  // safety: HWM should always be on profit side
                    double pctDistPts = (TrailPercent > 0) ? (TrailPercent * hwmProfitPts) : 0.0;
                    double effDistPts = Math.Max(TrailDistancePoints, pctDistPts);

                    double stopPx = (currentEntryDir > 0)
                        ? trailWaterMark - effDistPts
                        : trailWaterMark + effDistPts;

                    bool breached = (currentEntryDir > 0) ? (c <= stopPx) : (c >= stopPx);
                    if (breached)
                    {
                        if (currentEntryDir > 0)
                            ExitLong(Position.Quantity, "TrailStopLong", "");
                        else
                            ExitShort(Position.Quantity, "TrailStopShort", "");
                        trailExitSubmitted = true;
                        return;   // skip zigzag logic this bar
                    }
                }
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

            // ─── Past entry cutoff: exit open position on reverse pivot, no new entry
            if (minsOfDay >= entryCutMins)
            {
                if (Position.MarketPosition == MarketPosition.Long && newPivotDir == +1)
                    ExitLong(Position.Quantity, "LongExitPastCutoff", "");
                else if (Position.MarketPosition == MarketPosition.Short && newPivotDir == -1)
                    ExitShort(Position.Quantity, "ShortExitPastCutoff", "");
                return;
            }

            // ─── Before cutoff: place/reverse via Enter (NT8 auto-closes opposite) ───
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
