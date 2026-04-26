// =============================================================================
// ZigzagRunnerHybrid 1.4.0-RC.REJECTED -- 2026-04-24
// =============================================================================
// STATUS: REJECTED — DO NOT DEPLOY.
//   Phase 2 backtest (2026-04-24) tripped both abort triggers:
//     1. Trade count diverged +22% vs v1.3-RC at N=5 (limit was ±5%).
//     2. Theoretical PnL OOS dropped −74% (was claimed to be invariant).
//   Root cause: 5s-grid retracement check catches intra-minute spikes the 1m
//   close averages out, generating noise pivots. Hybrid produces a different
//   pivot set, not the same set with faster confirmation.
//   Findings: reports/findings/2026-04-25_cascade_pivot_quality.md (top section).
//
//   File kept as a research artifact; do not compile into production NT8.
//
// VERSION POLICY (effective 2026-04-25):
//   - Released  : no suffix       (only v1.0 currently)
//   - RC        : -RC suffix      (in-development candidate)
//   - REJECTED  : -RC.REJECTED    (this file)
//
// CHANGELOG 1.4.0-RC.REJECTED:
//   - Hybrid timing: PIVOT EXTREMES update only on 1m close (preserves
//     v1.3 / v1.0 trade semantics), but RETRACEMENT CONFIRMATION check
//     runs on a 5s secondary data series. Up to 55-second latency
//     reduction on pivot-trigger orders.
//   - New Timing parameter: ConfirmationTfSeconds (default 5).
//     Setting 60 causes the secondary series to align with the primary
//     1m bars and behavior reduces to v1.3 single-TF semantics.
//   - Pattern: AddDataSeries(BarsPeriodType.Second, N) + BarsInProgress
//     branching, identical pattern to NT8_BayesianBridge.cs (line 150).
//   - Class is INDEPENDENT of ZigzagRunner v1.3. Both compile into the
//     same DLL and can be applied as separate strategies on different
//     charts for clean A/B comparison.
//   - Trail stop, EOD force-close, entry cutoff: all moved into the 5s
//     branch so they react on 5s bars instead of 1m bars.
//   - CSV logging via OnExecutionUpdate is unchanged (already fill-
//     triggered, not bar-triggered).
//
// LOGIC:
//   Primary series (1m): updates extremePrice when current 1m close
//     extends the running high/low in current direction. No retracement
//     check, no order placement here.
//   Secondary series (5s): checks if (extremePrice - close5s) >= R for
//     up-leg or (close5s - extremePrice) >= R for down-leg. On confirm,
//     fires reversal order. After confirmation, extremePrice is reset
//     to NaN and re-initializes at next 1m close.
//
// EXPECTED BEHAVIOR:
//   - Same trade COUNT as v1.3 (pivot detection invariant preserved)
//   - Entry timestamps 0-55 seconds earlier than v1.3 on matched pivots
//   - Same direction (long/short) for matched pivots
//   - ETD per trade lower than v1.3 (less profit given back to lag)
//
// INSTALLATION:
//   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Strategies\
//   2. In NT8: Tools > NinjaScript Editor > F5 to compile
//   3. Strategy dropdown shows BOTH ZigzagRunner and ZigzagRunnerHybrid
//   4. Apply ZigzagRunnerHybrid to MNQ 1m chart on Sim101.
//      Set ConfirmationTfSeconds = 5 for default hybrid timing.
//
// IMPORTANT:
//   - DO NOT deploy in production until v1.0 CAPA Verification has
//     formally passed. This is a v1.4 candidate, not yet validated.
//   - Run Sim101 first. Verify pivot count matches v1.3 over a
//     concurrent session before any live consideration.
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
    public class ZigzagRunnerHybrid : Strategy
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
        [Range(1, 60)]
        [Display(Name = "Confirmation TF (seconds)", Description = "Secondary series period for retracement confirmation. 5 = check every 5s. 60 = effectively single-TF v1.3 behavior. Range 1-60.", Order = 1, GroupName = "Timing")]
        public int ConfirmationTfSeconds { get; set; }

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
        private const string VERSION = "1.4.0-RC.REJECTED";
        private const string CSV_HEADER = "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason";

        // ── Zigzag state ─────────────────────────────────────────────────
        // direction: 0 = undefined (before first pivot), +1 = up leg, -1 = down leg
        // extremePrice: tracks 1m close highs/lows (NEVER updated on 5s).
        //               After pivot confirmation, set to NaN until next 1m close re-establishes it.
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
                Name                             = "ZigzagRunnerHybrid";
                Description                      = "Zigzag pivot strategy with hybrid timing: 1m pivots + Ns confirmation (default 5s). v" + VERSION;
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

                // Defaults — match v1.3 baseline plus new Timing default
                RPoints                          = 30.0;
                Contracts                        = 1;
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;
                ConfirmationTfSeconds            = 5;
                TrailActivatePoints              = 10.0;
                TrailDistancePoints              = 5.0;
                TrailPercent                     = 0.10;
                CsvPath                          = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_hybrid_trades.csv";
            }
            else if (State == State.Configure)
            {
                // Add the secondary data series for retracement confirmation.
                // Pattern: AddDataSeries(BarsPeriodType.Second, N) — same approach
                // used in NT8_BayesianBridge.cs (lines 150–166) for multi-TF data.
                AddDataSeries(BarsPeriodType.Second, ConfirmationTfSeconds);

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

        // ─── CSV logging (unchanged from v1.3) ────────────────────────────
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
                Print("ZigzagRunnerHybrid CSV init error: " + ex.Message);
            }
        }

        private void AppendTradeCsv(DateTime exitTime, string exitReason, double exitPrice, int qty)
        {
            if (string.IsNullOrWhiteSpace(CsvPath)) return;
            if (currentEntryDir == 0) return;

            try
            {
                double pnlPts = currentEntryDir * (exitPrice - currentEntryPrice);
                double pnlUsd = pnlPts * 2.0 * qty;
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
                Print("ZigzagRunnerHybrid CSV append error: " + ex.Message);
            }
        }

        private static string CsvEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.Contains(",") || s.Contains("\""))
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }

        // ─── OnExecutionUpdate (unchanged from v1.3) ──────────────────────
        protected override void OnExecutionUpdate(Execution execution, string executionId,
                                                  double price, int quantity,
                                                  MarketPosition marketPosition, string orderId,
                                                  DateTime time)
        {
            if (execution == null || execution.Order == null) return;

            string orderName = execution.Order.Name ?? execution.Order.OrderAction.ToString();

            if (currentEntryDir != 0)
            {
                bool flatAfter    = marketPosition == MarketPosition.Flat;
                bool flippedSign  = (currentEntryDir > 0 && marketPosition == MarketPosition.Short)
                                 || (currentEntryDir < 0 && marketPosition == MarketPosition.Long);

                if (flatAfter || flippedSign)
                {
                    AppendTradeCsv(time, orderName, price, currentEntryQty);

                    if (flatAfter)
                    {
                        currentEntryDir    = 0;
                        currentEntryQty    = 0;
                        currentEntryPrice  = 0.0;
                        currentEntryReason = string.Empty;
                        trailArmed         = false;
                        trailWaterMark     = 0.0;
                        trailExitSubmitted = false;
                    }
                    else
                    {
                        currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                        currentEntryPrice  = price;
                        currentEntryTime   = time;
                        currentEntryQty    = Math.Abs(Position.Quantity);
                        currentEntryReason = orderName;
                        trailArmed         = false;
                        trailWaterMark     = price;
                        trailExitSubmitted = false;
                    }
                    return;
                }
                return;
            }

            if (marketPosition != MarketPosition.Flat)
            {
                currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                currentEntryPrice  = price;
                currentEntryTime   = time;
                currentEntryQty    = quantity;
                currentEntryReason = orderName;
                trailArmed         = false;
                trailWaterMark     = price;
                trailExitSubmitted = false;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // OnBarUpdate routes by BarsInProgress:
        //   0 = primary (1m): update extreme tracking only
        //   1 = secondary (5s default): retracement check, trail, EOD, orders
        // ═══════════════════════════════════════════════════════════════════
        protected override void OnBarUpdate()
        {
            // Series-specific BarsRequiredToTrade check
            if (BarsInProgress == 0)
            {
                if (CurrentBars[0] < BarsRequiredToTrade) return;
                OnPrimaryBarUpdate();
                return;
            }

            if (BarsInProgress == 1)
            {
                // Wait for both primary AND secondary to have enough bars before trading
                if (CurrentBars[0] < BarsRequiredToTrade) return;
                if (CurrentBars[1] < BarsRequiredToTrade) return;
                OnSecondaryBarUpdate();
                return;
            }
        }

        // 1m primary close — update pivot extreme only. NO retracement check, NO orders.
        private void OnPrimaryBarUpdate()
        {
            double c = Close[0];   // 1m close

            // Initialize or re-initialize extreme on first 1m close after init or pivot
            if (double.IsNaN(extremePrice))
            {
                extremePrice  = c;
                extremeBarIdx = CurrentBar;
                return;
            }

            // direction == 0: still seeking first directional move.
            // Mirror v1.3: don't update extreme until retracement triggers direction.
            // (extremePrice stays at the very first 1m close until first pivot fires.)
            if (direction == +1)
            {
                if (c > extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBar;
                }
            }
            else if (direction == -1)
            {
                if (c < extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBar;
                }
            }
            // No retracement check here — that lives in the secondary branch.
        }

        // 5s secondary close — retracement check, trail, EOD, entry cutoff, order placement.
        private void OnSecondaryBarUpdate()
        {
            double c5 = Closes[1][0];           // 5s close
            DateTime barUtc = Times[1][0].ToUniversalTime();
            int minsOfDay      = barUtc.Hour * 60 + barUtc.Minute;
            int eodMins        = EodHourUtc * 60 + EodMinuteUtc;
            int entryCutMins   = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

            // ─── EOD force-close (5s granularity) ──────────────────────
            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "EodExitLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "EodExitShort", "");
                return;
            }

            // ─── Trailing stop check (5s granularity, uses 5s close as price) ──
            if (currentEntryDir != 0 && TrailActivatePoints > 0.0 && !trailExitSubmitted)
            {
                double unrealizedPts = currentEntryDir * (c5 - currentEntryPrice);

                if (currentEntryDir > 0)
                    trailWaterMark = Math.Max(trailWaterMark, c5);
                else
                    trailWaterMark = Math.Min(trailWaterMark, c5);

                if (!trailArmed && unrealizedPts >= TrailActivatePoints)
                    trailArmed = true;

                if (trailArmed)
                {
                    double hwmProfitPts = currentEntryDir * (trailWaterMark - currentEntryPrice);
                    if (hwmProfitPts < 0) hwmProfitPts = 0;
                    double pctDistPts = (TrailPercent > 0) ? (TrailPercent * hwmProfitPts) : 0.0;
                    double effDistPts = Math.Max(TrailDistancePoints, pctDistPts);

                    double stopPx = (currentEntryDir > 0)
                        ? trailWaterMark - effDistPts
                        : trailWaterMark + effDistPts;

                    bool breached = (currentEntryDir > 0) ? (c5 <= stopPx) : (c5 >= stopPx);
                    if (breached)
                    {
                        if (currentEntryDir > 0)
                            ExitLong(Position.Quantity, "TrailStopLong", "");
                        else
                            ExitShort(Position.Quantity, "TrailStopShort", "");
                        trailExitSubmitted = true;
                        return;
                    }
                }
            }

            // ─── Skip retracement check if no extreme yet ──────────────
            // extremePrice is NaN immediately after a pivot confirmation OR before the
            // first 1m close has occurred. In that window we don't fire any new pivots —
            // we wait for the next 1m close in OnPrimaryBarUpdate to re-initialize.
            if (double.IsNaN(extremePrice)) return;

            // ─── Zigzag retracement check on 5s close ──────────────────
            // extremePrice is 1m-derived (set by OnPrimaryBarUpdate). c5 is 5s-derived.
            // Hybrid logic: retracement DISTANCE is computed against 1m extreme, but
            // confirmed at 5s granularity (up to 55s faster than v1.3).
            bool pivotConfirmed = false;
            int newPivotDir     = 0;

            if (direction == 0)
            {
                if (c5 - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // initial extreme was a LOW pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = double.NaN;   // wait for next 1m close to re-init
                    extremeBarIdx  = -1;
                }
                else if (extremePrice - c5 >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // initial extreme was a HIGH pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = double.NaN;
                    extremeBarIdx  = -1;
                }
            }
            else if (direction == +1)
            {
                // In an UP leg — check for R retracement DOWN from extreme
                if (extremePrice - c5 >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;           // high pivot just left behind
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = double.NaN;
                    extremeBarIdx  = -1;
                }
                // No "new high" tracking on 5s — extremes only update on 1m closes.
            }
            else  // direction == -1 (down leg)
            {
                if (c5 - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;           // low pivot just left behind
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = double.NaN;
                    extremeBarIdx  = -1;
                }
            }

            if (!pivotConfirmed) return;
            lastPivotDir = newPivotDir;

            // ─── Past entry cutoff: exit on reverse pivot, no new entry
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
