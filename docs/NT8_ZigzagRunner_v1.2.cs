// =============================================================================
// ZigzagRunner 1.2 -- 2026-04-25  (RELEASED: trail + hard SL=-$50 + CSV ledger)
// =============================================================================
// CHANGELOG 1.2 (2026-04-25, in-place — DynamicRiskManager refactor):
//   * REFACTOR: Risk logic (Initial SL + two-tier trail) now lives in a
//     dedicated `DynamicRiskManager` class (defined below in this file,
//     same namespace). The strategy holds a single `riskMgr` instance and
//     drives it via:
//        - riskMgr.OnInitialFill(fillPx, dir, fillPx)  on each fresh fill
//          → places hard SL at fill_price ± initialStopPts via callback.
//        - riskMgr.EvaluateStopState(Position, Close[0]) per bar close
//          → drives state machine (Null → Initial → Tier1 → Tier2),
//            updates trail HWM, fires SetStopLoss callback when stop
//            should ratchet tighter.
//        - riskMgr.ResetState() on flat fill.
//
//     The manager exposes a single `Action<double,int,double>` callback
//     (newStopPrice, direction, currentPrice) which the strategy wires to
//     RouteStopOrder(): valid stop → SetStopLoss; invalid (would be on
//     wrong side of current price = missed-breach) → market exit with
//     "TrailMissedBreach{Long|Short}" reason.
//
//     Behavior parity with prior v1.2:
//        - Initial state: stop at entry ± HardStopLossPoints (default 25pt).
//        - Tier1 (peak >= TrailActivatePoints, default 10pt = $20):
//             trail = HWM ± TrailDistancePoints (default 5pt = $10 floor).
//        - Tier2 (peak >= TrailDistancePoints / TrailPercent, default 50pt
//          = $100): trail distance = max(TrailDistancePoints,
//          TrailPercent * peak), so 90% of HWM profit is locked once past
//          the crossover.
//        - Ratchet guard inside manager: stop only ever tightens.
//
//   * BUGFIX (carried forward): SetStopLoss must NOT be called from
//     OnExecutionUpdate as the FIRST stop placement on a fresh entry —
//     NT8 routing breaks subsequent Enter calls. The manager's
//     OnInitialFill IS invoked from OnExecutionUpdate, but the SetStopLoss
//     it routes targets the just-filled signal name and is gap-safe (uses
//     actual fill price as basis, not bar close).
//
//   * NEW: StagnationMonitor (separate class in same file). Counts
//     consecutive 1m bars of negative unrealized PnL; fires market exit
//     after MAX_NEGATIVE_BARS (compile-time const = 5) without ever
//     touching breakeven. Counter resets on any non-negative bar via
//     internal logic in RequiresFlatten().
//
//     Strategy holds a `stagnationMon` instance, RECREATES it on flip /
//     fresh-entry fills (because the class's ResetState is private), and
//     queries `stagnationMon.RequiresFlatten(Position, Close[0], CurrentBar)`
//     each bar after the riskMgr update. Exit reason: "StagnationExitLong"
//     / "StagnationExitShort". The lastEvaluatedBar guard inside the class
//     prevents intra-bar multi-increments.
//
//     This re-introduces the negative-bars timeout that was removed during
//     v1.2 release prep. Prior implementation was firing too aggressively;
//     this version is structurally cleaner (strategy-agnostic class, hard
//     guard against same-bar re-eval) and uses MAX_NEGATIVE_BARS=5 instead
//     of the old 10. To disable, comment out the RequiresFlatten block in
//     OnBarUpdate.
//
//   * Trail (v1.2 design): two-phase. Arms at TrailActivatePoints (default
//     10 pts = $20). Trail distance = max(TrailDistancePoints,
//     TrailPercent * peak_profit). Defaults 5pt / 10% so trail floor is
//     $10 minimum, switches to 90% lock above $100 profit.
//   * Hard SL: HardStopLossPoints default 25 pts ($50 max single-trade loss).
//   * MFE / MAE / capture_pct columns added to CSV ledger for audit.
//     (NegativeBarsTimeout: removed during release prep — was breaking
//      entry flow. Hard SL + trail handle the loss-cap role on their own.)
//
// LEGACY CHANGELOG (1.2.3 reference):
//   * NEW: Negative-bars timeout. If unrealized PnL stays < 0 for more than
//     NegativeBarsTimeout consecutive 1m closes (default 10), market-exit
//     flat with reason "NoProgressExit".
//
//     Counter resets to 0 on any bar where unrealized PnL >= 0, so this is
//     a "consecutive negative bars" timer, not a cumulative one. The trade
//     gets a fresh 10-bar window every time it touches breakeven.
//
//     Rationale: chop trades sit underwater for 10+ minutes without ever
//     making favorable progress. Cutting at -10 bars locks the loss before
//     the hard SL fires for catastrophic cases. Caps held duration on
//     "stuck" trades.
//
//     Set NegativeBarsTimeout <= 0 to disable.
//
// CHANGELOG 1.2.2 (2026-04-25):
//   * NEW: Hard stop loss (HardStopLossPoints, default 25 pts = $50 on MNQ).
//     Pre-placed on entry via SetStopLoss(CalculationMode.Price, ...) at
//     entry_price ± HardStopLossPoints. Caps catastrophic single-trade loss.
//
//     Interaction with trail:
//       - On entry, hard SL is the active stop level.
//       - Each bar in position, the strategy computes the FAVORABLE level:
//           long  : SL = max(entry - HardStopLossPoints, trail_stop_if_armed)
//           short : SL = min(entry + HardStopLossPoints, trail_stop_if_armed)
//         Stop is MONOTONIC in favorable direction — never loosens.
//       - Until trail arms (peak >= TrailActivatePoints), only hard SL applies.
//       - Once trail arms, the trail level (HWM ± eff_dist) becomes the
//         dominant stop because it's always closer to current price than
//         the entry-based hard SL once the trade is profitable.
//
//     Per-trade max loss bound: $50 (SL trigger) + ~$2 (commission) = ~$52.
//     CSV exit_reason will read "Stop loss" or NT8's auto-generated stop tag.
//     Set HardStopLossPoints <= 0 to disable hard SL (trail-only behavior).
//
// CHANGELOG 1.2.1 (2026-04-25):
//   * FIX: trail now uses NT8 SetStopLoss(CalculationMode.Price, ...) to
//     pre-place a real stop order each bar. Replaces the prior
//     "check-on-close-then-market-exit" pattern. Effects:
//       - Stop line is now VISIBLE on the chart and ratchets up each bar
//         (so you can see the trail moving with the trade).
//       - Stop fires intra-bar at the exact trigger price (not at next
//         bar's open with slippage), so chart PnL closely matches account
//         PnL on trail exits.
//       - The trail stop order is updated each bar; only the most-recent
//         stop level is live. NT8 cancels prior stop when we set new.
//   * NEW: CSV ledger now records peak unrealized PnL (MFE) per trade,
//     so we can audit "did the trail capture peak − eff_dist?" directly.
//     New columns: mfe_pts, mae_pts, capture_pct.
//
// CHANGELOG 1.2.0 (2026-04-25):
//   * NEW: Trailing stop with two-phase tightening (per user spec 2026-04-25).
//     Phase 1 (arming + minimum-distance):
//        Arms when unrealized profit reaches TrailActivatePoints
//        (default 10 pts = $20 on MNQ). Once armed, trail distance =
//        TrailDistancePoints (default 5 pts = $10).
//     Phase 2 (percent ratchet):
//        Once unrealized profit reaches the crossover where
//        TrailPercent * peak_profit > TrailDistancePoints (default $100
//        of profit @ 10%), trail switches to TrailPercent of HWM profit.
//        => effective trail distance = max(TrailDistancePoints,
//                                          TrailPercent * peak_profit_in_points)
//     Trail check on bar close (Calculate.OnBarClose). Exit fires as
//     market via ExitLong/ExitShort. Exit tags: "TrailStopLong" /
//     "TrailStopShort" (recorded in CSV ledger).
//     Set TrailActivatePoints <= 0 to disable.
//     Set TrailPercent = 0 to keep strict fixed-distance trail (Phase 1 only).
//
//   * No SL in v1.2 (the rejected v1.2-RC variant had a $20 hard SL that
//     fired on bar-level noise in the high-vol 2026 regime, regressing
//     vs v1.0 by ~$68/day. v1.2 keeps just the trail; SL stays out until
//     calibrated to current bar ranges).
//
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
    public class ZigzagRunner_v12 : Strategy
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
        [Display(Name = "Trail Activate (points)", Description = "Arm trailing stop once unrealized profit reaches this many points. Default 10 pts = $20 on MNQ. Set 0 to disable trail.", Order = 1, GroupName = "TrailStop")]
        public double TrailActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Distance (points)", Description = "Minimum trail distance in points (Phase 1 floor). Default 5 pts = $10 on MNQ. Trail never tighter than this.", Order = 2, GroupName = "TrailStop")]
        public double TrailDistancePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Percent", Description = "Phase 2: trail this fraction of HWM profit. Effective dist = max(TrailDistance, TrailPercent * peak_profit). 0.10 = 10% (locks 90% of HWM once past crossover). Set 0 for fixed-distance only (Phase 1).", Order = 3, GroupName = "TrailStop")]
        public double TrailPercent { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Hard Stop Loss (points)", Description = "Catastrophic backstop in points. Pre-placed via SetStopLoss on entry. Default 25 pts = $50 max loss/trade on MNQ. Set 0 to disable hard SL (trail-only).", Order = 1, GroupName = "StopLoss")]
        public double HardStopLossPoints { get; set; }

        // StagnationMonitor: hardcoded MAX_NEGATIVE_BARS=5 (compile-time
        // constant per the supplied module). No NinjaScriptProperty exposed —
        // tweak the constant in the StagnationMonitor class definition (below)
        // if you need to retune.

        [NinjaScriptProperty]
        [Display(Name = "Trade Log CSV Path", Description = "If non-empty, append one row per closed trade. Use full path. Empty = disable.", Order = 1, GroupName = "Logging")]
        public string CsvPath { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.2";
        private const string CSV_HEADER =
            "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason,mfe_pts,mae_pts,capture_pct";

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

        // ── Risk manager (DynamicRiskManager — see class below) ───────────
        private DynamicRiskManager riskMgr;

        // ── Stagnation monitor (StagnationMonitor — see class below) ──────
        // Re-instantiated on each fill (flip + fresh entry) since the class's
        // ResetState is private; recreating is the public-API-preserving way
        // to wipe consecutiveNegativeBars / lastEvaluatedBar between trades.
        private StagnationMonitor stagnationMon;

        // ── In-trade MFE / MAE tracking (for CSV audit) ───────────────────
        private double currentTradeMfePts;   // peak unrealized profit (points)
        private double currentTradeMaePts;   // peak unrealized loss (points, positive value)

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "ZigzagRunner_v1.2";
                Description                      = "Zigzag pivot-retracement + trailing stop + CSV ledger. v" + VERSION;
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
                TrailActivatePoints              = 10.0;   // 10 pts = $20 on MNQ
                TrailDistancePoints              = 5.0;    // 5 pts = $10 on MNQ (Phase 1 floor)
                TrailPercent                     = 0.10;   // 10% of peak profit (Phase 2 ratchet)
                HardStopLossPoints               = 25.0;   // 25 pts = $50 hard catastrophic backstop
                CsvPath                          = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_v1.2_trades.csv";
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
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;
                // Risk manager: parameterized in POINTS to match existing
                // NinjaScriptProperty inputs. T2 activation derived from
                // crossover (TrailDistancePoints / TrailPercent) so a single
                // pair of params drives both phases.
                double t2ActPts = (TrailPercent > 0.0)
                    ? (TrailDistancePoints / TrailPercent)
                    : double.MaxValue;
                riskMgr = new DynamicRiskManager(
                    HardStopLossPoints,
                    TrailActivatePoints, TrailDistancePoints,
                    t2ActPts, TrailPercent,
                    RouteStopOrder);
                stagnationMon = new StagnationMonitor();
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

                // Capture %: how much of MFE we kept. 100% = exited at peak; 0% = gave back all.
                // For losers MFE may be 0 -> capture undefined -> report 0.
                double capturePct = (currentTradeMfePts > 0)
                    ? (100.0 * pnlPts / currentTradeMfePts)
                    : 0.0;

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
                    currentTradeMfePts.ToString("F4", CultureInfo.InvariantCulture),
                    currentTradeMaePts.ToString("F4", CultureInfo.InvariantCulture),
                    capturePct.ToString("F2", CultureInfo.InvariantCulture),
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
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState();
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
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        // DynamicRiskManager: reset state and place Initial SL
                        // using ACTUAL fill price as basis (gap-safe). The
                        // stopRouter callback will fire SetStopLoss against
                        // the new entry's signal name.
                        riskMgr.OnInitialFill(price, currentEntryDir, price);
                        // StagnationMonitor: fresh instance per trade (flip
                        // case never sees Flat, so internal state would carry
                        // over otherwise).
                        stagnationMon = new StagnationMonitor();
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
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;
                // DynamicRiskManager: reset state and place Initial SL using
                // actual fill price (gap-safe vs prior bar's close).
                riskMgr.OnInitialFill(price, currentEntryDir, price);
                // StagnationMonitor: fresh instance per trade.
                stagnationMon = new StagnationMonitor();
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

            // ─── Risk-managed stop (DynamicRiskManager) ─────────────────
            // Per-bar drive of the state machine. Manager updates max-PnL,
            // transitions Initial → Tier1 → Tier2 by peak profit, and routes
            // a tighter stop via the RouteStopOrder callback when the ratchet
            // moves favorably. Missed-breach handling lives in the callback.
            if (currentEntryDir != 0)
            {
                double unrealizedPts = currentEntryDir * (c - currentEntryPrice);
                if (unrealizedPts >  currentTradeMfePts) currentTradeMfePts =  unrealizedPts;
                if (-unrealizedPts > currentTradeMaePts) currentTradeMaePts = -unrealizedPts;

                riskMgr.EvaluateStopState(Position, c);

                // ─── Stagnation timeout (StagnationMonitor) ─────────────
                // Hardcoded MAX_NEGATIVE_BARS=5 inside the monitor class.
                // RequiresFlatten increments only on first call per bar
                // (lastEvaluatedBar guard). Returns true when 5 consecutive
                // negative bars have accumulated since last reset.
                if (stagnationMon.RequiresFlatten(Position, c, CurrentBar))
                {
                    if (currentEntryDir > 0)
                        ExitLong(Position.Quantity, "StagnationExitLong", "");
                    else
                        ExitShort(Position.Quantity, "StagnationExitShort", "");
                    return;
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
            // SL placement: Hard SL is placed by riskMgr.OnInitialFill() from
            // OnExecutionUpdate AFTER the fill — uses the actual fill price as
            // basis (gap-safe vs bar-close basis). Subsequent trail tightening
            // happens in OnBarUpdate via riskMgr.EvaluateStopState().
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

        // ─── DynamicRiskManager callback ─────────────────────────────────
        // Routes the manager's computed stop level into NT8. If the stop
        // is on the correct side of currentPrice, we SetStopLoss against
        // the entry signal; otherwise the trail breach was missed (price
        // already past the stop) and we market-exit immediately.
        private void RouteStopOrder(double stopPrice, int direction, double currentPrice)
        {
            if (currentEntryDir == 0) return;
            string sig = (direction > 0) ? "LongAtLowPivot" : "ShortAtHighPivot";

            bool valid = (direction > 0) ? (stopPrice < currentPrice)
                                         : (stopPrice > currentPrice);
            if (valid)
            {
                SetStopLoss(sig, CalculationMode.Price, stopPrice, false);
            }
            else
            {
                if (direction > 0)
                    ExitLong(Position.Quantity, "TrailMissedBreachLong", "");
                else
                    ExitShort(Position.Quantity, "TrailMissedBreachShort", "");
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // DynamicRiskManager
    // ─────────────────────────────────────────────────────────────────────
    // State-machine stop manager. Tracks max unrealized profit (in points)
    // and walks Null → Initial → Tier1 → Tier2:
    //
    //   Initial : entry ± initialStopPts                  (hard SL)
    //   Tier1   : peak  ± t1TrailPts                      (fixed-distance trail)
    //   Tier2   : peak  ± max(t1TrailPts, t2TrailPct·peak)(percent-locked trail)
    //
    // Constraints:
    //   - All distances expressed in PRICE POINTS so the strategy's
    //     existing NinjaScriptProperty inputs (HardStopLossPoints,
    //     TrailActivatePoints, TrailDistancePoints, TrailPercent) drive
    //     behavior unchanged.
    //   - Ratchet is one-directional: stop can only move favorably.
    //   - The strategy supplies a routing callback. Manager does no
    //     SetStopLoss / ExitLong directly — keeps it Strategy-agnostic
    //     and unit-testable in isolation.
    // ═════════════════════════════════════════════════════════════════════
    public enum StopState { Null, Initial, Tier1, Tier2 }

    public class DynamicRiskManager
    {
        // Configuration (all in price points)
        private readonly double initialStopPts;
        private readonly double t1ActivationPts;
        private readonly double t1TrailPts;
        private readonly double t2ActivationPts;
        private readonly double t2TrailPct;

        // Callback: (newStopPrice, direction +1/-1, currentPrice for missed-breach check)
        private readonly Action<double, int, double> stopRouter;

        // State
        private StopState currentState   = StopState.Null;
        private double maxUnrealizedPts  = 0.0;
        private double currentStopPrice  = 0.0;

        public DynamicRiskManager(
            double initialStopPts,
            double t1ActivationPts, double t1TrailPts,
            double t2ActivationPts, double t2TrailPct,
            Action<double, int, double> stopRouter)
        {
            this.initialStopPts   = initialStopPts;
            this.t1ActivationPts  = t1ActivationPts;
            this.t1TrailPts       = t1TrailPts;
            this.t2ActivationPts  = t2ActivationPts;
            this.t2TrailPct       = t2TrailPct;
            this.stopRouter       = stopRouter;
        }

        public StopState State          { get { return currentState; } }
        public double    MaxUnrealized  { get { return maxUnrealizedPts; } }
        public double    CurrentStop    { get { return currentStopPrice; } }

        // Driven from OnBarUpdate per bar close while in position.
        public void EvaluateStopState(Position position, double currentPrice)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return;

            int direction = (position.MarketPosition == MarketPosition.Long) ? 1 : -1;
            double unrealizedPts = direction * (currentPrice - position.AveragePrice);
            if (unrealizedPts > maxUnrealizedPts) maxUnrealizedPts = unrealizedPts;

            DetermineState(maxUnrealizedPts);
            CalculateAndRouteStop(position, currentPrice);
        }

        // Driven from OnExecutionUpdate at fill time. Resets state and places
        // the Initial SL using the ACTUAL fill price (gap-safe).
        public void OnInitialFill(double fillPrice, int direction, double currentPrice)
        {
            ResetState();
            currentState = StopState.Initial;
            if (initialStopPts > 0.0)
            {
                double initialStop = fillPrice - direction * initialStopPts;
                currentStopPrice   = initialStop;
                if (stopRouter != null)
                    stopRouter(initialStop, direction, currentPrice);
            }
        }

        public void ResetState()
        {
            currentState      = StopState.Null;
            maxUnrealizedPts  = 0.0;
            currentStopPrice  = 0.0;
        }

        private void DetermineState(double maxPts)
        {
            if (t2ActivationPts > 0.0 && maxPts >= t2ActivationPts)
                currentState = StopState.Tier2;
            else if (t1ActivationPts > 0.0 && maxPts >= t1ActivationPts)
                currentState = StopState.Tier1;
            else if (currentState == StopState.Null)
                currentState = StopState.Initial;
        }

        private void CalculateAndRouteStop(Position position, double currentPrice)
        {
            int direction = (position.MarketPosition == MarketPosition.Long) ? 1 : -1;
            double entry  = position.AveragePrice;
            double newStop;

            if (currentState == StopState.Initial)
            {
                if (initialStopPts <= 0.0) return;
                newStop = entry - direction * initialStopPts;
            }
            else if (currentState == StopState.Tier1)
            {
                if (t1TrailPts <= 0.0) return;
                double peakPrice = entry + direction * maxUnrealizedPts;
                newStop = peakPrice - direction * t1TrailPts;
            }
            else if (currentState == StopState.Tier2)
            {
                double peakPrice = entry + direction * maxUnrealizedPts;
                double trailPts  = Math.Max(t1TrailPts, t2TrailPct * maxUnrealizedPts);
                newStop = peakPrice - direction * trailPts;
            }
            else
            {
                return;
            }

            EnforceOrderModification(newStop, direction, currentPrice);
        }

        private void EnforceOrderModification(double calculatedStop, int direction, double currentPrice)
        {
            // Ratchet: only update if favorable (or first placement).
            bool shouldUpdate = false;
            if (currentStopPrice == 0.0)
                shouldUpdate = true;
            else if (direction > 0 && calculatedStop > currentStopPrice)
                shouldUpdate = true;
            else if (direction < 0 && calculatedStop < currentStopPrice)
                shouldUpdate = true;

            if (shouldUpdate)
            {
                currentStopPrice = calculatedStop;
                if (stopRouter != null)
                    stopRouter(calculatedStop, direction, currentPrice);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // StagnationMonitor (supplied verbatim by user, 2026-04-25)
    // ─────────────────────────────────────────────────────────────────────
    // Tracks consecutive bars of negative unrealized PnL while in position.
    // Returns true from RequiresFlatten when the count reaches a hardcoded
    // threshold (MAX_NEGATIVE_BARS = 5). Strategy is responsible for the
    // actual ExitLong/ExitShort call. Class self-resets on Flat detection;
    // strategy recreates the instance on flip fills (since ResetState is
    // private — see usage in OnExecutionUpdate flip + fresh-entry branches).
    // ═════════════════════════════════════════════════════════════════════
    public class StagnationMonitor
    {
        // Compile-time constant for max allowable negative bars
        private const int MAX_NEGATIVE_BARS = 5;

        private int consecutiveNegativeBars = 0;
        private int lastEvaluatedBar = -1;

        /// <summary>
        /// Evaluates temporal degradation of position.
        /// Must be called on OnBarUpdate to synchronize with queue position.
        /// Fails fast on null state or redundant evaluation.
        /// </summary>
        public bool RequiresFlatten(Position position, double currentPrice, int currentBarIdx)
        {
            // Defensive validation: Reset state if flat
            if (position == null || position.MarketPosition == MarketPosition.Flat)
            {
                ResetState();
                return false;
            }

            // Prevent intra-bar multiple increments
            if (currentBarIdx == lastEvaluatedBar)
            {
                return false;
            }

            lastEvaluatedBar = currentBarIdx;

            double currentPnlUsd = position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, currentPrice);

            if (currentPnlUsd < 0.0)
            {
                consecutiveNegativeBars++;
            }
            else
            {
                // Reset counter if state change returns to positive
                consecutiveNegativeBars = 0;
            }

            return consecutiveNegativeBars >= MAX_NEGATIVE_BARS;
        }

        private void ResetState()
        {
            consecutiveNegativeBars = 0;
            lastEvaluatedBar = -1;
        }
    }
}
