// =============================================================================
// ZigzagRunner 1.5-RC -- 2026-04-27  (CHOP-SPECIALIST: 2-feature bleed-score filter)
// =============================================================================
// CHANGELOG 1.5-RC (2026-04-27):
//   * UPGRADE from v1.4-RC's single-feature regime filter (mean_range_5d) to a
//     2-feature COMBINED Z-SCORE BLEED CLASSIFIER. Empirical basis from a
//     95-day NT8 backtest (1/2-4/24/2026, 1,678 trades, R=50, 1s primary):
//        Working window 1/2-2/26 = +$4,096 (+$89/day)
//        Bleed window  2/27-4/24 = -$4,648 (-$95/day)
//     The strategy is a chop specialist; on trend days it bleeds.
//
//   * NEW METHODOLOGY: bleed_score = z(prior_range) + z(range_compression)
//     where:
//        prior_range       = yesterday's high - yesterday's low
//        range_compression = prior_range / mean_range_20d
//     Z-scores use IS-calibrated normalization (1/2-3/1/2026 N=48 days).
//     When bleed_score > BleedThresholdZ, today is predicted bleed regime;
//     no new entries fire. Existing positions still flow through trail/SL/EOD.
//
//   * MVP THRESHOLD: BleedThresholdZ = -0.34 (IS-median, OOS-validated).
//     Catches 82% of OOS bleed days, OOS lift +$6,202.
//
//   * VALIDATION TOOLS:
//        tools/nt8_bleed_harvest_classifier.py  -- the underlying classifier
//        tools/v15_filter_apply.py              -- retroactive filter on trade ledger
//        tools/v15_alt_classifiers.py           -- compared vs decision tree, LR
//        tools/v15_hour_filter_walkforward.py   -- proved hour-mask doesn't help
//        tools/v15_cum_pnl_chart.py             -- visualization
//
//   * CAVEAT: in pure-chop regimes filter HURTS by ~$2k (no bleed to catch).
//     Filter is regime-volatility play, not universal alpha. Net 3-fold
//     expected +$1,272/test-period with substantial variance.
//
//   * v1.4-RC's `MaxMeanRange5dPts` property REMOVED. Replaced by
//     `BleedThresholdZ`. All other v1.4-RC properties / risk machinery /
//     RideWithTrend / OnHighPivot/OnLowPivot UNCHANGED.
//
//   * IS-calibrated constants (HARDCODED — recompute quarterly):
//        MEAN_PRIOR_RANGE       = 385.32
//        STD_PRIOR_RANGE        = 219.83
//        MEAN_RANGE_COMPRESSION = 1.0315
//        STD_RANGE_COMPRESSION  = 0.5502
//
// =============================================================================
//
// CHANGELOG 1.4-RC (2026-04-26 -- superseded by 1.5-RC):
//   * NEW: Daily regime filter. Adds a Day(1) data series via AddDataSeries
//     and computes the rolling 5-day mean daily range at the start of each
//     new trading session. If the mean range exceeds MaxMeanRange5dPts, the
//     entire session is SKIPPED (no new entries fired; existing positions
//     still managed normally). EnableRegimeFilter toggles the gate.
//
//     Empirical basis (2026-04-26 sweep, 100 random days, default params):
//        prior_day_range Spearman rho vs day PnL = -0.31
//        mean_range_5d   Spearman rho vs day PnL = -0.30
//        range_expansion Spearman rho vs day PnL = -0.27
//
//     Day WR jumps from 31% (all days) to 41% when mean_range_5d is in the
//     bottom quartile, AND mean PnL flips from -$162/day to +$20/day on the
//     filtered subset. The filter targets exactly this discrimination.
//
//     Threshold default: 350pt mean of last 5 days' (high-low). Tune via
//     MaxMeanRange5dPts in the dashboard. Lower = stricter filter (more
//     skipped days). Higher = looser filter (more days traded).
//
//     Implementation:
//        - AddDataSeries(BarsPeriodType.Day, 1) gives BarsArray[BIP_DAILY=4].
//        - At start of each new session (detected via Times[BIP_PIVOT][0].Date
//          change), iterate the prior 5 daily bars (Highs[BIP_DAILY][0..4]
//          minus Lows[BIP_DAILY][0..4]) and average.
//        - If mean range > threshold, set tradeAllowedToday = false until
//          the next session boundary.
//        - Pivot order placement gates on tradeAllowedToday.
//        - Stagnation, trail, EOD all run unchanged.
//
//   * MINIMAL LOG NOISE: per-bar DRM CALC / TRANSITION Print statements
//     from the v1.3-RC diagnostic build are removed. Only essential events
//     log to Output: regime filter decision (1/day), fresh-entry fill,
//     missed-breach. Diagnostic logger remains in the DRM class behind
//     an opt-in (riskMgr.SetDiagLogger) — wired only if needed.
//
// CHANGELOG 1.3-RC (2026-04-26):
//   * STRUCTURAL: Strategy is now multi-TF. Primary chart TF is fast-execution
//     (typically 1s), with three CONFIGURABLE secondary TFs added via
//     AddDataSeries. OnBarUpdate routes by BarsInProgress:
//
//        BarsInProgress = 0 (primary chart) — order fills, MFE/MAE update
//        BarsInProgress = 1 (Pivot TF)       — pivot detection, EOD, entry
//                                              cutoff, stagnation evaluation
//        BarsInProgress = 2 (Hard SL TF)     — Initial-state SL + Tier1 trail
//                                              evaluation
//        BarsInProgress = 3 (Trail TF)       — Tier2 fast-ratchet trail
//
//     This decouples "when do we check the stop" from "when do we trade",
//     letting risk evaluation happen at sub-pivot granularity. Defaults:
//        PivotTfSeconds   = 60  (1m pivots — preserves v1.0/v1.2 semantics)
//        HardSlTfSeconds  = 5   (5s hard SL + Tier1 trail tightening)
//        TrailTpTfSeconds = 1   (1s Tier2 fast lock-in)
//
//     Direct fix for the 2026-04-26 v1.2 cap-breach issue: with HardSl on
//     5s instead of 60s, the SL placement+modify cycle catches fast moves
//     12× sooner. Combined with isSimulatedStop=false (carried forward
//     from v1.2.7) the −739pt outliers should disappear.
//
//   * NEW: Tier2ActivationPoints is now an explicit NinjaScriptProperty
//     (default 50pt = $100 on MNQ at 1 contract). Previous v1.2 derived
//     this from TrailDistancePoints / TrailPercent, which coupled the
//     Tier1 trail floor to the Tier1→Tier2 transition. Now decoupled:
//     you can set "Tier2 starts at $200 in profit" without changing the
//     Tier1 trail floor.
//
//   * REFACTOR: DynamicRiskManager_v15.EvaluateStopState_v15() split into
//        UpdateMaxPnlAndState(pos, price) — always updates peak + state
//        RouteStopForState(pos, price, eligibleStates) — only routes
//          SetStopLoss if currentState matches one of the requested tiers
//     This lets the strategy fire stop-modify orders ONLY from the
//     appropriate TF branch (Hard SL TF for Initial+Tier1, Trail TF for
//     Tier2) without redundant order traffic.
//
//   * BUGFIX: OnInitialFill is back in OnExecutionUpdate (where it was
//     before the v1.2.6 refactor moved it to OnBarUpdate, opening a 60s
//     placement gap). The Initial-state SL is now placed at the actual
//     fill price the moment NT8 reports the fill — gap-safe.
//
//   * StagnationMonitor_v15: still per-trade-recreated, evaluated on the
//     PIVOT TF branch (so MAX_NEGATIVE_BARS counts CONSECUTIVE PIVOT-TF
//     BARS, not consecutive 1s ticks). Threshold remains MaxNegativeBars
//     property (default 5 — same semantics as v1.2 since Pivot TF defaults
//     to 60s).
//
// SETUP:
//   1. Compile in NT8 NinjaScript Editor (F5).
//   2. Apply ZigzagRunner_v1.5-RC to ANY chart — primary TF is your
//      execution-fill resolution. 1s is recommended.
//   3. Defaults match v1.0/v1.2 trade timing (60s pivots) but with
//      sub-pivot risk responsiveness.
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
    public class ZigzagRunner_v15 : Strategy
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

        // ── Multi-TF risk machinery ──────────────────────────────────────
        // All in seconds. Express minutes as 60×N. Each is added as a
        // separate data series via AddDataSeries(BarsPeriodType.Second, N).
        // BarsInProgress maps: 0=primary, 1=Pivot, 2=HardSl, 3=TrailTp.

        [NinjaScriptProperty]
        [Range(1, 86400)]
        [Display(Name = "Pivot TF (seconds)", Description = "Bar period for pivot detection / EOD / entry cutoff / stagnation. 60 = 1-minute (matches v1.0/v1.2 semantics).", Order = 1, GroupName = "Multi-TF")]
        public int PivotTfSeconds { get; set; }

        [NinjaScriptProperty]
        [Range(1, 86400)]
        [Display(Name = "Hard SL TF (seconds)", Description = "Bar period for Initial-state SL and Tier1 trail evaluation. 5 = check every 5 seconds. Lower = tighter cap enforcement.", Order = 2, GroupName = "Multi-TF")]
        public int HardSlTfSeconds { get; set; }

        [NinjaScriptProperty]
        [Range(1, 86400)]
        [Display(Name = "Trail TP TF (seconds)", Description = "Bar period for Tier2 high-profit trail (the percent ratchet). 1 = ratchet every second. Lower = tighter trail.", Order = 3, GroupName = "Multi-TF")]
        public int TrailTpTfSeconds { get; set; }

        // ── Stop loss & trail tier configuration ─────────────────────────

        [NinjaScriptProperty]
        [Display(Name = "Hard Stop Loss (points)", Description = "Initial-state catastrophic backstop. 25 pts = $50 max loss/trade on MNQ. Set 0 to disable.", Order = 1, GroupName = "StopLoss")]
        public double HardStopLossPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Tier1 Activate (points)", Description = "Peak unrealized profit (in points) needed to arm Tier1 trail. Default 10pt = $20 on MNQ.", Order = 1, GroupName = "TrailStop")]
        public double Tier1ActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Tier1 Trail Distance (points)", Description = "Tier1 fixed-distance trail floor. Trail = HWM - this many points (or MAX(this, percent×peak) once Tier2 is active).", Order = 2, GroupName = "TrailStop")]
        public double Tier1TrailDistancePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Tier2 Activate (points)", Description = "Peak unrealized profit (in points) at which trail switches from fixed-distance to percentage. Default 50pt = $100 on MNQ. Was derived in v1.2; now configurable independently.", Order = 3, GroupName = "TrailStop")]
        public double Tier2ActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 1.0)]
        [Display(Name = "Tier2 Trail Percent", Description = "Tier2 trail fraction of HWM profit. 0.10 = locks 90%. Effective dist = MAX(Tier1TrailDistance, this × peak). Set 0 to keep fixed-distance only.", Order = 4, GroupName = "TrailStop")]
        public double Tier2TrailPercent { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Negative Bars", Description = "Consecutive negative PIVOT-TF bars before stagnation flatten. 0 = disable but keep tracking. Counts in Pivot TF units (default 60s).", Order = 1, GroupName = "Stagnation")]
        public int MaxNegativeBars { get; set; }

        // ── Regime filter (v1.5-RC) ──────────────────────────────────────
        // At the start of each session, compute mean(high-low) across the
        // prior 5 daily bars. If it exceeds MaxMeanRange5dPts, skip the day.
        // EnableRegimeFilter=false bypasses entirely (= v1.3-RC behavior).

        [NinjaScriptProperty]
        [Display(Name = "Enable Bleed Filter", Description = "If true, skip trading on sessions where the bleed_score exceeds BleedThresholdZ. Filter evaluated once per session at start of first Pivot TF bar. Existing positions still managed; only new entries are gated. False = behave like v1.3-RC (no filter).", Order = 1, GroupName = "RegimeFilter")]
        public bool EnableRegimeFilter { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Bleed Threshold Z", Description = "Z-score threshold for the v1.5-RC bleed filter. bleed_score = z(prior_range) + z(range_compression). When bleed_score > this threshold, skip the session. MVP default = -0.34 (IS-median, OOS-validated). Conservative -0.5 (highest $/kept-day). Aggressive +0.75 (biggest aggregate). AVOID 0.0 (empirical local min on the threshold curve).", Order = 2, GroupName = "RegimeFilter")]
        public double BleedThresholdZ { get; set; }

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "Range Mean Lookback Days", Description = "Number of prior daily bars used for the mean_range_20d computation in range_compression. Default 20.", Order = 3, GroupName = "RegimeFilter")]
        public int RangeMeanLookbackDays { get; set; }

        // ── Direction modifier (v1.5-RC, added 2026-04-26) ───────────────
        // The strategy's pivot logic produces newPivotDir ∈ {+1, −1}:
        //   +1 = HIGH pivot just confirmed  → default mapping: EnterShort
        //   −1 = LOW  pivot just confirmed  → default mapping: EnterLong
        // RideWithTrend=true multiplies newPivotDir by −1 before the entry
        // decision, flipping the strategy from MEAN-REVERSION (fade the
        // breakout) to TREND-FOLLOWING (ride with the prior leg's
        // continuation).
        // Empirical basis: 100-day Python sweep showed flipped direction
        // produces +$162/day mean (vs −$162 for counter), 68.6% Day WR
        // (vs 31.4%), and the regime correlation flips from −0.30 to +0.30
        // (high-vol days now favor the strategy instead of breaking it).

        [NinjaScriptProperty]
        [Display(Name = "Ride With Trend", Description = "If true, FLIP direction at every pivot: HIGH pivot -> LONG (continuation), LOW pivot -> SHORT. Default false = counter-trend (fade pivots, original v1.0 logic). Toggle this to A/B test direction without redeploying.", Order = 3, GroupName = "RegimeFilter")]
        public bool RideWithTrend { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Log CSV Path", Description = "If non-empty, append one row per closed trade. Use full path. Empty = disable.", Order = 1, GroupName = "Logging")]
        public string CsvPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Regime Log CSV Path", Description = "If non-empty, append one row per session-start regime decision (date, mean_range_5d, threshold, trade_today). Empty = disable.", Order = 2, GroupName = "Logging")]
        public string RegimeLogPath { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.5-RC";
        private const string CSV_HEADER =
            "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason,mfe_pts,mae_pts,capture_pct,max_neg_bars";

        // ── BarsInProgress index constants ───────────────────────────────
        private const int BIP_PRIMARY = 0;   // chart TF (typically 1s for fills)
        private const int BIP_PIVOT   = 1;   // PivotTfSeconds
        private const int BIP_HARDSL  = 2;   // HardSlTfSeconds
        private const int BIP_TRAILTP = 3;   // TrailTpTfSeconds
        private const int BIP_DAILY   = 4;   // Daily series for regime filter (v1.4)

        // ── Zigzag state ─────────────────────────────────────────────────
        // direction: 0 = undefined (before first pivot), +1 = up leg, -1 = down leg
        private int direction;
        private double extremePrice;
        private int extremeBarIdx;
        private int lastPivotDir;      // +1 = high pivot just formed, -1 = low pivot, 0 = none
        private double lastPivotPrice;

        // ── Trade tracking for CSV logging ────────────────────────────────
        private double currentEntryPrice;
        private DateTime currentEntryTime;
        private int currentEntryDir;       // +1 long, -1 short, 0 flat
        private int currentEntryQty;
        private string currentEntryReason;
        private readonly object csvLock = new object();

        // ── Risk manager + stagnation monitor ─────────────────────────────
        private DynamicRiskManager_v15 riskMgr;
        private StagnationMonitor_v15 stagnationMon;

        // ── In-trade MFE / MAE tracking (for CSV audit) ───────────────────
        private double currentTradeMfePts;
        private double currentTradeMaePts;

        // ── Regime filter state (v1.4) ────────────────────────────────────
        // tradeAllowedToday is set at the start of each session based on the
        // 5-day mean daily range vs MaxMeanRange5dPts. When false, no new
        // entries are submitted; existing positions are still managed.
        private DateTime currentSessionDate = DateTime.MinValue;
        private bool tradeAllowedToday      = true;
        private double lastEvalBleedScore   = 0.0;     // most recent bleed_score
        private double lastEvalPriorRange   = 0.0;
        private double lastEvalRangeComp    = 0.0;
        private readonly object regimeLogLock = new object();

        // ── IS-calibrated bleed-score normalization (v1.5-RC) ─────────────
        // Computed from 1/2-3/1/2026, N=48 days. Recompute quarterly via
        // tools/nt8_bleed_harvest_classifier.py.
        private const double MEAN_PRIOR_RANGE       = 385.32;
        private const double STD_PRIOR_RANGE        = 219.83;
        private const double MEAN_RANGE_COMPRESSION = 1.0315;
        private const double STD_RANGE_COMPRESSION  = 0.5502;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "ZigzagRunner_v1.5-RC";
                Description                      = "Multi-TF zigzag pivots + 3-TF risk machinery (Pivot/SL/Trail) + StagnationMonitor_v15 + CSV ledger. v" + VERSION;
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

                // Trade-trigger defaults (match v1.0/v1.2 sweet spot)
                RPoints                          = 30.0;
                Contracts                        = 1;
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;

                // Multi-TF defaults — Pivot 1m, Hard SL 5s, Trail 1s.
                PivotTfSeconds                   = 60;
                HardSlTfSeconds                  = 5;
                TrailTpTfSeconds                 = 1;

                // Risk tier defaults (= v1.2 defaults, now decoupled).
                HardStopLossPoints               = 25.0;   // $50 catastrophic cap
                Tier1ActivatePoints              = 10.0;   // arm Tier1 at $20 unrealized
                Tier1TrailDistancePoints         = 5.0;    // $10 trail floor
                Tier2ActivatePoints              = 50.0;   // switch to percent at $100 unrealized
                Tier2TrailPercent                = 0.10;   // 10% lock (90% giveback ceiling)

                MaxNegativeBars                  = 5;

                // Regime filter defaults (v1.4)
                EnableRegimeFilter               = true;
                BleedThresholdZ                  = -0.34;   // OOS-validated MVP threshold
                RangeMeanLookbackDays            = 20;
                RideWithTrend                    = false;   // false = counter (v1.0 logic), true = with-trend

                CsvPath                          = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_v1.5_trades.csv";
                RegimeLogPath                    = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_v1.5_regime_log.csv";
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
                currentTradeMfePts   = 0.0;
                currentTradeMaePts   = 0.0;
                currentSessionDate   = DateTime.MinValue;
                tradeAllowedToday    = true;
                lastEvalBleedScore   = 0.0;
                lastEvalPriorRange   = 0.0;
                lastEvalRangeComp    = 0.0;

                // Add the three intra-day secondary series. Order matters:
                // BarsInProgress indexes follow the call order.
                AddDataSeries(BarsPeriodType.Second, PivotTfSeconds);    // BIP_PIVOT   = 1
                AddDataSeries(BarsPeriodType.Second, HardSlTfSeconds);   // BIP_HARDSL  = 2
                AddDataSeries(BarsPeriodType.Second, TrailTpTfSeconds);  // BIP_TRAILTP = 3
                // Daily series for regime filter (v1.4)
                AddDataSeries(BarsPeriodType.Day, 1);                     // BIP_DAILY   = 4

                riskMgr = new DynamicRiskManager_v15(
                    HardStopLossPoints,
                    Tier1ActivatePoints, Tier1TrailDistancePoints,
                    Tier2ActivatePoints, Tier2TrailPercent,
                    RouteStopOrder);
                // DRM diag logger left null (silent). Uncomment the line below
                // to enable per-bar state-transition + stop-calc Prints during
                // troubleshooting. Adds significant Output panel noise.
                // riskMgr.SetDiagLogger(s => Print("[v1.5-RC " + Time[0].ToString("HH:mm:ss") + "] " + s));
                stagnationMon = new StagnationMonitor_v15(MaxNegativeBars);
                EnsureCsvHeader();
                EnsureRegimeLogHeader();
            }
        }

        // ─── CSV ledger ───────────────────────────────────────────────────
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
                Print("ZigzagRunner v1.5-RC CSV init error: " + ex.Message);
            }
        }

        private static string CsvEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.Contains(",") || s.Contains("\""))
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }

        // ─── Regime filter log (v1.4) ─────────────────────────────────────
        private void EnsureRegimeLogHeader()
        {
            if (string.IsNullOrWhiteSpace(RegimeLogPath)) return;
            try
            {
                string dir = Path.GetDirectoryName(RegimeLogPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);
                if (!File.Exists(RegimeLogPath))
                {
                    lock (regimeLogLock)
                    {
                        File.WriteAllText(RegimeLogPath,
                            "log_time_utc,session_date,prior_range,range_compression,bleed_score,threshold_z,trade_today,enable_filter" + Environment.NewLine);
                    }
                }
            }
            catch (Exception ex)
            {
                Print("ZigzagRunner v1.5-RC regime-log init error: " + ex.Message);
            }
        }

        private void AppendRegimeLog(DateTime sessionDate, double bleedScore, bool tradeToday)
        {
            if (string.IsNullOrWhiteSpace(RegimeLogPath)) return;
            try
            {
                string row = string.Join(",", new string[] {
                    DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                    sessionDate.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                    lastEvalPriorRange.ToString("F4", CultureInfo.InvariantCulture),
                    lastEvalRangeComp.ToString("F4", CultureInfo.InvariantCulture),
                    bleedScore.ToString("F4", CultureInfo.InvariantCulture),
                    BleedThresholdZ.ToString("F4", CultureInfo.InvariantCulture),
                    tradeToday ? "true" : "false",
                    EnableRegimeFilter ? "true" : "false",
                });
                lock (regimeLogLock)
                {
                    File.AppendAllText(RegimeLogPath, row + Environment.NewLine);
                }
            }
            catch (Exception ex)
            {
                Print("ZigzagRunner v1.5-RC regime-log append error: " + ex.Message);
            }
        }

        // Compute prior_range (yesterday's high-low) and mean_range_20d
        // (rolling N-day mean prior to today). Returns (NaN, NaN) if insufficient
        // daily history.
        private (double priorRange, double rangeCompression) ComputeBleedFeatures()
        {
            if (CurrentBars[BIP_DAILY] < RangeMeanLookbackDays)
                return (double.NaN, double.NaN);
            double priorRange = Highs[BIP_DAILY][0] - Lows[BIP_DAILY][0];
            double sum = 0.0;
            for (int i = 0; i < RangeMeanLookbackDays; i++)
                sum += Highs[BIP_DAILY][i] - Lows[BIP_DAILY][i];
            double meanRangeN = sum / RangeMeanLookbackDays;
            double rangeCompression = (meanRangeN > 0)
                ? priorRange / meanRangeN
                : double.NaN;
            return (priorRange, rangeCompression);
        }

        // Called from the Pivot TF branch on the first bar of each new session.
        // v1.5-RC: combined-z bleed-score classifier.
        //   bleed_score = z(prior_range) + z(range_compression)
        // tradeAllowedToday = (bleed_score <= BleedThresholdZ).
        private void EvaluateRegimeFilter(DateTime sessionDate)
        {
            if (!EnableRegimeFilter)
            {
                tradeAllowedToday = true;
                return;
            }
            var (priorRange, rangeCompression) = ComputeBleedFeatures();
            if (double.IsNaN(priorRange) || double.IsNaN(rangeCompression))
            {
                // Warmup -- default to trading.
                tradeAllowedToday = true;
                lastEvalBleedScore = 0.0;
                lastEvalPriorRange = 0.0;
                lastEvalRangeComp  = 0.0;
                return;
            }
            double zPriorRange = (priorRange       - MEAN_PRIOR_RANGE)       / STD_PRIOR_RANGE;
            double zRangeComp  = (rangeCompression - MEAN_RANGE_COMPRESSION) / STD_RANGE_COMPRESSION;
            double bleedScore  = zPriorRange + zRangeComp;
            lastEvalPriorRange = priorRange;
            lastEvalRangeComp  = rangeCompression;
            lastEvalBleedScore = bleedScore;
            tradeAllowedToday  = bleedScore <= BleedThresholdZ;
            string mode = RideWithTrend ? "WITH-trend" : "COUNTER-trend";
            Print("[v1.5-RC " + sessionDate.ToString("yyyy-MM-dd") +
                  "] BleedFilter: priorRange=" + priorRange.ToString("F1") +
                  " rangeComp=" + rangeCompression.ToString("F3") +
                  " z(pr)=" + zPriorRange.ToString("F2") +
                  " z(rc)=" + zRangeComp.ToString("F2") +
                  " bleed=" + bleedScore.ToString("F2") +
                  " thr=" + BleedThresholdZ.ToString("F2") +
                  " -> trade=" + tradeAllowedToday +
                  " mode=" + mode);
            AppendRegimeLog(sessionDate, bleedScore, tradeAllowedToday);
        }

        private void AppendTradeCsv(DateTime exitTime, string exitReason, double exitPrice, int qty)
        {
            if (string.IsNullOrWhiteSpace(CsvPath)) return;
            if (currentEntryDir == 0) return;
            try
            {
                double pnlPts  = currentEntryDir * (exitPrice - currentEntryPrice);
                double pnlUsd  = pnlPts * 2.0 * qty;   // MNQ: $2/pt/contract
                double heldMin = (exitTime - currentEntryTime).TotalMinutes;
                string dir     = (currentEntryDir > 0) ? "long" : "short";
                string day     = currentEntryTime.ToUniversalTime().ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
                DateTime exitUtc  = exitTime.ToUniversalTime();
                DateTime entryUtc = currentEntryTime.ToUniversalTime();

                double capturePct = (currentTradeMfePts > 0)
                    ? (100.0 * pnlPts / currentTradeMfePts) : 0.0;

                int trackedMaxBars = (stagnationMon != null) ? stagnationMon.MaxConsecutiveNegative : 0;

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
                    trackedMaxBars.ToString(CultureInfo.InvariantCulture),
                });

                lock (csvLock)
                {
                    File.AppendAllText(CsvPath, row + Environment.NewLine);
                }
            }
            catch (Exception ex)
            {
                Print("ZigzagRunner v1.5-RC CSV append error: " + ex.Message);
            }
        }

        // OnExecutionUpdate fires on every fill. Tracks entry state, appends
        // CSV row when a trade closes, and places the Initial-state hard SL
        // at the actual fill price (gap-safe — closes the 60s placement gap
        // that v1.2.6 introduced).
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
                        // Flip in same fill. New direction inferred from
                        // resulting market position. Initial SL placed
                        // immediately at the actual fill price.
                        currentEntryDir    = (marketPosition == MarketPosition.Long) ? +1 : -1;
                        currentEntryPrice  = price;
                        currentEntryTime   = time;
                        currentEntryQty    = Contracts;
                        // Signal name reflects the active mode (counter vs with-trend).
                        if (marketPosition == MarketPosition.Long)
                            currentEntryReason = RideWithTrend ? "LongAtHighPivot" : "LongAtLowPivot";
                        else
                            currentEntryReason = RideWithTrend ? "ShortAtLowPivot" : "ShortAtHighPivot";
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        // Place Initial SL at fill price NOW (no 60s gap).
                        riskMgr.OnInitialFill(price, currentEntryDir, price);
                        // Fresh stagnation monitor per trade.
                        stagnationMon = new StagnationMonitor_v15(MaxNegativeBars);
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
                currentEntryQty    = Contracts;
                currentEntryReason = orderName;
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;
                Print("[v1.5-RC " + time.ToString("HH:mm:ss") + "] STRAT FRESH-ENTRY dir=" + currentEntryDir +
                      " fillPx=" + price.ToString("F2") + " posAvg=" + Position.AveragePrice.ToString("F2") +
                      " orderName=" + orderName);
                riskMgr.OnInitialFill(price, currentEntryDir, price);
                stagnationMon = new StagnationMonitor_v15(MaxNegativeBars);
            }
        }

        // OnBarUpdate is multi-TF. Route by BarsInProgress.
        protected override void OnBarUpdate()
        {
            if (CurrentBars[BIP_PRIMARY] < BarsRequiredToTrade) return;

            // Defensive panic-close (1.0.1). Runs on every BIP — cheap and
            // catches stale orders before they compound.
            if (Math.Abs(Position.Quantity) > Contracts)
            {
                Print("ZigzagRunner v1.5-RC SAFETY: Position.Quantity=" + Position.Quantity +
                      " exceeds Contracts=" + Contracts + ", panic-closing.");
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "SafetyPanicLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "SafetyPanicShort", "");
                return;
            }

            if (BarsInProgress == BIP_PRIMARY)
            {
                OnPrimaryBarUpdate();
                return;
            }
            if (BarsInProgress == BIP_PIVOT)
            {
                if (CurrentBars[BIP_PIVOT] < BarsRequiredToTrade) return;
                OnPivotBarUpdate();
                return;
            }
            if (BarsInProgress == BIP_HARDSL)
            {
                if (CurrentBars[BIP_HARDSL] < BarsRequiredToTrade) return;
                OnHardSlBarUpdate();
                return;
            }
            if (BarsInProgress == BIP_TRAILTP)
            {
                if (CurrentBars[BIP_TRAILTP] < BarsRequiredToTrade) return;
                OnTrailTpBarUpdate();
                return;
            }
            // BIP_DAILY = no-op (daily series is read on demand, not driven).
            if (BarsInProgress == BIP_DAILY)
            {
                return;
            }
        }

        // Primary chart bars (typically 1s). MFE/MAE update only — peak
        // tracked at finest resolution available. Risk routing happens on
        // the dedicated risk-TF branches.
        private void OnPrimaryBarUpdate()
        {
            if (currentEntryDir == 0) return;
            double c = Closes[BIP_PRIMARY][0];
            double unrealizedPts = currentEntryDir * (c - currentEntryPrice);
            if (unrealizedPts >  currentTradeMfePts) currentTradeMfePts =  unrealizedPts;
            if (-unrealizedPts > currentTradeMaePts) currentTradeMaePts = -unrealizedPts;
            // Pass strategy-captured entry (NOT Position.AveragePrice) so DRM
            // tracks against the same basis the strategy uses for MFE/MAE.
            riskMgr.UpdateMaxPnlAndState(Position, c, currentEntryPrice, currentEntryDir);
        }

        // Pivot TF (default 60s). Drives:
        //   - Pivot extreme tracking + R-retracement detection
        //   - EOD force-close
        //   - Entry cutoff window
        //   - Stagnation evaluation (counts CONSECUTIVE PIVOT-TF BARS underwater)
        //   - Order placement (EnterLong/EnterShort)
        private void OnPivotBarUpdate()
        {
            double c = Closes[BIP_PIVOT][0];
            DateTime barUtc    = Times[BIP_PIVOT][0].ToUniversalTime();
            DateTime sessionDate = Times[BIP_PIVOT][0].Date;  // local date — used as session marker
            int minsOfDay      = barUtc.Hour * 60 + barUtc.Minute;
            int eodMins        = EodHourUtc * 60 + EodMinuteUtc;
            int entryCutMins   = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

            // ─── Regime filter (v1.4) ────────────────────────────────────
            // Re-evaluate on session-date change. Uses the prior 5 closed
            // daily bars; doesn't need the current day's data.
            if (sessionDate != currentSessionDate)
            {
                currentSessionDate = sessionDate;
                EvaluateRegimeFilter(sessionDate);
            }

            // EOD force-close
            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "EodExitLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "EodExitShort", "");
                return;
            }

            // Initialize extreme on first Pivot-TF close
            if (double.IsNaN(extremePrice))
            {
                extremePrice  = c;
                extremeBarIdx = CurrentBars[BIP_PIVOT];
                return;
            }

            // Zigzag state machine
            bool pivotConfirmed = false;
            int newPivotDir     = 0;

            if (direction == 0)
            {
                if (c - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;       // extreme was a LOW pivot
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
                else if (extremePrice - c >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;       // HIGH pivot
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
            }
            else if (direction == +1)
            {
                if (c > extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBars[BIP_PIVOT];
                }
                else if (extremePrice - c >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = +1;
                    lastPivotPrice = extremePrice;
                    direction      = -1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
            }
            else  // direction == -1
            {
                if (c < extremePrice)
                {
                    extremePrice  = c;
                    extremeBarIdx = CurrentBars[BIP_PIVOT];
                }
                else if (c - extremePrice >= RPoints)
                {
                    pivotConfirmed = true;
                    newPivotDir    = -1;
                    lastPivotPrice = extremePrice;
                    direction      = +1;
                    extremePrice   = c;
                    extremeBarIdx  = CurrentBars[BIP_PIVOT];
                }
            }

            bool isFlipping = false;

            if (pivotConfirmed)
            {
                lastPivotDir = newPivotDir;

                // Gate new entries on (a) entry cutoff window AND (b) regime
                // filter. If filter says skip-the-day, NO new entry orders
                // fire. Existing positions still flow through trail/SL/EOD.
                if (minsOfDay < entryCutMins && tradeAllowedToday)
                {
                    // Apply direction modifier. dirMod = -1 flips entries:
                    //   HIGH pivot → LONG (with-trend continuation)
                    //   LOW  pivot → SHORT (with-trend continuation)
                    int dirMod = RideWithTrend ? -1 : +1;
                    int effectiveDir = newPivotDir * dirMod;

                    if (effectiveDir == +1)
                    {
                        // Want SHORT
                        if (!(Position.MarketPosition == MarketPosition.Short && Position.Quantity >= Contracts))
                        {
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong(Position.Quantity, "FlipExitLong", "");
                            string sig = RideWithTrend ? "ShortAtLowPivot" : "ShortAtHighPivot";
                            EnterShort(Contracts, sig);
                            isFlipping = true;
                        }
                    }
                    else
                    {
                        // Want LONG
                        if (!(Position.MarketPosition == MarketPosition.Long && Position.Quantity >= Contracts))
                        {
                            if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort(Position.Quantity, "FlipExitShort", "");
                            string sig = RideWithTrend ? "LongAtHighPivot" : "LongAtLowPivot";
                            EnterLong(Contracts, sig);
                            isFlipping = true;
                        }
                    }
                }
            }

            // Stagnation evaluation — only on the Pivot TF branch so
            // MAX_NEGATIVE_BARS counts CONSECUTIVE PIVOT-TF BARS, not 1s ticks.
            // Skip on bars where we just flipped (fresh trade gets full window).
            if (!isFlipping && currentEntryDir != 0 && stagnationMon != null)
            {
                if (stagnationMon.RequiresFlatten(Position, c, CurrentBars[BIP_PIVOT], currentEntryPrice))
                {
                    if (currentEntryDir > 0)
                        ExitLong(Position.Quantity, "StagnationExitLong", "");
                    else
                        ExitShort(Position.Quantity, "StagnationExitShort", "");
                }
            }
        }

        // Hard SL TF (default 5s). Drives Initial-state SL re-evaluation
        // and Tier1 trail tightening. Tier2 is handled on the Trail TF
        // branch (typically faster for fine ratcheting).
        private void OnHardSlBarUpdate()
        {
            if (currentEntryDir == 0) return;
            double c = Closes[BIP_HARDSL][0];
            riskMgr.UpdateMaxPnlAndState(Position, c, currentEntryPrice, currentEntryDir);
            riskMgr.RouteStopForState(Position, c, currentEntryPrice, currentEntryDir, StopState_v15.Initial, StopState_v15.Tier1);
        }

        // Trail TP TF (default 1s). Drives Tier2 percent-trail ratcheting.
        // Highest frequency = tightest profit lock-in once trade reaches
        // Tier2ActivatePoints of unrealized profit.
        private void OnTrailTpBarUpdate()
        {
            if (currentEntryDir == 0) return;
            double c = Closes[BIP_TRAILTP][0];
            riskMgr.UpdateMaxPnlAndState(Position, c, currentEntryPrice, currentEntryDir);
            riskMgr.RouteStopForState(Position, c, currentEntryPrice, currentEntryDir, StopState_v15.Tier2);
        }

        // ─── DynamicRiskManager_v15 callback ─────────────────────────────────
        // Validates the computed stop against the bar-close basis price.
        // If the stop is on the correct side of price (= still ahead of us
        // by at least gapBuffer pts), routes via SetStopLoss. Otherwise
        // the trail breach was missed (price has already crossed the level)
        // — fire a market exit.
        private void RouteStopOrder(double stopPrice, int direction, double currentPrice)
        {
            if (currentEntryDir == 0) return;

            // Signal name MUST match the EnterLong/EnterShort signal that
            // opened the trade. The with-trend mode uses different signal
            // strings so the CSV ledger reflects the actual entry semantics.
            string sig;
            if (direction > 0)
                sig = RideWithTrend ? "LongAtHighPivot" : "LongAtLowPivot";
            else
                sig = RideWithTrend ? "ShortAtLowPivot" : "ShortAtHighPivot";
            double gapBuffer = 1.0;

            bool valid;
            if (direction > 0)
                valid = stopPrice <= (currentPrice - gapBuffer);
            else
                valid = stopPrice >= (currentPrice + gapBuffer);

            if (valid)
            {
                // isSimulatedStop = false: route as exchange-stop semantics.
                SetStopLoss(sig, CalculationMode.Price, stopPrice, false);
            }
            else
            {
                Print("[v1.5-RC " + Time[0].ToString("HH:mm:ss") +
                      "] MISSED-BREACH sig=" + sig +
                      " stopPx=" + stopPrice.ToString("F2") +
                      " currentPx=" + currentPrice.ToString("F2"));
                if (direction > 0)
                    ExitLong(Position.Quantity, "TrailMissedBreachLong", "");
                else
                    ExitShort(Position.Quantity, "TrailMissedBreachShort", "");
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // StopState_v15 + DynamicRiskManager_v15 (v1.5-RC variant)
    // ─────────────────────────────────────────────────────────────────────
    // CHANGE FROM v1.2: EvaluateStopState_v15() split into two operations so
    // the strategy can drive max-PnL tracking from one TF and stop
    // routing from another. Old EvaluateStopState_v15() retained as a
    // convenience wrapper.
    //
    //   UpdateMaxPnlAndState(pos, price)
    //     → updates internal maxUnrealizedPts + transitions state machine
    //     → returns current state for caller's information
    //     → DOES NOT route SetStopLoss
    //
    //   RouteStopForState(pos, price, params StopState_v15[] eligible)
    //     → if currentState ∈ eligible: computes the tier's stop price
    //       and invokes the routing callback (subject to ratchet guard).
    //     → DOES NOT update max-PnL
    //
    //   EvaluateStopState_v15(pos, price)  ← back-compat (calls both)
    // ═════════════════════════════════════════════════════════════════════
    public enum StopState_v15 { Null, Initial, Tier1, Tier2 }

    public class DynamicRiskManager_v15
    {
        // Configuration (price points)
        private readonly double initialStopPts;
        private readonly double t1ActivationPts;
        private readonly double t1TrailPts;
        private readonly double t2ActivationPts;
        private readonly double t2TrailPct;

        // Callback: (newStopPrice, direction +1/-1, currentPrice for missed-breach check)
        private readonly Action<double, int, double> stopRouter;

        // Diagnostic logger (wired to Strategy's Print method). Null = silent.
        private Action<string> diagLogger;

        // State
        private StopState_v15 currentState   = StopState_v15.Null;
        private double maxUnrealizedPts  = 0.0;
        private double currentStopPrice  = 0.0;

        public DynamicRiskManager_v15(
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

        public void SetDiagLogger(Action<string> logger)
        {
            this.diagLogger = logger;
        }

        public StopState_v15 State          { get { return currentState; } }
        public double    MaxUnrealized  { get { return maxUnrealizedPts; } }
        public double    CurrentStop    { get { return currentStopPrice; } }

        // Update peak unrealized + transition state machine. NO routing.
        // entryPrice is passed EXPLICITLY by the strategy (= currentEntryPrice
        // captured at fill time) instead of reading position.AveragePrice,
        // because Strategy Analyzer multi-TF backtests can leave AveragePrice
        // stale during the tick-by-tick simulation, causing DRM to see a
        // wrong basis and never update maxPts even when price went favorable.
        // Bug surfaced 2026-04-26: trades with strategy MFE=28pt were stopping
        // at entry±5pt (Tier1 formula with maxPts≈0) because DRM's
        // maxUnrealizedPts wasn't tracking. Strategy passes the same
        // currentEntryPrice it uses for its own MFE tracker — both stay in
        // sync now.
        public StopState_v15 UpdateMaxPnlAndState(Position position, double currentPrice, double entryPrice, int entryDir)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
                return currentState;
            if (entryDir == 0) return currentState;

            double unrealizedPts = entryDir * (currentPrice - entryPrice);
            if (unrealizedPts > maxUnrealizedPts) maxUnrealizedPts = unrealizedPts;

            DetermineState(maxUnrealizedPts);
            return currentState;
        }

        // Route SetStopLoss callback if currentState matches one of the
        // eligible states. Strategy passes its own captured entryPrice +
        // entryDir (not Position.AveragePrice / MarketPosition) so DRM
        // computes against the same basis the strategy is tracking.
        public void RouteStopForState(Position position, double currentPrice, double entryPrice, int entryDir, params StopState_v15[] eligibleStates)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat) return;
            if (entryDir == 0) return;
            if (eligibleStates == null || eligibleStates.Length == 0) return;

            bool matches = false;
            for (int i = 0; i < eligibleStates.Length; i++)
            {
                if (eligibleStates[i] == currentState) { matches = true; break; }
            }
            if (!matches) return;

            CalculateAndRouteStop(entryPrice, entryDir, currentPrice);
        }

        // Place Initial-state SL at fill time (called from OnExecutionUpdate).
        public void OnInitialFill(double fillPrice, int direction, double currentPrice)
        {
            ResetState();
            currentState = StopState_v15.Initial;
            if (diagLogger != null)
                diagLogger("DRM OnInitialFill fill=" + fillPrice.ToString("F2") +
                           " dir=" + direction + " initialStopPts=" + initialStopPts);
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
            currentState      = StopState_v15.Null;
            maxUnrealizedPts  = 0.0;
            currentStopPrice  = 0.0;
        }

        private void DetermineState(double maxPts)
        {
            StopState_v15 priorState = currentState;
            if (t2ActivationPts > 0.0 && maxPts >= t2ActivationPts)
                currentState = StopState_v15.Tier2;
            else if (t1ActivationPts > 0.0 && maxPts >= t1ActivationPts)
                currentState = StopState_v15.Tier1;
            else if (currentState == StopState_v15.Null)
                currentState = StopState_v15.Initial;
            // Diagnostic: log every state transition.
            if (priorState != currentState && diagLogger != null)
                diagLogger("DRM TRANSITION " + priorState + " -> " + currentState +
                           " maxPts=" + maxPts.ToString("F2") +
                           " (t1Act=" + t1ActivationPts + " t2Act=" + t2ActivationPts + ")");
        }

        private void CalculateAndRouteStop(double entry, int direction, double currentPrice)
        {
            double newStop;

            if (currentState == StopState_v15.Initial)
            {
                if (initialStopPts <= 0.0) return;
                newStop = entry - direction * initialStopPts;
            }
            else if (currentState == StopState_v15.Tier1)
            {
                if (t1TrailPts <= 0.0) return;
                double peakPrice = entry + direction * maxUnrealizedPts;
                newStop = peakPrice - direction * t1TrailPts;
            }
            else if (currentState == StopState_v15.Tier2)
            {
                double peakPrice = entry + direction * maxUnrealizedPts;
                double trailPts  = Math.Max(t1TrailPts, t2TrailPct * maxUnrealizedPts);
                newStop = peakPrice - direction * trailPts;
            }
            else
            {
                return;
            }

            // Diagnostic: log the computed stop and the state used to compute it.
            if (diagLogger != null)
                diagLogger("DRM CALC state=" + currentState +
                           " maxPts=" + maxUnrealizedPts.ToString("F2") +
                           " entry=" + entry.ToString("F2") +
                           " dir=" + direction +
                           " -> newStop=" + newStop.ToString("F2"));

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
    // StagnationMonitor_v15 (carried forward from v1.2 — runs on Pivot TF in
    // v1.5-RC so MAX_NEGATIVE_BARS counts consecutive PIVOT-TF BARS, not
    // 1s ticks. Strategy recreates the instance per trade since
    // ResetState is private.)
    // ═════════════════════════════════════════════════════════════════════
    public class StagnationMonitor_v15
    {
        private readonly int maxNegativeBars;
        private int consecutiveNegativeBars = 0;
        private int lastEvaluatedBar = -1;

        public int MaxConsecutiveNegative { get; private set; }

        public StagnationMonitor_v15(int maxNegativeBars)
        {
            this.maxNegativeBars = maxNegativeBars;
            this.MaxConsecutiveNegative = 0;
        }

        public bool RequiresFlatten(Position position, double currentPrice, int currentBarIdx, double entryPrice)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
            {
                ResetState();
                return false;
            }

            if (currentBarIdx == lastEvaluatedBar) return false;
            lastEvaluatedBar = currentBarIdx;

            int direction = (position.MarketPosition == MarketPosition.Long) ? 1 : -1;
            double currentPnlPts = direction * (currentPrice - entryPrice);

            if (currentPnlPts < 0.0)
            {
                consecutiveNegativeBars++;
                if (consecutiveNegativeBars > MaxConsecutiveNegative)
                    MaxConsecutiveNegative = consecutiveNegativeBars;
            }
            else
            {
                consecutiveNegativeBars = 0;
            }

            if (maxNegativeBars > 0)
                return consecutiveNegativeBars >= maxNegativeBars;
            return false;
        }

        private void ResetState()
        {
            consecutiveNegativeBars = 0;
            lastEvaluatedBar = -1;
            MaxConsecutiveNegative = 0;
        }
    }
}
