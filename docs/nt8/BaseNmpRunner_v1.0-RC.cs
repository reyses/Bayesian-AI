// =============================================================================
// BaseNmpRunner 1.0-RC -- 2026-04-30 (Standalone NT8 port of BASE_NMP tier)
// =============================================================================
//
// PURPOSE:
//   Native NT8 implementation of BASE_NMP (Nightmare Mean Pullback) tier.
//   This is the "trending regime / DOWN-day" specialist — captures the profit
//   that v1.0.4 / v1.0.7-RC counter-trend leaves on the table when the macro
//   regime is bearish/trending.
//
// EDGE BASIS (per overnight EDA 2026-04-29):
//   - BASE_NMP fired exclusively in late-2025 / Jan-Mar 2026 window when
//     other tiers (FADE_CALM, RIDE_AGAINST) stopped firing.
//   - Entry feature signature: lower 1m_reversion_prob (0.91), higher 1m_hurst
//     (0.71), positive 1D_dmi_diff (1.5) — i.e., trending / less mean-reverting.
//   - Tier produced 1,195 trades, $19,997 total, $16.7/trade in Python sim
//     across Jan-Mar 2026 OOS period (= ~$50/day NT8 equivalent after the 2x
//     trade-count translation factor).
//   - LinReg slope filter (T=0.5) added another +$4,162 to BASE_NMP per
//     overnight validation (within-tier 70/30 holdout: confirmed generalizes).
//
// LOGIC (from training/nightmare.py):
//
//   ENTRY (when flat):
//     |z_se| > 2.0 (ROCHE) AND vr < 1.0
//       z_se > +2  ->  SHORT (fade upper deviation)
//       z_se < -2  ->  LONG  (fade lower deviation)
//
//   EXIT (when in position):
//     |z_se| < 0.5         ->  exit (mean reached)
//     vr > 1.0             ->  exit (regime flipped to trending)
//     unrealized -loss > MaxLossPoints  ->  exit (hard stop)
//     EOD time             ->  exit (force flat)
//
//   FORMULAS:
//     z_se = (close - LinReg(N)) / StdDev(residuals, N)
//     residual[i] = close[i] - LinReg(N)[i]
//     vr = var(N-bar return) / (N * var(1-bar return))  (Lo-MacKinlay)
//
//   Where N defaults match the SFE: regression period 30 bars; VR uses
//   aggregation 5 over a sample window of 30.
//
// OPTIONAL: LINREG SLOPE FILTER (validated per overnight EDA):
//   When UseEntrySlopeFilter=true, skip an entry if abs(LinRegSlope) is
//   beyond threshold AND opposes the trade direction (don't fade against
//   strong slope). Validated improvement +$4,162/14mo Python sim, +$809
//   on within-tier holdout.
//
// INSTALLATION (RC GATE — does NOT auto-deploy live):
//   1. File lives in docs/nt8/BaseNmpRunner_v1.0-RC.cs (this file).
//   2. To run in NT8: copy to bin/Custom/Strategies/.
//   3. F5 in NT8 NinjaScript Editor.
//   4. Apply BaseNmpRunner_v1.0-RC to a 1-minute MNQ chart (or matching TF).
//   5. NT8 Strategy Analyzer for backtest validation BEFORE Sim or Live.
//
// =============================================================================

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class BaseNmpRunner_v10 : Strategy
    {
        // ── Entry / exit thresholds (BASE_NMP canonical) ─────────────────

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Z-SE Entry Threshold (ROCHE)",
                 Description = "abs(z_se) must exceed this for entry. SFE default = 2.0.",
                 Order = 1, GroupName = "NMP Entry")]
        public double ZseEntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "VR Entry Threshold",
                 Description = "variance_ratio must be BELOW this for entry. SFE default = 1.0 (mean-reverting only).",
                 Order = 2, GroupName = "NMP Entry")]
        public double VrEntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 3.0)]
        [Display(Name = "Z-SE Exit Threshold",
                 Description = "abs(z_se) below this -> exit (mean reached). SFE default = 0.5.",
                 Order = 1, GroupName = "NMP Exit")]
        public double ZseExitThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "VR Exit Threshold",
                 Description = "variance_ratio above this -> exit (regime flip). SFE default = 1.0.",
                 Order = 2, GroupName = "NMP Exit")]
        public double VrExitThreshold { get; set; }

        // ── Computation windows ──────────────────────────────────────────

        [NinjaScriptProperty]
        [Range(5, 500)]
        [Display(Name = "Regression Period (N)",
                 Description = "Bars for OLS regression in z_se denominator. SFE default = 30.",
                 Order = 1, GroupName = "Compute Windows")]
        public int RegressionPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(2, 100)]
        [Display(Name = "VR Aggregation Period",
                 Description = "q in VR(q) = var(q-bar return) / (q * var(1-bar return)). Default 5.",
                 Order = 2, GroupName = "Compute Windows")]
        public int VrAggregationPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(10, 500)]
        [Display(Name = "Variance Sample Window",
                 Description = "Window for stddev of returns in VR computation. Default 30.",
                 Order = 3, GroupName = "Compute Windows")]
        public int VarianceSampleWindow { get; set; }

        // ── Risk / size ──────────────────────────────────────────────────

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Contracts", Description = "Contracts per trade.",
                 Order = 1, GroupName = "Risk")]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Range(0, 500)]
        [Display(Name = "Max Loss Points",
                 Description = "Hard stop: flatten when unrealized loss reaches this many points. " +
                               "Default 25pt = -$50 on MNQ (matches SFE MAX_DRAWDOWN).",
                 Order = 2, GroupName = "Risk")]
        public double MaxLossPoints { get; set; }

        // ── Schedule ─────────────────────────────────────────────────────

        [NinjaScriptProperty]
        [Display(Name = "EOD Hour UTC", Order = 1, GroupName = "Schedule")]
        public int EodHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Minute UTC", Order = 2, GroupName = "Schedule")]
        public int EodMinuteUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Hour UTC", Order = 3, GroupName = "Schedule")]
        public int EntryCutoffHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Minute UTC", Order = 4, GroupName = "Schedule")]
        public int EntryCutoffMinuteUtc { get; set; }

        // ── Optional LinReg slope filter (validated overnight) ───────────

        [NinjaScriptProperty]
        [Display(Name = "Use Entry Slope Filter",
                 Description = "Skip entries where LinReg slope opposes the trade direction. " +
                               "Validated +$4,162 improvement (within-tier 70/30 holdout: +$809 generalizes). " +
                               "Default off — enable for the v1.1 expected lift.",
                 Order = 1, GroupName = "Slope Filter")]
        public bool UseEntrySlopeFilter { get; set; }

        [NinjaScriptProperty]
        [Range(2, 500)]
        [Display(Name = "Slope Filter Period",
                 Description = "LinReg lookback in bars for the entry slope filter. Default 30.",
                 Order = 2, GroupName = "Slope Filter")]
        public int SlopeFilterPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 50.0)]
        [Display(Name = "Slope Filter Threshold",
                 Description = "Skip entry if abs(slope) > this AND slope opposes direction. " +
                               "Default 0.5 = best for BASE_NMP per overnight validation.",
                 Order = 3, GroupName = "Slope Filter")]
        public double SlopeFilterThreshold { get; set; }

        // ── Costs ────────────────────────────────────────────────────────

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "Slippage (points)",
                 Description = "Per-fill slippage. MNQ tick = 0.25 pts. Default 0.25 = 1 tick.",
                 Order = 1, GroupName = "Costs")]
        public double SlippagePoints { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 20.0)]
        [Display(Name = "Commission per round-trip (USD)",
                 Description = "Default 1.90 = $0.95/side x 2 (NinjaTrader Brokerage Free).",
                 Order = 2, GroupName = "Costs")]
        public double CommissionPerRoundtripUsd { get; set; }

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0-RC";

        // ── Indicators ───────────────────────────────────────────────────
        // NOTE: NT8 has TWO separate indicators for linear regression:
        //   LinReg(period) — returns the regression LINE value as Series<double>
        //   LinRegSlope(period) — returns the SLOPE value as Series<double>
        // We need both: LinReg for z_se (line value), LinRegSlope for filter.
        private LinReg primaryLinReg;            // For z_se: regression line on close
        private LinRegSlope slopeFilterLinReg;   // For optional slope filter
        private Series<double> residuals;        // close - linreg, used for SE_N
        private Series<double> ret1;             // 1-bar returns
        private Series<double> retN;             // N-bar returns (where N = VrAggregationPeriod)

        // ── State ────────────────────────────────────────────────────────
        private double currentZSe;
        private double currentVr;
        private double currentSlope;
        private int closedRoundtripsCount;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                             = "BaseNmpRunner_v1.0-RC";
                Description                      = "Standalone BASE_NMP tier port. z_se>2 AND vr<1 fade. v" + VERSION;
                Calculate                        = Calculate.OnBarClose;
                EntriesPerDirection              = 1;
                EntryHandling                    = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy     = true;
                ExitOnSessionCloseSeconds        = 30;
                IsFillLimitOnTouch               = false;
                MaximumBarsLookBack              = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution              = OrderFillResolution.Standard;
                Slippage                         = 0;  // driven from SlippagePoints in Configure
                StartBehavior                    = StartBehavior.WaitUntilFlat;
                TimeInForce                      = TimeInForce.Gtc;
                TraceOrders                      = false;
                RealtimeErrorHandling            = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling               = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade              = 50;  // need warmup for regression + variance

                // ── BASE_NMP canonical thresholds (per training/nightmare.py) ──
                ZseEntryThreshold                = 2.0;   // ROCHE
                VrEntryThreshold                 = 1.0;   // VR_ENTRY
                ZseExitThreshold                 = 0.5;   // Z_EXIT
                VrExitThreshold                  = 1.0;   // VR_EXIT

                // Compute windows
                RegressionPeriod                 = 30;
                VrAggregationPeriod              = 5;
                VarianceSampleWindow             = 30;

                // Risk
                Contracts                        = 1;
                MaxLossPoints                    = 25.0;  // ~$50 stop on MNQ ($2/pt)

                // Schedule
                EodHourUtc                       = 20;
                EodMinuteUtc                     = 55;
                EntryCutoffHourUtc               = 20;
                EntryCutoffMinuteUtc             = 30;

                // Slope filter (off by default; recommended T=0.5 if enabled)
                UseEntrySlopeFilter              = false;
                SlopeFilterPeriod                = 30;
                SlopeFilterThreshold             = 0.5;

                // Costs
                SlippagePoints                   = 0.25;
                CommissionPerRoundtripUsd        = 1.90;
            }
            else if (State == State.Configure)
            {
                // NT8 simulation engine slippage (used in SA + Playback).
                Slippage = SlippagePoints;
                closedRoundtripsCount = 0;
            }
            else if (State == State.DataLoaded)
            {
                primaryLinReg = LinReg(Close, RegressionPeriod);
                if (UseEntrySlopeFilter)
                {
                    slopeFilterLinReg = LinRegSlope(Close, SlopeFilterPeriod);
                }
                residuals = new Series<double>(this);
                ret1      = new Series<double>(this);
                retN      = new Series<double>(this);
            }
            else if (State == State.Terminated)
            {
                int rt = closedRoundtripsCount;
                if (SystemPerformance != null && SystemPerformance.AllTrades != null)
                    rt = SystemPerformance.AllTrades.Count;
                double estCommission = rt * CommissionPerRoundtripUsd;
                Print(string.Format(
                    "[BaseNmpRunner_v{0} COST SUMMARY] roundtrips={1}  est_commission=${2:F2}  " +
                    "slippage_pts={3:F4}  per_roundtrip=${4:F2}  use_slope_filter={5}",
                    VERSION, rt, estCommission, SlippagePoints, CommissionPerRoundtripUsd,
                    UseEntrySlopeFilter));
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;

            // ─── Compute current bar features ─────────────────────────────
            double linRegVal = primaryLinReg[0];

            // residual = close - regression line value at this bar
            residuals[0] = Close[0] - linRegVal;

            // SE = stddev of residuals over the regression window
            double seVal = StdDev(residuals, RegressionPeriod)[0];
            if (seVal <= 0)
                return;  // no variance yet — skip
            currentZSe = (Close[0] - linRegVal) / seVal;

            // 1-bar and N-bar returns for variance ratio
            ret1[0] = Close[0] - Close[1];
            if (CurrentBar >= VrAggregationPeriod)
                retN[0] = Close[0] - Close[VrAggregationPeriod];
            else
                retN[0] = 0;

            double std1 = StdDev(ret1, VarianceSampleWindow)[0];
            double stdN = StdDev(retN, VarianceSampleWindow)[0];
            double var1 = std1 * std1;
            double varN = stdN * stdN;

            if (var1 <= 0)
                return;  // no 1-bar variance yet
            currentVr = varN / (VrAggregationPeriod * var1);

            // Slope (only if filter enabled, otherwise leave 0 for diagnostic)
            currentSlope = 0;
            if (UseEntrySlopeFilter)
                currentSlope = slopeFilterLinReg[0];

            // ─── Time-of-day ──────────────────────────────────────────────
            DateTime barUtc = Time[0].ToUniversalTime();
            int minsOfDay      = barUtc.Hour * 60 + barUtc.Minute;
            int eodMins        = EodHourUtc * 60 + EodMinuteUtc;
            int entryCutMins   = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

            // ─── EOD force-close ──────────────────────────────────────────
            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong(Position.Quantity, "EodExitLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort(Position.Quantity, "EodExitShort", "");
                return;
            }

            // ─── Position-state: in-trade exits ───────────────────────────
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                double refPrice = Close[0];
                double unrealizedPts = Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Points, refPrice);

                // Hard SL: if unrealized loss exceeds threshold
                if (MaxLossPoints > 0 && unrealizedPts <= -MaxLossPoints)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(Position.Quantity, "HardStopLong", "");
                    else
                        ExitShort(Position.Quantity, "HardStopShort", "");
                    return;
                }

                // Mean reached: |z_se| < ZExitThreshold
                if (Math.Abs(currentZSe) < ZseExitThreshold)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(Position.Quantity, "MeanReachedLong", "");
                    else
                        ExitShort(Position.Quantity, "MeanReachedShort", "");
                    return;
                }

                // Regime flip: vr > VrExitThreshold
                if (currentVr > VrExitThreshold)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(Position.Quantity, "RegimeFlipLong", "");
                    else
                        ExitShort(Position.Quantity, "RegimeFlipShort", "");
                    return;
                }

                // Otherwise hold position, no entry decision needed
                return;
            }

            // ─── Position-state: flat — entry decision ────────────────────
            if (minsOfDay >= entryCutMins) return;  // past cutoff, no new entries

            // BASE_NMP entry: |z_se| > ROCHE AND vr < VR_ENTRY
            if (Math.Abs(currentZSe) > ZseEntryThreshold && currentVr < VrEntryThreshold)
            {
                int actionSide = currentZSe > 0 ? -1 : +1;  // z>0 -> short, z<0 -> long

                // Optional slope filter: skip if slope strongly opposes direction
                if (UseEntrySlopeFilter && SlopeFilterThreshold > 0)
                {
                    bool slopeOpposes =
                        (actionSide > 0 && currentSlope < -SlopeFilterThreshold) ||
                        (actionSide < 0 && currentSlope > +SlopeFilterThreshold);
                    if (slopeOpposes)
                        return;  // skip entry
                }

                if (actionSide > 0)
                    EnterLong(Contracts, "NmpFadeLong");
                else
                    EnterShort(Contracts, "NmpFadeShort");
            }
        }

        protected override void OnExecutionUpdate(Cbi.Execution execution, string executionId,
            double price, int quantity, Cbi.MarketPosition marketPosition,
            string orderId, DateTime time)
        {
            // Track closed roundtrips for the cost summary in Terminated state
            if (execution != null && execution.Order != null &&
                execution.Order.OrderState == Cbi.OrderState.Filled &&
                Position.MarketPosition == MarketPosition.Flat)
            {
                closedRoundtripsCount++;
            }
        }
    }
}
