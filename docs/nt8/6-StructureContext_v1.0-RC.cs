//
// 6-StructureContext_v1.0-RC.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: Structure Context Dashboard
//
// ORIGIN (2026-07-07): built from a same-day Bayesian-AI research session
// (research/level_hold/) that tested three pieces of the user's discretionary
// method causally against 63 days of MNQ data:
//   1. Macro position-in-range (Figure_3's "where are we in the bigger
//      structure" read) -- MEASURED REAL, weak (+1-2pp on where swings
//      terminate, p=.007, research/level_hold/reports/
//      LEVEL_HOLD_FINDINGS_2026-07-07.md + pivot_level_proximity_thr20.txt).
//   2. Sigma-relative "how near" -- the user corrected the original fixed-
//      tick-radius methodology mid-session: his own band indicators (1a/1b)
//      draw at N-SIGMA, a width that breathes with volatility, so "how near"
//      must scale with the band's CURRENT sigma, never a constant. This
//      indicator follows that rule throughout.
//   3. Touch/visit COUNT as a standalone predictor of hold-vs-break --
//      MEASURED NOT ROBUST once corrected to be sigma-relative (flat across
//      buckets at 0.5sigma/1sigma zones; see docs/daily/2026-07-07.md,
//      "CORRECTION" section). Deliberately OMITTED here -- it did not
//      survive scrutiny and has no business being displayed as if it did.
//
// WHAT THIS IS: a state DASHBOARD, not a signal. Every effect behind it is a
// few-percentage-point statistical tendency, not a trigger. It exists to
// make three things a discretionary trader has to hold in their head
// simultaneously -- (a) macro position, (b) volatility-relative distance to
// the nearest structural level, (c) local curvature phase -- visible in one
// glance, so a micro-timeframe entry doesn't get taken (or panicked out of)
// without the macro context that was missing in the user's Figure_1-4
// trade postmortem (docs/daily/2026-07-07.md / memory
// project-moises-trade-postmortem).
//
// Cubic slope/curvature are NOT recomputed here -- they are pulled directly
// from 2-CubicRegressionEndpoint_v1.0-RC (already tested, already on your
// charts) via NinjaScript's standard cross-indicator composition. The fast/
// slow band mean+sigma ARE recomputed locally (self-contained OLS, same
// formula family as 1a/1b's OlsMeanSe) rather than cross-referenced, so this
// indicator compiles and runs standalone even if 1a/1b are not on the chart.
//
// VERSION: 1.0-RC
// -----------------------------------------------------------------------------
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class _6_StructureContext_v10 : Indicator
    {
        private const string VERSION = "1.0-RC";
        private const int BIP_FAST = 1;
        private const int BIP_SLOW = 2;

        // Self-contained rolling OLS state (mirrors 1a's OlsMeanSe formula)
        private double curFastMean, curFastSigma;
        private double curSlowMean, curSlowSigma;
        private bool haveFast, haveSlow;

        // Session-anchored macro cluster (running extremes of the band edges
        // seen so far this session -- the "look back at the whole session"
        // the user described; grows monotonically until the next session).
        private double clusterMax = double.NaN;
        private double clusterMin = double.NaN;

        // Cross-referenced cubic indicator (their tested implementation --
        // NOT recomputed here). See file header for why.
        private _2_CubicRegressionEndpoint_v10 cubic;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Structure-context dashboard: macro position-in-range, " +
                    "sigma-relative distance to the nearest tracked level, and curvature " +
                    "phase (from 2-CubicRegressionEndpoint), combined into one glanceable " +
                    "read. NOT a signal -- every component is a measured, weak (1-5pp) " +
                    "statistical tendency (research/level_hold/, 2026-07-07). Built to keep " +
                    "macro context visible while reading micro price action (the miss in " +
                    "Figure_3 of the 2026-07-07 trade postmortem).";
                Name = "6-StructureContext_v1.0-RC";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;              // own panel: MacroPosPct is a 0-100 oscillator
                DisplayInDataBox = true;
                DrawOnPricePanel = false;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = false;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                // Fast band: ~10 min structure (default 5s bars x 120 = 10 min)
                FastTimeFrameType = BarsPeriodType.Second;
                FastTimeFrameValue = 5;
                FastPeriod = 120;

                // Slow band: ~60 min structure (default 1m bars x 60 = 60 min)
                SlowTimeFrameType = BarsPeriodType.Minute;
                SlowTimeFrameValue = 1;
                SlowPeriod = 60;

                // Cubic: matches 2-CubicRegressionEndpoint's own documented
                // default (1s bars x 450 = 7.5 min) so the phase read lines
                // up with what's already validated/plotted on your charts.
                CubicTimeFrameType = BarsPeriodType.Second;
                CubicTimeFrameValue = 1;
                CubicPeriod = 450;

                ZoneSigmaMultiplier = 1.0;   // "how near" = this many FAST sigmas

                AddPlot(new Stroke(Brushes.DodgerBlue, 2), PlotStyle.Line, "MacroPosPct");
                AddPlot(new Stroke(Brushes.Orange, 1),     PlotStyle.Line, "NearestLevelSigmaDist");
                AddPlot(new Stroke(Brushes.Cyan, 1),       PlotStyle.Line, "CurvaturePhaseCode");

                AddLine(new Stroke(Brushes.Gray, DashStyleHelper.Dash, 1), 50, "MidRange");
                AddLine(new Stroke(Brushes.DarkGray, DashStyleHelper.Dot, 1), 85, "NearTop");
                AddLine(new Stroke(Brushes.DarkGray, DashStyleHelper.Dot, 1), 15, "NearBottom");
            }
            else if (State == State.Configure)
            {
                AddDataSeries(FastTimeFrameType, FastTimeFrameValue);   // BIP_FAST = 1
                AddDataSeries(SlowTimeFrameType, SlowTimeFrameValue);   // BIP_SLOW = 2
            }
            else if (State == State.DataLoaded)
            {
                haveFast = false;
                haveSlow = false;
                clusterMax = double.NaN;
                clusterMin = double.NaN;
                cubic = _2_CubicRegressionEndpoint_v10(CubicTimeFrameType, CubicTimeFrameValue, CubicPeriod);
            }
        }

        /// <summary>Trailing-window OLS endpoint (mean) + residual RMS (sigma).
        /// Same formula family as 1a-StatCloseRegressionBands' OlsMeanSe --
        /// kept self-contained here so this indicator has no compile-time
        /// dependency on 1a/1b being present. x-grid is bar index (0..W-1);
        /// only the SHAPE of the fit matters for sigma, not physical time
        /// units, so this is deliberately simpler than the cubic weight
        /// derivation (which needs real time units for slope/curvature).</summary>
        private static bool OlsMeanSigma(ISeries<double> closes, int W, out double mean, out double sigma)
        {
            mean = double.NaN; sigma = double.NaN;
            if (W < 3) return false;

            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (int i = 0; i < W; i++)
            {
                double x = i;                 // i=0 is the OLDEST bar in the window
                double y = closes[W - 1 - i];
                sx += x; sy += y; sxx += x * x; sxy += x * y;
            }
            double n = W;
            double denom = n * sxx - sx * sx;
            if (Math.Abs(denom) < 1e-9) return false;
            double b = (n * sxy - sx * sy) / denom;
            double a = (sy - b * sx) / n;

            double sse = 0;
            for (int i = 0; i < W; i++)
            {
                double x = i;
                double y = closes[W - 1 - i];
                double resid = y - (a + b * x);
                sse += resid * resid;
            }
            sigma = Math.Sqrt(sse / n);
            mean = a + b * (n - 1);   // endpoint, x = W-1 (the newest bar)
            return true;
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress == BIP_FAST)
            {
                if (CurrentBars[BIP_FAST] >= FastPeriod - 1)
                {
                    double m, s;
                    if (OlsMeanSigma(Closes[BIP_FAST], FastPeriod, out m, out s))
                    {
                        curFastMean = m; curFastSigma = s; haveFast = true;
                    }
                }
                return;
            }

            if (BarsInProgress == BIP_SLOW)
            {
                if (CurrentBars[BIP_SLOW] >= SlowPeriod - 1)
                {
                    double m, s;
                    if (OlsMeanSigma(Closes[BIP_SLOW], SlowPeriod, out m, out s))
                    {
                        curSlowMean = m; curSlowSigma = s; haveSlow = true;
                    }
                }
                return;
            }

            // ── Primary series: combine + render ──
            if (BarsInProgress != 0) return;
            if (!haveFast || !haveSlow) return;

            // Reset the session-anchored macro cluster at the start of each
            // new session (mirrors pivot_level_proximity.py's per-day reset;
            // here it's continuous/expanding through the LIVE session rather
            // than frozen at a single formation cutoff, since a discretionary
            // trader is reading "how far have we ranged so far today", live).
            if (Bars.IsFirstBarOfSession)
            {
                clusterMax = double.NaN;
                clusterMin = double.NaN;
            }

            double fastUp = curFastMean + ZoneSigmaMultiplier * curFastSigma;
            double fastLo = curFastMean - ZoneSigmaMultiplier * curFastSigma;
            double slowUp = curSlowMean + ZoneSigmaMultiplier * curSlowSigma;
            double slowLo = curSlowMean - ZoneSigmaMultiplier * curSlowSigma;

            double dayHi = Math.Max(fastUp, slowUp);
            double dayLo = Math.Min(fastLo, slowLo);
            clusterMax = double.IsNaN(clusterMax) ? dayHi : Math.Max(clusterMax, dayHi);
            clusterMin = double.IsNaN(clusterMin) ? dayLo : Math.Min(clusterMin, dayLo);

            double close = Close[0];
            double range = clusterMax - clusterMin;
            double macroPosPct = range > 1e-9
                ? Math.Max(0.0, Math.Min(100.0, (close - clusterMin) / range * 100.0))
                : 50.0;

            // Distance to the nearer cluster edge, in FAST-sigma units --
            // the corrected, volatility-relative "how near" (2026-07-07).
            double distToMax = Math.Abs(clusterMax - close);
            double distToMin = Math.Abs(close - clusterMin);
            bool nearMax = distToMax <= distToMin;
            double nearestDist = nearMax ? distToMax : distToMin;
            double sigmaDist = curFastSigma > 1e-9 ? nearestDist / curFastSigma : double.NaN;
            double tickDist = nearestDist / TickSize;

            // Curvature phase, straight from the tested cubic indicator --
            // raw slope/curvature = plotted value minus the endpoint offset
            // (see 2-CubicRegressionEndpoint's OnBarUpdate: CubicSlope[0] =
            // CubicValue[0] + rawSlope, CubicCurvature[0] = CubicValue[0] + rawCurv).
            // Guard: the sub-indicator's Series<double> defaults to 0.0 until
            // it has CubicPeriod bars on its own independent timeframe --
            // reading it earlier would show a false confident phase read
            // from all-zero values. A real MNQ close is never <= 0, so this
            // is a safe, simple readiness check without depending on the
            // sub-indicator's private internal bar count.
            double phaseCode;
            string phaseLabel;
            if (cubic.CubicValue[0] <= 0.0)
            {
                phaseCode = 0; phaseLabel = "warming up";
            }
            else
            {
                double rawSlope = cubic.CubicSlope[0] - cubic.CubicValue[0];
                double rawCurv = cubic.CubicCurvature[0] - cubic.CubicValue[0];
                // Code: +2 accelerating up, +1 decelerating up, 0 flat,
                //       -1 decelerating down, -2 accelerating down
                if (Math.Abs(rawSlope) < 1e-9)
                {
                    phaseCode = 0; phaseLabel = "flat";
                }
                else if (rawSlope > 0)
                {
                    phaseCode = rawCurv >= 0 ? 2 : 1;
                    phaseLabel = rawCurv >= 0 ? "UP-accelerating" : "UP-decelerating";
                }
                else
                {
                    phaseCode = rawCurv <= 0 ? -2 : -1;
                    phaseLabel = rawCurv <= 0 ? "DOWN-accelerating" : "DOWN-decelerating";
                }
            }

            MacroPosPct[0] = macroPosPct;
            NearestLevelSigmaDist[0] = sigmaDist;
            CurvaturePhaseCode[0] = phaseCode;

            string edge = nearMax ? "CLUSTER-TOP" : "CLUSTER-BOTTOM";
            string summary = string.Format(
                "Macro: {0:F0}% of session range | Nearest: {1} @ {2:F1}σ / {3:F1} ticks | Curve: {4}\n" +
                "(weak, measured tendencies -- not a signal; see docs/daily/2026-07-07.md)",
                macroPosPct, edge, sigmaDist, tickDist, phaseLabel);
            Draw.TextFixed(this, "StructureContextSummary", summary, TextPosition.BottomLeft);
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Fast Timeframe Unit", Order = 0, GroupName = "Fast Band")]
        public BarsPeriodType FastTimeFrameType { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Fast Timeframe Value", Order = 1, GroupName = "Fast Band")]
        public int FastTimeFrameValue { get; set; }

        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Fast Rolling Window (Bars)", Order = 2, GroupName = "Fast Band")]
        public int FastPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Slow Timeframe Unit", Order = 3, GroupName = "Slow Band")]
        public BarsPeriodType SlowTimeFrameType { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Slow Timeframe Value", Order = 4, GroupName = "Slow Band")]
        public int SlowTimeFrameValue { get; set; }

        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Slow Rolling Window (Bars)", Order = 5, GroupName = "Slow Band")]
        public int SlowPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Cubic Timeframe Unit", Order = 6, GroupName = "Cubic (cross-ref)")]
        public BarsPeriodType CubicTimeFrameType { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Cubic Timeframe Value", Order = 7, GroupName = "Cubic (cross-ref)")]
        public int CubicTimeFrameValue { get; set; }

        [Range(4, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Cubic Rolling Window (Bars)", Order = 8, GroupName = "Cubic (cross-ref)")]
        public int CubicPeriod { get; set; }

        [Range(0.1, 10.0), NinjaScriptProperty]
        [Display(Name = "Zone Width (x Fast Sigma)", Description =
            "'How near' is measured in units of the FAST band's own current " +
            "sigma, never a fixed tick count -- corrected 2026-07-07 after " +
            "the fixed-tick version was shown to give a misleading answer.",
            Order = 9, GroupName = "Parameters")]
        public double ZoneSigmaMultiplier { get; set; }

        [Browsable(false)] [XmlIgnore] public Series<double> MacroPosPct           => Values[0];
        [Browsable(false)] [XmlIgnore] public Series<double> NearestLevelSigmaDist => Values[1];
        [Browsable(false)] [XmlIgnore] public Series<double> CurvaturePhaseCode    => Values[2];
        #endregion
    }
}
