//
// StatRegressionBands.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: linear regression mean + z-bands at +/-1, +/-2, +/-3,
// +/-4 standard errors (SE).
//
// Mirrors the visual reference in
// charts/chart_reg_z_2025_06_09.png (price + regression mean + +/-2 SE bands).
//
// IMPORTANT: this indicator computes SE = sqrt(SSE / (n-2)) — the proper
// standard error of regression — to match the Python SFE z_se formulation
// used everywhere else in the Bayesian-AI project. NT8's built-in
// RegressionChannel uses stddev-of-|residuals| instead, which gives subtly
// different bands. If you want bands that match Python z_se, use this
// indicator, NOT RegressionChannel.
//
// Plots:
//   0  Mean          (DodgerBlue, thick)
//   1  Upper 1 SE    (LightGray dashed)
//   2  Lower 1 SE    (LightGray dashed)
//   3  Upper 2 SE    (Orange)              <- the "fade trigger" zone
//   4  Lower 2 SE    (Orange)
//   5  Upper 3 SE    (Red)
//   6  Lower 3 SE    (Red)
//   7  Upper 4 SE    (DarkRed)             <- extreme, rare
//   8  Lower 4 SE    (DarkRed)
//
// Parameters:
//   Period  -- lookback bars for OLS regression (default 20)
//
// -----------------------------------------------------------------------------
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    /// <summary>
    /// Linear regression mean + z-bands at +/-1, +/-2, +/-3, +/-4 SE
    /// (proper RMS of residuals).
    /// </summary>
    public class StatRegressionBands : Indicator
    {
        // Cached series (per-bar) so we can read back history
        private Series<double> meanSeries;
        private Series<double> seSeries;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Linear regression mean + z-bands at +/-1, +/-2, +/-3, +/-4 SE. " +
                              "SE = sqrt(SSE/(n-2)). Matches Python SFE z_se. Adjustable period.";
                Name        = "StatRegressionBands";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                Period = 20;

                AddPlot(new Stroke(Brushes.DodgerBlue, 2),                          PlotStyle.Line, "Mean");
                AddPlot(new Stroke(Brushes.LightGray,  DashStyleHelper.Dash, 1),    PlotStyle.Line, "Upper1");
                AddPlot(new Stroke(Brushes.LightGray,  DashStyleHelper.Dash, 1),    PlotStyle.Line, "Lower1");
                AddPlot(new Stroke(Brushes.Orange,                          1),     PlotStyle.Line, "Upper2");
                AddPlot(new Stroke(Brushes.Orange,                          1),     PlotStyle.Line, "Lower2");
                AddPlot(new Stroke(Brushes.Red,                             1),     PlotStyle.Line, "Upper3");
                AddPlot(new Stroke(Brushes.Red,                             1),     PlotStyle.Line, "Lower3");
                AddPlot(new Stroke(Brushes.DarkRed,                         1),     PlotStyle.Line, "Upper4");
                AddPlot(new Stroke(Brushes.DarkRed,                         1),     PlotStyle.Line, "Lower4");
            }
            else if (State == State.DataLoaded)
            {
                meanSeries = new Series<double>(this);
                seSeries   = new Series<double>(this);
            }
        }

        protected override void OnBarUpdate()
        {
            // Need at least Period bars (and >= 3 for SE with n-2 divisor)
            if (CurrentBar < Period - 1 || Period < 3)
                return;

            // ----- OLS slope/intercept over the last Period bars -----
            // Map x = 0 .. Period-1 chronologically, so x = Period-1 is current bar.
            // Input[i] in NT8 indexes BACKWARDS: Input[0] = current, Input[Period-1] = oldest.
            // So bar at x = count corresponds to Input[Period - 1 - count].
            double sumX  = Period * (Period - 1) * 0.5;
            double sumX2 = (Period - 1) * Period * (2.0 * Period - 1) / 6.0;  // sum_{x=0}^{n-1} x^2
            double divisor = Period * sumX2 - sumX * sumX;

            if (Math.Abs(divisor) < 1e-12)
                return;

            double sumXy = 0.0;
            double sumY  = 0.0;
            for (int count = 0; count < Period; count++)
            {
                double y = Input[Period - 1 - count];
                sumXy += count * y;
                sumY  += y;
            }

            double slope     = (Period * sumXy - sumX * sumY) / divisor;
            double intercept = (sumY - slope * sumX) / Period;

            // Mean = regression value at the current bar (x = Period-1)
            double mean = intercept + slope * (Period - 1);

            // ----- SE of regression: sqrt(SSE / (n - 2)) -----
            double sse = 0.0;
            for (int count = 0; count < Period; count++)
            {
                double yHat = intercept + slope * count;
                double y    = Input[Period - 1 - count];
                double res  = y - yHat;
                sse += res * res;
            }
            double se = Math.Sqrt(sse / (Period - 2));

            meanSeries[0] = mean;
            seSeries[0]   = se;

            Mean[0]   = mean;
            Upper1[0] = mean + 1.0 * se;
            Lower1[0] = mean - 1.0 * se;
            Upper2[0] = mean + 2.0 * se;
            Lower2[0] = mean - 2.0 * se;
            Upper3[0] = mean + 3.0 * se;
            Lower3[0] = mean - 3.0 * se;
            Upper4[0] = mean + 4.0 * se;
            Lower4[0] = mean - 4.0 * se;
        }

        #region Properties

        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Period", Description = "Lookback bars for OLS regression (>=3)", Order = 0, GroupName = "Parameters")]
        public int Period { get; set; }

        [Browsable(false)] [XmlIgnore] public Series<double> Mean   => Values[0];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper1 => Values[1];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower1 => Values[2];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper2 => Values[3];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower2 => Values[4];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper3 => Values[5];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower3 => Values[6];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper4 => Values[7];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower4 => Values[8];

        /// <summary>Helper accessor for SE from other scripts.</summary>
        [Browsable(false)] [XmlIgnore] public Series<double> SE => seSeries;

        #endregion
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
    public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
    {
        private StatRegressionBands[] cacheStatRegressionBands;
        public StatRegressionBands StatRegressionBands(int period)
        {
            return StatRegressionBands(Input, period);
        }

        public StatRegressionBands StatRegressionBands(ISeries<double> input, int period)
        {
            if (cacheStatRegressionBands != null)
                for (int idx = 0; idx < cacheStatRegressionBands.Length; idx++)
                    if (cacheStatRegressionBands[idx] != null && cacheStatRegressionBands[idx].Period == period && cacheStatRegressionBands[idx].EqualsInput(input))
                        return cacheStatRegressionBands[idx];
            return CacheIndicator<StatRegressionBands>(new StatRegressionBands() { Period = period }, input, ref cacheStatRegressionBands);
        }
    }
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
    public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
    {
        public Indicators.StatRegressionBands StatRegressionBands(int period)
        {
            return indicator.StatRegressionBands(Input, period);
        }

        public Indicators.StatRegressionBands StatRegressionBands(ISeries<double> input, int period)
        {
            return indicator.StatRegressionBands(input, period);
        }
    }
}

namespace NinjaTrader.NinjaScript.Strategies
{
    public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
    {
        public Indicators.StatRegressionBands StatRegressionBands(int period)
        {
            return indicator.StatRegressionBands(Input, period);
        }

        public Indicators.StatRegressionBands StatRegressionBands(ISeries<double> input, int period)
        {
            return indicator.StatRegressionBands(input, period);
        }
    }
}

#endregion
