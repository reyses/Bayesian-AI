//
// 1a-StatCloseRegressionBands_v1.0-RC.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: Independent Close Stat Regression Bands
//
// Close-based z-bands at +/-1, 2, 3, 4 standard errors (SE) on an
// INDEPENDENT Close timeframe.
// Supports toggling individual sigma bands (1, 2, 3, 4) on/off.
//
// Performance Optimized: Bypasses standard error residual calculations entirely
// if no sigma bands are enabled, saving significant CPU/memory resources.
//
// SE = sqrt(SSE / (n-2)) matches Python SFE z_se.
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
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class _1a_StatCloseRegressionBands_v10 : Indicator
    {
        private const string VERSION = "1.0-RC";
        private const int BIP_CLOSE = 1;

        // Cached values
        private double curMClose, curSeClose;
        private bool haveCloseData;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Close Stat Regression Bands on an independent Close timeframe. Supports toggling individual sigmas.";
                Name        = "1a-StatCloseRegressionBands_v1.0-RC";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = false;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                // Timeframe & window defaults
                CloseTimeFrameType = BarsPeriodType.Minute;
                CloseTimeFrameValue = 1;
                Period = 3600;

                // Sigma toggles
                ShowSigma1 = true;
                ShowSigma2 = true;
                ShowSigma3 = true;
                ShowSigma4 = true;

                // Plots (0-8)
                AddPlot(new Stroke(Brushes.DodgerBlue, 2),                       PlotStyle.Line, "MeanClose");
                AddPlot(new Stroke(Brushes.LightGray, DashStyleHelper.Dash, 1),  PlotStyle.Line, "Upper1Close");
                AddPlot(new Stroke(Brushes.LightGray, DashStyleHelper.Dash, 1),  PlotStyle.Line, "Lower1Close");
                AddPlot(new Stroke(Brushes.Orange,                          1),  PlotStyle.Line, "Upper2Close");
                AddPlot(new Stroke(Brushes.Orange,                          1),  PlotStyle.Line, "Lower2Close");
                AddPlot(new Stroke(Brushes.Red,                             1),  PlotStyle.Line, "Upper3Close");
                AddPlot(new Stroke(Brushes.Red,                             1),  PlotStyle.Line, "Lower3Close");
                AddPlot(new Stroke(Brushes.DarkRed,                         1),  PlotStyle.Line, "Upper4Close");
                AddPlot(new Stroke(Brushes.DarkRed,                         1),  PlotStyle.Line, "Lower4Close");
            }
            else if (State == State.Configure)
            {
                AddDataSeries(CloseTimeFrameType, CloseTimeFrameValue);
                curMClose = double.NaN;
                curSeClose = double.NaN;
                haveCloseData = false;
            }
        }

        protected override void OnBarUpdate()
        {
            // ── Close Series updates (BIP_CLOSE) ──
            if (BarsInProgress == BIP_CLOSE)
            {
                if (CurrentBars[BIP_CLOSE] >= Period - 1 && Period >= 3)
                {
                    double mc, sec;
                    bool needSe = ShowSigma1 || ShowSigma2 || ShowSigma3 || ShowSigma4;
                    if (OlsMeanSe(Closes[BIP_CLOSE], Period, needSe, out mc, out sec))
                    {
                        curMClose = mc;
                        curSeClose = sec;
                        haveCloseData = true;
                    }
                }
                return;
            }

            // ── Primary Series updates (Rendering) ──
            if (BarsInProgress != 0) return;

            if (haveCloseData)
            {
                MeanClose[0] = curMClose;

                // Sigma 1
                if (ShowSigma1)
                {
                    Upper1Close[0] = curMClose + 1.0 * curSeClose;
                    Lower1Close[0] = curMClose - 1.0 * curSeClose;
                }
                else
                {
                    Upper1Close.Reset(); Lower1Close.Reset();
                }

                // Sigma 2
                if (ShowSigma2)
                {
                    Upper2Close[0] = curMClose + 2.0 * curSeClose;
                    Lower2Close[0] = curMClose - 2.0 * curSeClose;
                }
                else
                {
                    Upper2Close.Reset(); Lower2Close.Reset();
                }

                // Sigma 3
                if (ShowSigma3)
                {
                    Upper3Close[0] = curMClose + 3.0 * curSeClose;
                    Lower3Close[0] = curMClose - 3.0 * curSeClose;
                }
                else
                {
                    Upper3Close.Reset(); Lower3Close.Reset();
                }

                // Sigma 4
                if (ShowSigma4)
                {
                    Upper4Close[0] = curMClose + 4.0 * curSeClose;
                    Lower4Close[0] = curMClose - 4.0 * curSeClose;
                }
                else
                {
                    Upper4Close.Reset(); Lower4Close.Reset();
                }
            }
        }

        private bool OlsMeanSe(ISeries<double> series, int period, bool calculateSe, out double mean, out double se)
        {
            mean = 0.0; se = 0.0;
            double sumX  = period * (period - 1) * 0.5;
            double sumX2 = (period - 1) * period * (2.0 * period - 1) / 6.0;
            double divisor = period * sumX2 - sumX * sumX;
            if (Math.Abs(divisor) < 1e-12) return false;

            double sumXy = 0.0, sumY = 0.0;
            for (int count = 0; count < period; count++)
            {
                double y = series[period - 1 - count];
                sumXy += count * y;
                sumY  += y;
            }
            double slope     = (period * sumXy - sumX * sumY) / divisor;
            double intercept = (sumY - slope * sumX) / period;
            mean = intercept + slope * (period - 1);

            if (calculateSe)
            {
                double sse = 0.0;
                for (int count = 0; count < period; count++)
                {
                    double yHat = intercept + slope * count;
                    double y    = series[period - 1 - count];
                    double res  = y - yHat;
                    sse += res * res;
                }
                se = Math.Sqrt(sse / (period - 2));
            }
            return true;
        }

        private double CalculateMinutes(int period, BarsPeriodType type, int value)
        {
            double multiplier = 0;
            switch (type)
            {
                case BarsPeriodType.Second: multiplier = value / 60.0; break;
                case BarsPeriodType.Minute: multiplier = value; break;
                case BarsPeriodType.Day: multiplier = value * 1440.0; break;
                case BarsPeriodType.Week: multiplier = value * 10080.0; break;
            }
            return period * multiplier;
        }

        #region Properties

        // Close series parameters
        [NinjaScriptProperty]
        [Display(Name = "Timeframe Unit", Description = "Timeframe type for Close series", Order = 0, GroupName = "Parameters")]
        public BarsPeriodType CloseTimeFrameType { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Timeframe Value", Description = "Timeframe period value for Close series", Order = 1, GroupName = "Parameters")]
        public int CloseTimeFrameValue { get; set; }

        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Rolling Window (Bars)", Description = "Lookback bars for Close OLS regression (>=3)", Order = 2, GroupName = "Parameters")]
        public int Period { get; set; }

        // Sigma toggles
        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 1", Description = "Toggle +/- 1 SE bands", Order = 3, GroupName = "Parameters")]
        public bool ShowSigma1 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 2", Description = "Toggle +/- 2 SE bands", Order = 4, GroupName = "Parameters")]
        public bool ShowSigma2 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 3", Description = "Toggle +/- 3 SE bands", Order = 5, GroupName = "Parameters")]
        public bool ShowSigma3 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 4", Description = "Toggle +/- 4 SE bands", Order = 6, GroupName = "Parameters")]
        public bool ShowSigma4 { get; set; }

        // Calculators
        [Browsable(true)]
        [Display(Name = "Close Lookback (Min)", Description = "Calculated lookback in minutes for Close bands.", GroupName = "Calculators", Order = 0)]
        public double CalculatedPrimaryLookbackMinutes => CalculateMinutes(Period, CloseTimeFrameType, CloseTimeFrameValue);

        // Plots
        [Browsable(false)] [XmlIgnore] public Series<double> MeanClose   => Values[0];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper1Close => Values[1];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower1Close => Values[2];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper2Close => Values[3];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower2Close => Values[4];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper3Close => Values[5];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower3Close => Values[6];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper4Close => Values[7];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower4Close => Values[8];

        #endregion
    }
}
