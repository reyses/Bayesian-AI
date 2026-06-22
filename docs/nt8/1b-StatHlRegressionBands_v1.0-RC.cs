//
// 1b-StatHlRegressionBands_v1.0-RC.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: Independent High/Low Stat Regression Bands
//
// High/Low-based z-bands at +1,2,3,4 (High) and -1,2,3,4 (Low) SE on an
// INDEPENDENT HL timeframe.
// Supports toggling:
//   - High/Low Far Side & Near Side independently
//   - Individual sigma bands (1, 2, 3, 4) independently
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
    public class _1b_StatHlRegressionBands_v10 : Indicator
    {
        private const string VERSION = "1.0-RC";
        private const int BIP_HL = 1;

        // Cached values
        private double curMHigh, curSeHigh, curMLow, curSeLow;
        private bool haveHlData;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "High/Low Stat Regression Bands on an independent HL timeframe. Supports toggling Near/Far sides and individual sigmas.";
                Name        = "1b-StatHlRegressionBands_v1.0-RC";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = false;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                // Timeframe & window defaults
                HlTimeFrameType = BarsPeriodType.Minute;
                HlTimeFrameValue = 8;
                HlPeriod = 3600;

                // Visibility toggles
                ShowHighFarSide  = true;
                ShowHighNearSide = false;
                ShowLowFarSide   = true;
                ShowLowNearSide  = false;

                // Sigma toggles
                ShowSigma1 = true;
                ShowSigma2 = true;
                ShowSigma3 = true;
                ShowSigma4 = true;

                // Plots (0-17)
                // High Far (0-4)
                AddPlot(new Stroke(Brushes.LimeGreen, 1.5f),                     PlotStyle.Line, "MeanHigh");
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dash, 1),  PlotStyle.Line, "Upper1Hl");
                AddPlot(new Stroke(Brushes.LimeGreen,                       1),  PlotStyle.Line, "Upper2Hl");
                AddPlot(new Stroke(Brushes.LimeGreen,                       1),  PlotStyle.Line, "Upper3Hl");
                AddPlot(new Stroke(Brushes.LimeGreen,                       1),  PlotStyle.Line, "Upper4Hl");

                // Low Far (5-9)
                AddPlot(new Stroke(Brushes.OrangeRed, 1.5f),                     PlotStyle.Line, "MeanLow");
                AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dash, 1),  PlotStyle.Line, "Lower1Hl");
                AddPlot(new Stroke(Brushes.OrangeRed,                       1),  PlotStyle.Line, "Lower2Hl");
                AddPlot(new Stroke(Brushes.OrangeRed,                       1),  PlotStyle.Line, "Lower3Hl");
                AddPlot(new Stroke(Brushes.OrangeRed,                       1),  PlotStyle.Line, "Lower4Hl");

                // High Near (10-13)
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Lower1HlNear");
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Lower2HlNear");
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Lower3HlNear");
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Lower4HlNear");

                // Low Near (14-17)
                AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Upper1HlNear");
                AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Upper2HlNear");
                AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Upper3HlNear");
                AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dot, 1),   PlotStyle.Line, "Upper4HlNear");
            }
            else if (State == State.Configure)
            {
                AddDataSeries(HlTimeFrameType, HlTimeFrameValue);
                curMHigh  = double.NaN;
                curSeHigh = double.NaN;
                curMLow   = double.NaN;
                curSeLow  = double.NaN;
                haveHlData = false;
            }
        }

        protected override void OnBarUpdate()
        {
            // ── Secondary Series updates (BIP_HL) ──
            if (BarsInProgress == BIP_HL)
            {
                if (CurrentBars[BIP_HL] >= HlPeriod - 1 && HlPeriod >= 3)
                {
                    double mh, seh, ml, sel;
                    bool anySigma = ShowSigma1 || ShowSigma2 || ShowSigma3 || ShowSigma4;
                    bool needSeHigh = (ShowHighFarSide || ShowHighNearSide) && anySigma;
                    bool needSeLow  = (ShowLowFarSide || ShowLowNearSide) && anySigma;

                    if (OlsMeanSe(Highs[BIP_HL], HlPeriod, needSeHigh, out mh, out seh) &&
                        OlsMeanSe(Lows[BIP_HL],  HlPeriod, needSeLow,  out ml, out sel))
                    {
                        curMHigh   = mh;
                        curSeHigh  = seh;
                        curMLow    = ml;
                        curSeLow   = sel;
                        haveHlData = true;
                    }
                }
                return;
            }

            // ── Primary Series updates (Rendering) ──
            if (BarsInProgress != 0) return;

            if (haveHlData)
            {
                MeanHigh[0] = curMHigh;
                MeanLow[0]  = curMLow;

                // High Far Side
                if (ShowHighFarSide)
                {
                    if (ShowSigma1) Upper1Hl[0] = curMHigh + 1.0 * curSeHigh; else Upper1Hl.Reset();
                    if (ShowSigma2) Upper2Hl[0] = curMHigh + 2.0 * curSeHigh; else Upper2Hl.Reset();
                    if (ShowSigma3) Upper3Hl[0] = curMHigh + 3.0 * curSeHigh; else Upper3Hl.Reset();
                    if (ShowSigma4) Upper4Hl[0] = curMHigh + 4.0 * curSeHigh; else Upper4Hl.Reset();
                }
                else
                {
                    Upper1Hl.Reset(); Upper2Hl.Reset(); Upper3Hl.Reset(); Upper4Hl.Reset();
                }

                // High Near Side
                if (ShowHighNearSide)
                {
                    if (ShowSigma1) Lower1HlNear[0] = curMHigh - 1.0 * curSeHigh; else Lower1HlNear.Reset();
                    if (ShowSigma2) Lower2HlNear[0] = curMHigh - 2.0 * curSeHigh; else Lower2HlNear.Reset();
                    if (ShowSigma3) Lower3HlNear[0] = curMHigh - 3.0 * curSeHigh; else Lower3HlNear.Reset();
                    if (ShowSigma4) Lower4HlNear[0] = curMHigh - 4.0 * curSeHigh; else Lower4HlNear.Reset();
                }
                else
                {
                    Lower1HlNear.Reset(); Lower2HlNear.Reset(); Lower3HlNear.Reset(); Lower4HlNear.Reset();
                }

                // Low Far Side
                if (ShowLowFarSide)
                {
                    if (ShowSigma1) Lower1Hl[0] = curMLow - 1.0 * curSeLow; else Lower1Hl.Reset();
                    if (ShowSigma2) Lower2Hl[0] = curMLow - 2.0 * curSeLow; else Lower2Hl.Reset();
                    if (ShowSigma3) Lower3Hl[0] = curMLow - 3.0 * curSeLow; else Lower3Hl.Reset();
                    if (ShowSigma4) Lower4Hl[0] = curMLow - 4.0 * curSeLow; else Lower4Hl.Reset();
                }
                else
                {
                    Lower1Hl.Reset(); Lower2Hl.Reset(); Lower3Hl.Reset(); Lower4Hl.Reset();
                }

                // Low Near Side
                if (ShowLowNearSide)
                {
                    if (ShowSigma1) Upper1HlNear[0] = curMLow + 1.0 * curSeLow; else Upper1HlNear.Reset();
                    if (ShowSigma2) Upper2HlNear[0] = curMLow + 2.0 * curSeLow; else Upper2HlNear.Reset();
                    if (ShowSigma3) Upper3HlNear[0] = curMLow + 3.0 * curSeLow; else Upper3HlNear.Reset();
                    if (ShowSigma4) Upper4HlNear[0] = curMLow + 4.0 * curSeLow; else Upper4HlNear.Reset();
                }
                else
                {
                    Upper1HlNear.Reset(); Upper2HlNear.Reset(); Upper3HlNear.Reset(); Upper4HlNear.Reset();
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

        // High/Low series parameters
        [NinjaScriptProperty]
        [Display(Name = "Timeframe Unit", Description = "Timeframe type for High/Low series", Order = 0, GroupName = "Parameters")]
        public BarsPeriodType HlTimeFrameType { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Timeframe Value", Description = "Timeframe period value for High/Low series", Order = 1, GroupName = "Parameters")]
        public int HlTimeFrameValue { get; set; }

        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Rolling Window (Bars)", Description = "Lookback bars for HL OLS regression (>=3)", Order = 2, GroupName = "Parameters")]
        public int HlPeriod { get; set; }

        // Visibility Toggles (Checkboxes)
        [NinjaScriptProperty]
        [Display(Name = "Show High Far Side", Description = "Show upper bands of the High regression channel (resistance)", Order = 3, GroupName = "Parameters")]
        public bool ShowHighFarSide { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show High Near Side", Description = "Show lower bands of the High regression channel (inner bands)", Order = 4, GroupName = "Parameters")]
        public bool ShowHighNearSide { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Low Far Side", Description = "Show lower bands of the Low regression channel (support)", Order = 5, GroupName = "Parameters")]
        public bool ShowLowFarSide { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Low Near Side", Description = "Show upper bands of the Low regression channel (inner bands)", Order = 6, GroupName = "Parameters")]
        public bool ShowLowNearSide { get; set; }

        // Sigma toggles
        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 1", Description = "Toggle +/- 1 SE bands", Order = 7, GroupName = "Parameters")]
        public bool ShowSigma1 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 2", Description = "Toggle +/- 2 SE bands", Order = 8, GroupName = "Parameters")]
        public bool ShowSigma2 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 3", Description = "Toggle +/- 3 SE bands", Order = 9, GroupName = "Parameters")]
        public bool ShowSigma3 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Sigma 4", Description = "Toggle +/- 4 SE bands", Order = 10, GroupName = "Parameters")]
        public bool ShowSigma4 { get; set; }

        // Calculators
        [Browsable(true)]
        [Display(Name = "HL Lookback (Min)", Description = "Calculated lookback in minutes for HL bands.", GroupName = "Calculators", Order = 0)]
        public double CalculatedHlLookbackMinutes => CalculateMinutes(HlPeriod, HlTimeFrameType, HlTimeFrameValue);

        // High plots
        [Browsable(false)] [XmlIgnore] public Series<double> MeanHigh    => Values[0];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper1Hl    => Values[1];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper2Hl    => Values[2];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper3Hl    => Values[3];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper4Hl    => Values[4];

        // Low plots
        [Browsable(false)] [XmlIgnore] public Series<double> MeanLow     => Values[5];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower1Hl    => Values[6];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower2Hl    => Values[7];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower3Hl    => Values[8];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower4Hl    => Values[9];

        // High Near plots
        [Browsable(false)] [XmlIgnore] public Series<double> Lower1HlNear => Values[10];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower2HlNear => Values[11];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower3HlNear => Values[12];
        [Browsable(false)] [XmlIgnore] public Series<double> Lower4HlNear => Values[13];

        // Low Near plots
        [Browsable(false)] [XmlIgnore] public Series<double> Upper1HlNear => Values[14];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper2HlNear => Values[15];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper3HlNear => Values[16];
        [Browsable(false)] [XmlIgnore] public Series<double> Upper4HlNear => Values[17];

        #endregion
    }
}
