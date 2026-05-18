//
// HLCRegressionMeans.cs   v2.0  (multi-timeframe)
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: 3-body regression envelope on FIXED timeframes.
//   M_high  = OLS regression mean on 1h HIGH series
//   M_low   = OLS regression mean on 1h LOW series
//   M_close = OLS regression mean on 15m CLOSE series
//
// Why fixed TFs:
//   M_high / M_low (1h)  — slow, stable boundary "anchors"; the 3-body envelope
//                          from MEMORY 2026-05-09 uses 1h for HL because of the
//                          ~5% bar rate at ±2σ_high/low and the rare extreme
//                          ±3σ entries (1.5-1.6% of bars).
//   M_close (15m)        — fast, tactical mean. Used as the reversion target
//                          in cusp/CRM strategies (z_15m_crm signal).
//
// Output is plotted on the CURRENT chart timeframe (whatever you opened the
// chart in — 5s, 1m, 5m, etc.). The 1h and 15m values are held flat across
// every bar in their period (so you see the regression mean as a step
// function locked to the parent TF).
//
// Optional bands on M_close (15m SE): ±1/2/3 σ.
//
// Parameters:
//   HlPeriod    : 20    OLS lookback in 1h BARS  for M_high/M_low (= 20 hours)
//   ClosePeriod : 20    OLS lookback in 15m BARS for M_close      (= 5 hours)
//   ShowHigh    : true
//   ShowLow     : true
//   ShowBands   : true
//
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
    /// <summary>
    /// Multi-timeframe 3-body regression envelope. M_high/M_low computed on
    /// 1h bars, M_close on 15m bars. SE = sqrt(SSE/(n-2)) on the 15m close
    /// regression — matches Python z_se.
    /// </summary>
    public class HLCRegressionMeans : Indicator
    {
        // BarsInProgress indices:
        //   0 = primary chart TF (rendering surface)
        //   1 = 1h series  (for M_high / M_low)
        //   2 = 15m series (for M_close)
        private const int BIP_1H  = 1;
        private const int BIP_15M = 2;

        // Cross-bar state — last computed values are held until next parent close
        private double curMHigh, curMLow, curMClose, curSeClose;
        private bool   haveHl, haveClose;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "MTF 3-body envelope: M_high/M_low on 1h bars, M_close on 15m. " +
                              "OLS over HlPeriod / ClosePeriod bars per parent TF.";
                Name        = "HLCRegressionMeans";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = false;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                HlPeriod    = 20;
                ClosePeriod = 20;
                ShowHigh    = true;
                ShowLow     = true;
                ShowBands   = true;

                MCloseBrush = Brushes.DodgerBlue;
                MHighBrush  = Brushes.LimeGreen;
                MLowBrush   = Brushes.OrangeRed;
                Band1Brush  = Brushes.LightGray;
                Band2Brush  = Brushes.Orange;
                Band3Brush  = Brushes.DarkRed;

                AddPlot(new Stroke(Brushes.DodgerBlue, 2),                       PlotStyle.Line, "M_close_15m");
                AddPlot(new Stroke(Brushes.LimeGreen,  1),                       PlotStyle.Line, "M_high_1h");
                AddPlot(new Stroke(Brushes.OrangeRed,  1),                       PlotStyle.Line, "M_low_1h");
                AddPlot(new Stroke(Brushes.LightGray, DashStyleHelper.Dash, 1),  PlotStyle.Line, "Upper1_15m");
                AddPlot(new Stroke(Brushes.LightGray, DashStyleHelper.Dash, 1),  PlotStyle.Line, "Lower1_15m");
                AddPlot(new Stroke(Brushes.Orange,                          1),  PlotStyle.Line, "Upper2_15m");
                AddPlot(new Stroke(Brushes.Orange,                          1),  PlotStyle.Line, "Lower2_15m");
                AddPlot(new Stroke(Brushes.DarkRed,                         1),  PlotStyle.Line, "Upper3_15m");
                AddPlot(new Stroke(Brushes.DarkRed,                         1),  PlotStyle.Line, "Lower3_15m");
            }
            else if (State == State.Configure)
            {
                // Secondary series — order matters: 1h first → BIP=1, 15m → BIP=2
                AddDataSeries(BarsPeriodType.Minute, 60);   // BarsArray[1]  (1h)
                AddDataSeries(BarsPeriodType.Minute, 15);   // BarsArray[2]  (15m)

                curMHigh   = double.NaN;
                curMLow    = double.NaN;
                curMClose  = double.NaN;
                curSeClose = double.NaN;
                haveHl     = false;
                haveClose  = false;
            }
            else if (State == State.DataLoaded)
            {
                // Apply user-chosen brushes
                Plots[0].Brush = MCloseBrush;
                Plots[1].Brush = MHighBrush;
                Plots[2].Brush = MLowBrush;
                Plots[3].Brush = Band1Brush;
                Plots[4].Brush = Band1Brush;
                Plots[5].Brush = Band2Brush;
                Plots[6].Brush = Band2Brush;
                Plots[7].Brush = Band3Brush;
                Plots[8].Brush = Band3Brush;
            }
        }

        protected override void OnBarUpdate()
        {
            // ── 1h secondary series — recompute M_high & M_low on each 1h close ──
            if (BarsInProgress == BIP_1H)
            {
                if (CurrentBars[BIP_1H] >= HlPeriod - 1 && HlPeriod >= 3)
                {
                    double mh, _seh, ml, _sel;
                    if (OlsMeanSe(Highs[BIP_1H], HlPeriod, out mh, out _seh) &&
                        OlsMeanSe(Lows[BIP_1H],  HlPeriod, out ml, out _sel))
                    {
                        curMHigh = mh;
                        curMLow  = ml;
                        haveHl   = true;
                    }
                }
                return;
            }

            // ── 15m secondary series — recompute M_close & SE on each 15m close ──
            if (BarsInProgress == BIP_15M)
            {
                if (CurrentBars[BIP_15M] >= ClosePeriod - 1 && ClosePeriod >= 3)
                {
                    double mc, sec;
                    if (OlsMeanSe(Closes[BIP_15M], ClosePeriod, out mc, out sec))
                    {
                        curMClose  = mc;
                        curSeClose = sec;
                        haveClose  = true;
                    }
                }
                return;
            }

            // ── Primary series — render the held values ────────────────────
            if (BarsInProgress != 0) return;
            if (!haveHl || !haveClose) return;

            MCloseOut[0] = curMClose;

            if (ShowHigh) MHighOut[0] = curMHigh; else MHighOut.Reset();
            if (ShowLow)  MLowOut[0]  = curMLow;  else MLowOut.Reset();

            if (ShowBands)
            {
                Upper1[0] = curMClose + curSeClose;
                Lower1[0] = curMClose - curSeClose;
                Upper2[0] = curMClose + 2.0 * curSeClose;
                Lower2[0] = curMClose - 2.0 * curSeClose;
                Upper3[0] = curMClose + 3.0 * curSeClose;
                Lower3[0] = curMClose - 3.0 * curSeClose;
            }
            else
            {
                Upper1.Reset(); Lower1.Reset();
                Upper2.Reset(); Lower2.Reset();
                Upper3.Reset(); Lower3.Reset();
            }
        }

        // -----------------------------------------------------------------
        // OLS regression on a NinjaScript series. Mean = value at current
        // bar; SE = sqrt(SSE / (n-2)).
        // -----------------------------------------------------------------
        private bool OlsMeanSe(ISeries<double> series, int period, out double mean, out double se)
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

            double sse = 0.0;
            for (int count = 0; count < period; count++)
            {
                double yHat = intercept + slope * count;
                double y    = series[period - 1 - count];
                double res  = y - yHat;
                sse += res * res;
            }
            se = Math.Sqrt(sse / (period - 2));
            return true;
        }

        // ─────────────────────────────────────────────────────────────────
        #region Properties
        // ─────────────────────────────────────────────────────────────────
        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "HL period (1h bars)",
                 Description = "OLS lookback in 1h bars for M_high/M_low (20 = 20 hours)",
                 GroupName = "Periods", Order = 0)]
        public int HlPeriod { get; set; }

        [Range(3, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Close period (15m bars)",
                 Description = "OLS lookback in 15m bars for M_close (20 = 5 hours)",
                 GroupName = "Periods", Order = 1)]
        public int ClosePeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show M_high (1h)", GroupName = "Plot toggles", Order = 0)]
        public bool ShowHigh { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show M_low (1h)", GroupName = "Plot toggles", Order = 1)]
        public bool ShowLow { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show ±1/2/3 σ bands on M_close (15m)",
                 GroupName = "Plot toggles", Order = 2)]
        public bool ShowBands { get; set; }

        [XmlIgnore] [Display(Name = "M_close color (15m)", GroupName = "Plot colors", Order = 0)]
        public Brush MCloseBrush { get; set; }
        [Browsable(false)] public string MCloseBrushSerialize { get { return Serialize.BrushToString(MCloseBrush); } set { MCloseBrush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "M_high color (1h)", GroupName = "Plot colors", Order = 1)]
        public Brush MHighBrush { get; set; }
        [Browsable(false)] public string MHighBrushSerialize { get { return Serialize.BrushToString(MHighBrush); } set { MHighBrush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "M_low color (1h)", GroupName = "Plot colors", Order = 2)]
        public Brush MLowBrush { get; set; }
        [Browsable(false)] public string MLowBrushSerialize { get { return Serialize.BrushToString(MLowBrush); } set { MLowBrush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "Band 1σ color", GroupName = "Plot colors", Order = 3)]
        public Brush Band1Brush { get; set; }
        [Browsable(false)] public string Band1BrushSerialize { get { return Serialize.BrushToString(Band1Brush); } set { Band1Brush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "Band 2σ color", GroupName = "Plot colors", Order = 4)]
        public Brush Band2Brush { get; set; }
        [Browsable(false)] public string Band2BrushSerialize { get { return Serialize.BrushToString(Band2Brush); } set { Band2Brush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "Band 3σ color", GroupName = "Plot colors", Order = 5)]
        public Brush Band3Brush { get; set; }
        [Browsable(false)] public string Band3BrushSerialize { get { return Serialize.BrushToString(Band3Brush); } set { Band3Brush = Serialize.StringToBrush(value); } }

        [Browsable(false), XmlIgnore] public Series<double> MCloseOut => Values[0];
        [Browsable(false), XmlIgnore] public Series<double> MHighOut  => Values[1];
        [Browsable(false), XmlIgnore] public Series<double> MLowOut   => Values[2];
        [Browsable(false), XmlIgnore] public Series<double> Upper1    => Values[3];
        [Browsable(false), XmlIgnore] public Series<double> Lower1    => Values[4];
        [Browsable(false), XmlIgnore] public Series<double> Upper2    => Values[5];
        [Browsable(false), XmlIgnore] public Series<double> Lower2    => Values[6];
        [Browsable(false), XmlIgnore] public Series<double> Upper3    => Values[7];
        [Browsable(false), XmlIgnore] public Series<double> Lower3    => Values[8];
        #endregion
    }
}
