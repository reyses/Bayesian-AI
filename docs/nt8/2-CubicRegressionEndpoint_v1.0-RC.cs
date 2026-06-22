//
// 2-CubicRegressionEndpoint_v1.0-RC.cs
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: Causal Cubic Regression Filter
//
// Fits a cubic polynomial: y(t) = a*t^3 + b*t^2 + c*t + d
// over a rolling Period of bars on an INDEPENDENT timeframe, where t is measured
// in minutes (t = index * (timeframe_value / 60.0) or custom seconds scaling).
// Pre-calculates OLS pseudo-inverse weights on startup to evaluate the Endpoint (t_last)
// Value, Slope (first derivative), and Curvature (second derivative) in O(1) per bar.
//
// Matches Python SFE orange-line (7.5m cubic) logic.
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
    public class _2_CubicRegressionEndpoint_v10 : Indicator
    {
        private const string VERSION = "1.0-RC";
        private const int BIP_CUBIC = 1;

        // Weights arrays
        private double[] valWeights;
        private double[] slopeWeights;
        private double[] curvWeights;

        // Cached values
        private double curVal, curSlope, curCurv;
        private bool haveCubicData;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Fast O(1) endpoint value, slope, and curvature of a rolling Cubic Regression on an independent timeframe. " +
                              "Matches Python SFE 7.5m cubic.";
                Name        = "2-CubicRegressionEndpoint_v1.0-RC";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                // Timeframe & window defaults
                CubicTimeFrameType = BarsPeriodType.Second;
                CubicTimeFrameValue = 1;
                Period = 450; // default 7.5 minutes on a 1-second chart (450 bars)
                
                AddPlot(new Stroke(Brushes.DarkOrange, 2), PlotStyle.Line, "CubicValue");
                AddPlot(new Stroke(Brushes.Magenta, 1),    PlotStyle.Line, "CubicSlope");
                AddPlot(new Stroke(Brushes.Cyan, 1),       PlotStyle.Line, "CubicCurvature");
            }
            else if (State == State.Configure)
            {
                AddDataSeries(CubicTimeFrameType, CubicTimeFrameValue);
                curVal = double.NaN;
                curSlope = double.NaN;
                curCurv = double.NaN;
                haveCubicData = false;
            }
            else if (State == State.DataLoaded)
            {
                ComputeWeights();
            }
        }

        protected override void OnBarUpdate()
        {
            // ── Cubic Series updates (BIP_CUBIC) ──
            if (BarsInProgress == BIP_CUBIC)
            {
                if (CurrentBars[BIP_CUBIC] >= Period - 1 && Period >= 4)
                {
                    double val = 0.0;
                    double slope = 0.0;
                    double curv = 0.0;

                    for (int i = 0; i < Period; i++)
                    {
                        double price = Closes[BIP_CUBIC][i];
                        int idx = Period - 1 - i; // map to chronological weights index
                        val   += valWeights[idx] * price;
                        slope += slopeWeights[idx] * price;
                        curv  += curvWeights[idx] * price;
                    }

                    curVal = val;
                    curSlope = slope;
                    curCurv = curv;
                    haveCubicData = true;
                }
                return;
            }

            // ── Primary Series updates (Rendering) ──
            if (BarsInProgress != 0) return;

            if (haveCubicData)
            {
                CubicValue[0]     = curVal;
                CubicSlope[0]     = curVal + curSlope;
                CubicCurvature[0] = curVal + curCurv;
            }
        }

        private void ComputeWeights()
        {
            int n = Period;
            valWeights   = new double[n];
            slopeWeights = new double[n];
            curvWeights  = new double[n];

            if (n < 4) return;

            // 1. Build time grid x (minutes)
            double[] x = new double[n];
            double barMin = 1.0 / 60.0; // default for 1s
            if (CubicTimeFrameType == BarsPeriodType.Second)
                barMin = CubicTimeFrameValue / 60.0;
            else if (CubicTimeFrameType == BarsPeriodType.Minute)
                barMin = CubicTimeFrameValue;
            else if (CubicTimeFrameType == BarsPeriodType.Day)
                barMin = CubicTimeFrameValue * 1440.0;
            else if (CubicTimeFrameType == BarsPeriodType.Week)
                barMin = CubicTimeFrameValue * 10080.0;

            for (int i = 0; i < n; i++)
            {
                x[i] = i * barMin;
            }
            double xe = x[n - 1]; // endpoint

            // 2. Sum of powers S_p = sum(x_i^p)
            double[] S = new double[7]; // S[0] to S[6]
            for (int i = 0; i < n; i++)
            {
                double val = 1.0;
                for (int p = 0; p <= 6; p++)
                {
                    S[p] += val;
                    val *= x[i];
                }
            }

            // 3. Populate normal equations matrix M (4x4)
            double[,] M = new double[4, 4] {
                { S[6], S[5], S[4], S[3] },
                { S[5], S[4], S[3], S[2] },
                { S[4], S[3], S[2], S[1] },
                { S[3], S[2], S[1], S[0] }
            };

            // 4. Invert M
            double[,] Minv = new double[4, 4];
            if (!InvertMatrix4x4(M, Minv))
            {
                return;
            }

            // 5. Compute weights for each index i
            for (int i = 0; i < n; i++)
            {
                double xi = x[i];
                double xi2 = xi * xi;
                double xi3 = xi2 * xi;

                double p0 = Minv[0, 0] * xi3 + Minv[0, 1] * xi2 + Minv[0, 2] * xi + Minv[0, 3];
                double p1 = Minv[1, 0] * xi3 + Minv[1, 1] * xi2 + Minv[1, 2] * xi + Minv[1, 3];
                double p2 = Minv[2, 0] * xi3 + Minv[2, 1] * xi2 + Minv[2, 2] * xi + Minv[2, 3];
                double p3 = Minv[3, 0] * xi3 + Minv[3, 1] * xi2 + Minv[3, 2] * xi + Minv[3, 3];

                valWeights[i] = (xe * xe * xe) * p0 + (xe * xe) * p1 + xe * p2 + p3;
                slopeWeights[i] = 3.0 * (xe * xe) * p0 + 2.0 * xe * p1 + p2;
                curvWeights[i] = 6.0 * xe * p0 + 2.0 * p1;
            }
        }

        private bool InvertMatrix4x4(double[,] mat, double[,] inv)
        {
            int n = 4;
            double[,] temp = new double[n, 2 * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    temp[i, j] = mat[i, j];
                    temp[i, j + n] = (i == j) ? 1.0 : 0.0;
                }
            }

            for (int i = 0; i < n; i++)
            {
                double maxVal = Math.Abs(temp[i, i]);
                int maxRow = i;
                for (int k = i + 1; k < n; k++)
                {
                    if (Math.Abs(temp[k, i]) > maxVal)
                    {
                        maxVal = Math.Abs(temp[k, i]);
                        maxRow = k;
                    }
                }

                if (maxVal < 1e-12) return false;

                for (int k = 0; k < 2 * n; k++)
                {
                    double t = temp[i, k];
                    temp[i, k] = temp[maxRow, k];
                    temp[maxRow, k] = t;
                }

                double pivot = temp[i, i];
                for (int k = 0; k < 2 * n; k++)
                {
                    temp[i, k] /= pivot;
                }

                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        double factor = temp[j, i];
                        for (int k = 0; k < 2 * n; k++)
                        {
                            temp[j, k] -= factor * temp[i, k];
                        }
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    inv[i, j] = temp[i, j + n];
                }
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

        [NinjaScriptProperty]
        [Display(Name = "Timeframe Unit", Description = "Timeframe type for Cubic series", Order = 0, GroupName = "Parameters")]
        public BarsPeriodType CubicTimeFrameType { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Timeframe Value", Description = "Timeframe period value for Cubic series", Order = 1, GroupName = "Parameters")]
        public int CubicTimeFrameValue { get; set; }

        [Range(4, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Rolling Window (Bars)", Description = "Lookback bars for Cubic OLS regression (>=4)", Order = 2, GroupName = "Parameters")]
        public int Period { get; set; }

        [Browsable(true)]
        [Display(Name = "Cubic Lookback (Min)", Description = "Calculated lookback in minutes for Cubic bands.", GroupName = "Calculators", Order = 0)]
        public double CalculatedCubicLookbackMinutes => CalculateMinutes(Period, CubicTimeFrameType, CubicTimeFrameValue);

        [Browsable(false)] [XmlIgnore] public Series<double> CubicValue     => Values[0];
        [Browsable(false)] [XmlIgnore] public Series<double> CubicSlope     => Values[1];
        [Browsable(false)] [XmlIgnore] public Series<double> CubicCurvature => Values[2];

        #endregion
    }
}
