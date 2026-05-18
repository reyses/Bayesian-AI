//
// ZigZagATR.cs   v2.0  (multi-timeframe)
// -----------------------------------------------------------------------------
// NinjaTrader 8 indicator: ATR-adaptive ZigZag on FIXED timeframes — exact
// parity with the Python pipeline (tools/build_zigzag_pivot_dataset.py +
// tools/auto_swing_marker.detect_swings).
//
// Architecture:
//   AtrTfMinutes  : ATR is computed on bars at this minute timeframe (default 1m)
//                   Period = AtrPeriod (default 14 bars = 14 minutes).
//                   With UseMedianAtr=true, uses median of last (period × 3)
//                   true ranges (matches Python compute_atr).
//
//   ZigZagTfSeconds: swing detection runs on bars at this second timeframe
//                    (default 5s = matches Python detect_swings on 5s closes).
//                    min_bars defaults to 36 (= 3 minutes at 5s = same as Python).
//
//   Primary chart TF (whatever you opened — any) is just the rendering canvas.
//   Indicator pulls 1m + 5s in the background regardless of chart TF.
//
// Threshold:
//     min_reversal_price = ATR(period) × AtrMult
//     pivots confirmed when reversal ≥ min_reversal AND ≥ min_bars elapsed
//
// Plots (rendered on the primary chart TF as step functions):
//   0  ZigZag pivot line (Yellow)
//   1  Running extreme   (Orange)
//   2  R-trigger         (Cyan)
//
// -----------------------------------------------------------------------------
#region Using declarations
using System;
using System.Collections.Generic;
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
    /// Multi-timeframe ATR-adaptive ZigZag. ATR on AtrTfMinutes minute bars,
    /// swing detection on ZigZagTfSeconds second bars. Mirrors the Python
    /// build_zigzag_pivot_dataset.py + detect_swings calibration exactly.
    /// </summary>
    public class ZigZagATR : Indicator
    {
        // BarsInProgress:
        //   0 = primary chart TF (render)
        //   1 = ATR timeframe (minute bars)
        //   2 = ZigZag detection timeframe (second bars)
        private const int BIP_ATR    = 1;
        private const int BIP_ZIGZAG = 2;

        // ATR state (computed on BIP=1 ticks)
        private Queue<double> trBuffer;
        private int           trBufferMax;
        private double        curAtr;
        private ATR           atrBuiltin;     // when UseMedianAtr=false
        private bool          haveAtr;

        // ZigZag state (computed on BIP=2 ticks)
        private int     direction;
        private double  extremePrice;
        private int     extremeBar;
        private int     lastPivotBar;
        private double  lastPivotPrice;

        // Held values for rendering on the primary
        private double  curExtreme;
        private double  curTrigger;
        private double  curZigZag;
        private bool    haveZz;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "MTF ATR-adaptive ZigZag. ATR on 1m bars × AtrMult, " +
                              "swings on 5s closes. Mirrors Python detect_swings.";
                Name        = "ZigZagATR";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                IsSuspendedWhileInactive = true;
                PaintPriceMarkers = false;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;

                AtrTfMinutes    = 1;
                AtrPeriod       = 14;
                AtrMult         = 4.0;
                UseMedianAtr    = true;
                ZigZagTfSeconds = 5;
                MinBars         = 36;     // 3 min equivalent at 5s
                ShowExtreme     = true;
                ShowTrigger     = true;

                ZigZagBrush   = Brushes.Yellow;
                ExtremeBrush  = Brushes.Orange;
                TriggerBrush  = Brushes.Cyan;

                AddPlot(new Stroke(Brushes.Yellow,  2), PlotStyle.Line, "ZigZag");
                AddPlot(new Stroke(Brushes.Orange,  1), PlotStyle.Line, "Extreme");
                AddPlot(new Stroke(Brushes.Cyan,    1), PlotStyle.Line, "RTrigger");
            }
            else if (State == State.Configure)
            {
                // Order matters — BIP=1 then BIP=2
                AddDataSeries(BarsPeriodType.Minute, AtrTfMinutes);
                AddDataSeries(BarsPeriodType.Second, ZigZagTfSeconds);

                trBufferMax    = Math.Max(AtrPeriod * 3, AtrPeriod + 1);
                trBuffer       = new Queue<double>(trBufferMax);
                curAtr         = 0.0;
                haveAtr        = false;

                direction      = 0;
                extremePrice   = 0.0;
                extremeBar     = -1;
                lastPivotBar   = -1;
                lastPivotPrice = 0.0;

                curExtreme = curTrigger = curZigZag = 0.0;
                haveZz     = false;
            }
            else if (State == State.DataLoaded)
            {
                Plots[0].Brush = ZigZagBrush;
                Plots[1].Brush = ExtremeBrush;
                Plots[2].Brush = TriggerBrush;

                if (!UseMedianAtr)
                    atrBuiltin = ATR(BarsArray[BIP_ATR], AtrPeriod);
            }
        }

        protected override void OnBarUpdate()
        {
            // ── ATR timeframe (minute) — refresh ATR on each bar close ──
            if (BarsInProgress == BIP_ATR)
            {
                if (CurrentBars[BIP_ATR] < 1) return;
                if (UseMedianAtr)
                {
                    double h  = Highs[BIP_ATR][0];
                    double l  = Lows[BIP_ATR][0];
                    double pc = CurrentBars[BIP_ATR] > 0
                                ? Closes[BIP_ATR][1] : Closes[BIP_ATR][0];
                    double tr = Math.Max(h - l,
                                  Math.Max(Math.Abs(h - pc), Math.Abs(l - pc)));
                    trBuffer.Enqueue(tr);
                    while (trBuffer.Count > trBufferMax)
                        trBuffer.Dequeue();
                    if (trBuffer.Count >= AtrPeriod)
                    {
                        double[] arr = new double[trBuffer.Count];
                        trBuffer.CopyTo(arr, 0);
                        Array.Sort(arr);
                        int n = arr.Length;
                        curAtr = (n % 2 == 0)
                                  ? (arr[n / 2 - 1] + arr[n / 2]) / 2.0
                                  : arr[n / 2];
                        haveAtr = true;
                    }
                }
                else
                {
                    if (CurrentBars[BIP_ATR] >= AtrPeriod)
                    {
                        curAtr  = atrBuiltin[0];
                        haveAtr = true;
                    }
                }
                return;
            }

            // ── ZigZag timeframe (second) — process swing logic on closes ──
            if (BarsInProgress == BIP_ZIGZAG)
            {
                if (!haveAtr || curAtr <= 0) return;
                if (CurrentBars[BIP_ZIGZAG] < MinBars + 1) return;

                double close  = Closes[BIP_ZIGZAG][0];
                double tick   = TickSize;
                int    rTicks = Math.Max(2, (int)Math.Round(curAtr / tick * AtrMult));
                double rPrice = rTicks * tick;

                int    barIdx = CurrentBars[BIP_ZIGZAG];

                if (direction == 0)
                {
                    if (extremeBar == -1)
                    {
                        extremePrice   = close;
                        extremeBar     = barIdx;
                        lastPivotBar   = barIdx;
                        lastPivotPrice = close;
                    }
                    double sp = lastPivotPrice;
                    if (close > sp && (close - sp) >= rPrice)
                    {
                        direction = 1; extremePrice = close; extremeBar = barIdx;
                    }
                    else if (close < sp && (sp - close) >= rPrice)
                    {
                        direction = -1; extremePrice = close; extremeBar = barIdx;
                    }
                    else if (close > extremePrice)
                    {
                        extremePrice = close; extremeBar = barIdx;
                    }
                }
                else if (direction == 1)
                {
                    if (close >= extremePrice)
                    {
                        extremePrice = close; extremeBar = barIdx;
                    }
                    else if ((extremePrice - close) >= rPrice
                             && (barIdx - extremeBar) >= MinBars)
                    {
                        lastPivotBar = extremeBar; lastPivotPrice = extremePrice;
                        direction = -1; extremePrice = close; extremeBar = barIdx;
                    }
                }
                else // direction == -1
                {
                    if (close <= extremePrice)
                    {
                        extremePrice = close; extremeBar = barIdx;
                    }
                    else if ((close - extremePrice) >= rPrice
                             && (barIdx - extremeBar) >= MinBars)
                    {
                        lastPivotBar = extremeBar; lastPivotPrice = extremePrice;
                        direction = 1; extremePrice = close; extremeBar = barIdx;
                    }
                }

                // Update render-cache
                curExtreme = extremePrice;
                curZigZag  = direction != 0 ? extremePrice : close;
                if (direction == 1)       curTrigger = extremePrice - rPrice;
                else if (direction == -1) curTrigger = extremePrice + rPrice;
                else                       curTrigger = close;
                haveZz = true;
                return;
            }

            // ── Primary chart TF — render the held values ───────────────────
            if (BarsInProgress != 0) return;
            if (!haveZz) return;

            Values[0][0] = curZigZag;
            if (ShowExtreme) Values[1][0] = curExtreme; else Values[1].Reset();
            if (ShowTrigger) Values[2][0] = curTrigger; else Values[2].Reset();
        }

        // ─────────────────────────────────────────────────────────────────
        #region Properties
        // ─────────────────────────────────────────────────────────────────
        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ATR timeframe (minutes)",
                 Description = "Minute timeframe for ATR calc (default 1)",
                 GroupName = "ATR settings", Order = 0)]
        public int AtrTfMinutes { get; set; }

        [Range(2, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ATR period",
                 Description = "Lookback bars on the ATR TF (default 14)",
                 GroupName = "ATR settings", Order = 1)]
        public int AtrPeriod { get; set; }

        [Range(0.1, 100.0), NinjaScriptProperty]
        [Display(Name = "ATR multiplier",
                 Description = "R-threshold = ATR × this. 4.0 = visually-calibrated default.",
                 GroupName = "ATR settings", Order = 2)]
        public double AtrMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use median ATR (Python parity)",
                 Description = "true = median of last (period × 3) TRs (matches Python compute_atr)",
                 GroupName = "ATR settings", Order = 3)]
        public bool UseMedianAtr { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ZigZag timeframe (seconds)",
                 Description = "Second timeframe for swing detection (default 5)",
                 GroupName = "ZigZag settings", Order = 0)]
        public int ZigZagTfSeconds { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Min bars between pivots",
                 Description = "On the zigzag TF — default 36 = 3 minutes at 5s",
                 GroupName = "ZigZag settings", Order = 1)]
        public int MinBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show extreme trail",
                 Description = "Orange running-extreme stairstep",
                 GroupName = "Plot toggles", Order = 0)]
        public bool ShowExtreme { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show R-trigger",
                 Description = "Cyan reversal-trigger line (extreme ± R)",
                 GroupName = "Plot toggles", Order = 1)]
        public bool ShowTrigger { get; set; }

        [XmlIgnore] [Display(Name = "ZigZag color", GroupName = "Plot colors", Order = 0)]
        public Brush ZigZagBrush { get; set; }
        [Browsable(false)] public string ZigZagBrushSerialize { get { return Serialize.BrushToString(ZigZagBrush); } set { ZigZagBrush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "Extreme color", GroupName = "Plot colors", Order = 1)]
        public Brush ExtremeBrush { get; set; }
        [Browsable(false)] public string ExtremeBrushSerialize { get { return Serialize.BrushToString(ExtremeBrush); } set { ExtremeBrush = Serialize.StringToBrush(value); } }

        [XmlIgnore] [Display(Name = "R-trigger color", GroupName = "Plot colors", Order = 2)]
        public Brush TriggerBrush { get; set; }
        [Browsable(false)] public string TriggerBrushSerialize { get { return Serialize.BrushToString(TriggerBrush); } set { TriggerBrush = Serialize.StringToBrush(value); } }

        [Browsable(false), XmlIgnore] public Series<double> ZigZagPlot  { get { return Values[0]; } }
        [Browsable(false), XmlIgnore] public Series<double> ExtremePlot { get { return Values[1]; } }
        [Browsable(false), XmlIgnore] public Series<double> TriggerPlot { get { return Values[2]; } }
        #endregion
    }
}
