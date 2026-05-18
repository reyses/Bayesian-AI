// =============================================================================
// ZigzagRunnerNative 1.0.0-RC -- 2026-05-17
// =============================================================================
//
// PURPOSE
//   Native NT8 zigzag-runner strategy that EXACTLY matches the Python
//   pipeline calibration. Foundation for the hybrid (Python sidecar) build.
//
// PYTHON PIPELINE PARITY (source: tools/build_zigzag_pivot_dataset.py)
//   - ATR period 14 on 1m bars              ->  AtrPeriod=14, AtrTfMinutes=1
//   - ATR multiplier 4.0                     ->  AtrMult=4.0
//   - Median ATR (last period*3 TRs)         ->  UseMedianAtr=true
//   - Pivot detection on 5s closes           ->  ZigZagTfSeconds=5
//   - min_bars between pivots = 36 (3 min)   ->  MinBars=36
//   - Tick size 0.25, min reversal floor 4 ticks (= 1 point)
//
// IMPORTANT: All prior ZigzagRunner versions in this folder did NOT match
// Python calibration:
//   - v1.0.cs (deployed as ZigzagRunner.cs): static R=30, single TF
//   - v1.0.8-RC: UseDynamicR available but ATR computed on the pivot
//     series, defaults AtrLookbackBars=60, AtrMultiplier=5.0 (wrong)
//   The ZigZagATR.cs INDICATOR has the right architecture; this strategy
//   inlines that same logic.
//
// LOGIC (mirrors ZigZagATR.cs indicator state machine)
//   BIP=0  primary chart TF                  -- render + order management
//   BIP=1  1m bars                           -- median ATR(14) buffer
//   BIP=2  5s bars                           -- pivot state machine
//
//   On each 5s close:
//     rTicks = max(2, round(curAtr / tickSize * AtrMult))
//     rPrice = rTicks * tickSize
//     run zigzag state machine (direction, extremePrice, extremeBar)
//     when retracement >= rPrice AND bars-since-extreme >= MinBars,
//     PIVOT IS CONFIRMED  -> set lastPivot, emit entry signal
//
// ENTRY/EXIT (v1.0.4 structural-fix pattern)
//   Counter-trend by default:
//     HIGH pivot  -> SHORT (fade the rally)
//     LOW pivot   -> LONG  (fade the dip)
//   ALWAYS exit current position BEFORE placing new entry (decouples
//   exit from entry-as-side-effect; protects skip/always-long/always-short).
//
// SAFETY
//   - EOD force-close at EodHourUtc:EodMinuteUtc
//   - Entry cutoff: no new entries after EntryCutoffHourUtc:EntryCutoffMinuteUtc
//   - IsExitOnSessionCloseStrategy=true, ExitOnSessionCloseSeconds=30
//
// VERSIONING
//   v1.0.0-RC  -- 2026-05-17 -- initial port, defaults match Python pipeline
//
// DEPLOYMENT GATE
//   This file lives in docs/nt8/. It is NOT auto-deployed. To run:
//     1. Open NT8 NinjaScript Editor
//     2. Copy this file to Documents/NinjaTrader 8/bin/Custom/Strategies/
//        (requires explicit user approval per CLAUDE.md)
//     3. F5 compile
//     4. Apply ZigzagRunnerNative_v1.0.0-RC to MNQ chart
//
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public enum PivotActionNative
    {
        Long,
        Short,
        Skip
    }

    public class ZigzagRunnerNative_v100RC : Strategy
    {
        // ── BarsInProgress routing ───────────────────────────────────────
        // BIP=0 primary chart TF (render + orders)
        // BIP=1 1m bars (ATR source)
        // BIP=2 5s bars (zigzag state machine)
        private const int BIP_ATR    = 1;
        private const int BIP_ZIGZAG = 2;

        // ── Version ──────────────────────────────────────────────────────
        private const string VERSION = "1.0.0-RC";

        // ── Properties: ATR (Python parity) ──────────────────────────────
        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ATR timeframe (minutes)",
                 Description = "Minute TF for ATR. Python = 1m.",
                 GroupName = "1. ATR (Python parity)", Order = 0)]
        public int AtrTfMinutes { get; set; }

        [Range(2, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ATR period",
                 Description = "Bars on ATR TF. Python = 14.",
                 GroupName = "1. ATR (Python parity)", Order = 1)]
        public int AtrPeriod { get; set; }

        [Range(0.1, 100.0), NinjaScriptProperty]
        [Display(Name = "ATR multiplier",
                 Description = "R-threshold = ATR x this. Python = 4.0 (visually calibrated).",
                 GroupName = "1. ATR (Python parity)", Order = 2)]
        public double AtrMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use median ATR",
                 Description = "true = median of last (period x 3) TRs. Python compute_atr uses median.",
                 GroupName = "1. ATR (Python parity)", Order = 3)]
        public bool UseMedianAtr { get; set; }

        // ── Properties: ZigZag (Python parity) ───────────────────────────
        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "ZigZag timeframe (seconds)",
                 Description = "Second TF for swing detection. Python = 5s.",
                 GroupName = "2. ZigZag (Python parity)", Order = 0)]
        public int ZigZagTfSeconds { get; set; }

        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(Name = "Min bars between pivots",
                 Description = "Required bars since extreme before confirming. Python = 36 (= 3 min at 5s).",
                 GroupName = "2. ZigZag (Python parity)", Order = 1)]
        public int MinBars { get; set; }

        // ── Properties: Execution ────────────────────────────────────────
        [Range(1, 100), NinjaScriptProperty]
        [Display(Name = "Contracts",
                 Description = "Contracts per entry (flat sizing = baseline).",
                 GroupName = "3. Execution", Order = 0)]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "On High Pivot",
                 Description = "Action at HIGH pivot. Short = counter-trend (v1.0 baseline).",
                 GroupName = "3. Execution", Order = 1)]
        public PivotActionNative OnHighPivot { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "On Low Pivot",
                 Description = "Action at LOW pivot. Long = counter-trend (v1.0 baseline).",
                 GroupName = "3. Execution", Order = 2)]
        public PivotActionNative OnLowPivot { get; set; }

        // ── Properties: Schedule ─────────────────────────────────────────
        [Range(0, 23), NinjaScriptProperty]
        [Display(Name = "EOD Hour UTC",
                 Description = "Force-close hour UTC (20 = before NYSE close).",
                 GroupName = "4. Schedule (UTC)", Order = 0)]
        public int EodHourUtc { get; set; }

        [Range(0, 59), NinjaScriptProperty]
        [Display(Name = "EOD Minute UTC",
                 Description = "Force-close minute.",
                 GroupName = "4. Schedule (UTC)", Order = 1)]
        public int EodMinuteUtc { get; set; }

        [Range(0, 23), NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Hour UTC",
                 Description = "No new entries after this time.",
                 GroupName = "4. Schedule (UTC)", Order = 2)]
        public int EntryCutoffHourUtc { get; set; }

        [Range(0, 59), NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Minute UTC",
                 Description = "No new entries after this minute.",
                 GroupName = "4. Schedule (UTC)", Order = 3)]
        public int EntryCutoffMinuteUtc { get; set; }

        // ── Plot accessors ───────────────────────────────────────────────
        [System.Xml.Serialization.XmlIgnore]
        [Browsable(false)]
        public Series<double> ZigZagPlot  { get { return Values[0]; } }
        [System.Xml.Serialization.XmlIgnore]
        [Browsable(false)]
        public Series<double> ExtremePlot { get { return Values[1]; } }
        [System.Xml.Serialization.XmlIgnore]
        [Browsable(false)]
        public Series<double> TriggerPlot { get { return Values[2]; } }

        // ── ATR state (BIP=1) ────────────────────────────────────────────
        private Queue<double> trBuffer;
        private int           trBufferMax;
        private double        curAtr;
        private bool          haveAtr;

        // ── ZigZag state (BIP=2) ─────────────────────────────────────────
        private int    direction;          // 0 undef, +1 up leg, -1 down leg
        private double extremePrice;
        private int    extremeBar;
        private int    lastPivotBar;
        private double lastPivotPrice;

        // ── Pivot-fire signal (set on BIP=2, consumed on next primary bar) ─
        private bool   pendingPivotFire;
        private int    pendingPivotDir;    // +1 high pivot, -1 low pivot
        private double pendingPivotPrice;

        // ── Render-cache (for primary plots) ─────────────────────────────
        private double curExtreme;
        private double curTrigger;
        private double curZigZag;
        private bool   haveZz;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                          = "ZigzagRunnerNative_v" + VERSION;
                Description                   = "Native NT8 zigzag-runner. ATR x 4 dynamic R on 1m bars, " +
                                                "pivots on 5s. Matches Python build_zigzag_pivot_dataset.py.";
                Calculate                     = Calculate.OnBarClose;
                EntriesPerDirection           = 1;
                EntryHandling                 = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy  = true;
                ExitOnSessionCloseSeconds     = 30;
                IsFillLimitOnTouch            = false;
                MaximumBarsLookBack           = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution           = OrderFillResolution.Standard;
                Slippage                      = 0;
                StartBehavior                 = StartBehavior.WaitUntilFlat;
                TimeInForce                   = TimeInForce.Gtc;
                TraceOrders                   = false;
                RealtimeErrorHandling         = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling            = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade           = 2;

                // Python pipeline defaults
                AtrTfMinutes                  = 1;
                AtrPeriod                     = 14;
                AtrMult                       = 4.0;
                UseMedianAtr                  = true;
                ZigZagTfSeconds               = 5;
                MinBars                       = 36;

                Contracts                     = 1;
                OnHighPivot                   = PivotActionNative.Short;  // counter-trend
                OnLowPivot                    = PivotActionNative.Long;   // counter-trend

                EodHourUtc                    = 20;
                EodMinuteUtc                  = 55;
                EntryCutoffHourUtc            = 20;
                EntryCutoffMinuteUtc          = 30;

                AddPlot(Brushes.Yellow, "ZigZag");
                AddPlot(Brushes.Orange, "Extreme");
                AddPlot(Brushes.Cyan,   "RTrigger");
            }
            else if (State == State.Configure)
            {
                // Order matters: BIP=1 ATR, BIP=2 ZigZag
                AddDataSeries(BarsPeriodType.Minute, AtrTfMinutes);
                AddDataSeries(BarsPeriodType.Second, ZigZagTfSeconds);

                trBufferMax     = Math.Max(AtrPeriod * 3, AtrPeriod + 1);
                trBuffer        = new Queue<double>(trBufferMax);
                curAtr          = 0.0;
                haveAtr         = false;

                direction       = 0;
                extremePrice    = 0.0;
                extremeBar      = -1;
                lastPivotBar    = -1;
                lastPivotPrice  = 0.0;

                pendingPivotFire  = false;
                pendingPivotDir   = 0;
                pendingPivotPrice = 0.0;

                curExtreme = curTrigger = curZigZag = 0.0;
                haveZz     = false;
            }
        }

        protected override void OnBarUpdate()
        {
            // ── BIP=1 (1m bars): update median ATR ──────────────────────
            if (BarsInProgress == BIP_ATR)
            {
                if (CurrentBars[BIP_ATR] < 1) return;
                double h  = Highs[BIP_ATR][0];
                double l  = Lows[BIP_ATR][0];
                double pc = CurrentBars[BIP_ATR] > 0
                            ? Closes[BIP_ATR][1] : Closes[BIP_ATR][0];
                double tr = Math.Max(h - l,
                                Math.Max(Math.Abs(h - pc), Math.Abs(l - pc)));

                if (UseMedianAtr)
                {
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
                    // Wilder smoothing (mimics built-in ATR)
                    if (!haveAtr && CurrentBars[BIP_ATR] >= AtrPeriod)
                    {
                        curAtr  = tr;
                        haveAtr = true;
                    }
                    else if (haveAtr)
                    {
                        curAtr = (curAtr * (AtrPeriod - 1) + tr) / AtrPeriod;
                    }
                }
                return;
            }

            // ── BIP=2 (5s bars): zigzag state machine ───────────────────
            if (BarsInProgress == BIP_ZIGZAG)
            {
                if (!haveAtr || curAtr <= 0) return;
                if (CurrentBars[BIP_ZIGZAG] < MinBars + 1) return;

                double close  = Closes[BIP_ZIGZAG][0];
                double tick   = TickSize;
                int    rTicks = Math.Max(2, (int)Math.Round(curAtr / tick * AtrMult));
                double rPrice = rTicks * tick;
                int    barIdx = CurrentBars[BIP_ZIGZAG];

                bool pivotConfirmed = false;
                int  newPivotDir    = 0;   // +1 high pivot, -1 low pivot

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
                        // HIGH pivot confirmed at extremePrice
                        pivotConfirmed   = true;
                        newPivotDir      = +1;
                        lastPivotBar     = extremeBar;
                        lastPivotPrice   = extremePrice;
                        direction        = -1;
                        extremePrice     = close;
                        extremeBar       = barIdx;
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
                        // LOW pivot confirmed at extremePrice
                        pivotConfirmed   = true;
                        newPivotDir      = -1;
                        lastPivotBar     = extremeBar;
                        lastPivotPrice   = extremePrice;
                        direction        = 1;
                        extremePrice     = close;
                        extremeBar       = barIdx;
                    }
                }

                // Update render cache
                curExtreme = extremePrice;
                curZigZag  = direction != 0 ? lastPivotPrice : close;
                if (direction == 1)       curTrigger = extremePrice - rPrice;
                else if (direction == -1) curTrigger = extremePrice + rPrice;
                else                       curTrigger = close;
                haveZz = true;

                if (pivotConfirmed)
                {
                    pendingPivotFire  = true;
                    pendingPivotDir   = newPivotDir;
                    pendingPivotPrice = lastPivotPrice;
                }
                return;
            }

            // ── BIP=0 (primary chart TF): render plots + place orders ───
            if (BarsInProgress != 0) return;
            if (CurrentBar < BarsRequiredToTrade) return;

            // Render
            if (haveZz)
            {
                Values[0][0] = curZigZag;
                Values[1][0] = curExtreme;
                Values[2][0] = curTrigger;
            }

            // EOD force-close
            DateTime nowUtc = Time[0].ToUniversalTime();
            int      minsOfDay = nowUtc.Hour * 60 + nowUtc.Minute;
            int      eodMins   = EodHourUtc * 60 + EodMinuteUtc;
            int      cutMins   = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("EOD_ExitLong", "");
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort("EOD_ExitShort", "");
                pendingPivotFire = false;
                return;
            }

            if (!pendingPivotFire) return;
            pendingPivotFire = false;

            // v1.0.4 structural fix: ALWAYS exit current position FIRST
            if (Position.MarketPosition == MarketPosition.Long)
                ExitLong("PivotExitLong", "");
            else if (Position.MarketPosition == MarketPosition.Short)
                ExitShort("PivotExitShort", "");

            // Entry cutoff window (skip new entries but allow exits)
            if (minsOfDay >= cutMins) return;

            // Determine entry action
            PivotActionNative action = (pendingPivotDir == +1) ? OnHighPivot : OnLowPivot;
            string label = (pendingPivotDir == +1) ? "HighPivot" : "LowPivot";

            if (action == PivotActionNative.Long)
                EnterLong(Contracts, label + "_Long");
            else if (action == PivotActionNative.Short)
                EnterShort(Contracts, label + "_Short");
            // else Skip: stay flat
        }
    }
}
