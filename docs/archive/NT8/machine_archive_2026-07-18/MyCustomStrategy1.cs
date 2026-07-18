// =============================================================================
// MyCustomStrategy1 -- Universal Primary Chart + Absolute HTF Anchoring
// =============================================================================
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class MyCustomStrategy1 : Strategy
    {
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name="R (points)", Order=1, GroupName="Parameters")]
        public int R { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contracts", Order = 2, GroupName = "Execution")]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Activate (points)", Order = 1, GroupName = "Risk Management")]
        public double TrailActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Distance (points)", Order = 2, GroupName = "Risk Management")]
        public double TrailDistancePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Percent", Order = 3, GroupName = "Risk Management")]
        public double TrailPercent { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Hard Stop Loss (points)", Order = 4, GroupName = "Risk Management")]
        public double HardStopLossPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Negative Primary Bars", Order = 5, GroupName = "Risk Management")]
        public int MaxNegativeBars { get; set; }

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> Plot
        {
            get { return Values[0]; }
        }

        private SMA sma50_15m;
        private SMA sma50_60m;

        private int direction;
        private double extremePrice;
        private int extremeBarIdx;

        private double currentEntryPrice;
        private DateTime currentEntryTime;
        private int currentEntryDir;       
        private int currentEntryQty;

        private C1_DynamicRiskManager riskMgr;
        private C1_StagnationMonitor stagnationMon;
        private double currentTradeMfePts;   
        private double currentTradeMaePts;   

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                                        = "MyCustomStrategy1";
                Calculate                                   = Calculate.OnBarClose;
                EntriesPerDirection                         = 1;
                EntryHandling                               = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy                = true;
                ExitOnSessionCloseSeconds                   = 30;
                IsFillLimitOnTouch                          = false;
                MaximumBarsLookBack                         = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution                         = OrderFillResolution.Standard;
                Slippage                                    = 1;
                StartBehavior                               = StartBehavior.WaitUntilFlat;
                TimeInForce                                 = TimeInForce.Gtc;
                TraceOrders                                 = false;
                RealtimeErrorHandling                       = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling                          = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade                         = 50;
                IsInstantiatedOnEachOptimizationIteration   = false;

                R                                           = 30;
                Contracts                                   = 1;
                TrailActivatePoints                         = 10.0;   
                TrailDistancePoints                         = 5.0;    
                TrailPercent                                = 0.10;   
                HardStopLossPoints                          = 25.0;   
                MaxNegativeBars                             = 300; // Adjusted for sub-minute evaluation

                AddPlot(Brushes.Orange, "ZigzagExtreme");
            }
            else if (State == State.Configure)
            {
                // BarsArray[0] is user-defined at runtime (e.g., 1s, 10s, 1m)
                // Absolute structural anchoring for HTF validation
                AddDataSeries(BarsPeriodType.Minute, 1);  // BarsArray[1]
                AddDataSeries(BarsPeriodType.Minute, 5);  // BarsArray[2]
                AddDataSeries(BarsPeriodType.Minute, 15); // BarsArray[3]
                AddDataSeries(BarsPeriodType.Minute, 60); // BarsArray[4]

                direction = 0;
                extremePrice = double.NaN;
                extremeBarIdx = -1;
                currentEntryPrice = 0.0;
                currentEntryTime = DateTime.MinValue;
                currentEntryDir = 0;
                currentEntryQty = 0;
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;

                double t2ActPts = double.MaxValue;
                if (TrailPercent > 0.0)
                {
                    t2ActPts = TrailDistancePoints / TrailPercent;
                }

                riskMgr = new C1_DynamicRiskManager(HardStopLossPoints, TrailActivatePoints, TrailDistancePoints, t2ActPts, TrailPercent, RouteStopOrder);
                stagnationMon = new C1_StagnationMonitor(MaxNegativeBars);
            }
            else if (State == State.DataLoaded)
            {
                // Bind SMAs strictly to the absolute HTF arrays
                sma50_15m = SMA(BarsArray[3], 50);
                sma50_60m = SMA(BarsArray[4], 50);
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution == null || execution.Order == null) 
            {
                return;
            }

            if (currentEntryDir != 0)
            {
                bool flatAfter = false;
                if (marketPosition == MarketPosition.Flat)
                {
                    flatAfter = true;
                }

                bool flippedSign = false;
                if ((currentEntryDir > 0) && (marketPosition == MarketPosition.Short))
                {
                    flippedSign = true;
                }
                else if ((currentEntryDir < 0) && (marketPosition == MarketPosition.Long))
                {
                    flippedSign = true;
                }

                if (flatAfter || flippedSign)
                {
                    if (flatAfter)
                    {
                        currentEntryDir = 0;
                        currentEntryQty = 0;
                        currentEntryPrice = 0.0;
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState();
                    }
                    else
                    {
                        currentEntryDir = -1;
                        if (marketPosition == MarketPosition.Long)
                        {
                            currentEntryDir = 1;
                        }
                        
                        currentEntryPrice = price;
                        currentEntryTime = time;
                        currentEntryQty = Contracts;
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState(); 
                        stagnationMon = new C1_StagnationMonitor(MaxNegativeBars);
                    }
                    return;
                }
                return;
            }

            if (marketPosition != MarketPosition.Flat)
            {
                currentEntryDir = -1;
                if (marketPosition == MarketPosition.Long)
                {
                    currentEntryDir = 1;
                }

                currentEntryPrice = price;
                currentEntryTime = time;
                currentEntryQty = Contracts;   
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;
                riskMgr.ResetState(); 
                stagnationMon = new C1_StagnationMonitor(MaxNegativeBars);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;
            if (CurrentBars[0] < 1) return;
            if (CurrentBars[1] < 1) return;
            if (CurrentBars[2] < 1) return;
            if (CurrentBars[3] < 1) return;
            if (CurrentBars[4] < 1) return;

            // Restrict order routing strictly to primary event thread (BarsArray[0])
            if (BarsInProgress != 0) return;

            // Evaluates Zigzag entirely on the selected primary chart
            double c = Close[0];

            if (!double.IsNaN(extremePrice))
            {
                Plot[0] = extremePrice;
            }

            // --- ZIGZAG EVALUATION PIPELINE ---
            if (double.IsNaN(extremePrice))
            {
                extremePrice = c;
                extremeBarIdx = CurrentBar;
                return;
            }

            bool pivotConfirmed = false;
            int newPivotDir = 0;   

            if (direction == 0)
            {
                if ((c - extremePrice) >= R) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = -1; 
                    direction = 1; 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
                else if ((extremePrice - c) >= R) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = 1; 
                    direction = -1; 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
            }
            else if (direction == 1)
            {
                if (c > extremePrice) 
                { 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
                else if ((extremePrice - c) >= R) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = 1; 
                    direction = -1; 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
            }
            else 
            {
                if (c < extremePrice) 
                { 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
                else if ((c - extremePrice) >= R) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = -1; 
                    direction = 1; 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
            }

            // --- ENTRY GATE PIPELINE ---
            bool isFlipping = false;

            if (pivotConfirmed)
            {
                // Strict isolation: evaluating HTF SMAs against their respective HTF Close arrays
                double current15mSma = sma50_15m[1];
                double current60mSma = sma50_60m[1];
                double current15mClose = Closes[3][1]; 
                double current60mClose = Closes[4][1]; 

                bool htfBullish = false;
                if ((current15mClose > current15mSma) && (current60mClose > current60mSma))
                {
                    htfBullish = true;
                }

                bool htfBearish = false;
                if ((current15mClose < current15mSma) && (current60mClose < current60mSma))
                {
                    htfBearish = true;
                }

                if ((newPivotDir == 1) && htfBearish)
                {
                    if (Position.MarketPosition == MarketPosition.Short && Position.Quantity >= Contracts) 
                    {
                        return;
                    }
                    if (Position.MarketPosition == MarketPosition.Long) 
                    {
                        ExitLong(Position.Quantity, "FlipExitLong", "");
                    }
                    EnterShort(Contracts, "ShortAtHighPivot");
                    isFlipping = true;
                }
                else if ((newPivotDir == -1) && htfBullish)
                {
                    if (Position.MarketPosition == MarketPosition.Long && Position.Quantity >= Contracts) 
                    {
                        return; 
                    }
                    if (Position.MarketPosition == MarketPosition.Short) 
                    {
                        ExitShort(Position.Quantity, "FlipExitShort", "");
                    }
                    EnterLong(Contracts, "LongAtLowPivot");
                    isFlipping = true;
                }
            }

            // --- RISK EVALUATION PIPELINE ---
            if (!isFlipping && currentEntryDir != 0)
            {
                if (riskMgr.State == C1_StopState.Null) 
                {
                    riskMgr.OnInitialFill(currentEntryPrice, currentEntryDir, c);
                }

                double unrealizedPts = currentEntryDir * (c - currentEntryPrice);
                if (unrealizedPts > currentTradeMfePts) 
                {
                    currentTradeMfePts = unrealizedPts;
                }
                if (-unrealizedPts > currentTradeMaePts) 
                {
                    currentTradeMaePts = -unrealizedPts;
                }

                riskMgr.EvaluateStopState(Position, c);

                if (stagnationMon.RequiresFlatten(Position, c, CurrentBar, currentEntryPrice))
                {
                    if (currentEntryDir > 0) 
                    {
                        ExitLong(Position.Quantity, "StagnationExitLong", "");
                    }
                    else 
                    {
                        ExitShort(Position.Quantity, "StagnationExitShort", "");
                    }
                }
            }
        }

        private void RouteStopOrder(double stopPrice, int direction, double currentPrice)
        {
            if (currentEntryDir == 0) 
            {
                return;
            }
            
            string sig = "ShortAtHighPivot";
            if (direction > 0)
            {
                sig = "LongAtLowPivot";
            }
            
            double gapBuffer = 1.0; 
            bool valid = false;
            
            if (direction > 0)
            {
                valid = stopPrice <= (currentPrice - gapBuffer);
            }
            else
            {
                valid = stopPrice >= (currentPrice + gapBuffer);
            }
            
            if (valid) 
            {
                SetStopLoss(sig, CalculationMode.Price, stopPrice, true);
            }
            else
            {
                if (direction > 0) 
                {
                    ExitLong(Position.Quantity, "TrailMissedBreachLong", "");
                }
                else 
                {
                    ExitShort(Position.Quantity, "TrailMissedBreachShort", "");
                }
            }
        }
    }

    public enum C1_StopState { Null, Initial, Tier1, Tier2 }

    public class C1_DynamicRiskManager
    {
        private readonly double initialStopPts;
        private readonly double t1ActivationPts;
        private readonly double t1TrailPts;
        private readonly double t2ActivationPts;
        private readonly double t2TrailPct;
        private readonly Action<double, int, double> stopRouter;

        private C1_StopState currentState = C1_StopState.Null;
        private double maxUnrealizedPts = 0.0;
        private double currentStopPrice = 0.0;

        public C1_DynamicRiskManager(double initialStopPts, double t1ActivationPts, double t1TrailPts, double t2ActivationPts, double t2TrailPct, Action<double, int, double> stopRouter)
        {
            this.initialStopPts = initialStopPts;
            this.t1ActivationPts = t1ActivationPts;
            this.t1TrailPts = t1TrailPts;
            this.t2ActivationPts = t2ActivationPts;
            this.t2TrailPct = t2TrailPct;
            this.stopRouter = stopRouter;
        }

        public C1_StopState State { get { return currentState; } }
        public double MaxUnrealized { get { return maxUnrealizedPts; } }
        public double CurrentStop { get { return currentStopPrice; } }

        public void EvaluateStopState(Position position, double currentPrice)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat) 
            {
                return;
            }

            int direction = -1;
            if (position.MarketPosition == MarketPosition.Long)
            {
                direction = 1;
            }

            double unrealizedPts = direction * (currentPrice - position.AveragePrice);
            if (unrealizedPts > maxUnrealizedPts) 
            {
                maxUnrealizedPts = unrealizedPts;
            }

            DetermineState(maxUnrealizedPts);
            CalculateAndRouteStop(position, currentPrice);
        }

        public void OnInitialFill(double fillPrice, int direction, double currentPrice)
        {
            ResetState();
            currentState = C1_StopState.Initial;
            if (initialStopPts > 0.0)
            {
                double initialStop = fillPrice - (direction * initialStopPts);
                currentStopPrice = initialStop;
                if (stopRouter != null) 
                {
                    stopRouter(initialStop, direction, currentPrice);
                }
            }
        }

        public void ResetState()
        {
            currentState = C1_StopState.Null;
            maxUnrealizedPts = 0.0;
            currentStopPrice = 0.0;
        }

        private void DetermineState(double maxPts)
        {
            if ((t2ActivationPts > 0.0) && (maxPts >= t2ActivationPts)) 
            {
                currentState = C1_StopState.Tier2;
            }
            else if ((t1ActivationPts > 0.0) && (maxPts >= t1ActivationPts)) 
            {
                currentState = C1_StopState.Tier1;
            }
            else if (currentState == C1_StopState.Null) 
            {
                currentState = C1_StopState.Initial;
            }
        }

        private void CalculateAndRouteStop(Position position, double currentPrice)
        {
            int direction = -1;
            if (position.MarketPosition == MarketPosition.Long)
            {
                direction = 1;
            }

            double entry = position.AveragePrice;
            double newStop = 0.0;

            if (currentState == C1_StopState.Initial)
            {
                if (initialStopPts <= 0.0) 
                {
                    return;
                }
                newStop = entry - (direction * initialStopPts);
            }
            else if (currentState == C1_StopState.Tier1)
            {
                if (t1TrailPts <= 0.0) 
                {
                    return;
                }
                double peakPrice = entry + (direction * maxUnrealizedPts);
                newStop = peakPrice - (direction * t1TrailPts);
            }
            else if (currentState == C1_StopState.Tier2)
            {
                double peakPrice = entry + (direction * maxUnrealizedPts);
                double dynamicTrail = t2TrailPct * maxUnrealizedPts;
                double trailPts = Math.Max(t1TrailPts, dynamicTrail);
                newStop = peakPrice - (direction * trailPts);
            }
            else 
            {
                return;
            }

            EnforceOrderModification(newStop, direction, currentPrice);
        }

        private void EnforceOrderModification(double calculatedStop, int direction, double currentPrice)
        {
            bool shouldUpdate = false;
            if (currentStopPrice == 0.0) 
            {
                shouldUpdate = true;
            }
            else if ((direction > 0) && (calculatedStop > currentStopPrice)) 
            {
                shouldUpdate = true;
            }
            else if ((direction < 0) && (calculatedStop < currentStopPrice)) 
            {
                shouldUpdate = true;
            }

            if (shouldUpdate)
            {
                currentStopPrice = calculatedStop;
                if (stopRouter != null) 
                {
                    stopRouter(calculatedStop, direction, currentPrice);
                }
            }
        }
    }

    public class C1_StagnationMonitor
    {
        private readonly int maxNegativeBars;
        private int consecutiveNegativeBars = 0;
        private int lastEvaluatedBar = -1;

        public C1_StagnationMonitor(int maxNegativeBars)
        {
            this.maxNegativeBars = maxNegativeBars;
        }

        public bool RequiresFlatten(Position position, double currentPrice, int currentBarIdx, double entryPrice)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
            {
                ResetState();
                return false;
            }

            if (currentBarIdx == lastEvaluatedBar) 
            {
                return false;
            }
            lastEvaluatedBar = currentBarIdx;

            int direction = -1;
            if (position.MarketPosition == MarketPosition.Long)
            {
                direction = 1;
            }

            double currentPnlPts = direction * (currentPrice - entryPrice);

            if (currentPnlPts < 0.0)
            {
                consecutiveNegativeBars++;
            }
            else 
            {
                consecutiveNegativeBars = 0;
            }

            if (maxNegativeBars > 0) 
            {
                return consecutiveNegativeBars >= maxNegativeBars;
            }
            
            return false;
        }

        private void ResetState()
        {
            consecutiveNegativeBars = 0;
            lastEvaluatedBar = -1;
        }
    }
}