// =============================================================================
// VWAP_MTF_Reversion v1.2 -- Full Genetic Risk Surface
// =============================================================================
#region Using declarations
using System;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class VWAP_MTF_Reversion : Strategy
    {
        [NinjaScriptProperty]
        [Display(Name = "Contracts", Order = 1, GroupName = "1. Execution")]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 4.0)]
        [Display(Name = "VWAP Target Band (Multiplier)", Order = 1, GroupName = "2. Optimization Space")]
        public double VwapMult1 { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 5.0)]
        [Display(Name = "VWAP Entry Band (Multiplier)", Order = 2, GroupName = "2. Optimization Space")]
        public double VwapMult2 { get; set; }

        [NinjaScriptProperty]
        [Range(10, 200)]
        [Display(Name = "HTF SMA Period", Order = 3, GroupName = "2. Optimization Space")]
        public int SmaPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(10.0, 100.0)]
        [Display(Name = "Hard Stop (Points)", Order = 4, GroupName = "2. Optimization Space")]
        public double HardStopPoints { get; set; }

        [NinjaScriptProperty]
        [Range(5.0, 100.0)]
        [Display(Name = "Trail Activate (Points)", Order = 5, GroupName = "2. Optimization Space")]
        public double TrailActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Range(5.0, 50.0)]
        [Display(Name = "Trail Distance (Points)", Order = 6, GroupName = "2. Optimization Space")]
        public double TrailDistancePoints { get; set; }

        [NinjaScriptProperty]
        [Range(1, 50)]
        [Display(Name = "Max Stale Bars", Order = 7, GroupName = "2. Optimization Space")]
        public int MaxNegativeBars { get; set; }

        private OrderFlowVWAP vwap;
        private SMA smaHTF;

        private double currentEntryPrice;
        private int currentEntryDir;
        private double currentTradeMfePts;
        private double currentTradeMaePts;

        private VWAP_RiskManager riskMgr;
        private VWAP_StagnationMonitor stagnationMon;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "VWAP_MTF_Reversion_v1.2";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                BarsRequiredToTrade = 50;
                IsInstantiatedOnEachOptimizationIteration = false;

                Contracts = 1;
                VwapMult1 = 1.0;
                VwapMult2 = 2.0;
                SmaPeriod = 50;
                HardStopPoints = 25.0;
                TrailActivatePoints = 15.0;
                TrailDistancePoints = 10.0;
                MaxNegativeBars = 5;
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, 15);

                currentEntryPrice = 0.0;
                currentEntryDir = 0;
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;

                riskMgr = new VWAP_RiskManager(HardStopPoints, TrailActivatePoints, TrailDistancePoints, RouteStopOrder);
                stagnationMon = new VWAP_StagnationMonitor(MaxNegativeBars);
            }
            else if (State == State.DataLoaded)
            {
                vwap = OrderFlowVWAP(VWAPResolution.Standard, Bars.TradingHours, VWAPStandardDeviations.Three, VwapMult1, VwapMult2, 3.0);
                smaHTF = SMA(BarsArray[1], SmaPeriod);
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
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState();
                        stagnationMon = new VWAP_StagnationMonitor(MaxNegativeBars);
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
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;
                riskMgr.ResetState();
                stagnationMon = new VWAP_StagnationMonitor(MaxNegativeBars);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;
            if (CurrentBars[0] < 1 || CurrentBars[1] < 1) return;
            if (BarsInProgress != 0) return;

            // Theoretical Boundary Invalidation
            if (VwapMult1 >= VwapMult2)
            {
                return;
            }

            double upper1 = vwap.StdDev1Upper[0];
            double upper2 = vwap.StdDev2Upper[0];
            double lower1 = vwap.StdDev1Lower[0];
            double lower2 = vwap.StdDev2Lower[0];

            double htfTrend = smaHTF[1];
            double htfClose = Closes[1][1];

            bool isUptrend = false;
            if (htfClose > htfTrend)
            {
                isUptrend = true;
            }

            bool isDowntrend = false;
            if (htfClose < htfTrend)
            {
                isDowntrend = true;
            }

            double c = Close[0];
            double h = High[0];
            double l = Low[0];

            // --- ENTRY GATES ---
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                if (isDowntrend)
                {
                    if ((h >= upper2) && (c < upper2))
                    {
                        EnterShort(Contracts, "Short_VWAP");
                        SetProfitTarget("Short_VWAP", CalculationMode.Price, upper1);
                    }
                }

                if (isUptrend)
                {
                    if ((l <= lower2) && (c > lower2))
                    {
                        EnterLong(Contracts, "Long_VWAP");
                        SetProfitTarget("Long_VWAP", CalculationMode.Price, lower1);
                    }
                }
            }

            // --- RISK EVALUATION PIPELINE ---
            if (currentEntryDir != 0)
            {
                if (riskMgr.State == VWAP_StopState.Null)
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

            string sig = "Short_VWAP";
            if (direction > 0)
            {
                sig = "Long_VWAP";
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
                    ExitLong(Position.Quantity, "TrailBreachLong", "");
                }
                else
                {
                    ExitShort(Position.Quantity, "TrailBreachShort", "");
                }
            }
        }
    }

    public enum VWAP_StopState { Null, Initial, Trailing }

    public class VWAP_RiskManager
    {
        private readonly double initialStopPts;
        private readonly double trailActivatePts;
        private readonly double trailDistancePts;
        private readonly Action<double, int, double> stopRouter;

        private VWAP_StopState currentState = VWAP_StopState.Null;
        private double maxUnrealizedPts = 0.0;
        private double currentStopPrice = 0.0;

        public VWAP_RiskManager(double initialStopPts, double trailActivatePts, double trailDistancePts, Action<double, int, double> stopRouter)
        {
            this.initialStopPts = initialStopPts;
            this.trailActivatePts = trailActivatePts;
            this.trailDistancePts = trailDistancePts;
            this.stopRouter = stopRouter;
        }

        public VWAP_StopState State { get { return currentState; } }

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

            if ((currentState == VWAP_StopState.Initial) && (maxUnrealizedPts >= trailActivatePts))
            {
                currentState = VWAP_StopState.Trailing;
            }

            CalculateAndRouteStop(position, currentPrice);
        }

        public void OnInitialFill(double fillPrice, int direction, double currentPrice)
        {
            ResetState();
            currentState = VWAP_StopState.Initial;
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
            currentState = VWAP_StopState.Null;
            maxUnrealizedPts = 0.0;
            currentStopPrice = 0.0;
        }

        private void CalculateAndRouteStop(Position position, double currentPrice)
        {
            if (currentState != VWAP_StopState.Trailing)
            {
                return;
            }

            int direction = -1;
            if (position.MarketPosition == MarketPosition.Long)
            {
                direction = 1;
            }

            double entry = position.AveragePrice;
            double peakPrice = entry + (direction * maxUnrealizedPts);
            double newStop = peakPrice - (direction * trailDistancePts);

            bool shouldUpdate = false;
            if (direction > 0 && newStop > currentStopPrice)
            {
                shouldUpdate = true;
            }
            else if (direction < 0 && newStop < currentStopPrice)
            {
                shouldUpdate = true;
            }

            if (shouldUpdate)
            {
                currentStopPrice = newStop;
                if (stopRouter != null)
                {
                    stopRouter(newStop, direction, currentPrice);
                }
            }
        }
    }

    public class VWAP_StagnationMonitor
    {
        private readonly int maxNegativeBars;
        private int consecutiveNegativeBars = 0;
        private int lastEvaluatedBar = -1;

        public VWAP_StagnationMonitor(int maxNegativeBars)
        {
            this.maxNegativeBars = maxNegativeBars;
        }

        public bool RequiresFlatten(Position position, double currentPrice, int currentBarIdx, double entryPrice)
        {
            if (position == null || position.MarketPosition == MarketPosition.Flat)
            {
                consecutiveNegativeBars = 0;
                lastEvaluatedBar = -1;
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
    }
}
