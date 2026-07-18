// =============================================================================
// MultiTimeframeRunner v1.0.2 -- Strict Syntax Enforcement
// =============================================================================
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class MultiTimeframeRunner : Strategy
    {
        [NinjaScriptProperty]
        [Display(Name = "Contracts", Order = 1, GroupName = "Execution")]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Hour UTC", Order = 1, GroupName = "Schedule")]
        public int EodHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Minute UTC", Order = 2, GroupName = "Schedule")]
        public int EodMinuteUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Hour UTC", Order = 3, GroupName = "Schedule")]
        public int EntryCutoffHourUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Entry Cutoff Minute UTC", Order = 4, GroupName = "Schedule")]
        public int EntryCutoffMinuteUtc { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Activate (points)", Order = 1, GroupName = "TrailStop")]
        public double TrailActivatePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Distance (points)", Order = 2, GroupName = "TrailStop")]
        public double TrailDistancePoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trail Percent", Order = 3, GroupName = "TrailStop")]
        public double TrailPercent { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Hard Stop Loss (points)", Order = 1, GroupName = "StopLoss")]
        public double HardStopLossPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Negative Bars", Description = "Consecutive negative bars before flattening. 0 to disable.", Order = 1, GroupName = "Stagnation")]
        public int MaxNegativeBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Log CSV Path", Order = 1, GroupName = "Logging")]
        public string CsvPath { get; set; }

        private SMA sma50B0;
        private SMA sma50B1;
        private SMA sma50B2;
        private SMA sma5B0;
        private SMA sma5B1;
        private SMA sma5B2;

        private const string CSV_HEADER = "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason,mfe_pts,mae_pts,capture_pct,max_neg_bars";

        private double currentEntryPrice;
        private DateTime currentEntryTime;
        private int currentEntryDir;       
        private int currentEntryQty;
        private string currentEntryReason;
        private readonly object csvLock = new object();

        private MTF_DynamicRiskManager riskMgr;
        private MTF_StagnationMonitor stagnationMon;
        private double currentTradeMfePts;   
        private double currentTradeMaePts;   

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "MultiTimeframeRunner_v1.0.2";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 50; 

                Contracts = 1;
                EodHourUtc = 20;
                EodMinuteUtc = 55;
                EntryCutoffHourUtc = 20;
                EntryCutoffMinuteUtc = 30;
                TrailActivatePoints = 10.0;   
                TrailDistancePoints = 5.0;    
                TrailPercent = 0.10;   
                HardStopLossPoints = 25.0;   
                MaxNegativeBars = 5;
                CsvPath = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_mtf_trades.csv";
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Minute, 5);
                AddDataSeries(BarsPeriodType.Minute, 15);

                currentEntryPrice = 0.0;
                currentEntryTime = DateTime.MinValue;
                currentEntryDir = 0;
                currentEntryQty = 0;
                currentEntryReason = string.Empty;
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;

                double t2ActPts = double.MaxValue;
                if (TrailPercent > 0.0)
                {
                    t2ActPts = TrailDistancePoints / TrailPercent;
                }

                riskMgr = new MTF_DynamicRiskManager(HardStopLossPoints, TrailActivatePoints, TrailDistancePoints, t2ActPts, TrailPercent, RouteStopOrder);
                stagnationMon = new MTF_StagnationMonitor(MaxNegativeBars);
                EnsureCsvHeader();
            }
            else if (State == State.DataLoaded)
            {
                sma50B0 = SMA(50);
                sma5B0 = SMA(5);
                
                sma50B1 = SMA(BarsArray[1], 50);
                sma50B2 = SMA(BarsArray[2], 50);
                sma5B1  = SMA(BarsArray[1], 5);
                sma5B2  = SMA(BarsArray[2], 5);
            }
        }

        private void EnsureCsvHeader()
        {
            if (string.IsNullOrWhiteSpace(CsvPath)) 
            {
                return;
            }
            try
            {
                string dir = Path.GetDirectoryName(CsvPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) 
                {
                    Directory.CreateDirectory(dir);
                }
                if (!File.Exists(CsvPath)) 
                {
                    lock (csvLock) 
                    { 
                        File.WriteAllText(CsvPath, CSV_HEADER + Environment.NewLine); 
                    }
                }
            }
            catch (Exception ex) 
            { 
                Print("CSV init error: " + ex.Message); 
            }
        }

        private static string CsvEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) 
            {
                return "";
            }
            if (s.Contains(",") || s.Contains("\"")) 
            {
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            }
            return s;
        }

        private void AppendTradeCsv(DateTime exitTime, string exitReason, double exitPrice, int qty)
        {
            if (string.IsNullOrWhiteSpace(CsvPath) || currentEntryDir == 0) 
            {
                return;
            }
            try
            {
                double pnlPts = currentEntryDir * (exitPrice - currentEntryPrice);
                double pnlUsd = pnlPts * 2.0 * qty;   
                double heldMin = (exitTime - currentEntryTime).TotalMinutes;
                
                string dir = "short";
                if (currentEntryDir > 0)
                {
                    dir = "long";
                }

                string day = currentEntryTime.ToUniversalTime().ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
                DateTime exitUtc = exitTime.ToUniversalTime();
                DateTime entryUtc = currentEntryTime.ToUniversalTime();

                double capturePct = 0.0;
                if (currentTradeMfePts > 0) 
                {
                    capturePct = 100.0 * pnlPts / currentTradeMfePts;
                }

                int trackedMaxBars = 0;
                if (stagnationMon != null) 
                {
                    trackedMaxBars = stagnationMon.MaxConsecutiveNegative;
                }

                string row = string.Join(",", new string[] {
                    DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture), day,
                    entryUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    exitUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    dir, currentEntryPrice.ToString("F4", CultureInfo.InvariantCulture),
                    exitPrice.ToString("F4", CultureInfo.InvariantCulture), qty.ToString(CultureInfo.InvariantCulture),
                    pnlPts.ToString("F4", CultureInfo.InvariantCulture), pnlUsd.ToString("F2", CultureInfo.InvariantCulture),
                    heldMin.ToString("F2", CultureInfo.InvariantCulture), CsvEscape(currentEntryReason), CsvEscape(exitReason),
                    currentTradeMfePts.ToString("F4", CultureInfo.InvariantCulture), currentTradeMaePts.ToString("F4", CultureInfo.InvariantCulture),
                    capturePct.ToString("F2", CultureInfo.InvariantCulture), trackedMaxBars.ToString(CultureInfo.InvariantCulture)
                });

                lock (csvLock) 
                { 
                    File.AppendAllText(CsvPath, row + Environment.NewLine); 
                }
            }
            catch (Exception ex) 
            { 
                Print("CSV append error: " + ex.Message); 
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution == null || execution.Order == null) 
            {
                return;
            }

            string orderName = execution.Order.Name ?? execution.Order.OrderAction.ToString();

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
                    AppendTradeCsv(time, orderName, price, currentEntryQty);

                    if (flatAfter)
                    {
                        currentEntryDir = 0;
                        currentEntryQty = 0;
                        currentEntryPrice = 0.0;
                        currentEntryReason = string.Empty;
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState();
                    }
                    else
                    {
                        currentEntryDir = -1;
                        currentEntryReason = "EnterShort_SMA";
                        if (marketPosition == MarketPosition.Long)
                        {
                            currentEntryDir = 1;
                            currentEntryReason = "EnterLong_SMA";
                        }
                        
                        currentEntryPrice = price;
                        currentEntryTime = time;
                        currentEntryQty = Contracts;
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState(); 
                        stagnationMon = new MTF_StagnationMonitor(MaxNegativeBars);
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
                currentEntryReason = orderName;   
                currentTradeMfePts = 0.0;
                currentTradeMaePts = 0.0;
                riskMgr.ResetState(); 
                stagnationMon = new MTF_StagnationMonitor(MaxNegativeBars);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) return;
            if (CurrentBars[0] < 1) return;
            if (CurrentBars[1] < 1) return;
            if (CurrentBars[2] < 1) return;
            if (BarsInProgress != 0) return;

            if (Math.Abs(Position.Quantity) > Contracts)
            {
                if (Position.MarketPosition == MarketPosition.Long) 
                {
                    ExitLong(Position.Quantity, "SafetyPanicLong", "");
                }
                else if (Position.MarketPosition == MarketPosition.Short) 
                {
                    ExitShort(Position.Quantity, "SafetyPanicShort", "");
                }
                return;
            }

            double c = Close[0];
            DateTime barUtc = Time[0].ToUniversalTime();
            int minsOfDay = (barUtc.Hour * 60) + barUtc.Minute;
            int eodMins = (EodHourUtc * 60) + EodMinuteUtc;
            int entryCutMins = (EntryCutoffHourUtc * 60) + EntryCutoffMinuteUtc;

            if (minsOfDay >= eodMins)
            {
                if (Position.MarketPosition == MarketPosition.Long) 
                {
                    ExitLong(Position.Quantity, "EodExitLong", "");
                }
                else if (Position.MarketPosition == MarketPosition.Short) 
                {
                    ExitShort(Position.Quantity, "EodExitShort", "");
                }
                return;
            }

            bool isFlipping = false;

            // --- ENTRY LOGIC PIPELINE ---
            if (minsOfDay < entryCutMins)
            {
                bool htfBullish = ((sma5B1[1] > sma50B1[1]) && (sma5B2[1] > sma50B2[1]));
                bool htfBearish = ((sma5B1[1] < sma50B1[1]) && (sma5B2[1] < sma50B2[1]));

                if (htfBullish && CrossAbove(sma5B0, sma50B0, 1))
                {
                    if (Position.MarketPosition == MarketPosition.Short && Position.Quantity >= Contracts) 
                    {
                        return; 
                    }
                    if (Position.MarketPosition == MarketPosition.Short) 
                    {
                        ExitShort(Position.Quantity, "FlipExitShort", "");
                    }
                    EnterLong(Contracts, "EnterLong_SMA");
                    isFlipping = true;
                }
                else if (htfBearish && CrossBelow(sma5B0, sma50B0, 1))
                {
                    if (Position.MarketPosition == MarketPosition.Long && Position.Quantity >= Contracts) 
                    {
                        return; 
                    }
                    if (Position.MarketPosition == MarketPosition.Long) 
                    {
                        ExitLong(Position.Quantity, "FlipExitLong", "");
                    }
                    EnterShort(Contracts, "EnterShort_SMA");
                    isFlipping = true;
                }
            }

            // --- RISK EVALUATION PIPELINE ---
            if (!isFlipping && currentEntryDir != 0)
            {
                if (riskMgr.State == MTF_StopState.Null) 
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
            
            string sig = "EnterShort_SMA";
            if (direction > 0)
            {
                sig = "EnterLong_SMA";
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

    public enum MTF_StopState { Null, Initial, Tier1, Tier2 }

    public class MTF_DynamicRiskManager
    {
        private readonly double initialStopPts;
        private readonly double t1ActivationPts;
        private readonly double t1TrailPts;
        private readonly double t2ActivationPts;
        private readonly double t2TrailPct;
        private readonly Action<double, int, double> stopRouter;

        private MTF_StopState currentState = MTF_StopState.Null;
        private double maxUnrealizedPts = 0.0;
        private double currentStopPrice = 0.0;

        public MTF_DynamicRiskManager(double initialStopPts, double t1ActivationPts, double t1TrailPts, double t2ActivationPts, double t2TrailPct, Action<double, int, double> stopRouter)
        {
            this.initialStopPts = initialStopPts;
            this.t1ActivationPts = t1ActivationPts;
            this.t1TrailPts = t1TrailPts;
            this.t2ActivationPts = t2ActivationPts;
            this.t2TrailPct = t2TrailPct;
            this.stopRouter = stopRouter;
        }

        public MTF_StopState State { get { return currentState; } }
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
            currentState = MTF_StopState.Initial;
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
            currentState = MTF_StopState.Null;
            maxUnrealizedPts = 0.0;
            currentStopPrice = 0.0;
        }

        private void DetermineState(double maxPts)
        {
            if ((t2ActivationPts > 0.0) && (maxPts >= t2ActivationPts)) 
            {
                currentState = MTF_StopState.Tier2;
            }
            else if ((t1ActivationPts > 0.0) && (maxPts >= t1ActivationPts)) 
            {
                currentState = MTF_StopState.Tier1;
            }
            else if (currentState == MTF_StopState.Null) 
            {
                currentState = MTF_StopState.Initial;
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

            if (currentState == MTF_StopState.Initial)
            {
                if (initialStopPts <= 0.0) 
                {
                    return;
                }
                newStop = entry - (direction * initialStopPts);
            }
            else if (currentState == MTF_StopState.Tier1)
            {
                if (t1TrailPts <= 0.0) 
                {
                    return;
                }
                double peakPrice = entry + (direction * maxUnrealizedPts);
                newStop = peakPrice - (direction * t1TrailPts);
            }
            else if (currentState == MTF_StopState.Tier2)
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

    public class MTF_StagnationMonitor
    {
        private readonly int maxNegativeBars;
        private int consecutiveNegativeBars = 0;
        private int lastEvaluatedBar = -1;
        public int MaxConsecutiveNegative { get; private set; }

        public MTF_StagnationMonitor(int maxNegativeBars)
        {
            this.maxNegativeBars = maxNegativeBars;
            this.MaxConsecutiveNegative = 0;
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
                if (consecutiveNegativeBars > MaxConsecutiveNegative) 
                {
                    MaxConsecutiveNegative = consecutiveNegativeBars;
                }
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
            MaxConsecutiveNegative = 0;
        }
    }
}