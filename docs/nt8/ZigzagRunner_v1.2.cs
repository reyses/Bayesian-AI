// =============================================================================
// ZigzagRunner 1.2.6 -- Continuous Stagnation Tracking & CSV Export
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
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ZigzagRunner_v12 : Strategy
    {
        [NinjaScriptProperty]
        [Display(Name = "R (points)", Order = 1, GroupName = "Zigzag")]
        public double RPoints { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Contracts", Order = 2, GroupName = "Zigzag")]
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
        [Display(Name = "Max Negative Bars", Description = "Consecutive negative bars before flattening. 0 to disable but keep tracking.", Order = 1, GroupName = "Stagnation")]
        public int MaxNegativeBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Log CSV Path", Order = 1, GroupName = "Logging")]
        public string CsvPath { get; set; }

        private const string VERSION = "1.2.6";
        private const string CSV_HEADER = "close_time_utc,day,entry_time_utc,exit_time_utc,direction,entry_price,exit_price,qty,pnl_points,pnl_usd,held_minutes,entry_reason,exit_reason,mfe_pts,mae_pts,capture_pct,max_neg_bars";

        private int direction;
        private double extremePrice;
        private int extremeBarIdx;
        private int lastPivotDir;     
        private double lastPivotPrice;

        private double currentEntryPrice;
        private DateTime currentEntryTime;
        private int currentEntryDir;       
        private int currentEntryQty;
        private string currentEntryReason;
        private readonly object csvLock = new object();

        private DynamicRiskManager_v12 riskMgr;
        private StagnationMonitor_v12 stagnationMon;
        private double currentTradeMfePts;   
        private double currentTradeMaePts;   

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "ZigzagRunner_v1.2";
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
                BarsRequiredToTrade = 2;

                RPoints = 30.0;
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
                CsvPath = @"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\reports\findings\nt8_zigzag_v1.2_trades.csv";
            }
            else if (State == State.Configure)
            {
                direction = 0;
                extremePrice = double.NaN;
                extremeBarIdx = -1;
                lastPivotDir = 0;
                lastPivotPrice = double.NaN;
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

                riskMgr = new DynamicRiskManager_v12(HardStopLossPoints, TrailActivatePoints, TrailDistancePoints, t2ActPts, TrailPercent, RouteStopOrder);
                stagnationMon = new StagnationMonitor_v12(MaxNegativeBars);
                EnsureCsvHeader();
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
                    DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture), 
                    day,
                    entryUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    exitUtc.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    dir, 
                    currentEntryPrice.ToString("F4", CultureInfo.InvariantCulture),
                    exitPrice.ToString("F4", CultureInfo.InvariantCulture), 
                    qty.ToString(CultureInfo.InvariantCulture),
                    pnlPts.ToString("F4", CultureInfo.InvariantCulture), 
                    pnlUsd.ToString("F2", CultureInfo.InvariantCulture),
                    heldMin.ToString("F2", CultureInfo.InvariantCulture), 
                    CsvEscape(currentEntryReason), 
                    CsvEscape(exitReason),
                    currentTradeMfePts.ToString("F4", CultureInfo.InvariantCulture), 
                    currentTradeMaePts.ToString("F4", CultureInfo.InvariantCulture),
                    capturePct.ToString("F2", CultureInfo.InvariantCulture),
                    trackedMaxBars.ToString(CultureInfo.InvariantCulture)
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
                bool flatAfter = marketPosition == MarketPosition.Flat;
                bool flippedSign = (currentEntryDir > 0 && marketPosition == MarketPosition.Short) || (currentEntryDir < 0 && marketPosition == MarketPosition.Long);

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
                        currentEntryReason = "ShortAtHighPivot";
                        if (marketPosition == MarketPosition.Long)
                        {
                            currentEntryDir = 1;
                            currentEntryReason = "LongAtLowPivot";
                        }
                        
                        currentEntryPrice = price;
                        currentEntryTime = time;
                        currentEntryQty = Contracts;
                        currentTradeMfePts = 0.0;
                        currentTradeMaePts = 0.0;
                        riskMgr.ResetState(); 
                        stagnationMon = new StagnationMonitor_v12(MaxNegativeBars);
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
                stagnationMon = new StagnationMonitor_v12(MaxNegativeBars);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < BarsRequiredToTrade) 
            {
                return;
            }

            if (Math.Abs(Position.Quantity) > Contracts)
            {
                Print("ZigzagRunner SAFETY: Position.Quantity=" + Position.Quantity + " exceeds Contracts=" + Contracts + ", panic-closing.");
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
            int minsOfDay = barUtc.Hour * 60 + barUtc.Minute;
            int eodMins = EodHourUtc * 60 + EodMinuteUtc;
            int entryCutMins = EntryCutoffHourUtc * 60 + EntryCutoffMinuteUtc;

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
                if (c - extremePrice >= RPoints) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = -1; 
                    lastPivotPrice = extremePrice; 
                    direction = 1; 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
                else if (extremePrice - c >= RPoints) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = 1; 
                    lastPivotPrice = extremePrice; 
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
                else if (extremePrice - c >= RPoints) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = 1; 
                    lastPivotPrice = extremePrice; 
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
                else if (c - extremePrice >= RPoints) 
                { 
                    pivotConfirmed = true; 
                    newPivotDir = -1; 
                    lastPivotPrice = extremePrice; 
                    direction = 1; 
                    extremePrice = c; 
                    extremeBarIdx = CurrentBar; 
                }
            }

            if (pivotConfirmed) 
            {
                lastPivotDir = newPivotDir;
            }

            bool isFlipping = false;

            if (pivotConfirmed && minsOfDay < entryCutMins)
            {
                if (newPivotDir == 1)
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
                else
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

            if (!isFlipping && currentEntryDir != 0)
            {
                if (riskMgr.State == StopState_v12.Null)
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

                riskMgr.EvaluateStopState_v12(Position, c);

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

    public enum StopState_v12 { Null, Initial, Tier1, Tier2 }

    public class DynamicRiskManager_v12
    {
        private readonly double initialStopPts;
        private readonly double t1ActivationPts;
        private readonly double t1TrailPts;
        private readonly double t2ActivationPts;
        private readonly double t2TrailPct;
        private readonly Action<double, int, double> stopRouter;

        private StopState_v12 currentState = StopState_v12.Null;
        private double maxUnrealizedPts = 0.0;
        private double currentStopPrice = 0.0;

        public DynamicRiskManager_v12(double initialStopPts, double t1ActivationPts, double t1TrailPts, double t2ActivationPts, double t2TrailPct, Action<double, int, double> stopRouter)
        {
            this.initialStopPts = initialStopPts;
            this.t1ActivationPts = t1ActivationPts;
            this.t1TrailPts = t1TrailPts;
            this.t2ActivationPts = t2ActivationPts;
            this.t2TrailPct = t2TrailPct;
            this.stopRouter = stopRouter;
        }

        public StopState_v12 State { get { return currentState; } }
        public double MaxUnrealized { get { return maxUnrealizedPts; } }
        public double CurrentStop { get { return currentStopPrice; } }

        public void EvaluateStopState_v12(Position position, double currentPrice)
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
            currentState = StopState_v12.Initial;
            
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
            currentState = StopState_v12.Null;
            maxUnrealizedPts = 0.0;
            currentStopPrice = 0.0;
        }

        private void DetermineState(double maxPts)
        {
            if (t2ActivationPts > 0.0 && maxPts >= t2ActivationPts) 
            {
                currentState = StopState_v12.Tier2;
            }
            else if (t1ActivationPts > 0.0 && maxPts >= t1ActivationPts) 
            {
                currentState = StopState_v12.Tier1;
            }
            else if (currentState == StopState_v12.Null) 
            {
                currentState = StopState_v12.Initial;
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

            if (currentState == StopState_v12.Initial)
            {
                if (initialStopPts <= 0.0) 
                {
                    return;
                }
                newStop = entry - (direction * initialStopPts);
            }
            else if (currentState == StopState_v12.Tier1)
            {
                if (t1TrailPts <= 0.0) 
                {
                    return;
                }
                double peakPrice = entry + (direction * maxUnrealizedPts);
                newStop = peakPrice - (direction * t1TrailPts);
            }
            else if (currentState == StopState_v12.Tier2)
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
            else if (direction > 0 && calculatedStop > currentStopPrice) 
            {
                shouldUpdate = true;
            }
            else if (direction < 0 && calculatedStop < currentStopPrice) 
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

    public class StagnationMonitor_v12
    {
        private readonly int maxNegativeBars;
        private int consecutiveNegativeBars = 0;
        private int lastEvaluatedBar = -1;
        
        public int MaxConsecutiveNegative { get; private set; }

        public StagnationMonitor_v12(int maxNegativeBars)
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